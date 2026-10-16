"""Training_Harness: real gradient-descent training for the two torch models
(Requirement 4, `design.md` Section 3.4).

What this module owns, and what it reuses
-------------------------------------------------------------------------

The *semantics* of every loss term, the lambda grid, the selection rule,
early stopping, and seed derivation already exist and are imported, not
restated:

- `models.losses.load_lambda_grid` -- the frozen 3x3 grid read from
  `configs/experiments/lambda_grid.yaml` (`decisions.md` Section 14.3);
- `models.losses.select_lambda_by_validation_concordance` -- the selection
  rule, fed validation-split triples that it scores with
  `evaluation.metrics.concordance_index`;
- `models.losses.total_loss` -- the frozen combination
  `L_survival + lambda_state * L_CD4/CD8 + lambda_balance * L_balance`. It is
  a pure arithmetic combination, so it is applied to tensors directly and
  autograd flows through it unchanged;
- `models.losses.masked_huber` / `mmd` and
  `models.training.discrete_time_censored_nll` -- the scalar specifications
  of record. This module supplies differentiable batched equivalents
  (`masked_huber_tensor`, `mmd_tensor`, `survival_nll_tensor`) whose
  definitions follow those functions term for term; task 10.3's gradient
  tests use the pure-Python functions as the oracle;
- `models.training.EarlyStopping` -- the patience rule, unchanged;
- `infra.seeding.derive_seed` -- replica sub-seeds, exactly as
  `uncertainty.ensemble.DeepEnsemble` derives them.

Frozen training configuration (`decisions.md` Section 25)
-------------------------------------------------------------------------

Adam with `betas=(0.9, 0.999)` and `weight_decay=1e-4`; `lr=1e-3` with no
schedule; full-batch (one optimizer step per epoch over the whole
development batch); global-norm gradient clipping at `1.0` every step;
early-stopping patience `20` on validation loss; `max_epochs=500`; the
checkpoint is the validation-loss-minimizing epoch, written as a plain dict
via `torch.save`. These are `FROZEN_TRAINING_CONFIG`. `TrainingConfig`
exists so tests can run a short `max_epochs`; `TrainingConfig.is_frozen`
reports whether a config is the frozen one, and production callers pass
the default.

Survival NLL conventions
-------------------------------------------------------------------------

Per patient with assigned bin `b` (`TensorBatch.bin_index`) and event flag
`e` (`TensorBatch.event_indicator`):

- event (`e = True`): `-sum_{k<b} log(1 - h_k) - log(h_b)`, which is
  `discrete_time_censored_nll(h, event_time_index=b)`;
- censored (`e = False`): `-sum_{k<b} log(1 - h_k)`, i.e. the patient is
  credited with surviving every bin *before* the one in which censoring
  happened -- `discrete_time_censored_nll(h[:b], None)` (and `0` when
  `b = 0`). The partially observed censoring bin is not credited.

`log h` and `log(1 - h)` are computed as `logsigmoid(logit)` and
`logsigmoid(-logit)` from the models' `hazard_logits`, the numerically
stable form of the oracle's epsilon-clamped logs. The batch value is the
mean over patients, matching `models.training.censoring_aware_loss`.

`TensorBatch.censoring_mask` is not consumed. It flags the narrower case
of a censoring time falling before the assigned bin's start, which on the
real ACTG175 v3.2 data marks no patient (the censoring and outcome tables
agree); for such a row the convention above credits survival up to the
assigned bin's start.

Masked Huber conventions (Requirement 4.2)
-------------------------------------------------------------------------

The per-element Huber term (threshold `delta = 1.0`, `masked_huber`'s
default) is multiplied by the observed mask *before* reduction, so a
masked-missing entry contributes exactly zero to the value and exactly zero
gradient to the prediction at that position, whatever placeholder the
target holds. Reduction follows `masked_huber`: the sum of the CD4 and CD8
components per (patient, landmark) position, averaged over positions with
at least one observed component. When a position's two masks agree (the
case `masked_huber` can express), the two definitions coincide exactly;
where only one component is observed -- week-96 CD8 is 100% missing on the
real data -- the missing component contributes zero and the position
still counts once.

Representation for the balance term
-------------------------------------------------------------------------

`MMD(z_treated, z_control)` uses each model's final-landmark hidden state
under the patient's *factual* arm: the representation the hazard heads
read. For `TorchSequenceGRU` that is the only representation there is; for
`TorchInterventionAwareModel` it is the same choice, so the two families'
balance terms measure the same object. The linear-kernel form follows
`models.losses.mmd`.

Split isolation (Requirements 4.7, 4.8, 18.5)
-------------------------------------------------------------------------

Every training entry point takes a `TrainingBatches(dev, val)` pair and
nothing else that carries patient data, so there is no parameter through
which a test batch could arrive. `TrainingBatches` rejects a batch whose
`split_name` is not the one its slot expects -- a `"test"` batch in either
slot raises `ReferenceAccessError`, the same error the reference guard
raises for unauthorized test-split access. This module calls neither
`require_test_split` nor `unseal` and imports neither.

Torch dependency scope
-------------------------------------------------------------------------

`torch` is a leaf dependency confined to `src/models/torch_*.py` and
`src/models/dataset.py` (`design.md` Section 2.4). Nothing torch-free may
import this module; tests live under `tests/neural/`.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from infra.manifest import ManifestWriter
from infra.reference_guard import ReferenceAccessError
from infra.seeding import derive_seed
from models.dataset import TensorBatch
from models.losses import (
    load_lambda_grid,
    select_lambda_by_validation_concordance,
    total_loss,
)
from models.torch_intervention_aware import TorchInterventionAwareModel, outcome_from_hazards
from models.torch_sequence_gru import TorchSequenceGRU
from models.training import EarlyStopping

# ---------------------------------------------------------------------------
# Frozen configuration (decisions.md Section 25)
# ---------------------------------------------------------------------------

#: Number of deep-ensemble replicas (`decisions.md` Section 15).
N_ENSEMBLE_REPLICAS: int = 5

#: Number of `(lambda_state, lambda_balance)` combinations in the frozen
#: grid (`decisions.md` Section 14.3: "9 combinations").
LAMBDA_GRID_SIZE: int = 9

#: Huber threshold, `models.losses.masked_huber`'s default.
HUBER_DELTA: float = 1.0

#: Supported device strings (`decisions.md` Section 25: "`cpu` default,
#: `cuda` when available").
SUPPORTED_DEVICES: tuple[str, ...] = ("cpu", "cuda")


@dataclass(frozen=True)
class TrainingConfig:
    """The optimizer and stopping configuration of one training run.

    Every default is the `decisions.md` Section 25 value. `max_epochs` is
    the one field tests shorten; `is_frozen` says whether a config is the
    frozen one, so a caller that must use it can check.
    """

    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    patience: int = 20
    max_epochs: int = 500

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs!r}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience!r}")

    @property
    def is_frozen(self) -> bool:
        return self == FROZEN_TRAINING_CONFIG


#: `decisions.md` Section 25's frozen training configuration.
FROZEN_TRAINING_CONFIG = TrainingConfig()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeviceUnavailableError(RuntimeError):
    """A supported device was requested but is not available on this host."""


class UnsupportedDeviceError(ValueError):
    """A device string outside `SUPPORTED_DEVICES` was requested."""


class TrainingDivergedError(RuntimeError):
    """The development-split training loss became non-finite."""


# ---------------------------------------------------------------------------
# Device and seeding
# ---------------------------------------------------------------------------


def resolve_device(requested: str = "cpu") -> str:
    """Validate a device request before any tensor is allocated.

    Accepts exactly `"cpu"` and `"cuda"` (`decisions.md` Section 25). The
    match is exact: `"CUDA"`, `"cuda:0"`, `" cpu"`, `"mps"`, and non-strings
    are all unsupported. Nothing here allocates a tensor, so a caller that
    resolves first fails before any allocation or training (Requirement 5.6).

    Raises:
        UnsupportedDeviceError: for any other value, including `"mps"`.
        DeviceUnavailableError: for `"cuda"` when `torch.cuda.is_available()`
            is false, naming the device.
    """
    if not isinstance(requested, str) or requested not in SUPPORTED_DEVICES:
        raise UnsupportedDeviceError(
            f"device {requested!r} is not supported; expected one of {SUPPORTED_DEVICES!r}"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError(
            "device 'cuda' was requested but torch.cuda.is_available() is False"
        )
    return requested


def seed_everything(seed: int) -> None:
    """Seed torch's generators and request deterministic kernels.

    `decisions.md` Section 25: `torch.manual_seed` / `torch.cuda.manual_seed_all`
    from the run's seed, and `torch.use_deterministic_algorithms(True)`
    "wherever a deterministic CUDA kernel exists" -- hence `warn_only=True`,
    which keeps an op without a deterministic kernel from aborting the run.

    `torch.cuda.manual_seed_all` is called unconditionally: torch defers it
    until CUDA is initialized, so it is a safe no-op on a CPU-only host and
    the CUDA stream is seeded from the same top-level seed (Requirement 5.4).
    The deterministic-algorithms switch sits behind a capability check, and
    falls back to the strict form on a torch without `warn_only`.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Full fp32 on CUDA (Requirement 5.5). PyTorch lets cuDNN -- which runs
    # the GRU layers -- use TF32 by default on Ampere and newer GPUs, and
    # TF32's 10-bit mantissa moved CPU/CUDA final weights apart by ~1e-4 on
    # an H200 (5 full-batch epochs); with TF32 off they agree to ~1e-7.
    # Both switches are set explicitly; neither has any effect on CPU.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    use_deterministic = getattr(torch, "use_deterministic_algorithms", None)
    if use_deterministic is not None:
        try:
            use_deterministic(True, warn_only=True)
        except TypeError:  # pragma: no cover - torch < 1.11 has no warn_only
            use_deterministic(True)


# ---------------------------------------------------------------------------
# Typed split container (Requirements 4.7, 4.8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TrainingBatches:
    """The only patient data any training entry point receives.

    `dev` trains; `val` drives early stopping, checkpoint selection, and
    lambda selection. There is deliberately no third slot.

    Raises:
        ReferenceAccessError: if either slot holds a `"test"` batch.
        ValueError: if a slot holds the other slot's split, if both slots are
            the same object, or if the two batches were built to different
            dimensions.
    """

    dev: TensorBatch
    val: TensorBatch

    def __post_init__(self) -> None:
        for slot, batch in (("dev", self.dev), ("val", self.val)):
            if batch.split_name == "test":
                raise ReferenceAccessError(
                    f"a 'test' batch was passed as the {slot!r} batch of a training "
                    "run; training, early stopping, and selection read the "
                    "Development_Split and Validation_Split only (Requirement 4.8)"
                )
            if batch.split_name != slot:
                raise ValueError(
                    f"the {slot!r} slot requires a batch with split_name={slot!r}, "
                    f"got {batch.split_name!r}"
                )
        if self.dev is self.val:
            raise ValueError("dev and val must be distinct batches")
        if self.dev.dimensions != self.val.dimensions:
            raise ValueError(
                f"dev and val dimensions disagree: {self.dev.dimensions!r} vs "
                f"{self.val.dimensions!r}"
            )


def _tensor_field_names() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in dataclasses.fields(TensorBatch)
        if field.name not in ("split_name", "patient_ids", "dimensions")
    )


def batch_to_device(batch: TensorBatch, device: str) -> TensorBatch:
    """`batch` with every tensor on `device` (the same object on a no-op move)."""
    moved = {name: getattr(batch, name).to(device) for name in _tensor_field_names()}
    if all(moved[name] is getattr(batch, name) for name in moved):
        return batch
    return dataclasses.replace(batch, **moved)


# ---------------------------------------------------------------------------
# Model adapter: factual outputs for either family
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FactualOutputs:
    """What the loss reads from one forward pass, under each row's observed arm.

    Attributes:
        state_pred: `(N, LANDMARK_COUNT, 2)` CD4/CD8 predictions.
        hazard_logits: `(N, SURVIVAL_BIN_COUNT)`.
        hazards: `(N, SURVIVAL_BIN_COUNT)`, `sigmoid(hazard_logits)`.
        representation: `(N, hidden_size)`, the final-landmark hidden state
            (see module docstring, "Representation for the balance term").
    """

    state_pred: Tensor
    hazard_logits: Tensor
    hazards: Tensor
    representation: Tensor


def factual_outputs(model: nn.Module, batch: TensorBatch) -> FactualOutputs:
    """Run `model` on `batch` and return each row's factual-arm outputs.

    `TorchSequenceGRU` consumes the observed arm as an input column, so its
    single forward is already factual. `TorchInterventionAwareModel` rolls
    out both arms and each row takes the rollout matching its observed
    `treatment_arm`.

    Raises:
        TypeError: for any other model type.
    """
    if isinstance(model, TorchSequenceGRU):
        output = model(batch)
        return FactualOutputs(
            state_pred=output.state_pred,
            hazard_logits=output.hazard_logits,
            hazards=output.hazards,
            representation=output.hidden_states[:, -1, :],
        )
    if isinstance(model, TorchInterventionAwareModel):
        pair = model.rollout_both_arms(batch)
        treated = batch.treatment_arm == 1.0

        def pick(arm_1: Tensor, arm_0: Tensor) -> Tensor:
            mask = treated.reshape((-1,) + (1,) * (arm_1.dim() - 1))
            return torch.where(mask, arm_1, arm_0)

        hazard_logits = pick(pair.arm_1.hazard_logits, pair.arm_0.hazard_logits)
        return FactualOutputs(
            state_pred=pick(pair.arm_1.state_pred, pair.arm_0.state_pred),
            hazard_logits=hazard_logits,
            hazards=torch.sigmoid(hazard_logits),
            representation=pick(
                pair.arm_1.hidden_states[:, -1, :], pair.arm_0.hidden_states[:, -1, :]
            ),
        )
    raise TypeError(
        "train_one_model supports TorchSequenceGRU and TorchInterventionAwareModel, "
        f"got {type(model).__name__}"
    )


# ---------------------------------------------------------------------------
# Differentiable loss terms
# ---------------------------------------------------------------------------


def survival_nll_tensor(
    hazard_logits: Tensor, bin_index: Tensor, event_indicator: Tensor
) -> Tensor:
    """Mean discrete-time censored NLL over the batch (see module docstring).

    Args:
        hazard_logits: `(N, K)` pre-sigmoid hazards.
        bin_index: `(N,)` int64 in `0..K-1`.
        event_indicator: `(N,)` bool.
    """
    n_bins = hazard_logits.shape[1]
    positions = torch.arange(n_bins, device=hazard_logits.device).unsqueeze(0)
    bins = bin_index.unsqueeze(1)
    survived = (positions < bins).to(hazard_logits.dtype)
    at_event = ((positions == bins) & event_indicator.unsqueeze(1)).to(hazard_logits.dtype)
    log_survive = functional.logsigmoid(-hazard_logits)
    log_event = functional.logsigmoid(hazard_logits)
    per_patient = -(survived * log_survive).sum(dim=1) - (at_event * log_event).sum(dim=1)
    return per_patient.mean()


def _huber_elementwise(residual: Tensor, delta: float) -> Tensor:
    abs_residual = residual.abs()
    quadratic = 0.5 * residual * residual
    linear = delta * (abs_residual - 0.5 * delta)
    return torch.where(abs_residual <= delta, quadratic, linear)


def masked_huber_tensor(
    state_pred: Tensor,
    cd4: Tensor,
    cd8: Tensor,
    cd4_mask: Tensor,
    cd8_mask: Tensor,
    delta: float = HUBER_DELTA,
) -> Tensor:
    """Masked CD4/CD8 Huber loss, mask applied before reduction (Req 4.2).

    Args:
        state_pred: `(N, L, 2)` predictions, `[..., 0]` CD4 and `[..., 1]` CD8.
        cd4, cd8: `(N, L)` targets (placeholder at missing entries).
        cd4_mask, cd8_mask: `(N, L)` bool, `True = observed`.
    """
    targets = torch.stack((cd4, cd8), dim=-1)
    mask = torch.stack((cd4_mask, cd8_mask), dim=-1).to(state_pred.dtype)
    per_element = _huber_elementwise(state_pred - targets, delta) * mask
    observed_positions = (cd4_mask | cd8_mask).sum()
    if int(observed_positions) == 0:
        return per_element.sum() * 0.0
    return per_element.sum() / observed_positions.to(state_pred.dtype)


def mmd_tensor(representation: Tensor, treatment_arm: Tensor) -> Tensor:
    """Linear-kernel MMD `||mean(z_treated) - mean(z_control)||^2`.

    Zero (with a graph attached) when either group is empty, as
    `models.losses.mmd` returns `0.0` then.
    """
    treated = treatment_arm == 1.0
    control = ~treated
    if not bool(treated.any()) or not bool(control.any()):
        return representation.sum() * 0.0
    difference = representation[treated].mean(dim=0) - representation[control].mean(dim=0)
    return (difference * difference).sum()


@dataclass(frozen=True, eq=False)
class LossTerms:
    """The three objective terms and their frozen combination."""

    total: Tensor
    survival: Tensor
    state: Tensor
    balance: Tensor


def compute_loss_terms(
    model: nn.Module, batch: TensorBatch, lambda_state: float, lambda_balance: float
) -> LossTerms:
    """`survival_NLL + lambda_state * masked_huber + lambda_balance * MMD` on `batch`."""
    outputs = factual_outputs(model, batch)
    survival = survival_nll_tensor(outputs.hazard_logits, batch.bin_index, batch.event_indicator)
    state = masked_huber_tensor(
        outputs.state_pred, batch.cd4, batch.cd8, batch.cd4_mask, batch.cd8_mask
    )
    balance = mmd_tensor(outputs.representation, batch.treatment_arm)
    return LossTerms(
        total=total_loss(survival, state, balance, lambda_state, lambda_balance),
        survival=survival,
        state=state,
        balance=balance,
    )


# ---------------------------------------------------------------------------
# Checkpoints (decisions.md Section 25: plain dict via torch.save)
# ---------------------------------------------------------------------------


def save_torch_checkpoint(
    path: str | Path,
    *,
    model_state_dict: dict[str, Tensor],
    optimizer_state_dict: dict[str, Any],
    epoch: int,
    seed: int,
    val_loss: float,
) -> Path:
    """Write the frozen plain-dict checkpoint via `torch.save`.

    The payload is exactly `{"model_state_dict", "optimizer_state_dict",
    "epoch", "seed", "val_loss"}` -- never a pickled model object, the same
    "plain dict, not a class instance" convention as
    `models.training.save_checkpoint`. That function writes JSON floats and
    so cannot hold tensors; this is its tensor counterpart. No wall-clock
    timestamp is stored, so identical training yields identical bytes.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "epoch": int(epoch),
            "seed": int(seed),
            "val_loss": float(val_loss),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint written by `save_torch_checkpoint`.

    `weights_only=True`: the payload is tensors and primitives only, so the
    restricted unpickler suffices and no arbitrary object is ever
    reconstructed from disk.
    """
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def checkpoint_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a checkpoint file's bytes (bound into tokens/audits)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# train_one_model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TrainingResult:
    """One training run's outcome.

    `model` holds the weights of `best_epoch` (the validation-loss minimum),
    in `eval()` mode. Epochs are 1-based: epoch `e` is the state after `e`
    optimizer steps.
    """

    model: nn.Module
    seed: int
    device: str
    lambda_state: float
    lambda_balance: float
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    stopped_early: bool
    train_loss_curve: tuple[float, ...]
    val_loss_curve: tuple[float, ...]
    checkpoint_path: Path | None
    checkpoint_sha256: str | None


def train_one_model(
    model: nn.Module,
    batches: TrainingBatches,
    *,
    lambda_state: float,
    lambda_balance: float,
    seed: int,
    device: str = "cpu",
    config: TrainingConfig = FROZEN_TRAINING_CONFIG,
    checkpoint_path: str | Path | None = None,
    manifest: ManifestWriter | None = None,
    manifest_label: str = "training",
) -> TrainingResult:
    """Train `model` on `batches.dev`, early-stopping on `batches.val`.

    Full-batch Adam with the frozen configuration; each epoch is one
    optimizer step on the development loss followed by an `eval()`-mode,
    no-grad validation loss. `EarlyStopping` decides when to stop, and the
    weights of the validation-minimizing epoch are restored into `model`
    and, if `checkpoint_path` is given, written there.

    With `manifest`, the finished run is recorded into it through
    `record_training_manifest(manifest, result, label=manifest_label)`
    (Requirements 4.9, 5.7).

    Raises:
        UnsupportedDeviceError / DeviceUnavailableError: from
            `resolve_device`, before the model or batches are moved.
        TypeError: if `batches` is not a `TrainingBatches`.
        TrainingDivergedError: if the development loss becomes non-finite.
    """
    if not isinstance(batches, TrainingBatches):
        raise TypeError(
            f"batches must be a TrainingBatches(dev, val), got {type(batches).__name__}"
        )
    device = resolve_device(device)
    seed_everything(seed)

    model = model.to(device)
    dev = batch_to_device(batches.dev, device)
    val = batch_to_device(batches.val, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    stopper = EarlyStopping(patience=config.patience)

    train_curve: list[float] = []
    val_curve: list[float] = []
    best_epoch = 0
    best_val_loss = math.inf
    best_model_state: dict[str, Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_terms = compute_loss_terms(model, dev, lambda_state, lambda_balance)
        train_value = float(train_terms.total.detach())
        if not math.isfinite(train_value):
            raise TrainingDivergedError(
                f"development loss is non-finite ({train_value!r}) at epoch {epoch}"
            )
        train_terms.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_value = float(compute_loss_terms(model, val, lambda_state, lambda_balance).total)

        train_curve.append(train_value)
        val_curve.append(val_value)

        should_stop = stopper.should_stop(val_value)
        if stopper.epochs_since_improvement == 0:
            best_epoch = epoch
            best_val_loss = val_value
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        if should_stop:
            stopped_early = True
            break

    if best_model_state is None or best_optimizer_state is None:
        # Every validation loss was non-finite: keep the final weights so the
        # run is still inspectable, and report the non-finite minimum.
        best_epoch = len(val_curve)
        best_val_loss = val_curve[-1]
        best_model_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()

    written: Path | None = None
    digest: str | None = None
    if checkpoint_path is not None:
        written = save_torch_checkpoint(
            checkpoint_path,
            model_state_dict={k: v.detach().cpu() for k, v in best_model_state.items()},
            optimizer_state_dict=best_optimizer_state,
            epoch=best_epoch,
            seed=seed,
            val_loss=best_val_loss,
        )
        digest = checkpoint_sha256(written)

    result = TrainingResult(
        model=model,
        seed=seed,
        device=device,
        lambda_state=float(lambda_state),
        lambda_balance=float(lambda_balance),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        epochs_run=len(val_curve),
        stopped_early=stopped_early,
        train_loss_curve=tuple(train_curve),
        val_loss_curve=tuple(val_curve),
        checkpoint_path=written,
        checkpoint_sha256=digest,
    )
    if manifest is not None:
        record_training_manifest(manifest, result, label=manifest_label, config=config)
    return result


# ---------------------------------------------------------------------------
# Manifest recording (Requirements 4.9, 5.7)
# ---------------------------------------------------------------------------


def record_training_manifest(
    manifest: ManifestWriter,
    result: TrainingResult,
    *,
    label: str = "training",
    config: TrainingConfig | None = None,
) -> None:
    """Record one finished run into `manifest` through its existing API.

    Uses only `set_model_config`, `add_checkpoint_path`, `add_metric`, and
    `set_device`; the `ManifestWriter` schema is unchanged. What lands where:

    - `model_config[label]`: `device`, `seed`, `lambda_state`,
      `lambda_balance`, `selected_epoch`, `checkpoint_sha256`, and the
      training configuration. Any existing `model_config` entries under other
      keys are preserved (`set_model_config` replaces, so this merges first).
    - `metrics[f"{label}.<name>"]`: `train_loss_curve` and `val_loss_curve`
      (per-epoch lists), `selected_epoch`, `best_val_loss`, `epochs_run`,
      `stopped_early`, `lambda_state`, `lambda_balance`, `checkpoint_sha256`.
    - `checkpoint_paths`: the checkpoint file, when one was written.
    - `device`: `result.device` via `set_device` (Requirement 5.7).

    `label` keeps several runs in one manifest apart (e.g. one per replica).
    A run without a checkpoint records `checkpoint_sha256=None` explicitly
    rather than omitting it.
    """
    if not label or not label.strip():
        raise ValueError("label must be a non-empty string")
    record: dict[str, Any] = {
        "device": result.device,
        "seed": int(result.seed),
        "lambda_state": float(result.lambda_state),
        "lambda_balance": float(result.lambda_balance),
        "selected_epoch": int(result.best_epoch),
        "checkpoint_sha256": result.checkpoint_sha256,
    }
    if config is not None:
        record["training_config"] = {
            "optimizer": "adam",
            "learning_rate": config.learning_rate,
            "betas": list(config.betas),
            "weight_decay": config.weight_decay,
            "grad_clip_norm": config.grad_clip_norm,
            "patience": config.patience,
            "max_epochs": config.max_epochs,
            "is_frozen": config.is_frozen,
        }
    manifest.set_model_config({**manifest.model_config, label: record})

    if result.checkpoint_path is not None:
        manifest.add_checkpoint_path(str(result.checkpoint_path))

    metrics: dict[str, Any] = {
        "train_loss_curve": list(result.train_loss_curve),
        "val_loss_curve": list(result.val_loss_curve),
        "selected_epoch": int(result.best_epoch),
        "best_val_loss": float(result.best_val_loss),
        "epochs_run": int(result.epochs_run),
        "stopped_early": bool(result.stopped_early),
        "lambda_state": float(result.lambda_state),
        "lambda_balance": float(result.lambda_balance),
        "checkpoint_sha256": result.checkpoint_sha256,
    }
    for name, value in metrics.items():
        manifest.add_metric(f"{label}.{name}", value)

    manifest.set_device(result.device)


# ---------------------------------------------------------------------------
# Validation scoring and lambda selection (Requirement 4.3)
# ---------------------------------------------------------------------------

#: `(risk_scores, event_times, event_indicators)` -- the triple
#: `evaluation.metrics.concordance_index` consumes.
ScoringTriple = tuple[tuple[float, ...], tuple[float, ...], tuple[bool, ...]]


def validation_scoring_triple(model: nn.Module, val: TensorBatch) -> ScoringTriple:
    """Score `val` for composite-endpoint concordance.

    Risk is the factual-arm 96-week cumulative event probability
    `1 - prod(1 - h_k)` (`outcome_from_hazards`); times and indicators are
    `val.event_time_weeks` / `val.event_indicator`. Requires a `"val"`
    batch, so a development or test batch cannot be scored for selection.
    """
    if val.split_name != "val":
        if val.split_name == "test":
            raise ReferenceAccessError(
                "lambda selection scored a 'test' batch; selection reads the "
                "Validation_Split only (Requirement 4.8)"
            )
        raise ValueError(
            f"selection scores the validation split only, got split_name={val.split_name!r}"
        )
    model.eval()
    with torch.no_grad():
        risk = outcome_from_hazards(factual_outputs(model, val).hazards)
    return (
        tuple(float(value) for value in risk.cpu().tolist()),
        tuple(float(value) for value in val.event_time_weeks.cpu().tolist()),
        tuple(bool(value) for value in val.event_indicator.cpu().tolist()),
    )


@dataclass(frozen=True)
class LambdaSelection:
    """The outcome of `select_lambdas`.

    Attributes:
        selected: the chosen `(lambda_state, lambda_balance)`.
        grid: the nine combinations evaluated, in `load_lambda_grid` order.
        seed: the seed every candidate was trained with.
        scored_split: always `"val"`.
        candidate_triples: each combination's validation scoring triple,
            exactly what `select_lambda_by_validation_concordance` received.
    """

    selected: tuple[float, float]
    grid: tuple[tuple[float, float], ...]
    seed: int
    scored_split: str
    candidate_triples: dict[tuple[float, float], ScoringTriple]


ModelFactory = Callable[[], nn.Module]


def select_lambdas(
    model_factory: ModelFactory,
    batches: TrainingBatches,
    seed: int,
    device: str = "cpu",
    *,
    config: TrainingConfig = FROZEN_TRAINING_CONFIG,
    grid: Sequence[tuple[float, float]] | None = None,
) -> LambdaSelection:
    """Evaluate the frozen 3x3 grid and pick by validation concordance.

    For each combination: `seed_everything(seed)`, construct a fresh model,
    train it on `batches.dev` with early stopping on `batches.val`, and score
    `batches.val`. The nine validation triples go to
    `select_lambda_by_validation_concordance` unchanged.

    `grid` defaults to `load_lambda_grid()`; it is a parameter only so a
    test can inject the loaded grid. Any grid that is not nine combinations
    is rejected.
    """
    device = resolve_device(device)
    combinations = tuple(
        (float(a), float(b)) for a, b in (load_lambda_grid() if grid is None else grid)
    )
    if len(combinations) != LAMBDA_GRID_SIZE or len(set(combinations)) != LAMBDA_GRID_SIZE:
        raise ValueError(
            f"the frozen lambda grid has {LAMBDA_GRID_SIZE} distinct combinations "
            f"(decisions.md Section 14.3), got {combinations!r}"
        )

    candidate_triples: dict[tuple[float, float], ScoringTriple] = {}
    for lambda_state, lambda_balance in combinations:
        seed_everything(seed)
        result = train_one_model(
            model_factory(),
            batches,
            lambda_state=lambda_state,
            lambda_balance=lambda_balance,
            seed=seed,
            device=device,
            config=config,
        )
        candidate_triples[(lambda_state, lambda_balance)] = validation_scoring_triple(
            result.model, batch_to_device(batches.val, device)
        )

    selected = select_lambda_by_validation_concordance(candidate_triples)
    return LambdaSelection(
        selected=selected,
        grid=combinations,
        seed=seed,
        scored_split="val",
        candidate_triples=candidate_triples,
    )


# ---------------------------------------------------------------------------
# Ensemble (Requirement 4.6)
# ---------------------------------------------------------------------------


def replica_seeds(master_seed: int, n_replicas: int = N_ENSEMBLE_REPLICAS) -> tuple[int, ...]:
    """`derive_seed(master_seed, f"replica_{i}")` for `i` in `0..n-1`.

    The same derivation `uncertainty.ensemble.DeepEnsemble.replica_seeds`
    uses, so the harness and the ensemble agree on which seed produced
    which replica.
    """
    if n_replicas <= 0:
        raise ValueError(f"n_replicas must be positive, got {n_replicas!r}")
    return tuple(derive_seed(master_seed, f"replica_{i}") for i in range(n_replicas))


@dataclass(frozen=True, eq=False)
class EnsembleResult:
    """`train_ensemble`'s output: the selection and the trained replicas."""

    master_seed: int
    replica_seeds: tuple[int, ...]
    lambda_selection: LambdaSelection
    replicas: tuple[TrainingResult, ...]

    @property
    def models(self) -> tuple[nn.Module, ...]:
        return tuple(replica.model for replica in self.replicas)


def train_ensemble(
    model_factory: ModelFactory,
    batches: TrainingBatches,
    master_seed: int,
    device: str = "cpu",
    n_replicas: int = N_ENSEMBLE_REPLICAS,
    *,
    config: TrainingConfig = FROZEN_TRAINING_CONFIG,
    checkpoint_dir: str | Path | None = None,
) -> EnsembleResult:
    """Select lambdas once, then train `n_replicas` replicas at that pair.

    Lambda selection runs once, on a single replica -- replica 0's seed,
    `derive_seed(master_seed, "replica_0")`. Each replica `i` is then
    constructed after `seed_everything(seed_i)` and trained at the selected
    pair. With `checkpoint_dir`, replica `i`'s best-epoch checkpoint is
    written to `checkpoint_dir / f"replica_{i}.pt"`.
    """
    device = resolve_device(device)
    seeds = replica_seeds(master_seed, n_replicas)
    selection = select_lambdas(model_factory, batches, seeds[0], device, config=config)
    lambda_state, lambda_balance = selection.selected

    replicas: list[TrainingResult] = []
    for index, seed in enumerate(seeds):
        seed_everything(seed)
        replicas.append(
            train_one_model(
                model_factory(),
                batches,
                lambda_state=lambda_state,
                lambda_balance=lambda_balance,
                seed=seed,
                device=device,
                config=config,
                checkpoint_path=(
                    None if checkpoint_dir is None else Path(checkpoint_dir) / f"replica_{index}.pt"
                ),
            )
        )

    return EnsembleResult(
        master_seed=master_seed,
        replica_seeds=seeds,
        lambda_selection=selection,
        replicas=tuple(replicas),
    )
