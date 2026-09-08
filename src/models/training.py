"""Model-agnostic training infrastructure (Requirement 8.3).

This module provides the training-support machinery every predictive model
in `src/models` needs -- early stopping, checkpointing, class weighting,
a censoring-aware loss, a deterministic-evaluation convention, and seed
recording -- without importing any concrete model class. It is written
against a minimal duck-typed `TrainableModel` protocol (parameter
get/set via `flat_parameters()`/`load_flat_parameters()`, matching the
interface `models.sequence_baseline.SequenceGRU` and `GRUCell` already
expose) so the same helpers are reusable by the intervention-aware model
(task 9.x) once it exists, without this module depending on either.

Dependency scope note: as elsewhere in `src/models` and `src/causal`
(`pyproject.toml` declares no `numpy`/`torch`), this module is pure
standard library (`json`, `math`, `time`, `dataclasses`).

Early-stopping / model-selection scope note (`decisions.md` Section 6):
the historical randomized-trial effect must never influence early
stopping, checkpoint selection, or any other training-time decision --
`EarlyStopping` here only ever sees a caller-supplied validation loss
computed from the observational development split.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from infra.seeding import derive_seed

# ---------------------------------------------------------------------------
# Trainable-model protocol (duck typing -- no import of any concrete model)
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainableModel(Protocol):
    """The minimal interface this module's helpers rely on.

    Matches the flat-parameter (de)serialization interface
    `models.sequence_baseline.SequenceGRU` already exposes (and that any
    future intervention-aware model should expose to reuse this module),
    rather than a `get_params()`/`set_params()` naming convention.
    """

    def flat_parameters(self) -> Sequence[float]: ...

    def load_flat_parameters(self, flat: Sequence[float]) -> None: ...


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopping:
    """Tracks validation loss across epochs and signals when to stop.

    `should_stop(validation_loss)` is called once per epoch with that
    epoch's validation loss. It returns `True` once `patience` consecutive
    epochs have failed to improve on the best validation loss seen so far
    by at least `min_delta`; otherwise it returns `False` (and, if the
    epoch improved on the best loss, resets the non-improvement counter).

    Args:
        patience: number of consecutive non-improving epochs tolerated
            before signalling a stop. Must be a positive integer.
        min_delta: the minimum decrease in validation loss, relative to
            the best loss seen so far, required to count as an
            improvement. Defaults to `0.0` (any strict decrease counts).
    """

    patience: int
    min_delta: float = 0.0
    best_loss: float = field(default=math.inf, init=False)
    epochs_since_improvement: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")

    def should_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.epochs_since_improvement = 0
            return False

        self.epochs_since_improvement += 1
        return self.epochs_since_improvement >= self.patience


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    model_params: Any,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Serialize `model_params` (and optional `metadata`) to `path` as JSON.

    `model_params` is expected to be a JSON-serializable structure of
    plain floats (e.g. the flat parameter list `SequenceGRU.flat_parameters()`
    returns, or a dict of such lists) -- this function does not know about
    any specific model class. Python's `json` module renders floats via
    their shortest round-trip `repr`, so `load_checkpoint` recovers the
    exact same float values (bit-for-bit), not merely an approximation.

    Creates parent directories if they do not already exist.
    """
    payload = {"model_params": model_params, "metadata": metadata or {}}
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, sort_keys=True))


def load_checkpoint(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a checkpoint written by `save_checkpoint`.

    Returns:
        A `(model_params, metadata)` tuple, exactly as passed to
        `save_checkpoint` (metadata is `{}` if none was supplied).
    """
    payload = json.loads(Path(path).read_text())
    return payload["model_params"], payload["metadata"]


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------


def compute_class_weights(labels: Sequence[int]) -> dict[int, float]:
    """Inverse-frequency class weights, `sklearn`'s `class_weight="balanced"`
    formula: `weight[c] = n_samples / (n_classes * count[c])`.

    Works for binary or small-cardinality integer label sets. Raises
    `ValueError` on an empty `labels` sequence (weights are undefined).
    """
    if not labels:
        raise ValueError("labels must not be empty")

    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    n_samples = len(labels)
    n_classes = len(counts)
    return {cls: n_samples / (n_classes * count) for cls, count in counts.items()}


# ---------------------------------------------------------------------------
# Censoring-aware loss: discrete-time survival negative log-likelihood
# ---------------------------------------------------------------------------


def discrete_time_censored_nll(
    hazards: Sequence[float],
    event_time_index: int | None,
    epsilon: float = 1e-9,
) -> float:
    """Discrete-time censored negative log-likelihood for one patient.

    Standard discrete-time survival loss (see e.g. the discrete-hazard
    formulation used throughout survival analysis): given a sequence of
    per-timestep hazard probabilities `h_0, ..., h_{T-1}` (as produced by
    a model's hazard head, e.g. `SequenceGRU.forward`'s
    `hazard_predictions`),

    - if the event occurs at timestep `t = event_time_index`, the patient
      survived timesteps `0..t-1` and then had the event at `t`:

          loss = -sum_{k=0}^{t-1} log(1 - h_k) - log(h_t)

    - if the patient is censored (no event observed through the last
      timestep, `event_time_index is None`), the patient is only known to
      have survived every observed timestep:

          loss = -sum_{k=0}^{T-1} log(1 - h_k)

    `epsilon` clamps `h_k` / `1 - h_k` away from `0` before taking `log`,
    so a saturated hazard prediction cannot produce `-inf`/`nan`.

    Raises:
        ValueError: if `hazards` is empty, or `event_time_index` is out of
            range for `hazards`.
    """
    if not hazards:
        raise ValueError("hazards must not be empty")
    if event_time_index is not None and not (0 <= event_time_index < len(hazards)):
        raise ValueError(
            f"event_time_index {event_time_index} out of range for "
            f"hazards of length {len(hazards)}"
        )

    survived_through = event_time_index if event_time_index is not None else len(hazards)

    loss = 0.0
    for k in range(survived_through):
        loss += -math.log(max(1.0 - hazards[k], epsilon))
    if event_time_index is not None:
        loss += -math.log(max(hazards[event_time_index], epsilon))
    return loss


def censoring_aware_loss(
    batch_hazards: Sequence[Sequence[float]],
    batch_event_time_indices: Sequence[int | None],
) -> float:
    """Mean `discrete_time_censored_nll` over a batch of patients.

    `batch_hazards[i]` is patient `i`'s hazard sequence;
    `batch_event_time_indices[i]` is that patient's event timestep index,
    or `None` if censored. Both sequences must have the same length.
    """
    if len(batch_hazards) != len(batch_event_time_indices):
        raise ValueError(
            "batch_hazards and batch_event_time_indices must have the same length, "
            f"got {len(batch_hazards)} and {len(batch_event_time_indices)}"
        )
    if not batch_hazards:
        raise ValueError("batch must not be empty")

    total = sum(
        discrete_time_censored_nll(hazards, event_time_index)
        for hazards, event_time_index in zip(
            batch_hazards, batch_event_time_indices, strict=True
        )
    )
    return total / len(batch_hazards)


# ---------------------------------------------------------------------------
# Deterministic evaluation
# ---------------------------------------------------------------------------


def evaluate_deterministically(
    model: TrainableModel, forward_fn: Any, *args: Any, **kwargs: Any
) -> Any:
    """Run `forward_fn(*args, **kwargs)` as a deterministic evaluation call.

    Convention, not machinery: every model this module is designed for
    (`SequenceGRU.forward`, and any future intervention-aware model built
    to the same `TrainableModel` protocol) is a pure function of its
    current parameters and its inputs -- no dropout, no sampling, no
    hidden RNG draws inside `forward`. Evaluation is therefore already
    deterministic given fixed parameters, *provided* the caller does not
    interleave a `train_step`-style parameter update between the forward
    passes being compared. This helper exists to make that convention
    explicit and give evaluation call sites a single, documented place to
    call through, rather than adding unnecessary state or a context
    manager where none is needed.

    `model` is accepted (and typed against `TrainableModel`) purely to
    make the caller's intent legible; it is not otherwise used, since
    `forward_fn` is expected to be a bound method of `model` (e.g.
    `model.forward`).
    """
    return forward_fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Seed recording
# ---------------------------------------------------------------------------


def derive_training_seed(master_seed: int, run_label: str = "training") -> int:
    """Derive the seed for one training run via `infra.seeding.derive_seed`.

    Reuses the single frozen HMAC-SHA-256 derivation rule (Requirement
    1.6) rather than deriving a training seed ad hoc, so a checkpoint's
    recorded seed is always reproducible from `(master_seed, run_label)`.
    """
    return derive_seed(master_seed, run_label)


def make_checkpoint_metadata(
    *,
    epoch: int,
    master_seed: int,
    run_label: str = "training",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard checkpoint-metadata dict: `epoch`, the derived
    training `seed`, the `master_seed` it was derived from, and a wall-clock
    `timestamp`. `extra` fields (if any) are merged in, so callers can add
    e.g. a training-loss value without this function needing to know about
    every possible metadata field. Every checkpoint written via
    `save_checkpoint` using this metadata therefore always records which
    seed produced it, per Requirement 8.3.
    """
    metadata: dict[str, Any] = {
        "epoch": epoch,
        "seed": derive_training_seed(master_seed, run_label),
        "master_seed": master_seed,
        "timestamp": time.time(),
    }
    if extra:
        metadata.update(extra)
    return metadata
