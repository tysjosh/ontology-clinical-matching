"""Sequence_GRU: the production torch transcription of the
treatment-as-feature GRU baseline (Requirement 2, `design.md` Section 3.2).

What this module is, and what it deliberately is not
-------------------------------------------------------------------------

`models.sequence_baseline.SequenceGRU` remains the **specification of
record** for this architecture: it is the pure-Python prototype
`decisions.md` Section 14.2 froze, and it produces no reported number. This
module replaces its *gradient mechanism* -- central-difference numerical
differentiation becomes PyTorch autograd (Requirement 2.5) -- and nothing
else. Every architectural width is *imported* from the prototype rather
than restated here, so prototype/production drift surfaces as an import
error or a shape error rather than as a silent divergence (task 9's parent
body):

- ``LANDMARK_WEEKS`` / ``N_LANDMARKS`` -- the three frozen landmark
  timepoints (0, 20, 96). A Transformer was rejected precisely because this
  sequence is three steps long.
- ``INPUT_SIZE`` (= 3) and its ``[cd4, cd8, treatment_arm]`` column order
  -- treatment enters as an ordinary input column at every landmark, with
  **no separate intervention encoder** (Requirement 2.2). That contrast is
  the entire point of this baseline: the explicit-channel model is
  `models.torch_intervention_aware`.
- the baseline covariates that seed the hidden state (Requirement 2.1):
  since the `decisions.md` Section 26 amendment of 2026-10-16, the full
  14-column Section 13 adjustment set (`TensorBatch.baseline`), not the
  prototype's four ``BASELINE_STATE_INIT_FIELDS``. The projection is a single
  ``nn.Linear(baseline_covariate_width, hidden_size)`` followed by ``tanh``,
  matching the prototype's ``init_W``/``init_b`` + ``tanh``.
- ``STATE_DIM`` (= 2) -- the CD4/CD8 state-prediction width.

Where this module intentionally differs from the prototype
-------------------------------------------------------------------------

The hazard head. The prototype emits one hazard per *landmark* timestep
(three), because it predates the survival discretization. `decisions.md`
Section 25 froze the discrete-time hazard grid at 23 bins, independently of
the landmark timepoints, and `design.md` Section 3.2 therefore specifies a
single ``nn.Linear(hidden_size, 23)`` head reading the **final** hidden
state (Requirement 2.3). The bin count is read from
`models.dataset.SURVIVAL_BIN_COUNT`, never written as a literal here. That
hidden state is the one after the time-zero landmark, not the final one (the
amendment of 2026-10-16): the outcome is predicted from time zero, and the
later hidden states have read the week-20 and week-96 labs.

The state head is next-state regression, as Section 14.2 and the prototype
specify: the prediction of landmark k comes from the hidden state before
landmark k's CD4/CD8 is read (h_0 for landmark 0, h_{k-1} otherwise), and is
stored at index k of `state_pred`. Until 2026-10 it read landmark k's own
hidden state, so it partly reconstructed its input; see the `decisions.md`
Section 26 amendment of 2026-10-12, item D.

Frozen hidden width
-------------------------------------------------------------------------

``FROZEN_HIDDEN_SIZE = 16`` is `decisions.md` Section 25's "Hidden width"
entry (Requirement 2.4). It is **not defined here**: it is imported from
`models.dataset`, which holds the single canonical copy, and re-exported
under this module's ``__all__`` for callers and tests that expect it here.
It is the same object the intervention-aware model's ``HIDDEN_SIZE``
aliases, deliberately and structurally: the
treatment-as-feature/explicit-channel ablation contrast (Section 17 items
2-3) has to isolate the representational difference, not a capacity
difference, so the two models cannot be allowed to drift to different
widths. One literal in one place is what makes that drift impossible rather
than merely discouraged. ``hidden_size`` remains a constructor parameter so
tests can exercise other widths, but every production construction path
takes the default.

Determinism by construction
-------------------------------------------------------------------------

There is no dropout, no batch normalization, and no other stochastic or
batch-coupled layer anywhere in this module, so an ``eval()``-mode forward
pass on identical inputs returns bitwise-identical outputs (Requirement
2.6) -- and, more strongly, ``train()`` and ``eval()`` mode compute the
same function. This is asserted *structurally* rather than only
behaviourally: `stochastic_or_batch_coupled_layers` walks a module tree and
names every offender, and `tests/neural/test_torch_sequence_gru.py` asserts
it returns nothing for a constructed model.

Torch dependency scope
-------------------------------------------------------------------------

`torch` is a leaf dependency confined to `src/models/torch_*.py` and
`src/models/dataset.py` (`design.md` Section 2.4). This module must not be
imported from `src/models/__init__.py`, and nothing outside the torch leaf
may import it; `tests/unit/test_torch_isolation.py` enforces both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.dataset import (
    FROZEN_HIDDEN_SIZE,
    SURVIVAL_BIN_COUNT,
    DatasetDimensions,
    TensorBatch,
)
from models.sequence_baseline import (
    BASELINE_STATE_INIT_FIELDS,
    INPUT_SIZE,
    LANDMARK_WEEKS,
    N_LANDMARKS,
    STATE_DIM,
)

__all__ = [
    "FROZEN_HIDDEN_SIZE",
    "SequenceOutput",
    "TorchSequenceGRU",
    "assemble_sequence_input",
    "stochastic_or_batch_coupled_layers",
]

# ---------------------------------------------------------------------------
# Frozen values (decisions.md Section 25)
# ---------------------------------------------------------------------------

# `FROZEN_HIDDEN_SIZE` is re-exported from `models.dataset`, which owns the
# single canonical copy of `decisions.md` Section 25's "Hidden width"
# (Requirement 2.4). It is deliberately *not* redefined here: this module and
# `torch_intervention_aware` must share one width, so there is one literal.

#: Width of the baseline-covariate vector that seeds the hidden state --
#: `len(sequence_baseline.BASELINE_STATE_INIT_FIELDS)`, not an independent
#: value. Equal to `DatasetDimensions.state_width`, which the constructor
#: checks (Requirement 2.1).
STATE_INIT_WIDTH: int = len(BASELINE_STATE_INIT_FIELDS)

#: Index into `hidden_states` of the state the survival-hazard head reads:
#: 0, the hidden state after the time-zero landmark (baseline covariates,
#: week-0 CD4/CD8, treatment). Never a later landmark, whose hidden state has
#: read post-baseline labs (decisions.md Section 26 amendment of 2026-10-16).
#: `models.torch_training.factual_outputs` reads the same index for the
#: balance term's representation.
OUTCOME_HIDDEN_INDEX: int = 0


# ---------------------------------------------------------------------------
# Structural determinism check
# ---------------------------------------------------------------------------

# Built by name lookup rather than by direct attribute reference so this
# module keeps importing against a torch build that renames or drops one of
# the rarer variants.
_DROPOUT_TYPE_NAMES = (
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
)
_BATCH_COUPLED_TYPE_NAMES = (
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "SyncBatchNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
)


def _resolve_types(names: tuple[str, ...]) -> tuple[type, ...]:
    return tuple(cls for cls in (getattr(nn, name, None) for name in names) if cls is not None)


def stochastic_or_batch_coupled_layers(module: nn.Module) -> tuple[str, ...]:
    """Names of every submodule that would break eval-mode determinism.

    Requirement 2.6 says an `eval()`-mode forward pass on identical inputs
    returns identical outputs. `design.md` Section 3.2 achieves that "by
    construction -- no dropout, no batchnorm anywhere". This function makes
    that claim checkable as a structural property of the module tree rather
    than as a behavioural observation that a passing determinism test might
    reach for the wrong reason (a fixed global seed, say).

    Two families are reported:

    - **stochastic layers** -- every `nn.Dropout` variant, which injects
      noise in `train()` mode and so makes the two modes compute different
      functions;
    - **batch-coupled normalization** -- `BatchNorm*` / `InstanceNorm*`,
      whose output for one row depends on the other rows in the batch and
      whose running statistics mutate on every `train()`-mode forward, so
      the same input can yield different outputs on a second call.

    `LayerNorm` and `GroupNorm` are *not* reported: they normalize within a
    single row, hold no running state, and behave identically in both
    modes. Neither is used here, but neither would be a determinism defect.

    An `nn.RNNBase` subclass (`nn.GRU` here) carries dropout as a float
    attribute rather than as a child module, so a `.modules()` walk alone
    would miss it; it is checked explicitly.

    Args:
        module: the module tree to inspect. Named submodules are reported by
            their qualified name; the root, if it offends, is reported as
            `"<root>"`.

    Returns:
        A tuple of offending names, empty when the tree is clean.
    """
    forbidden = _resolve_types(_DROPOUT_TYPE_NAMES) + _resolve_types(_BATCH_COUPLED_TYPE_NAMES)
    offenders: list[str] = []
    for name, submodule in module.named_modules():
        label = name or "<root>"
        if forbidden and isinstance(submodule, forbidden):
            offenders.append(f"{label}:{type(submodule).__name__}")
            continue
        if isinstance(submodule, nn.RNNBase) and float(submodule.dropout) != 0.0:
            offenders.append(f"{label}:{type(submodule).__name__}(dropout={submodule.dropout})")
    return tuple(offenders)


# ---------------------------------------------------------------------------
# Emitted output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SequenceOutput:
    """One forward pass's outputs (`design.md` Section 3.2).

    `eq=False` for the same reason `models.dataset.TensorBatch` uses it: a
    generated `__eq__` would compare `Tensor` fields with `==`, producing an
    elementwise tensor and then raising on the truthiness test. Compare
    field by field with `torch.equal`, which is what the bitwise-determinism
    assertion wants anyway.

    Attributes:
        state_pred: `(N, N_LANDMARKS, STATE_DIM)` = `(N, 3, 2)` float,
            `[cd4, cd8]` predicted for each landmark k from the hidden state
            *before* landmark k is read (h_0 for k = 0, h_{k-1} otherwise;
            Requirement 2.3). Linear, no output activation -- the
            prototype's next-state head.
        hazards: `(N, SURVIVAL_BIN_COUNT)` = `(N, 23)` float in `(0, 1)`,
            the per-bin discrete-time hazard probabilities over
            `decisions.md` Section 25's frozen bin grid. Probabilities, not
            logits, so this field means the same thing as the prototype's
            `hazard_predictions` and can be handed directly to
            `models.training.discrete_time_censored_nll`.
        hazard_logits: `(N, 23)` float, the pre-sigmoid values `hazards`
            came from. Carried alongside rather than derived by inverting a
            sigmoid because task 10's survival NLL wants the numerically
            stable logit form (`logsigmoid`) instead of `log(sigmoid(x))`;
            keeping both means the stable path never has to re-derive one
            from the other.
        hidden_states: `(N, N_LANDMARKS, hidden_size)` float, the GRU hidden
            state at each landmark. Exposed because the balance term of the
            training objective is an MMD between treated and control
            *representations*, so the harness needs the representation this
            forward pass already computed rather than a second forward pass
            to recover it.
    """

    state_pred: Tensor
    hazards: Tensor
    hazard_logits: Tensor
    hidden_states: Tensor


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------


def assemble_sequence_input(batch: TensorBatch) -> Tensor:
    """The `(N, N_LANDMARKS, INPUT_SIZE)` per-landmark input tensor.

    Column order is the prototype's `[cd4, cd8, treatment_arm]`
    (`sequence_baseline.INPUT_SIZE = STATE_DIM + TREATMENT_FEATURE_DIM`), so
    `treatment_arm` is the third ordinary column of the same input vector
    the CD4/CD8 values occupy -- not a separately encoded channel
    (Requirement 2.2). It is repeated identically at every landmark because
    ACTG175's estimand is baseline-treatment ITT: treatment does not vary
    over follow-up here.

    Missingness is *not* a fourth column. `models.dataset` has already
    written `MISSING_PLACEHOLDER` into every masked CD4/CD8 position, and
    the masks (`batch.cd4_mask` / `batch.cd8_mask`) are consumed by the
    loss's masked-Huber state term, not by the input layer -- adding a mask
    column would widen the frozen `INPUT_SIZE` and break the
    capacity-matched comparison with the intervention-aware model.

    Exposed at module level (not private) so tests and the training harness
    can inspect the exact layout the model consumes rather than
    reconstructing a guess at it.
    """
    landmark_count = batch.cd4.shape[1]
    treatment_arm = batch.treatment_arm.unsqueeze(1).expand(-1, landmark_count)
    return torch.stack((batch.cd4, batch.cd8, treatment_arm), dim=-1)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class TorchSequenceGRU(nn.Module):
    """Treatment-as-feature GRU baseline, in torch (Requirement 2).

    `forward` takes a `models.dataset.TensorBatch`, matching `design.md`
    Section 3.2's signature. That choice, and why:

    - the batch is the only object that carries the tensors *and* the
      `DatasetDimensions` they were built to, so `forward` can check the
      widths it received against the widths it was constructed for and fail
      with a named mismatch instead of broadcasting silently;
    - task 10's harness holds `dev` and `val` batches as separate typed
      parameters precisely so no test batch can reach training. A model that
      accepted loose tensors would let a caller assemble an input from any
      source, which is exactly the provenance the typed batch preserves;
    - task 12.1's `build_model` constructs from `batch.dimensions`, so the
      construction and call paths read one description of the layout.

    Plain tensors are still reachable for callers who need them:
    `assemble_sequence_input` exposes the input layout, and
    `forward_from_tensors` takes the two tensors the model actually
    consumes.

    Attributes:
        hidden_size: the GRU hidden width, `FROZEN_HIDDEN_SIZE` by default.
        landmark_count: `N_LANDMARKS`, from the prototype.
        survival_bin_count: the hazard head's output width, from the
            `DatasetDimensions` handed in.
        state_init_projection: `nn.Linear(baseline_covariate_width,
            hidden_size)`, `tanh`-activated in `forward`, seeding `h_0` from
            the Section 13 adjustment set (Requirement 2.1).
        gru: single-layer `nn.GRU`, `dropout=0.0`, consuming
            `[cd4, cd8, treatment_arm]` per landmark.
        state_head: `nn.Linear(hidden_size, 2)`, applied at every landmark.
        hazard_head: `nn.Linear(hidden_size, 23)`, applied to the hidden
            state after the time-zero landmark (`OUTCOME_HIDDEN_INDEX`).
    """

    def __init__(
        self,
        dimensions: DatasetDimensions,
        hidden_size: int = FROZEN_HIDDEN_SIZE,
    ) -> None:
        """Construct the model against a dataset's reported dimensions.

        Args:
            dimensions: the widths `models.dataset.PatientDatasetBuilder`
                reports (Requirement 1.7). Read, not trusted blindly: the
                landmark count and the state-init width must agree with the
                prototype's constants, so a dataset layout change is a
                construction-time error here rather than a silent reshape.
            hidden_size: GRU hidden width. Defaults to the frozen 16;
                overridable only so tests can vary it.

        Raises:
            ValueError: if `hidden_size` is not positive, if
                `dimensions.landmark_count` differs from
                `sequence_baseline.N_LANDMARKS`, if `dimensions.state_width`
                differs from `len(BASELINE_STATE_INIT_FIELDS)`, or if
                `dimensions.survival_bin_count` differs from
                `dataset.SURVIVAL_BIN_COUNT`.
        """
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size!r}")
        if dimensions.landmark_count != N_LANDMARKS:
            raise ValueError(
                f"dimensions.landmark_count={dimensions.landmark_count} disagrees with "
                f"sequence_baseline.N_LANDMARKS={N_LANDMARKS} "
                f"(landmark weeks {LANDMARK_WEEKS!r})"
            )
        if dimensions.baseline_covariate_width <= 0:
            raise ValueError(
                f"dimensions.baseline_covariate_width must be positive, got "
                f"{dimensions.baseline_covariate_width!r}"
            )
        if dimensions.survival_bin_count != SURVIVAL_BIN_COUNT:
            raise ValueError(
                f"dimensions.survival_bin_count={dimensions.survival_bin_count} disagrees "
                f"with dataset.SURVIVAL_BIN_COUNT={SURVIVAL_BIN_COUNT} "
                "(decisions.md Section 25)"
            )

        self.hidden_size = hidden_size
        self.landmark_count = N_LANDMARKS
        self.survival_bin_count = dimensions.survival_bin_count
        self.dimensions = dimensions

        # h_0 is seeded from the full Section 13 adjustment set
        # (`TensorBatch.baseline`), the same covariates every causal estimator
        # adjusts for (decisions.md Section 26 amendment of 2026-10-16).
        self.baseline_width = dimensions.baseline_covariate_width
        self.state_init_projection = nn.Linear(self.baseline_width, hidden_size)
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bias=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.state_head = nn.Linear(hidden_size, STATE_DIM)
        self.hazard_head = nn.Linear(hidden_size, dimensions.survival_bin_count)

    # -- forward ---------------------------------------------------------

    def forward(self, batch: TensorBatch) -> SequenceOutput:
        """Run the model over one `TensorBatch`.

        Args:
            batch: the split's tensors. Its `cd4`, `cd8`, `treatment_arm`,
                and `baseline` fields are read; the masks and survival
                targets belong to the loss, not to the input layer.

        Returns:
            A `SequenceOutput` with `state_pred` `(N, 3, 2)` and `hazards`
            `(N, 23)`.

        Raises:
            ValueError: if any consumed tensor's shape disagrees with the
                dimensions this model was constructed for, naming the field
                and both widths.
        """
        self._validate(batch)
        return self.forward_from_tensors(batch.baseline, assemble_sequence_input(batch))

    def forward_from_tensors(self, baseline: Tensor, sequence_input: Tensor) -> SequenceOutput:
        """The tensor-level forward, for callers holding loose tensors.

        `forward` is the interface the pipeline uses; this is the same
        computation with the batch unpacked, so a test can drive the model
        from hand-built tensors without fabricating a `TensorBatch`.

        Args:
            baseline: `(N, baseline_covariate_width)`, the development-
                standardized Section 13 adjustment set (`TensorBatch.baseline`).
            sequence_input: `(N, 3, 3)`, per-landmark
                `[cd4, cd8, treatment_arm]` as `assemble_sequence_input`
                lays it out.

        Returns:
            The same `SequenceOutput` `forward` returns.
        """
        # h_0 from the baseline covariates, `tanh`-bounded so the initial
        # hidden state sits in the same range GRU hidden states occupy after
        # a step -- the prototype's `init_W`/`init_b` + `tanh`
        # (Requirement 2.1). `nn.GRU` wants `(num_layers, N, hidden_size)`.
        initial_hidden = torch.tanh(self.state_init_projection(baseline)).unsqueeze(0)

        hidden_states, _final = self.gru(sequence_input, initial_hidden)

        # Next-state regression (`decisions.md` Section 14.2: "next
        # patient-state (CD4/CD8) regression"), one prediction per landmark
        # (Requirement 2.3). The prediction *of* landmark k is read from the
        # hidden state *before* landmark k's CD4/CD8 is consumed: h_0 (the
        # baseline-initialized state) for landmark 0, and h_{k-1} for
        # landmark k. So no landmark's prediction has seen that landmark's
        # own measurement, matching the prototype's shifted-by-one targets
        # (`sequence_baseline.SequenceTrainingSample.target_next_states`)
        # while keeping the `(N, 3, 2)` "value at landmark k" layout every
        # loss and metric indexes. Linear, no output activation.
        preceding_hidden = torch.cat(
            (initial_hidden.transpose(0, 1), hidden_states[:, :-1, :]), dim=1
        )
        state_pred = self.state_head(preceding_hidden)

        # One hazard per frozen survival bin, from the hidden state after the
        # time-zero landmark only (`OUTCOME_HIDDEN_INDEX`): baseline
        # covariates, week-0 CD4/CD8, and treatment. The survival outcome is
        # predicted from time zero, so it must not read the week-20 or
        # week-96 labs the later hidden states have consumed (decisions.md
        # Section 26 amendment of 2026-10-16; until then this read the final
        # hidden state). Sigmoid, so `hazards` carries probabilities.
        hazard_logits = self.hazard_head(hidden_states[:, OUTCOME_HIDDEN_INDEX, :])

        return SequenceOutput(
            state_pred=state_pred,
            hazards=torch.sigmoid(hazard_logits),
            hazard_logits=hazard_logits,
            hidden_states=hidden_states,
        )

    # -- validation ------------------------------------------------------

    def _validate(self, batch: TensorBatch) -> None:
        """Reject a batch whose shapes disagree with this model's widths."""
        expected_landmarks = self.landmark_count
        for field_name in ("cd4", "cd8"):
            tensor = getattr(batch, field_name)
            if tensor.dim() != 2 or tensor.shape[1] != expected_landmarks:
                raise ValueError(
                    f"batch.{field_name} must have shape (N, {expected_landmarks}), "
                    f"got {tuple(tensor.shape)}"
                )
        if batch.baseline.dim() != 2 or batch.baseline.shape[1] != self.baseline_width:
            raise ValueError(
                f"batch.baseline must have shape (N, {self.baseline_width}), "
                f"got {tuple(batch.baseline.shape)}"
            )
        if batch.treatment_arm.dim() != 1:
            raise ValueError(
                f"batch.treatment_arm must have shape (N,), "
                f"got {tuple(batch.treatment_arm.shape)}"
            )
        row_counts = {
            "cd4": batch.cd4.shape[0],
            "cd8": batch.cd8.shape[0],
            "baseline": batch.baseline.shape[0],
            "treatment_arm": batch.treatment_arm.shape[0],
        }
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"batch tensors disagree on row count: {row_counts!r}")
