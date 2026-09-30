"""Intervention_Aware_Model: the PyTorch transcription of
`models.intervention_aware` (Requirement 3, `design.md` Section 3.3;
`decisions.md` Section 14.3 for the architecture, Section 25 for the widths).

What this module is, and is not
-------------------------------------------------------------------------

It is not a second architecture. Every structural element and every width
comes from the pure-Python prototype, which stays in place as the
specification of record:

- the three separately-encoded channels and their concatenation order
  `z = [z_state; z_intervention; z_protocol]` -- `StateEncoder`,
  `InterventionEncoder`, `ProtocolEncoder`, `concatenate_channels`;
- the channel input widths, **imported** as
  `models.intervention_aware.STATE_INPUT_DIM` (4),
  `INTERVENTION_INPUT_DIM` (3), `PROTOCOL_INPUT_DIM` (2), never restated
  here, so a prototype/production divergence is an import error or a shape
  error rather than a silent drift (task 9's parent body);
- the GRU state-transition rollout over the three landmark timesteps, and
  its zero initial hidden state -- `GRUTransition.rollout`'s documented
  default, which is also why this model has no baseline-covariate hidden
  init (that role belongs to the state channel's contribution to `z`, and
  is what distinguishes this family from `TorchSequenceGRU`);
- the treatment-specific hazard heads and the shared CD4/CD8 state head --
  `models.branch_heads.BranchHead`'s `branch_1`/`branch_0` pair.

What it replaces is the *gradient mechanism* (the prototype's
central-difference numerical gradient becomes autograd) and the *batching*
(per-patient Python lists become `(N, ...)` tensors read off a
`models.dataset.TensorBatch`).

Widths: `hidden_size = 16` and the 4/3/2 channel widths are frozen in
`decisions.md` Section 25. Each encoder's *output* width equals its input
width, which is that section's own arithmetic: "three concatenated encoder
channels (`STATE_INPUT_DIM=4 + INTERVENTION_INPUT_DIM=3 +
PROTOCOL_INPUT_DIM=2 = 9`-wide input per landmark)". So the concatenated
representation is 9 wide in every mode (Correctness Property 13).

Which tensors feed which channel (and why)
-------------------------------------------------------------------------

- **state channel** <- `TensorBatch.state_init`, the `(N, 4)`
  `BASELINE_STATE_INIT_FIELDS` block. This is the prototype's documented
  default for `StateEncoder` ("the baseline-covariate vector, encoded once
  per patient and reused identically at every landmark timestep"), and the
  frozen `STATE_INPUT_DIM = 4` admits nothing else: the per-landmark
  CD4/CD8 pair is 2 wide, not 4, and the standardized `TensorBatch.baseline`
  block cannot supply these four fields either -- `baseline_cd8` is not in
  `causal.propensity.COVARIATE_FIELDS`. It is also the only choice that
  keeps the CD4/CD8 state head honest: feeding the observed CD4/CD8 at
  landmark `t` into the encoder would make the head that predicts CD4/CD8
  at landmark `t` partly a copy of its own input.

  Known limitation, recorded rather than worked around. `state_init` carries
  *raw* values (CD4 ~ 300, CD8 ~ 900, Karnofsky ~ 90, age ~ 35) --
  `models.dataset` standardizes the `baseline` block but emits `state_init`
  unstandardized -- so this encoder's `linear + tanh` saturates at +/-1 and
  its gradient is effectively zero. The state channel therefore contributes
  a near-constant per-patient signature rather than a trained projection.
  This is inherited, not introduced here: the prototype's `StateEncoder`
  and `SequenceGRU`'s baseline hidden-state init apply the same `tanh` to
  the same raw vector, so it is a property of the architecture frozen in
  `decisions.md` Sections 14.2/14.3 as applied to raw-scale inputs. Fixing
  it means either standardizing `state_init` in `models.dataset` (task 8's
  contract, and a change the sequence-GRU baseline would have to share to
  stay capacity- and input-comparable) or adding a normalization layer here
  (an unfrozen structural element, and the batchnorm family is ruled out by
  the determinism convention). Neither belongs to this task; it is flagged
  here so a reported intervention-aware result is not read as evidence that
  the state channel was trained.
- **intervention channel** <- `TensorBatch.intervention`
  (`[treatment_arm, dose_or_default, adherence_or_default]`), with the
  `treatment_arm` column replaced by the arm being rolled out. This is the
  one tensor that differs between the arm-0 and arm-1 rollouts
  (Requirement 3.6).
- **protocol channel** <- `TensorBatch.protocol`
  (`[is_eligible, is_time_zero_valid]`).

All three channel inputs are per-patient constants, so `z` is constant
across the three landmark timesteps and the rollout's hidden states still
evolve through the recurrence -- exactly the prototype's behaviour, where
`GRUTransition.rollout` is handed one `z_t` per landmark drawn from
encoders that are documented as per-patient-constant under ACTG175's
baseline-treatment ITT estimand.

`TensorBatch.baseline`, `cd4_mask`, and `cd8_mask` are carried through the
rollout on `BranchOutput.shared_inputs` without being consumed by an
encoder: the masks are what the training harness's masked Huber term needs
alongside the state predictions (`models.losses.masked_huber`), and
`baseline` is the wider covariate block a downstream adjustment step reads.
Carrying them makes Requirement 3.6 checkable by object identity over every
non-intervention input the two rollouts receive, not only over the ones an
encoder happens to read (Correctness Property 14).

The 96-week outcome and its boundary convention (Requirement 3.5)
-------------------------------------------------------------------------

There is no outcome head in this module. The 96-week outcome is
`1 - prod_k (1 - h_k)` over the discrete-hazard bins up to week 96, so the
outcome is a function of the hazard sequence by construction and cannot
drift from it. `_assert_no_outcome_head` checks structurally at
construction that the module's submodules are exactly the encoders, the
GRU cell, the two hazard heads, and the state head -- adding an
outcome-producing `nn.Linear` later fails at construction rather than
silently producing a second, unconstrained answer.

**Which bins are "up to week 96".** Bins are the half-open 8-week
intervals of `decisions.md` Section 25, so bin `k` covers
`[8k, 8k + 8)`. The bins included are those whose entire span lies at or
before week 96 -- indices 0..11, covering `[0, 96)`. The count is derived,
not written down: `OUTCOME_HORIZON_BIN_COUNT =
models.dataset.survival_bin_index(HORIZON_WEEKS)` with `HORIZON_WEEKS =
96.0` imported from `causal.baselines`. `survival_bin_index(96.0)` is 12,
the index of the bin that *starts* at week 96, and that bin is
**excluded**.

Justification, including the one place this convention is visibly
imperfect: bin 12 covers `[96, 104)`, so including it would charge the
96-week risk with events occurring up to 8 weeks after the horizon --
a systematic overshoot of a full bin. Excluding it yields the risk over
`[0, 96)`, which differs from `causal.baselines._horizon_outcome`'s
`elapsed <= 96` convention only for an event at exactly 96.0 elapsed weeks
(672 days). The 23-bin grid cannot represent a partial bin, so one of the
two boundary treatments has to be chosen; a boundary-instant difference is
strictly smaller than an 8-week overshoot, and it is the conservative
direction (the neural risk never claims credit for post-horizon events).
This is a layout consequence of the frozen discretization, not a new
frozen decision, and it is stated here so a reader of a reported 96-week
risk knows exactly which bins produced it.

Determinism
-------------------------------------------------------------------------

No dropout and no batchnorm anywhere, so an `eval()`-mode forward is
deterministic by construction (the same reason `torch_sequence_gru.py`
carries none). Nothing here draws random numbers at forward time; the only
randomness is parameter initialization, which the training harness fixes
through `seed_everything` before construction.

Torch dependency scope
-------------------------------------------------------------------------

`torch` is a leaf dependency confined to `src/models/torch_*.py` and
`src/models/dataset.py` (`design.md` Section 2.4). `src/models/__init__.py`
must not import this module, and nothing outside those files may either --
`tests/unit/test_torch_isolation.py` enforces both. Tests for this module
live in `tests/neural/`, behind `pytest.importorskip("torch")`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from causal.baselines import HORIZON_WEEKS
from models.dataset import (
    FROZEN_HIDDEN_SIZE,
    LANDMARK_COUNT,
    SURVIVAL_BIN_COUNT,
    DatasetDimensions,
    TensorBatch,
    survival_bin_index,
)
from models.intervention_aware import (
    INTERVENTION_INPUT_DIM,
    PROTOCOL_INPUT_DIM,
    STATE_INPUT_DIM,
)

# ---------------------------------------------------------------------------
# Frozen widths and derived counts
# ---------------------------------------------------------------------------

#: GRU hidden width (`decisions.md` Section 25: "`hidden_size = 16` for both
#: the sequence-GRU baseline and the intervention-aware model's
#: `GRUTransition`"). An **alias** of `models.dataset.FROZEN_HIDDEN_SIZE`,
#: which holds the single canonical copy -- not an independent literal. That
#: is what makes this model and `TorchSequenceGRU` capacity-comparable by
#: construction: the ablation contrast of `decisions.md` Section 17 items 2-3
#: isolates the representational difference (treatment as a feature vs. an
#: explicit channel) only if the two share one width, and two literals could
#: drift apart with every test still passing.
HIDDEN_SIZE: int = FROZEN_HIDDEN_SIZE

#: `[cd4, cd8]` -- the width of the CD4/CD8 state head's output, matching
#: `models.sequence_baseline.STATE_DIM`'s `[next_cd4, next_cd8]` pair.
STATE_PREDICTION_WIDTH: int = 2

#: Risk horizon for the derived outcome, imported rather than restated
#: (`causal.baselines.HORIZON_WEEKS = 96.0`, the same horizon the sealed
#: reference's `risk_arm0`/`risk_arm1`/`risk_difference` quantities use).
OUTCOME_HORIZON_WEEKS: float = HORIZON_WEEKS

#: How many leading discrete-hazard bins the 96-week outcome accumulates.
#: Derived from the frozen bin definition via `survival_bin_index`, never
#: hard-coded: `survival_bin_index(96.0)` is the index of the bin *starting*
#: at week 96, and that bin is excluded, so the accumulated bins are exactly
#: those spanning `[0, 96)`. See the module docstring for the boundary
#: justification.
OUTCOME_HORIZON_BIN_COUNT: int = survival_bin_index(OUTCOME_HORIZON_WEEKS)

#: The names of every non-intervention input a rollout receives. Requirement
#: 3.6 is "hold every non-intervention input identical across the two
#: rollouts", and this tuple is what "every" means, so a test enumerates it
#: rather than re-listing the fields (Correctness Property 14).
SHARED_INPUT_FIELDS: tuple[str, ...] = (
    "baseline",
    "state",
    "protocol",
    "cd4_mask",
    "cd8_mask",
)

#: The exact set of submodules `TorchInterventionAwareModel` may own. Checked
#: at construction so a later-added outcome head is a construction failure
#: rather than a silent second answer (Requirement 3.5).
_EXPECTED_SUBMODULES: frozenset[str] = frozenset(
    {
        "state_encoder",
        "intervention_encoder",
        "protocol_encoder",
        "gru_cell",
        "hazard_head_arm0",
        "hazard_head_arm1",
        "state_head",
    }
)

if OUTCOME_HORIZON_BIN_COUNT >= SURVIVAL_BIN_COUNT:  # pragma: no cover - import guard
    raise AssertionError(
        f"OUTCOME_HORIZON_BIN_COUNT={OUTCOME_HORIZON_BIN_COUNT} must be fewer than "
        f"the frozen SURVIVAL_BIN_COUNT={SURVIVAL_BIN_COUNT}"
    )


# ---------------------------------------------------------------------------
# Channel modes (design.md Section 3.3)
# ---------------------------------------------------------------------------


class InterventionMode(Enum):
    """How the intervention channel is filled, per ablation variant.

    The channel's *width* is `INTERVENTION_INPUT_DIM` in every mode, and the
    encoder's parameter count is identical in every mode -- only the content
    changes (Requirement 3.3, Correctness Property 13).

    - `LEARNED`: the real channel, `[arm, dose_or_default,
      adherence_or_default]` with `arm` set to the arm being rolled out.
    - `ZEROED`: an all-zero channel of the same width -- the
      "no-intervention-channel" ablation. Note this removes the channel's
      information, not its capacity, so it is not by itself capacity-matched
      against a *narrower* model; it is capacity-matched against `LEARNED`.
    - `PLACEBO`: the frozen placebo/null channel of `decisions.md` Section
      14.3 / Ablation 10 -- "a fixed random vector matching the intervention
      channel's dimensionality, redrawn once per seed and frozen", supplied
      by the caller as `placebo_vector` (task 14 draws it from
      `derive_seed(master_seed, "placebo_intervention_channel")`).

    In `ZEROED` and `PLACEBO` the channel input does not depend on the arm,
    so the representation and every rolled-out hidden state are identical
    across the two arms. The hazards still differ, because the arm also
    selects which treatment-specific hazard head reads the hidden state --
    those heads are structural (`decisions.md` Section 14.3) and are not
    what these modes ablate. That is exactly the ablation's content: with a
    zeroed or placebo channel the treatment reaches the outcome only through
    the output heads, never through the state representation the GRU rolls
    forward.
    """

    LEARNED = "learned"
    ZEROED = "zeroed"
    PLACEBO = "placebo"


class ProtocolMode(Enum):
    """Whether the protocol channel carries its features or zeros.

    Same capacity-matching contract as `InterventionMode`: the width stays
    `PROTOCOL_INPUT_DIM` and the encoder stays the same size in both modes.
    """

    ACTIVE = "active"
    ZEROED = "zeroed"


# ---------------------------------------------------------------------------
# Emitted data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SharedRolloutInputs:
    """Every non-intervention tensor a rollout receives.

    One instance is built per `forward` / `rollout_both_arms` call and handed
    to both arms unchanged, so `rollout_both_arms`' two `BranchOutput`s carry
    the *same object* here -- which is how Requirement 3.6 is checked by
    identity rather than by value (Correctness Property 14). Field names are
    `SHARED_INPUT_FIELDS`.

    `eq=False` for the same reason `TensorBatch` sets it: a generated
    `__eq__` would compare tensors with `==` and then raise on the truthiness
    test.

    Attributes:
        baseline: `(N, baseline_covariate_width)`, carried through unread by
            any encoder (see module docstring).
        state: `(N, STATE_INPUT_DIM)`, the state channel's input.
        protocol: `(N, PROTOCOL_INPUT_DIM)`, the protocol channel's input
            after `ProtocolMode` is applied.
        cd4_mask: `(N, LANDMARK_COUNT)` bool, `True = observed`.
        cd8_mask: `(N, LANDMARK_COUNT)` bool, `True = observed`.
    """

    baseline: Tensor
    state: Tensor
    protocol: Tensor
    cd4_mask: Tensor
    cd8_mask: Tensor


@dataclass(frozen=True, eq=False)
class TorchBranchOutput:
    """One arm's rollout for a whole batch.

    `design.md` Section 3.3 names this type `BranchOutput`, which this module
    also binds as an alias below. It is the batched tensor analogue of
    `models.branch_heads.BranchOutput`, not the same class: that one holds
    three Python floats for one patient and a single `factual`/
    `counterfactual` tag, which cannot describe a batch in which the tag
    varies row by row. `factual_mask` carries that per-row tagging here, and
    keeps Requirement 9.5 of the parent spec ("only one potential outcome per
    patient SHALL be described as factually observed") expressible: across
    the two arms of one `rollout_both_arms` call, exactly one arm's
    `factual_mask` is set for any given row.

    Attributes:
        arm: the arm this rollout was run under, `0` or `1`.
        representation: `(N, representation_width)` -- the concatenated
            `[z_state; z_intervention; z_protocol]`, constant across
            landmarks (see module docstring).
        hidden_states: `(N, LANDMARK_COUNT, hidden_size)`, one per landmark.
        state_pred: `(N, LANDMARK_COUNT, 2)` CD4/CD8 predictions.
        hazard_logits: `(N, SURVIVAL_BIN_COUNT)` pre-sigmoid hazards, for a
            numerically stable survival NLL.
        hazards: `(N, SURVIVAL_BIN_COUNT)` discrete hazards in `(0, 1)`.
        outcome_96w: `(N,)` cumulative event probability by week 96, derived
            from `hazards` (Requirement 3.5).
        factual_mask: `(N,)` bool, `True` where the batch's observed
            `treatment_arm` equals `arm`.
        shared_inputs: the non-intervention inputs this rollout received.
        intervention_input: `(N, INTERVENTION_INPUT_DIM)` -- the only input
            that differs between the two arms of one `rollout_both_arms`
            call.
    """

    arm: Literal[0, 1]
    representation: Tensor
    hidden_states: Tensor
    state_pred: Tensor
    hazard_logits: Tensor
    hazards: Tensor
    outcome_96w: Tensor
    factual_mask: Tensor
    shared_inputs: SharedRolloutInputs
    intervention_input: Tensor


#: `design.md` Section 3.3's name for `TorchBranchOutput`. Bound as an alias
#: so either spelling imports the one class; the `Torch` prefix is kept as
#: the primary name because `models.branch_heads.BranchOutput` already exists
#: with per-patient scalar semantics.
BranchOutput = TorchBranchOutput


class RolloutPair(NamedTuple):
    """`rollout_both_arms`' return value: the arm-0 and arm-1 rollouts.

    A `NamedTuple`, so it *is* the `tuple[BranchOutput, BranchOutput]` of
    `design.md` Section 3.3 and unpacks positionally, while `.arm_0` /
    `.arm_1` remove any doubt about which element is which.
    """

    arm_0: TorchBranchOutput
    arm_1: TorchBranchOutput


# ---------------------------------------------------------------------------
# The derived outcome (Requirement 3.5)
# ---------------------------------------------------------------------------


def outcome_from_hazards(hazards: Tensor, n_bins: int = OUTCOME_HORIZON_BIN_COUNT) -> Tensor:
    """`1 - prod_{k < n_bins} (1 - h_k)`: cumulative event probability.

    The only route to a 96-week outcome in this module. Exposed as a plain
    function so the derivation is testable without a model, and so it is
    visibly a function of the hazards alone (Requirement 3.5).

    Args:
        hazards: `(N, SURVIVAL_BIN_COUNT)` discrete hazards in `[0, 1]`.
        n_bins: how many leading bins to accumulate, defaulting to the
            derived `OUTCOME_HORIZON_BIN_COUNT` (bins spanning `[0, 96)`;
            see the module docstring for the boundary convention).

    Returns:
        `(N,)` cumulative event probability by the end of bin `n_bins - 1`.

    Raises:
        ValueError: if `hazards` is not 2-D, or `n_bins` is outside
            `1..hazards.shape[-1]`.
    """
    if hazards.dim() != 2:
        raise ValueError(f"hazards must be 2-D (N, bins), got shape {tuple(hazards.shape)}")
    if not 1 <= n_bins <= hazards.shape[-1]:
        raise ValueError(
            f"n_bins must be in 1..{hazards.shape[-1]}, got {n_bins!r}"
        )
    survival = torch.prod(1.0 - hazards[:, :n_bins], dim=1)
    return 1.0 - survival


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class TorchInterventionAwareModel(nn.Module):
    """Intervention_Aware_Model (Requirement 3, `design.md` Section 3.3).

    Structure, all of it from `decisions.md` Section 14.3: three separate
    channel encoders (linear + `tanh`, mirroring the prototype's
    `_LinearEncoder`), a GRU state-transition cell rolled out over the three
    landmarks from a zero initial hidden state, two treatment-specific hazard
    heads, and one shared CD4/CD8 state head.

    Args:
        dimensions: the emitted-tensor widths from
            `models.dataset.PatientDatasetBuilder.dimensions`. Read, not
            trusted blindly: the channel widths and bin count are checked
            against the frozen constants, so a dataset-layout change surfaces
            here as a `ValueError` rather than downstream as a shape error.
        hidden_size: GRU hidden width, `decisions.md` Section 25's frozen 16.
        intervention_mode: see `InterventionMode`.
        protocol_mode: see `ProtocolMode`.
        placebo_vector: required exactly when `intervention_mode` is
            `PLACEBO`, and rejected otherwise so a supplied vector is never
            silently ignored. Length must be `INTERVENTION_INPUT_DIM`. Stored
            as a non-trainable buffer, so it travels with `.to(device)` and
            appears in `state_dict()` -- a frozen input, not a parameter.

    Raises:
        ValueError: on a `dimensions` mismatch against the frozen widths, a
            placebo vector that is missing, unexpected, or the wrong length,
            or a non-positive `hidden_size`.
    """

    def __init__(
        self,
        dimensions: DatasetDimensions,
        hidden_size: int = HIDDEN_SIZE,
        intervention_mode: InterventionMode = InterventionMode.LEARNED,
        protocol_mode: ProtocolMode = ProtocolMode.ACTIVE,
        placebo_vector: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        _validate_dimensions(dimensions)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size!r}")

        self.dimensions = dimensions
        self.hidden_size = hidden_size
        self.intervention_mode = intervention_mode
        self.protocol_mode = protocol_mode
        self.landmark_count = dimensions.landmark_count
        self.survival_bin_count = dimensions.survival_bin_count

        # Encoder output widths are fixed at construction and equal to the
        # frozen input widths, independent of either mode (Requirement 3.3).
        self.state_encoder = nn.Linear(dimensions.state_width, dimensions.state_width)
        self.intervention_encoder = nn.Linear(
            dimensions.intervention_width, dimensions.intervention_width
        )
        self.protocol_encoder = nn.Linear(dimensions.protocol_width, dimensions.protocol_width)

        self.gru_cell = nn.GRUCell(self.representation_width, hidden_size)

        # Treatment-specific hazard heads, one per branch. Indexed by arm
        # through `_hazard_head`, never by a learned gate.
        self.hazard_head_arm0 = nn.Linear(hidden_size, self.survival_bin_count)
        self.hazard_head_arm1 = nn.Linear(hidden_size, self.survival_bin_count)

        # One shared CD4/CD8 state head, applied at each landmark.
        self.state_head = nn.Linear(hidden_size, STATE_PREDICTION_WIDTH)

        self.register_buffer(
            "placebo_vector",
            _resolved_placebo_vector(placebo_vector, intervention_mode, dimensions),
        )

        self._assert_no_outcome_head()

    # -- structure -------------------------------------------------------

    @property
    def state_channel_width(self) -> int:
        """`z_state`'s width -- fixed at construction, mode-independent."""
        return self.state_encoder.out_features

    @property
    def intervention_channel_width(self) -> int:
        """`z_intervention`'s width -- identical in all three
        `InterventionMode`s, which is what capacity-matches the ablations
        (Requirement 3.3, Correctness Property 13)."""
        return self.intervention_encoder.out_features

    @property
    def protocol_channel_width(self) -> int:
        """`z_protocol`'s width -- identical in both `ProtocolMode`s."""
        return self.protocol_encoder.out_features

    @property
    def representation_width(self) -> int:
        """`len(z) = len(z_state) + len(z_intervention) + len(z_protocol)`,
        the prototype's `concatenate_channels` arithmetic and Section 25's
        `4 + 3 + 2 = 9`."""
        return (
            self.state_channel_width
            + self.intervention_channel_width
            + self.protocol_channel_width
        )

    def _assert_no_outcome_head(self) -> None:
        """Fail construction if this module owns any submodule beyond the
        frozen structure -- in particular any outcome-producing head
        (Requirement 3.5).

        Structural rather than documentary: the 96-week outcome must remain
        a function of the hazard sequence, and the way that stops being true
        is someone adding an `nn.Linear(hidden, 1)` here later. Then this
        raises, naming the requirement.
        """
        owned = set(dict(self.named_children()))
        unexpected = owned - _EXPECTED_SUBMODULES
        missing = _EXPECTED_SUBMODULES - owned
        if unexpected or missing:
            raise AssertionError(
                "TorchInterventionAwareModel's submodules must be exactly "
                f"{sorted(_EXPECTED_SUBMODULES)}; unexpected={sorted(unexpected)}, "
                f"missing={sorted(missing)}. The 96-week outcome is derived from the "
                "discrete-hazard sequence and no separate unconstrained outcome head "
                "may exist in this module (Requirement 3.5)."
            )

    def _hazard_head(self, arm: int) -> nn.Linear:
        return self.hazard_head_arm1 if arm == 1 else self.hazard_head_arm0

    # -- channel inputs --------------------------------------------------

    def shared_inputs(self, batch: TensorBatch) -> SharedRolloutInputs:
        """Build the non-intervention inputs once for a batch.

        `rollout_both_arms` calls this exactly once and hands the result to
        both arms, so every field is the same object in both rollouts
        (Requirement 3.6). `batch.baseline`, `batch.state_init`,
        `batch.cd4_mask`, and `batch.cd8_mask` pass through untouched -- the
        same objects the batch holds. `protocol` is the batch's own tensor
        under `ProtocolMode.ACTIVE`, and a zero tensor of identical shape and
        dtype under `ProtocolMode.ZEROED`.
        """
        if self.protocol_mode is ProtocolMode.ACTIVE:
            protocol = batch.protocol
        else:
            protocol = torch.zeros_like(batch.protocol)
        return SharedRolloutInputs(
            baseline=batch.baseline,
            state=batch.state_init,
            protocol=protocol,
            cd4_mask=batch.cd4_mask,
            cd8_mask=batch.cd8_mask,
        )

    def intervention_input(self, batch: TensorBatch, arm: int) -> Tensor:
        """The intervention channel's input for `arm`.

        A freshly constructed tensor on every call, in every mode, so the two
        rollouts of one `rollout_both_arms` call never share this object even
        when its *values* coincide (`ZEROED` and `PLACEBO` ignore `arm`,
        which is the ablation's whole point). Requirement 3.6 is about this
        tensor being the only one that differs; identity of the others is
        what carries it.

        Under `LEARNED`, column 0 of `batch.intervention` -- the
        `treatment_arm` column, per `models.dataset._intervention_features` --
        is replaced by `arm`, and the remaining `dose`/`adherence` columns
        are taken from the batch unchanged. Nothing is written in place:
        `batch.intervention` is never mutated.
        """
        _validate_arm(arm)
        source = batch.intervention
        if self.intervention_mode is InterventionMode.ZEROED:
            return torch.zeros_like(source)
        if self.intervention_mode is InterventionMode.PLACEBO:
            placebo = self.placebo_vector.to(dtype=source.dtype, device=source.device)
            return placebo.unsqueeze(0).expand(source.shape[0], -1).clone()
        arm_column = torch.full(
            (source.shape[0], 1), float(arm), dtype=source.dtype, device=source.device
        )
        return torch.cat([arm_column, source[:, 1:]], dim=1)

    def encode_representation(self, batch: TensorBatch, arm: int) -> Tensor:
        """`z = [z_state; z_intervention; z_protocol]` for `arm`, `(N, 9)`.

        Exposed so the channel-width property (Correctness Property 13) can
        be checked without running a full rollout.
        """
        shared = self.shared_inputs(batch)
        return self._encode(shared, self.intervention_input(batch, arm))

    def _encode(self, shared: SharedRolloutInputs, intervention_input: Tensor) -> Tensor:
        z_state = torch.tanh(self.state_encoder(shared.state))
        z_intervention = torch.tanh(self.intervention_encoder(intervention_input))
        z_protocol = torch.tanh(self.protocol_encoder(shared.protocol))
        return torch.cat([z_state, z_intervention, z_protocol], dim=1)

    # -- forward ---------------------------------------------------------

    def forward(self, batch: TensorBatch, arm: Literal[0, 1]) -> TorchBranchOutput:
        """Roll out `batch` under treatment `arm`.

        Args:
            batch: one split's tensors from `PatientDatasetBuilder.build`.
            arm: `0` or `1`.

        Returns:
            That arm's `TorchBranchOutput`, including the hazard-derived
            96-week outcome.

        Raises:
            ValueError: if `arm` is not `0` or `1`, or if `batch`'s trailing
                widths disagree with this model's.
        """
        _validate_arm(arm)
        return self._rollout(batch, self.shared_inputs(batch), arm)

    def rollout_both_arms(self, batch: TensorBatch) -> RolloutPair:
        """Roll out both arms for every patient (Requirements 3.4, 3.6).

        The non-intervention inputs are built once, before either rollout,
        and handed to both: `pair.arm_0.shared_inputs is
        pair.arm_1.shared_inputs`, and every tensor inside it is likewise one
        object. The intervention-channel input is built per arm and is the
        only tensor that differs (`pair.arm_0.intervention_input is not
        pair.arm_1.intervention_input`), which makes Requirement 3.6
        checkable by identity rather than by value (Correctness Property 14).

        Returns:
            `RolloutPair(arm_0=..., arm_1=...)`, a `tuple` subclass, so it
            unpacks as `(arm_0_output, arm_1_output)`.
        """
        shared = self.shared_inputs(batch)
        return RolloutPair(
            arm_0=self._rollout(batch, shared, 0),
            arm_1=self._rollout(batch, shared, 1),
        )

    def _rollout(
        self, batch: TensorBatch, shared: SharedRolloutInputs, arm: Literal[0, 1]
    ) -> TorchBranchOutput:
        """One arm's rollout from prebuilt shared inputs."""
        self._validate_batch_widths(batch)
        intervention_input = self.intervention_input(batch, arm)
        representation = self._encode(shared, intervention_input)

        n_rows = representation.shape[0]
        hidden = torch.zeros(
            n_rows,
            self.hidden_size,
            dtype=representation.dtype,
            device=representation.device,
        )
        hidden_states: list[Tensor] = []
        for _ in range(self.landmark_count):
            hidden = self.gru_cell(representation, hidden)
            hidden_states.append(hidden)
        stacked_hidden = torch.stack(hidden_states, dim=1)

        state_pred = self.state_head(stacked_hidden)
        hazard_logits = self._hazard_head(arm)(hidden)
        hazards = torch.sigmoid(hazard_logits)

        return TorchBranchOutput(
            arm=arm,
            representation=representation,
            hidden_states=stacked_hidden,
            state_pred=state_pred,
            hazard_logits=hazard_logits,
            hazards=hazards,
            outcome_96w=outcome_from_hazards(hazards),
            factual_mask=batch.treatment_arm == float(arm),
            shared_inputs=shared,
            intervention_input=intervention_input,
        )

    def _validate_batch_widths(self, batch: TensorBatch) -> None:
        """Reject a batch whose trailing widths disagree with this model's.

        A clearer failure than the `nn.Linear` shape error it would otherwise
        become, and it names the tensor.
        """
        expected = (
            ("state_init", batch.state_init, self.dimensions.state_width),
            ("intervention", batch.intervention, self.dimensions.intervention_width),
            ("protocol", batch.protocol, self.dimensions.protocol_width),
            ("baseline", batch.baseline, self.dimensions.baseline_covariate_width),
        )
        for name, tensor, width in expected:
            if tensor.shape[-1] != width:
                raise ValueError(
                    f"batch.{name} has trailing width {tensor.shape[-1]}, but this model "
                    f"was constructed for {width}"
                )
        for name, tensor in (("cd4_mask", batch.cd4_mask), ("cd8_mask", batch.cd8_mask)):
            if tensor.shape[-1] != self.landmark_count:
                raise ValueError(
                    f"batch.{name} has trailing width {tensor.shape[-1]}, but this model "
                    f"was constructed for {self.landmark_count} landmarks"
                )


# ---------------------------------------------------------------------------
# Construction-time validation helpers
# ---------------------------------------------------------------------------


def _validate_arm(arm: int) -> None:
    if arm not in (0, 1):
        raise ValueError(f"arm must be 0 or 1, got {arm!r}")


def _validate_dimensions(dimensions: DatasetDimensions) -> None:
    """Check `dimensions` against the frozen widths of `decisions.md`
    Section 25 and `models.dataset`.

    The widths are imported, so this cannot drift silently; what it catches
    is a *caller* handing in a `DatasetDimensions` built for a different
    layout.
    """
    frozen = (
        ("state_width", dimensions.state_width, STATE_INPUT_DIM),
        ("intervention_width", dimensions.intervention_width, INTERVENTION_INPUT_DIM),
        ("protocol_width", dimensions.protocol_width, PROTOCOL_INPUT_DIM),
        ("landmark_count", dimensions.landmark_count, LANDMARK_COUNT),
        ("survival_bin_count", dimensions.survival_bin_count, SURVIVAL_BIN_COUNT),
    )
    mismatches = [
        f"{name}={value!r} (frozen: {expected!r})"
        for name, value, expected in frozen
        if value != expected
    ]
    if mismatches:
        raise ValueError(
            "DatasetDimensions disagrees with the frozen widths of decisions.md "
            "Section 25: " + "; ".join(mismatches)
        )
    if dimensions.baseline_covariate_width <= 0:
        raise ValueError(
            f"baseline_covariate_width must be positive, got "
            f"{dimensions.baseline_covariate_width!r}"
        )


def _resolved_placebo_vector(
    placebo_vector: tuple[float, ...] | None,
    intervention_mode: InterventionMode,
    dimensions: DatasetDimensions,
) -> Tensor:
    """The buffer backing `InterventionMode.PLACEBO`.

    A zero-length tensor in the other two modes, so the buffer always exists
    (a `state_dict` key that appears and disappears by mode is worse than an
    empty one) while carrying no values that could be mistaken for a channel.

    Raises:
        ValueError: if the vector is absent under `PLACEBO`, supplied under
            another mode (which would otherwise be silently ignored), or of
            the wrong length.
    """
    if intervention_mode is not InterventionMode.PLACEBO:
        if placebo_vector is not None:
            raise ValueError(
                f"placebo_vector was supplied with intervention_mode="
                f"{intervention_mode.name}; it is used only by "
                "InterventionMode.PLACEBO and would otherwise be silently ignored"
            )
        return torch.zeros(0, dtype=torch.float32)
    if placebo_vector is None:
        raise ValueError(
            "InterventionMode.PLACEBO requires placebo_vector: decisions.md Section "
            "14.3 freezes the placebo channel as 'a fixed random vector matching the "
            "intervention channel's dimensionality, redrawn once per seed and frozen', "
            "so the vector is drawn by the caller (from derive_seed(master_seed, "
            "'placebo_intervention_channel')) and never here"
        )
    if len(placebo_vector) != dimensions.intervention_width:
        raise ValueError(
            f"placebo_vector has length {len(placebo_vector)}, but the intervention "
            f"channel is {dimensions.intervention_width} wide; the placebo channel must "
            "match the intervention channel's dimensionality exactly (decisions.md "
            "Section 14.3)"
        )
    values = [float(value) for value in placebo_vector]
    if not all(value == value and abs(value) != float("inf") for value in values):
        raise ValueError(f"placebo_vector must be finite, got {placebo_vector!r}")
    return torch.tensor(values, dtype=torch.float32)
