"""Intervention-aware architecture: separately-encoded channels + GRU
rollout (Requirements 9.2, 9.3, 9.8; `decisions.md` Section 14.3
"Intervention-aware model family and loss"; Correctness Property 7).

`decisions.md` Section 14.3 freezes this architecture:

- baseline-state encoder;
- separate structured intervention encoder;
- protocol encoder;
- GRU state-transition module;
- treatment-specific event-hazard heads (one per branch -- added in task
  9.2/`branch_1`/`branch_0`, **not** implemented here);
- CD4/CD8 state-prediction head (also task 9.2).

Representation (Section 14.3):

```text
z_t = [z_t^state; z_t^intervention; z_t^protocol]
```

-- the three encoders below each produce their own fixed-width vector, and
`concatenate_channels` joins them via plain Python list concatenation.
This module implements exactly the three encoders, the concatenation, the
placebo/null intervention channel used by the capacity-matched ablation
(Requirement 9.8, `decisions.md` Section 14.3's "Placebo/null channel",
Correctness Property 7), and the GRU rollout module (`GRUTransition`) that
consumes the concatenated sequence across the three landmark timepoints
(baseline, week 20, week 96). The event-hazard and CD4/CD8-prediction
output heads belong to task 9.2 (`branch_1`/`branch_0`); this module stops
at the rollout hidden states.

Dependency scope note (read before extending this module)
------------------------------------------------------------

As in `causal/propensity.py`, `causal/baselines.py`, and
`models/sequence_baseline.py`, `pyproject.toml` declares only `pydantic`
and `PyYAML` as runtime dependencies -- no `numpy`/`torch`. Continuing that
established convention, every encoder below is a **minimal, from-scratch
linear (+ `tanh`) projection**, pure Python (lists of floats), and
`GRUTransition` reuses `models.sequence_baseline.GRUCell` directly (rather
than reimplementing GRU gate equations a second time) for the actual
multi-step recurrent update.

Per-channel input conventions (documented choices -- see each encoder's
docstring for the rationale)
-------------------------------------------------------------------------

- `StateEncoder`: baseline covariates, the same small fixed list
  `models.sequence_baseline.BASELINE_STATE_INIT_FIELDS` already uses to
  seed `SequenceGRU`'s hidden state (`decisions.md` Section 14.3 calls
  this the "baseline-state encoder"). The encoder itself is a plain linear
  projection agnostic to what vector it is handed, so a caller could later
  feed it a per-timestep state vector instead -- but the default,
  documented usage here is the baseline-covariate vector, encoded once per
  patient and reused identically at every landmark timestep in the
  concatenation (mirroring how `SequenceGRU` treats `treatment_arm`: a
  per-patient-constant value repeated at each timestep, since ACTG175's
  estimand is baseline-treatment ITT).
- `InterventionEncoder`: a short, fixed-width per-patient vector
  `[treatment_arm, dose_or_default, adherence_or_default]`, drawing
  `dose`/`adherence` from `models.intervention.Intervention` when an
  `Intervention` object is available (`0.0` default when the field is
  `None` or no `Intervention` object is supplied -- ACTG175 does not
  require dose/adherence for its primary analysis, so this only enriches
  the channel when the data is present rather than requiring it).
  Constant across landmark timesteps for the same ITT reason as
  `StateEncoder` above.
- `ProtocolEncoder`: a small placeholder feature set,
  `[is_eligible, is_time_zero_valid]` (both `1.0`/`0.0` indicators),
  reflecting protocol-adherence-to-design concerns (eligibility,
  time-zero validity) rather than clinical state or treatment content.
  Documented as a placeholder: richer protocol features (e.g. visit
  windows, deviation flags) can be added later without changing the
  encoder's interface, only its `input_size`.

None of these choices are `decisions.md`-frozen parameters; they are this
implementation's own defaults, kept small, fixed, and documented so the
capacity-matching property (Correctness Property 7) is checkable
independent of exactly which features are chosen.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from models.sequence_baseline import GRUCell

# ---------------------------------------------------------------------------
# Small pure-Python vector/matrix helpers (mirrors sequence_baseline.py's
# module-local helpers -- kept self-contained per this project's convention
# of not sharing private helpers across modules).
# ---------------------------------------------------------------------------


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(w * v for w, v in zip(row, vector, strict=True)) for row in matrix]


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b, strict=True)]


def _random_matrix(rng: random.Random, n_rows: int, n_cols: int, scale: float) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(n_cols)] for _ in range(n_rows)]


def _random_vector(rng: random.Random, n: int, scale: float) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(n)]


# ---------------------------------------------------------------------------
# Default per-channel input widths (documented choices, see module docstring)
# ---------------------------------------------------------------------------

#: `StateEncoder` default input width -- matches
#: `models.sequence_baseline.BASELINE_STATE_INIT_FIELDS`
#: (`baseline_cd4, baseline_cd8, karnofsky_score, age`).
STATE_INPUT_DIM: int = 4

#: `InterventionEncoder` default input width -- `[treatment_arm,
#: dose_or_default, adherence_or_default]`.
INTERVENTION_INPUT_DIM: int = 3

#: `ProtocolEncoder` default input width -- `[is_eligible,
#: is_time_zero_valid]` placeholder feature set (see module docstring).
PROTOCOL_INPUT_DIM: int = 2


# ---------------------------------------------------------------------------
# Shared linear (+ tanh) projection, used by all three real encoders.
# ---------------------------------------------------------------------------


@dataclass
class _LinearEncoder:
    """A minimal linear projection + `tanh`, `output_size = W @ x + b`.

    Shared building block for `StateEncoder`, `InterventionEncoder`, and
    `ProtocolEncoder` -- each wraps one of these with its own documented
    `input_size` and semantics (see module docstring).
    """

    input_size: int
    output_size: int
    W: list[list[float]]
    b: list[float]

    @classmethod
    def initialize(
        cls, input_size: int, output_size: int, rng: random.Random, scale: float = 0.5
    ) -> _LinearEncoder:
        return cls(
            input_size=input_size,
            output_size=output_size,
            W=_random_matrix(rng, output_size, input_size, scale),
            b=_random_vector(rng, output_size, scale),
        )

    def encode(self, x: Sequence[float]) -> list[float]:
        if len(x) != self.input_size:
            raise ValueError(f"expected input of length {self.input_size}, got {len(x)}")
        return [math.tanh(v) for v in _add(_matvec(self.W, x), self.b)]


# ---------------------------------------------------------------------------
# StateEncoder
# ---------------------------------------------------------------------------


@dataclass
class StateEncoder:
    """Baseline-state encoder (`decisions.md` Section 14.3): projects a
    baseline-covariate vector (default width `STATE_INPUT_DIM`, matching
    `BASELINE_STATE_INIT_FIELDS`) into `z_t^state` of configurable width.
    """

    encoder: _LinearEncoder

    @classmethod
    def initialize(
        cls, output_size: int, seed: int = 0, input_size: int = STATE_INPUT_DIM
    ) -> StateEncoder:
        rng = random.Random(seed)
        return cls(encoder=_LinearEncoder.initialize(input_size, output_size, rng))

    @property
    def output_size(self) -> int:
        return self.encoder.output_size

    def encode(self, baseline_covariates: Sequence[float]) -> list[float]:
        """`z_t^state` for one patient. `baseline_covariates` is expected
        to be `models.sequence_baseline.extract_baseline_vector(patient)`
        (or an equivalent fixed-order vector); the same vector is reused
        for every landmark timestep (see module docstring)."""
        return self.encoder.encode(baseline_covariates)


# ---------------------------------------------------------------------------
# InterventionEncoder + PlaceboInterventionEncoder
# ---------------------------------------------------------------------------


@dataclass
class InterventionEncoder:
    """Separate structured intervention encoder (`decisions.md` Section
    14.3): projects `[treatment_arm, dose_or_default, adherence_or_default]`
    (default width `INTERVENTION_INPUT_DIM`) into `z_t^intervention` of
    configurable width -- the "intervention channel width" referenced by
    the capacity-matched placebo ablation (Requirement 9.8, Correctness
    Property 7).
    """

    encoder: _LinearEncoder

    @classmethod
    def initialize(
        cls, output_size: int, seed: int = 0, input_size: int = INTERVENTION_INPUT_DIM
    ) -> InterventionEncoder:
        rng = random.Random(seed)
        return cls(encoder=_LinearEncoder.initialize(input_size, output_size, rng))

    @property
    def output_size(self) -> int:
        return self.encoder.output_size

    def encode(self, intervention_features: Sequence[float]) -> list[float]:
        """`z_t^intervention` for one patient. `intervention_features` is
        expected to be `[treatment_arm, dose_or_default,
        adherence_or_default]` (see module docstring); the same vector is
        reused for every landmark timestep."""
        return self.encoder.encode(intervention_features)


def intervention_features_from(
    treatment_arm: float,
    intervention: object | None = None,
) -> list[float]:
    """Build the default `[treatment_arm, dose_or_default,
    adherence_or_default]` feature vector for `InterventionEncoder.encode`.

    `intervention`, if supplied, is expected to be a
    `models.intervention.Intervention` (typed loosely here to avoid a hard
    import dependency); its `dose`/`adherence` fields are used when not
    `None`, else `0.0`. `intervention=None` (no `Intervention` object
    available for this patient/timestep) also defaults both to `0.0`.
    """
    dose = 0.0
    adherence = 0.0
    if intervention is not None:
        dose = getattr(intervention, "dose", None) or 0.0
        adherence = getattr(intervention, "adherence", None) or 0.0
    return [float(treatment_arm), float(dose), float(adherence)]


@dataclass
class PlaceboInterventionEncoder:
    """Capacity-matched placebo/null intervention channel (Requirement
    9.8; `decisions.md` Section 14.3 "Placebo/null channel"; Correctness
    Property 7): "a fixed random vector matching the intervention
    channel's dimensionality, redrawn once per seed and frozen."

    Unlike `InterventionEncoder`, this is **not** a trainable projection:
    on construction (`initialize(output_size, seed)`), it draws exactly
    one random vector of length `output_size` and caches it. Every
    subsequent `encode(...)` call -- for any patient, any timestep, any
    input -- returns that same cached vector (a fresh copy, so callers
    cannot mutate the cached original), never redrawing. Substituting this
    for the real `InterventionEncoder` in an ablation therefore disables
    the intervention channel's *information content* while leaving
    `z_t`'s total width unchanged -- the capacity-matching property Property
    7 requires.
    """

    output_size: int
    _cached_vector: list[float]

    @classmethod
    def initialize(cls, output_size: int, seed: int = 0) -> PlaceboInterventionEncoder:
        rng = random.Random(seed)
        return cls(output_size=output_size, _cached_vector=_random_vector(rng, output_size, 0.5))

    def encode(self, intervention_features: Sequence[float] | None = None) -> list[float]:
        """Return the cached placebo vector, ignoring `intervention_features`
        entirely (accepted only so this can be swapped in for
        `InterventionEncoder.encode` without changing call sites)."""
        return list(self._cached_vector)


# ---------------------------------------------------------------------------
# ProtocolEncoder
# ---------------------------------------------------------------------------


@dataclass
class ProtocolEncoder:
    """Protocol encoder (`decisions.md` Section 14.3): projects a small
    placeholder protocol-feature vector -- default `[is_eligible,
    is_time_zero_valid]`, width `PROTOCOL_INPUT_DIM` -- into
    `z_t^protocol` of configurable width. See module docstring for why
    this feature set is a documented placeholder.
    """

    encoder: _LinearEncoder

    @classmethod
    def initialize(
        cls, output_size: int, seed: int = 0, input_size: int = PROTOCOL_INPUT_DIM
    ) -> ProtocolEncoder:
        rng = random.Random(seed)
        return cls(encoder=_LinearEncoder.initialize(input_size, output_size, rng))

    @property
    def output_size(self) -> int:
        return self.encoder.output_size

    def encode(self, protocol_features: Sequence[float]) -> list[float]:
        """`z_t^protocol` for one patient. `protocol_features` is expected
        to be `[is_eligible, is_time_zero_valid]` (see module docstring);
        the same vector is reused for every landmark timestep unless a
        caller has richer, time-varying protocol features."""
        return self.encoder.encode(protocol_features)


# ---------------------------------------------------------------------------
# Channel concatenation: z_t = [z_t^state; z_t^intervention; z_t^protocol]
# ---------------------------------------------------------------------------


def concatenate_channels(
    z_state: Sequence[float], z_intervention: Sequence[float], z_protocol: Sequence[float]
) -> list[float]:
    """`z_t = [z_t^state; z_t^intervention; z_t^protocol]` (`decisions.md`
    Section 14.3) -- plain Python list concatenation of the three encoder
    outputs. `len(...)` of the result is `len(z_state) + len(z_intervention)
    + len(z_protocol)` regardless of which concrete intervention encoder
    (`InterventionEncoder` or `PlaceboInterventionEncoder`) produced
    `z_intervention`, as long as both have the same `output_size` -- the
    capacity-matching property (Correctness Property 7).
    """
    return [*z_state, *z_intervention, *z_protocol]


# ---------------------------------------------------------------------------
# GRUTransition: multi-step rollout across baseline / week-20 / week-96
# ---------------------------------------------------------------------------


@dataclass
class GRUTransition:
    """GRU state-transition module (`decisions.md` Section 14.3):
    consumes the concatenated `z_t` sequence across the three landmark
    timepoints (baseline, week 20, week 96) and produces a rollout of
    hidden states -- the multi-step rollout required for the "minimal
    structured patient world model" designation (Requirement 9.11,
    `decisions.md` Section 14.4).

    Reuses `models.sequence_baseline.GRUCell` directly for the recurrent
    update (see module docstring) rather than reimplementing GRU gate
    equations. The event-hazard and CD4/CD8-prediction output heads that
    would normally sit on top of this rollout's hidden states belong to
    task 9.2 (`branch_1`/`branch_0`, see `models.branch_heads.BranchHead`
    and `models.branch_heads.evaluate_both_branches`) and are deliberately
    not implemented here -- `rollout()` returns hidden states only.
    """

    input_size: int
    hidden_size: int
    gru_cell: GRUCell

    @classmethod
    def initialize(cls, input_size: int, hidden_size: int, seed: int = 0) -> GRUTransition:
        rng = random.Random(seed)
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            gru_cell=GRUCell.initialize(input_size, hidden_size, rng),
        )

    def rollout(
        self,
        z_sequence: Sequence[Sequence[float]],
        initial_hidden_state: Sequence[float] | None = None,
    ) -> list[list[float]]:
        """Multi-step rollout: one `GRUCell.step` per `z_t` in
        `z_sequence` (expected length `N_LANDMARKS = 3` -- baseline, week
        20, week 96, though this method itself accepts any length).

        `initial_hidden_state` defaults to a zero vector of width
        `hidden_size` when not supplied (no baseline-covariate-init
        projection here -- that role is filled by `StateEncoder`'s
        contribution to `z_t` itself, unlike `SequenceGRU` where baseline
        covariates instead seed `h_0` directly).

        Returns one `hidden_size`-length hidden state per timestep
        (length `len(z_sequence)`).
        """
        if initial_hidden_state is not None:
            h = list(initial_hidden_state)
        else:
            h = [0.0] * self.hidden_size
        hidden_states: list[list[float]] = []
        for z_t in z_sequence:
            h = self.gru_cell.step(z_t, h)
            hidden_states.append(h)
        return hidden_states
