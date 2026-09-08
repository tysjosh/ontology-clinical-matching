"""Sequence-model baseline: capacity-matched GRU patient-state model
(Requirement 8.2; `decisions.md` Section 14.2 "Sequence-model baseline").

`decisions.md` Section 14.2 freezes this architecture:

- baseline covariates initialize the GRU's hidden state;
- CD4/CD8 observations form the longitudinal input sequence;
- treatment is supplied as an ordinary input feature, undifferentiated from
  other covariates (**not** a separate channel -- that is the
  intervention-aware model of task 9.1/`src/models/intervention_aware.py`,
  which encodes state/intervention/protocol through *separately
  identifiable* channels. This model is deliberately the "treatment as
  ordinary feature" comparison arm the ablation matrix contrasts against
  it);
- two output heads: next patient-state (CD4/CD8) regression, and
  discrete-time event-hazard.

"A Transformer architecture is rejected as unnecessary given only three
principal clinical timepoints (baseline, week 20, week 96)" -- so the
sequence length this model consumes is fixed at ``N_LANDMARKS = 3``
(`decisions.md` Section 5's landmark measurements).

Dependency scope note (read before extending this module)
------------------------------------------------------------

As in `causal/propensity.py` (primary propensity model) and
`causal/baselines.py` (Cox baselines), `pyproject.toml` declares only
`pydantic` and `PyYAML` as runtime dependencies -- no
`numpy`/`torch`/`scikit-learn`. Adding any of those (in particular a full
autodiff/tensor library) would be a real, heavyweight dependency decision
this task does not make unilaterally, and is disproportionate for a
~2,000-patient dataset with a 3-timestep sequence and a hidden state on the
order of 4-16 units. Continuing the established project convention, this
module implements:

- ``GRUCell``: a **minimal, from-scratch single-layer GRU cell**, pure
  Python (lists of floats), computing the standard update/reset-gate and
  candidate-hidden-state equations explicitly via manual matrix-vector
  products -- no generic autodiff engine.
- ``SequenceGRU``: wraps ``GRUCell`` with a baseline-state-init linear
  projection and the two frozen output heads (next-state regression,
  discrete-time hazard).
- Training (``SequenceGRU.train_step``): rather than a full, hand-derived
  backprop-through-time (BPTT) implementation -- which is easy to get
  subtly wrong for a GRU's coupled gates without a test oracle to check
  against -- this uses a **central-difference numerical gradient** over
  the model's flat parameter vector. With ~100-200 parameters at the
  hidden sizes this module targets (4-16) and a handful of training
  samples, a full central-difference gradient computation is cheap (a few
  hundred forward passes per step, each forward pass itself only three GRU
  steps), and is *exact* in the numerical sense (matches the true gradient
  up to O(epsilon^2) truncation error) rather than an ad-hoc
  approximation. This is documented here as the deliberate simplification
  in place of BPTT: correct enough to demonstrably reduce training loss
  for a smoke test, but not the training method this project would use at
  a scale where numerical differentiation's O(n_params) forward-pass cost
  became a bottleneck. Revisit (analytic BPTT, or a real autodiff
  dependency) if this model is ever trained on the full development
  cohort rather than smoke-tested.

Sequence input and baseline-state-init conventions
----------------------------------------------------

- Per-timestep input vector (length ``INPUT_SIZE = 3``): ``[cd4, cd8,
  treatment_arm]`` -- CD4/CD8 at that landmark timepoint, followed by
  ``treatment_arm`` concatenated in as an ordinary scalar feature (frozen
  choice above). ``treatment_arm`` is repeated identically at every
  timestep, since ACTG175's estimand is baseline-treatment ITT (Section
  7.3/`msm_or_narrowing.py`) -- treatment does not vary over time here.
- Baseline-state-init vector: a small, fixed covariate list read directly
  from ``PatientRecord.timeline.static_covariates`` --
  ``BASELINE_STATE_INIT_FIELDS`` below. This module defines its own list
  (rather than reusing `causal.propensity.COVARIATE_FIELDS`) because the
  baseline-state-init projection here does not need cross-fold-safe
  standardization the way a propensity model's regression does -- it only
  needs *some* reasonable fixed-width baseline summary to seed the hidden
  state. Kept small and documented for exactly this reason.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cohorts.cohort_construction import PatientRecord

# ---------------------------------------------------------------------------
# Frozen architecture constants (decisions.md Section 14.2, Section 5)
# ---------------------------------------------------------------------------

#: CD4 and CD8 -- the two longitudinal state variables this model predicts.
STATE_DIM: int = 2

#: Treatment supplied as one ordinary scalar feature (undifferentiated from
#: CD4/CD8), per decisions.md Section 14.2 -- not a separate channel.
TREATMENT_FEATURE_DIM: int = 1

#: Per-timestep input width: [cd4, cd8, treatment_arm].
INPUT_SIZE: int = STATE_DIM + TREATMENT_FEATURE_DIM

#: The three frozen clinical landmark timepoints (decisions.md Section 5):
#: baseline, week 20, week 96. A Transformer is explicitly rejected as
#: unnecessary given this fixed, short sequence length.
LANDMARK_WEEKS: tuple[int, ...] = (0, 20, 96)
N_LANDMARKS: int = len(LANDMARK_WEEKS)

#: `concept_id`s under which longitudinal CD4/CD8 are recorded in
#: `PatientTimeline.timestamped_observations` (matches the normalized
#: concept names used throughout `src/data`, e.g. `causal/propensity.py`
#: and `causal/negative_controls.py`'s `BASELINE_CD8_FIELD`).
CD4_LAB_CONCEPT_ID: str = "cd4_count"
CD8_LAB_CONCEPT_ID: str = "cd8_count"

#: Small, fixed baseline-covariate list used only to initialize the GRU's
#: hidden state (see module docstring for why this is its own list rather
#: than reusing `causal.propensity.COVARIATE_FIELDS`). Read directly from
#: `PatientTimeline.static_covariates`; missing/non-numeric values default
#: to `0.0`.
BASELINE_STATE_INIT_FIELDS: tuple[str, ...] = (
    "baseline_cd4",
    "baseline_cd8",
    "karnofsky_score",
    "age",
)


# ---------------------------------------------------------------------------
# Small pure-Python vector/matrix helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(w * v for w, v in zip(row, vector, strict=True)) for row in matrix]


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b, strict=True)]


def _elementwise_mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [ai * bi for ai, bi in zip(a, b, strict=True)]


def _random_matrix(rng: random.Random, n_rows: int, n_cols: int, scale: float) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(n_cols)] for _ in range(n_rows)]


def _random_vector(rng: random.Random, n: int, scale: float) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(n)]


# ---------------------------------------------------------------------------
# GRUCell: minimal, from-scratch single-layer GRU cell
# ---------------------------------------------------------------------------


@dataclass
class GRUCell:
    """A single-layer GRU cell, pure Python, operating on small fixed-size
    vectors (`input_size` on the order of 3, `hidden_size` on the order of
    4-16 -- see module docstring for why this stays small).

    Standard GRU equations (update gate `z`, reset gate `r`, candidate
    hidden state `h_tilde`, blended output `h`), applied explicitly via
    manual matrix-vector products -- no autodiff engine:

    ```text
    combined       = concat(x_t, h_prev)
    z_t            = sigmoid(Wz @ combined + bz)
    r_t            = sigmoid(Wr @ combined + br)
    combined_reset = concat(x_t, r_t * h_prev)
    h_tilde_t      = tanh(Wh @ combined_reset + bh)
    h_t            = (1 - z_t) * h_prev + z_t * h_tilde_t
    ```

    Each of `Wz`, `Wr`, `Wh` has shape `(hidden_size, input_size +
    hidden_size)`; each of `bz`, `br`, `bh` has shape `(hidden_size,)`.
    """

    input_size: int
    hidden_size: int
    Wz: list[list[float]]
    bz: list[float]
    Wr: list[list[float]]
    br: list[float]
    Wh: list[list[float]]
    bh: list[float]

    @classmethod
    def initialize(cls, input_size: int, hidden_size: int, rng: random.Random) -> GRUCell:
        combined_size = input_size + hidden_size
        # Small-uniform initialization -- deterministic given `rng`'s seed,
        # keeps the initial forward pass numerically well-behaved for the
        # small dimensions this module targets.
        scale = 0.5
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            Wz=_random_matrix(rng, hidden_size, combined_size, scale),
            bz=_random_vector(rng, hidden_size, scale),
            Wr=_random_matrix(rng, hidden_size, combined_size, scale),
            br=_random_vector(rng, hidden_size, scale),
            Wh=_random_matrix(rng, hidden_size, combined_size, scale),
            bh=_random_vector(rng, hidden_size, scale),
        )

    def step(self, x_t: Sequence[float], h_prev: Sequence[float]) -> list[float]:
        """Advance the hidden state by one timestep. Returns the new
        hidden state (`hidden_size` floats)."""
        if len(x_t) != self.input_size:
            raise ValueError(f"expected input of length {self.input_size}, got {len(x_t)}")
        if len(h_prev) != self.hidden_size:
            raise ValueError(
                f"expected hidden state of length {self.hidden_size}, got {len(h_prev)}"
            )

        combined = [*x_t, *h_prev]
        z = [_sigmoid(v) for v in _add(_matvec(self.Wz, combined), self.bz)]
        r = [_sigmoid(v) for v in _add(_matvec(self.Wr, combined), self.br)]

        combined_reset = [*x_t, *_elementwise_mul(r, h_prev)]
        h_tilde = [math.tanh(v) for v in _add(_matvec(self.Wh, combined_reset), self.bh)]

        return [
            (1.0 - z_i) * h_prev_i + z_i * h_tilde_i
            for z_i, h_prev_i, h_tilde_i in zip(z, h_prev, h_tilde, strict=True)
        ]

    def flat_parameters(self) -> list[float]:
        flat: list[float] = []
        for matrix in (self.Wz, self.Wr, self.Wh):
            for row in matrix:
                flat.extend(row)
        flat.extend(self.bz)
        flat.extend(self.br)
        flat.extend(self.bh)
        return flat

    def load_flat_parameters(self, flat: Sequence[float]) -> None:
        offset = 0
        combined_size = self.input_size + self.hidden_size
        for matrix in (self.Wz, self.Wr, self.Wh):
            for row in matrix:
                row[:] = flat[offset : offset + combined_size]
                offset += combined_size
        for bias in (self.bz, self.br, self.bh):
            bias[:] = flat[offset : offset + self.hidden_size]
            offset += self.hidden_size

    def n_parameters(self) -> int:
        combined_size = self.input_size + self.hidden_size
        return 3 * self.hidden_size * combined_size + 3 * self.hidden_size


# ---------------------------------------------------------------------------
# SequenceGRU: baseline-state init + GRUCell + two output heads
# ---------------------------------------------------------------------------


@dataclass
class SequenceGRU:
    """Capacity-matched GRU patient-state model (`decisions.md` Section
    14.2): the "treatment as ordinary feature" baseline arm.

    Attributes (all pure-Python lists of floats -- see module docstring
    for the dependency-scope rationale):
        baseline_dim: width of the baseline-covariate vector used to
            initialize the hidden state (see `BASELINE_STATE_INIT_FIELDS`).
        hidden_size: GRU hidden-state width (kept small, 4-16, per module
            docstring).
        gru_cell: the `GRUCell` consuming `[cd4, cd8, treatment_arm]` at
            each of the three landmark timesteps.
        init_W, init_b: the linear projection from baseline covariates to
            the initial hidden state `h_0` (`init_W` shape
            `(hidden_size, baseline_dim)`, `init_b` shape `(hidden_size,)`),
            passed through `tanh` to keep `h_0` in the same bounded range
            GRU hidden states occupy after each step.
        next_state_W, next_state_b: the next-state regression head, linear,
            no output activation (`next_state_W` shape
            `(STATE_DIM, hidden_size)`, `next_state_b` shape `(STATE_DIM,)`)
            -- predicts `[next_cd4, next_cd8]` from that timestep's hidden
            state.
        hazard_W, hazard_b: the discrete-time hazard head, linear + sigmoid
            (`hazard_W` shape `(1, hidden_size)`, `hazard_b` shape `(1,)`)
            -- predicts one hazard probability per timestep.
    """

    baseline_dim: int
    hidden_size: int
    gru_cell: GRUCell
    init_W: list[list[float]]
    init_b: list[float]
    next_state_W: list[list[float]]
    next_state_b: list[float]
    hazard_W: list[list[float]]
    hazard_b: list[float]

    @classmethod
    def initialize(cls, baseline_dim: int, hidden_size: int, seed: int = 0) -> SequenceGRU:
        rng = random.Random(seed)
        scale = 0.5
        return cls(
            baseline_dim=baseline_dim,
            hidden_size=hidden_size,
            gru_cell=GRUCell.initialize(INPUT_SIZE, hidden_size, rng),
            init_W=_random_matrix(rng, hidden_size, baseline_dim, scale),
            init_b=_random_vector(rng, hidden_size, scale),
            next_state_W=_random_matrix(rng, STATE_DIM, hidden_size, scale),
            next_state_b=_random_vector(rng, STATE_DIM, scale),
            hazard_W=_random_matrix(rng, 1, hidden_size, scale),
            hazard_b=_random_vector(rng, 1, scale),
        )

    # -- forward -----------------------------------------------------------

    def forward(
        self, baseline_covariates: Sequence[float], sequence_input: Sequence[Sequence[float]]
    ) -> tuple[list[list[float]], list[list[float]], list[float]]:
        """Run the model forward over `sequence_input` (a sequence of
        per-timestep `[cd4, cd8, treatment_arm]` feature vectors, expected
        length `N_LANDMARKS = 3`, each of length `INPUT_SIZE = 3`).

        Returns:
            hidden_states: one `hidden_size`-length list per timestep
                (length `len(sequence_input)`).
            next_state_predictions: one `STATE_DIM`-length (`[cd4, cd8]`)
                list per timestep (length `len(sequence_input)`).
            hazard_predictions: one hazard probability (float in `[0, 1]`)
                per timestep (length `len(sequence_input)`).
        """
        if len(baseline_covariates) != self.baseline_dim:
            raise ValueError(
                f"expected baseline_covariates of length {self.baseline_dim}, "
                f"got {len(baseline_covariates)}"
            )

        h = [math.tanh(v) for v in _add(_matvec(self.init_W, baseline_covariates), self.init_b)]

        hidden_states: list[list[float]] = []
        next_state_predictions: list[list[float]] = []
        hazard_predictions: list[float] = []

        for x_t in sequence_input:
            h = self.gru_cell.step(x_t, h)
            hidden_states.append(h)
            next_state_predictions.append(_add(_matvec(self.next_state_W, h), self.next_state_b))
            hazard_logit = _add(_matvec(self.hazard_W, h), self.hazard_b)[0]
            hazard_predictions.append(_sigmoid(hazard_logit))

        return hidden_states, next_state_predictions, hazard_predictions

    # -- flat parameter (de)serialization, for numerical-gradient training -

    def flat_parameters(self) -> list[float]:
        flat: list[float] = []
        for row in self.init_W:
            flat.extend(row)
        flat.extend(self.init_b)
        flat.extend(self.gru_cell.flat_parameters())
        for row in self.next_state_W:
            flat.extend(row)
        flat.extend(self.next_state_b)
        for row in self.hazard_W:
            flat.extend(row)
        flat.extend(self.hazard_b)
        return flat

    def load_flat_parameters(self, flat: Sequence[float]) -> None:
        offset = 0
        for row in self.init_W:
            row[:] = flat[offset : offset + self.baseline_dim]
            offset += self.baseline_dim
        self.init_b[:] = flat[offset : offset + self.hidden_size]
        offset += self.hidden_size

        n_gru = self.gru_cell.n_parameters()
        self.gru_cell.load_flat_parameters(flat[offset : offset + n_gru])
        offset += n_gru

        for row in self.next_state_W:
            row[:] = flat[offset : offset + self.hidden_size]
            offset += self.hidden_size
        self.next_state_b[:] = flat[offset : offset + STATE_DIM]
        offset += STATE_DIM

        for row in self.hazard_W:
            row[:] = flat[offset : offset + self.hidden_size]
            offset += self.hidden_size
        self.hazard_b[:] = flat[offset : offset + 1]
        offset += 1

    # -- loss ----------------------------------------------------------

    def loss(self, batch: Sequence[SequenceTrainingSample]) -> float:
        """Mean per-sample loss over `batch`:
        `mean_squared_error(next_state_predictions, target_next_states) +
        binary_cross_entropy(hazard_predictions, target_hazards)`, masked
        by each sample's `mask` (so timesteps with no observed target, e.g.
        the final landmark's "next" state, do not contribute).
        """
        if not batch:
            raise ValueError("Cannot compute loss on an empty batch")

        total_loss = 0.0
        for sample in batch:
            _, next_state_predictions, hazard_predictions = self.forward(
                sample.baseline_covariates, sample.sequence_input
            )
            total_loss += _sample_loss(
                next_state_predictions, hazard_predictions, sample.target_next_states,
                sample.target_hazards, sample.mask,
            )
        return total_loss / len(batch)

    # -- training: central-difference numerical gradient descent -----------

    def train_step(
        self,
        batch: Sequence[SequenceTrainingSample],
        learning_rate: float = 0.05,
        epsilon: float = 1e-4,
    ) -> float:
        """One parameter-update step over `batch` via central-difference
        numerical gradient descent (see module docstring for why this
        replaces analytic BPTT here).

        Returns the loss computed *before* this step's update (so callers
        can track a per-iteration loss curve, e.g. for a training-loop
        smoke test asserting the loss decreases across iterations).
        """
        params = self.flat_parameters()
        loss_before = self.loss(batch)

        gradient = [0.0] * len(params)
        for i in range(len(params)):
            original = params[i]

            params[i] = original + epsilon
            self.load_flat_parameters(params)
            loss_plus = self.loss(batch)

            params[i] = original - epsilon
            self.load_flat_parameters(params)
            loss_minus = self.loss(batch)

            gradient[i] = (loss_plus - loss_minus) / (2.0 * epsilon)
            params[i] = original

        updated = [p - learning_rate * g for p, g in zip(params, gradient, strict=True)]
        self.load_flat_parameters(updated)

        return loss_before


# ---------------------------------------------------------------------------
# Training sample container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceTrainingSample:
    """One patient's training example for `SequenceGRU.train_step`.

    Attributes:
        baseline_covariates: length `baseline_dim` (see
            `BASELINE_STATE_INIT_FIELDS`).
        sequence_input: `N_LANDMARKS` per-timestep `[cd4, cd8,
            treatment_arm]` vectors (length `INPUT_SIZE` each).
        target_next_states: `N_LANDMARKS` `[cd4, cd8]` targets -- typically
            the *following* timestep's observed state (shifted by one),
            with the final timestep's target left as a placeholder and
            excluded via `mask`.
        target_hazards: `N_LANDMARKS` binary event-indicator targets (`0.0`
            or `1.0`) for the discrete-time hazard head.
        mask: `N_LANDMARKS` booleans; `False` excludes that timestep's
            next-state target from the loss (e.g. the last landmark, which
            has no "next" observation).
    """

    baseline_covariates: tuple[float, ...]
    sequence_input: tuple[tuple[float, ...], ...]
    target_next_states: tuple[tuple[float, float], ...]
    target_hazards: tuple[float, ...]
    mask: tuple[bool, ...]


def _sample_loss(
    next_state_predictions: Sequence[Sequence[float]],
    hazard_predictions: Sequence[float],
    target_next_states: Sequence[Sequence[float]],
    target_hazards: Sequence[float],
    mask: Sequence[bool],
) -> float:
    """`mean_squared_error` (next-state head, masked) + `binary
    cross-entropy` (hazard head, every timestep) for one sample."""
    squared_errors: list[float] = []
    for prediction, target, keep in zip(
        next_state_predictions, target_next_states, mask, strict=True
    ):
        if not keep:
            continue
        squared_errors.extend((p - t) ** 2 for p, t in zip(prediction, target, strict=True))
    state_loss = sum(squared_errors) / len(squared_errors) if squared_errors else 0.0

    epsilon = 1e-9
    cross_entropy_terms = [
        -(
            target * math.log(prediction + epsilon)
            + (1.0 - target) * math.log(1.0 - prediction + epsilon)
        )
        for prediction, target in zip(hazard_predictions, target_hazards, strict=True)
    ]
    hazard_loss = (
        sum(cross_entropy_terms) / len(cross_entropy_terms) if cross_entropy_terms else 0.0
    )

    return state_loss + hazard_loss


# ---------------------------------------------------------------------------
# Optional convenience adapters: PatientRecord -> model inputs
# ---------------------------------------------------------------------------


def extract_baseline_vector(patient: PatientRecord) -> list[float]:
    """The fixed-order `BASELINE_STATE_INIT_FIELDS` baseline vector for one
    patient, read from `patient.timeline.static_covariates`. Missing or
    non-numeric fields default to `0.0`."""
    static_covariates = patient.timeline.static_covariates
    values: list[float] = []
    for field_name in BASELINE_STATE_INIT_FIELDS:
        raw_value = static_covariates.get(field_name)
        try:
            values.append(float(raw_value) if raw_value is not None else 0.0)
        except (TypeError, ValueError):
            values.append(0.0)
    return values
