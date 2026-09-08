"""Factual/counterfactual branch heads (Requirements 9.4, 9.5;
`decisions.md` Section 14.3: "treatment-specific event-hazard heads (one
per branch); CD4/CD8 state-prediction head ... For every eligible
patient, both branches are evaluated: Ŷ_i(1), Ŷ_i(0).").

This module sits directly on top of `models.intervention_aware.GRUTransition`
(task 9.1): `GRUTransition.rollout(z_sequence)` produces a `hidden_size`-
length hidden state per landmark timestep; `BranchHead` (`branch_1`,
`branch_0`) maps ONE such hidden state to `(event_hazard, next_cd4,
next_cd8)`. `evaluate_both_branches` runs both heads on the SAME hidden
state and tags each output `"factual"`/`"counterfactual"` according to
which branch matches the patient's observed treatment
(`cohorts.cohort_construction.PatientRecord.treatment_arm`, `0` or `1`) --
Requirement 9.5: "Only one potential outcome per patient SHALL be
described as factually observed."

This module deliberately does not implement the event-hazard/CD4/CD8
*loss* (task 9.3) or treatment-effect aggregation over branch outputs
(task 9.4 in the plan's task-9.x numbering, i.e. plan task 9.4
"treatment-effect aggregation" -- distinct from Requirement 9.4 quoted
above); it only produces the two per-patient potential-outcome
predictions and their factual/counterfactual tags.

Following the same from-scratch, pure-Python convention as
`models.intervention_aware` (no `numpy`/`torch` runtime dependency, see
that module's docstring), `BranchHead` reuses the identical minimal
linear-projection pattern (`W @ h + b`): `sigmoid` for the bounded
`event_hazard` output, identity/linear for the `next_cd4`/`next_cd8`
regression outputs. Rather than re-import `intervention_aware`'s private
`_matvec`/`_add`/`_random_matrix`/`_random_vector` helpers (this project's
established convention is to keep such helpers module-local -- see
`intervention_aware.py`'s own module docstring), this module defines its
own small equivalents below.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Small pure-Python vector/matrix helpers (mirrors intervention_aware.py's
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


def _sigmoid(x: float) -> float:
    # Numerically-stable logistic sigmoid.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


#: Number of raw outputs a `BranchHead` projects to before splitting into
#: `(event_hazard, next_cd4, next_cd8)`.
_BRANCH_OUTPUT_WIDTH = 3


@dataclass
class BranchHead:
    """One treatment-specific branch output head (`branch_1` or
    `branch_0`, `decisions.md` Section 14.3's "treatment-specific
    event-hazard heads (one per branch); CD4/CD8 state-prediction head").

    A single linear projection `W @ h + b` (`h` = a hidden state from
    `models.intervention_aware.GRUTransition.rollout()`) produces three
    raw outputs, which are then mapped to `(event_hazard, next_cd4,
    next_cd8)`: `event_hazard = sigmoid(raw[0])` (bounded to `[0, 1]`,
    since it is a hazard); `next_cd4 = raw[1]`, `next_cd8 = raw[2]`
    (identity/linear, since these are unbounded regression targets).
    """

    hidden_size: int
    W: list[list[float]]
    b: list[float]

    @classmethod
    def initialize(cls, hidden_size: int, seed: int = 0, scale: float = 0.5) -> BranchHead:
        rng = random.Random(seed)
        return cls(
            hidden_size=hidden_size,
            W=_random_matrix(rng, _BRANCH_OUTPUT_WIDTH, hidden_size, scale),
            b=_random_vector(rng, _BRANCH_OUTPUT_WIDTH, scale),
        )

    def predict(self, hidden_state: Sequence[float]) -> tuple[float, float, float]:
        """`(event_hazard, next_cd4, next_cd8)` for one hidden state.

        Raises `ValueError` if `hidden_state`'s length does not match
        `self.hidden_size` (mirrors `intervention_aware._LinearEncoder`'s
        input-length check).
        """
        if len(hidden_state) != self.hidden_size:
            raise ValueError(
                f"expected hidden_state of length {self.hidden_size}, got {len(hidden_state)}"
            )
        raw = _add(_matvec(self.W, hidden_state), self.b)
        event_hazard = _sigmoid(raw[0])
        next_cd4 = raw[1]
        next_cd8 = raw[2]
        return event_hazard, next_cd4, next_cd8


# ---------------------------------------------------------------------------
# BranchOutput + evaluate_both_branches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchOutput:
    """One branch's potential-outcome prediction for one patient, tagged
    according to whether it matches the patient's observed treatment.

    Requirement 9.4: "The model SHALL produce both potential-outcome
    predictions, Ŷ_i(1) and Ŷ_i(0), while identifying which corresponds
    to the observed treatment." Requirement 9.5: "Only one potential
    outcome per patient SHALL be described as factually observed" --
    enforced by `evaluate_both_branches` below (exactly one of the two
    `BranchOutput`s it returns is tagged `"factual"`).
    """

    event_hazard: float
    next_cd4: float
    next_cd8: float
    tag: Literal["factual", "counterfactual"]


def evaluate_both_branches(
    hidden_state: Sequence[float],
    branch_1_head: BranchHead,
    branch_0_head: BranchHead,
    observed_treatment_arm: int,
) -> tuple[BranchOutput, BranchOutput]:
    """Run BOTH `branch_1_head` and `branch_0_head` on the SAME
    `hidden_state` -- `decisions.md` Section 14.3: "For every eligible
    patient, both branches are evaluated: Ŷ_i(1), Ŷ_i(0)." -- and tag each
    result `"factual"`/`"counterfactual"` against `observed_treatment_arm`
    (the patient's `cohorts.cohort_construction.PatientRecord
    .treatment_arm`, `0` or `1`).

    Returns `(branch_1_output, branch_0_output)`. `branch_1_output.tag ==
    "factual"` iff `observed_treatment_arm == 1`; `branch_0_output.tag ==
    "factual"` iff `observed_treatment_arm == 0`. Since
    `observed_treatment_arm` is always exactly `0` or `1` (raises
    `ValueError` otherwise), exactly one of the two returned outputs is
    tagged `"factual"` and the other `"counterfactual"` -- Requirement
    9.5.
    """
    if observed_treatment_arm not in (0, 1):
        raise ValueError(
            f"observed_treatment_arm must be 0 or 1, got {observed_treatment_arm!r}"
        )

    branch_1_hazard, branch_1_cd4, branch_1_cd8 = branch_1_head.predict(hidden_state)
    branch_0_hazard, branch_0_cd4, branch_0_cd8 = branch_0_head.predict(hidden_state)

    branch_1_tag: Literal["factual", "counterfactual"] = (
        "factual" if observed_treatment_arm == 1 else "counterfactual"
    )
    branch_0_tag: Literal["factual", "counterfactual"] = (
        "factual" if observed_treatment_arm == 0 else "counterfactual"
    )

    branch_1_output = BranchOutput(
        event_hazard=branch_1_hazard, next_cd4=branch_1_cd4, next_cd8=branch_1_cd8, tag=branch_1_tag
    )
    branch_0_output = BranchOutput(
        event_hazard=branch_0_hazard, next_cd4=branch_0_cd4, next_cd8=branch_0_cd8, tag=branch_0_tag
    )
    return branch_1_output, branch_0_output
