"""Treatment-effect aggregation (Requirements 9.6, 9.7; Correctness
Property 5; `decisions.md` Section 14.3's `aggregation.py` entry in
`design.md` Section 3.6).

`PatientLevelEstimate` is defined here exactly as frozen in `design.md`'s
Data Models section: `kind: Literal["model_estimate"]` is a *single-value*
`Literal`, so no other string (in particular `"observed"`) can ever be
assigned to it without a type-checker violation -- the structural
enforcement of Requirement 9.7 ("Patient-level potential-outcome
differences SHALL be labelled model-dependent estimates, not observed
individual effects.") and Correctness Property 5 ("Patient-level
potential-outcome differences are never labeled as observed ... with no
`"observed"` variant existing in the type at all.").

`estimate_ate` implements the standardization/g-computation ATE required
by Requirement 9.6 ("Cohort treatment effects SHALL be computed by a
declared standardization or g-computation procedure over a declared
target population."): the g-computation formula for the ATE is
`mean_i(Y_hat_i(1) - Y_hat_i(0))` over a target population -- i.e. predict
both potential outcomes for every patient in the population, then average
the individual contrasts. **The declared target population is exactly
the set of `PatientLevelEstimate`s the caller passes in** -- this module
does not filter, resample, or otherwise redefine the population; the
caller is responsible for constructing `patient_estimates` from whichever
population (full eligible cohort, support-restricted cohort, a subgroup,
...) is the declared target population for a given estimate (Requirement
10.11: restricting the cohort redefines the supported target population,
and the corresponding estimand is reported explicitly by the caller
choosing which patients to pass here).

Scalar choice for `y1_hat`/`y0_hat`
------------------------------------

`models.branch_heads.BranchOutput` gives three scalars per branch
(`event_hazard`, `next_cd4`, `next_cd8`). Of these, `event_hazard` is the
one directly on the hazard/risk scale that corresponds to a single-number
treatment effect (a risk difference), so `build_patient_level_estimate`
below uses `branch_1_output.event_hazard` as `y1_hat` and
`branch_0_output.event_hazard` as `y0_hat`. This is a documented choice
of this module, not a `decisions.md`-frozen parameter; aggregating
`next_cd4`/`next_cd8` treatment effects, if ever needed, would require a
separate helper and is out of scope for this task.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from models.branch_heads import BranchOutput


@dataclass(frozen=True)
class PatientLevelEstimate:
    """One patient's modelled potential-outcome pair, exactly matching
    `design.md`'s Data Models section.

    `kind` is a single-value `Literal["model_estimate"]` -- there is no
    `"observed"` (or any other) variant in the type at all, so a
    type-checker rejects any attempt to construct one with a different
    `kind`. This is the structural enforcement of Requirement 9.7 and
    Correctness Property 5.
    """

    patient_id: str
    y1_hat: float
    y0_hat: float
    kind: Literal["model_estimate"]


def build_patient_level_estimate(
    patient_id: str,
    branch_1_output: BranchOutput,
    branch_0_output: BranchOutput,
) -> PatientLevelEstimate:
    """Bridge a patient's two `models.branch_heads.BranchOutput`s into a
    `PatientLevelEstimate`, using `event_hazard` as the potential-outcome
    scalar (see module docstring for the rationale).

    This function expects `branch_1_output`/`branch_0_output`
    specifically -- i.e. the output of `branch_1_head`/`branch_0_head`
    respectively -- **regardless of each output's `.tag`**
    (`"factual"`/`"counterfactual"`). `BranchOutput.tag` reflects which
    branch matches the *observed* treatment for this particular patient
    (Requirement 9.5), not which branch is "1" vs "0"; `y1_hat` is always
    `branch_1_output.event_hazard` (the potential outcome under treatment
    arm 1) and `y0_hat` is always `branch_0_output.event_hazard` (the
    potential outcome under treatment arm 0), whether or not arm 1 was
    the patient's actual observed arm.
    """
    return PatientLevelEstimate(
        patient_id=patient_id,
        y1_hat=branch_1_output.event_hazard,
        y0_hat=branch_0_output.event_hazard,
        kind="model_estimate",
    )


def estimate_ate(patient_estimates: Sequence[PatientLevelEstimate]) -> float:
    """Standardization/g-computation ATE (Requirement 9.6):
    `mean_i(y1_hat_i - y0_hat_i)` over the target population, where the
    target population is exactly the patients in `patient_estimates` --
    the caller determines the target population by which patients they
    include (see module docstring).

    Raises `ValueError` on an empty sequence (the ATE over an empty
    population is undefined).
    """
    if len(patient_estimates) == 0:
        raise ValueError("estimate_ate requires at least one PatientLevelEstimate")
    return sum(pe.y1_hat - pe.y0_hat for pe in patient_estimates) / len(patient_estimates)


def estimate_cate(
    patient_estimates: Sequence[PatientLevelEstimate],
    subgroup_of: dict[str, str] | Callable[[str], str],
) -> dict[str, float]:
    """Conditional ATE per subgroup (Requirement 9.6, applied within each
    declared subgroup rather than the whole population): groups
    `patient_estimates` by `subgroup_of[patient_id]` (if `subgroup_of` is
    a `dict`) or `subgroup_of(patient_id)` (if callable), then computes
    `estimate_ate()` within each subgroup.

    Returns `{subgroup_label: cate}`. Raises whatever `estimate_ate`
    raises if constructing an empty subgroup were possible -- it is not,
    since subgroups are built only from patients actually present in
    `patient_estimates`.
    """
    lookup: Callable[[str], str]
    if callable(subgroup_of):
        lookup = subgroup_of
    else:
        lookup = lambda patient_id: subgroup_of[patient_id]  # noqa: E731

    groups: dict[str, list[PatientLevelEstimate]] = defaultdict(list)
    for pe in patient_estimates:
        groups[lookup(pe.patient_id)].append(pe)

    return {label: estimate_ate(members) for label, members in groups.items()}
