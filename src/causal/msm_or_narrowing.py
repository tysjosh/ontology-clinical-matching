"""Baseline-treatment (ITT) estimand narrowing check (Requirement 7.11;
`decisions.md` Section 11.3, last paragraph).

`decisions.md` Section 11.3 freezes the following resolution of Requirement
7.11's either/or clause ("use an appropriate longitudinal estimator or
explicitly narrow the trial to a baseline-treatment estimand"):

    If treatment or censoring is found to vary in a way that requires a
    longitudinal (time-varying) estimator, the trial is narrowed to the
    baseline (ITT, single time-zero) estimand rather than implementing a
    marginal structural model -- resolves Requirement 7.11 by choosing the
    narrowing branch, since ACTG 175's primary strategy is
    ITT-at-randomization.

`decisions.md` Section 3-4 and `configs/target_trials/main.yaml` describe
*what* ACTG175's ITT strategy actually tolerates: treatment is assigned
once, at randomization (`assignment_time: "randomization"`,
`strategy: "intention_to_treat"`), and post-randomization treatment
switching is expected to occur and is handled by being
`"recorded_and_described_descriptively_ignored_in_primary_itt_analysis"` --
i.e. a patient's `timeline.treatment_events` MAY legitimately contain more
than one recorded treatment-modality event (a switch), and this is *not* a
structural obstruction to the baseline estimand. Rejecting every cohort
with any recorded switch would be inconsistent with ACTG175's actual frozen
design, which tolerates exactly this under ITT.

What this check actually validates
------------------------------------

Since ordinary recorded treatment switching is tolerated (not an
obstruction), the narrowing check below is scoped to a narrower, genuinely
disqualifying condition: **the randomized `treatment_arm` assigned to each
patient (`decisions.md` Section 3-4: fixed at randomization, never
re-derived) must agree with the treatment concept_id of that patient's
*earliest* recorded `treatment_events` entry.**

Rationale: the baseline/ITT estimand's entire definition rests on treatment
being fixed and known at a single time zero (randomization). If a patient's
first recorded treatment event does not match their randomized arm, that is
not an ordinary post-randomization switch (which occurs *after* time zero
and is explicitly tolerated) -- it is a data-consistency failure that means
time zero's treatment value cannot be trusted, which is exactly the kind of
defect that would force abandoning the single-time-zero estimand in favor
of a longitudinal model that treats treatment as evolving from an
uncertain start. Any later switch (second, third, ... recorded event with a
different concept_id) is left untouched by this check, per the tolerated
ITT design above.

`assert_baseline_treatment_estimand_valid` therefore does not reject
ACTG175's actual ITT structure (including its normal, tolerated
switching) -- it only rejects the narrower, genuinely-invalid case where
`PatientRecord.treatment_arm` and the patient's own recorded treatment
history disagree at time zero.
"""

from __future__ import annotations

from collections.abc import Sequence

from cohorts.cohort_construction import RandomizedCohort

#: The treatment concept_id each retained arm (`decisions.md` Section 3-4;
#: `configs/target_trials/main.yaml`'s `intervention.concept_id` /
#: `comparator.concept_id`) is expected to correspond to at time zero.
TREATMENT_CONCEPT_ID_BY_ARM: dict[int, str] = {
    0: "zidovudine",
    1: "zidovudine+didanosine",
}


class TimeVaryingTreatmentDetectedError(Exception):
    """Raised when a cohort's treatment history is structurally inconsistent
    with the baseline (ITT, single time-zero) estimand.

    This does NOT fire on ordinary, tolerated post-randomization treatment
    switching (see module docstring) -- only when one or more patients'
    randomized `treatment_arm` disagrees with the treatment concept_id of
    their own earliest recorded `treatment_events` entry, which would make
    time zero's treatment value untrustworthy for a baseline estimand.
    """

    def __init__(self, patient_ids: Sequence[str]):
        self.patient_ids: tuple[str, ...] = tuple(patient_ids)
        super().__init__(
            "Baseline-treatment estimand is invalid for patient(s) "
            f"{list(self.patient_ids)!r}: randomized treatment_arm does not "
            "match the earliest recorded treatment_events concept_id. This "
            "indicates the treatment value at time zero cannot be trusted, "
            "which would require a longitudinal (marginal-structural-model) "
            "estimator rather than the frozen baseline/ITT estimand "
            "(decisions.md Section 11.3)."
        )


def assert_baseline_treatment_estimand_valid(cohort: RandomizedCohort) -> None:
    """Assert that `cohort` is consistent with the frozen baseline (ITT) estimand.

    For each patient in `cohort` (skipping any patient with no recorded
    `treatment_events`, which is not expected to occur in a
    `RandomizedCohort` -- see `cohorts.cohort_construction._adapt_to_endpoint_record`,
    which excludes patients lacking a treatment-start time -- but handled
    defensively rather than assumed away), this compares the patient's
    randomized `treatment_arm` against the treatment concept_id of their
    *earliest* recorded treatment event (`TREATMENT_CONCEPT_ID_BY_ARM`).

    Any later recorded treatment event with a different concept_id (an
    ordinary treatment switch, `decisions.md` Section 3-4) is NOT
    inspected and does NOT cause this function to raise -- ACTG175's ITT
    strategy explicitly tolerates such switches
    (`configs/target_trials/main.yaml`'s
    `treatment_switching_handling: "recorded_and_described_descriptively_ignored_in_primary_itt_analysis"`).

    Raises:
        TimeVaryingTreatmentDetectedError: if any patient's `treatment_arm`
            disagrees with their earliest recorded treatment event's
            concept_id, listing every offending `patient_id` (sorted, for
            deterministic error messages).
    """
    offending_patient_ids: list[str] = []
    for patient in cohort.patients:
        treatment_events = patient.timeline.treatment_events
        if not treatment_events:
            continue

        earliest_event = min(treatment_events, key=lambda event: event.observed_at)
        expected_concept_id = TREATMENT_CONCEPT_ID_BY_ARM.get(patient.treatment_arm)

        if expected_concept_id is not None and earliest_event.concept_id != expected_concept_id:
            offending_patient_ids.append(patient.patient_id)

    if offending_patient_ids:
        raise TimeVaryingTreatmentDetectedError(sorted(offending_patient_ids))
