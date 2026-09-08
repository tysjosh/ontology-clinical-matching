"""Unit tests for `src/causal/msm_or_narrowing.py`.

Builds two synthetic `RandomizedCohort`s (same hand-built-event style as
`test_overlap.py`):

- an "ITT-consistent" cohort reflecting ACTG175's actual ITT structure:
  each patient's `treatment_arm` is fixed at randomization and matches the
  concept_id of their earliest recorded `treatment_events` entry, but some
  patients also have a *later* recorded treatment-switch event (a second,
  different concept_id) -- the ordinary, tolerated ITT pattern
  (`decisions.md` Section 3-4; `configs/target_trials/main.yaml`'s
  `treatment_switching_handling`). Asserts
  `assert_baseline_treatment_estimand_valid` does NOT raise.
- a "time-zero-inconsistent" cohort where a patient's randomized
  `treatment_arm` disagrees with their earliest recorded treatment event's
  concept_id -- asserts `TimeVaryingTreatmentDetectedError` is raised,
  listing the offending patient_id.

**Validates: Requirements 7.11**
"""

from datetime import datetime, timedelta

import pytest

from causal.msm_or_narrowing import (
    TREATMENT_CONCEPT_ID_BY_ARM,
    TimeVaryingTreatmentDetectedError,
    assert_baseline_treatment_estimand_valid,
)
from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


def _treatment_event(patient_id: str, concept_id: str, observed_at: datetime) -> ClinicalEvent:
    return ClinicalEvent(
        patient_id=patient_id,
        encounter_id=None,
        concept_id=concept_id,
        value=None,
        unit=None,
        modality="treatment",
        observed_at=observed_at,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )


def _patient_record(
    patient_id: str,
    *,
    treatment_arm: int,
    initial_concept_id: str,
    switch_concept_id: str | None = None,
) -> PatientRecord:
    events = [_treatment_event(patient_id, initial_concept_id, ASSIGNMENT_TIME)]
    if switch_concept_id is not None:
        events.append(
            _treatment_event(patient_id, switch_concept_id, ASSIGNMENT_TIME + timedelta(weeks=20))
        )
    timeline = build_patient_timeline(events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME,
            event_indicator=False,
            censoring_indicator=True,
            event_type=None,
        ),
    )


class TestIttConsistentCohortPasses:
    def test_no_switching_passes(self):
        patients = (
            _patient_record("P001", treatment_arm=0, initial_concept_id="zidovudine"),
            _patient_record(
                "P002", treatment_arm=1, initial_concept_id="zidovudine+didanosine"
            ),
        )
        cohort = RandomizedCohort(patients=patients)

        assert_baseline_treatment_estimand_valid(cohort)  # should not raise

    def test_tolerated_post_randomization_switch_passes(self):
        """A recorded later switch does not violate the baseline/ITT estimand."""
        patients = (
            _patient_record("P001", treatment_arm=0, initial_concept_id="zidovudine"),
            _patient_record(
                "P002",
                treatment_arm=1,
                initial_concept_id="zidovudine+didanosine",
                switch_concept_id="zidovudine",  # switched off study arm mid-trial
            ),
        )
        cohort = RandomizedCohort(patients=patients)

        assert_baseline_treatment_estimand_valid(cohort)  # should not raise


class TestTimeZeroInconsistentCohortRaises:
    def test_mismatched_time_zero_treatment_raises(self):
        patients = (
            _patient_record("P001", treatment_arm=0, initial_concept_id="zidovudine"),
            # P002 is randomized to arm 1 but their earliest recorded
            # treatment event is arm 0's concept_id -- a time-zero
            # data-consistency failure, not a tolerated later switch.
            _patient_record("P002", treatment_arm=1, initial_concept_id="zidovudine"),
        )
        cohort = RandomizedCohort(patients=patients)

        with pytest.raises(TimeVaryingTreatmentDetectedError) as exc_info:
            assert_baseline_treatment_estimand_valid(cohort)

        assert exc_info.value.patient_ids == ("P002",)

    def test_error_lists_all_offending_patients_sorted(self):
        patients = (
            _patient_record("P002", treatment_arm=1, initial_concept_id="zidovudine"),
            _patient_record("P001", treatment_arm=1, initial_concept_id="zidovudine"),
        )
        cohort = RandomizedCohort(patients=patients)

        with pytest.raises(TimeVaryingTreatmentDetectedError) as exc_info:
            assert_baseline_treatment_estimand_valid(cohort)

        assert exc_info.value.patient_ids == ("P001", "P002")


def test_treatment_concept_id_by_arm_matches_frozen_arms():
    assert TREATMENT_CONCEPT_ID_BY_ARM == {0: "zidovudine", 1: "zidovudine+didanosine"}
