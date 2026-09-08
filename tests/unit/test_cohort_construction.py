"""Unit/integration tests for `src/cohorts/cohort_construction.py`.

Builds a small synthetic set of patients (mix of eligible/ineligible, arms
0/1/2/3, varying endpoint outcomes) as `data.timelines.PatientTimeline`s,
plus the loaded `TargetTrialSpec`, and verifies `build_randomized_cohort`:

- excludes ineligible patients;
- excludes arm-2/3 patients (even when otherwise eligible);
- retains eligible arm-0/1 patients with endpoint results attached;
- is deterministic: two calls on identical input produce identical cohort
  membership (patient_id set) and counts.

**Validates: Requirements 6.1, 7.1, 7.2**
"""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from cohorts.cohort_construction import RandomizedCohort, build_randomized_cohort
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.schema import TargetTrialSpec

MAIN_SPEC_PATH = Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = datetime(2020, 6, 4, 9, 0, 0)
END_OF_FOLLOWUP = datetime(2021, 1, 15, 9, 0, 0)


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


def _event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="cd4_count",
        value=350.0,
        unit="cells/mm3",
        modality="lab",
        observed_at=WEEK_20,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


def _make_patient_events(
    patient_id: str,
    *,
    hiv_positive: bool = True,
    baseline_cd4: float = 350.0,
    age: float = 34.0,
    week_20_cd4: float | None = 300.0,
    death_time: datetime | None = None,
    aids_event_time: datetime | None = None,
    treatment_concept: str = "zidovudine",
) -> list[ClinicalEvent]:
    """Build a small, hand-constructed event list for one synthetic patient."""
    events = [
        _event(
            patient_id,
            concept_id="hiv_positive",
            value=hiv_positive,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="baseline_cd4",
            value=baseline_cd4,
            unit="cells/mm3",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="age",
            value=age,
            unit="years",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id=treatment_concept,
            value=1.0,
            unit=None,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="administrative_censoring",
            value=1.0,
            unit=None,
            modality="censoring",
            observed_at=END_OF_FOLLOWUP,
        ),
    ]
    if week_20_cd4 is not None:
        events.append(
            _event(
                patient_id,
                concept_id="cd4_count",
                value=week_20_cd4,
                unit="cells/mm3",
                modality="lab",
                observed_at=WEEK_20,
            )
        )
    if death_time is not None:
        events.append(
            _event(
                patient_id,
                concept_id="death",
                value=1.0,
                unit=None,
                modality="endpoint",
                observed_at=death_time,
            )
        )
    if aids_event_time is not None:
        events.append(
            _event(
                patient_id,
                concept_id="aids_defining_event",
                value=1.0,
                unit=None,
                modality="endpoint",
                observed_at=aids_event_time,
            )
        )
    return events


@pytest.fixture
def synthetic_cohort_inputs():
    """A mixed synthetic cohort: eligible/ineligible, arms 0/1/2/3, varied outcomes."""
    patients = {
        # Eligible, arm 0, censored (no event).
        "p_arm0_eligible_censored": (
            _make_patient_events("p_arm0_eligible_censored", week_20_cd4=320.0),
            0,
        ),
        # Eligible, arm 1, dies during follow-up (event).
        "p_arm1_eligible_death": (
            _make_patient_events(
                "p_arm1_eligible_death",
                treatment_concept="zidovudine+didanosine",
                death_time=datetime(2020, 9, 1),
            ),
            1,
        ),
        # Eligible, arm 1, AIDS-defining event.
        "p_arm1_eligible_aids": (
            _make_patient_events(
                "p_arm1_eligible_aids",
                treatment_concept="zidovudine+didanosine",
                aids_event_time=datetime(2020, 7, 1),
            ),
            1,
        ),
        # Eligible, arm 0, >=50% CD4 decline event.
        "p_arm0_eligible_cd4_decline": (
            _make_patient_events(
                "p_arm0_eligible_cd4_decline",
                baseline_cd4=400.0,
                week_20_cd4=180.0,  # 55% decline
            ),
            0,
        ),
        # Ineligible: baseline CD4 out of ACTG175 range (900 > 500).
        "p_ineligible_cd4": (
            _make_patient_events("p_ineligible_cd4", baseline_cd4=900.0),
            1,
        ),
        # Ineligible: HIV negative.
        "p_ineligible_hiv": (
            _make_patient_events("p_ineligible_hiv", hiv_positive=False),
            0,
        ),
        # Eligible but arm 2 (zalcitabine) -- must be excluded regardless.
        "p_arm2_eligible": (
            _make_patient_events("p_arm2_eligible", treatment_concept="zalcitabine"),
            2,
        ),
        # Eligible but arm 3 (zidovudine+zalcitabine) -- must be excluded regardless.
        "p_arm3_eligible": (
            _make_patient_events(
                "p_arm3_eligible", treatment_concept="zidovudine+zalcitabine"
            ),
            3,
        ),
    }

    timelines = [
        build_patient_timeline(events) for events, _arm in patients.values()
    ]
    treatment_arms = {patient_id: arm for patient_id, (_events, arm) in patients.items()}
    return timelines, treatment_arms


def test_eligible_arm_0_1_patients_are_retained_with_endpoint_results(
    synthetic_cohort_inputs, target_trial_spec
):
    timelines, treatment_arms = synthetic_cohort_inputs

    cohort = build_randomized_cohort(timelines, treatment_arms, target_trial_spec)

    retained_ids = set(cohort.patient_ids)
    assert retained_ids == {
        "p_arm0_eligible_censored",
        "p_arm1_eligible_death",
        "p_arm1_eligible_aids",
        "p_arm0_eligible_cd4_decline",
    }

    by_id = {patient.patient_id: patient for patient in cohort}

    censored = by_id["p_arm0_eligible_censored"]
    assert censored.treatment_arm == 0
    assert censored.endpoint_result.event_indicator is False
    assert censored.endpoint_result.censoring_indicator is True

    death = by_id["p_arm1_eligible_death"]
    assert death.treatment_arm == 1
    assert death.endpoint_result.event_indicator is True
    assert death.endpoint_result.event_type == "death"

    aids = by_id["p_arm1_eligible_aids"]
    assert aids.endpoint_result.event_type == "aids_defining_event"

    cd4_decline = by_id["p_arm0_eligible_cd4_decline"]
    assert cd4_decline.endpoint_result.event_type == "cd4_decline"


def test_ineligible_patients_are_excluded(synthetic_cohort_inputs, target_trial_spec):
    timelines, treatment_arms = synthetic_cohort_inputs

    cohort = build_randomized_cohort(timelines, treatment_arms, target_trial_spec)

    retained_ids = set(cohort.patient_ids)
    assert "p_ineligible_cd4" not in retained_ids
    assert "p_ineligible_hiv" not in retained_ids


def test_arm_2_and_3_patients_are_excluded_even_if_eligible(
    synthetic_cohort_inputs, target_trial_spec
):
    timelines, treatment_arms = synthetic_cohort_inputs

    cohort = build_randomized_cohort(timelines, treatment_arms, target_trial_spec)

    retained_ids = set(cohort.patient_ids)
    assert "p_arm2_eligible" not in retained_ids
    assert "p_arm3_eligible" not in retained_ids

    retained_arms = {patient.treatment_arm for patient in cohort}
    assert retained_arms <= {0, 1}


def test_repeated_construction_is_deterministic(synthetic_cohort_inputs, target_trial_spec):
    """Repeated calls with identical input reproduce identical cohort membership and counts."""
    timelines, treatment_arms = synthetic_cohort_inputs

    cohort_1 = build_randomized_cohort(timelines, treatment_arms, target_trial_spec)
    cohort_2 = build_randomized_cohort(timelines, treatment_arms, target_trial_spec)

    assert cohort_1.patient_ids == cohort_2.patient_ids
    assert len(cohort_1) == len(cohort_2)
    assert cohort_1 == cohort_2


def test_repeated_construction_from_shuffled_input_order_is_still_deterministic(
    synthetic_cohort_inputs, target_trial_spec
):
    """Reordering the input timelines/arm mapping must not change the result."""
    timelines, treatment_arms = synthetic_cohort_inputs

    reversed_timelines = list(reversed(timelines))
    cohort_forward = build_randomized_cohort(timelines, treatment_arms, target_trial_spec)
    cohort_reversed = build_randomized_cohort(
        reversed_timelines, treatment_arms, target_trial_spec
    )

    assert cohort_forward.patient_ids == cohort_reversed.patient_ids
    assert cohort_forward == cohort_reversed


def test_by_arm_returns_only_matching_patients_sorted_by_patient_id(
    synthetic_cohort_inputs, target_trial_spec
):
    timelines, treatment_arms = synthetic_cohort_inputs

    cohort = build_randomized_cohort(timelines, treatment_arms, target_trial_spec)

    arm_0_ids = [patient.patient_id for patient in cohort.by_arm(0)]
    arm_1_ids = [patient.patient_id for patient in cohort.by_arm(1)]

    assert arm_0_ids == sorted(arm_0_ids)
    assert arm_1_ids == sorted(arm_1_ids)
    assert set(arm_0_ids) == {"p_arm0_eligible_censored", "p_arm0_eligible_cd4_decline"}
    assert set(arm_1_ids) == {"p_arm1_eligible_death", "p_arm1_eligible_aids"}


def test_empty_cohort_returns_empty_randomized_cohort(target_trial_spec):
    cohort = build_randomized_cohort([], {}, target_trial_spec)
    assert isinstance(cohort, RandomizedCohort)
    assert len(cohort) == 0
    assert cohort.patient_ids == ()
