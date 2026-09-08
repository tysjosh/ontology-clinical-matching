"""Unit test for `src/cohorts/cohort_flow.py`.

Builds a small synthetic candidate population (mix of eligible/ineligible,
arms 0/1/2/3, censored patients and patients with events) as
`data.timelines.PatientTimeline`s and verifies `build_cohort_flow_table`'s
consistency invariants, plus that the Mermaid source is well-formed.

**Validates: Requirements 7.3, 15.3**
"""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from cohorts.cohort_flow import build_cohort_flow_table
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
    """Build a small, hand-constructed event list for one synthetic patient.

    Adapted from `tests/unit/test_cohort_construction.py::_make_patient_events`.
    """
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
def synthetic_flow_inputs():
    """A mixed synthetic candidate population: eligible/ineligible, arms 0-3, varied outcomes."""
    patients = {
        # Eligible, arm 0, censored (no event).
        "p_arm0_censored_a": (
            _make_patient_events("p_arm0_censored_a", week_20_cd4=320.0),
            0,
        ),
        "p_arm0_censored_b": (
            _make_patient_events("p_arm0_censored_b", week_20_cd4=330.0),
            0,
        ),
        # Eligible, arm 1, dies during follow-up (event).
        "p_arm1_death": (
            _make_patient_events(
                "p_arm1_death",
                treatment_concept="zidovudine+didanosine",
                death_time=datetime(2020, 9, 1),
            ),
            1,
        ),
        # Eligible, arm 1, AIDS-defining event.
        "p_arm1_aids": (
            _make_patient_events(
                "p_arm1_aids",
                treatment_concept="zidovudine+didanosine",
                aids_event_time=datetime(2020, 7, 1),
            ),
            1,
        ),
        # Eligible, arm 0, >=50% CD4 decline event.
        "p_arm0_cd4_decline": (
            _make_patient_events(
                "p_arm0_cd4_decline",
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
        # Eligible but arm 2 (zalcitabine) -- excluded at the arm-assignment stage.
        "p_arm2_eligible": (
            _make_patient_events("p_arm2_eligible", treatment_concept="zalcitabine"),
            2,
        ),
        # Eligible but arm 3 (zidovudine+zalcitabine) -- excluded at the arm-assignment stage.
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


def test_cohort_flow_table_counts_are_internally_consistent(
    synthetic_flow_inputs, target_trial_spec
):
    timelines, treatment_arms = synthetic_flow_inputs

    result = build_cohort_flow_table(timelines, treatment_arms, target_trial_spec)
    flow_table = result["flow_table"]

    # initial_population_count == eligible_count + ineligible_excluded_count
    assert (
        flow_table["initial_population_count"]
        == flow_table["eligible_count"] + flow_table["ineligible_excluded_count"]
    )

    # eligible_count == arm_0_1_count + arm_invalid_or_excluded_count
    assert (
        flow_table["eligible_count"]
        == flow_table["arm_0_1_count"] + flow_table["arm_invalid_or_excluded_count"]
    )

    # arm_0_1_count == censored_count + event_count == analyzed_population_count
    assert (
        flow_table["arm_0_1_count"]
        == flow_table["censored_count"] + flow_table["event_count"]
    )
    assert flow_table["arm_0_1_count"] == flow_table["analyzed_population_count"]

    # Sanity check against the known synthetic composition.
    assert flow_table["initial_population_count"] == len(timelines)
    assert flow_table["ineligible_excluded_count"] == 2  # p_ineligible_cd4, p_ineligible_hiv
    assert flow_table["arm_invalid_or_excluded_count"] == 2  # p_arm2_eligible, p_arm3_eligible
    assert flow_table["analyzed_population_count"] == 5
    assert flow_table["censored_count"] == 2
    assert flow_table["event_count"] == 3

    # Stages list is present and its stage counts agree with the flat fields.
    stage_names = [stage["name"] for stage in flow_table["stages"]]
    assert stage_names == [
        "initial_population",
        "eligibility_exclusions",
        "arm_assignment",
        "censoring",
        "analyzed_population",
    ]


def test_mermaid_source_is_well_formed(synthetic_flow_inputs, target_trial_spec):
    timelines, treatment_arms = synthetic_flow_inputs

    result = build_cohort_flow_table(timelines, treatment_arms, target_trial_spec)
    mermaid = result["mermaid"]
    analyzed_population_count = result["flow_table"]["analyzed_population_count"]

    assert mermaid
    assert "flowchart" in mermaid
    assert str(analyzed_population_count) in mermaid


def test_empty_population_returns_zeroed_consistent_table(target_trial_spec):
    result = build_cohort_flow_table([], {}, target_trial_spec)
    flow_table = result["flow_table"]

    assert flow_table["initial_population_count"] == 0
    assert flow_table["eligible_count"] == 0
    assert flow_table["ineligible_excluded_count"] == 0
    assert flow_table["arm_0_1_count"] == 0
    assert flow_table["analyzed_population_count"] == 0
    assert "flowchart" in result["mermaid"]
