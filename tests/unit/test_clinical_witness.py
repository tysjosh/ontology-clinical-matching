"""Unit tests for `src/witnesses/clinical.py::ClinicalWitness`.

One test per `ProtocolChecker` violation category (ineligible, invalid
assignment, treatment-before-eligibility, outcome-before-intervention,
impossible sequence, endpoint-window violation), confirming
`ClinicalWitness.evaluate()` returns `status="fail"` with the violation
correctly attributed in `evidence["violations_by_rule"]`, plus a
clean-cohort test confirming `status="pass"` with `violation_count=0`.

Fixtures are adapted from `tests/unit/test_consistency_checker.py`.

**Validates: Requirements 11.6, 11.7, 11.8**
"""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from protocols.consistency_checker import PatientProtocolRecord
from protocols.eligibility_dsl import PatientTimeline
from protocols.endpoints import PatientEndpointRecord
from protocols.schema import TargetTrialSpec
from witnesses.clinical import ClinicalWitness

MAIN_SPEC_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"
)

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
LAST_OBSERVED = datetime(2021, 1, 15, 9, 0, 0)


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


def _eligible_timeline(**overrides) -> PatientTimeline:
    baseline_covariates = {
        "hiv_positive": True,
        "baseline_cd4": 350,
        "age": 34,
    }
    baseline_covariates.update(overrides.pop("baseline_covariates", {}))
    kwargs = {
        "baseline_covariates": baseline_covariates,
        "assignment_time": ASSIGNMENT_TIME,
        "baseline_covariate_timestamps": {
            "baseline_cd4": datetime(2020, 1, 10, 8, 0, 0),
            "age": datetime(2020, 1, 5, 8, 0, 0),
        },
    }
    kwargs.update(overrides)
    return PatientTimeline(**kwargs)


def _clean_endpoint_record(**overrides) -> PatientEndpointRecord:
    kwargs = {
        "baseline_cd4": 400,
        "assignment_time": ASSIGNMENT_TIME,
        "last_observed_time": LAST_OBSERVED,
        "cd4_observations": [(datetime(2020, 6, 1), 380)],
        "aids_defining_event_time": None,
        "death_time": None,
    }
    kwargs.update(overrides)
    return PatientEndpointRecord(**kwargs)


def _clean_record(patient_id: str = "p1", **overrides) -> PatientProtocolRecord:
    kwargs = {
        "patient_id": patient_id,
        "eligibility_timeline": _eligible_timeline(),
        "assigned_arm": 1,
        "endpoint_record": _clean_endpoint_record(),
        "treatment_start_time": ASSIGNMENT_TIME,
    }
    kwargs.update(overrides)
    return PatientProtocolRecord(**kwargs)


def test_clean_cohort_passes(target_trial_spec):
    cohort = [_clean_record("p1"), _clean_record("p2")]

    result = ClinicalWitness.evaluate(cohort, target_trial_spec)

    assert result.witness == "clinical"
    assert result.status == "pass"
    assert result.evidence["violation_count"] == 0
    assert result.evidence["violations_by_rule"] == {}
    assert result.evidence["violations"] == []


def test_ineligible_patient_fails(target_trial_spec):
    """Baseline CD4 outside the ACTG175 eligibility range -> eligibility category."""
    timeline = _eligible_timeline(baseline_covariates={"baseline_cd4": 900})
    record = _clean_record(eligibility_timeline=timeline)

    result = ClinicalWitness.evaluate([record], target_trial_spec)

    assert result.status == "fail"
    assert result.evidence["violations_by_rule"]["ineligible"] == 1
    assert result.evidence["violation_count"] == 1


def test_invalid_assignment_fails(target_trial_spec):
    """assigned_arm not in {0, 1} -> treatment-assignment-permitted category."""
    record = _clean_record(assigned_arm=None)

    result = ClinicalWitness.evaluate([record], target_trial_spec)

    assert result.status == "fail"
    assert result.evidence["violations_by_rule"]["invalid_assignment"] == 1


def test_treatment_before_eligibility_fails(target_trial_spec):
    """Treatment starting before assignment/eligibility -> event-timing category."""
    early_treatment = datetime(2020, 1, 1, 0, 0, 0)  # before ASSIGNMENT_TIME
    record = _clean_record(treatment_start_time=early_treatment)

    result = ClinicalWitness.evaluate([record], target_trial_spec)

    assert result.status == "fail"
    assert result.evidence["violations_by_rule"]["treatment_before_eligibility"] == 1


def test_outcome_before_intervention_fails(target_trial_spec):
    """Endpoint event_time before treatment_start_time -> event-timing category."""
    death_time = datetime(2020, 1, 1, 0, 0, 0)  # before treatment start
    endpoint_record = _clean_endpoint_record(death_time=death_time)
    record = _clean_record(
        endpoint_record=endpoint_record,
        treatment_start_time=ASSIGNMENT_TIME,
    )

    result = ClinicalWitness.evaluate([record], target_trial_spec)

    assert result.status == "fail"
    assert result.evidence["violations_by_rule"]["outcome_before_intervention"] == 1


def test_impossible_sequence_fails(target_trial_spec):
    """death_time before assignment_time -> event-timing category."""
    death_time = datetime(2020, 1, 1, 0, 0, 0)  # before ASSIGNMENT_TIME
    endpoint_record = _clean_endpoint_record(
        death_time=death_time,
        last_observed_time=LAST_OBSERVED,
    )
    record = _clean_record(endpoint_record=endpoint_record, treatment_start_time=None)

    result = ClinicalWitness.evaluate([record], target_trial_spec)

    assert result.status == "fail"
    assert result.evidence["violations_by_rule"]["impossible_sequence"] == 1


def test_endpoint_window_violation_fails(target_trial_spec):
    """An event far beyond the declared follow-up horizon -> endpoint-construction category."""
    far_future_death = ASSIGNMENT_TIME.replace(year=ASSIGNMENT_TIME.year + 10)
    endpoint_record = _clean_endpoint_record(
        death_time=far_future_death,
        last_observed_time=far_future_death,
    )
    record = _clean_record(endpoint_record=endpoint_record, treatment_start_time=ASSIGNMENT_TIME)

    result = ClinicalWitness.evaluate([record], target_trial_spec)

    assert result.status == "fail"
    assert result.evidence["violations_by_rule"]["endpoint_window_violation"] == 1


def test_multiple_violations_are_all_attributed(target_trial_spec):
    """A patient with several violations has each rule counted in violations_by_rule."""
    bad_timeline = _eligible_timeline(baseline_covariates={"baseline_cd4": 900})
    bad_record = _clean_record(
        patient_id="bad",
        eligibility_timeline=bad_timeline,
        assigned_arm=None,
    )
    good_record = _clean_record(patient_id="good")

    result = ClinicalWitness.evaluate([bad_record, good_record], target_trial_spec)

    assert result.status == "fail"
    assert result.evidence["violations_by_rule"]["ineligible"] == 1
    assert result.evidence["violations_by_rule"]["invalid_assignment"] == 1
    assert result.evidence["violation_count"] == 2
    patient_ids = {v["patient_id"] for v in result.evidence["violations"]}
    assert patient_ids == {"bad"}
