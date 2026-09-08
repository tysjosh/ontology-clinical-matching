"""Unit tests for `src/protocols/consistency_checker.py::ProtocolChecker`.

One test per violation category (ineligible, invalid assignment,
treatment-before-eligibility, time-zero violation, outcome-before-intervention,
impossible sequence, endpoint-window violation), plus a clean-cohort test
with no violations.

**Validates: Requirements 3.13**
"""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from protocols.consistency_checker import PatientProtocolRecord, ProtocolChecker
from protocols.eligibility_dsl import PatientTimeline
from protocols.endpoints import PatientEndpointRecord
from protocols.schema import TargetTrialSpec

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


def test_clean_cohort_has_no_violations(target_trial_spec):
    cohort = [_clean_record("p1"), _clean_record("p2")]
    violations = ProtocolChecker.check(cohort, target_trial_spec)
    assert violations == []


def test_ineligible_patient_is_flagged(target_trial_spec):
    """Baseline CD4 outside the ACTG175 eligibility range -> rule='ineligible'."""
    timeline = _eligible_timeline(baseline_covariates={"baseline_cd4": 900})
    record = _clean_record(eligibility_timeline=timeline)

    violations = ProtocolChecker.check([record], target_trial_spec)

    ineligible = [v for v in violations if v.rule == "ineligible"]
    assert len(ineligible) == 1
    assert ineligible[0].patient_id == "p1"
    assert "reason" in ineligible[0].evidence
    assert "missing_fields" in ineligible[0].evidence


def test_invalid_assignment_is_flagged(target_trial_spec):
    """assigned_arm not in {0, 1} -> rule='invalid_assignment'."""
    record = _clean_record(assigned_arm=None)

    violations = ProtocolChecker.check([record], target_trial_spec)

    invalid = [v for v in violations if v.rule == "invalid_assignment"]
    assert len(invalid) == 1
    assert invalid[0].evidence["assigned_arm"] is None


def test_invalid_assignment_rejects_out_of_range_value(target_trial_spec):
    """assigned_arm outside {0, 1} (e.g. arm 2) is also invalid."""
    record = _clean_record(assigned_arm=2)

    violations = ProtocolChecker.check([record], target_trial_spec)

    invalid = [v for v in violations if v.rule == "invalid_assignment"]
    assert len(invalid) == 1
    assert invalid[0].evidence["assigned_arm"] == 2


def test_treatment_before_eligibility_is_flagged(target_trial_spec):
    """Treatment starting before assignment/eligibility -> rule='treatment_before_eligibility'."""
    early_treatment = datetime(2020, 1, 1, 0, 0, 0)  # before ASSIGNMENT_TIME
    record = _clean_record(treatment_start_time=early_treatment)

    violations = ProtocolChecker.check([record], target_trial_spec)

    flagged = [v for v in violations if v.rule == "treatment_before_eligibility"]
    assert len(flagged) == 1
    assert flagged[0].timestamp == early_treatment
    assert flagged[0].evidence["assignment_time"] == ASSIGNMENT_TIME


def test_time_zero_violation_is_translated_from_validate_time_zero(target_trial_spec):
    """A leaked baseline-field timestamp is translated into rule='time_zero_violation'."""
    leaked_time = datetime(2020, 2, 1, 8, 0, 0)  # after ASSIGNMENT_TIME
    timeline = _eligible_timeline(
        baseline_covariates={"week_20_cd4": 200},
        baseline_covariate_timestamps={
            "baseline_cd4": datetime(2020, 1, 10, 8, 0, 0),
            "age": datetime(2020, 1, 5, 8, 0, 0),
            "week_20_cd4": leaked_time,
        },
    )
    record = _clean_record(eligibility_timeline=timeline)

    violations = ProtocolChecker.check([record], target_trial_spec)

    flagged = [v for v in violations if v.rule == "time_zero_violation"]
    assert len(flagged) == 1
    assert flagged[0].timestamp == leaked_time
    assert flagged[0].evidence["field_name"] == "week_20_cd4"


def test_outcome_before_intervention_is_flagged(target_trial_spec):
    """Endpoint event_time before treatment_start_time -> rule='outcome_before_intervention'."""
    death_time = datetime(2020, 1, 1, 0, 0, 0)  # before treatment start
    endpoint_record = _clean_endpoint_record(death_time=death_time)
    record = _clean_record(
        endpoint_record=endpoint_record,
        treatment_start_time=ASSIGNMENT_TIME,
    )

    violations = ProtocolChecker.check([record], target_trial_spec)

    flagged = [v for v in violations if v.rule == "outcome_before_intervention"]
    assert len(flagged) == 1
    assert flagged[0].timestamp == death_time
    assert flagged[0].evidence["reference_field"] == "treatment_start_time"


def test_impossible_sequence_death_before_assignment(target_trial_spec):
    """death_time before assignment_time -> rule='impossible_sequence'."""
    death_time = datetime(2020, 1, 1, 0, 0, 0)  # before ASSIGNMENT_TIME
    endpoint_record = _clean_endpoint_record(
        death_time=death_time,
        last_observed_time=LAST_OBSERVED,
    )
    record = _clean_record(endpoint_record=endpoint_record, treatment_start_time=None)

    violations = ProtocolChecker.check([record], target_trial_spec)

    flagged = [v for v in violations if v.rule == "impossible_sequence"]
    assert len(flagged) == 1
    assert flagged[0].evidence["reason"] == "death_time_before_assignment_time"


def test_impossible_sequence_last_observed_before_assignment(target_trial_spec):
    """last_observed_time before assignment_time -> rule='impossible_sequence'."""
    bad_last_observed = datetime(2019, 1, 1, 0, 0, 0)  # before ASSIGNMENT_TIME
    endpoint_record = _clean_endpoint_record(last_observed_time=bad_last_observed)
    record = _clean_record(endpoint_record=endpoint_record, treatment_start_time=None)

    violations = ProtocolChecker.check([record], target_trial_spec)

    flagged = [v for v in violations if v.rule == "impossible_sequence"]
    assert len(flagged) == 1
    assert flagged[0].evidence["reason"] == "last_observed_time_before_assignment_time"


def test_endpoint_window_violation_is_flagged(target_trial_spec):
    """An event far beyond the declared follow-up horizon -> rule='endpoint_window_violation'."""
    # secondary_horizon_weeks == 96; ENDPOINT_WINDOW_MULTIPLIER == 2 -> 192 weeks max.
    far_future_death = ASSIGNMENT_TIME.replace(year=ASSIGNMENT_TIME.year + 10)
    endpoint_record = _clean_endpoint_record(
        death_time=far_future_death,
        last_observed_time=far_future_death,
    )
    record = _clean_record(endpoint_record=endpoint_record, treatment_start_time=ASSIGNMENT_TIME)

    violations = ProtocolChecker.check([record], target_trial_spec)

    flagged = [v for v in violations if v.rule == "endpoint_window_violation"]
    assert len(flagged) == 1
    assert flagged[0].timestamp == far_future_death


def test_endpoint_within_window_is_not_flagged(target_trial_spec):
    """An event within the declared follow-up horizon does not trigger the window check."""
    within_window_death = datetime(2021, 6, 1)  # well within 192 weeks of ASSIGNMENT_TIME
    endpoint_record = _clean_endpoint_record(
        death_time=within_window_death,
        last_observed_time=within_window_death,
    )
    record = _clean_record(endpoint_record=endpoint_record, treatment_start_time=ASSIGNMENT_TIME)

    violations = ProtocolChecker.check([record], target_trial_spec)

    flagged = [v for v in violations if v.rule == "endpoint_window_violation"]
    assert flagged == []


def test_missing_endpoint_record_skips_endpoint_derived_checks(target_trial_spec):
    """A patient with no endpoint_record does not trigger endpoint-derived checks."""
    record = _clean_record(endpoint_record=None)

    violations = ProtocolChecker.check([record], target_trial_spec)

    endpoint_derived_rules = {
        "outcome_before_intervention",
        "impossible_sequence",
        "endpoint_window_violation",
    }
    assert not any(v.rule in endpoint_derived_rules for v in violations)


def test_multiple_patients_each_accumulate_their_own_violations(target_trial_spec):
    """Violations are attributed to the correct patient_id and a patient may have several."""
    bad_timeline = _eligible_timeline(baseline_covariates={"baseline_cd4": 900})
    bad_record = _clean_record(
        patient_id="bad",
        eligibility_timeline=bad_timeline,
        assigned_arm=None,
    )
    good_record = _clean_record(patient_id="good")

    violations = ProtocolChecker.check([bad_record, good_record], target_trial_spec)

    bad_violations = {v.rule for v in violations if v.patient_id == "bad"}
    good_violations = [v for v in violations if v.patient_id == "good"]

    assert "ineligible" in bad_violations
    assert "invalid_assignment" in bad_violations
    assert good_violations == []
