"""Unit tests for `src/protocols/time_zero.py::validate_time_zero`.

Covers a clean timeline, an intentionally leaked baseline field, the
boundary case where a baseline timestamp exactly equals assignment time, and
multiple leaked fields being reported together.

**Validates: Requirements 3.10, 3.11**
"""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from protocols.eligibility_dsl import PatientTimeline
from protocols.schema import TargetTrialSpec
from protocols.time_zero import validate_time_zero

MAIN_SPEC_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"
)


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


def test_clean_timeline_has_no_violations(target_trial_spec):
    """All baseline timestamps at or before assignment time -> no violations."""
    timeline = PatientTimeline(
        baseline_covariates={"baseline_cd4": 350, "age": 34},
        assignment_time=ASSIGNMENT_TIME,
        baseline_covariate_timestamps={
            "baseline_cd4": datetime(2020, 1, 10, 8, 0, 0),
            "age": datetime(2020, 1, 5, 8, 0, 0),
        },
    )
    violations = validate_time_zero(timeline, target_trial_spec)
    assert violations == []


def test_leaked_baseline_field_is_rejected(target_trial_spec):
    """A baseline field observed strictly after assignment must be flagged."""
    leaked_time = datetime(2020, 2, 1, 8, 0, 0)  # after ASSIGNMENT_TIME
    timeline = PatientTimeline(
        baseline_covariates={"baseline_cd4": 350, "week_20_cd4": 200},
        assignment_time=ASSIGNMENT_TIME,
        baseline_covariate_timestamps={
            "baseline_cd4": datetime(2020, 1, 10, 8, 0, 0),
            "week_20_cd4": leaked_time,
        },
    )
    violations = validate_time_zero(timeline, target_trial_spec)

    assert len(violations) == 1
    violation = violations[0]
    assert violation.field_name == "week_20_cd4"
    assert violation.observed_at == leaked_time
    assert violation.assignment_time == ASSIGNMENT_TIME


def test_timestamp_exactly_equal_to_assignment_time_is_accepted(target_trial_spec):
    """Coincidence (timestamp == assignment time) is allowed per Requirement 3.10."""
    timeline = PatientTimeline(
        baseline_covariates={"baseline_cd4": 350},
        assignment_time=ASSIGNMENT_TIME,
        baseline_covariate_timestamps={
            "baseline_cd4": ASSIGNMENT_TIME,
        },
    )
    violations = validate_time_zero(timeline, target_trial_spec)
    assert violations == []


def test_multiple_leaked_fields_are_all_reported(target_trial_spec):
    """Every leaked field is reported, not just the first one encountered."""
    leaked_a = datetime(2020, 2, 1, 8, 0, 0)
    leaked_b = datetime(2020, 3, 1, 8, 0, 0)
    timeline = PatientTimeline(
        baseline_covariates={"baseline_cd4": 350},
        assignment_time=ASSIGNMENT_TIME,
        baseline_covariate_timestamps={
            "baseline_cd4": datetime(2020, 1, 10, 8, 0, 0),
            "week_20_cd4": leaked_a,
            "week_96_cd4": leaked_b,
        },
    )
    violations = validate_time_zero(timeline, target_trial_spec)

    flagged_fields = {violation.field_name for violation in violations}
    assert flagged_fields == {"week_20_cd4", "week_96_cd4"}
    assert len(violations) == 2


def test_no_assignment_time_yields_no_violations(target_trial_spec):
    """With no assignment time recorded, there is nothing to compare against."""
    timeline = PatientTimeline(
        baseline_covariates={"baseline_cd4": 350},
        assignment_time=None,
        baseline_covariate_timestamps={
            "baseline_cd4": datetime(2020, 1, 10, 8, 0, 0),
        },
    )
    violations = validate_time_zero(timeline, target_trial_spec)
    assert violations == []


def test_no_timestamps_recorded_yields_no_violations(target_trial_spec):
    """With no per-field timestamps recorded, there is nothing to check."""
    timeline = PatientTimeline(
        baseline_covariates={"baseline_cd4": 350},
        assignment_time=ASSIGNMENT_TIME,
        baseline_covariate_timestamps={},
    )
    violations = validate_time_zero(timeline, target_trial_spec)
    assert violations == []


def test_rejects_unsupported_time_zero_definition(target_trial_spec):
    """A spec whose time_zero.definition isn't randomized_treatment_assignment
    is not silently misinterpreted."""
    unsupported_spec = target_trial_spec.model_copy(
        update={
            "time_zero": target_trial_spec.time_zero.model_copy(
                update={"definition": "some_other_definition"}
            )
        }
    )
    timeline = PatientTimeline(assignment_time=ASSIGNMENT_TIME)

    with pytest.raises(ValueError):
        validate_time_zero(timeline, unsupported_spec)
