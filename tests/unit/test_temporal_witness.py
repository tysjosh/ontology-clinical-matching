"""Unit tests for `src/witnesses/temporal.py::TemporalWitness.evaluate`.

Covers each of the five individual checks (baseline-precedes-treatment via
`time_zero` violations present/absent, outcome-after-time-zero pass/fail,
chronological-ordering pass/fail, the 5%-tolerance longitudinal-error check
pass/fail at the boundary) plus the aggregate pass/fail/uncertain
classification.

**Validates: Requirements 11.1**
"""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from protocols.eligibility_dsl import PatientTimeline
from protocols.schema import TargetTrialSpec
from witnesses.temporal import TemporalWitness, WitnessResult

MAIN_SPEC_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"
)


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
BASELINE_TIME = datetime(2020, 1, 10, 8, 0, 0)
WEEK_20_TIME = datetime(2020, 6, 1, 8, 0, 0)
WEEK_96_TIME = datetime(2021, 12, 1, 8, 0, 0)


def _clean_timeline(spec: TargetTrialSpec) -> tuple[PatientTimeline, TargetTrialSpec]:
    timeline = PatientTimeline(
        baseline_covariates={"baseline_cd4": 350},
        assignment_time=ASSIGNMENT_TIME,
        baseline_covariate_timestamps={"baseline_cd4": BASELINE_TIME},
    )
    return timeline, spec


def _leaked_timeline(spec: TargetTrialSpec) -> tuple[PatientTimeline, TargetTrialSpec]:
    leaked_time = datetime(2020, 2, 1, 8, 0, 0)  # after ASSIGNMENT_TIME
    timeline = PatientTimeline(
        baseline_covariates={"baseline_cd4": 350, "week_20_cd4": 200},
        assignment_time=ASSIGNMENT_TIME,
        baseline_covariate_timestamps={
            "baseline_cd4": BASELINE_TIME,
            "week_20_cd4": leaked_time,
        },
    )
    return timeline, spec


# ---------------------------------------------------------------------------
# Individual check: baseline-precedes-treatment / no-future-leakage
# ---------------------------------------------------------------------------


def test_baseline_precedes_treatment_passes_with_no_time_zero_violations(target_trial_spec):
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=1.0,
        factual_baseline_cd4_cd8_mae=1.0,
    )
    sub_result = result.evidence["baseline_precedes_treatment_and_no_future_leakage"]
    assert sub_result["pass"] is True
    assert sub_result["violation_count"] == 0


def test_baseline_precedes_treatment_fails_with_leaked_baseline_field(target_trial_spec):
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_leaked_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=1.0,
        factual_baseline_cd4_cd8_mae=1.0,
    )
    sub_result = result.evidence["baseline_precedes_treatment_and_no_future_leakage"]
    assert sub_result["pass"] is False
    assert sub_result["violation_count"] == 1
    assert sub_result["violations"][0]["field_name"] == "week_20_cd4"
    assert result.status == "fail"


# ---------------------------------------------------------------------------
# Individual check: outcome-after-time-zero
# ---------------------------------------------------------------------------


def test_outcome_after_time_zero_passes(target_trial_spec):
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=1.0,
        factual_baseline_cd4_cd8_mae=1.0,
    )
    sub_result = result.evidence["outcomes_after_time_zero"]
    assert sub_result["pass"] is True
    assert sub_result["violation_count"] == 0


def test_outcome_before_time_zero_fails(target_trial_spec):
    event_before_assignment = datetime(2019, 1, 1, 0, 0, 0)
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[event_before_assignment],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=1.0,
        factual_baseline_cd4_cd8_mae=1.0,
    )
    sub_result = result.evidence["outcomes_after_time_zero"]
    assert sub_result["pass"] is False
    assert sub_result["violation_count"] == 1
    assert sub_result["violating_indices"] == [0]
    assert result.status == "fail"


def test_mismatched_event_and_assignment_lengths_raise():
    with pytest.raises(ValueError):
        TemporalWitness.evaluate(
            timelines_and_specs=[],
            event_times=[WEEK_96_TIME],
            assignment_times=[],
            landmark_timestamps=[],
        )


# ---------------------------------------------------------------------------
# Individual check: chronological-ordering preservation
# ---------------------------------------------------------------------------


def test_chronological_ordering_passes_for_nondecreasing_landmarks(target_trial_spec):
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=1.0,
        factual_baseline_cd4_cd8_mae=1.0,
    )
    sub_result = result.evidence["chronological_ordering_preserved"]
    assert sub_result["pass"] is True
    assert sub_result["violation_count"] == 0


def test_chronological_ordering_fails_for_out_of_order_landmarks(target_trial_spec):
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_96_TIME, WEEK_20_TIME]],  # out of order
        held_out_cd4_cd8_mae=1.0,
        factual_baseline_cd4_cd8_mae=1.0,
    )
    sub_result = result.evidence["chronological_ordering_preserved"]
    assert sub_result["pass"] is False
    assert sub_result["violation_count"] == 1
    assert sub_result["violating_indices"] == [0]
    assert result.status == "fail"


# ---------------------------------------------------------------------------
# Individual check: held-out longitudinal (CD4/CD8) error tolerance
# ---------------------------------------------------------------------------


def test_longitudinal_error_passes_at_exact_boundary(target_trial_spec):
    """held_out_mae == factual_baseline_mae * 1.05 exactly -> pass (<=, not <)."""
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=10.5,
        factual_baseline_cd4_cd8_mae=10.0,
    )
    sub_result = result.evidence["longitudinal_error_within_tolerance"]
    assert sub_result["pass"] is True
    assert result.status == "pass"


def test_longitudinal_error_fails_just_above_boundary(target_trial_spec):
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=10.51,
        factual_baseline_cd4_cd8_mae=10.0,
    )
    sub_result = result.evidence["longitudinal_error_within_tolerance"]
    assert sub_result["pass"] is False
    assert result.status == "fail"


# ---------------------------------------------------------------------------
# Aggregate classification
# ---------------------------------------------------------------------------


def test_aggregate_pass_when_all_five_conditions_hold(target_trial_spec):
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=10.0,
        factual_baseline_cd4_cd8_mae=10.0,
    )
    assert isinstance(result, WitnessResult)
    assert result.witness == "temporal"
    assert result.status == "pass"


def test_aggregate_fail_on_any_hard_violation_even_with_good_longitudinal_error(
    target_trial_spec,
):
    """A hard violation (leaked baseline field) fails the witness even when
    the longitudinal-error check alone would pass."""
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_leaked_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=1.0,
        factual_baseline_cd4_cd8_mae=1.0,
    )
    assert result.status == "fail"


def test_aggregate_uncertain_when_longitudinal_error_inputs_missing(target_trial_spec):
    """No hard violation, but longitudinal-error inputs are missing -> uncertain,
    not a clean pass or fail."""
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_clean_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=None,
        factual_baseline_cd4_cd8_mae=None,
    )
    assert result.status == "uncertain"
    sub_result = result.evidence["longitudinal_error_within_tolerance"]
    assert sub_result["pass"] is None
    assert sub_result["inputs_available"] is False


def test_aggregate_fail_takes_priority_over_uncertain(target_trial_spec):
    """A hard violation combined with missing longitudinal-error inputs still
    resolves to fail, not uncertain -- fail takes priority."""
    result = TemporalWitness.evaluate(
        timelines_and_specs=[_leaked_timeline(target_trial_spec)],
        event_times=[WEEK_96_TIME],
        assignment_times=[ASSIGNMENT_TIME],
        landmark_timestamps=[[BASELINE_TIME, WEEK_20_TIME, WEEK_96_TIME]],
        held_out_cd4_cd8_mae=None,
        factual_baseline_cd4_cd8_mae=None,
    )
    assert result.status == "fail"
