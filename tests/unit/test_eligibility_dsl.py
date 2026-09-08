"""Unit tests for `src/protocols/eligibility_dsl.py`.

Covers negation, logical scope (And/Or/nesting), threshold boundaries,
missing-value behavior, temporal eligibility, and the concrete
ACTG175_ELIGIBILITY rule against synthetic eligible/ineligible patients.

**Validates: Requirements 3.8, 3.9**
"""

from datetime import date

import pytest

from protocols.eligibility_dsl import (
    ACTG175_ELIGIBILITY,
    And,
    DiagnosisHistory,
    MedicationExposure,
    Not,
    Or,
    PatientTimeline,
    TemporalWindow,
    Threshold,
)


def _timeline(**baseline_covariates) -> PatientTimeline:
    return PatientTimeline(baseline_covariates=baseline_covariates)


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------


def test_not_inverts_a_passing_clause():
    timeline = _timeline(age=30)
    result = Not(Threshold("age", ">=", 18)).evaluate(timeline)
    assert result.eligible is False
    assert result.missing_fields == []


def test_not_inverts_a_failing_clause():
    timeline = _timeline(age=10)
    result = Not(Threshold("age", ">=", 18)).evaluate(timeline)
    assert result.eligible is True
    assert result.missing_fields == []


def test_not_does_not_launder_missing_value_into_eligible():
    """A missing field under Not must stay ineligible, not flip to eligible."""
    timeline = _timeline()  # age absent
    result = Not(Threshold("age", ">=", 18)).evaluate(timeline)
    assert result.eligible is False
    assert result.missing_fields == ["age"]


# ---------------------------------------------------------------------------
# Logical scope: And / Or / nesting
# ---------------------------------------------------------------------------


def test_and_requires_all_clauses_to_pass():
    timeline = _timeline(age=30, baseline_cd4=300)
    passing = And(Threshold("age", ">=", 18), Threshold("baseline_cd4", ">=", 200))
    failing = And(Threshold("age", ">=", 18), Threshold("baseline_cd4", ">=", 400))

    assert passing.evaluate(timeline).eligible is True
    assert failing.evaluate(timeline).eligible is False


def test_or_requires_at_least_one_clause_to_pass():
    timeline = _timeline(age=10, baseline_cd4=300)
    expr = Or(Threshold("age", ">=", 18), Threshold("baseline_cd4", ">=", 200))
    result = expr.evaluate(timeline)
    assert result.eligible is True


def test_or_fails_when_no_clause_passes():
    timeline = _timeline(age=10, baseline_cd4=100)
    expr = Or(Threshold("age", ">=", 18), Threshold("baseline_cd4", ">=", 200))
    result = expr.evaluate(timeline)
    assert result.eligible is False


def test_nested_and_or_scope():
    """(age>=18 AND cd4>=200) OR (age>=12 AND cd4>=400) style nesting."""
    timeline = _timeline(age=13, baseline_cd4=450)
    expr = Or(
        And(Threshold("age", ">=", 18), Threshold("baseline_cd4", ">=", 200)),
        And(Threshold("age", ">=", 12), Threshold("baseline_cd4", ">=", 400)),
    )
    result = expr.evaluate(timeline)
    assert result.eligible is True


def test_and_requires_at_least_one_clause_constructor_guard():
    with pytest.raises(ValueError):
        And()


def test_or_requires_at_least_one_clause_constructor_guard():
    with pytest.raises(ValueError):
        Or()


# ---------------------------------------------------------------------------
# Threshold boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "boundary_value", "observed", "expected"),
    [
        (">=", 200, 200, True),
        (">=", 200, 199, False),
        ("<=", 500, 500, True),
        ("<=", 500, 501, False),
        (">", 200, 200, False),
        (">", 200, 201, True),
        ("<", 500, 500, False),
        ("<", 500, 499, True),
        ("==", 200, 200, True),
        ("==", 200, 201, False),
        ("!=", 200, 201, True),
        ("!=", 200, 200, False),
    ],
)
def test_threshold_boundary_behavior(op, boundary_value, observed, expected):
    timeline = _timeline(baseline_cd4=observed)
    result = Threshold("baseline_cd4", op, boundary_value).evaluate(timeline)
    assert result.eligible is expected


def test_threshold_rejects_unsupported_operator():
    with pytest.raises(ValueError):
        Threshold("age", "~=", 18)


# ---------------------------------------------------------------------------
# Missing-value behavior
# ---------------------------------------------------------------------------


def test_threshold_missing_field_is_ineligible_and_reported():
    timeline = _timeline()  # no fields at all
    result = Threshold("age", ">=", 18).evaluate(timeline)
    assert result.eligible is False
    assert result.missing_fields == ["age"]


def test_and_unions_missing_fields_from_all_clauses():
    timeline = _timeline()
    expr = And(Threshold("age", ">=", 18), Threshold("baseline_cd4", ">=", 200))
    result = expr.evaluate(timeline)
    assert result.eligible is False
    assert set(result.missing_fields) == {"age", "baseline_cd4"}


def test_or_reports_missing_fields_even_when_another_clause_passes():
    """A passing Or must not hide that a different clause's data was missing."""
    timeline = _timeline(age=30)  # baseline_cd4 missing
    expr = Or(Threshold("age", ">=", 18), Threshold("baseline_cd4", ">=", 200))
    result = expr.evaluate(timeline)
    assert result.eligible is True
    assert result.missing_fields == ["baseline_cd4"]


def test_diagnosis_history_missing_is_ineligible():
    timeline = PatientTimeline(baseline_covariates={}, diagnosis_history=None)
    result = DiagnosisHistory("aids_defining_event").evaluate(timeline)
    assert result.eligible is False
    assert result.missing_fields == ["diagnosis_history"]


def test_medication_history_missing_is_ineligible():
    timeline = PatientTimeline(baseline_covariates={}, medication_history=None)
    result = MedicationExposure("zidovudine").evaluate(timeline)
    assert result.eligible is False
    assert result.missing_fields == ["medication_history"]


def test_empty_history_is_not_missing():
    """An empty list (no diagnoses recorded) is a legitimate non-missing state."""
    timeline = PatientTimeline(baseline_covariates={}, diagnosis_history=[])
    result = DiagnosisHistory("aids_defining_event", present=False).evaluate(timeline)
    assert result.eligible is True
    assert result.missing_fields == []


# ---------------------------------------------------------------------------
# Temporal eligibility
# ---------------------------------------------------------------------------


def test_temporal_window_inside_window_passes():
    timeline = _timeline(enrollment_date=date(2020, 6, 15))
    expr = TemporalWindow("enrollment_date", date(2020, 1, 1), date(2020, 12, 31))
    result = expr.evaluate(timeline)
    assert result.eligible is True


def test_temporal_window_outside_window_fails():
    timeline = _timeline(enrollment_date=date(2021, 1, 1))
    expr = TemporalWindow("enrollment_date", date(2020, 1, 1), date(2020, 12, 31))
    result = expr.evaluate(timeline)
    assert result.eligible is False


def test_temporal_window_on_boundary_is_inclusive():
    timeline = _timeline(enrollment_date=date(2020, 12, 31))
    expr = TemporalWindow("enrollment_date", date(2020, 1, 1), date(2020, 12, 31))
    result = expr.evaluate(timeline)
    assert result.eligible is True


def test_temporal_window_missing_field_is_ineligible():
    timeline = _timeline()
    expr = TemporalWindow("enrollment_date", date(2020, 1, 1), date(2020, 12, 31))
    result = expr.evaluate(timeline)
    assert result.eligible is False
    assert result.missing_fields == ["enrollment_date"]


def test_temporal_window_rejects_start_after_end():
    with pytest.raises(ValueError):
        TemporalWindow("enrollment_date", date(2020, 12, 31), date(2020, 1, 1))


def test_diagnosis_history_window_restricts_matches():
    history = [{"concept_id": "aids_defining_event", "date": date(2019, 1, 1)}]
    timeline = PatientTimeline(baseline_covariates={}, diagnosis_history=history)
    expr = DiagnosisHistory(
        "aids_defining_event",
        window_start=date(2020, 1, 1),
        window_end=date(2020, 12, 31),
    )
    result = expr.evaluate(timeline)
    assert result.eligible is False  # the recorded event predates the window


def test_medication_exposure_window_restricts_matches():
    history = [{"concept_id": "zidovudine", "start_time": date(2020, 6, 1)}]
    timeline = PatientTimeline(baseline_covariates={}, medication_history=history)
    expr = MedicationExposure(
        "zidovudine",
        window_start=date(2020, 1, 1),
        window_end=date(2020, 12, 31),
    )
    result = expr.evaluate(timeline)
    assert result.eligible is True


# ---------------------------------------------------------------------------
# ACTG175's concrete eligibility rule
# ---------------------------------------------------------------------------


def test_actg175_eligibility_passes_for_synthetic_eligible_patient():
    timeline = _timeline(hiv_positive=True, baseline_cd4=350, age=34)
    result = ACTG175_ELIGIBILITY.evaluate(timeline)
    assert result.eligible is True
    assert result.missing_fields == []


def test_actg175_eligibility_fails_for_synthetic_ineligible_patient():
    # CD4 too high (above the 200-500 frozen range) and below the age floor.
    timeline = _timeline(hiv_positive=True, baseline_cd4=650, age=10)
    result = ACTG175_ELIGIBILITY.evaluate(timeline)
    assert result.eligible is False


def test_actg175_eligibility_fails_and_flags_missing_field():
    # HIV status and age present, CD4 entirely unrecorded.
    timeline = _timeline(hiv_positive=True, age=25)
    result = ACTG175_ELIGIBILITY.evaluate(timeline)
    assert result.eligible is False
    assert "baseline_cd4" in result.missing_fields
