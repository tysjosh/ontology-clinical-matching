"""Unit tests for `evaluation.uncertainty_report.evaluate_uncertainty` and
`fit_calibration_on_validation` (task 16.1; Requirement 10.1-10.7).

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7**
"""

from __future__ import annotations

import pytest

from evaluation.uncertainty_report import (
    UncertaintyReport,
    error_at_matched_coverage_for_gated_vs_ungated,
    evaluate_uncertainty,
    fit_calibration_on_validation,
)
from uncertainty.calibration import ValidationSetRequiredError


def _well_supported_cohort(n: int = 20):
    predictions = [0.1 + 0.03 * i for i in range(n)]
    labels = [i % 2 == 0 for i in range(n)]
    propensities = [0.5] * n  # well within common support
    ensemble_uncertainties = [0.01 * (i % 5) for i in range(n)]
    protocol_valid_flags = [True] * n
    no_hard_violation_flags = [True] * n
    event_times = [50.0 + i for i in range(n)]
    event_indicators = [i % 3 == 0 for i in range(n)]
    return (
        predictions,
        labels,
        propensities,
        ensemble_uncertainties,
        protocol_valid_flags,
        no_hard_violation_flags,
        event_times,
        event_indicators,
    )


def test_evaluate_uncertainty_returns_a_report_with_populated_support_decisions():
    (
        predictions,
        labels,
        propensities,
        ensemble_uncertainties,
        protocol_valid_flags,
        no_hard_violation_flags,
        event_times,
        event_indicators,
    ) = _well_supported_cohort()

    report = evaluate_uncertainty(
        predictions,
        labels,
        propensities,
        ensemble_uncertainties,
        validation_uncertainty_90th_percentile=0.5,
        protocol_valid_flags=protocol_valid_flags,
        no_hard_violation_flags=no_hard_violation_flags,
        event_times=event_times,
        event_indicators=event_indicators,
        horizon=96.0,
    )

    assert isinstance(report, UncertaintyReport)
    assert len(report.support_decisions) == len(predictions)
    assert len(report.retained_flags) == len(predictions)
    assert all(report.retained_flags)  # everything well within support/uncertainty bounds


def test_patients_outside_common_support_are_not_retained():
    (
        predictions,
        labels,
        _propensities,
        ensemble_uncertainties,
        protocol_valid_flags,
        no_hard_violation_flags,
        event_times,
        event_indicators,
    ) = _well_supported_cohort()

    extreme_propensities = [0.99] * len(predictions)  # outside [0.10, 0.90]

    report = evaluate_uncertainty(
        predictions,
        labels,
        extreme_propensities,
        ensemble_uncertainties,
        validation_uncertainty_90th_percentile=0.5,
        protocol_valid_flags=protocol_valid_flags,
        no_hard_violation_flags=no_hard_violation_flags,
        event_times=event_times,
        event_indicators=event_indicators,
        horizon=96.0,
    )

    assert not any(report.retained_flags)
    assert "empty_supported_population" in " ".join(report.notes)


def test_cd4_cd8_mae_computed_when_supplied_nan_when_absent():
    (
        predictions,
        labels,
        propensities,
        ensemble_uncertainties,
        protocol_valid_flags,
        no_hard_violation_flags,
        event_times,
        event_indicators,
    ) = _well_supported_cohort()

    report_without = evaluate_uncertainty(
        predictions,
        labels,
        propensities,
        ensemble_uncertainties,
        validation_uncertainty_90th_percentile=0.5,
        protocol_valid_flags=protocol_valid_flags,
        no_hard_violation_flags=no_hard_violation_flags,
        event_times=event_times,
        event_indicators=event_indicators,
        horizon=96.0,
    )
    assert report_without.cd4_mae != report_without.cd4_mae  # nan
    assert report_without.cd8_mae != report_without.cd8_mae  # nan

    report_with = evaluate_uncertainty(
        predictions,
        labels,
        propensities,
        ensemble_uncertainties,
        validation_uncertainty_90th_percentile=0.5,
        protocol_valid_flags=protocol_valid_flags,
        no_hard_violation_flags=no_hard_violation_flags,
        event_times=event_times,
        event_indicators=event_indicators,
        horizon=96.0,
        cd4_predicted=[300.0] * len(predictions),
        cd4_observed=[310.0] * len(predictions),
        cd8_predicted=[900.0] * len(predictions),
        cd8_observed=[880.0] * len(predictions),
    )
    assert report_with.cd4_mae == 10.0
    assert report_with.cd8_mae == 20.0


def test_fit_calibration_on_validation_wraps_in_validation_set():
    predictions = [0.1, 0.5, 0.9, 0.2, 0.8]
    labels = [False, True, True, False, True]

    result = fit_calibration_on_validation(predictions, labels)

    assert result.brier_score == pytest.approx(
        sum(
            (prediction - (1.0 if label else 0.0)) ** 2
            for prediction, label in zip(predictions, labels, strict=True)
        )
        / len(predictions)
    )


def test_fit_calibration_on_validation_cannot_be_bypassed_with_a_bare_sequence():
    """A plain list, passed directly to `Calibrator.fit` without going
    through `fit_calibration_on_validation`, is rejected -- confirming
    this module has no code path around that enforcement."""
    from uncertainty.calibration import Calibrator

    with pytest.raises(ValidationSetRequiredError):
        Calibrator().fit([0.1, 0.5])  # type: ignore[arg-type]


def test_error_at_matched_coverage_for_gated_vs_ungated_returns_both_models():
    predictions = [0.1 + 0.04 * i for i in range(15)]
    labels = [i % 2 == 0 for i in range(15)]
    gated_confidence = [1.0 - abs(0.5 - p) for p in predictions]
    ungated_confidence = [0.5] * len(predictions)

    risk_gated, risk_ungated = error_at_matched_coverage_for_gated_vs_ungated(
        predictions, gated_confidence, predictions, ungated_confidence, labels, target_coverage=0.8
    )

    assert isinstance(risk_gated, float)
    assert isinstance(risk_ungated, float)


def test_empty_cohort_does_not_raise():
    report = evaluate_uncertainty(
        [], [], [], [], validation_uncertainty_90th_percentile=0.5,
        protocol_valid_flags=[], no_hard_violation_flags=[],
        event_times=[], event_indicators=[], horizon=96.0,
    )
    assert report.support_decisions == ()
    assert report.retained_flags == ()
