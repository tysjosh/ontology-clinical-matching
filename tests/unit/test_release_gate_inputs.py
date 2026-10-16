"""Unit tests for the two release-gate inputs frozen by the `decisions.md`
Section 26 amendment of 2026-10-12:

- item A: the cohort-level `uncertainty_valid` rule
  (`evaluation.uncertainty_report.evaluate_uncertainty_validity`);
- item B: the common-support trimming rule
  (`causal.overlap.trim_to_common_support`).
"""

from __future__ import annotations

import math

import pytest

from causal.overlap import PRIMARY_SUPPORT_LOWER, PRIMARY_SUPPORT_UPPER, trim_to_common_support
from evaluation.uncertainty_report import (
    REQUIRED_ENSEMBLE_REPLICAS,
    evaluate_uncertainty_validity,
)
from uncertainty.calibration import CalibrationResult


class _Patient:
    """The two attributes `trim_to_common_support` reads."""

    def __init__(self, patient_id: str, treatment_arm: int) -> None:
        self.patient_id = patient_id
        self.treatment_arm = treatment_arm


def _calibration(slope: float, intercept: float) -> CalibrationResult:
    return CalibrationResult(
        brier_score=0.2,
        calibration_intercept=intercept,
        calibration_slope=slope,
        reliability_diagram=[],
    )


# ---------------------------------------------------------------------------
# Item B: common-support trimming
# ---------------------------------------------------------------------------


def test_trim_removes_exactly_the_patients_outside_the_primary_band():
    patients = [_Patient(f"P{i}", i % 2) for i in range(6)]
    propensities = {"P0": 0.05, "P1": 0.10, "P2": 0.5, "P3": 0.90, "P4": 0.933, "P5": 0.95}

    trim = trim_to_common_support(patients, propensities)

    assert [p.patient_id for p in trim.retained] == ["P1", "P2", "P3"]
    assert trim.trimmed_patient_ids == ("P0", "P4", "P5")
    assert trim.trimmed_count_by_arm == {0: 2, 1: 1}
    assert trim.band == (PRIMARY_SUPPORT_LOWER, PRIMARY_SUPPORT_UPPER) == (0.10, 0.90)


def test_trim_reports_zero_counts_when_nothing_is_removed():
    patients = [_Patient("A", 0), _Patient("B", 1)]
    trim = trim_to_common_support(patients, {"A": 0.4, "B": 0.6})

    assert len(trim.retained) == 2
    assert trim.trimmed_count_by_arm == {0: 0, 1: 0}
    assert trim.to_dict()["trimmed_patient_ids"] == []


def test_trim_refuses_a_patient_without_a_propensity():
    with pytest.raises(ValueError):
        trim_to_common_support([_Patient("A", 0)], {})


def test_trimmed_population_has_no_positivity_warning():
    """After trimming, every remaining score is inside the primary band,
    which is what `OverlapReport.no_unresolved_positivity_warning` checks."""
    patients = [_Patient(f"P{i}", i % 2) for i in range(20)]
    propensities = {p.patient_id: i / 19 for i, p in enumerate(patients)}

    trim = trim_to_common_support(patients, propensities)

    assert all(
        PRIMARY_SUPPORT_LOWER <= propensities[p.patient_id] <= PRIMARY_SUPPORT_UPPER
        for p in trim.retained
    )


# ---------------------------------------------------------------------------
# Item A: uncertainty_valid
# ---------------------------------------------------------------------------


def test_uncertainty_valid_when_all_three_parts_hold():
    validity = evaluate_uncertainty_validity(
        REQUIRED_ENSEMBLE_REPLICAS, _calibration(0.9, -0.1), [True, False, True]
    )
    assert validity.valid
    assert len(validity.reasons) == 3
    assert all("pass" in reason for reason in validity.reasons)


_FULL = REQUIRED_ENSEMBLE_REPLICAS


@pytest.mark.parametrize(
    ("replicas", "calibration", "flags", "failing_part"),
    [
        (4, _calibration(1.0, 0.0), [True], "ensemble_complete"),
        (_FULL, None, [True], "calibration_finite"),
        (_FULL, _calibration(math.nan, math.nan), [True], "calibration_finite"),
        (_FULL, _calibration(1.0, math.inf), [True], "calibration_finite"),
        (_FULL, _calibration(1.0, 0.0), [False, False], "supported_population"),
        (_FULL, _calibration(1.0, 0.0), [], "supported_population"),
    ],
)
def test_uncertainty_invalid_when_any_part_fails(replicas, calibration, flags, failing_part):
    validity = evaluate_uncertainty_validity(replicas, calibration, flags)

    assert not validity.valid
    failing = [reason for reason in validity.reasons if ": fail" in reason]
    assert len(failing) == 1
    assert failing[0].startswith(failing_part)


def test_uncertainty_validity_adds_no_threshold_on_calibration_quality():
    """Item A is procedural: a finite but poor calibration still passes."""
    validity = evaluate_uncertainty_validity(
        REQUIRED_ENSEMBLE_REPLICAS, _calibration(0.1, 3.0), [True]
    )
    assert validity.valid


def test_required_ensemble_replicas_matches_section_15():
    assert REQUIRED_ENSEMBLE_REPLICAS == 5
    torch_training = pytest.importorskip("models.torch_training")
    assert REQUIRED_ENSEMBLE_REPLICAS == torch_training.N_ENSEMBLE_REPLICAS
