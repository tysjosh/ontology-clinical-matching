"""Unit tests for `src/uncertainty/ood.py`.

Constructs synthetic "in-distribution" patients (propensity near 0.5,
missingness rate close to a reference rate) and synthetic
"out-of-distribution" patients (propensity near the extremes, e.g. 0.98;
missingness rate far from the reference, e.g. all fields missing vs. a low
reference rate), and confirms the OOD patients' scores are strictly higher
than the in-distribution patients' scores for both the propensity-extremity
score (primary) and the missingness-deviation score (secondary).

**Validates: Requirements 10.8**
"""

import pytest

from causal.overlap import PRIMARY_SUPPORT_LOWER, PRIMARY_SUPPORT_UPPER
from uncertainty.ood import (
    missingness_deviation_score,
    patient_missingness_rate_from_indicators,
    propensity_extremity_score,
)

# Reference missingness rate for an assumed development cohort: mostly
# complete baseline data.
_REFERENCE_MISSINGNESS_RATE = 0.05

# In-distribution synthetic patients: propensity near 0.5 (well inside the
# primary common-support band), missingness rate close to the reference.
_IN_DISTRIBUTION_PATIENTS = [
    {"propensity": 0.50, "missingness_rate": 0.05},
    {"propensity": 0.45, "missingness_rate": 0.04},
    {"propensity": 0.55, "missingness_rate": 0.06},
]

# Out-of-distribution synthetic patients: propensity near the extremes
# (outside the primary common-support band), missingness rate far from the
# reference (e.g. nearly all fields missing).
_OUT_OF_DISTRIBUTION_PATIENTS = [
    {"propensity": 0.98, "missingness_rate": 0.95},
    {"propensity": 0.01, "missingness_rate": 1.00},
    {"propensity": 0.99, "missingness_rate": 0.90},
]


def test_propensity_extremity_score_zero_inside_primary_band() -> None:
    assert propensity_extremity_score(0.5) == 0.0
    assert propensity_extremity_score(PRIMARY_SUPPORT_LOWER) == 0.0
    assert propensity_extremity_score(PRIMARY_SUPPORT_UPPER) == 0.0


def test_propensity_extremity_score_positive_and_monotonic_outside_band() -> None:
    just_outside = propensity_extremity_score(PRIMARY_SUPPORT_UPPER + 0.01)
    further_outside = propensity_extremity_score(PRIMARY_SUPPORT_UPPER + 0.08)
    assert just_outside > 0.0
    assert further_outside > just_outside

    # Symmetric on the low side.
    low_just_outside = propensity_extremity_score(PRIMARY_SUPPORT_LOWER - 0.01)
    assert low_just_outside > 0.0


def test_missingness_deviation_score_basic() -> None:
    assert missingness_deviation_score(0.05, 0.05) == 0.0
    assert missingness_deviation_score(0.95, 0.05) == pytest.approx(0.9)


def test_patient_missingness_rate_from_indicators() -> None:
    assert patient_missingness_rate_from_indicators({}) == 0.0
    assert patient_missingness_rate_from_indicators({"a": 0, "b": 0}) == 0.0
    assert patient_missingness_rate_from_indicators({"a": 1, "b": 0}) == 0.5
    assert patient_missingness_rate_from_indicators({"a": 1, "b": 1}) == 1.0


def test_ood_patients_score_strictly_higher_than_in_distribution_patients() -> None:
    in_distribution_propensity_scores = [
        propensity_extremity_score(p["propensity"]) for p in _IN_DISTRIBUTION_PATIENTS
    ]
    ood_propensity_scores = [
        propensity_extremity_score(p["propensity"]) for p in _OUT_OF_DISTRIBUTION_PATIENTS
    ]

    in_distribution_missingness_scores = [
        missingness_deviation_score(p["missingness_rate"], _REFERENCE_MISSINGNESS_RATE)
        for p in _IN_DISTRIBUTION_PATIENTS
    ]
    ood_missingness_scores = [
        missingness_deviation_score(p["missingness_rate"], _REFERENCE_MISSINGNESS_RATE)
        for p in _OUT_OF_DISTRIBUTION_PATIENTS
    ]

    # Every OOD patient must strictly outscore every in-distribution
    # patient, for BOTH score functions.
    assert max(in_distribution_propensity_scores) < min(ood_propensity_scores)
    assert max(in_distribution_missingness_scores) < min(ood_missingness_scores)
