"""Unit tests for `src/uncertainty/calibration.py`.

Uses a small, hand-computable synthetic validation set (6 predictions/
labels) to check:

- `.fit()` called with a bare list/tuple (not wrapped in `ValidationSet`)
  raises `ValidationSetRequiredError`.
- the Brier score matches a hand-computed value.
- the calibration slope/intercept matches calling
  `evaluation.metrics.calibration_slope_intercept` directly on the same
  predictions/labels (consistency check, since `Calibrator` reuses that
  function rather than re-implementing the fit).
- the reliability diagram's bin membership/counts match hand-verified
  bins, including two bins that each receive two predictions.

**Validates: Requirements 10.3, 10.4**
"""

import pytest

from evaluation.metrics import calibration_slope_intercept
from uncertainty.calibration import (
    Calibrator,
    ValidationSet,
    ValidationSetRequiredError,
)

# 6 predictions, deliberately arranged so two land in bin [0.0, 0.1) and
# two land in bin [0.5, 0.6), and two land in singleton bins ([0.1, 0.2)
# and [0.9, 1.0]).
PREDICTIONS = [0.05, 0.08, 0.12, 0.55, 0.58, 0.92]
LABELS = [False, True, False, True, False, True]


# ---------------------------------------------------------------------------
# ValidationSet enforcement
# ---------------------------------------------------------------------------


def test_fit_rejects_bare_list():
    calibrator = Calibrator()
    with pytest.raises(ValidationSetRequiredError):
        calibrator.fit(PREDICTIONS)  # bare list, not a ValidationSet


def test_fit_rejects_bare_tuple_pair():
    calibrator = Calibrator()
    with pytest.raises(ValidationSetRequiredError):
        calibrator.fit((PREDICTIONS, LABELS))


def test_fit_accepts_validation_set():
    calibrator = Calibrator()
    result = calibrator.fit(ValidationSet(predictions=PREDICTIONS, labels=LABELS))
    assert result is calibrator.result_


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_score_hand_computed():
    # (0.05-0)^2 + (0.08-1)^2 + (0.12-0)^2 + (0.55-1)^2 + (0.58-0)^2 + (0.92-1)^2
    #   = 0.0025 + 0.8464 + 0.0144 + 0.2025 + 0.3364 + 0.0064 = 1.4086
    # mean over 6 = 0.234766666...
    expected = 1.4086 / 6
    result = Calibrator().fit(ValidationSet(predictions=PREDICTIONS, labels=LABELS))
    assert result.brier_score == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Calibration slope/intercept consistency with evaluation.metrics
# ---------------------------------------------------------------------------


def test_calibration_slope_intercept_matches_metrics_function():
    expected_intercept, expected_slope = calibration_slope_intercept(PREDICTIONS, LABELS)
    result = Calibrator().fit(ValidationSet(predictions=PREDICTIONS, labels=LABELS))
    assert result.calibration_intercept == pytest.approx(expected_intercept)
    assert result.calibration_slope == pytest.approx(expected_slope)


# ---------------------------------------------------------------------------
# Reliability diagram bin membership/counts
# ---------------------------------------------------------------------------


def test_reliability_diagram_bin_membership_and_counts():
    result = Calibrator().fit(ValidationSet(predictions=PREDICTIONS, labels=LABELS))
    diagram = result.reliability_diagram

    # Expected non-empty bins, in bin order:
    #   bin [0.0, 0.1): predictions 0.05, 0.08 -> mean_pred=0.065,
    #       labels False, True -> mean_observed=0.5, count=2
    #   bin [0.1, 0.2): prediction 0.12 -> mean_pred=0.12,
    #       label False -> mean_observed=0.0, count=1
    #   bin [0.5, 0.6): predictions 0.55, 0.58 -> mean_pred=0.565,
    #       labels True, False -> mean_observed=0.5, count=2
    #   bin [0.9, 1.0]: prediction 0.92 -> mean_pred=0.92,
    #       label True -> mean_observed=1.0, count=1
    assert len(diagram) == 4

    mean_pred_0, mean_obs_0, count_0 = diagram[0]
    assert mean_pred_0 == pytest.approx(0.065)
    assert mean_obs_0 == pytest.approx(0.5)
    assert count_0 == 2

    mean_pred_1, mean_obs_1, count_1 = diagram[1]
    assert mean_pred_1 == pytest.approx(0.12)
    assert mean_obs_1 == pytest.approx(0.0)
    assert count_1 == 1

    mean_pred_2, mean_obs_2, count_2 = diagram[2]
    assert mean_pred_2 == pytest.approx(0.565)
    assert mean_obs_2 == pytest.approx(0.5)
    assert count_2 == 2

    mean_pred_3, mean_obs_3, count_3 = diagram[3]
    assert mean_pred_3 == pytest.approx(0.92)
    assert mean_obs_3 == pytest.approx(1.0)
    assert count_3 == 1


def test_reliability_diagram_boundary_prediction_of_one_falls_in_last_bin():
    predictions = [1.0, 0.0]
    labels = [True, False]
    result = Calibrator().fit(ValidationSet(predictions=predictions, labels=labels))
    diagram = result.reliability_diagram
    assert len(diagram) == 2
    # First non-empty bin is [0.0, 0.1) containing the 0.0 prediction.
    assert diagram[0] == (0.0, 0.0, 1)
    # Last bin [0.9, 1.0] (closed on both ends) contains the 1.0 prediction.
    assert diagram[1] == (1.0, 1.0, 1)


def test_validation_set_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ValidationSet(predictions=[0.1, 0.2], labels=[True])
