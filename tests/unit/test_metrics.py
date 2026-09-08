"""Unit tests for `src/evaluation/metrics.py` (factual and selective-prediction
metrics).

Each of the five frozen factual metrics (`decisions.md` Section 18) is
tested against a small, hand-computable synthetic example:

- `concordance_index`: 4 patients, hand-counted comparable/concordant pairs.
- `auroc` / `auprc`: 5 patients (3 positive, 2 negative), hand-computed
  against the standard rank-based/precision-recall definitions.
- `ipcw_brier_score`: perfect predictions with no censoring reduce to the
  plain Brier score (near-zero); a second case checks the plain-Brier
  reduction against a hand-computed value when every patient is observed.
- `calibration_slope_intercept`: well-calibrated synthetic probabilities
  recover slope ~= 1, intercept ~= 0.
- `mean_absolute_error`: hand-computed on a small paired set (CD4/CD8-style).

**Validates: Requirements 8.4, 8.5**

The seven frozen selective-prediction metrics (`decisions.md` Section 18;
Requirement 10.12) are likewise tested against small, hand-computable
synthetic examples of retained/abstained patients.

**Validates: Requirements 10.12**
"""

import math
from datetime import UTC, datetime

import pytest

from cohorts.sealed_reference import InternalReference
from evaluation.metrics import (
    PublishedReference,
    absolute_log_hr_error,
    aurcc,
    auprc,
    auroc,
    calibration_slope_intercept,
    ci_overlap,
    concordance_index,
    effect_direction_agreement,
    error_at_matched_coverage,
    estimated_hazard_ratio,
    incorrect_release_rate,
    ipcw_brier_score,
    mean_absolute_error,
    retained_coverage,
    risk_coverage_curve,
    risk_difference_error_vs_internal_reference,
    selective_risk,
    unnecessary_abstention_rate,
)

# ---------------------------------------------------------------------------
# 1. concordance_index
# ---------------------------------------------------------------------------


def test_concordance_index_hand_computed():
    # Four patients. Higher risk_score should predict an earlier event time.
    #   A: time=1 (event), risk=4.0   (earliest event, should have highest risk)
    #   B: time=2 (event), risk=3.0
    #   C: time=3 (censored), risk=2.0
    #   D: time=4 (event), risk=1.0   (latest event, lowest risk -- as expected)
    #
    # Comparable pairs (shorter time has the event):
    #   (A,B): A(t=1,event) < B(t=2) -> comparable. risk_A(4.0) > risk_B(3.0) -> concordant.
    #   (A,C): A(t=1,event) < C(t=3) -> comparable. risk_A(4.0) > risk_C(2.0) -> concordant.
    #   (A,D): A(t=1,event) < D(t=4) -> comparable. risk_A(4.0) > risk_D(1.0) -> concordant.
    #   (B,C): B(t=2,event) < C(t=3) -> comparable. risk_B(3.0) > risk_C(2.0) -> concordant.
    #   (B,D): B(t=2,event) < D(t=4) -> comparable. risk_B(3.0) > risk_D(1.0) -> concordant.
    #   (C,D): C is censored at t=3 < D's t=4 -> C is NOT the earlier-event member
    #          (C has no event) -> not comparable (censored-before-later-time pairs
    #          where the earlier one is censored are not comparable).
    #
    # -> 5 comparable pairs, all 5 concordant -> C-index = 1.0
    risk_scores = [4.0, 3.0, 2.0, 1.0]
    event_times = [1.0, 2.0, 3.0, 4.0]
    event_indicators = [True, True, False, True]

    result = concordance_index(risk_scores, event_times, event_indicators)
    assert math.isclose(result, 1.0)


def test_concordance_index_hand_computed_with_discordant_pair():
    # Same four patients, but swap the risk scores of A and D so the model
    # gets the (A,D) comparison wrong.
    #   A: time=1 (event), risk=1.0   (should be high, but is now low)
    #   B: time=2 (event), risk=3.0
    #   C: time=3 (censored), risk=2.0
    #   D: time=4 (event), risk=4.0   (should be low, but is now high)
    #
    # Comparable pairs (same 5 as above):
    #   (A,B): risk_A(1.0) > risk_B(3.0)? No -> discordant.
    #   (A,C): risk_A(1.0) > risk_C(2.0)? No -> discordant.
    #   (A,D): risk_A(1.0) > risk_D(4.0)? No -> discordant.
    #   (B,C): risk_B(3.0) > risk_C(2.0)? Yes -> concordant.
    #   (B,D): risk_B(3.0) > risk_D(4.0)? No -> discordant.
    # -> 1 concordant / 5 comparable = 0.2
    risk_scores = [1.0, 3.0, 2.0, 4.0]
    event_times = [1.0, 2.0, 3.0, 4.0]
    event_indicators = [True, True, False, True]

    result = concordance_index(risk_scores, event_times, event_indicators)
    assert math.isclose(result, 0.2)


def test_concordance_index_no_comparable_pairs_is_nan():
    # All censored -> no comparable pairs.
    result = concordance_index([1.0, 2.0], [1.0, 2.0], [False, False])
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# 2. auroc / auprc
# ---------------------------------------------------------------------------

# 5 patients: 3 positive (P1, P2, P3), 2 negative (N1, N2).
# Scores sorted descending: P1=0.9, P2=0.8, N1=0.6, P3=0.4, N2=0.2
_SCORES = [0.9, 0.8, 0.6, 0.4, 0.2]
_LABELS = [True, True, False, True, False]


def test_auroc_hand_computed():
    # Positive scores: 0.9, 0.8, 0.4. Negative scores: 0.6, 0.2.
    # 6 pos/neg pairs total:
    #   (0.9,0.6): pos>neg -> concordant
    #   (0.9,0.2): concordant
    #   (0.8,0.6): concordant
    #   (0.8,0.2): concordant
    #   (0.4,0.6): pos<neg -> discordant
    #   (0.4,0.2): concordant
    # -> 5/6 concordant
    result = auroc(_SCORES, _LABELS)
    assert math.isclose(result, 5.0 / 6.0)


def test_auroc_no_positives_or_no_negatives_is_nan():
    assert math.isnan(auroc([0.1, 0.2], [True, True]))
    assert math.isnan(auroc([0.1, 0.2], [False, False]))


def test_auprc_hand_computed():
    # Sorted descending: 0.9(P), 0.8(P), 0.6(N), 0.4(P), 0.2(N)
    # Running (TP, FP) after each item, precision computed only at positives:
    #   0.9(P): TP=1, FP=0 -> precision = 1/1 = 1.0
    #   0.8(P): TP=2, FP=0 -> precision = 2/2 = 1.0
    #   0.6(N): TP=2, FP=1 (no precision recorded)
    #   0.4(P): TP=3, FP=1 -> precision = 3/4 = 0.75
    #   0.2(N): TP=3, FP=2 (no precision recorded)
    # AUPRC (average precision) = mean(1.0, 1.0, 0.75) = 2.75 / 3
    result = auprc(_SCORES, _LABELS)
    assert math.isclose(result, 2.75 / 3.0)


def test_auprc_no_positives_is_nan():
    assert math.isnan(auprc([0.1, 0.2], [False, False]))


def test_auprc_perfect_ranking_is_one():
    # All positives score higher than all negatives -> AUPRC = 1.0
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    labels = [True, True, True, False, False]
    assert math.isclose(auprc(scores, labels), 1.0)


# ---------------------------------------------------------------------------
# 3. ipcw_brier_score
# ---------------------------------------------------------------------------


def test_ipcw_brier_score_perfect_predictions_near_zero():
    # No censoring before the horizon -> weights are all 1.0 (S_C == 1),
    # and predictions exactly match outcomes -> Brier score is exactly 0.
    risk_scores = [1.0, 0.0, 1.0, 0.0]
    event_times = [10.0, 50.0, 20.0, 60.0]
    event_indicators = [True, False, True, False]
    horizon = 96.0

    result = ipcw_brier_score(risk_scores, event_times, event_indicators, horizon)
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_ipcw_brier_score_reduces_to_plain_brier_when_no_censoring():
    # No patient is censored at all (every patient has `event_indicator=True`)
    # -> the reverse-KM censoring curve has zero censoring events, so
    # S_C(t) == 1.0 everywhere and the IPCW Brier score reduces exactly to
    # the plain Brier score: mean((Y_i - risk_i)^2). Two events occur
    # before the horizon (Y=1); two occur after it (Y=0, but still
    # "observed" since the event -- and hence the true horizon outcome --
    # is known).
    risk_scores = [0.8, 0.3, 0.5, 0.5]
    event_times = [10.0, 20.0, 150.0, 200.0]
    event_indicators = [True, True, True, True]
    horizon = 96.0

    expected = ((1 - 0.8) ** 2 + (1 - 0.3) ** 2 + (0 - 0.5) ** 2 + (0 - 0.5) ** 2) / 4
    result = ipcw_brier_score(risk_scores, event_times, event_indicators, horizon)
    assert math.isclose(result, expected, abs_tol=1e-9)


def test_ipcw_brier_score_worse_predictions_score_higher():
    event_times = [10.0, 20.0, 96.0, 96.0]
    event_indicators = [True, True, False, False]
    horizon = 96.0

    good = ipcw_brier_score([0.9, 0.9, 0.1, 0.1], event_times, event_indicators, horizon)
    bad = ipcw_brier_score([0.1, 0.1, 0.9, 0.9], event_times, event_indicators, horizon)
    assert good < bad


# ---------------------------------------------------------------------------
# 4. calibration_slope_intercept
# ---------------------------------------------------------------------------


def test_calibration_slope_near_one_for_well_calibrated_predictions():
    # Generate labels via a deterministic threshold matching the predicted
    # probability's rank, so the predicted probabilities are, on average,
    # a faithful (well-calibrated) description of outcome frequency.
    import random

    rng = random.Random(42)
    predicted_risks = []
    labels = []
    for i in range(500):
        risk = (i % 100) / 100.0
        risk = min(max(risk, 0.01), 0.99)
        predicted_risks.append(risk)
        labels.append(rng.random() < risk)

    intercept, slope = calibration_slope_intercept(predicted_risks, labels)
    assert math.isfinite(intercept)
    assert math.isfinite(slope)
    assert abs(slope - 1.0) < 0.3
    assert abs(intercept) < 0.3


def test_calibration_slope_near_zero_for_uninformative_predictions():
    # Predicted risk carries no information about outcome (constant
    # predictions can't identify a slope, so vary risk slightly but keep
    # labels independent of it) -> slope should be near 0, not near 1.
    import random

    rng = random.Random(7)
    predicted_risks = [0.3 + 0.4 * rng.random() for _ in range(500)]
    labels = [rng.random() < 0.5 for _ in range(500)]

    _intercept, slope = calibration_slope_intercept(predicted_risks, labels)
    assert abs(slope) < 0.3


# ---------------------------------------------------------------------------
# 5. mean_absolute_error
# ---------------------------------------------------------------------------


def test_mean_absolute_error_hand_computed():
    # |10-8| + |20-25| + |30-30| + |5-9| = 2 + 5 + 0 + 4 = 11; /4 = 2.75
    predicted = [10.0, 20.0, 30.0, 5.0]
    observed = [8.0, 25.0, 30.0, 9.0]
    result = mean_absolute_error(predicted, observed)
    assert math.isclose(result, 2.75)


def test_mean_absolute_error_zero_for_identical_sequences():
    values = [350.0, 420.0, 280.0]
    assert math.isclose(mean_absolute_error(values, values), 0.0)


def test_mean_absolute_error_empty_is_nan():
    assert math.isnan(mean_absolute_error([], []))


# ---------------------------------------------------------------------------
# 6. Selective-prediction metrics
# ---------------------------------------------------------------------------

# Shared 5-patient synthetic set used across several selective-prediction
# tests below. Patients are already ordered by confidence descending (index
# 0 = most confident), so `confidence_scores = [5, 4, 3, 2, 1]` sorts them
# in the same order as their index -- keeping the hand computation simple.
_SEL_PREDICTIONS = [0.9, 0.7, 0.6, 0.3, 0.1]
_SEL_LABELS = [True, True, False, True, False]
_SEL_CONFIDENCE = [5.0, 4.0, 3.0, 2.0, 1.0]
# Patients 0, 1, 3 retained; patients 2, 4 abstained.
_SEL_RETAINED = [True, True, False, True, False]


def test_retained_coverage_hand_computed():
    # 3 of 5 retained -> 0.6
    assert math.isclose(retained_coverage(_SEL_RETAINED), 0.6)


def test_retained_coverage_empty_is_nan():
    assert math.isnan(retained_coverage([]))


def test_selective_risk_hand_computed():
    # Retained patients: idx0 (pred=0.9, label=1) -> (0.9-1)^2 = 0.01
    #                    idx1 (pred=0.7, label=1) -> (0.7-1)^2 = 0.09
    #                    idx3 (pred=0.3, label=1) -> (0.3-1)^2 = 0.49
    # mean = (0.01 + 0.09 + 0.49) / 3 = 0.59 / 3
    result = selective_risk(_SEL_PREDICTIONS, _SEL_LABELS, _SEL_RETAINED)
    assert math.isclose(result, 0.59 / 3.0)


def test_selective_risk_no_retained_is_nan():
    result = selective_risk([0.5, 0.5], [True, False], [False, False])
    assert math.isnan(result)


def test_selective_risk_custom_loss_fn():
    # A trivial loss_fn (always 1.0) makes the "average loss" trivially 1.0
    # regardless of predictions/labels, confirming loss_fn is actually used.
    result = selective_risk(
        _SEL_PREDICTIONS, _SEL_LABELS, _SEL_RETAINED, loss_fn=lambda pred, label: 1.0
    )
    assert math.isclose(result, 1.0)


def test_risk_coverage_curve_hand_computed():
    # Sorted by confidence descending == index order 0..4 already.
    #   k=1: retained {0}                 -> risk = (0.9-1)^2 = 0.01,        coverage=0.2
    #   k=2: retained {0,1}                -> risk = mean(0.01, 0.09) = 0.05, coverage=0.4
    #   k=4: retained {0,1,2,3}            -> risk = mean(0.01,0.09,0.36,0.49)
    #                                                = 0.95/4 = 0.2375,       coverage=0.8
    curve = risk_coverage_curve(_SEL_PREDICTIONS, _SEL_LABELS, _SEL_CONFIDENCE)

    assert len(curve) == 5
    coverage_1, risk_1 = curve[0]
    assert math.isclose(coverage_1, 0.2)
    assert math.isclose(risk_1, 0.01)

    coverage_2, risk_2 = curve[1]
    assert math.isclose(coverage_2, 0.4)
    assert math.isclose(risk_2, 0.05)

    coverage_4, risk_4 = curve[3]
    assert math.isclose(coverage_4, 0.8)
    assert math.isclose(risk_4, 0.2375)

    # Full coverage retains everyone -> plain (unweighted) mean loss.
    coverage_5, risk_5 = curve[4]
    assert math.isclose(coverage_5, 1.0)
    assert math.isclose(risk_5, 0.96 / 5.0)


def test_risk_coverage_curve_empty_is_empty_list():
    assert risk_coverage_curve([], [], []) == []


def test_aurcc_hand_computed_small_curve():
    # 3 patients, confidence order == index order.
    #   k=1: retained {0} (pred=0.8, label=1) -> risk=0.04,  coverage=1/3
    #   k=2: retained {0,1} (pred=0.5, label=0) -> risk=(0.04+0.25)/2=0.145, coverage=2/3
    #   k=3: retained {0,1,2} (pred=0.2, label=1) -> risk=(0.04+0.25+0.64)/3=0.31, coverage=1
    # Trapezoidal area = (1/3)*(0.04+0.145)/2 + (1/3)*(0.145+0.31)/2
    #                   = 0.32 / 3
    predictions = [0.8, 0.5, 0.2]
    labels = [True, False, True]
    confidence = [3.0, 2.0, 1.0]

    result = aurcc(predictions, labels, confidence)
    assert math.isclose(result, 0.32 / 3.0, rel_tol=1e-9)


def test_aurcc_perfect_predictions_near_zero():
    predictions = [1.0, 0.0, 1.0]
    labels = [True, False, True]
    confidence = [3.0, 2.0, 1.0]

    result = aurcc(predictions, labels, confidence)
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_aurcc_bad_predictions_score_higher_than_good_predictions():
    labels = [True, True, False, True, False]
    confidence = [5.0, 4.0, 3.0, 2.0, 1.0]

    good = aurcc([0.9, 0.9, 0.1, 0.9, 0.1], labels, confidence)
    bad = aurcc([0.1, 0.1, 0.9, 0.1, 0.9], labels, confidence)
    assert good < bad


def test_aurcc_fewer_than_two_patients_is_nan():
    assert math.isnan(aurcc([0.5], [True], [1.0]))


def test_incorrect_release_rate_hand_computed():
    # Retained: idx0 (pred=0.9>=0.5 -> positive, label=True) -> correct
    #           idx1 (pred=0.7>=0.5 -> positive, label=True) -> correct
    #           idx3 (pred=0.3<0.5 -> negative, label=True) -> incorrect
    # -> 1 incorrect out of 3 retained = 1/3
    result = incorrect_release_rate(_SEL_PREDICTIONS, _SEL_LABELS, _SEL_RETAINED)
    assert math.isclose(result, 1.0 / 3.0)


def test_incorrect_release_rate_no_retained_is_nan():
    result = incorrect_release_rate([0.5, 0.5], [True, False], [False, False])
    assert math.isnan(result)


def test_unnecessary_abstention_rate_hand_computed():
    # Abstained: idx2 (pred=0.6>=0.5 -> positive, label=False) -> would be wrong
    #            idx4 (pred=0.1<0.5 -> negative, label=False) -> would be correct
    # -> 1 would-be-correct out of 2 abstained = 1/2
    result = unnecessary_abstention_rate(_SEL_PREDICTIONS, _SEL_LABELS, _SEL_RETAINED)
    assert math.isclose(result, 0.5)


def test_unnecessary_abstention_rate_no_abstained_is_nan():
    result = unnecessary_abstention_rate([0.5, 0.5], [True, False], [True, True])
    assert math.isnan(result)


def test_error_at_matched_coverage_matches_risk_coverage_curve_point():
    # Target coverage 0.8 lands exactly on the k=4 point of the hand-computed
    # curve above (risk = 0.2375).
    result = error_at_matched_coverage(
        _SEL_PREDICTIONS, _SEL_LABELS, _SEL_CONFIDENCE, target_coverage=0.8
    )
    assert math.isclose(result, 0.2375)


def test_error_at_matched_coverage_two_model_comparison():
    # Model B: a constant, uninformative prediction (0.6) for every patient,
    # same confidence ranking as model A.
    #   k=4 (coverage=0.8): retained {0,1,2,3}, losses:
    #     idx0 (label=1): (0.6-1)^2=0.16
    #     idx1 (label=1): (0.6-1)^2=0.16
    #     idx2 (label=0): (0.6-0)^2=0.36
    #     idx3 (label=1): (0.6-1)^2=0.16
    #   mean = (0.16*3 + 0.36) / 4 = 0.84 / 4 = 0.21
    predictions_b = [0.6, 0.6, 0.6, 0.6, 0.6]

    risk_a, risk_b = error_at_matched_coverage(
        _SEL_PREDICTIONS,
        _SEL_LABELS,
        _SEL_CONFIDENCE,
        target_coverage=0.8,
        predictions_b=predictions_b,
        confidence_scores_b=_SEL_CONFIDENCE,
    )
    assert math.isclose(risk_a, 0.2375)
    assert math.isclose(risk_b, 0.21)


# ---------------------------------------------------------------------------
# 7. Causal-recovery metrics
# ---------------------------------------------------------------------------
#
# **Validates: Requirements 14.8, 14.9**


def test_estimated_hazard_ratio_hand_computed():
    # log(0.5) -> exp(log(0.5)) == 0.5
    assert math.isclose(estimated_hazard_ratio(math.log(0.5)), 0.5)
    # log_hr = 0 -> HR = 1.0 (no effect)
    assert math.isclose(estimated_hazard_ratio(0.0), 1.0)


def test_absolute_log_hr_error_hand_computed():
    # log_hr_hat == log(0.50) exactly -> error is 0.
    assert math.isclose(absolute_log_hr_error(math.log(0.50)), 0.0, abs_tol=1e-12)

    # log_hr_hat == log(0.60) -> |log(0.60) - log(0.50)| == log(0.60/0.50)
    result = absolute_log_hr_error(math.log(0.60))
    assert math.isclose(result, math.log(0.60 / 0.50))

    # A log_hr_hat on the *harmful* side (HR=2.0) still measures absolute
    # distance from log(0.50).
    result = absolute_log_hr_error(math.log(2.0))
    assert math.isclose(result, abs(math.log(2.0) - math.log(0.50)))


def test_effect_direction_agreement_same_sign_is_true():
    # Both protective (negative log-HR) -> agree.
    assert effect_direction_agreement(math.log(0.7), reference_log_hr=math.log(0.5)) is True
    # Both harmful (positive log-HR) -> agree.
    assert effect_direction_agreement(1.2, reference_log_hr=0.3) is True


def test_effect_direction_agreement_opposite_sign_is_false():
    # log_hr_hat harmful (positive), reference protective (negative, the
    # default log(0.50)) -> disagree.
    assert effect_direction_agreement(0.4) is False
    assert effect_direction_agreement(-0.4, reference_log_hr=0.4) is False


def test_effect_direction_agreement_default_reference_is_published_hr():
    # Default reference_log_hr is log(0.50) (protective) -- a protective
    # estimate agrees by default.
    assert effect_direction_agreement(math.log(0.6)) is True


def test_ci_overlap_overlapping_intervals():
    # [0.3, 0.7] and [0.5, 0.9] overlap (share [0.5, 0.7]).
    assert ci_overlap(0.3, 0.7, 0.5, 0.9) is True
    # Touching endpoints count as overlapping.
    assert ci_overlap(0.0, 0.5, 0.5, 1.0) is True


def test_ci_overlap_non_overlapping_intervals():
    # [0.1, 0.2] and [0.5, 0.9] do not overlap.
    assert ci_overlap(0.1, 0.2, 0.5, 0.9) is False
    # Order of arguments (A vs B) shouldn't matter.
    assert ci_overlap(0.5, 0.9, 0.1, 0.2) is False


def test_risk_difference_error_vs_internal_reference_with_real_internal_reference():
    reference = InternalReference(
        value=0.15,
        cohort_size=100,
        computed_at=datetime.now(UTC),
        metadata={},
    )
    # |0.1 - 0.15| == 0.05
    result = risk_difference_error_vs_internal_reference(0.1, reference)
    assert math.isclose(result, 0.05)


def test_risk_difference_error_vs_internal_reference_rejects_published_reference():
    published = PublishedReference(value=0.5, source="Hammer et al.")
    with pytest.raises(TypeError):
        risk_difference_error_vs_internal_reference(0.1, published)


def test_risk_difference_error_vs_internal_reference_rejects_other_types():
    with pytest.raises(TypeError):
        risk_difference_error_vs_internal_reference(0.1, 0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        risk_difference_error_vs_internal_reference(0.1, None)  # type: ignore[arg-type]
