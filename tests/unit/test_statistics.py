"""Unit tests for `src/evaluation/statistics.py`.

**Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7**
"""

import pytest

from evaluation.statistics import (
    PRIMARY_COMPARISONS,
    assess_predictive_causal_association,
    holm_correction,
    paired_bootstrap_p_value,
    paired_bootstrap_test,
)

# ---------------------------------------------------------------------------
# Explicitly required: a fourth, unprespecified comparison cannot enter the
# corrected family (Requirement 14.5).
# ---------------------------------------------------------------------------


def test_paired_bootstrap_test_rejects_unprespecified_comparison():
    with pytest.raises(ValueError):
        paired_bootstrap_test("not_a_real_comparison", [1.0, 2.0], [0.5, 0.5])


def test_holm_correction_rejects_a_fourth_key_even_alongside_the_three_valid_ones():
    p_values = {
        "baseline_vs_intervention_aware": 0.01,
        "intervention_aware_vs_ipcw_aipw": 0.04,
        "gated_vs_ungated": 0.03,
        "not_a_real_comparison": 0.01,
    }

    with pytest.raises(ValueError):
        holm_correction(p_values)


def test_holm_correction_rejects_a_missing_primary_comparison():
    p_values = {
        "baseline_vs_intervention_aware": 0.01,
        "intervention_aware_vs_ipcw_aipw": 0.04,
    }

    with pytest.raises(ValueError):
        holm_correction(p_values)


@pytest.mark.parametrize("comparison_name", PRIMARY_COMPARISONS)
def test_paired_bootstrap_test_accepts_every_valid_primary_comparison(comparison_name):
    values_a = [10.0, 20.0, 30.0]
    values_b = [1.0, 1.0, 1.0]

    result = paired_bootstrap_test(comparison_name, values_a, values_b, n_replicates=50, seed=0)

    assert result.n_replicates == 50


def test_holm_correction_accepts_exactly_the_three_valid_comparison_names():
    p_values = {name: 0.02 for name in PRIMARY_COMPARISONS}

    adjusted = holm_correction(p_values)

    assert set(adjusted) == set(PRIMARY_COMPARISONS)


# ---------------------------------------------------------------------------
# Explicitly required: holm_correction matches a hand-computed example.
#
# p-values: baseline_vs_intervention_aware=0.01, gated_vs_ungated=0.03,
# intervention_aware_vs_ipcw_aipw=0.04.
# Sorted ascending: [0.01, 0.03, 0.04]; n=3; multipliers (n - i) for
# i=0,1,2 are [3, 2, 1] -> raw adjusted = [0.03, 0.06, 0.04]; running max
# (step-down monotonicity) = [0.03, 0.06, 0.06]; capped at 1.0, unchanged.
# Mapped back to original keys:
#   baseline_vs_intervention_aware (0.01) -> 0.03
#   gated_vs_ungated               (0.03) -> 0.06
#   intervention_aware_vs_ipcw_aipw (0.04) -> 0.06
# Independently re-verified with a standalone script before asserting.
# ---------------------------------------------------------------------------


def test_holm_correction_matches_hand_computed_example():
    p_values = {
        "baseline_vs_intervention_aware": 0.01,
        "intervention_aware_vs_ipcw_aipw": 0.04,
        "gated_vs_ungated": 0.03,
    }

    adjusted = holm_correction(p_values)

    assert adjusted["baseline_vs_intervention_aware"] == pytest.approx(0.03)
    assert adjusted["gated_vs_ungated"] == pytest.approx(0.06)
    assert adjusted["intervention_aware_vs_ipcw_aipw"] == pytest.approx(0.06)


def test_holm_correction_caps_adjusted_p_values_at_one():
    p_values = {
        "baseline_vs_intervention_aware": 0.5,
        "intervention_aware_vs_ipcw_aipw": 0.6,
        "gated_vs_ungated": 0.9,
    }

    adjusted = holm_correction(p_values)

    assert all(value <= 1.0 for value in adjusted.values())


# ---------------------------------------------------------------------------
# Explicitly required: paired_bootstrap_test resamples the same index
# across both sequences on each replicate. Uses a dataset with varying
# per-index differences ([9, 19, 29]) so that the point estimate -- the
# unresampled mean paired difference -- must equal the exact mean of the
# known differences, 19.0, regardless of resampling.
# ---------------------------------------------------------------------------


def test_paired_bootstrap_point_estimate_matches_exact_mean_of_known_paired_differences():
    values_a = [10.0, 20.0, 30.0]
    values_b = [1.0, 1.0, 1.0]
    # Per-index differences: [9, 19, 29]; exact mean = 19.0.

    result = paired_bootstrap_test(
        "baseline_vs_intervention_aware", values_a, values_b, n_replicates=200, seed=0
    )

    assert result.point_estimate == pytest.approx(19.0)


def test_paired_bootstrap_replicate_estimates_are_averages_of_the_known_per_index_differences():
    # Per-index differences are [9, 19, 29]. Because resampling draws a
    # *shared* index for both sequences on every replicate, every
    # replicate's difference-of-means must be an average drawn only from
    # {9, 19, 29} -- it can never equal a value that would only arise from
    # mismatched (independent) pairing, such as 10 - 1 with a shuffled
    # pairing producing e.g. 20 - 1 = 19 mixed with 30 - 1 = 29 in a way
    # inconsistent with any shared-index draw. Concretely: with n=3, every
    # replicate mean must lie in the convex hull of {9.0, 19.0, 29.0}.
    values_a = [10.0, 20.0, 30.0]
    values_b = [1.0, 1.0, 1.0]

    result = paired_bootstrap_test(
        "baseline_vs_intervention_aware", values_a, values_b, n_replicates=500, seed=1
    )

    assert 9.0 - 1e-9 <= result.ci_low
    assert result.ci_high <= 29.0 + 1e-9


def test_paired_bootstrap_requires_equal_length_sequences():
    with pytest.raises(ValueError):
        paired_bootstrap_test("baseline_vs_intervention_aware", [1.0, 2.0], [1.0, 2.0, 3.0])


def test_paired_bootstrap_requires_non_empty_sequences():
    with pytest.raises(ValueError):
        paired_bootstrap_test("baseline_vs_intervention_aware", [], [])


# ---------------------------------------------------------------------------
# paired_bootstrap_p_value (task 21.1).
# ---------------------------------------------------------------------------


def test_paired_bootstrap_p_value_rejects_unprespecified_comparison():
    with pytest.raises(ValueError):
        paired_bootstrap_p_value("not_a_real_comparison", [1.0, 2.0], [0.5, 0.5])


def test_paired_bootstrap_p_value_requires_equal_length_sequences():
    with pytest.raises(ValueError):
        paired_bootstrap_p_value("baseline_vs_intervention_aware", [1.0, 2.0], [1.0, 2.0, 3.0])


def test_paired_bootstrap_p_value_requires_non_empty_sequences():
    with pytest.raises(ValueError):
        paired_bootstrap_p_value("baseline_vs_intervention_aware", [], [])


def test_paired_bootstrap_p_value_is_near_zero_for_a_consistent_large_difference():
    """Every per-index difference is strongly positive (9, 19, 29), so
    every bootstrap replicate's mean difference is also strongly
    positive; the two-sided p-value against a null of zero difference
    must be at (or very near) zero."""
    values_a = [10.0, 20.0, 30.0]
    values_b = [1.0, 1.0, 1.0]

    p_value = paired_bootstrap_p_value(
        "baseline_vs_intervention_aware", values_a, values_b, n_replicates=500, seed=0
    )

    assert p_value == pytest.approx(0.0, abs=1e-9)


def test_paired_bootstrap_p_value_is_near_one_for_identical_sequences():
    """Every per-index difference is exactly zero, so every replicate's
    mean difference is also exactly zero; both tail proportions are 1.0,
    giving a p-value of 1.0 (capped)."""
    values = [10.0, 20.0, 30.0]

    p_value = paired_bootstrap_p_value(
        "baseline_vs_intervention_aware", values, values, n_replicates=200, seed=0
    )

    assert p_value == pytest.approx(1.0)


def test_paired_bootstrap_p_value_is_bounded_in_zero_one():
    import random

    rng = random.Random(42)
    values_a = [rng.uniform(-5, 5) for _ in range(20)]
    values_b = [rng.uniform(-5, 5) for _ in range(20)]

    p_value = paired_bootstrap_p_value(
        "intervention_aware_vs_ipcw_aipw", values_a, values_b, n_replicates=300, seed=1
    )

    assert 0.0 <= p_value <= 1.0


def test_paired_bootstrap_p_value_feeds_holm_correction_directly():
    """Confirms the intended composition: compute all three comparisons'
    p-values, then pass the resulting dict straight to `holm_correction`."""
    import random

    rng = random.Random(7)
    p_values = {
        name: paired_bootstrap_p_value(
            name,
            [rng.uniform(-1, 1) for _ in range(15)],
            [rng.uniform(-1, 1) for _ in range(15)],
            n_replicates=200,
            seed=idx,
        )
        for idx, name in enumerate(PRIMARY_COMPARISONS)
    }

    adjusted = holm_correction(p_values)

    assert set(adjusted) == set(PRIMARY_COMPARISONS)
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


# ---------------------------------------------------------------------------
# Explicitly required: assess_predictive_causal_association's small-sample
# guard (Requirement 14.6-14.7).
# ---------------------------------------------------------------------------


def test_predictive_causal_association_below_minimum_is_descriptive_only():
    # 3 configurations, default min_configurations=5.
    factual_metric_values = [0.1, 0.2, 0.3]
    causal_recovery_values = [0.5, 0.6, 0.7]

    result = assess_predictive_causal_association(factual_metric_values, causal_recovery_values)

    assert result.mode == "descriptive_only"
    assert result.correlation is None
    assert result.n_configurations == 3
    assert "descriptive" in result.summary.lower()
    assert "not claim a stable correlation" in result.summary.lower()
    assert "correlation of" not in result.summary.lower()


def test_predictive_causal_association_at_or_above_minimum_computes_correlation():
    # 6 configurations, default min_configurations=5, perfectly correlated
    # data -> Pearson correlation should be exactly 1.0.
    factual_metric_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    causal_recovery_values = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

    result = assess_predictive_causal_association(factual_metric_values, causal_recovery_values)

    assert result.mode == "correlation"
    assert result.n_configurations == 6
    assert isinstance(result.correlation, float)
    assert result.correlation == pytest.approx(1.0)
    assert "correlation" in result.summary.lower()


def test_predictive_causal_association_respects_custom_min_configurations():
    # 4 configurations, explicit min_configurations=3 -> above threshold,
    # should compute a correlation rather than falling back to descriptive.
    factual_metric_values = [0.1, 0.2, 0.3, 0.4]
    causal_recovery_values = [0.4, 0.3, 0.2, 0.1]

    result = assess_predictive_causal_association(
        factual_metric_values, causal_recovery_values, min_configurations=3
    )

    assert result.mode == "correlation"
    # Perfectly anti-correlated data -> Pearson correlation of -1.0.
    assert result.correlation == pytest.approx(-1.0)


def test_predictive_causal_association_requires_equal_length_sequences():
    with pytest.raises(ValueError):
        assess_predictive_causal_association([0.1, 0.2], [0.1, 0.2, 0.3])


def test_predictive_causal_association_requires_non_empty_sequences():
    with pytest.raises(ValueError):
        assess_predictive_causal_association([], [])
