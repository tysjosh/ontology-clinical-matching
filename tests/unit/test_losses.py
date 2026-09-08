"""Unit tests for `src/models/losses.py` (task 9.3; `decisions.md`
Section 14.3 "Frozen training objective").

**Validates: decisions.md Section 14.3 (training objective)**
"""

from __future__ import annotations

import math

import pytest

from models.losses import (
    load_lambda_grid,
    masked_huber,
    mmd,
    select_lambda_by_validation_concordance,
    total_loss,
)

# ---------------------------------------------------------------------------
# masked_huber
# ---------------------------------------------------------------------------


def test_masked_huber_hand_computed_with_masked_entries():
    """Three patient/timestep pairs: one observed with a small residual
    (quadratic regime), one observed with a large residual (linear
    regime), one masked (`None`) that must NOT contribute to the loss.
    """
    predicted = [(500.0, 800.0), (500.0, 800.0), (999.0, 999.0)]
    observed = [(500.5, 799.5), (400.0, 700.0), None]
    delta = 1.0

    # Pair 0: residuals (-0.5, 0.5), both |r| <= delta -> quadratic: 0.5*r^2
    huber_0 = 0.5 * 0.5**2 + 0.5 * 0.5**2  # = 0.25

    # Pair 1: residuals (100.0, 100.0), both |r| > delta -> linear: delta*(|r|-0.5*delta)
    huber_1 = delta * (100.0 - 0.5 * delta) + delta * (100.0 - 0.5 * delta)  # = 199.0

    expected = (huber_0 + huber_1) / 2  # mean over the 2 observed pairs (pair 2 masked out)

    result = masked_huber(predicted, observed, delta=delta)
    assert result == pytest.approx(expected)


def test_masked_huber_all_masked_returns_zero():
    predicted = [(1.0, 2.0), (3.0, 4.0)]
    observed = [None, None]
    assert masked_huber(predicted, observed) == 0.0


def test_masked_huber_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        masked_huber([(1.0, 2.0)], [(1.0, 2.0), None])


# ---------------------------------------------------------------------------
# mmd
# ---------------------------------------------------------------------------


def test_mmd_hand_computed_known_mean_difference():
    """Two groups of 1-D-ish (2-D) vectors with a known mean difference:
    mean(z_treated) = (2.0, 0.0), mean(z_control) = (0.0, 0.0) ->
    MMD^2 = (2.0-0.0)^2 + (0.0-0.0)^2 = 4.0.
    """
    z_treated = [(1.0, 1.0), (3.0, -1.0)]
    z_control = [(-1.0, 2.0), (1.0, -2.0)]

    result = mmd(z_treated, z_control)
    assert result == pytest.approx(4.0)


def test_mmd_zero_when_means_identical():
    z_treated = [(1.0, 2.0), (3.0, 4.0)]
    z_control = [(5.0, 0.0), (-1.0, 6.0)]  # mean = (2.0, 3.0), same as z_treated's mean
    assert mmd(z_treated, z_control) == pytest.approx(0.0, abs=1e-12)


def test_mmd_empty_group_returns_zero():
    assert mmd([], [(1.0, 2.0)]) == 0.0
    assert mmd([(1.0, 2.0)], []) == 0.0


def test_mmd_rejects_inconsistent_dimensions():
    with pytest.raises(ValueError):
        mmd([(1.0, 2.0)], [(1.0, 2.0, 3.0)])


# ---------------------------------------------------------------------------
# total_loss
# ---------------------------------------------------------------------------


def test_total_loss_exact_formula_combination():
    survival_nll_value = 2.0
    masked_huber_value = 0.5
    mmd_value = 1.25
    lambda_state = 0.3
    lambda_balance = 1.0

    expected = survival_nll_value + lambda_state * masked_huber_value + lambda_balance * mmd_value

    result = total_loss(
        survival_nll_value, masked_huber_value, mmd_value, lambda_state, lambda_balance
    )
    assert result == pytest.approx(expected)
    assert result == pytest.approx(2.0 + 0.3 * 0.5 + 1.0 * 1.25)


# ---------------------------------------------------------------------------
# load_lambda_grid
# ---------------------------------------------------------------------------


def test_load_lambda_grid_all_nine_frozen_combinations():
    grid = load_lambda_grid()

    assert len(grid) == 9
    expected_values = {0.1, 0.3, 1.0}
    expected_combinations = {
        (lambda_state, lambda_balance)
        for lambda_state in expected_values
        for lambda_balance in expected_values
    }

    actual_combinations = {
        (round(lambda_state, 10), round(lambda_balance, 10))
        for lambda_state, lambda_balance in grid
    }
    assert actual_combinations == expected_combinations
    # No duplicates.
    assert len(set(grid)) == 9


# ---------------------------------------------------------------------------
# select_lambda_by_validation_concordance
#
# THE EXPLICITLY REQUIRED TEST: confirm selection is based ONLY on the
# validation-split data passed in. Because the function's signature has
# no development- or test-split parameter at all (structural
# enforcement, documented in its docstring), the strongest test
# available is: construct validation-split-shaped candidate results
# where exactly one lambda combination is engineered to have obviously
# the highest validation concordance, and confirm that combination is
# selected -- by construction, no other data source could have entered
# the computation, since none is accepted.
# ---------------------------------------------------------------------------


def test_select_lambda_by_validation_concordance_reads_only_validation_split():
    """Only validation-split (risk_scores, event_times, event_indicators)
    triples are passed in for each of three candidate lambda
    combinations. One combination's validation-split risk scores are
    engineered to perfectly rank-order the validation-split event times
    (concordance = 1.0); the other two are engineered to anti-correlate
    or be uninformative (much lower concordance). The winner must be the
    perfectly-concordant combination, confirming the decision is driven
    entirely by the validation-split inputs supplied -- the only inputs
    this function's signature can ever accept.
    """
    event_times = [1.0, 2.0, 3.0, 4.0]
    event_indicators = [True, True, True, True]

    # Perfectly concordant: higher risk score for earlier event time.
    best_risk_scores = [4.0, 3.0, 2.0, 1.0]
    # Anti-concordant: higher risk score for later event time.
    worst_risk_scores = [1.0, 2.0, 3.0, 4.0]
    # Uninformative (all tied): concordance is undefined/low (ties -> 0.5 credit).
    tied_risk_scores = [1.0, 1.0, 1.0, 1.0]

    candidate_results = {
        (0.1, 0.1): (worst_risk_scores, event_times, event_indicators),
        (0.3, 0.3): (tied_risk_scores, event_times, event_indicators),
        (1.0, 1.0): (best_risk_scores, event_times, event_indicators),
    }

    selected = select_lambda_by_validation_concordance(candidate_results)
    assert selected == (1.0, 1.0)


def test_select_lambda_by_validation_concordance_rejects_empty_input():
    with pytest.raises(ValueError):
        select_lambda_by_validation_concordance({})


def test_select_lambda_by_validation_concordance_signature_has_no_dev_or_test_parameter():
    """Structural-enforcement smoke check: the function accepts exactly
    one parameter (the validation-split candidate-results mapping), so
    there is no argument position through which development- or
    test-split data could ever be supplied.
    """
    import inspect

    signature = inspect.signature(select_lambda_by_validation_concordance)
    assert list(signature.parameters) == ["candidate_results"]
