"""Unit tests for `causal.solvers` (neural-experiment-pipeline task 7.7).

No numpy: references are closed forms or hand-checked properties.
"""

from __future__ import annotations

import math
import random

import pytest

from causal.solvers import (
    SOLVER_MAX_ITERATIONS,
    SOLVER_TOLERANCE,
    SolverConvergenceError,
    fit_cox_newton,
    fit_linear_exact,
    fit_logistic_newton,
    solve_linear_system,
)


def test_frozen_rule_constants():
    assert SOLVER_TOLERANCE == 1e-8
    assert SOLVER_MAX_ITERATIONS == 100


def test_solve_linear_system_and_singular_detection():
    assert solve_linear_system([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0]) == pytest.approx([0.8, 1.4])
    assert solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]) is None


def test_logistic_one_binary_covariate_equals_closed_form_log_odds():
    # x=1: 30 of 40 positive; x=0: 10 of 50 positive.
    features = [[1.0]] * 40 + [[0.0]] * 50
    labels = [1.0] * 30 + [0.0] * 10 + [1.0] * 10 + [0.0] * 40
    bias, weights, diag = fit_logistic_newton(features, labels)
    assert diag.converged and diag.iterations <= 20
    assert bias == pytest.approx(math.log(10 / 40), abs=1e-10)
    assert weights[0] == pytest.approx(math.log(30 / 10) - math.log(10 / 40), abs=1e-10)


def test_logistic_perfect_separation_is_reported_not_converged():
    features = [[float(x)] for x in (-3, -2, -1, 1, 2, 3)]
    labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    _, _, diag = fit_logistic_newton(features, labels)
    assert not diag.converged
    assert diag.reason in {"singular_hessian", "iteration_cap", "line_search_failed", "non_finite"}
    with pytest.raises(SolverConvergenceError, match="did not converge"):
        raise SolverConvergenceError("separated", diag)


def test_linear_matches_hand_computed_ols_and_zero_columns_are_inert():
    # y = 1 + 2 x exactly, plus an all-zero column.
    features = [[float(x), 0.0] for x in range(6)]
    labels = [1.0 + 2.0 * x for x in range(6)]
    bias, weights, diag = fit_linear_exact(features, labels)
    assert diag.converged
    assert bias == pytest.approx(1.0, abs=1e-12)
    assert weights == pytest.approx([2.0, 0.0], abs=1e-12)


def _cox_data(seed: int = 3, n: int = 60):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = [rng.gauss(0, 1), float(rng.random() < 0.5)]
        hazard = math.exp(0.7 * x[0] - 0.5 * x[1])
        time = round(rng.expovariate(hazard), 1)  # rounding creates ties
        rows.append((time, x, rng.random() < 0.8))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows], [r[0] for r in rows], [r[2] for r in rows]


def _cox_score(beta, features, times, events, weights):
    n, k = len(features), len(beta)
    score = [0.0] * k
    i = 0
    while i < n:
        j = i
        while j + 1 < n and times[j + 1] == times[i]:
            j += 1
        d = sum(weights[m] for m in range(i, j + 1) if events[m])
        if d:
            risk = range(i, n)
            risk_weight = {
                m: weights[m] * math.exp(sum(b * x for b, x in zip(beta, features[m])))
                for m in risk
            }
            s0 = sum(risk_weight.values())
            for a in range(k):
                s1 = sum(risk_weight[m] * features[m][a] for m in risk)
                score[a] += sum(weights[m] * features[m][a] for m in range(i, j + 1) if events[m])
                score[a] -= d * s1 / s0
        i = j + 1
    return score


def test_cox_score_vanishes_at_the_solution_with_ties():
    features, times, events = _cox_data()
    weights = [1.0] * len(features)
    beta, diag, information = fit_cox_newton(features, times, events, weights, eta_clip=30.0)
    assert diag.converged
    assert max(abs(s) for s in _cox_score(beta, features, times, events, weights)) < 1e-8
    assert information[0][0] > 0 and information[1][1] > 0


def test_cox_zero_weight_rows_are_inert():
    features, times, events = _cox_data(seed=5)
    weights = [1.0] * len(features)
    base, _, _ = fit_cox_newton(features, times, events, weights, eta_clip=30.0)
    extra_features = [*features, [9.0, 1.0]]
    extra_times = [*times, times[-1] + 1.0]
    beta, diag, _ = fit_cox_newton(
        extra_features, extra_times, [*events, True], [*weights, 0.0], 30.0
    )
    assert diag.converged
    assert beta == pytest.approx(base, abs=1e-9)


def test_cox_without_events_returns_the_zero_start():
    features, times, _ = _cox_data(seed=7, n=10)
    beta, diag, _ = fit_cox_newton(features, times, [False] * 10, [1.0] * 10, 30.0)
    assert diag.converged and diag.reason == "zero_gradient_at_start"
    assert beta == [0.0, 0.0]


def test_solvers_are_deterministic():
    features, times, events = _cox_data(seed=11)
    first = fit_cox_newton(features, times, events, [1.0] * len(features), 30.0)
    second = fit_cox_newton(features, times, events, [1.0] * len(features), 30.0)
    assert first[0] == second[0] and first[1] == second[1]
    labels = [float(e) for e in events]
    assert fit_logistic_newton(features, labels) == fit_logistic_newton(features, labels)
