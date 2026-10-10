"""The estimators use the converged solvers (neural-experiment-pipeline task 7.8)."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import pytest

from causal.baselines import _CoxRow, _fit_cox
from causal.solvers import SolverConvergenceError, fit_cox_newton
from evaluation.metrics import calibration_slope_intercept
from uncertainty.bootstrap import patient_level_bootstrap


def _rows(seed: int = 1, n: int = 80) -> list[_CoxRow]:
    rng = random.Random(seed)
    start = datetime(2000, 1, 1)
    rows = []
    for _ in range(n):
        arm = float(rng.random() < 0.5)
        days = int(rng.expovariate(math.exp(-0.6 * arm)) * 100) + 1
        rows.append(_CoxRow((arm,), start + timedelta(days=days), rng.random() < 0.8, 1.0))
    return rows


def test_fit_cox_returns_the_converged_newton_solution():
    rows = _rows()
    beta, inverse_information = _fit_cox(rows)
    ordered = sorted(rows, key=lambda r: r.event_time)
    reference, diagnostic, _ = fit_cox_newton(
        [r.features for r in ordered],
        [r.event_time for r in ordered],
        [r.event_indicator for r in ordered],
        [r.weight for r in ordered],
        30.0,
    )
    assert diagnostic.converged
    assert beta == reference
    assert inverse_information is not None and inverse_information[0][0] > 0


def test_calibration_on_separated_predictions_raises_instead_of_reporting_a_slope():
    with pytest.raises(SolverConvergenceError):
        calibration_slope_intercept([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [False] * 3 + [True] * 3)


def test_calibration_recovers_identity_on_well_calibrated_predictions():
    rng = random.Random(0)
    risks = [rng.uniform(0.05, 0.95) for _ in range(4000)]
    labels = [rng.random() < r for r in risks]
    intercept, slope = calibration_slope_intercept(risks, labels)
    assert intercept == pytest.approx(0.0, abs=0.15)
    assert slope == pytest.approx(1.0, abs=0.15)


def test_bootstrap_skips_replicates_whose_fit_does_not_converge():
    calls = {"n": 0}

    def estimator(sample):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            from causal.solvers import ConvergenceDiagnostic

            raise SolverConvergenceError(
                "test", ConvergenceDiagnostic("x", 0, 0.0, 0.0, 0.0, False, "iteration_cap")
            )
        return sum(sample) / len(sample)

    result = patient_level_bootstrap(estimator, list(range(20)), n_replicates=30, seed=1)
    assert math.isfinite(result.ci_low) and math.isfinite(result.ci_high)
