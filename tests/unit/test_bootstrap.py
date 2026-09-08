"""Unit tests for `src/uncertainty/bootstrap.py`.

The explicitly required test is bootstrap CI coverage on a synthetic
dataset with a known true value: draw a sample from `Normal(mu=10,
sigma=2)`, bootstrap the sample mean, and confirm the resulting CI
contains the true mean, `mu=10`, in the large majority of repeated
trials.

**Validates: Requirement 10.7**
"""

import math
import random

import pytest

from uncertainty.bootstrap import BootstrapCI, patient_level_bootstrap


def _sample_mean(data):
    return sum(data) / len(data)


def test_point_estimate_matches_full_data_estimator():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    result = patient_level_bootstrap(_sample_mean, data, n_replicates=200, seed=0)

    assert result.point_estimate == pytest.approx(3.0)


def test_returns_bootstrap_ci_with_expected_replicate_counts():
    data = [float(i) for i in range(1, 51)]

    result = patient_level_bootstrap(_sample_mean, data, n_replicates=500, seed=1)

    assert isinstance(result, BootstrapCI)
    assert result.n_replicates == 500
    assert result.n_successful == 500
    assert result.ci_low <= result.point_estimate <= result.ci_high


def test_deterministic_for_a_fixed_seed():
    data = [float(i) for i in range(1, 31)]

    first = patient_level_bootstrap(_sample_mean, data, n_replicates=300, seed=42)
    second = patient_level_bootstrap(_sample_mean, data, n_replicates=300, seed=42)

    assert first == second


def test_different_seeds_produce_different_ci_bounds():
    data = [float(i) for i in range(1, 31)]

    first = patient_level_bootstrap(_sample_mean, data, n_replicates=300, seed=1)
    second = patient_level_bootstrap(_sample_mean, data, n_replicates=300, seed=2)

    assert (first.ci_low, first.ci_high) != (second.ci_low, second.ci_high)


def test_empty_data_raises():
    with pytest.raises(ValueError):
        patient_level_bootstrap(_sample_mean, [], n_replicates=100, seed=0)


def test_nan_replicates_are_skipped_and_counted():
    # An estimator that occasionally returns NaN for degenerate resamples
    # (e.g. a resample containing only one distinct value) -- here forced
    # deterministically to always return NaN, so n_successful must be 0
    # and the CI bounds must be NaN rather than raising.
    def always_nan(data):
        return float("nan")

    result = patient_level_bootstrap(always_nan, [1.0, 2.0, 3.0], n_replicates=50, seed=0)

    assert result.n_successful == 0
    assert math.isnan(result.ci_low)
    assert math.isnan(result.ci_high)


def test_single_element_data_bootstraps_to_a_point_mass():
    data = [7.0]

    result = patient_level_bootstrap(_sample_mean, data, n_replicates=100, seed=0)

    assert result.point_estimate == pytest.approx(7.0)
    assert result.ci_low == pytest.approx(7.0)
    assert result.ci_high == pytest.approx(7.0)


def test_bootstrap_ci_coverage_on_synthetic_normal_data_with_known_true_mean():
    """The explicitly required test: bootstrap CI coverage on a synthetic
    dataset with a known true value.

    Repeats the coverage check over several independently seeded
    synthetic datasets drawn from `Normal(mu=10, sigma=2)` and confirms
    the known true mean falls inside the resulting 95% CI in at least
    90% of trials -- consistent with roughly 95% nominal coverage, with
    slack for this being a finite-sample sanity check rather than a
    rigorous statistical validation.
    """
    true_mean = 10.0
    sigma = 2.0
    n_patients = 200
    n_trials = 30

    covered = 0
    for trial_seed in range(n_trials):
        data_rng = random.Random(1000 + trial_seed)
        data = [data_rng.gauss(true_mean, sigma) for _ in range(n_patients)]

        result = patient_level_bootstrap(
            _sample_mean, data, n_replicates=500, seed=trial_seed
        )

        if result.ci_low <= true_mean <= result.ci_high:
            covered += 1

    coverage_rate = covered / n_trials
    assert coverage_rate >= 0.90
