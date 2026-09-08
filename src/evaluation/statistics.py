"""The locked statistical analysis plan: paired bootstrap comparisons and
Holm multiplicity correction for the three frozen primary model
comparisons (Requirement 14.1-14.5; `decisions.md` Section 19
"Statistical plan").

`decisions.md` Section 19 freezes the primary comparison set to exactly
three named model comparisons:

    - `baseline_vs_intervention_aware` (baseline GRU vs. intervention-aware
      model);
    - `intervention_aware_vs_ipcw_aipw` (intervention-aware model vs.
      IPCW-AIPW);
    - `gated_vs_ungated` (complete gated model vs. ungated model).

and mandates "Holm correction for the three primary model comparisons."
Both `paired_bootstrap_test` and `holm_correction` hardcode this set via
`PRIMARY_COMPARISONS` and reject any comparison name outside it -- this is
the enforcement mechanism that a fourth, unprespecified comparison cannot
enter the corrected family (Requirement 14.5).

Paired resampling
------------------

Section 19 also specifies "paired bootstrap comparisons on identical test
patients" with "1,000 patient-level bootstrap replicates" (matching the
`n_replicates=1000` default used throughout `src/uncertainty/bootstrap.py`,
Requirement 10.7). Because the two sequences being compared are two
models' per-patient values on the *same* patients (Requirement 14.4: "model
comparisons on the same evaluation population SHALL use paired
resampling"), each bootstrap replicate must resample the same patient
index for both sequences -- resampling `values_a` and `values_b`
independently would break the pairing and understate/overstate the
variance of the difference. `paired_bootstrap_test` therefore implements
its own small resampling loop (drawing one shared index vector per
replicate and indexing into both sequences with it) rather than delegating
to `uncertainty.bootstrap.patient_level_bootstrap`, whose `estimator_fn`
receives a single resampled sequence and has no way to guarantee two
independently-resampled sequences share the same drawn indices. The
result is still expressed as a `uncertainty.bootstrap.BootstrapCI` over
the per-replicate differences `values_a[i] - values_b[i]`, reusing that
module's result type (`design.md` Section 3.7: one shared engine/result
shape, multiple callers) even though the resampling loop itself is
implemented locally here.

Holm-Bonferroni step-down correction
---------------------------------------

Standard algorithm: sort p-values ascending; for the i-th smallest
(0-indexed) p-value, multiply by `(n - i)`; take the running maximum
across the sorted order to enforce monotonicity (a step-down procedure
must never produce a smaller adjusted p-value for a larger raw p-value);
cap every adjusted value at `1.0`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from uncertainty.bootstrap import BootstrapCI

#: The exact three frozen primary comparisons (`decisions.md` Section 19).
#: Hardcoded and must not be extended without a documented amendment
#: (Section 25) -- this is the "fourth comparison cannot enter" guard for
#: both `paired_bootstrap_test` and `holm_correction`.
PRIMARY_COMPARISONS: tuple[str, ...] = (
    "baseline_vs_intervention_aware",
    "intervention_aware_vs_ipcw_aipw",
    "gated_vs_ungated",
)


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence.

    Reimplemented locally rather than imported from
    `uncertainty.bootstrap._percentile`: that helper is private
    (`_`-prefixed) to its own module, and this mirrors the same small
    formula already duplicated across `causal.weighting._percentile` and
    `causal.negative_controls._percentile` (see `uncertainty.bootstrap`'s
    module docstring, "Percentile method"), matching NumPy's default
    `numpy.percentile(..., method="linear")` convention.
    """
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values[0]

    rank = (percentile / 100.0) * (n - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = rank - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def paired_bootstrap_test(
    comparison_name: str,
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_replicates: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> BootstrapCI:
    """Paired patient-level bootstrap CI for the difference `values_a -
    values_b` on one of the three frozen primary comparisons.

    `values_a[i]` and `values_b[i]` must refer to the same test patient
    (Requirement 14.4, "paired resampling"): on every replicate, one shared
    set of indices is drawn with replacement and used to index into *both*
    sequences, so the per-index pairing is preserved. The point estimate
    and CI are computed over the resulting distribution of per-replicate
    mean paired differences.

    Args:
        comparison_name: must be one of `PRIMARY_COMPARISONS`; any other
            value raises `ValueError` (Requirement 14.5 -- a fourth,
            unprespecified comparison cannot enter the corrected family).
        values_a: per-patient values for the first model/method in the
            comparison, one value per test patient.
        values_b: per-patient values for the second model/method in the
            comparison, aligned index-for-index with `values_a` (same
            patient at the same index).
        n_replicates: number of bootstrap replicates. Defaults to `1000`
            per `decisions.md` Section 19.
        seed: seeds a `random.Random(seed)` instance used for all
            resampling in this call, for determinism.
        ci_level: the confidence level, e.g. `0.95` for a 95% CI (the
            default, per `decisions.md` Section 19).

    Returns:
        A `BootstrapCI` over the paired-difference distribution: the point
        estimate is the mean of `values_a[i] - values_b[i]` across all
        patients (unresampled), and `ci_low`/`ci_high` are the empirical
        percentile bounds of the resampled replicate means.

    Raises:
        ValueError: if `comparison_name` is not one of
            `PRIMARY_COMPARISONS`, or if `values_a` and `values_b` do not
            have the same nonzero length.
    """
    if comparison_name not in PRIMARY_COMPARISONS:
        raise ValueError(
            f"comparison_name={comparison_name!r} is not one of the three frozen "
            f"primary comparisons {PRIMARY_COMPARISONS!r} (decisions.md Section 19); "
            f"an unprespecified comparison cannot enter the corrected family."
        )
    if len(values_a) != len(values_b):
        raise ValueError(
            f"values_a (len={len(values_a)}) and values_b (len={len(values_b)}) must "
            f"have the same length -- this is a paired comparison on identical test "
            f"patients."
        )
    if not values_a:
        raise ValueError("paired_bootstrap_test requires non-empty values_a/values_b")

    n = len(values_a)

    def _mean_paired_diff(indices: Sequence[int]) -> float:
        return sum(values_a[i] - values_b[i] for i in indices) / len(indices)

    full_indices = range(n)
    point_estimate = _mean_paired_diff(full_indices)

    rng = random.Random(seed)
    replicate_estimates: list[float] = []
    for _ in range(n_replicates):
        shared_indices = [rng.randrange(n) for _ in range(n)]
        replicate_estimates.append(_mean_paired_diff(shared_indices))
    replicate_estimates.sort()

    lower_percentile = (1.0 - ci_level) / 2.0 * 100.0
    upper_percentile = (1.0 - (1.0 - ci_level) / 2.0) * 100.0
    ci_low = _percentile(replicate_estimates, lower_percentile)
    ci_high = _percentile(replicate_estimates, upper_percentile)

    return BootstrapCI(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_replicates=n_replicates,
        n_successful=len(replicate_estimates),
    )


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down multiplicity correction, hardcoded to the
    three frozen primary comparisons.

    Standard algorithm: sort p-values ascending; multiply the i-th
    smallest (0-indexed) p-value by `(n - i)`; take the running maximum
    across the sorted order (step-down monotonicity); cap at `1.0`.

    Args:
        p_values: `{comparison_name: raw_p_value}`. The key set must be
            exactly `PRIMARY_COMPARISONS` (order-independent) -- a subset,
            superset, or any other key raises `ValueError` (Requirement
            14.5's "fourth comparison cannot enter" guard, applied to the
            correction step).

    Returns:
        `{comparison_name: holm_adjusted_p_value}`, same keys as the
        input.

    Raises:
        ValueError: if `set(p_values) != set(PRIMARY_COMPARISONS)`.
    """
    if set(p_values) != set(PRIMARY_COMPARISONS):
        raise ValueError(
            f"p_values keys {sorted(p_values)!r} must be exactly the three frozen "
            f"primary comparisons {sorted(PRIMARY_COMPARISONS)!r} (decisions.md "
            f"Section 19); an unprespecified comparison cannot enter the corrected "
            f"family."
        )

    n = len(p_values)
    sorted_names = sorted(p_values, key=lambda name: p_values[name])

    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, name in enumerate(sorted_names):
        raw_adjusted = p_values[name] * (n - i)
        running_max = max(running_max, raw_adjusted)
        adjusted[name] = min(running_max, 1.0)

    return adjusted


# ---------------------------------------------------------------------------
# Predictive-causal association analysis with the small-sample guard
# (Requirement 14.6-14.7; `decisions.md` Section 19: "No predictive-causal
# correlation claim unless enough independent configurations exist to
# support it").
# ---------------------------------------------------------------------------

#: Default minimum number of independent model/seed configurations required
#: before a correlation claim is permitted. `decisions.md` Section 19 does
#: not freeze this exact number, but it freezes "5 random seeds per neural
#: configuration" elsewhere -- 5 is chosen here as a plausible, defensible
#: default tied to that frozen number, and is exposed as an overridable
#: parameter rather than hardcoded.
DEFAULT_MIN_CONFIGURATIONS: int = 5


@dataclass(frozen=True)
class PredictiveCausalAssociationResult:
    """Result of `assess_predictive_causal_association`.

    Attributes:
        mode: `"correlation"` if enough independent configurations were
            available to support a correlation claim; `"descriptive_only"`
            otherwise (Requirement 14.7).
        n_configurations: the number of independent model/seed
            configurations (value pairs) the assessment was run on.
        correlation: the Pearson correlation coefficient across
            `factual_metric_values` and `causal_recovery_values`, populated
            only when `mode == "correlation"`; `None` when
            `mode == "descriptive_only"`.
        summary: a human-readable description of the result, always
            populated regardless of `mode`.
        values: the raw `(factual_metric_values, causal_recovery_values)`
            pair the assessment was computed from, for traceability.
    """

    mode: Literal["correlation", "descriptive_only"]
    n_configurations: int
    correlation: float | None
    summary: str
    values: tuple[Sequence[float], Sequence[float]]


def _pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient, implemented from scratch with the
    standard formula:

        sum((x - mean_x) * (y - mean_y))
        / sqrt(sum((x - mean_x)^2) * sum((y - mean_y)^2))

    Assumes `x` and `y` are non-empty and of equal length; callers are
    responsible for validating this beforehand.
    """
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))

    return numerator / math.sqrt(sum_sq_x * sum_sq_y)


def assess_predictive_causal_association(
    factual_metric_values: Sequence[float],
    causal_recovery_values: Sequence[float],
    min_configurations: int = DEFAULT_MIN_CONFIGURATIONS,
) -> PredictiveCausalAssociationResult:
    """Assess the association between factual predictive performance and
    causal recovery performance across independent model/seed
    configurations, guarded against small-sample correlation claims.

    `factual_metric_values[i]` and `causal_recovery_values[i]` must refer
    to the same independent model/seed configuration (e.g. one value pair
    per trained model instance), aligned index-for-index.

    Args:
        factual_metric_values: one factual predictive-performance value
            per independent model/seed configuration.
        causal_recovery_values: one causal-recovery-performance value per
            independent model/seed configuration, aligned index-for-index
            with `factual_metric_values`.
        min_configurations: the minimum number of independent
            configurations required to support a correlation claim.
            Defaults to `DEFAULT_MIN_CONFIGURATIONS` (5).

    Returns:
        A `PredictiveCausalAssociationResult`. If fewer than
        `min_configurations` value pairs are available, `mode` is
        `"descriptive_only"`, `correlation` is `None`, and `summary`
        explicitly states that the analysis is descriptive only and does
        not claim a stable correlation (Requirement 14.7). Otherwise,
        `mode` is `"correlation"`, `correlation` holds the Pearson
        correlation coefficient across all configurations, and `summary`
        describes it as a correlation across N configurations (Requirement
        14.6).

    Raises:
        ValueError: if `factual_metric_values` and `causal_recovery_values`
            do not have the same length, or if either is empty.
    """
    if len(factual_metric_values) != len(causal_recovery_values):
        raise ValueError(
            f"factual_metric_values (len={len(factual_metric_values)}) and "
            f"causal_recovery_values (len={len(causal_recovery_values)}) must have "
            f"the same length -- one value pair per independent model/seed "
            f"configuration."
        )
    if not factual_metric_values:
        raise ValueError(
            "assess_predictive_causal_association requires non-empty "
            "factual_metric_values/causal_recovery_values"
        )

    n = len(factual_metric_values)

    if n < min_configurations:
        summary = (
            f"Only {n} independent model/seed configuration(s) are available, below "
            f"the minimum of {min_configurations} required to support a correlation "
            f"claim (Requirement 14.6). This analysis is descriptive only and does "
            f"not claim a stable correlation between factual predictive performance "
            f"and causal recovery performance (Requirement 14.7)."
        )
        return PredictiveCausalAssociationResult(
            mode="descriptive_only",
            n_configurations=n,
            correlation=None,
            summary=summary,
            values=(factual_metric_values, causal_recovery_values),
        )

    correlation = _pearson_correlation(factual_metric_values, causal_recovery_values)
    summary = (
        f"Pearson correlation of {correlation:.4f} between factual predictive "
        f"performance and causal recovery performance across {n} independent "
        f"model/seed configurations."
    )
    return PredictiveCausalAssociationResult(
        mode="correlation",
        n_configurations=n,
        correlation=correlation,
        summary=summary,
        values=(factual_metric_values, causal_recovery_values),
    )
