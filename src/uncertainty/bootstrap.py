"""Shared patient-level bootstrap engine (Requirement 10.7; `design.md`
Section 3.7).

`design.md` Section 3.7 specifies one shared bootstrap engine:

    `patient_level_bootstrap(estimator_fn, data, n_replicates=1000, seed) ->
    BootstrapCI`; used for both cohort-level treatment-effect intervals and
    negative-control CIs (Section 11.3), and for the paired-comparison
    tests in `src/evaluation/statistics.py` -- one bootstrap engine,
    multiple callers.

This module is the "one bootstrap engine." `causal.negative_controls`
(`run_negative_control_exposure`/`run_negative_control_outcome`) currently
implements its own inline copy of this same resampling/CI pattern
(patient-level resample-with-replacement, N replicates, 2.5th/97.5th
percentile CI, skipping NaN replicate estimates) predating this shared
module; refactoring those call sites to use this function instead is a
separate, out-of-scope cleanup and is deliberately not done here.

Percentile method
-------------------

Percentiles are computed by linear interpolation between the two closest
ranks in the sorted replicate-estimate list, matching the convention
already established in `causal.weighting._percentile` and
`causal.negative_controls._percentile` (in turn matching NumPy's default
`numpy.percentile(..., method="linear")`). Reimplemented locally rather
than imported from either `causal` module: `uncertainty` is intentionally
independent of `causal` (this bootstrap engine is equally used by
`evaluation.statistics`'s paired-comparison tests, which have nothing to
do with the causal module), so importing the helper from `causal` would
introduce an odd, one-directional `uncertainty` -> `causal` dependency for
a few lines of arithmetic. See `causal.weighting` module docstring for the
percentile formula itself.

Dependency scope note
-----------------------

As elsewhere in this codebase (see `causal.propensity` module docstring),
only `pydantic`/`PyYAML` are approved runtime dependencies; this module
uses only the standard library (`random`, `math`, `dataclasses`).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from causal.solvers import SolverConvergenceError

DataT = TypeVar("DataT")


@dataclass(frozen=True)
class BootstrapCI:
    """The result of one `patient_level_bootstrap` call.

    Attributes:
        point_estimate: `estimator_fn(data)` on the full, unresampled
            input.
        ci_low: the `(1 - ci_level) / 2` percentile of the successful
            bootstrap replicate estimates (e.g. the 2.5th percentile for
            the default 95% CI).
        ci_high: the `1 - (1 - ci_level) / 2` percentile of the successful
            bootstrap replicate estimates (e.g. the 97.5th percentile for
            the default 95% CI).
        n_replicates: the number of bootstrap replicates requested
            (`n_replicates` as passed in, regardless of how many
            succeeded).
        n_successful: the number of replicates whose `estimator_fn` result
            was not NaN and was therefore included in the percentile
            computation.
    """

    point_estimate: float
    ci_low: float
    ci_high: float
    n_replicates: int
    n_successful: int


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence.

    See module docstring for the method. Returns `float("nan")` for an
    empty input (e.g. every bootstrap replicate produced NaN) rather than
    raising, so callers can surface the degenerate case explicitly instead
    of crashing.
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


def patient_level_bootstrap(
    estimator_fn: Callable[[Sequence[DataT]], float],
    data: Sequence[DataT],
    n_replicates: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> BootstrapCI:
    """Patient-level bootstrap confidence interval for `estimator_fn(data)`.

    The shared bootstrap engine (`design.md` Section 3.7): computes a
    point estimate on the full data, then resamples `data` with
    replacement (same length as `data`, i.e. a standard nonparametric
    bootstrap at the patient level -- one independence unit per element
    of `data`) `n_replicates` times, recomputing `estimator_fn` on each
    resample, and reports the empirical percentile interval of the
    resulting replicate estimates.

    Args:
        estimator_fn: `estimator_fn(data) -> float` -- any statistic
            computed over a sequence of patient-level records/values
            (e.g. a treatment-effect estimate, a negative-control
            statistic, or a paired metric difference). Called once on the
            full `data` for the point estimate, then once per replicate.
        data: the patient-level data to resample. Each element is treated
            as one independence unit (e.g. one patient record); resampling
            draws `len(data)` elements with replacement per replicate.
        n_replicates: number of bootstrap replicates. Defaults to `1000`
            per `design.md` Section 3.7.
        seed: seeds a `random.Random(seed)` instance used for all
            resampling in this call, for determinism. Defaults to `0`.
        ci_level: the confidence level, e.g. `0.95` for a 95% CI (the
            default). The CI bounds are the `(1 - ci_level) / 2` and
            `1 - (1 - ci_level) / 2` percentiles of the successful
            replicate estimates.

    Returns:
        A `BootstrapCI` with the point estimate, CI bounds, and replicate
        counts. If every replicate produces NaN, `ci_low`/`ci_high` are
        `float("nan")` rather than raising.

    Raises:
        ValueError: if `data` is empty (there is nothing to resample).
    """
    if not data:
        raise ValueError("patient_level_bootstrap requires non-empty data")

    point_estimate = estimator_fn(data)

    rng = random.Random(seed)
    n = len(data)
    replicate_estimates: list[float] = []
    for _ in range(n_replicates):
        resample = [data[rng.randrange(n)] for _ in range(n)]
        try:
            estimate = estimator_fn(resample)
        except SolverConvergenceError:
            # A replicate whose fit does not meet the frozen convergence rule
            # is dropped, exactly as a NaN replicate already is.
            continue
        if not math.isnan(estimate):
            replicate_estimates.append(estimate)
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
