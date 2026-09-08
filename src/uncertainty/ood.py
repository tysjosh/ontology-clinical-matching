"""Out-of-distribution (OOD) scoring (Requirement 10.8).

Requirement 10.8 states: "Out-of-distribution support SHALL prioritize
propensity extremity and one prespecified representation or missingness
diagnostic; implementing multiple redundant scores is optional." This
module implements exactly the two prespecified scores -- no more:

- `propensity_extremity_score` -- **primary**.
- `missingness_deviation_score` -- **secondary**.

Propensity-extremity score (primary)
-------------------------------------

`causal.overlap` freezes the primary common-support band as
`PRIMARY_SUPPORT_LOWER = 0.10` / `PRIMARY_SUPPORT_UPPER = 0.90`
(`decisions.md` Section 12). This module reuses those exact constants
rather than re-hardcoding the thresholds.

`propensity_extremity_score(propensity)` is defined as the signed distance
*outside* the primary common-support band:

    score = max(0.0, PRIMARY_SUPPORT_LOWER - propensity, propensity - PRIMARY_SUPPORT_UPPER)

This is `0.0` for any propensity inside `[0.10, 0.90]` (i.e. "in common
support, not OOD by this score") and increases monotonically the further
`propensity` falls outside that band in either direction. A propensity of
`0.98` therefore scores `0.98 - 0.90 = 0.08`, and a propensity of `0.50`
scores `0.0`. This is the "simple, monotonic score measuring how far
`propensity` is from the safe middle of the common-support band" called
for by the task -- deliberately zero-valued for the whole in-support
range rather than just the exact center, since "in common support" is
itself the in-distribution condition the rest of the pipeline (task 7.1's
`OverlapDiagnostics`) already uses.

Missingness-deviation score (secondary)
-----------------------------------------

`missingness_deviation_score(patient_missingness_rate, reference_missingness_rate)`
is the plain absolute deviation:

    score = abs(patient_missingness_rate - reference_missingness_rate)

where `patient_missingness_rate` is the fraction of a patient's monitored
baseline fields that are missing, and `reference_missingness_rate` is a
caller-supplied reference rate (e.g. the development cohort's average
missingness rate, computed by the caller once and passed in for every
patient). This module does not compute the reference rate itself -- the
caller decides what "reference" means for a given run (per
`decisions.md`/task framing, typically the development-cohort average).

`patient_missingness_rate_from_indicators` is a small convenience helper
that computes `patient_missingness_rate` from a per-patient indicator
dict shaped like one value of
`data.imputation.missingness_indicators_from_timelines`'s return value
(`{field_name: 1 or 0}`), so callers already holding that shape do not
need to re-derive the fraction by hand. It is a thin mean over the `0`/`1`
indicator values; the deviation formula above remains the core score.
"""

from __future__ import annotations

from collections.abc import Mapping

from causal.overlap import PRIMARY_SUPPORT_LOWER, PRIMARY_SUPPORT_UPPER


def propensity_extremity_score(propensity: float) -> float:
    """Primary OOD score: distance of `propensity` outside the primary
    common-support band `[PRIMARY_SUPPORT_LOWER, PRIMARY_SUPPORT_UPPER]`.

    Returns `0.0` for any `propensity` inside the band (inclusive), and an
    increasingly positive value the further `propensity` falls outside it
    in either direction.
    """
    return max(
        0.0,
        PRIMARY_SUPPORT_LOWER - propensity,
        propensity - PRIMARY_SUPPORT_UPPER,
    )


def missingness_deviation_score(
    patient_missingness_rate: float, reference_missingness_rate: float
) -> float:
    """Secondary OOD score: absolute deviation of a patient's missingness
    rate from a caller-supplied reference missingness rate.

    Args:
        patient_missingness_rate: fraction (in `[0.0, 1.0]`) of the
            patient's monitored baseline fields that are missing.
        reference_missingness_rate: the reference rate to compare against,
            e.g. the development cohort's average missingness rate.

    Returns:
        `abs(patient_missingness_rate - reference_missingness_rate)`.
    """
    return abs(patient_missingness_rate - reference_missingness_rate)


def patient_missingness_rate_from_indicators(indicators: Mapping[str, int]) -> float:
    """Convenience helper: fraction of missing fields in a per-patient
    indicator dict shaped like one value of
    `data.imputation.missingness_indicators_from_timelines`'s return value
    (`{field_name: 1 or 0}`, `1` meaning missing).

    Returns `0.0` for an empty `indicators` mapping (no monitored fields to
    be missing).
    """
    if not indicators:
        return 0.0
    return sum(indicators.values()) / len(indicators)
