"""Overlap diagnostics (Requirements 7.6, 7.7; `decisions.md` Section 12
"Overlap and support rules").

`decisions.md` Section 12 freezes the following (copied verbatim, not
re-derived):

- **Primary common-support rule:** `0.10 <= e_hat(X) <= 0.90`.
- **Sensitivity rule:** `0.05 <= e_hat(X) <= 0.95`.
- A causal estimate is **support-valid** only if ALL of:
    1. both arms retain >=100 patients;
    2. effective sample size >=100 per arm;
    3. every adjusted baseline covariate has absolute standardized mean
       difference (SMD) < 0.10;
    4. no unresolved positivity warning remains.

Interface / decoupling from `causal.propensity`
--------------------------------------------------

`OverlapDiagnostics.compute()` accepts a plain `{patient_id: propensity}`
mapping rather than a `causal.propensity.CrossFittedPredictions` instance
directly. `CrossFittedPredictions.propensity_scores` already has exactly
this shape, so a caller holding cross-fitted predictions simply passes
`cross_fitted.propensity_scores`. This keeps `overlap.py` usable with any
propensity source (cross-fitted or not) without importing
`causal.propensity`, and keeps the two modules independently testable.

Every `patient_id` in `patients` MUST have a corresponding entry in
`propensities` -- a missing entry raises `ValueError` rather than silently
skipping the patient, since a partially-scored cohort would silently bias
every summary statistic below.

Extreme-weight-count proxy
----------------------------

Full inverse-probability-weight computation (with clipping/truncation
presets) is `causal.weighting`'s job (task 7.2), not this module's. As a
simple, clearly-documented proxy, "extreme weight" here means "propensity
score outside the *primary* common-support band `[0.10, 0.90]`" -- i.e. the
same patients who fail the primary common-support rule. This is a
reasonable proxy because IPW weights (`1/e` for arm 1, `1/(1-e)` for arm 0)
diverge exactly as propensity scores approach 0 or 1, so distance from the
primary band is a direct (if unscaled) indicator of weight extremity. If a
future task needs the true post-clipping weight distribution, it should use
`causal.weighting` directly rather than this proxy.

Positivity-warning proxy
--------------------------

`no_unresolved_positivity_warning` (the fourth frozen condition) is
resolved here as `extreme_weight_count == 0` -- i.e. no patient falls
outside the primary common-support band. This reuses the same proxy above
rather than introducing a second, differently-scoped notion of "positivity
warning"; documented here so the choice is auditable.

Effective sample size (ESS)
------------------------------

Per arm, using the standard formula `ESS = (sum(w))^2 / sum(w^2)` with
inverse-propensity weights `w_i = 1/e_i` for arm-1 patients and
`w_i = 1/(1-e_i)` for arm-0 patients (unstabilized, unclipped -- this
module's job is diagnostics, not the primary estimator's weighting scheme).

Standardized mean difference (SMD)
-------------------------------------

Per covariate in `adjustment_set`, the pooled-SD SMD:

    SMD = (mean_1 - mean_0) / sqrt((var_1 + var_0) / 2)

Raw covariate values are read from `PatientRecord.timeline.static_covariates`
and coerced to `float`; a missing or non-numeric value for a given patient
is excluded from that covariate's mean/variance computation (never
defaulted to `0.0`, which would silently distort the balance check). Sample
variance (`statistics.variance`, i.e. `ddof=1`) is used when an arm has at
least two observed values for a covariate; an arm with fewer than two
observed values contributes a variance of `0.0` for that covariate. If both
arms have zero variance and equal means, SMD is defined as `0.0` (no
imbalance to report); if the pooled SD is `0.0` but the means differ, SMD is
reported as `float("inf")` (an unambiguous imbalance signal rather than a
`ZeroDivisionError`).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from cohorts.cohort_construction import PatientRecord

#: `decisions.md` Section 12 -- primary common-support band.
PRIMARY_SUPPORT_LOWER: float = 0.10
PRIMARY_SUPPORT_UPPER: float = 0.90

#: `decisions.md` Section 12 -- sensitivity common-support band.
SENSITIVITY_SUPPORT_LOWER: float = 0.05
SENSITIVITY_SUPPORT_UPPER: float = 0.95

#: `decisions.md` Section 12 -- minimum retained patients per arm.
MIN_ARM_COUNT: int = 100

#: `decisions.md` Section 12 -- minimum effective sample size per arm.
MIN_EFFECTIVE_SAMPLE_SIZE: float = 100.0

#: `decisions.md` Section 12 -- maximum absolute SMD for every adjusted covariate.
MAX_ABS_SMD: float = 0.10

#: The two retained treatment arms (mirrors `cohort_construction.RETAINED_ARMS`).
_ARM_1_INTERVENTION = 1
_ARM_0_COMPARATOR = 0


@dataclass(frozen=True)
class ArmPropensitySummary:
    """Simple summary statistics of one arm's propensity distribution."""

    n: int
    minimum: float
    maximum: float
    mean: float
    median: float


@dataclass(frozen=True)
class CommonSupportSummary:
    """Cohort-wide common-support counts/fractions under both frozen bands."""

    primary_count: int
    primary_fraction: float
    sensitivity_count: int
    sensitivity_fraction: float


@dataclass(frozen=True)
class OverlapReport:
    """The full result of `OverlapDiagnostics.compute()`.

    Attributes:
        arm_counts: `{arm: retained patient count}`.
        arm_propensity_summary: `{arm: ArmPropensitySummary}`.
        common_support: cohort-wide common-support counts/fractions.
        extreme_weight_count: count of patients outside the primary
            common-support band (see module docstring, "extreme-weight
            count proxy").
        extreme_weight_fraction: `extreme_weight_count / len(patients)`
            (`0.0` if `patients` is empty).
        ess_by_arm: `{arm: effective sample size}`.
        smd_by_covariate: `{covariate name: signed SMD}`, one entry per
            `adjustment_set` field that has at least one observed value in
            each arm; a field observed in neither/only one arm is omitted
            (there is no balance to compute).
        arm_counts_sufficient: both arms retain >= `MIN_ARM_COUNT` patients.
        ess_sufficient: both arms' ESS >= `MIN_EFFECTIVE_SAMPLE_SIZE`.
        smd_sufficient: every computed SMD has `abs(smd) < MAX_ABS_SMD`.
        no_unresolved_positivity_warning: `extreme_weight_count == 0`.
        support_valid: all four conditions above.
    """

    arm_counts: dict[int, int]
    arm_propensity_summary: dict[int, ArmPropensitySummary]
    common_support: CommonSupportSummary
    extreme_weight_count: int
    extreme_weight_fraction: float
    ess_by_arm: dict[int, float]
    smd_by_covariate: dict[str, float]
    arm_counts_sufficient: bool
    ess_sufficient: bool
    smd_sufficient: bool
    no_unresolved_positivity_warning: bool
    support_valid: bool = field(default=False)


def _propensity_for(patient: PatientRecord, propensities: Mapping[str, float]) -> float:
    if patient.patient_id not in propensities:
        raise ValueError(
            f"Missing propensity score for patient_id={patient.patient_id!r}; "
            "OverlapDiagnostics.compute() requires a propensity for every "
            "patient in `patients`."
        )
    return propensities[patient.patient_id]


def _summarize_arm(propensity_values: Sequence[float]) -> ArmPropensitySummary:
    if not propensity_values:
        return ArmPropensitySummary(n=0, minimum=0.0, maximum=0.0, mean=0.0, median=0.0)
    return ArmPropensitySummary(
        n=len(propensity_values),
        minimum=min(propensity_values),
        maximum=max(propensity_values),
        mean=statistics.fmean(propensity_values),
        median=statistics.median(propensity_values),
    )


def _effective_sample_size(weights: Sequence[float]) -> float:
    """`ESS = (sum(w))^2 / sum(w^2)`; `0.0` for an empty arm."""
    if not weights:
        return 0.0
    sum_w = sum(weights)
    sum_w2 = sum(w * w for w in weights)
    if sum_w2 == 0.0:
        return 0.0
    return (sum_w * sum_w) / sum_w2


def _raw_numeric_covariate(patient: PatientRecord, field_name: str) -> float | None:
    raw_value = patient.timeline.static_covariates.get(field_name)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _smd(values_1: Sequence[float], values_0: Sequence[float]) -> float | None:
    """Pooled-SD SMD for one covariate, or `None` if either arm has no observed value."""
    if not values_1 or not values_0:
        return None

    mean_1 = statistics.fmean(values_1)
    mean_0 = statistics.fmean(values_0)
    var_1 = statistics.variance(values_1) if len(values_1) > 1 else 0.0
    var_0 = statistics.variance(values_0) if len(values_0) > 1 else 0.0

    pooled_sd = math.sqrt((var_1 + var_0) / 2.0)
    if pooled_sd == 0.0:
        return 0.0 if mean_1 == mean_0 else float("inf")
    return (mean_1 - mean_0) / pooled_sd


class OverlapDiagnostics:
    """Overlap and support diagnostics for a cohort's propensity scores.

    Stateless -- `compute()` is a `staticmethod`-style entry point kept as
    a class (rather than a bare module function) to mirror the
    `SomeThing.compute(...)` / `SomeThing.evaluate(...)` naming convention
    already used by `CompositeEndpoint.compute()` and other Stage-3/7
    modules in this codebase.
    """

    @staticmethod
    def compute(
        patients: Sequence[PatientRecord],
        propensities: Mapping[str, float],
        adjustment_set: Sequence[str],
    ) -> OverlapReport:
        """Compute the full `OverlapReport` for `patients`.

        Args:
            patients: the cohort (or cohort subset, e.g. one split) to
                diagnose. Only `treatment_arm` (`0` or `1`) and
                `timeline.static_covariates` are read.
            propensities: `{patient_id: propensity}`, e.g.
                `cross_fitted_predictions.propensity_scores` from
                `causal.propensity.CrossFittedPredictions`. Every patient
                in `patients` must have an entry (see module docstring).
            adjustment_set: covariate names to balance-check, read from
                `PatientRecord.timeline.static_covariates`. Typically the
                frozen baseline adjustment set (`decisions.md` Section 13).

        Returns:
            An `OverlapReport` with arm-specific propensity distributions,
            common support, the extreme-weight-count proxy, per-arm ESS,
            per-covariate SMD, the four frozen boolean conditions, and
            `support_valid`.
        """
        arm_counts: dict[int, int] = {}
        propensity_by_arm: dict[int, list[float]] = {}
        weights_by_arm: dict[int, list[float]] = {}
        covariate_values_by_arm: dict[int, dict[str, list[float]]] = {
            _ARM_0_COMPARATOR: {name: [] for name in adjustment_set},
            _ARM_1_INTERVENTION: {name: [] for name in adjustment_set},
        }

        primary_count = 0
        sensitivity_count = 0
        extreme_weight_count = 0

        for patient in patients:
            arm = patient.treatment_arm
            propensity = _propensity_for(patient, propensities)

            arm_counts[arm] = arm_counts.get(arm, 0) + 1
            propensity_by_arm.setdefault(arm, []).append(propensity)

            if PRIMARY_SUPPORT_LOWER <= propensity <= PRIMARY_SUPPORT_UPPER:
                primary_count += 1
            else:
                extreme_weight_count += 1
            if SENSITIVITY_SUPPORT_LOWER <= propensity <= SENSITIVITY_SUPPORT_UPPER:
                sensitivity_count += 1

            # Inverse-propensity weight, per arm (unstabilized, unclipped --
            # diagnostics only; see module docstring).
            if arm == _ARM_1_INTERVENTION:
                weight = 1.0 / propensity if propensity != 0.0 else math.inf
            elif arm == _ARM_0_COMPARATOR:
                weight = 1.0 / (1.0 - propensity) if propensity != 1.0 else math.inf
            else:
                weight = None
            if weight is not None:
                weights_by_arm.setdefault(arm, []).append(weight)

            if arm in covariate_values_by_arm:
                for field_name in adjustment_set:
                    value = _raw_numeric_covariate(patient, field_name)
                    if value is not None:
                        covariate_values_by_arm[arm][field_name].append(value)

        n_patients = len(patients)
        arm_propensity_summary = {
            arm: _summarize_arm(values) for arm, values in propensity_by_arm.items()
        }
        common_support = CommonSupportSummary(
            primary_count=primary_count,
            primary_fraction=(primary_count / n_patients) if n_patients else 0.0,
            sensitivity_count=sensitivity_count,
            sensitivity_fraction=(sensitivity_count / n_patients) if n_patients else 0.0,
        )
        ess_by_arm = {
            arm: _effective_sample_size(weights) for arm, weights in weights_by_arm.items()
        }

        smd_by_covariate: dict[str, float] = {}
        for field_name in adjustment_set:
            values_1 = covariate_values_by_arm[_ARM_1_INTERVENTION][field_name]
            values_0 = covariate_values_by_arm[_ARM_0_COMPARATOR][field_name]
            smd = _smd(values_1, values_0)
            if smd is not None:
                smd_by_covariate[field_name] = smd

        arm_counts_sufficient = (
            arm_counts.get(_ARM_0_COMPARATOR, 0) >= MIN_ARM_COUNT
            and arm_counts.get(_ARM_1_INTERVENTION, 0) >= MIN_ARM_COUNT
        )
        ess_sufficient = (
            ess_by_arm.get(_ARM_0_COMPARATOR, 0.0) >= MIN_EFFECTIVE_SAMPLE_SIZE
            and ess_by_arm.get(_ARM_1_INTERVENTION, 0.0) >= MIN_EFFECTIVE_SAMPLE_SIZE
        )
        smd_sufficient = all(abs(smd) < MAX_ABS_SMD for smd in smd_by_covariate.values())
        no_unresolved_positivity_warning = extreme_weight_count == 0

        support_valid = (
            arm_counts_sufficient
            and ess_sufficient
            and smd_sufficient
            and no_unresolved_positivity_warning
        )

        return OverlapReport(
            arm_counts=arm_counts,
            arm_propensity_summary=arm_propensity_summary,
            common_support=common_support,
            extreme_weight_count=extreme_weight_count,
            extreme_weight_fraction=(extreme_weight_count / n_patients) if n_patients else 0.0,
            ess_by_arm=ess_by_arm,
            smd_by_covariate=smd_by_covariate,
            arm_counts_sufficient=arm_counts_sufficient,
            ess_sufficient=ess_sufficient,
            smd_sufficient=smd_sufficient,
            no_unresolved_positivity_warning=no_unresolved_positivity_warning,
            support_valid=support_valid,
        )
