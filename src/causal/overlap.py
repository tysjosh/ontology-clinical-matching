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

Standardized mean difference (SMD): what the third condition gates on
-----------------------------------------------------------------------

`decisions.md` Section 26 (2026-09-30 amendment; task 7.6) fixes how Section
12's "every **adjusted** baseline covariate" is computed:

- **Weighted basis.** `smd_by_covariate`, the only SMD `smd_sufficient`
  reads, is computed under **stabilized inverse-probability weights**:
  `P(A=1)/e` for arm 1 and `P(A=0)/(1-e)` for arm 0, with the marginal
  prevalences taken from `patients`, clipped with the frozen Section 11
  `"primary"` 1st/99th-percentile preset through
  `causal.weighting.clip_weights`. Weighted means are `sum(w x) / sum(w)`;
  weighted variances are the reliability-weighted analogue of `ddof=1`
  (denominator `sum(w) - sum(w^2)/sum(w)`), so with all weights equal to 1
  they reduce exactly to the raw computation. Pooled-SD form:
  `SMD = (mean_1 - mean_0) / sqrt((var_1 + var_0) / 2)`.
- **Full adjustment set, one shared encoding.** Each `adjustment_set` name
  that is a Section 13 covariate is read through `causal.adjustment_set`
  (stored-key resolution such as `weight` -> `weight_kg`) and contributes
  one entry per encoded column: continuous values on their raw scale,
  booleans as 1/0, binary categories as 1 for the non-reference level and
  0 for the reference level, prior ART as an exposure indicator plus a
  duration. Names outside Section 13 are read as raw numerics. A missing
  or unusable value is excluded from that column's moments, never
  defaulted to `0.0`.
- **No silent omission.** A column with no observed value in one arm cannot
  be balance-checked. It gets an entry in `smd_exclusions` naming the
  reason, and it **counts as not balanced**: `smd_sufficient` requires
  `smd_exclusions` to be empty.
- **Raw SMDs are descriptive only.** `raw_smd_by_covariate` holds the same
  columns' unweighted SMDs. Section 9 deliberately induces raw imbalance,
  so this field is reported, labelled raw, and never gated on.

Degenerate cases (both bases): an arm with fewer than two observed values
contributes variance `0.0`; zero pooled SD with equal means gives SMD `0.0`;
zero pooled SD with different means gives `float("inf")`, which fails the
gate.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from causal.adjustment_set import (
    ADJUSTMENT_COVARIATES_BY_NAME,
    AdjustmentCovariate,
    resolve_raw_value,
)
from causal.weighting import clip_weights
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

#: `decisions.md` Section 26 (2026-09-30): the clipping preset for the
#: weighted-SMD basis (Section 11's frozen `"primary"` 1st/99th percentile).
SMD_WEIGHT_CLIP_PRESET: str = "primary"

#: Recorded in `OverlapReport.smd_basis`.
SMD_BASIS: str = "stabilized_ipw_clipped_primary_1_99"

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
        smd_by_covariate: `{encoded column: signed weighted SMD}` on the
            stabilized-IPW basis (module docstring). The quantity the third
            frozen condition gates on.
        arm_counts_sufficient: both arms retain >= `MIN_ARM_COUNT` patients.
        ess_sufficient: both arms' ESS >= `MIN_EFFECTIVE_SAMPLE_SIZE`.
        smd_sufficient: `smd_exclusions` is empty and every weighted SMD
            has `abs(smd) < MAX_ABS_SMD`.
        no_unresolved_positivity_warning: `extreme_weight_count == 0`.
        support_valid: all four conditions above.
        raw_smd_by_covariate: the same columns' unweighted SMDs. Descriptive
            only; never read by `smd_sufficient` or `support_valid`.
        smd_exclusions: `{encoded column: reason}` for every column that
            could not be balance-checked. Each one makes `smd_sufficient`
            false.
        smd_basis: the weighting basis of `smd_by_covariate`.
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
    raw_smd_by_covariate: dict[str, float] = field(default_factory=dict)
    smd_exclusions: dict[str, str] = field(default_factory=dict)
    smd_basis: str = SMD_BASIS


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


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    return sum(v * w for v, w in zip(values, weights, strict=True)) / sum(weights)


def _weighted_variance(values: Sequence[float], weights: Sequence[float]) -> float:
    """Reliability-weighted variance; equals `statistics.variance` at unit weights."""
    if len(values) < 2:
        return 0.0
    mean = _weighted_mean(values, weights)
    sum_w = sum(weights)
    denominator = sum_w - sum(w * w for w in weights) / sum_w
    if denominator <= 0.0:
        return 0.0
    return sum(w * (v - mean) ** 2 for v, w in zip(values, weights, strict=True)) / denominator


def _weighted_smd(
    values_1: Sequence[float],
    weights_1: Sequence[float],
    values_0: Sequence[float],
    weights_0: Sequence[float],
) -> float:
    """Pooled-SD SMD under weights, with the same degenerate cases as `_smd`.

    Values are first shifted by a common constant (the first observed
    value). The SMD is shift-invariant, and the shift makes a constant
    column exactly zero, so it cannot produce a spurious SMD from rounding
    in `sum(w x) / sum(w)` (variance ~1e-26 against a mean difference of
    one ulp).
    """
    shift = values_1[0]
    values_1 = [v - shift for v in values_1]
    values_0 = [v - shift for v in values_0]
    mean_1 = _weighted_mean(values_1, weights_1)
    mean_0 = _weighted_mean(values_0, weights_0)
    pooled_sd = math.sqrt(
        (_weighted_variance(values_1, weights_1) + _weighted_variance(values_0, weights_0)) / 2.0
    )
    if pooled_sd == 0.0:
        return 0.0 if mean_1 == mean_0 else float("inf")
    return (mean_1 - mean_0) / pooled_sd


def _stabilized_clipped_weights(
    patients: Sequence[PatientRecord], propensities: Mapping[str, float]
) -> dict[str, float]:
    """Stabilized IPW weights for arms 0/1, clipped with the frozen primary preset."""
    in_arms = [p for p in patients if p.treatment_arm in (_ARM_0_COMPARATOR, _ARM_1_INTERVENTION)]
    if not in_arms:
        return {}
    p_treated = sum(1 for p in in_arms if p.treatment_arm == _ARM_1_INTERVENTION) / len(in_arms)
    unclipped: dict[str, float] = {}
    for patient in in_arms:
        e = _propensity_for(patient, propensities)
        if patient.treatment_arm == _ARM_1_INTERVENTION:
            unclipped[patient.patient_id] = p_treated / e if e > 0.0 else math.inf
        else:
            unclipped[patient.patient_id] = (1.0 - p_treated) / (1.0 - e) if e < 1.0 else math.inf
    clipped = clip_weights(unclipped, SMD_WEIGHT_CLIP_PRESET)
    assert isinstance(clipped, dict)
    return clipped


def _to_float(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _section_13_column_values(
    patient: PatientRecord, covariate: AdjustmentCovariate
) -> list[float | None]:
    """One value per encoded column of `covariate`, `None` where missing.

    Raw scale for continuous columns. Missing values stay `None` (unlike the
    propensity encoding, which fills them), so they are excluded from the
    balance moments rather than counted as zero.
    """
    raw = resolve_raw_value(patient.timeline.static_covariates, covariate)
    if covariate.encoding == "continuous":
        return [_to_float(raw)]
    if covariate.encoding == "boolean":
        return [None if raw is None else (1.0 if bool(raw) else 0.0)]
    if covariate.encoding == "binary_category":
        if not isinstance(raw, str):
            return [None]
        if covariate.allowed_levels is not None and raw not in covariate.allowed_levels:
            return [None]
        return [0.0 if raw == covariate.reference_level else 1.0]
    duration = None
    for concept_id in covariate.concept_ids:
        duration = _to_float(patient.timeline.static_covariates.get(concept_id))
        if duration is not None:
            break
    if duration is None:
        return [None, None]
    return [1.0 if duration > 0 else 0.0, duration]


def _balance_columns(patient: PatientRecord, name: str) -> list[tuple[str, float | None]]:
    """`[(column, value or None)]` for one `adjustment_set` name."""
    covariate = ADJUSTMENT_COVARIATES_BY_NAME.get(name)
    if covariate is None:
        return [(name, _raw_numeric_covariate(patient, name))]
    return list(zip(covariate.columns, _section_13_column_values(patient, covariate), strict=True))


def _column_order(adjustment_set: Sequence[str]) -> list[str]:
    order: list[str] = []
    for name in adjustment_set:
        covariate = ADJUSTMENT_COVARIATES_BY_NAME.get(name)
        for column in covariate.columns if covariate is not None else (name,):
            if column not in order:
                order.append(column)
    return order


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
        columns = _column_order(adjustment_set)
        smd_weights = _stabilized_clipped_weights(patients, propensities)
        # {arm: {column: [(value, weight)]}}
        covariate_values_by_arm: dict[int, dict[str, list[tuple[float, float]]]] = {
            _ARM_0_COMPARATOR: {column: [] for column in columns},
            _ARM_1_INTERVENTION: {column: [] for column in columns},
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
                weight_for_smd = smd_weights[patient.patient_id]
                for name in adjustment_set:
                    for column, value in _balance_columns(patient, name):
                        if value is not None:
                            covariate_values_by_arm[arm][column].append((value, weight_for_smd))

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
        raw_smd_by_covariate: dict[str, float] = {}
        smd_exclusions: dict[str, str] = {}
        for column in columns:
            pairs_1 = covariate_values_by_arm[_ARM_1_INTERVENTION][column]
            pairs_0 = covariate_values_by_arm[_ARM_0_COMPARATOR][column]
            if not pairs_1 or not pairs_0:
                empty = [
                    str(arm)
                    for arm, pairs in ((_ARM_1_INTERVENTION, pairs_1), (_ARM_0_COMPARATOR, pairs_0))
                    if not pairs
                ]
                smd_exclusions[column] = (
                    f"no observed value in arm {' and arm '.join(empty)}; balance cannot be "
                    "checked, so the covariate is not counted as balanced"
                )
                continue
            values_1 = [v for v, _ in pairs_1]
            values_0 = [v for v, _ in pairs_0]
            raw = _smd(values_1, values_0)
            assert raw is not None
            raw_smd_by_covariate[column] = raw
            smd_by_covariate[column] = _weighted_smd(
                values_1, [w for _, w in pairs_1], values_0, [w for _, w in pairs_0]
            )

        arm_counts_sufficient = (
            arm_counts.get(_ARM_0_COMPARATOR, 0) >= MIN_ARM_COUNT
            and arm_counts.get(_ARM_1_INTERVENTION, 0) >= MIN_ARM_COUNT
        )
        ess_sufficient = (
            ess_by_arm.get(_ARM_0_COMPARATOR, 0.0) >= MIN_EFFECTIVE_SAMPLE_SIZE
            and ess_by_arm.get(_ARM_1_INTERVENTION, 0.0) >= MIN_EFFECTIVE_SAMPLE_SIZE
        )
        smd_sufficient = not smd_exclusions and all(
            abs(smd) < MAX_ABS_SMD for smd in smd_by_covariate.values()
        )
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
            raw_smd_by_covariate=raw_smd_by_covariate,
            smd_exclusions=smd_exclusions,
            smd_basis=SMD_BASIS,
        )


# ---------------------------------------------------------------------------
# Common-support trimming (`decisions.md` Section 26 amendment of
# 2026-10-12, item B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommonSupportTrim:
    """The result of `trim_to_common_support`.

    Attributes:
        retained: the patients inside the band, in input order -- the
            estimation population every causal estimate, diagnostic, and
            witness then runs on.
        trimmed_patient_ids: the patients removed, sorted.
        trimmed_count_by_arm: `{arm: count}` removed, for both arms (zero
            included), so the report always states what trimming did.
        band: the `(lower, upper)` band applied.
    """

    retained: tuple[PatientRecord, ...]
    trimmed_patient_ids: tuple[str, ...]
    trimmed_count_by_arm: dict[int, int]
    band: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "band": list(self.band),
            "retained_count": len(self.retained),
            "trimmed_patient_ids": list(self.trimmed_patient_ids),
            "trimmed_count_by_arm": {str(arm): n for arm, n in self.trimmed_count_by_arm.items()},
        }


def trim_to_common_support(
    patients: Sequence[PatientRecord],
    propensities: Mapping[str, float],
    lower: float = PRIMARY_SUPPORT_LOWER,
    upper: float = PRIMARY_SUPPORT_UPPER,
) -> CommonSupportTrim:
    """Remove patients whose propensity lies outside `[lower, upper]`.

    The frozen trimming rule (`decisions.md` Section 26 amendment of
    2026-10-12, item B): before the support check, every patient outside
    the Section 12 primary common-support band `[0.10, 0.90]` of the
    primary (logistic) propensity model is removed from the estimation
    population. Applied the same way to every analysis population -- every
    split, every seed, every sensitivity variant that refits the primary
    propensity model. A positivity warning resolved this way counts as
    resolved under Section 12's fourth condition, and the count removed
    per arm is always reported.

    This changes the estimand to the patients with overlap. Propensities
    are not refit after trimming: the band is defined by the fit that
    produced it.
    """
    retained: list[PatientRecord] = []
    trimmed: list[str] = []
    trimmed_by_arm: dict[int, int] = {_ARM_0_COMPARATOR: 0, _ARM_1_INTERVENTION: 0}
    for patient in patients:
        propensity = _propensity_for(patient, propensities)
        if lower <= propensity <= upper:
            retained.append(patient)
        else:
            trimmed.append(patient.patient_id)
            trimmed_by_arm[patient.treatment_arm] = trimmed_by_arm.get(patient.treatment_arm, 0) + 1
    return CommonSupportTrim(
        retained=tuple(retained),
        trimmed_patient_ids=tuple(sorted(trimmed)),
        trimmed_count_by_arm=trimmed_by_arm,
        band=(lower, upper),
    )
