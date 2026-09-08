"""Negative controls (Requirement 7.12; Correctness Property 4;
`decisions.md` Section 11.3).

`decisions.md` Section 11.3 freezes two negative-control checks:

1. **Negative-control exposure** — a synthetic sham treatment label,
   generated via `HMAC-SHA-256(patient_id, negative_control_seed)`
   thresholded to reproduce the observed real-treatment prevalence,
   generated without reference to any covariate, outcome, censoring
   indicator, or real treatment assignment. The primary causal estimator
   is rerun with the sham label substituted for the real treatment; since
   the sham label cannot have a true causal effect, a detected effect
   indicates estimator instability or a pipeline defect.
2. **Negative-control outcome** — pre-treatment baseline CD8 used as a
   pseudo-outcome (treatment cannot causally affect a value measured
   before assignment), with baseline CD8 excluded from its own
   adjustment set (the CD8 double-use safeguard,
   `data.leakage_audit.assert_no_self_adjustment`) so the check is not
   tautological.

DELIBERATE, DOCUMENTED ADAPTATION — read before changing thresholds
---------------------------------------------------------------------

`decisions.md` Section 11.3 states the negative-control-*exposure*
pass/fail/uncertain rule on a **hazard-ratio scale**: pass if the 95% CI
of the estimated hazard ratio contains 1.0 *and* the point estimate falls
within `[0.90, 1.10]`; fail if the CI excludes 1.0; uncertain if the CI
contains 1.0 but the point estimate falls outside `[0.90, 1.10]`.

This codebase's primary causal baseline, `causal.baselines.ipcw_aipw`, does
**not** produce a hazard ratio — it produces a **risk difference**
(`scale="risk_difference"`), and it never computes a standard error/CI
(`se=None` always, per that module's documented limitation). There is
currently no hazard-ratio-scale primary estimator with a CI anywhere in
this codebase to rerun with the sham label.

Rather than block this task on reconciling that mismatch (a design
decision out of scope here), `run_negative_control_exposure` below applies
the frozen HR-scale *rule* — pragmatically adapted — to the
risk-difference-scale estimator that actually exists:

- the null hypothesis "no effect" is `hazard_ratio == 1.0` on the HR
  scale; its direct analogue on the risk-difference scale is
  `risk_difference == 0.0`, used as the null value here;
- the `[0.90, 1.10]` HR-scale tolerance band around the null is likewise
  translated to an analogous `[-0.10, 0.10]` risk-difference-scale
  tolerance band around `0.0` (`NULL_TOLERANCE_BAND` below);
- since `ipcw_aipw` has no analytic CI, a **simple nonparametric bootstrap**
  (patient-level resampling with replacement, `n_bootstrap` refits, 2.5th/
  97.5th empirical percentiles) is used in place of the HR-scale CI.

This is a documented substitution of scale and CI method, not a
reinterpretation of the underlying falsification logic: the check still
asks "does re-running the primary estimator with a label that cannot have
a true effect produce an estimate distinguishable from the null,
sign-appropriate tolerance band?" — exactly what the frozen rule is for.
A future reviewer adding a hazard-ratio-scale primary estimator with a
real CI should retarget this function at that estimator and drop this
adaptation note.

The negative-control-*outcome* rule (mean-difference scale, `decisions.md`
Section 11.3) already matches this codebase's estimators directly (no
adaptation needed): pass if the adjusted standardized mean difference is
below 0.10 in absolute value *and* the bootstrap CI for the adjusted mean
difference contains zero; fail if the CI excludes zero; uncertain if the
CI contains zero but the absolute standardized mean difference is >= 0.10.

Dependency scope note
-----------------------

As elsewhere in `causal`/`models` (see `causal.propensity`'s module
docstring), only `pydantic`/`PyYAML` are approved runtime dependencies.
The linear regression fit for the negative-control-outcome check is a
minimal, from-scratch batch-gradient-descent implementation, following the
same pattern as `causal.baselines._fit_logistic` and
`causal.propensity.LogisticPropensityModel`.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from causal.baselines import HORIZON_WEEKS, ipcw_aipw
from cohorts.cohort_construction import PatientRecord
from data.leakage_audit import assert_no_self_adjustment
from infra.seeding import derive_seed

#: The `concept_id` under which baseline CD8 is recorded in
#: `PatientTimeline.static_covariates` (matches
#: `data.loaders.actg175.BaselineCovariateRecord.baseline_cd8`).
BASELINE_CD8_FIELD = "baseline_cd8"

#: Risk-difference-scale analogue of the frozen `[0.90, 1.10]` hazard-ratio
#: tolerance band (see module docstring, "DELIBERATE, DOCUMENTED
#: ADAPTATION"). A risk difference within this band of the null (`0.0`) is
#: treated as consistent with "no detectable effect."
NULL_TOLERANCE_BAND: tuple[float, float] = (-0.10, 0.10)

#: `decisions.md` Section 11.3 — frozen standardized-mean-difference
#: threshold for the negative-control-outcome check.
SMD_THRESHOLD: float = 0.10

#: Gradient-descent configuration for the negative-control-outcome linear
#: regression (mirrors `causal.baselines`/`causal.propensity`'s fixed,
#: documented training defaults — not a `decisions.md`-frozen parameter).
_LR_LEARNING_RATE: float = 0.1
_LR_N_ITERATIONS: int = 300


# ---------------------------------------------------------------------------
# Sham-label generation
# ---------------------------------------------------------------------------


class SyntheticShamExposure:
    """Deterministic sham-treatment-label generator (`decisions.md` Section 11.3).

    `generate()` depends ONLY on each `patient_id` and the frozen
    `negative_control_seed` (via `HMAC-SHA-256(key=seed, msg=patient_id)`)
    — never on any covariate, outcome, censoring indicator, or the real
    treatment assignment. `target_prevalence` is a plain threshold
    parameter (not clinical data derived from the sham label itself); the
    caller supplies it from the *real* cohort's observed treatment
    prevalence so the sham arm sizes resemble the real study's, but the
    per-patient sham *label* is computed purely from `(patient_id, seed)`.
    """

    #: Number of leading hex digits of the HMAC digest used to derive the
    #: per-patient uniform value in `[0, 1)`. 8 hex digits (32 bits) is far
    #: more resolution than needed to threshold against a prevalence, while
    #: keeping the conversion simple and exact (`16**8` is exactly
    #: representable as a Python int/float divisor).
    _HEX_DIGITS: int = 8

    @staticmethod
    def generate(
        patient_ids: Sequence[str], seed: int, target_prevalence: float = 0.5
    ) -> dict[str, int]:
        """Compute a deterministic sham label (`0`/`1`) for each `patient_id`.

        Args:
            patient_ids: the patients to generate sham labels for.
            seed: the frozen negative-control seed. Used as the HMAC key
                (`str(seed)`, UTF-8 encoded) — the same `(patient_id, seed)`
                pair always produces the same label, on any machine, in any
                process.
            target_prevalence: the fraction of patients that should receive
                a sham label of `1`, e.g. `sum(arm == 1 for ...) / n` from
                the real cohort. Must be in `[0.0, 1.0]`.

        Returns:
            `{patient_id: 0 or 1}`, covering every `patient_id` in
            `patient_ids` exactly once.
        """
        if not 0.0 <= target_prevalence <= 1.0:
            raise ValueError(
                f"target_prevalence must be in [0.0, 1.0], got {target_prevalence}"
            )

        key = str(seed).encode("utf-8")
        denominator = 16**SyntheticShamExposure._HEX_DIGITS

        labels: dict[str, int] = {}
        for patient_id in patient_ids:
            digest = hmac.new(key, patient_id.encode("utf-8"), hashlib.sha256).hexdigest()
            uniform_value = int(digest[: SyntheticShamExposure._HEX_DIGITS], 16) / denominator
            labels[patient_id] = 1 if uniform_value < target_prevalence else 0
        return labels


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NegativeControlResult:
    """The result of one negative-control check.

    Attributes:
        name: `"negative_control_exposure"` or `"negative_control_outcome"`.
        classification: `"pass"`, `"fail"`, or `"uncertain"` per the frozen
            (exposure: adapted, see module docstring; outcome: verbatim)
            `decisions.md` Section 11.3 rule.
        point_estimate: the point estimate on the scale documented for
            each check (risk difference for exposure; adjusted
            standardized mean difference for outcome).
        ci_low: the 2.5th-percentile bootstrap CI bound.
        ci_high: the 97.5th-percentile bootstrap CI bound.
        evidence: structured, rule-specific supporting detail (bootstrap
            replicate counts, thresholds applied, etc.).
    """

    name: str
    classification: Literal["pass", "fail", "uncertain"]
    point_estimate: float
    ci_low: float
    ci_high: float
    evidence: dict


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence.

    Returns `float("nan")` for an empty input (all bootstrap replicates
    failed) rather than raising, so callers can classify the degenerate
    case explicitly.
    """
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[int(rank)]
    lower_weight = upper_index - rank
    upper_weight = rank - lower_index
    return sorted_values[lower_index] * lower_weight + sorted_values[upper_index] * upper_weight


# ---------------------------------------------------------------------------
# Negative-control exposure (sham treatment)
# ---------------------------------------------------------------------------


def run_negative_control_exposure(
    patients: Sequence[PatientRecord],
    propensity_scores: Mapping[str, float],
    seed: int,
    n_bootstrap: int = 200,
    horizon_weeks: float = HORIZON_WEEKS,
) -> NegativeControlResult:
    """Rerun `ipcw_aipw` with a sham treatment label; classify the result.

    See module docstring ("DELIBERATE, DOCUMENTED ADAPTATION") for why this
    applies the frozen hazard-ratio-scale rule, adapted, to
    `causal.baselines.ipcw_aipw`'s risk-difference-scale estimate with a
    bootstrap CI in place of an analytic hazard-ratio CI.

    Args:
        patients: the real cohort (real `treatment_arm` is only used here
            to compute the observed treatment prevalence the sham label is
            thresholded to match; it is never read by
            `SyntheticShamExposure.generate` itself).
        propensity_scores: `{patient_id: propensity}`, e.g. from
            `causal.propensity.LogisticPropensityModel.fit_cross_fitted(...)`
            on the REAL cohort. Reused unchanged for every sham-labeled
            resample (not refit per sham label / bootstrap replicate) — a
            documented cost-saving simplification, since propensity here
            only needs to reflect covariate overlap, not the (causally
            inert, by construction) sham label.
        seed: the frozen negative-control seed, passed through to
            `SyntheticShamExposure.generate` and used (via `derive_seed`)
            to seed the bootstrap resampling RNG.
        n_bootstrap: number of bootstrap replicates (default 200, "modest
            given pure-Python" per this task's scope).
        horizon_weeks: forwarded to `ipcw_aipw`.

    Returns:
        A `NegativeControlResult` named `"negative_control_exposure"`.
    """
    n = len(patients)
    n_treated = sum(1 for patient in patients if patient.treatment_arm == 1)
    target_prevalence = (n_treated / n) if n else 0.0

    sham_labels = SyntheticShamExposure.generate(
        [patient.patient_id for patient in patients], seed, target_prevalence
    )
    sham_patients = [
        replace(patient, treatment_arm=sham_labels[patient.patient_id]) for patient in patients
    ]

    point_estimate = ipcw_aipw(sham_patients, propensity_scores, horizon_weeks).estimate

    bootstrap_seed = derive_seed(seed, "negative_control_exposure_bootstrap")
    rng = random.Random(bootstrap_seed)
    bootstrap_estimates: list[float] = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(sham_patients) for _ in range(len(sham_patients))]
        estimate = ipcw_aipw(resample, propensity_scores, horizon_weeks).estimate
        if not math.isnan(estimate):
            bootstrap_estimates.append(estimate)
    bootstrap_estimates.sort()

    ci_low = _percentile(bootstrap_estimates, 2.5)
    ci_high = _percentile(bootstrap_estimates, 97.5)

    ci_contains_null = (not math.isnan(ci_low)) and (not math.isnan(ci_high)) and (
        ci_low <= 0.0 <= ci_high
    )
    point_in_band = NULL_TOLERANCE_BAND[0] <= point_estimate <= NULL_TOLERANCE_BAND[1]

    if not ci_contains_null:
        classification: Literal["pass", "fail", "uncertain"] = "fail"
    elif point_in_band:
        classification = "pass"
    else:
        classification = "uncertain"

    return NegativeControlResult(
        name="negative_control_exposure",
        classification=classification,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        evidence={
            "scale": "risk_difference",
            "scale_adaptation_note": (
                "decisions.md Section 11.3 specifies this rule on a "
                "hazard-ratio scale with a [0.90, 1.10] band around 1.0; "
                "adapted here to the risk-difference scale (null=0.0, "
                "band=[-0.10, 0.10]) since ipcw_aipw produces a risk "
                "difference with no analytic CI -- see module docstring."
            ),
            "null_tolerance_band": NULL_TOLERANCE_BAND,
            "target_prevalence": target_prevalence,
            "n_bootstrap": n_bootstrap,
            "n_bootstrap_successful": len(bootstrap_estimates),
        },
    )


# ---------------------------------------------------------------------------
# Negative-control outcome (baseline CD8)
# ---------------------------------------------------------------------------


def _raw_numeric(patient: PatientRecord, field_name: str) -> float | None:
    raw_value = patient.timeline.static_covariates.get(field_name)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _GenericStandardization:
    means: Mapping[str, float]
    stdevs: Mapping[str, float]


def _fit_generic_standardization(
    patients: Sequence[PatientRecord], fields: Sequence[str]
) -> _GenericStandardization:
    """Per-field mean/stdev for an arbitrary covariate list, fit on `patients`.

    Mirrors `causal.propensity._fit_standardization`'s pattern, generalized
    to an arbitrary field list (that module's version is hardcoded to its
    own fixed covariate set). A field with fewer than two observed numeric
    values, or zero stdev, standardizes to `0.0` at transform time.
    """
    means: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    for field_name in fields:
        values = [
            value
            for value in (_raw_numeric(patient, field_name) for patient in patients)
            if value is not None
        ]
        if not values:
            means[field_name] = 0.0
            stdevs[field_name] = 0.0
            continue
        means[field_name] = statistics.fmean(values)
        stdevs[field_name] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return _GenericStandardization(means=means, stdevs=stdevs)


def _generic_feature_vector(
    patient: PatientRecord, fields: Sequence[str], standardization: _GenericStandardization
) -> list[float]:
    row: list[float] = []
    for field_name in fields:
        raw_value = _raw_numeric(patient, field_name)
        if raw_value is None:
            row.append(0.0)
            continue
        mean = standardization.means.get(field_name, 0.0)
        stdev = standardization.stdevs.get(field_name, 0.0)
        row.append((raw_value - mean) / stdev if stdev > 0 else 0.0)
    return row


def _fit_linear_regression(
    features: Sequence[Sequence[float]], labels: Sequence[float]
) -> tuple[float, list[float]]:
    """Minimal from-scratch linear regression (bias + weights) via batch
    gradient descent on squared error -- same pattern as
    `causal.baselines._fit_logistic`/`causal.propensity.LogisticPropensityModel`,
    without the sigmoid link.
    """
    n = len(features)
    p = len(features[0]) if n else 0
    weights = [0.0] * p
    bias = 0.0
    if n == 0:
        return bias, weights

    for _ in range(_LR_N_ITERATIONS):
        bias_gradient = 0.0
        weight_gradients = [0.0] * p
        for feature_row, label in zip(features, labels, strict=True):
            prediction = bias + sum(w * x for w, x in zip(weights, feature_row, strict=True))
            error = prediction - label
            bias_gradient += error
            for index, x in enumerate(feature_row):
                weight_gradients[index] += error * x
        bias -= _LR_LEARNING_RATE * (bias_gradient / n)
        weights = [
            w - _LR_LEARNING_RATE * (gradient / n)
            for w, gradient in zip(weights, weight_gradients, strict=True)
        ]

    return bias, weights


def _fit_cd8_regression_smd(
    patients: Sequence[PatientRecord], regression_covariates: Sequence[str]
) -> float:
    """Fit `baseline_cd8 ~ treatment_arm + regression_covariates` and return
    the treatment-arm coefficient standardized by baseline CD8's pooled-arm
    SD (the "adjusted standardized mean difference").

    Baseline CD8 itself is standardized (z-scored using its own pooled-arm
    SD, per the pooled-SD SMD convention already used in
    `causal.overlap._smd`) before fitting, so the fitted treatment
    coefficient *is* the standardized mean difference directly -- no
    separate rescaling step is needed after the fit.
    """
    rows: list[tuple[PatientRecord, float]] = []
    cd8_values_by_arm: dict[int, list[float]] = {0: [], 1: []}
    for patient in patients:
        cd8 = _raw_numeric(patient, BASELINE_CD8_FIELD)
        if cd8 is None:
            continue
        rows.append((patient, cd8))
        if patient.treatment_arm in cd8_values_by_arm:
            cd8_values_by_arm[patient.treatment_arm].append(cd8)

    if not rows:
        return 0.0

    var_1 = statistics.variance(cd8_values_by_arm[1]) if len(cd8_values_by_arm[1]) > 1 else 0.0
    var_0 = statistics.variance(cd8_values_by_arm[0]) if len(cd8_values_by_arm[0]) > 1 else 0.0
    pooled_sd = math.sqrt((var_1 + var_0) / 2.0)
    if pooled_sd == 0.0:
        # Degenerate: no variance to standardize by (e.g. a tiny/constant
        # resample). Reporting SMD=0.0 (no detectable imbalance) is more
        # defensible than raising or dividing by zero.
        return 0.0

    mean_cd8 = statistics.fmean(cd8 for _, cd8 in rows)
    covariate_patients = [patient for patient, _ in rows]
    standardization = _fit_generic_standardization(covariate_patients, regression_covariates)

    features: list[list[float]] = []
    labels: list[float] = []
    for patient, cd8 in rows:
        covariate_row = _generic_feature_vector(patient, regression_covariates, standardization)
        features.append([float(patient.treatment_arm), *covariate_row])
        labels.append((cd8 - mean_cd8) / pooled_sd)

    _bias, weights = _fit_linear_regression(features, labels)
    return weights[0]


def run_negative_control_outcome(
    patients: Sequence[PatientRecord],
    adjustment_set: Sequence[str],
    seed: int,
    n_bootstrap: int = 200,
) -> NegativeControlResult:
    """Regress baseline CD8 on treatment + adjustment_set (minus CD8 itself);
    classify the adjusted standardized mean difference.

    `decisions.md` Section 11.3's negative-control-outcome rule (verbatim,
    no scale adaptation needed): pass if the adjusted standardized mean
    difference is below 0.10 in absolute value AND the bootstrap CI for
    the adjusted mean difference contains zero; fail if the CI excludes
    zero; uncertain if the CI contains zero but the absolute standardized
    mean difference is >= 0.10.

    Args:
        patients: the cohort (real treatment assignment and adjustment
            covariates, per `decisions.md` Section 11.3).
        adjustment_set: the full baseline adjustment set (`decisions.md`
            Section 13), which conventionally includes `"baseline_cd8"`.
            This function excludes `"baseline_cd8"` from the right-hand
            side before fitting (all other covariates are retained) and
            calls `assert_no_self_adjustment("baseline_cd8", ...)` on the
            resulting covariate list as a safeguard that the exclusion
            actually took effect (the CD8 double-use safeguard,
            `decisions.md` Section 11.3) -- this call must never raise if
            the exclusion above is implemented correctly.
        seed: the frozen negative-control seed, used (via `derive_seed`) to
            seed the bootstrap resampling RNG.
        n_bootstrap: number of bootstrap replicates (default 200).

    Returns:
        A `NegativeControlResult` named `"negative_control_outcome"`.
    """
    regression_covariates = [
        field_name for field_name in adjustment_set if field_name != BASELINE_CD8_FIELD
    ]
    # Safeguard: confirms the exclusion above actually removed baseline CD8
    # from its own adjustment set. Must not raise.
    assert_no_self_adjustment(BASELINE_CD8_FIELD, regression_covariates)

    point_estimate = _fit_cd8_regression_smd(patients, regression_covariates)

    bootstrap_seed = derive_seed(seed, "negative_control_outcome_bootstrap")
    rng = random.Random(bootstrap_seed)
    patient_list = list(patients)
    bootstrap_estimates: list[float] = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(patient_list) for _ in range(len(patient_list))]
        estimate = _fit_cd8_regression_smd(resample, regression_covariates)
        if not math.isnan(estimate):
            bootstrap_estimates.append(estimate)
    bootstrap_estimates.sort()

    ci_low = _percentile(bootstrap_estimates, 2.5)
    ci_high = _percentile(bootstrap_estimates, 97.5)

    ci_contains_null = (not math.isnan(ci_low)) and (not math.isnan(ci_high)) and (
        ci_low <= 0.0 <= ci_high
    )
    abs_below_threshold = abs(point_estimate) < SMD_THRESHOLD

    if not ci_contains_null:
        classification: Literal["pass", "fail", "uncertain"] = "fail"
    elif abs_below_threshold:
        classification = "pass"
    else:
        classification = "uncertain"

    return NegativeControlResult(
        name="negative_control_outcome",
        classification=classification,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        evidence={
            "scale": "standardized_mean_difference",
            "smd_threshold": SMD_THRESHOLD,
            "regression_covariates": list(regression_covariates),
            "n_bootstrap": n_bootstrap,
            "n_bootstrap_successful": len(bootstrap_estimates),
        },
    )
