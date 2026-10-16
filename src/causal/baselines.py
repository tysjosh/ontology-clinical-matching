"""Causal baseline estimators (Requirements 7.8, 7.9, 7.10; `decisions.md`
Sections 9, 11 "Propensity modeling").

Six baselines, all consuming a `RandomizedCohort`'s `PatientRecord`
sequence (see `cohorts.cohort_construction`) plus, where needed, a
`{patient_id: propensity}` mapping such as
`causal.propensity.LogisticPropensityModel.fit_cross_fitted(...).propensity_scores`:

1. `unadjusted_cox` -- Cox PH with `treatment_arm` as the only covariate.
   `unadjusted_km` is also provided (per-arm Kaplan-Meier survival curves)
   as the descriptive complement Requirement 7.9 groups with it ("unadjusted
   KM/Cox"); it does not itself produce a `CausalEstimate` (a survival
   curve is not a scalar), so it is exposed separately.
2. `adjusted_cox` -- Cox PH with `treatment_arm` plus the full baseline
   covariate list (`causal.propensity.COVARIATE_FIELDS`, reused rather than
   re-derived -- see that module's docstring for the field list and
   extraction/standardization logic reused here).
3. `ps_weighted_marginal_cox` -- the same Cox partial-likelihood machinery,
   marginal (`treatment_arm`-only), with each patient's contribution
   weighted by a stabilized inverse-propensity-of-treatment weight.
4. `outcome_regression` -- g-computation: a from-scratch logistic
   regression predicts each patient's risk of the endpoint by
   `HORIZON_WEEKS` under both `treatment_arm=1` and `treatment_arm=0` (all
   other covariates held fixed), and the standardized risk difference is
   the average of the per-patient differences.
5. `ipcw_ipw` -- inverse-probability-of-treatment weighting combined with
   inverse-probability-of-censoring weighting (a reverse-Kaplan-Meier
   estimate of the censoring distribution), estimating the 96-week risk
   difference.
6. `ipcw_aipw` (`is_primary=True`) -- augmented IPW: combines (4)'s
   outcome-regression predictions with (5)'s IPCW weights via the standard
   AIPW formula.

Dependency scope note
-----------------------

As in `causal.propensity`, only `pydantic`/`PyYAML` are approved runtime
dependencies -- no numpy/scipy/lifelines/statsmodels/scikit-learn. The Cox
partial-likelihood optimizer, the Kaplan-Meier estimator, and the logistic
outcome-regression model below are all minimal, from-scratch, pure-Python
implementations, deliberately scoped to this project's small covariate
list and moderate patient counts (see `causal.propensity`'s module
docstring for the same reasoning, which applies here unchanged).

96-week risk simplification (documented, applies to baselines 4-6)
----------------------------------------------------------------------

Defining a *binary* "event by `HORIZON_WEEKS`" outcome from a right-censored
time-to-event record requires a decision about patients censored before the
horizon (their true horizon-outcome is unobserved). This module uses the
simplest defensible convention:

    Y_i = 1 if the patient had the endpoint event at or before
          `HORIZON_WEEKS` (`event_indicator=True` and elapsed time <=
          horizon); else `Y_i = 0` -- including patients censored before
          the horizon.

This is a *known-biased* point estimate on its own (it silently treats
early censoring as "no event"), which is exactly why baselines 5 and 6
apply an inverse-probability-of-censoring correction: `observed_i` flags
patients censored strictly before the horizon (whose `Y_i=0` is not
trustworthy), and `ipcw_ipw`/`ipcw_aipw` reweight the *complete cases*
(`observed_i=1`) by the inverse probability of remaining uncensored, rather
than trying to impute the unknown outcome for excluded patients directly.
`outcome_regression` (baseline 4) does not apply this correction -- it
fits directly on the (documented, biased) `Y_i` above, consistent with it
being the plain g-computation baseline that the IPCW baselines exist to
correct.

Censoring-weight simplification (documented, Requirement 5's "IPCW-IPW"
instruction)
--------------------------------------------------------------------------

The censoring distribution is estimated with a single reverse-Kaplan-Meier
curve fit on the *pooled* cohort (both arms together), not arm-specific.
This is a simplifying assumption: if censoring depends on treatment arm in
a way not otherwise captured, a pooled censoring model is mis-specified.
Given ACTG175's administrative (calendar-driven, not treatment-driven)
censoring pattern, a pooled estimate is a reasonable, cheap approximation
rather than fitting two separate small-sample KM curves. The resulting
survival probability is floored at `_MIN_CENSORING_SURVIVAL` before
inversion to bound the resulting weight.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby

from causal.propensity import _feature_vector, _fit_standardization
from causal.solvers import SolverConvergenceError, fit_cox_newton, fit_logistic_newton
from cohorts.cohort_construction import PatientRecord

#: The prespecified risk horizon for the outcome-regression / IPCW baselines.
HORIZON_WEEKS: float = 96.0

#: Floor applied to the estimated censoring-survival probability before
#: inversion, so a single, poorly-supported late time point cannot produce
#: an unbounded IPCW weight (see module docstring).
_MIN_CENSORING_SURVIVAL: float = 0.05

#: The Cox and logistic fits below are solved to convergence by Newton-type
#: methods in `causal.solvers` (`decisions.md` Section 26, task 7.9). Until
#: task 7.8 they were 300 fixed gradient steps at learning rate 0.05, which
#: shrank every Cox log-hazard ratio 40-50% toward the null.

#: Bound applied to any linear predictor before `exp()`, to avoid overflow.
_ETA_CLIP: float = 30.0


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalEstimate:
    """A single baseline's result, in a uniform shape across all six baselines.

    Attributes:
        name: a short, stable identifier (e.g. `"unadjusted_cox"`).
        estimate: the point estimate, on the scale named by `scale`.
        se: the estimated standard error, if computed; `None` when a cheap
            SE is not implemented (documented at each call site).
        is_primary: `True` only for `ipcw_aipw`'s result.
        scale: `"log_hazard_ratio"` for the three Cox variants, or
            `"risk_difference"` for the outcome-regression/IPCW variants.
    """

    name: str
    estimate: float
    se: float | None
    is_primary: bool
    scale: str


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (Cox information-matrix inversion)
# ---------------------------------------------------------------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _invert_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]] | None:
    """Gauss-Jordan inverse of a small square matrix, or `None` if singular.

    Minimal, pure-Python implementation intended only for the small
    (<= ~7x7) Cox information matrices computed in this module -- not a
    general-purpose linear-algebra utility.
    """
    n = len(matrix)
    # Augmented [matrix | identity].
    augmented = [
        list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)
    ]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            return None
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot_value = augmented[col][col]
        augmented[col] = [value / pivot_value for value in augmented[col]]

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value_row
                for value, pivot_value_row in zip(augmented[row], augmented[col], strict=True)
            ]

    return [row[n:] for row in augmented]


# ---------------------------------------------------------------------------
# Kaplan-Meier (unadjusted, per arm)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KaplanMeierCurve:
    """A step-function survival curve: `times[i] -> survival[i]`, sorted ascending."""

    times: tuple[datetime, ...]
    survival: tuple[float, ...]

    def survival_at(self, time: datetime) -> float:
        """The survival probability at `time` (right-continuous step lookup)."""
        result = 1.0
        for t, s in zip(self.times, self.survival, strict=True):
            if t <= time:
                result = s
            else:
                break
        return result


def _kaplan_meier(rows: Sequence[tuple[datetime, bool]]) -> KaplanMeierCurve:
    """Standard Kaplan-Meier survival curve.

    Args:
        rows: `(time, is_event)` pairs -- `is_event=True` marks an event at
            `time` for the definition being estimated (the endpoint event
            for the ordinary KM curve, or `censoring_indicator` for the
            reverse-KM censoring-distribution estimate used by the IPCW
            baselines).

    Returns:
        A `KaplanMeierCurve` with one entry per *distinct event time*
        (times with zero events at them do not produce a step, matching
        the standard KM step function).
    """
    sorted_rows = sorted(rows, key=lambda row: row[0])
    n_at_risk = len(sorted_rows)

    times: list[datetime] = []
    survival: list[float] = []
    running_survival = 1.0

    index = 0
    for time_key, group_iter in groupby(sorted_rows, key=lambda row: row[0]):
        group = list(group_iter)
        n_events = sum(1 for _, is_event in group if is_event)
        if n_events > 0 and n_at_risk > 0:
            running_survival *= 1.0 - (n_events / n_at_risk)
            times.append(time_key)
            survival.append(running_survival)
        n_at_risk -= len(group)
        index += len(group)

    return KaplanMeierCurve(times=tuple(times), survival=tuple(survival))


def unadjusted_km(patients: Sequence[PatientRecord]) -> dict[int, KaplanMeierCurve]:
    """Per-arm Kaplan-Meier survival curves for the composite endpoint.

    Descriptive complement to `unadjusted_cox` (see module docstring);
    does not itself return a `CausalEstimate`.
    """
    curves: dict[int, KaplanMeierCurve] = {}
    for arm in (0, 1):
        arm_rows = [
            (patient.endpoint_result.event_time, patient.endpoint_result.event_indicator)
            for patient in patients
            if patient.treatment_arm == arm
        ]
        curves[arm] = _kaplan_meier(arm_rows)
    return curves


# ---------------------------------------------------------------------------
# Cox partial-likelihood machinery (shared by baselines 1-3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CoxRow:
    features: tuple[float, ...]
    event_time: datetime
    event_indicator: bool
    weight: float


def _fit_cox(rows: Sequence[_CoxRow]) -> tuple[list[float], list[list[float]] | None]:
    """Fit a (optionally weighted) Cox PH model by Newton-Raphson on the
    Breslow-tie partial log-likelihood (`causal.solvers.fit_cox_newton`).

    Weighted partial likelihood (reduces to the ordinary Cox partial
    likelihood when every `weight == 1.0`), following the standard
    weighted/robust-Cox formulation used to fit marginal structural Cox
    models from IPW weights (Cole & Hernán 2004):

        LL(beta) = sum_j [ sum_{i in D_j} w_i * eta_i
                            - (sum_{i in D_j} w_i) * log(sum_{k in R_j} w_k * exp(eta_k)) ]

    where `D_j` is the set of patients with an event at time `t_j` and
    `R_j = {k : t_k >= t_j}` is the risk set. Ties are handled via the
    Breslow approximation (risk-set sum over the *whole* tied group, not
    per-event).

    Returns:
        `(beta, inverse_information)` -- the inverse of the observed
        information matrix at the fitted `beta` (`None` if it was singular
        and could not be used for inference).

    Raises:
        SolverConvergenceError: if Newton-Raphson does not meet the frozen
            convergence rule; shrunk coefficients are never returned.
    """
    sorted_rows = sorted(rows, key=lambda row: row.event_time)
    p = len(sorted_rows[0].features) if sorted_rows else 0
    beta, diagnostic, information = fit_cox_newton(
        [row.features for row in sorted_rows],
        [row.event_time for row in sorted_rows],
        [row.event_indicator for row in sorted_rows],
        [row.weight for row in sorted_rows],
        _ETA_CLIP,
    )
    if not diagnostic.converged:
        raise SolverConvergenceError(f"Cox fit on {len(sorted_rows)} rows", diagnostic)
    inverse_information = _invert_matrix(information) if p > 0 else None
    return beta, inverse_information


def _cox_estimate(
    name: str,
    patients: Sequence[PatientRecord],
    feature_fn,
    weight_fn,
    treatment_index: int,
) -> CausalEstimate:
    rows = [
        _CoxRow(
            features=tuple(feature_fn(patient)),
            event_time=patient.endpoint_result.event_time,
            event_indicator=patient.endpoint_result.event_indicator,
            weight=weight_fn(patient),
        )
        for patient in patients
    ]
    beta, inverse_information = _fit_cox(rows)
    se = None
    if inverse_information is not None:
        variance = inverse_information[treatment_index][treatment_index]
        if variance >= 0.0:
            se = math.sqrt(variance)
    return CausalEstimate(
        name=name,
        estimate=beta[treatment_index],
        se=se,
        is_primary=False,
        scale="log_hazard_ratio",
    )


def unadjusted_cox(patients: Sequence[PatientRecord]) -> CausalEstimate:
    """Baseline 1 (Cox half): Cox PH with `treatment_arm` as the only covariate."""
    return _cox_estimate(
        name="unadjusted_cox",
        patients=patients,
        feature_fn=lambda patient: [float(patient.treatment_arm)],
        weight_fn=lambda patient: 1.0,
        treatment_index=0,
    )


def adjusted_cox(patients: Sequence[PatientRecord]) -> CausalEstimate:
    """Baseline 2: Cox PH with `treatment_arm` plus the full baseline covariate list.

    The covariates are the fourteen encoded columns of the frozen Section 13
    adjustment set (`causal.propensity.COVARIATE_FIELDS`, task 7.5), so the
    model has fifteen coefficients: `treatment_arm` first, then those
    columns in order (`models.non_neural_baseline.COEFFICIENT_NAMES`).

    Reuses `causal.propensity`'s covariate extraction and standardization
    (fit once, on the full input `patients`, since there is no cross-fitting
    concern for this diagnostic-style baseline).
    """
    standardization = _fit_standardization(patients)

    def feature_fn(patient: PatientRecord) -> list[float]:
        return [float(patient.treatment_arm), *_feature_vector(patient, standardization)]

    return _cox_estimate(
        name="adjusted_cox",
        patients=patients,
        feature_fn=feature_fn,
        weight_fn=lambda patient: 1.0,
        treatment_index=0,
    )


# ---------------------------------------------------------------------------
# Stabilized IPTW weights (shared by baselines 3, 5, 6)
# ---------------------------------------------------------------------------


def _stabilized_iptw_weights(
    patients: Sequence[PatientRecord], propensity_scores: Mapping[str, float]
) -> dict[str, float]:
    """Stabilized inverse-probability-of-treatment weights.

    `w_i = P(A=1) / e_i` for treated patients, `P(A=a_i) = P(A=0) / (1-e_i)`
    for control patients, using the observed treatment prevalence in
    `patients` as the stabilization numerator.
    """
    n = len(patients)
    if n == 0:
        return {}
    n_treated = sum(1 for patient in patients if patient.treatment_arm == 1)
    p_treated = n_treated / n
    p_control = 1.0 - p_treated

    weights: dict[str, float] = {}
    for patient in patients:
        e_i = propensity_scores[patient.patient_id]
        if patient.treatment_arm == 1:
            weights[patient.patient_id] = p_treated / e_i if e_i > 0.0 else math.inf
        else:
            weights[patient.patient_id] = p_control / (1.0 - e_i) if e_i < 1.0 else math.inf
    return weights


def ps_weighted_marginal_cox(
    patients: Sequence[PatientRecord], propensity_scores: Mapping[str, float]
) -> CausalEstimate:
    """Baseline 3: marginal (treatment_arm-only) Cox PH, IPTW-weighted."""
    weights = _stabilized_iptw_weights(patients, propensity_scores)
    return _cox_estimate(
        name="ps_weighted_marginal_cox",
        patients=patients,
        feature_fn=lambda patient: [float(patient.treatment_arm)],
        weight_fn=lambda patient: weights[patient.patient_id],
        treatment_index=0,
    )


# ---------------------------------------------------------------------------
# 96-week binary outcome + logistic outcome regression (baseline 4, reused by 5/6)
# ---------------------------------------------------------------------------


def _elapsed_weeks(patient: PatientRecord) -> float:
    """Weeks from treatment assignment (time zero) to `endpoint_result.event_time`."""
    treatment_events = patient.timeline.treatment_events
    if not treatment_events:
        raise ValueError(
            f"patient_id={patient.patient_id!r} has no treatment_events; "
            "cannot compute elapsed time to the endpoint."
        )
    assignment_time = min(event.observed_at for event in treatment_events)
    elapsed_days = (patient.endpoint_result.event_time - assignment_time).days
    return elapsed_days / 7.0


def _horizon_outcome(patient: PatientRecord, horizon_weeks: float) -> int:
    """`Y_i`: `1` if the endpoint event occurred at/before `horizon_weeks`, else `0`.

    See module docstring ("96-week risk simplification") -- patients
    censored before the horizon are counted as `0` here, a documented
    simplification corrected for by `ipcw_ipw`/`ipcw_aipw`'s `observed_i`
    weighting, not by this function.
    """
    if patient.endpoint_result.event_indicator and _elapsed_weeks(patient) <= horizon_weeks:
        return 1
    return 0


def _observed_by_horizon(patient: PatientRecord, horizon_weeks: float) -> bool:
    """`True` unless the patient was censored strictly before `horizon_weeks`."""
    if patient.endpoint_result.event_indicator:
        return True
    return _elapsed_weeks(patient) >= horizon_weeks


def _fit_logistic(
    features: Sequence[Sequence[float]], labels: Sequence[float]
) -> tuple[float, list[float]]:
    """Minimal from-scratch logistic regression (bias + weights), fit to
    convergence by Newton / IRLS (`causal.solvers.fit_logistic_newton`) --
    the same solver as `causal.propensity.LogisticPropensityModel`.

    Raises:
        SolverConvergenceError: if the fit does not converge (e.g. a
            covariate perfectly separates the outcome).
    """
    bias, weights, diagnostic = fit_logistic_newton(features, labels)
    if not diagnostic.converged:
        raise SolverConvergenceError(f"outcome logistic fit on {len(features)} rows", diagnostic)
    return bias, weights


def _predict_logistic(bias: float, weights: Sequence[float], feature_row: Sequence[float]) -> float:
    logit = bias + _dot(weights, feature_row)
    logit = max(-_ETA_CLIP, min(_ETA_CLIP, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def _outcome_regression_predictions(
    patients: Sequence[PatientRecord], horizon_weeks: float
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Fit the outcome-regression logistic model and return per-patient
    `(mu1, mu0, Y)`: predicted risk under `treatment_arm=1`, under
    `treatment_arm=0` (covariates held fixed), and the observed `Y_i`.
    """
    standardization = _fit_standardization(patients)
    covariate_rows = {
        patient.patient_id: _feature_vector(patient, standardization) for patient in patients
    }
    labels = [float(_horizon_outcome(patient, horizon_weeks)) for patient in patients]
    features = [
        [float(patient.treatment_arm), *covariate_rows[patient.patient_id]] for patient in patients
    ]
    bias, weights = _fit_logistic(features, labels)

    mu1: dict[str, float] = {}
    mu0: dict[str, float] = {}
    outcomes: dict[str, int] = {}
    for patient in patients:
        covariates = covariate_rows[patient.patient_id]
        mu1[patient.patient_id] = _predict_logistic(bias, weights, [1.0, *covariates])
        mu0[patient.patient_id] = _predict_logistic(bias, weights, [0.0, *covariates])
        outcomes[patient.patient_id] = int(_horizon_outcome(patient, horizon_weeks))

    return mu1, mu0, outcomes


def outcome_regression(
    patients: Sequence[PatientRecord], horizon_weeks: float = HORIZON_WEEKS
) -> CausalEstimate:
    """Baseline 4: g-computation risk difference from a logistic outcome model."""
    mu1, mu0, _ = _outcome_regression_predictions(patients, horizon_weeks)
    n = len(patients)
    mean_mu1 = sum(mu1[patient.patient_id] for patient in patients) / n
    mean_mu0 = sum(mu0[patient.patient_id] for patient in patients) / n
    return CausalEstimate(
        name="outcome_regression",
        estimate=mean_mu1 - mean_mu0,
        se=None,
        is_primary=False,
        scale="risk_difference",
    )


# ---------------------------------------------------------------------------
# IPCW weights (reverse Kaplan-Meier), shared by baselines 5, 6
# ---------------------------------------------------------------------------


def _censoring_kaplan_meier_weeks(
    rows: Sequence[tuple[float, bool]]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Kaplan-Meier survival curve over plain `float` elapsed-weeks time values.

    Thin wrapper around `_kaplan_meier` (which only relies on its time
    values being sortable/comparable, so a `float` "elapsed weeks" works
    identically to a `datetime`) that returns `(times, survival)` tuples
    directly rather than the `datetime`-typed `KaplanMeierCurve`.
    """

    sorted_rows = sorted(rows, key=lambda row: row[0])
    n_at_risk = len(sorted_rows)
    times: list[float] = []
    survival: list[float] = []
    running_survival = 1.0
    for time_key, group_iter in groupby(sorted_rows, key=lambda row: row[0]):
        group = list(group_iter)
        n_events = sum(1 for _, is_event in group if is_event)
        if n_events > 0 and n_at_risk > 0:
            running_survival *= 1.0 - (n_events / n_at_risk)
            times.append(time_key)
            survival.append(running_survival)
        n_at_risk -= len(group)
    return tuple(times), tuple(survival)


def _survival_at_weeks(times: Sequence[float], survival: Sequence[float], t: float) -> float:
    result = 1.0
    for time_value, survival_value in zip(times, survival, strict=True):
        if time_value <= t:
            result = survival_value
        else:
            break
    return result


def _censoring_ipcw_weights(
    patients: Sequence[PatientRecord], horizon_weeks: float
) -> dict[str, float]:
    """`1 / S_C(min(elapsed_i, horizon))`, floored at `_MIN_CENSORING_SURVIVAL`.

    `S_C` is the pooled reverse-Kaplan-Meier estimate of the censoring
    survival function (event definition swapped to `censoring_indicator`;
    see module docstring). Patients censored before `horizon_weeks` are
    still assigned a weight here (used only by callers that separately
    gate on `_observed_by_horizon`).
    """
    censoring_rows = [
        (_elapsed_weeks(patient), patient.endpoint_result.censoring_indicator)
        for patient in patients
    ]
    times, survival = _censoring_kaplan_meier_weeks(censoring_rows)

    weights: dict[str, float] = {}
    for patient in patients:
        capped_weeks = min(_elapsed_weeks(patient), horizon_weeks)
        s_c = max(_survival_at_weeks(times, survival, capped_weeks), _MIN_CENSORING_SURVIVAL)
        weights[patient.patient_id] = 1.0 / s_c
    return weights


def ipcw_ipw(
    patients: Sequence[PatientRecord],
    propensity_scores: Mapping[str, float],
    horizon_weeks: float = HORIZON_WEEKS,
) -> CausalEstimate:
    """Baseline 5: IPTW x IPCW-weighted 96-week risk difference.

    See module docstring ("Censoring-weight simplification") for the
    pooled reverse-KM censoring model and the `observed_i` complete-case
    restriction.
    """
    iptw = _stabilized_iptw_weights(patients, propensity_scores)
    ipcw = _censoring_ipcw_weights(patients, horizon_weeks)

    def _arm_risk(arm: int) -> float:
        numerator = 0.0
        denominator = 0.0
        for patient in patients:
            if patient.treatment_arm != arm:
                continue
            if not _observed_by_horizon(patient, horizon_weeks):
                continue
            weight = iptw[patient.patient_id] * ipcw[patient.patient_id]
            numerator += weight * _horizon_outcome(patient, horizon_weeks)
            denominator += weight
        return numerator / denominator if denominator > 0.0 else float("nan")

    risk_1 = _arm_risk(1)
    risk_0 = _arm_risk(0)
    return CausalEstimate(
        name="ipcw_ipw",
        estimate=risk_1 - risk_0,
        se=None,
        is_primary=False,
        scale="risk_difference",
    )


def ipcw_aipw(
    patients: Sequence[PatientRecord],
    propensity_scores: Mapping[str, float],
    horizon_weeks: float = HORIZON_WEEKS,
    inverse_weight_clip_preset: str | None = None,
) -> CausalEstimate:
    """Baseline 6 (primary): augmented IPW 96-week risk difference.

    Combines `outcome_regression`'s per-patient predictions with the
    IPTW/IPCW weights via:

        psi_hat = mean_i observed_i * [
            (A_i/e_i)   * (Y_i - mu1_i) / S_C_i
            - ((1-A_i)/(1-e_i)) * (Y_i - mu0_i) / S_C_i
        ] + mean_i [mu1_i - mu0_i]

    where `S_C_i` is the (floored) pooled reverse-KM censoring-survival
    probability at `min(elapsed_i, horizon)` (see
    `_censoring_ipcw_weights`; note the formula divides by the censoring
    *survival probability*, equivalently multiplying by the IPCW weight).

    `inverse_weight_clip_preset` exists for the
    `alternative_weight_clipping` sensitivity variant (`decisions.md`
    Section 11's sensitivity clipping). `None` (the default, every primary
    caller) leaves the formula above exactly as written. A
    `causal.weighting.CLIP_PRESETS` name clamps every patient's
    inverse-propensity weight (`1/e_i` for arm 1, `1/(1-e_i)` for arm 0)
    into that preset's percentile bounds, computed over all patients,
    before the augmentation term uses it -- so only the weights change,
    never the outcome model or the censoring weights.
    """
    mu1, mu0, outcomes = _outcome_regression_predictions(patients, horizon_weeks)
    ipcw_weights = _censoring_ipcw_weights(patients, horizon_weeks)
    n = len(patients)

    clipped_inverse_weights: dict[str, float] | None = None
    if inverse_weight_clip_preset is not None:
        from causal.weighting import clip_weights

        raw_inverse_weights: dict[str, float] = {}
        for patient in patients:
            e_i = propensity_scores[patient.patient_id]
            if patient.treatment_arm == 1 and e_i > 0.0:
                raw_inverse_weights[patient.patient_id] = 1.0 / e_i
            elif patient.treatment_arm != 1 and e_i < 1.0:
                raw_inverse_weights[patient.patient_id] = 1.0 / (1.0 - e_i)
        clipped_inverse_weights = dict(
            clip_weights(raw_inverse_weights, preset=inverse_weight_clip_preset)
        )

    augmentation_sum = 0.0
    plug_in_sum = 0.0
    for patient in patients:
        pid = patient.patient_id
        plug_in_sum += mu1[pid] - mu0[pid]
        if not _observed_by_horizon(patient, horizon_weeks):
            continue
        e_i = propensity_scores[pid]
        y_i = outcomes[pid]
        ipcw_i = ipcw_weights[pid]
        if clipped_inverse_weights is not None:
            if pid not in clipped_inverse_weights:
                continue  # e_i of exactly 0 or 1: no finite weight, as below
            weight_i = clipped_inverse_weights[pid]
            if patient.treatment_arm == 1:
                augmentation_sum += weight_i * (y_i - mu1[pid]) * ipcw_i
            else:
                augmentation_sum -= weight_i * (y_i - mu0[pid]) * ipcw_i
            continue
        if patient.treatment_arm == 1:
            if e_i > 0.0:
                augmentation_sum += (1.0 / e_i) * (y_i - mu1[pid]) * ipcw_i
        else:
            if e_i < 1.0:
                augmentation_sum -= (1.0 / (1.0 - e_i)) * (y_i - mu0[pid]) * ipcw_i

    psi_hat = (augmentation_sum / n) + (plug_in_sum / n)
    return CausalEstimate(
        name="ipcw_aipw",
        estimate=psi_hat,
        se=None,
        is_primary=True,
        scale="risk_difference",
    )


# ---------------------------------------------------------------------------
# Convenience: run all six baselines
# ---------------------------------------------------------------------------


def run_all_baselines(
    patients: Sequence[PatientRecord],
    propensity_scores: Mapping[str, float],
    horizon_weeks: float = HORIZON_WEEKS,
) -> list[CausalEstimate]:
    """Run all six causal baselines and return their `CausalEstimate`s.

    `unadjusted_km`'s per-arm survival curves are not included (not a
    scalar `CausalEstimate`); call it separately if needed.
    """
    return [
        unadjusted_cox(patients),
        adjusted_cox(patients),
        ps_weighted_marginal_cox(patients, propensity_scores),
        outcome_regression(patients, horizon_weeks),
        ipcw_ipw(patients, propensity_scores, horizon_weeks),
        ipcw_aipw(patients, propensity_scores, horizon_weeks),
    ]
