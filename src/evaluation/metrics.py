"""Factual and selective-prediction metrics (Requirements 8.4, 8.5, 10.12;
`decisions.md` Section 18 "Primary evaluation metrics" -- factual-prediction
and selective-prediction groups).

`decisions.md` Section 18 freezes the factual-prediction metric set to:
time-dependent concordance, 96-week AUROC, 96-week AUPRC, IPCW Brier score,
calibration slope/intercept, and CD4/CD8 mean absolute error. This module
implements each as a small, independent, pure function operating on plain
sequences (risk scores, event times, binary labels, predicted/observed
values) -- **not** on `cohorts.cohort_construction.PatientRecord` -- so the
same functions are reusable across whichever model produced the
predictions (`models.non_neural_baseline.CoxBaseline`,
`models.sequence_baseline.SequenceGRU`, and later the intervention-aware
model), matching this module's role as a model-agnostic evaluation layer.

Per Requirement 8.5, every metric here is a **factual predictive**
quantity (how well a model ranks or predicts *observed* outcomes) and
must not be described as a measure of causal validity.

Dependency scope note
----------------------

As elsewhere in this project, only `pydantic`/`PyYAML` are approved
runtime dependencies -- no numpy/scipy/scikit-learn. Every metric below is
a minimal, from-scratch, pure-Python implementation.

IPCW Brier score -- censoring-weight approach
----------------------------------------------

`causal.baselines` already implements a pooled reverse-Kaplan-Meier
censoring-weight estimator (`_censoring_ipcw_weights`), but it is a
private helper scoped to that module's `PatientRecord`-based baselines.
Rather than reaching into another module's underscore-prefixed internals
(an awkward cross-module dependency), `ipcw_brier_score` below implements
its own small, generic reverse-KM estimator over plain `(time, censored)`
pairs -- the same standard method, restated locally so this module has no
dependency on `causal.baselines` at all. See `_reverse_km_censoring_survival`.

Time-tie handling (documented simplification, applies to `concordance_index`
and the reverse-KM curve below): exactly-equal event times between two
patients are treated as *not comparable* for concordance (rather than
applying a specific tie-breaking convention), keeping the pairwise
implementation simple; equal *risk scores* for an otherwise-comparable
pair are given half credit, the standard convention.

Selective-prediction metrics -- decoupling from `uncertainty.support_gate`
----------------------------------------------------------------------------

`decisions.md` Section 18's selective-prediction group (retained coverage,
selective risk, the risk-coverage curve, AURCC, incorrect-release rate,
unnecessary-abstention rate, error at matched coverage; Requirement 10.12)
is implemented below on the same plain-sequence footing as the
factual-prediction group above. In particular, these functions accept
generic boolean `retained_flags` (patient retained vs. abstained) and a
generic per-patient `confidence_scores` sequence rather than importing
`uncertainty.support_gate.PatientSupportGate` or its `SupportDecision`
type -- callers compute `retained_flags` however they like (e.g. from
`SupportDecision.supported`) and pass the plain booleans in, keeping this
module free of any dependency on `uncertainty`.

Default loss function: unless a `loss_fn` is supplied, every function
below uses the Brier-style squared error `(prediction - label)^2`, with
`label` coerced to `1.0`/`0.0`. `loss_fn` always has the signature
`(prediction: float, label: bool) -> float`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import groupby

from causal.baselines import CausalEstimate
from causal.solvers import SolverConvergenceError, fit_logistic_newton
from cohorts.sealed_reference import InternalReference, compare_to_reference

#: The `InternalReference.quantities` key naming the randomized 96-week
#: risk difference (`design.md` Section 4.5). Selected *by name* rather than
#: positionally so a reference carrying quantities on several scales cannot
#: have the wrong one differenced against a risk-difference estimate
#: (Requirements 14.3-14.4).
RISK_DIFFERENCE_QUANTITY = "risk_difference"

#: The 1-covariate calibration fit below is solved to convergence by
#: `causal.solvers.fit_logistic_newton` (`decisions.md` Section 26, task
#: 7.9); until task 7.8 it was 500 fixed gradient steps.

#: Bound applied to any logit before `exp()`/`log()`, avoiding overflow and
#: `log(0)`.
_LOGIT_CLIP: float = 30.0

#: Floor/ceiling applied to a probability before converting it to a logit,
#: so an exactly-0 or exactly-1 predicted risk does not blow up `logit()`.
_PROBABILITY_EPSILON: float = 1e-6

#: Floor applied to the estimated censoring-survival probability before
#: inversion, bounding the resulting IPCW weight (same role as
#: `causal.baselines._MIN_CENSORING_SURVIVAL`).
_MIN_CENSORING_SURVIVAL: float = 0.05


# ---------------------------------------------------------------------------
# 1. Time-dependent concordance (Harrell's C-index)
# ---------------------------------------------------------------------------


def concordance_index(
    risk_scores: Sequence[float],
    event_times: Sequence[float],
    event_indicators: Sequence[bool],
) -> float:
    """Harrell's concordance index for survival data.

    For every comparable pair (the member with the shorter observed time
    experienced the event), checks whether the model's risk ordering
    agrees: the earlier-event patient should have the higher risk score.
    `C-index = concordant / comparable`, with tied risk scores on a
    comparable pair given half credit.

    Standard `O(n^2)` pairwise implementation -- fine for this project's
    moderate patient counts; for much larger `n` this would need the
    `O(n log n)` sort-based algorithm instead.

    Args:
        risk_scores: higher score = higher predicted risk of an earlier event.
        event_times: observed time (event or censoring) for each patient.
        event_indicators: `True` if the patient had the event at `event_times[i]`.

    Returns:
        The concordance index in `[0, 1]`, or `nan` if there are no
        comparable pairs (e.g. every patient censored, or fewer than 2
        patients).
    """
    n = len(risk_scores)
    comparable = 0
    concordant = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            time_i, time_j = event_times[i], event_times[j]
            if time_i == time_j:
                # Documented simplification (module docstring): exactly-tied
                # times are not treated as comparable.
                continue
            if time_i < time_j:
                earlier_event = event_indicators[i]
                risk_earlier, risk_later = risk_scores[i], risk_scores[j]
            else:
                earlier_event = event_indicators[j]
                risk_earlier, risk_later = risk_scores[j], risk_scores[i]
            if not earlier_event:
                continue

            comparable += 1
            if risk_earlier > risk_later:
                concordant += 1.0
            elif risk_earlier == risk_later:
                concordant += 0.5

    return concordant / comparable if comparable > 0 else float("nan")


# ---------------------------------------------------------------------------
# 2. 96-week AUROC / AUPRC
# ---------------------------------------------------------------------------


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank-based AUROC: `P(score of a random positive > score of a random negative)`.

    Computed via the same pairwise-comparison logic as `concordance_index`
    (equivalent, for binary labels, to the trapezoidal ROC-curve area),
    with tied scores between a positive/negative pair given half credit.

    Args:
        scores: predicted risk scores (higher = more likely positive).
        labels: binary 96-week-outcome labels.

    Returns:
        AUROC in `[0, 1]`, or `nan` if there are no positives or no
        negatives (undefined).
    """
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return float("nan")

    concordant = 0.0
    for pos_score in positives:
        for neg_score in negatives:
            if pos_score > neg_score:
                concordant += 1.0
            elif pos_score == neg_score:
                concordant += 0.5

    return concordant / (len(positives) * len(negatives))


def auprc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the precision-recall curve, via rank-based average precision.

    Sorts by score descending (ties broken by original index, so the
    result is deterministic), then at each *positive* example computes
    precision-so-far; the area under the resulting step-function
    precision-recall curve equals the mean of these per-positive
    precision values (the standard "average precision" identity -- no
    separate trapezoidal-integration step is needed once sorted this way).

    Args:
        scores: predicted risk scores (higher = more likely positive).
        labels: binary 96-week-outcome labels.

    Returns:
        AUPRC in `[0, 1]`, or `nan` if there are no positive labels.
    """
    n_positive = sum(1 for label in labels if label)
    if n_positive == 0:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    true_positives = 0
    false_positives = 0
    precision_sum = 0.0
    for i in order:
        if labels[i]:
            true_positives += 1
            precision_sum += true_positives / (true_positives + false_positives)
        else:
            false_positives += 1

    return precision_sum / n_positive


# ---------------------------------------------------------------------------
# 3. IPCW Brier score
# ---------------------------------------------------------------------------


def _reverse_km_censoring_survival(
    event_times: Sequence[float], event_indicators: Sequence[bool]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pooled reverse-Kaplan-Meier estimate of the censoring-survival function.

    Standard reverse-KM: swaps the event definition to "was censored"
    (`not event_indicators[i]`) and runs the ordinary Kaplan-Meier
    recursion, giving `S_C(t) = P(censoring time > t)`. Fit on the pooled
    input (no stratification) -- a documented simplification consistent
    with `causal.baselines`'s equivalent estimator (see that module's
    docstring for the same reasoning); this implementation is
    self-contained and does not import from `causal.baselines`.

    Returns:
        `(times, survival)`, one entry per distinct time at which at
        least one censoring event occurred, sorted ascending by time.
    """
    rows = sorted(zip(event_times, event_indicators, strict=True), key=lambda row: row[0])
    n_at_risk = len(rows)

    times: list[float] = []
    survival: list[float] = []
    running_survival = 1.0

    for time_value, group_iter in groupby(rows, key=lambda row: row[0]):
        group = list(group_iter)
        n_censoring_events = sum(1 for _, is_event in group if not is_event)
        if n_censoring_events > 0 and n_at_risk > 0:
            running_survival *= 1.0 - (n_censoring_events / n_at_risk)
            times.append(time_value)
            survival.append(running_survival)
        n_at_risk -= len(group)

    return tuple(times), tuple(survival)


def _survival_at(times: Sequence[float], survival: Sequence[float], t: float) -> float:
    result = 1.0
    for time_value, survival_value in zip(times, survival, strict=True):
        if time_value <= t:
            result = survival_value
        else:
            break
    return result


def ipcw_brier_score(
    risk_scores: Sequence[float],
    event_times: Sequence[float],
    event_indicators: Sequence[bool],
    horizon: float,
) -> float:
    """Inverse-probability-of-censoring-weighted Brier score at `horizon`.

    `mean_i [ I(observed_i) * w_i * (Y_i - risk_i)^2 ]`, where:

    - `Y_i = 1` if the patient had the event at/before `horizon`, else `0`;
    - `observed_i` is `False` only for patients censored strictly before
      `horizon` (their true horizon outcome is unobserved, so they
      contribute `0` rather than a guessed value);
    - `w_i = 1 / S_C(min(event_times[i], horizon))`, `S_C` the pooled
      reverse-KM censoring-survival curve (see
      `_reverse_km_censoring_survival`), floored at `_MIN_CENSORING_SURVIVAL`
      before inversion so a single poorly-supported late time point cannot
      produce an unbounded weight.

    `risk_scores` are interpreted as predicted probabilities of the event
    by `horizon`, i.e. on the same `[0, 1]` scale as `Y_i`.

    Note: when no patient is censored before `horizon` (every patient is
    `observed_i=True` with `S_C == 1`), this reduces exactly to the plain
    (unweighted) Brier score -- the basis for the "perfect predictions give
    near-zero Brier score" sanity check used in this module's tests.

    Args:
        risk_scores: predicted probability of the event by `horizon`, per patient.
        event_times: observed time (event or censoring) for each patient.
        event_indicators: `True` if the patient had the event at `event_times[i]`.
        horizon: the fixed prediction horizon (e.g. 96 weeks).

    Returns:
        The IPCW Brier score (lower is better; `0.0` is a perfect score).
    """
    n = len(risk_scores)
    if n == 0:
        return float("nan")

    censoring_times, censoring_survival = _reverse_km_censoring_survival(
        event_times, event_indicators
    )

    total = 0.0
    for risk, time, had_event in zip(risk_scores, event_times, event_indicators, strict=True):
        observed = had_event or time >= horizon
        if not observed:
            continue
        outcome = 1.0 if (had_event and time <= horizon) else 0.0
        s_c = max(
            _survival_at(censoring_times, censoring_survival, min(time, horizon)),
            _MIN_CENSORING_SURVIVAL,
        )
        weight = 1.0 / s_c
        total += weight * (outcome - risk) ** 2

    return total / n


# ---------------------------------------------------------------------------
# 4. Calibration slope / intercept
# ---------------------------------------------------------------------------


def _logit(probability: float) -> float:
    clipped = min(max(probability, _PROBABILITY_EPSILON), 1.0 - _PROBABILITY_EPSILON)
    return math.log(clipped / (1.0 - clipped))


def _fit_logistic_1d(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Tiny 1-covariate logistic regression `logit(Y) = intercept + slope * x`,
    fit to convergence by Newton / IRLS (`causal.solvers.fit_logistic_newton`).

    Returns:
        `(intercept, slope)`.

    Raises:
        SolverConvergenceError: if the fit does not converge, e.g. when the
            predicted logits perfectly separate the labels (the calibration
            slope is then unbounded and no finite value is reported).
    """
    intercept, weights, diagnostic = fit_logistic_newton([[value] for value in x], y)
    if not diagnostic.converged:
        raise SolverConvergenceError(f"calibration fit on {len(x)} rows", diagnostic)
    return intercept, weights[0]


def calibration_slope_intercept(
    predicted_risks: Sequence[float], labels: Sequence[bool]
) -> tuple[float, float]:
    """Calibration slope and intercept for a set of predicted probabilities.

    Fits `logit(Y) = intercept + slope * logit(predicted_risk)` -- the
    standard calibration-model formulation. A well-calibrated model has
    `slope ~= 1` and `intercept ~= 0`.

    Args:
        predicted_risks: predicted probabilities in `(0, 1)` (values at the
            boundary are clipped, see `_logit`).
        labels: binary observed outcomes.

    Returns:
        `(intercept, slope)`.
    """
    predicted_logits = [_logit(risk) for risk in predicted_risks]
    observed = [1.0 if label else 0.0 for label in labels]
    return _fit_logistic_1d(predicted_logits, observed)


# ---------------------------------------------------------------------------
# 5. CD4 / CD8 mean absolute error
# ---------------------------------------------------------------------------


def mean_absolute_error(predicted: Sequence[float], observed: Sequence[float]) -> float:
    """`mean(|predicted_i - observed_i|)` -- used for CD4/CD8 trajectory error.

    Generic (not CD4/CD8-specific): reusable for any paired
    predicted/observed numeric sequence.
    """
    if len(predicted) == 0:
        return float("nan")
    return sum(
        abs(p - o) for p, o in zip(predicted, observed, strict=True)
    ) / len(predicted)


# ---------------------------------------------------------------------------
# 6. Selective-prediction metrics (Requirement 10.12)
# ---------------------------------------------------------------------------


def _default_loss(prediction: float, label: bool) -> float:
    """Brier-style squared error, the default loss for every metric below."""
    return (prediction - (1.0 if label else 0.0)) ** 2


def retained_coverage(retained_flags: Sequence[bool]) -> float:
    """Fraction of patients retained (not abstained).

    Args:
        retained_flags: `True` if the patient's prediction is retained
            (released); `False` if the patient was abstained on.

    Returns:
        `sum(retained_flags) / len(retained_flags)`, or `nan` if
        `retained_flags` is empty.
    """
    n = len(retained_flags)
    if n == 0:
        return float("nan")
    return sum(1 for flag in retained_flags if flag) / n


def selective_risk(
    predictions: Sequence[float],
    labels: Sequence[bool],
    retained_flags: Sequence[bool],
    loss_fn: Callable[[float, bool], float] | None = None,
) -> float:
    """Average loss computed only over retained (non-abstained) patients.

    Args:
        predictions: per-patient predicted values (e.g. predicted risk).
        labels: per-patient true binary outcomes.
        retained_flags: `True` if the patient's prediction is retained.
        loss_fn: `(prediction, label) -> loss`; defaults to the
            Brier-style squared error `(prediction - label)^2`.

    Returns:
        The mean loss over retained patients, or `nan` if no patient is
        retained.
    """
    loss = loss_fn if loss_fn is not None else _default_loss
    losses = [
        loss(prediction, label)
        for prediction, label, retained in zip(predictions, labels, retained_flags, strict=True)
        if retained
    ]
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def risk_coverage_curve(
    predictions: Sequence[float],
    labels: Sequence[bool],
    confidence_scores: Sequence[float],
    loss_fn: Callable[[float, bool], float] | None = None,
) -> list[tuple[float, float]]:
    """The risk-coverage curve: selective risk at every achievable coverage level.

    Patients are sorted by `confidence_scores` descending (most confident
    first). At each `k` from `1` to `n`, the top-`k` most confident
    patients are treated as retained and the remainder as abstained, and
    the selective risk over that retained set is computed. This traces
    out one `(coverage, risk)` point per possible coverage level, in
    increasing-coverage order.

    Args:
        predictions: per-patient predicted values.
        labels: per-patient true binary outcomes.
        confidence_scores: per-patient scalar confidence (higher = more
            confident); used only to rank/threshold patients, not as the
            prediction itself.
        loss_fn: `(prediction, label) -> loss`; defaults to the
            Brier-style squared error.

    Returns:
        A list of `(coverage, risk)` pairs, one per `k = 1..n`, with
        `coverage = k / n`. Empty if there are no patients.
    """
    n = len(predictions)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: (-confidence_scores[i], i))

    curve: list[tuple[float, float]] = []
    for k in range(1, n + 1):
        retained_indices = set(order[:k])
        retained_flags = [i in retained_indices for i in range(n)]
        coverage = k / n
        risk = selective_risk(predictions, labels, retained_flags, loss_fn)
        curve.append((coverage, risk))

    return curve


def aurcc(
    predictions: Sequence[float],
    labels: Sequence[bool],
    confidence_scores: Sequence[float],
    loss_fn: Callable[[float, bool], float] | None = None,
) -> float:
    """Area under the risk-coverage curve (AURCC), via trapezoidal integration.

    Computes `risk_coverage_curve(...)` and integrates the resulting
    `(coverage, risk)` points directly with the trapezoidal rule, over
    the domain the curve already covers (`coverage` in
    `[1/n, 1.0]` -- the curve does not include a `coverage=0` point,
    since selective risk is undefined for an empty retained set). Lower
    AURCC indicates a better confidence ranking (low risk sustained to
    high coverage).

    Args:
        predictions: per-patient predicted values.
        labels: per-patient true binary outcomes.
        confidence_scores: per-patient scalar confidence used to rank patients.
        loss_fn: `(prediction, label) -> loss`; defaults to the
            Brier-style squared error.

    Returns:
        The trapezoidal-integrated area under the risk-coverage curve, or
        `nan` if there are fewer than 2 patients (the curve has fewer
        than 2 points, so no interval can be integrated).
    """
    curve = risk_coverage_curve(predictions, labels, confidence_scores, loss_fn)
    if len(curve) < 2:
        return float("nan")

    area = 0.0
    for (coverage_a, risk_a), (coverage_b, risk_b) in zip(curve, curve[1:], strict=False):
        area += (coverage_b - coverage_a) * (risk_a + risk_b) / 2.0
    return area


def incorrect_release_rate(
    predictions: Sequence[float],
    labels: Sequence[bool],
    retained_flags: Sequence[bool],
    threshold: float = 0.5,
) -> float:
    """Among retained patients, the fraction whose released prediction was wrong.

    A retained patient's thresholded prediction (`prediction >= threshold`
    -> predicted positive) is a "release." This is the fraction of
    releases that disagree with the patient's true label.

    Args:
        predictions: per-patient predicted values.
        labels: per-patient true binary outcomes.
        retained_flags: `True` if the patient's prediction is retained.
        threshold: the decision threshold applied to `predictions`.

    Returns:
        `count(retained AND (pred>=threshold) != label) / count(retained)`,
        or `nan` if no patient is retained.
    """
    retained_count = 0
    incorrect_count = 0
    for prediction, label, retained in zip(predictions, labels, retained_flags, strict=True):
        if not retained:
            continue
        retained_count += 1
        predicted_positive = prediction >= threshold
        if predicted_positive != label:
            incorrect_count += 1

    if retained_count == 0:
        return float("nan")
    return incorrect_count / retained_count


def unnecessary_abstention_rate(
    predictions: Sequence[float],
    labels: Sequence[bool],
    retained_flags: Sequence[bool],
    threshold: float = 0.5,
) -> float:
    """Among abstained patients, the fraction that would have been released correctly.

    Args:
        predictions: per-patient predicted values.
        labels: per-patient true binary outcomes.
        retained_flags: `True` if the patient's prediction is retained;
            `False` marks an abstained patient.
        threshold: the decision threshold applied to `predictions`.

    Returns:
        `count(not retained AND (pred>=threshold) == label) / count(not retained)`,
        or `nan` if no patient abstained.
    """
    abstained_count = 0
    would_be_correct_count = 0
    for prediction, label, retained in zip(predictions, labels, retained_flags, strict=True):
        if retained:
            continue
        abstained_count += 1
        predicted_positive = prediction >= threshold
        if predicted_positive == label:
            would_be_correct_count += 1

    if abstained_count == 0:
        return float("nan")
    return would_be_correct_count / abstained_count


def _risk_at_coverage(
    curve: Sequence[tuple[float, float]], target_coverage: float
) -> float:
    """Look up the selective risk at the achievable coverage nearest `target_coverage`."""
    if not curve:
        return float("nan")
    _, risk = min(curve, key=lambda point: abs(point[0] - target_coverage))
    return risk


def error_at_matched_coverage(
    predictions_a: Sequence[float],
    labels: Sequence[bool],
    confidence_scores_a: Sequence[float],
    target_coverage: float,
    predictions_b: Sequence[float] | None = None,
    confidence_scores_b: Sequence[float] | None = None,
    loss_fn: Callable[[float, bool], float] | None = None,
) -> float | tuple[float, float]:
    """Selective risk at a specific target coverage level ("matched coverage").

    Looks up the nearest achievable coverage in model A's risk-coverage
    curve to `target_coverage` and returns the selective risk there. If
    `predictions_b`/`confidence_scores_b` are also supplied (model B, over
    the same `labels`), also looks up model B's risk at its own nearest
    achievable coverage to `target_coverage`, so the two models can be
    compared fairly at (as close as achievable to) the same coverage
    level.

    Args:
        predictions_a: model A's per-patient predicted values.
        labels: per-patient true binary outcomes (shared by both models).
        confidence_scores_a: model A's per-patient confidence scores.
        target_coverage: the desired coverage level (e.g. `0.8`).
        predictions_b: model B's per-patient predicted values, if comparing two models.
        confidence_scores_b: model B's per-patient confidence scores, if comparing two models.
        loss_fn: `(prediction, label) -> loss`; defaults to the
            Brier-style squared error, applied to both models.

    Returns:
        Model A's selective risk at the matched coverage as a scalar, or
        `(risk_a, risk_b)` if model B's arguments are also supplied.
    """
    curve_a = risk_coverage_curve(predictions_a, labels, confidence_scores_a, loss_fn)
    risk_a = _risk_at_coverage(curve_a, target_coverage)

    if predictions_b is None or confidence_scores_b is None:
        return risk_a

    curve_b = risk_coverage_curve(predictions_b, labels, confidence_scores_b, loss_fn)
    risk_b = _risk_at_coverage(curve_b, target_coverage)
    return risk_a, risk_b


# ---------------------------------------------------------------------------
# 7. Causal-recovery metrics (Requirements 14.8, 14.9; `decisions.md`
#    Section 18 "Causal recovery")
# ---------------------------------------------------------------------------

#: The frozen published external-reference hazard ratio (`decisions.md`
#: Section 8: HR = 0.50, zidovudine+didanosine vs. zidovudine alone, Hammer
#: et al., NEJM 1996). This is a fixed historical constant baked directly
#: into the `E_logHR` formula below -- it is **not** the internal
#: randomized reference, and is intentionally not read from
#: `InternalReference`/`SealedReference` (that would require unsealing the
#: internal reference just to fetch a number the formula already hardcodes).
_PUBLISHED_REFERENCE_HAZARD_RATIO: float = 0.50


def estimated_hazard_ratio(log_hr: float) -> float:
    """Exponentiate a log-hazard-ratio estimate into a hazard ratio.

    `causal.baselines.adjusted_cox`'s `.estimate` is already on the
    log-hazard-ratio scale (that module's `scale="log_hazard_ratio"`), so
    this is a one-line naming/documentation helper for the "estimated HR"
    concept in `decisions.md` Section 18's causal-recovery metric group,
    not a new estimation procedure.

    Args:
        log_hr: a log-hazard-ratio estimate (e.g. from `adjusted_cox`).

    Returns:
        `exp(log_hr)`, the hazard ratio.
    """
    return math.exp(log_hr)


def absolute_log_hr_error(log_hr_hat: float) -> float:
    """Absolute log-hazard-ratio error against the frozen published HR.

    `E_logHR = |log(HR_hat) - log(0.50)|` (`decisions.md` Section 18),
    where `0.50` is the frozen published external-reference hazard ratio
    (Section 8) -- a hardcoded constant in this formula, not a value read
    from `InternalReference`. `log_hr_hat` is already on the log scale
    (see `estimated_hazard_ratio`'s docstring), so no separate `log()` call
    is needed on it here.

    Args:
        log_hr_hat: the estimated log-hazard-ratio, `log(HR_hat)`.

    Returns:
        `|log_hr_hat - log(0.50)|`.
    """
    return abs(log_hr_hat - math.log(_PUBLISHED_REFERENCE_HAZARD_RATIO))


def effect_direction_agreement(
    log_hr_hat: float,
    reference_log_hr: float = math.log(_PUBLISHED_REFERENCE_HAZARD_RATIO),
) -> bool:
    """Whether an estimated log-hazard-ratio agrees in *direction* with a
    reference log-hazard-ratio.

    Sign convention: a negative log-HR indicates a protective effect
    (hazard reduced relative to the comparator arm, HR < 1); a positive
    log-HR indicates a harmful effect (HR > 1); exactly `0.0` indicates no
    effect either direction. "Agreement" means both values are on the same
    side of zero (both negative, or both positive) -- i.e. both estimates
    point toward the same qualitative conclusion (protective vs. harmful),
    regardless of magnitude.

    An exact-zero value on either side is treated as agreement only when
    the other side is also exactly zero (see the truth table implied
    below); any other zero/nonzero pairing disagrees, since "no effect" is
    not the same direction as "an effect."

    Args:
        log_hr_hat: the estimated log-hazard-ratio.
        reference_log_hr: the reference log-hazard-ratio to compare
            direction against. Defaults to `log(0.50)`, the published
            external reference's direction (a protective effect).

    Returns:
        `True` if `log_hr_hat` and `reference_log_hr` have the same sign
        (both `> 0`, both `< 0`, or both exactly `0.0`); `False` otherwise.
    """
    def _sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    return _sign(log_hr_hat) == _sign(reference_log_hr)


def ci_overlap(
    ci_low_a: float, ci_high_a: float, ci_low_b: float, ci_high_b: float
) -> bool:
    """Whether two confidence intervals overlap.

    Standard interval-overlap test: two closed intervals `[low_a, high_a]`
    and `[low_b, high_b]` overlap iff neither lies entirely to one side of
    the other, i.e. `low_a <= high_b and low_b <= high_a`. Touching
    endpoints (e.g. `high_a == low_b`) count as overlapping.

    Args:
        ci_low_a: lower bound of interval A.
        ci_high_a: upper bound of interval A.
        ci_low_b: lower bound of interval B.
        ci_high_b: upper bound of interval B.

    Returns:
        `True` if the two intervals overlap.
    """
    return ci_low_a <= ci_high_b and ci_low_b <= ci_high_a


@dataclass(frozen=True)
class PublishedReference:
    """A published (historical, non-internal) reference effect estimate.

    Deliberately a distinct, non-interchangeable type from
    `cohorts.sealed_reference.InternalReference`, even though it has a
    similar shape (`value`, `source`). This class exists **only** so that
    `risk_difference_error_vs_internal_reference`'s rejection of a
    published-reference argument is a meaningful, testable runtime check
    (`decisions.md` Section 18: the 96-week risk-difference error must be
    computed "against the internal randomized reference ... not the
    published reference, per the reference-effect policy").

    No real pipeline code path should ever construct an actual
    "published reference" object and pass it where an `InternalReference`
    is expected -- `decisions.md` Section 8's frozen non-independence /
    no-published-reference-for-primary-comparison rule. This type is a
    deliberately-wrong-shaped stand-in for that misuse, used by
    `tests/unit/test_metrics.py` to confirm the guard actually rejects it.

    Attributes:
        value: the published effect estimate (e.g. a hazard ratio or risk
            difference).
        source: a citation/description of where the published value comes
            from (e.g. `"Hammer et al."`).
    """

    value: float
    source: str


def risk_difference_error_vs_internal_reference(
    estimated_risk_difference: float, reference: InternalReference
) -> float:
    """96-week risk-difference error against the internal randomized
    reference only.

    `decisions.md` Section 18 requires this metric to be computed "against
    the internal randomized reference (Section 8, item 2) -- not the
    published reference, per the reference-effect policy." The type hint
    (`reference: InternalReference`) documents that requirement, but a type
    hint alone does not stop misuse at runtime in Python (the same pattern
    used elsewhere in this codebase, e.g.
    `infra.reference_guard.assert_reference_unlocked`), so this function
    also runtime-checks `isinstance(reference, InternalReference)` and
    raises `TypeError` for anything else -- including a `PublishedReference`
    instance, or any other non-`InternalReference` object.

    Scale selection (Requirements 14.3-14.4; `design.md` Section 2.6
    "Tension 4"): a format-version-2 reference carries several quantities
    on several scales, only one of which is comparable with a risk
    difference. This function therefore selects the `"risk_difference"`
    quantity *by name* and delegates to
    `cohorts.sealed_reference.compare_to_reference`, which asserts the
    selected quantity's `scale` equals the estimate's own
    (`"risk_difference"`) and raises `ScaleMismatchError` otherwise -- so a
    reference whose `"risk_difference"` slot somehow carried a log-hazard
    ratio would be refused rather than silently differenced.

    A version-1-shaped reference (empty `quantities`) keeps its original
    behavior exactly: the error is taken against `reference.value`, whose
    meaning `design.md` Section 4.5 deliberately preserves so existing
    consumers are unaffected.

    Args:
        estimated_risk_difference: the model's estimated 96-week risk
            difference.
        reference: the unsealed `InternalReference` carrying the internal
            randomized reference's risk difference -- as the
            `"risk_difference"` entry of `quantities` (format version 2),
            or as `value` (format version 1). See
            `cohorts.sealed_reference.InternalReference`.

    Returns:
        `abs(estimated_risk_difference - reference_risk_difference)`.

    Raises:
        TypeError: if `reference` is not an `InternalReference` instance
            (e.g. a `PublishedReference`, `None`, or a plain float).
        ScaleMismatchError: if the reference's `"risk_difference"` quantity
            is not on the `"risk_difference"` scale.
    """
    if not isinstance(reference, InternalReference):
        raise TypeError(
            "risk_difference_error_vs_internal_reference() requires an "
            "InternalReference (the sealed internal randomized reference, "
            "decisions.md Section 8, item 2); got "
            f"{type(reference).__name__!r}. The 96-week risk-difference "
            "error must never be computed against a published reference "
            "(decisions.md Section 18's reference-effect policy)."
        )

    if RISK_DIFFERENCE_QUANTITY in reference.quantities:
        estimate = CausalEstimate(
            name="estimated_risk_difference",
            estimate=estimated_risk_difference,
            se=None,
            is_primary=False,
            scale="risk_difference",
        )
        return compare_to_reference(estimate, reference, RISK_DIFFERENCE_QUANTITY)

    return abs(estimated_risk_difference - reference.value)
