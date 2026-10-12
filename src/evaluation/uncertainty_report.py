"""Uncertainty_Evaluator: ensemble uncertainty and selective-prediction
evaluation (Requirement 10; task 16.1; `decisions.md` Section 15, 16, 18).

This module is the single place that composes already-existing pieces --
`uncertainty.ensemble.DeepEnsemble`'s predictions,
`uncertainty.calibration.Calibrator`, `uncertainty.support_gate.PatientSupportGate`,
and `evaluation.metrics`'s factual/selective-prediction functions -- into
one per-cohort report. It computes nothing new: every number here is one
of those modules' existing functions, called with the right inputs.

Why a plain-sequence interface (matching `evaluation.metrics`'s convention)
-------------------------------------------------------------------------

Like every other module in `evaluation/`, `evaluate_uncertainty` operates
on already-computed per-patient scalar sequences (ensemble means,
ensemble variances, labels, propensities, protocol/clinical validity
flags) rather than on `TensorBatch` or `PatientRecord` objects directly.
This keeps the module torch-free and model-agnostic -- the caller is
responsible for having already run the ensemble (`DeepEnsemble.predict`
once per patient, or the torch-batched equivalent) and extracted plain
Python values, exactly as `scripts/run_causal_analysis.py` already does
for the causal baselines and witnesses.

Epistemic uncertainty convention (Requirement 10.3)
-------------------------------------------------------------------------

`decisions.md` Section 15: ensemble variance is the uncertainty
statistic. `ensemble_variances` below is expected to be each patient's
`DeepEnsemble.predict(x)[1]` (the `variance` half of the `(mean,
variance)` pair) -- never a transformed statistic (e.g. standard
deviation) -- so it is directly comparable against
`validation_uncertainty_90th_percentile`, itself expected to be the 90th
percentile of that *same* statistic over the validation split
(`PatientSupportGate`'s own documented convention, reused unchanged
here).

Validation-only calibration (Requirement 10.4)
-------------------------------------------------------------------------

`fit_calibration_on_validation` requires its predictions/labels to be
wrapped in `uncertainty.calibration.ValidationSet` before calling
`Calibrator.fit` -- the same enforcement `Calibrator` itself already
provides; this module does not add a second check, it simply never calls
`Calibrator.fit` on anything but a `ValidationSet`.

Empty supported population (Requirement 10.7)
-------------------------------------------------------------------------

If every patient in a configuration is abstained on (`retained_flags` is
all `False`), every selective-prediction metric below is naturally `nan`
(each of `evaluation.metrics`'s selective functions already returns `nan`
on an empty retained set) rather than raising. `UncertaintyReport.notes`
records the empty-population condition explicitly by name, per
Requirement 10.7's "report ... as undefined with the empty-population
reason," so a reader does not have to infer "empty population" from a
bare `nan`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from evaluation.metrics import (
    aurcc,
    error_at_matched_coverage,
    incorrect_release_rate,
    ipcw_brier_score,
    mean_absolute_error,
    retained_coverage,
    risk_coverage_curve,
    selective_risk,
    unnecessary_abstention_rate,
)
from uncertainty.calibration import CalibrationResult, Calibrator, ValidationSet
from uncertainty.support_gate import PatientSupportGate, SupportDecision


@dataclass(frozen=True)
class UncertaintyReport:
    """`evaluate_uncertainty`'s output: one cohort's full uncertainty and
    selective-prediction summary.

    Attributes:
        support_decisions: one `SupportDecision` per patient, in input
            order (Requirement 10.6).
        retained_flags: `support_decisions[i].supported`, extracted for
            convenience (every selective-prediction function below
            consumes this form).
        calibration: the validation-fitted `CalibrationResult`
            (Requirement 10.4), or `None` if `fit_calibration_on_validation`
            was not called (e.g. this report describes a non-validation
            split).
        cd4_mae / cd8_mae: `evaluation.metrics.mean_absolute_error` over
            the supplied predicted/observed CD4 and CD8 values.
        ipcw_brier: the IPCW Brier score at the supplied horizon.
        retained_coverage_value: fraction of patients retained.
        selective_risk_value: mean loss over retained patients only.
        risk_coverage_curve_points: the full `(coverage, risk)` curve.
        aurcc_value: area under the risk-coverage curve.
        incorrect_release_rate_value: fraction of releases that were wrong.
        unnecessary_abstention_rate_value: fraction of abstentions that
            would have been correct releases.
        notes: human-readable notes, including the empty-supported-
            population condition when it applies (Requirement 10.7).
    """

    support_decisions: tuple[SupportDecision, ...]
    retained_flags: tuple[bool, ...]
    calibration: CalibrationResult | None
    cd4_mae: float
    cd8_mae: float
    ipcw_brier: float
    retained_coverage_value: float
    selective_risk_value: float
    risk_coverage_curve_points: tuple[tuple[float, float], ...]
    aurcc_value: float
    incorrect_release_rate_value: float
    unnecessary_abstention_rate_value: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def fit_calibration_on_validation(
    predictions: Sequence[float], labels: Sequence[bool], n_bins: int = 10
) -> CalibrationResult:
    """Fit `Calibrator` on validation-split `predictions`/`labels` only.

    This is the one call site in this module that touches `Calibrator`,
    and it always wraps its arguments in `ValidationSet` before calling
    `.fit()` -- there is no code path through this module that fits
    calibration on anything else (Requirement 10.4).
    """
    return Calibrator(n_bins=n_bins).fit(ValidationSet(predictions=predictions, labels=labels))


def evaluate_uncertainty(
    predictions: Sequence[float],
    labels: Sequence[bool],
    propensities: Sequence[float],
    ensemble_uncertainties: Sequence[float],
    validation_uncertainty_90th_percentile: float,
    protocol_valid_flags: Sequence[bool],
    no_hard_violation_flags: Sequence[bool],
    event_times: Sequence[float],
    event_indicators: Sequence[bool],
    horizon: float,
    cd4_predicted: Sequence[float] | None = None,
    cd4_observed: Sequence[float] | None = None,
    cd8_predicted: Sequence[float] | None = None,
    cd8_observed: Sequence[float] | None = None,
    calibration: CalibrationResult | None = None,
) -> UncertaintyReport:
    """Evaluate the frozen patient-support condition and every
    selective-prediction / factual-uncertainty metric for one cohort.

    Every sequence argument is one value per patient, in a single
    consistent patient order -- this function does not re-sort or
    re-index anything; the caller is responsible for aligning indices
    across all arguments.

    Args:
        predictions: per-patient predicted event probability (the
            quantity `selective_risk`/`risk_coverage_curve`/etc. and
            `ipcw_brier_score` are computed over).
        labels: per-patient true binary outcome.
        propensities: per-patient estimated propensity score, for the
            patient-support gate's overlap condition.
        ensemble_uncertainties: per-patient `DeepEnsemble.predict(x)[1]`
            (ensemble variance) -- the Requirement 10.3 uncertainty
            statistic.
        validation_uncertainty_90th_percentile: the validation split's
            90th percentile of that same statistic.
        protocol_valid_flags: per-patient `protocol_valid` input to
            `PatientSupportGate.evaluate`.
        no_hard_violation_flags: per-patient `no_hard_violation` input to
            `PatientSupportGate.evaluate`.
        event_times: per-patient observed (event or censoring) time, for
            `ipcw_brier_score`.
        event_indicators: per-patient event/censoring indicator, for
            `ipcw_brier_score`.
        horizon: the fixed prediction horizon (e.g. 96 weeks) for
            `ipcw_brier_score`.
        cd4_predicted / cd4_observed: paired CD4 predicted/observed
            values, for `mean_absolute_error`. `None` (both) skips the
            CD4 MAE computation, returning `nan`.
        cd8_predicted / cd8_observed: paired CD8 predicted/observed
            values, analogous to the CD4 pair.
        calibration: an already-fitted `CalibrationResult` (via
            `fit_calibration_on_validation`), or `None` if this report
            does not include calibration (e.g. a non-validation split).

    Returns:
        An `UncertaintyReport` with every patient-support decision, every
        selective-prediction metric, and the supplied calibration result.
        If every patient is abstained on, the selective metrics are
        `nan` and `notes` records the empty-supported-population
        condition explicitly (Requirement 10.7).
    """
    n = len(predictions)
    support_decisions = tuple(
        PatientSupportGate.evaluate(
            propensity=propensities[i],
            ensemble_uncertainty=ensemble_uncertainties[i],
            validation_uncertainty_90th_percentile=validation_uncertainty_90th_percentile,
            protocol_valid=protocol_valid_flags[i],
            no_hard_violation=no_hard_violation_flags[i],
        )
        for i in range(n)
    )
    retained_flags = tuple(decision.supported for decision in support_decisions)

    cd4_mae = (
        mean_absolute_error(cd4_predicted, cd4_observed)
        if cd4_predicted is not None and cd4_observed is not None
        else float("nan")
    )
    cd8_mae = (
        mean_absolute_error(cd8_predicted, cd8_observed)
        if cd8_predicted is not None and cd8_observed is not None
        else float("nan")
    )
    ipcw_brier = ipcw_brier_score(predictions, event_times, event_indicators, horizon)

    notes: list[str] = []
    if n > 0 and not any(retained_flags):
        notes.append(
            "empty_supported_population: every patient was abstained on; every "
            "selective-prediction metric below is undefined (nan) for this reason "
            "(Requirement 10.7)"
        )

    curve = tuple(risk_coverage_curve(predictions, labels, ensemble_uncertainties))

    return UncertaintyReport(
        support_decisions=support_decisions,
        retained_flags=retained_flags,
        calibration=calibration,
        cd4_mae=cd4_mae,
        cd8_mae=cd8_mae,
        ipcw_brier=ipcw_brier,
        retained_coverage_value=retained_coverage(retained_flags),
        selective_risk_value=selective_risk(predictions, labels, retained_flags),
        risk_coverage_curve_points=curve,
        aurcc_value=aurcc(predictions, labels, ensemble_uncertainties),
        incorrect_release_rate_value=incorrect_release_rate(predictions, labels, retained_flags),
        unnecessary_abstention_rate_value=unnecessary_abstention_rate(
            predictions, labels, retained_flags
        ),
        notes=tuple(notes),
    )


def error_at_matched_coverage_for_gated_vs_ungated(
    gated_predictions: Sequence[float],
    gated_confidence_scores: Sequence[float],
    ungated_predictions: Sequence[float],
    ungated_confidence_scores: Sequence[float],
    labels: Sequence[bool],
    target_coverage: float,
) -> tuple[float, float]:
    """Error at matched coverage for a gated vs. ungated configuration
    (Requirement 10.5).

    A thin, named wrapper around `evaluation.metrics.error_at_matched_coverage`'s
    two-model form, so a caller comparing a gated and an ungated
    configuration does not have to remember which positional argument is
    which model.

    Returns:
        `(risk_gated, risk_ungated)`, each model's selective risk at its
        own nearest achievable coverage to `target_coverage`.
    """
    risk_gated, risk_ungated = error_at_matched_coverage(
        gated_predictions,
        labels,
        gated_confidence_scores,
        target_coverage,
        predictions_b=ungated_predictions,
        confidence_scores_b=ungated_confidence_scores,
    )
    return risk_gated, risk_ungated
