"""Propensity estimation with cross-fitting (Requirements 7.4, 7.5;
Correctness Property 3; `decisions.md` Section 11 "Propensity modeling").

`decisions.md` Section 11 freezes two propensity specifications:

- **primary model**: logistic regression (`LogisticPropensityModel` below);
- **sensitivity model**: gradient boosting, fixed config `XGBoost`,
  `max_depth=3`, `n_estimators=200`, `learning_rate=0.05`, seed-locked
  (`BoostedPropensityModel` below) -- "sensitivity analysis only, never
  primary."

Both models share one interface, `PropensityModel.fit_cross_fitted(data,
n_folds=5) -> CrossFittedPredictions`, so causal baselines (task 7.2) and
overlap diagnostics (task 7.1) can consume either without caring which
concrete model produced the scores.

Dependency scope note (read before extending this module)
------------------------------------------------------------

`pyproject.toml` currently declares only `pydantic` and `PyYAML` as runtime
dependencies -- no `numpy`/`scipy`/`scikit-learn`/`xgboost`. Adding any of
those is a real dependency decision this task does not make unilaterally.
Consequently:

- `LogisticPropensityModel` is a **minimal, from-scratch logistic
  regression** implemented in pure Python, fit to convergence by Newton /
  IRLS (`causal.solvers.fit_logistic_newton`) from a zero start: stop when
  the Newton step's max |component| is below 1e-8, at most 100 iterations,
  frozen in `decisions.md` Section 26 (task 7.9). Each fold's
  `ConvergenceDiagnostic` is returned in `CrossFittedPredictions.convergence`,
  and a fold that does not converge raises `SolverConvergenceError`. Until
  task 7.8 this was 500 fixed gradient-descent steps with no tolerance,
  which left the 14-column fit unconverged (gradient norm ~1e-2).
- `BoostedPropensityModel` stores the exact frozen hyperparameters
  (`MAX_DEPTH=3`, `N_ESTIMATORS=200`, `LEARNING_RATE=0.05`) as class
  constants, satisfies the shared `PropensityModel` protocol structurally,
  but its `fit_cross_fitted()` deliberately raises
  `MissingDependencyError` rather than silently substituting a different
  algorithm and calling it "gradient boosting." It is explicitly a
  **sensitivity-only** model (`decisions.md` Section 11: "never primary"),
  so the pipeline's primary path never depends on it. Revisit this once
  `xgboost` is added as an approved project dependency -- at that point
  `fit_cross_fitted()` should be implemented against the real library
  using exactly the frozen hyperparameters already stored here.

Cross-fitting
--------------

`fit_cross_fitted(data, n_folds=5)` partitions `data` into `n_folds` folds
deterministically (patients sorted by `patient_id`, then assigned
round-robin by `index % n_folds`), and for each fold trains a fresh model
on every *other* fold's patients before predicting only that fold's
held-out patients. The returned `CrossFittedPredictions` exposes, per fold,
both the training patient IDs and the held-out (prediction) patient IDs, so
non-leakage (Correctness Property 3: no patient's prediction came from a
model trained on data including that patient) can be verified directly
rather than only inferred.

Covariate list (Section 13 adjustment set)
-------------------------------------------

The propensity model adjusts for the full frozen `decisions.md` Section 13
adjustment set: thirteen covariates, encoded as fourteen columns
(`COVARIATE_FIELDS`). Until task 7.5 it used only Section 9's six
observationalization inputs, which did not conform to Section 13 (see the
Section 26 amendment of 2026-09-30, item 2). Section 9's risk score in
`cohorts.selection` is a separate list and is unchanged.

Each Section 13 name is resolved to its stored `static_covariates` key(s)
through `causal.adjustment_set.SECTION_13_ADJUSTMENT_SET`, never by name
equality (`weight` is stored as `weight_kg`, `intravenous_drug_use_history`
as `iv_drug_use_history`, and prior ART as `prior_art_exposure_days`).

Columns, in order, with their encoding and missing-value rule. "Missing"
means no mapped key holds a non-`None` value.

==  ===============================  ==================================  ===========================
#   column                           encoding                            missing / unusable value
==  ===============================  ==================================  ===========================
1   `age`                            continuous, z-scored                `0.0` (the fold mean)
2   `weight`                         continuous (kg), z-scored           `0.0` (the fold mean)
3   `sex_female`                     `1.0` if `"F"`, `0.0` if `"M"`      `0.0`; any value other than
                                     (reference level `"M"`)             `"M"`/`"F"` is missing
4   `race_ethnicity_non_white`       `1.0` for any fitted level other    `0.0`; an unseen level (not
                                     than `"white"`, `0.0` for           in the fitted category set)
                                     `"white"` (reference level)         is also `0.0`
5   `hemophilia`                     boolean, `1.0`/`0.0`                `0.0`
6   `homosexual_activity`            boolean, `1.0`/`0.0`                `0.0`
7   `intravenous_drug_use_history`   boolean, `1.0`/`0.0`                `0.0`
8   `karnofsky_score`                continuous, z-scored                `0.0` (the fold mean)
9   `prior_opportunistic_infection`  boolean, `1.0`/`0.0`                `0.0`
10  `symptomatic_status`             boolean, `1.0`/`0.0`                `0.0`
11  `baseline_cd4`                   continuous, z-scored                `0.0` (the fold mean)
12  `baseline_cd8`                   continuous, z-scored                `0.0` (the fold mean)
13  `prior_art`                      exposure indicator: `1.0` if the    `0.0`
                                     stored duration is `> 0` days
14  `prior_art_duration`             duration in days (zeros included),  `0.0` (the fold mean)
                                     z-scored
==  ===============================  ==================================  ===========================

Columns 13-14 together encode Section 13's "prior antiretroviral exposure
*and* duration": the indicator separates naive from experienced patients,
and the standardized duration carries how long. Column 13 is derived
exactly as `cohorts.selection._prior_art_indicator` derives it.

Details of each rule:

- **continuous** (z-scored): `(raw - mean) / stdev` with `mean` and the
  population `stdev` fit on the training fold's observed numeric values
  only. Missing or non-numeric values are excluded from the fit and encode
  as `0.0`, the fold mean. A column with fewer than two observed values, or
  zero stdev, encodes as `0.0` for everyone.
- **boolean**: Python truthiness of the stored value, as `1.0`/`0.0`. The
  loader stores real `bool`s for all five of these fields.
- **binary_category** (`sex`, `race_ethnicity`): the category set is the
  set of distinct stored string values seen in the training fold (for
  `sex`, only values in the closed set `{"M", "F"}` count). A value encodes
  as `1.0` only if it is in that fitted set and is not the reference level.
  The reference level, a value not seen in the training fold, and a missing
  value all encode as `0.0`, so an unseen category contributes like the
  reference level (the usual drop-first, ignore-unknown convention) rather
  than raising. `race_ethnicity` is a free string in the loader schema. The
  real ACTG175 v3.2 source codes it as exactly two levels, `"white"` and
  `"non-white"`, so the white / non-white contrast is lossless there. If a
  source ever supplied more than one non-white level, they would all
  collapse into this single column.
- **exposure_and_duration**: the first mapped key holding a numeric value
  gives the duration in days. Column 13 is `1.0` if it is `> 0`; column 14
  is that duration z-scored like any continuous column.

Leakage safety: every parameter above (means, stdevs, and category sets)
is fit by `_fit_standardization` on the patients it is handed, which in
cross-fitting is always the training folds only, never the fold being
predicted. `models.dataset` calls the same function on Development_Split
rows only. Only baseline keys are read (see `causal.adjustment_set`); no
week-20 / week-96 measurement, event-derived field, or post-time-zero
censoring information enters the feature vector.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from causal.adjustment_set import (
    ADJUSTMENT_COVARIATES_BY_NAME,
    ENCODED_COLUMNS,
    SECTION_13_ADJUSTMENT_SET,
    AdjustmentCovariate,
    resolve_raw_value,
)
from causal.solvers import ConvergenceDiagnostic, SolverConvergenceError, fit_logistic_newton
from cohorts.cohort_construction import PatientRecord

# ---------------------------------------------------------------------------
# Covariate extraction (shared by both models)
# ---------------------------------------------------------------------------

#: The prior-ART Section 13 covariate (exposure indicator + duration).
_PRIOR_ART = ADJUSTMENT_COVARIATES_BY_NAME["prior_antiretroviral_exposure_and_duration"]

#: The standardized (z-scored) column names, one per continuous covariate
#: plus the prior-ART duration column, in `COVARIATE_FIELDS` order.
_CONTINUOUS_FIELDS: tuple[str, ...] = tuple(
    covariate.columns[0] if covariate.encoding == "continuous" else covariate.columns[1]
    for covariate in SECTION_13_ADJUSTMENT_SET
    if covariate.encoding in ("continuous", "exposure_and_duration")
)

#: `{z-scored column: its continuous Section 13 covariate}` (the prior-ART
#: duration column is handled separately by `_prior_art_duration`).
_COVARIATE_BY_CONTINUOUS_COLUMN: dict[str, AdjustmentCovariate] = {
    covariate.columns[0]: covariate
    for covariate in SECTION_13_ADJUSTMENT_SET
    if covariate.encoding == "continuous"
}

#: The fourteen encoded columns both propensity models use, in order (see
#: the module docstring's table). `causal.baselines.adjusted_cox`,
#: `models.non_neural_baseline.COEFFICIENT_NAMES`, and
#: `models.dataset.TensorBatch.baseline` all follow this order.
COVARIATE_FIELDS: tuple[str, ...] = ENCODED_COLUMNS


def _to_float(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    try:
        return float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _covariate_numeric(patient: PatientRecord, covariate: AdjustmentCovariate) -> float | None:
    """A covariate's stored numeric value via the Section 13 mapping, or `None`."""
    return _to_float(resolve_raw_value(patient.timeline.static_covariates, covariate))


def _covariate_level(patient: PatientRecord, covariate: AdjustmentCovariate) -> str | None:
    """A categorical covariate's stored level, or `None` if missing/unusable.

    A non-string value, or a value outside `covariate.allowed_levels` when
    that set is closed, is treated as missing.
    """
    raw_value = resolve_raw_value(patient.timeline.static_covariates, covariate)
    if not isinstance(raw_value, str):
        return None
    if covariate.allowed_levels is not None and raw_value not in covariate.allowed_levels:
        return None
    return raw_value


def _prior_art_duration(patient: PatientRecord) -> float | None:
    """Prior-ART duration in days: the first mapped key holding a number."""
    static_covariates = patient.timeline.static_covariates
    for concept_id in _PRIOR_ART.concept_ids:
        value = _to_float(static_covariates.get(concept_id))
        if value is not None:
            return value
    return None


def _prior_art_indicator(patient: PatientRecord) -> float:
    """Derive the `prior_art` indicator (mirrors `cohorts.selection._prior_art_indicator`)."""
    duration = _prior_art_duration(patient)
    return 1.0 if duration is not None and duration > 0 else 0.0


def _boolean_value(patient: PatientRecord, covariate: AdjustmentCovariate) -> float:
    """A boolean covariate as `1.0`/`0.0` (missing = `0.0`)."""
    raw_value = resolve_raw_value(patient.timeline.static_covariates, covariate)
    if raw_value is None:
        return 0.0
    return 1.0 if bool(raw_value) else 0.0


def _continuous_raw(patient: PatientRecord, column: str) -> float | None:
    """The raw numeric value behind a z-scored column in `_CONTINUOUS_FIELDS`."""
    if column == _PRIOR_ART.columns[1]:
        return _prior_art_duration(patient)
    return _covariate_numeric(patient, _COVARIATE_BY_CONTINUOUS_COLUMN[column])


@dataclass(frozen=True)
class _StandardizationParams:
    """Every fitted encoding parameter, fit on one training fold only.

    Attributes:
        means / stdevs: per z-scored column (`_CONTINUOUS_FIELDS`).
        category_sets: per `binary_category` Section 13 name, the distinct
            levels seen in the fitting patients.
    """

    means: Mapping[str, float]
    stdevs: Mapping[str, float]
    category_sets: Mapping[str, frozenset[str]]


def _fit_standardization(patients: Sequence[PatientRecord]) -> _StandardizationParams:
    """Fit every encoding parameter on `patients` only.

    Fits the mean and population stdev of each z-scored column over its
    observed numeric values, and the category set of each categorical
    covariate. A column with fewer than two observed numeric values, or a
    computed stdev of `0.0`, standardizes to `0.0` for every patient at
    transform time (see `_standardized_value`) rather than dividing by zero.
    """
    means: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    for field_name in _CONTINUOUS_FIELDS:
        values = [
            value
            for value in (_continuous_raw(patient, field_name) for patient in patients)
            if value is not None
        ]
        if not values:
            means[field_name] = 0.0
            stdevs[field_name] = 0.0
            continue
        means[field_name] = statistics.fmean(values)
        stdevs[field_name] = statistics.pstdev(values) if len(values) > 1 else 0.0

    category_sets: dict[str, frozenset[str]] = {}
    for covariate in SECTION_13_ADJUSTMENT_SET:
        if covariate.encoding != "binary_category":
            continue
        category_sets[covariate.name] = frozenset(
            level
            for level in (_covariate_level(patient, covariate) for patient in patients)
            if level is not None
        )
    return _StandardizationParams(means=means, stdevs=stdevs, category_sets=category_sets)


def _standardized_value(
    patient: PatientRecord, field_name: str, params: _StandardizationParams
) -> float:
    raw_value = _continuous_raw(patient, field_name)
    if raw_value is None:
        return 0.0
    mean = params.means.get(field_name, 0.0)
    stdev = params.stdevs.get(field_name, 0.0)
    return (raw_value - mean) / stdev if stdev > 0 else 0.0


def _category_indicator(
    patient: PatientRecord, covariate: AdjustmentCovariate, params: _StandardizationParams
) -> float:
    """`1.0` for a fitted, non-reference level; `0.0` for the reference
    level, an unseen level, or a missing value."""
    level = _covariate_level(patient, covariate)
    if level is None or level == covariate.reference_level:
        return 0.0
    return 1.0 if level in params.category_sets.get(covariate.name, frozenset()) else 0.0


def _feature_vector(patient: PatientRecord, params: _StandardizationParams) -> list[float]:
    """The fixed-order, fold-safe feature vector for one patient.

    One entry per `COVARIATE_FIELDS` column. Every fitted quantity comes
    from `params`, which is always fit on the *training* fold, never on
    the fold being predicted. Encodings are in the module docstring.
    """
    row: list[float] = []
    for covariate in SECTION_13_ADJUSTMENT_SET:
        if covariate.encoding == "continuous":
            row.append(_standardized_value(patient, covariate.columns[0], params))
        elif covariate.encoding == "boolean":
            row.append(_boolean_value(patient, covariate))
        elif covariate.encoding == "binary_category":
            row.append(_category_indicator(patient, covariate, params))
        else:  # exposure_and_duration
            row.append(_prior_art_indicator(patient))
            row.append(_standardized_value(patient, covariate.columns[1], params))
    return row


# ---------------------------------------------------------------------------
# Cross-fitted prediction result and shared cross-fitting routine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossFittedPredictions:
    """The result of `PropensityModel.fit_cross_fitted()`.

    Attributes:
        propensity_scores: `{patient_id: predicted propensity}`, covering
            every patient in the input data exactly once -- each score was
            produced by a model trained on the *other* folds only.
        fold_of: `{patient_id: fold index}` -- the fold in which this
            patient was held out (and therefore whose score in
            `propensity_scores` came from a model trained without them).
        training_patient_ids_by_fold: `{fold index: training patient_ids}`
            -- the exact set of patient IDs used to fit that fold's model.
            Exists so cross-fitting non-leakage (Correctness Property 3)
            can be checked directly: for every fold, `set(training_patient_ids_by_fold[fold])`
            must be disjoint from `set(prediction_ids_for(fold))`.
        n_folds: the number of folds used.
        convergence: `{fold index: ConvergenceDiagnostic}` for models that
            report one (`LogisticPropensityModel` always does); empty
            otherwise.
    """

    propensity_scores: dict[str, float]
    fold_of: dict[str, int]
    training_patient_ids_by_fold: dict[int, tuple[str, ...]]
    n_folds: int
    convergence: dict[int, ConvergenceDiagnostic] = field(default_factory=dict)

    def prediction_ids_for(self, fold: int) -> tuple[str, ...]:
        """The patient IDs held out (and predicted) in `fold`, sorted."""
        return tuple(sorted(pid for pid, f in self.fold_of.items() if f == fold))


def _assign_folds(patient_ids: Sequence[str], n_folds: int) -> dict[str, int]:
    """Deterministic fold assignment: sort by `patient_id`, then round-robin."""
    return {pid: index % n_folds for index, pid in enumerate(sorted(patient_ids))}


def _cross_fit(
    data: Sequence[PatientRecord],
    n_folds: int,
    fit_predict_fold: Callable[
        [Sequence[PatientRecord], Sequence[PatientRecord]], dict[str, float]
    ],
) -> CrossFittedPredictions:
    """Shared k-fold cross-fitting routine used by both propensity models.

    For each fold: fit on every *other* fold's patients, predict only this
    fold's held-out patients. `fit_predict_fold(train_patients,
    holdout_patients)` is expected to return `{patient_id: score}` for
    exactly `holdout_patients`.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for cross-fitting, got {n_folds}")

    patients_by_id = {patient.patient_id: patient for patient in data}
    if len(patients_by_id) != len(data):
        raise ValueError("data contains duplicate patient_ids")

    fold_of = _assign_folds(list(patients_by_id), n_folds)

    training_by_fold: dict[int, tuple[str, ...]] = {}
    propensity_scores: dict[str, float] = {}

    for fold in range(n_folds):
        holdout_ids = sorted(pid for pid, f in fold_of.items() if f == fold)
        train_ids = sorted(pid for pid, f in fold_of.items() if f != fold)
        training_by_fold[fold] = tuple(train_ids)

        train_patients = [patients_by_id[pid] for pid in train_ids]
        holdout_patients = [patients_by_id[pid] for pid in holdout_ids]
        if not holdout_patients:
            continue

        fold_predictions = fit_predict_fold(train_patients, holdout_patients)
        propensity_scores.update(fold_predictions)

    return CrossFittedPredictions(
        propensity_scores=propensity_scores,
        fold_of=fold_of,
        training_patient_ids_by_fold=training_by_fold,
        n_folds=n_folds,
    )


# ---------------------------------------------------------------------------
# Shared protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PropensityModel(Protocol):
    """Shared interface for `LogisticPropensityModel` and `BoostedPropensityModel`."""

    def fit_cross_fitted(
        self, data: Sequence[PatientRecord], n_folds: int = 5
    ) -> CrossFittedPredictions:
        """Cross-fit this model on `data`, partitioned into `n_folds` folds.

        Returns a `CrossFittedPredictions` covering every patient in `data`
        exactly once, where no patient's score came from a model trained on
        data including that patient (Correctness Property 3).
        """
        ...


# ---------------------------------------------------------------------------
# LogisticPropensityModel (primary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FittedLogistic:
    weights: tuple[float, ...]
    bias: float
    standardization: _StandardizationParams
    diagnostic: ConvergenceDiagnostic | None = None


def _sigmoid(x: float) -> float:
    # Guard against overflow for large-magnitude logits.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class LogisticPropensityModel:
    """Primary propensity model: minimal from-scratch logistic regression.

    See module docstring for why this is pure Python (no numpy/scikit-learn
    dependency) and why that is an appropriate, deliberately minimal choice
    for this covariate set and patient count -- not a general-purpose ML
    library replacement.

    Deterministic training configuration (`decisions.md` Section 26, task
    7.9):

    - zero-initialized weights and bias, so no random draw is needed for this
      convex objective; `seed` is accepted for interface parity with
      `BoostedPropensityModel` and does not affect the fit;
    - Newton / IRLS on mean binary cross-entropy against `treatment_arm`
      (`causal.solvers.fit_logistic_newton`), stopping when the step's max
      |component| is below `causal.solvers.SOLVER_TOLERANCE` (1e-8), capped
      at `SOLVER_MAX_ITERATIONS` (100). A fold that does not converge raises
      `SolverConvergenceError`; its coefficients are never used.
    """

    def __init__(self, seed: int = 0) -> None:
        #: Accepted for interface parity with `BoostedPropensityModel` and
        #: for documentation of intended reproducibility; unused by this
        #: model's zero-initialized, deterministic Newton fit.
        self.seed = seed

    def _fit(self, train_patients: Sequence[PatientRecord]) -> _FittedLogistic:
        if not train_patients:
            raise ValueError("Cannot fit LogisticPropensityModel on zero training patients")

        standardization = _fit_standardization(train_patients)
        features = [_feature_vector(patient, standardization) for patient in train_patients]
        labels = [float(patient.treatment_arm) for patient in train_patients]

        bias, weights, diagnostic = fit_logistic_newton(features, labels)
        if not diagnostic.converged:
            raise SolverConvergenceError(
                f"LogisticPropensityModel on {len(train_patients)} training patients", diagnostic
            )
        return _FittedLogistic(
            weights=tuple(weights),
            bias=bias,
            standardization=standardization,
            diagnostic=diagnostic,
        )

    def _predict(
        self, fitted: _FittedLogistic, patients: Sequence[PatientRecord]
    ) -> dict[str, float]:
        predictions: dict[str, float] = {}
        for patient in patients:
            feature_row = _feature_vector(patient, fitted.standardization)
            logit = fitted.bias + sum(
                w * x for w, x in zip(fitted.weights, feature_row, strict=True)
            )
            predictions[patient.patient_id] = _sigmoid(logit)
        return predictions

    def fit_cross_fitted(
        self, data: Sequence[PatientRecord], n_folds: int = 5
    ) -> CrossFittedPredictions:
        diagnostics: list[ConvergenceDiagnostic] = []

        def fit_predict_fold(
            train_patients: Sequence[PatientRecord], holdout_patients: Sequence[PatientRecord]
        ) -> dict[str, float]:
            fitted = self._fit(train_patients)
            if fitted.diagnostic is not None:
                diagnostics.append(fitted.diagnostic)
            return self._predict(fitted, holdout_patients)

        result = _cross_fit(data, n_folds, fit_predict_fold)
        # `_cross_fit` visits folds in ascending order and skips only folds
        # with no held-out patient, so diagnostics align with these folds.
        fitted_folds = sorted(set(result.fold_of.values()))
        if len(diagnostics) != len(fitted_folds):
            return result
        return replace(result, convergence=dict(zip(fitted_folds, diagnostics, strict=True)))


# ---------------------------------------------------------------------------
# BoostedPropensityModel (sensitivity only)
# ---------------------------------------------------------------------------


class MissingDependencyError(RuntimeError):
    """Raised when `BoostedPropensityModel` is used without `xgboost` installed.

    `decisions.md` Section 11 freezes the sensitivity propensity model as
    gradient boosting via `XGBoost` with fixed hyperparameters
    (`max_depth=3`, `n_estimators=200`, `learning_rate=0.05`). `xgboost` is
    not currently an approved project dependency (see `pyproject.toml`), so
    this model deliberately refuses to run rather than silently substituting
    a different algorithm and calling it gradient boosting.
    """

    def __init__(self) -> None:
        super().__init__(
            "BoostedPropensityModel.fit_cross_fitted() requires xgboost, which "
            "is not currently a project dependency (see pyproject.toml). This "
            "model is the decisions.md Section 11 sensitivity-only propensity "
            "specification (never primary); add xgboost as an approved "
            "dependency and implement this method against it -- using exactly "
            "MAX_DEPTH, N_ESTIMATORS, and LEARNING_RATE below -- before using "
            "BoostedPropensityModel for a sensitivity run."
        )


class BoostedPropensityModel:
    """Sensitivity-only propensity model: frozen gradient-boosting hyperparameters.

    `decisions.md` Section 11: "sensitivity model: gradient boosting (fixed
    config: XGBoost, max_depth=3, n_estimators=200, learning_rate=0.05,
    seed-locked) -- ... WILL, as sensitivity analysis only, never primary."

    `xgboost` is not an approved runtime dependency yet (see module
    docstring), so `fit_cross_fitted()` raises `MissingDependencyError`
    rather than pretending to run gradient boosting via a substitute
    algorithm. The frozen hyperparameters are still stored here as class
    constants so they are available for documentation/provenance and ready
    to use once `xgboost` is added.
    """

    #: Frozen `decisions.md` Section 11 hyperparameters -- do not change
    #: without a documented amendment.
    MAX_DEPTH: int = 3
    N_ESTIMATORS: int = 200
    LEARNING_RATE: float = 0.05

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit_cross_fitted(
        self, data: Sequence[PatientRecord], n_folds: int = 5
    ) -> CrossFittedPredictions:
        raise MissingDependencyError()
