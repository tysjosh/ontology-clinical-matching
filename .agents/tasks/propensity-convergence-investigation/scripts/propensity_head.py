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
  regression** implemented in pure Python: batch gradient descent on
  binary cross-entropy over a small, fixed covariate list, with
  zero-initialized weights (deterministic without needing any random draw)
  and a fixed learning rate/iteration count. This is appropriate for the
  handful of covariates and moderate patient counts ACTG175 involves. It
  is **not** a general-purpose ML library replacement -- it does not do
  regularization, line search, or convergence diagnostics beyond a fixed
  iteration budget. If this ever needs to scale beyond a handful of
  covariates or patients, revisit the numpy/scikit-learn dependency
  question explicitly rather than growing this loop further.
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

Covariate list
---------------

Six baseline covariates, matching the fields `cohorts.selection.py` already
uses for the frozen observationalization risk score (`decisions.md`
Section 9's `r_i` inputs), read from
`PatientRecord.timeline.static_covariates`:

- `baseline_cd4`, `karnofsky_score`, `age` (continuous, standardized
  in-fold using the training fold's own mean/stdev -- never test-fold
  statistics, so no cross-fold leakage enters the standardization either);
- `symptomatic_status`, `prior_opportunistic_infection` (raw boolean
  indicators, coerced to `1.0`/`0.0`);
- `prior_art` (derived indicator: `1.0` if the first present, numeric
  value among `"prior_art_exposure_days"` /
  `"prior_antiretroviral_exposure_duration"` is `> 0`, else `0.0` --
  mirrors `cohorts.selection._prior_art_indicator`'s derivation exactly).

A missing or non-numeric raw value for any field defaults to `0.0` (for
continuous fields, this only affects that patient's own feature value, not
the fold's fitted mean/stdev, since missing values are excluded from the
mean/stdev computation).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cohorts.cohort_construction import PatientRecord

# ---------------------------------------------------------------------------
# Covariate extraction (shared by both models)
# ---------------------------------------------------------------------------

#: Continuous covariate fields, standardized (z-scored) per training fold.
_CONTINUOUS_FIELDS: tuple[str, ...] = ("baseline_cd4", "karnofsky_score", "age")

#: Raw boolean-indicator concept_ids, read directly (never standardized).
_SYMPTOMATIC_CONCEPT_ID = "symptomatic_status"
_PRIOR_OI_CONCEPT_ID = "prior_opportunistic_infection"

#: `concept_id`s checked, in order, when deriving the `prior_art` indicator
#: (mirrors `cohorts.selection._PRIOR_ART_CONCEPT_IDS`).
_PRIOR_ART_CONCEPT_IDS: tuple[str, ...] = (
    "prior_art_exposure_days",
    "prior_antiretroviral_exposure_duration",
)

#: The full, fixed covariate list both propensity models use, in order.
#: Exposed for tests/documentation; not itself consumed positionally by
#: callers outside this module.
COVARIATE_FIELDS: tuple[str, ...] = (
    *_CONTINUOUS_FIELDS,
    _SYMPTOMATIC_CONCEPT_ID,
    _PRIOR_OI_CONCEPT_ID,
    "prior_art",
)


def _raw_numeric(patient: PatientRecord, field_name: str) -> float | None:
    """Read a raw numeric covariate, or `None` if missing/non-numeric."""
    raw_value = patient.timeline.static_covariates.get(field_name)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _boolean_indicator(patient: PatientRecord, concept_id: str) -> float:
    """A raw boolean-indicator covariate, coerced to `1.0`/`0.0` (missing = `0.0`)."""
    raw_value = patient.timeline.static_covariates.get(concept_id)
    if raw_value is None:
        return 0.0
    return 1.0 if bool(raw_value) else 0.0


def _prior_art_indicator(patient: PatientRecord) -> float:
    """Derive the `prior_art` indicator (mirrors `cohorts.selection._prior_art_indicator`)."""
    static_covariates = patient.timeline.static_covariates
    for concept_id in _PRIOR_ART_CONCEPT_IDS:
        if concept_id not in static_covariates:
            continue
        raw_value = static_covariates[concept_id]
        if raw_value is None:
            continue
        try:
            exposure = float(raw_value)
        except (TypeError, ValueError):
            continue
        return 1.0 if exposure > 0 else 0.0
    return 0.0


@dataclass(frozen=True)
class _StandardizationParams:
    """Per-continuous-field `(mean, stdev)`, fit on one training fold only."""

    means: Mapping[str, float]
    stdevs: Mapping[str, float]


def _fit_standardization(patients: Sequence[PatientRecord]) -> _StandardizationParams:
    """Fit per-field mean/stdev for `_CONTINUOUS_FIELDS` on `patients` only.

    A field with fewer than two observed numeric values, or a computed
    stdev of `0.0`, standardizes to `0.0` for every patient at transform
    time (see `_standardized_value`) rather than dividing by zero.
    """
    means: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    for field_name in _CONTINUOUS_FIELDS:
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
    return _StandardizationParams(means=means, stdevs=stdevs)


def _standardized_value(
    patient: PatientRecord, field_name: str, params: _StandardizationParams
) -> float:
    raw_value = _raw_numeric(patient, field_name)
    if raw_value is None:
        return 0.0
    mean = params.means.get(field_name, 0.0)
    stdev = params.stdevs.get(field_name, 0.0)
    return (raw_value - mean) / stdev if stdev > 0 else 0.0


def _feature_vector(patient: PatientRecord, params: _StandardizationParams) -> list[float]:
    """The fixed-order, fold-safe feature vector for one patient.

    Continuous fields are standardized using `params` (always fit on the
    *training* fold, never on the fold being predicted); the three
    indicator fields are raw `1.0`/`0.0` values, never standardized.
    """
    continuous = [_standardized_value(patient, name, params) for name in _CONTINUOUS_FIELDS]
    indicators = [
        _boolean_indicator(patient, _SYMPTOMATIC_CONCEPT_ID),
        _boolean_indicator(patient, _PRIOR_OI_CONCEPT_ID),
        _prior_art_indicator(patient),
    ]
    return [*continuous, *indicators]


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
    """

    propensity_scores: dict[str, float]
    fold_of: dict[str, int]
    training_patient_ids_by_fold: dict[int, tuple[str, ...]]
    n_folds: int

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

    Fixed, deterministic training configuration (not a `decisions.md`
    frozen parameter -- these are this implementation's own defaults, kept
    fixed so `fit_cross_fitted` is reproducible):

    - zero-initialized weights and bias (deterministic; no random draw is
      needed for a convex objective like logistic regression, so `seed` is
      accepted for interface parity with `BoostedPropensityModel` but does
      not affect this model's fit);
    - fixed learning rate (`LEARNING_RATE`) and iteration count
      (`N_ITERATIONS`), used for plain batch gradient descent on binary
      cross-entropy against `treatment_arm`.
    """

    LEARNING_RATE: float = 0.1
    N_ITERATIONS: int = 500

    def __init__(self, seed: int = 0) -> None:
        #: Accepted for interface parity with `BoostedPropensityModel` and
        #: for documentation of intended reproducibility; unused by this
        #: model's zero-initialized, deterministic gradient descent.
        self.seed = seed

    def _fit(self, train_patients: Sequence[PatientRecord]) -> _FittedLogistic:
        if not train_patients:
            raise ValueError("Cannot fit LogisticPropensityModel on zero training patients")

        standardization = _fit_standardization(train_patients)
        features = [_feature_vector(patient, standardization) for patient in train_patients]
        labels = [float(patient.treatment_arm) for patient in train_patients]

        n_features = len(COVARIATE_FIELDS)
        weights = [0.0] * n_features
        bias = 0.0
        n = len(train_patients)

        for _ in range(self.N_ITERATIONS):
            weight_gradients = [0.0] * n_features
            bias_gradient = 0.0
            for feature_row, label in zip(features, labels, strict=True):
                logit = bias + sum(w * x for w, x in zip(weights, feature_row, strict=True))
                prediction = _sigmoid(logit)
                error = prediction - label
                bias_gradient += error
                for j, x in enumerate(feature_row):
                    weight_gradients[j] += error * x

            bias -= self.LEARNING_RATE * (bias_gradient / n)
            weights = [
                w - self.LEARNING_RATE * (grad / n)
                for w, grad in zip(weights, weight_gradients, strict=True)
            ]

        return _FittedLogistic(weights=tuple(weights), bias=bias, standardization=standardization)

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
        def fit_predict_fold(
            train_patients: Sequence[PatientRecord], holdout_patients: Sequence[PatientRecord]
        ) -> dict[str, float]:
            fitted = self._fit(train_patients)
            return self._predict(fitted, holdout_patients)

        return _cross_fit(data, n_folds, fit_predict_fold)


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
