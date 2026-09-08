"""Unit/integration tests for `src/causal/propensity.py`.

Builds a small synthetic set of `PatientRecord`s (same hand-built-event
style as `test_selection.py`) spanning both treatment arms and confirms:

- `LogisticPropensityModel.fit_cross_fitted(patients, n_folds=5)` produces
  exactly one prediction per input patient (full coverage, no duplicates);
- for every fold, the set of patient_ids used to fit that fold's model has
  ZERO overlap with the set of patient_ids that fold's model predicted for
  (Correctness Property 3 -- cross-fitting non-leakage), checked directly
  via `CrossFittedPredictions.training_patient_ids_by_fold` /
  `.prediction_ids_for`;
- `BoostedPropensityModel.fit_cross_fitted()` raises `MissingDependencyError`
  rather than silently pretending to run gradient boosting, and its frozen
  hyperparameters match `decisions.md` Section 11 exactly.

**Validates: Requirements 7.4, 7.5; Correctness Property 3**
"""

from datetime import datetime

import pytest

from causal.propensity import (
    BoostedPropensityModel,
    LogisticPropensityModel,
    MissingDependencyError,
)
from cohorts.cohort_construction import PatientRecord
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


def _event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="baseline_cd4",
        value=350.0,
        unit="cells/mm3",
        modality="baseline",
        observed_at=ASSIGNMENT_TIME,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


def _make_patient_record(
    patient_id: str,
    *,
    treatment_arm: int,
    baseline_cd4: float,
    karnofsky_score: float,
    age: float,
    symptomatic_status: bool,
    prior_opportunistic_infection: bool,
    prior_art_exposure_days: float,
) -> PatientRecord:
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=baseline_cd4),
        _event(patient_id, concept_id="karnofsky_score", value=karnofsky_score, unit="score"),
        _event(patient_id, concept_id="age", value=age, unit="years"),
        _event(patient_id, concept_id="symptomatic_status", value=symptomatic_status, unit=None),
        _event(
            patient_id,
            concept_id="prior_opportunistic_infection",
            value=prior_opportunistic_infection,
            unit=None,
        ),
        _event(
            patient_id,
            concept_id="prior_art_exposure_days",
            value=prior_art_exposure_days,
            unit="days",
        ),
    ]
    timeline = build_patient_timeline(events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME,
            event_indicator=False,
            censoring_indicator=True,
            event_type=None,
        ),
    )


@pytest.fixture
def patients() -> tuple[PatientRecord, ...]:
    """23 synthetic patients spanning both arms, with varied covariates so
    the logistic fit has real signal to learn (not used for any accuracy
    assertion here -- only for exercising cross-fitting mechanics)."""
    records = []
    for i in range(23):
        arm = i % 2
        records.append(
            _make_patient_record(
                f"P{i:03d}",
                treatment_arm=arm,
                baseline_cd4=200.0 + 10.0 * i,
                karnofsky_score=70.0 + (i % 4) * 5.0,
                age=25.0 + (i % 10),
                symptomatic_status=bool(i % 3 == 0),
                prior_opportunistic_infection=bool(i % 5 == 0),
                prior_art_exposure_days=30.0 * (i % 4),
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# Correctness Property 3: cross-fitting non-leakage
# ---------------------------------------------------------------------------


def test_cross_fitting_has_zero_train_predict_overlap_per_fold(patients):
    result = LogisticPropensityModel(seed=0).fit_cross_fitted(patients, n_folds=5)

    assert result.n_folds == 5
    for fold in range(result.n_folds):
        training_ids = set(result.training_patient_ids_by_fold[fold])
        prediction_ids = set(result.prediction_ids_for(fold))

        # Non-leakage: no patient predicted by this fold's model was also
        # used to train it.
        assert training_ids & prediction_ids == set()

        # Every held-out patient in this fold must have been excluded from
        # its own training set (a patient cannot both train and be held out).
        for pid in prediction_ids:
            assert pid not in training_ids


def test_full_coverage_no_duplicates(patients):
    result = LogisticPropensityModel(seed=0).fit_cross_fitted(patients, n_folds=5)

    expected_ids = {patient.patient_id for patient in patients}
    predicted_ids = set(result.propensity_scores.keys())

    assert predicted_ids == expected_ids
    # Every patient assigned to exactly one fold (no duplicate predictions).
    assert len(result.fold_of) == len(patients)
    assert len(result.propensity_scores) == len(patients)


def test_every_patient_held_out_exactly_once_across_folds(patients):
    result = LogisticPropensityModel(seed=0).fit_cross_fitted(patients, n_folds=5)

    seen: list[str] = []
    for fold in range(result.n_folds):
        seen.extend(result.prediction_ids_for(fold))

    assert sorted(seen) == sorted(patient.patient_id for patient in patients)
    assert len(seen) == len(set(seen))


def test_propensity_scores_are_probabilities(patients):
    result = LogisticPropensityModel(seed=0).fit_cross_fitted(patients, n_folds=5)

    for score in result.propensity_scores.values():
        assert 0.0 <= score <= 1.0


def test_fit_cross_fitted_rejects_too_few_folds(patients):
    with pytest.raises(ValueError):
        LogisticPropensityModel().fit_cross_fitted(patients, n_folds=1)


# ---------------------------------------------------------------------------
# BoostedPropensityModel (sensitivity-only): frozen hyperparameters + refusal
# ---------------------------------------------------------------------------


def test_boosted_model_hyperparameters_match_decisions_md_section_11():
    assert BoostedPropensityModel.MAX_DEPTH == 3
    assert BoostedPropensityModel.N_ESTIMATORS == 200
    assert BoostedPropensityModel.LEARNING_RATE == 0.05


def test_boosted_model_raises_missing_dependency_error(patients):
    with pytest.raises(MissingDependencyError):
        BoostedPropensityModel().fit_cross_fitted(patients, n_folds=5)
