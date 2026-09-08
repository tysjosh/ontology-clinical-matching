"""Unit tests for `src/models/non_neural_baseline.py` (`CoxBaseline`).

Reuses the same synthetic-cohort-with-known-protective-effect pattern as
`tests/unit/test_baselines.py` (exponential survival times, treatment
halves the hazard) to confirm:

- `.fit()` runs without error and produces finite, non-NaN coefficients
  ("training converges");
- the fitted `treatment_arm` coefficient has the correct (negative,
  protective) sign;
- per-patient risk scores can be retrieved, are finite, and vary across
  patients ("metrics are computed").

**Validates: Requirements 8.1**
"""

import math
import random
from datetime import datetime, timedelta

import pytest

from cohorts.cohort_construction import PatientRecord
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from models.non_neural_baseline import COEFFICIENT_NAMES, CoxBaseline, fit_cox_baseline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)

TRUE_HAZARD_RATIO = 0.5
BASELINE_HAZARD = 0.01  # per week
ADMINISTRATIVE_CENSORING_WEEKS = 120.0
N_PATIENTS = 250


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
    event_time_weeks: float,
    event_indicator: bool,
) -> PatientRecord:
    event_time = ASSIGNMENT_TIME + timedelta(weeks=event_time_weeks)
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
        _event(
            patient_id,
            concept_id="zidovudine" if treatment_arm == 0 else "zidovudine+didanosine",
            value=1.0,
            unit=None,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
    ]
    timeline = build_patient_timeline(events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=event_time,
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="death" if event_indicator else None,
        ),
    )


def _simulate_patients(seed: int = 12345, n: int = N_PATIENTS) -> tuple[PatientRecord, ...]:
    """Same exponential-time simulation as `test_baselines.py`: treatment
    halves the hazard (a known protective effect), administrative
    censoring at `ADMINISTRATIVE_CENSORING_WEEKS`."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        arm = i % 2
        hazard = BASELINE_HAZARD * (TRUE_HAZARD_RATIO if arm == 1 else 1.0)
        true_event_time = rng.expovariate(hazard)
        if true_event_time <= ADMINISTRATIVE_CENSORING_WEEKS:
            event_time_weeks = true_event_time
            event_indicator = True
        else:
            event_time_weeks = ADMINISTRATIVE_CENSORING_WEEKS
            event_indicator = False

        records.append(
            _make_patient_record(
                f"P{i:04d}",
                treatment_arm=arm,
                baseline_cd4=200.0 + rng.uniform(-50, 50),
                karnofsky_score=80.0 + rng.uniform(-10, 10),
                age=35.0 + rng.uniform(-10, 10),
                symptomatic_status=rng.random() < 0.3,
                prior_opportunistic_infection=rng.random() < 0.2,
                prior_art_exposure_days=rng.uniform(0, 60),
                event_time_weeks=event_time_weeks,
                event_indicator=event_indicator,
            )
        )
    return tuple(records)


@pytest.fixture(scope="module")
def patients() -> tuple[PatientRecord, ...]:
    return _simulate_patients()


def test_fit_converges_with_finite_coefficients(patients):
    baseline = CoxBaseline().fit(patients)

    fitted = baseline.fitted
    assert set(fitted.coefficients.keys()) == set(COEFFICIENT_NAMES)
    for name, value in fitted.coefficients.items():
        assert math.isfinite(value), f"coefficient {name!r} is not finite: {value}"


def test_fit_treatment_coefficient_sign_is_protective(patients):
    baseline = CoxBaseline().fit(patients)

    assert baseline.fitted.coefficients["treatment_arm"] < 0.0


def test_fit_cox_baseline_function_matches_class_interface(patients):
    fitted = fit_cox_baseline(patients)

    assert set(fitted.coefficients.keys()) == set(COEFFICIENT_NAMES)
    assert fitted.coefficients["treatment_arm"] < 0.0


def test_risk_scores_are_finite_and_vary_across_patients(patients):
    baseline = CoxBaseline().fit(patients)
    fitted = baseline.fitted

    risk_scores = [fitted.risk_scores[patient.patient_id] for patient in patients]

    assert len(risk_scores) == len(patients)
    assert all(math.isfinite(score) and score > 0.0 for score in risk_scores)
    # Risk scores should not all be identical (they depend on differing
    # covariates/treatment arm across patients).
    assert len(set(risk_scores)) > 1


def test_predict_risk_matches_stored_risk_scores(patients):
    baseline = CoxBaseline().fit(patients)
    fitted = baseline.fitted

    for patient in patients[:10]:
        predicted = baseline.predict_risk(patient)
        stored = fitted.risk_scores[patient.patient_id]
        assert math.isclose(predicted, stored, rel_tol=1e-9)
