"""Unit tests for `src/causal/baselines.py`.

Builds ONE synthetic cohort (300 patients) with a KNOWN protective
treatment effect baked in via a simple exponential-time simulation
(treatment halves the event hazard, `hazard_ratio = 0.5`), fits
`LogisticPropensityModel` cross-fitted propensity scores on it, and
confirms each of the six baselines:

- runs without error;
- produces an estimate with the correct SIGN (negative log-hazard-ratio
  for the three Cox variants; negative risk difference for the
  outcome-regression/IPCW variants), consistent with treatment being
  protective.

This is a synthetic sanity check (directional correctness, no crashes),
not a statistical validation suite -- exact numerical recovery of
`hazard_ratio=0.5` is not asserted.

**Validates: Requirements 7.8, 7.9, 7.10**
"""

import math
import random
from datetime import datetime, timedelta

import pytest

from causal.baselines import (
    CausalEstimate,
    adjusted_cox,
    ipcw_aipw,
    ipcw_ipw,
    outcome_regression,
    ps_weighted_marginal_cox,
    unadjusted_cox,
    unadjusted_km,
)
from causal.propensity import LogisticPropensityModel
from cohorts.cohort_construction import PatientRecord
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)

#: True simulated effect: treatment halves the baseline hazard.
TRUE_HAZARD_RATIO = 0.5
BASELINE_HAZARD = 0.01  # per week
ADMINISTRATIVE_CENSORING_WEEKS = 120.0
N_PATIENTS = 300


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
    """Simple exponential-time simulation with a known protective effect.

    Each patient's true event time is drawn `~ Exponential(hazard)`, with
    `hazard = BASELINE_HAZARD * TRUE_HAZARD_RATIO` for treated patients and
    `hazard = BASELINE_HAZARD` for control -- i.e. treatment halves the
    hazard, a known protective effect. Administrative censoring is applied
    at `ADMINISTRATIVE_CENSORING_WEEKS`. Covariates are drawn independently
    of treatment/outcome (a simple randomized-style synthetic cohort; not
    testing confounding-adjustment recovery, just that each estimator runs
    and points the correct direction).
    """
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


@pytest.fixture(scope="module")
def propensity_scores(patients) -> dict[str, float]:
    result = LogisticPropensityModel(seed=0).fit_cross_fitted(patients, n_folds=5)
    return result.propensity_scores


# ---------------------------------------------------------------------------
# unadjusted_km (descriptive, no CausalEstimate)
# ---------------------------------------------------------------------------


def test_unadjusted_km_runs_and_treated_arm_survives_longer(patients):
    curves = unadjusted_km(patients)

    assert set(curves.keys()) == {0, 1}
    for curve in curves.values():
        assert len(curve.times) == len(curve.survival)
        # Survival probabilities are non-increasing and in [0, 1].
        assert all(0.0 <= s <= 1.0 for s in curve.survival)

    # At a late follow-up time, the treated arm's survival should be >=
    # the control arm's (treatment is protective in the simulation).
    late_time = ASSIGNMENT_TIME + timedelta(weeks=ADMINISTRATIVE_CENSORING_WEEKS)
    survival_treated = curves[1].survival_at(late_time)
    survival_control = curves[0].survival_at(late_time)
    assert survival_treated >= survival_control


# ---------------------------------------------------------------------------
# Cox variants: negative log-hazard-ratio expected
# ---------------------------------------------------------------------------


def test_unadjusted_cox_runs_and_sign_is_protective(patients):
    result = unadjusted_cox(patients)

    assert isinstance(result, CausalEstimate)
    assert result.name == "unadjusted_cox"
    assert result.scale == "log_hazard_ratio"
    assert result.is_primary is False
    assert result.estimate < 0.0


def test_adjusted_cox_runs_and_sign_is_protective(patients):
    result = adjusted_cox(patients)

    assert result.scale == "log_hazard_ratio"
    assert result.is_primary is False
    assert result.estimate < 0.0


def test_ps_weighted_marginal_cox_runs_and_sign_is_protective(patients, propensity_scores):
    result = ps_weighted_marginal_cox(patients, propensity_scores)

    assert result.scale == "log_hazard_ratio"
    assert result.is_primary is False
    assert result.estimate < 0.0


# ---------------------------------------------------------------------------
# Risk-difference variants: negative risk difference expected
# ---------------------------------------------------------------------------


def test_outcome_regression_runs_and_sign_is_protective(patients):
    result = outcome_regression(patients)

    assert result.scale == "risk_difference"
    assert result.is_primary is False
    assert result.estimate < 0.0


def test_ipcw_ipw_runs_and_sign_is_protective(patients, propensity_scores):
    result = ipcw_ipw(patients, propensity_scores)

    assert result.scale == "risk_difference"
    assert result.is_primary is False
    assert not math.isnan(result.estimate)
    assert result.estimate < 0.0


def test_ipcw_aipw_runs_is_marked_primary_and_sign_is_protective(patients, propensity_scores):
    result = ipcw_aipw(patients, propensity_scores)

    assert result.scale == "risk_difference"
    assert result.is_primary is True
    assert not math.isnan(result.estimate)
    assert result.estimate < 0.0


def test_only_ipcw_aipw_is_primary(patients, propensity_scores):
    all_results = [
        unadjusted_cox(patients),
        adjusted_cox(patients),
        ps_weighted_marginal_cox(patients, propensity_scores),
        outcome_regression(patients),
        ipcw_ipw(patients, propensity_scores),
        ipcw_aipw(patients, propensity_scores),
    ]

    primary_flags = [result.is_primary for result in all_results]
    assert primary_flags == [False, False, False, False, False, True]
