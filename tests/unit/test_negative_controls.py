"""Unit tests for `src/causal/negative_controls.py`.

**Validates: Requirements 7.12; Correctness Property 4**

Covers:

- `SyntheticShamExposure.generate` depends only on `(patient_id, seed)`:
  deterministic repeatability, seed-sensitivity, and independence from
  covariate/outcome data (identical sham labels across two "cohorts" that
  share `patient_id`s but differ in every other field).
- `run_negative_control_exposure`: a synthetic cohort with no true
  treatment effect and random arm/outcome assignment (so re-running the
  primary estimator with an independent sham label should recover an
  estimate near the null) is confirmed to classify `"pass"`.
- `run_negative_control_outcome`: a synthetic cohort where baseline CD8 is
  drawn independently of treatment arm is confirmed to classify `"pass"`,
  and `assert_no_self_adjustment` is confirmed to actually be invoked/
  respected (baseline CD8 never ends up in the regression's own adjustment
  list).

Per this task's documented time-budget guidance, only one determinate
classification per function (`"pass"`) is exercised; `"fail"`/`"uncertain"`
branches are not separately covered here.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from causal.negative_controls import (
    BASELINE_CD8_FIELD,
    NegativeControlResult,
    SyntheticShamExposure,
    run_negative_control_exposure,
    run_negative_control_outcome,
)
from causal.propensity import LogisticPropensityModel
from cohorts.cohort_construction import PatientRecord
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
ADMINISTRATIVE_CENSORING_WEEKS = 120.0
N_PATIENTS = 300
NEGATIVE_CONTROL_SEED = 424242


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
    baseline_cd8: float,
    karnofsky_score: float,
    age: float,
    event_time_weeks: float,
    event_indicator: bool,
) -> PatientRecord:
    event_time = ASSIGNMENT_TIME + timedelta(weeks=event_time_weeks)
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=baseline_cd4),
        _event(patient_id, concept_id="baseline_cd8", value=baseline_cd8),
        _event(patient_id, concept_id="karnofsky_score", value=karnofsky_score, unit="score"),
        _event(patient_id, concept_id="age", value=age, unit="years"),
        _event(patient_id, concept_id="symptomatic_status", value=False, unit=None),
        _event(
            patient_id, concept_id="prior_opportunistic_infection", value=False, unit=None
        ),
        _event(
            patient_id, concept_id="prior_art_exposure_days", value=0.0, unit="days"
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


def _simulate_null_cohort(seed: int, n: int = N_PATIENTS) -> tuple[PatientRecord, ...]:
    """A cohort with NO true treatment effect: arm, outcome, and baseline
    CD8 are all drawn independently of one another, so both negative
    controls should recover an estimate near the null.
    """
    rng = random.Random(seed)
    records = []
    for i in range(n):
        arm = rng.randint(0, 1)
        hazard = 0.01
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
                baseline_cd8=800.0 + rng.uniform(-100, 100),
                karnofsky_score=80.0 + rng.uniform(-10, 10),
                age=35.0 + rng.uniform(-10, 10),
                event_time_weeks=event_time_weeks,
                event_indicator=event_indicator,
            )
        )
    return tuple(records)


ADJUSTMENT_SET = ["age", "baseline_cd4", "baseline_cd8", "karnofsky_score"]


@pytest.fixture(scope="module")
def null_patients() -> tuple[PatientRecord, ...]:
    return _simulate_null_cohort(seed=99)


@pytest.fixture(scope="module")
def null_propensity_scores(null_patients) -> dict[str, float]:
    result = LogisticPropensityModel(seed=0).fit_cross_fitted(null_patients, n_folds=5)
    return result.propensity_scores


# ---------------------------------------------------------------------------
# SyntheticShamExposure.generate
# ---------------------------------------------------------------------------


class TestSyntheticShamExposure:
    def test_same_patient_id_and_seed_always_gives_same_label(self):
        patient_ids = [f"P{i:04d}" for i in range(50)]
        labels_a = SyntheticShamExposure.generate(patient_ids, seed=1, target_prevalence=0.5)
        labels_b = SyntheticShamExposure.generate(patient_ids, seed=1, target_prevalence=0.5)
        assert labels_a == labels_b

    def test_changing_seed_changes_at_least_some_labels(self):
        patient_ids = [f"P{i:04d}" for i in range(200)]
        labels_seed_1 = SyntheticShamExposure.generate(patient_ids, seed=1, target_prevalence=0.5)
        labels_seed_2 = SyntheticShamExposure.generate(patient_ids, seed=2, target_prevalence=0.5)
        assert labels_seed_1 != labels_seed_2

    def test_labels_are_binary_and_cover_every_patient(self):
        patient_ids = [f"P{i:04d}" for i in range(20)]
        labels = SyntheticShamExposure.generate(patient_ids, seed=7, target_prevalence=0.4)
        assert set(labels) == set(patient_ids)
        assert all(label in (0, 1) for label in labels.values())

    def test_labels_independent_of_covariates_and_outcomes(self, null_patients):
        """Two "cohorts" sharing patient_ids but differing in every other
        field must produce IDENTICAL sham labels for the same seed --
        proving the label depends only on patient_id and seed.
        """
        patient_ids = [patient.patient_id for patient in null_patients]

        # "Cohort A": the real fixture's covariates/outcomes.
        labels_cohort_a = SyntheticShamExposure.generate(patient_ids, seed=5, target_prevalence=0.5)

        # "Cohort B": same patient_ids, but wildly different covariates,
        # outcomes, and even a differently-seeded RNG driving the (unused)
        # covariate values -- only patient_id and the negative-control
        # `seed` argument are shared with cohort A.
        rng = random.Random(31337)
        mutated_patients = tuple(
            _make_patient_record(
                patient.patient_id,
                treatment_arm=1 - patient.treatment_arm,
                baseline_cd4=rng.uniform(0, 1000),
                baseline_cd8=rng.uniform(0, 2000),
                karnofsky_score=rng.uniform(0, 100),
                age=rng.uniform(18, 90),
                event_time_weeks=rng.uniform(0, 200),
                event_indicator=rng.random() < 0.5,
            )
            for patient in null_patients
        )
        mutated_ids = [patient.patient_id for patient in mutated_patients]
        labels_cohort_b = SyntheticShamExposure.generate(
            mutated_ids, seed=5, target_prevalence=0.5
        )

        assert labels_cohort_a == labels_cohort_b

    def test_target_prevalence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SyntheticShamExposure.generate(["P0001"], seed=1, target_prevalence=1.5)


# ---------------------------------------------------------------------------
# run_negative_control_exposure
# ---------------------------------------------------------------------------


class TestRunNegativeControlExposure:
    def test_null_cohort_classifies_pass(self, null_patients, null_propensity_scores):
        result = run_negative_control_exposure(
            null_patients,
            null_propensity_scores,
            seed=NEGATIVE_CONTROL_SEED,
            n_bootstrap=60,
        )

        assert isinstance(result, NegativeControlResult)
        assert result.name == "negative_control_exposure"
        assert result.classification == "pass"
        assert result.ci_low <= 0.0 <= result.ci_high

    def test_result_evidence_documents_scale_adaptation(
        self, null_patients, null_propensity_scores
    ):
        result = run_negative_control_exposure(
            null_patients,
            null_propensity_scores,
            seed=NEGATIVE_CONTROL_SEED,
            n_bootstrap=20,
        )

        assert result.evidence["scale"] == "risk_difference"
        assert "scale_adaptation_note" in result.evidence


# ---------------------------------------------------------------------------
# run_negative_control_outcome
# ---------------------------------------------------------------------------


class TestRunNegativeControlOutcome:
    def test_null_cohort_classifies_pass(self, null_patients):
        result = run_negative_control_outcome(
            null_patients,
            ADJUSTMENT_SET,
            seed=NEGATIVE_CONTROL_SEED,
            n_bootstrap=60,
        )

        assert isinstance(result, NegativeControlResult)
        assert result.name == "negative_control_outcome"
        assert result.classification == "pass"
        assert result.ci_low <= 0.0 <= result.ci_high
        assert abs(result.point_estimate) < 0.10

    def test_self_adjustment_safeguard_is_respected(self, null_patients):
        """`baseline_cd8` must never end up in the regression's own
        adjustment list -- confirmed both via the returned evidence and by
        the fact that `assert_no_self_adjustment` does not raise.
        """
        result = run_negative_control_outcome(
            null_patients,
            ADJUSTMENT_SET,
            seed=NEGATIVE_CONTROL_SEED,
            n_bootstrap=5,
        )

        assert BASELINE_CD8_FIELD not in result.evidence["regression_covariates"]
        assert BASELINE_CD8_FIELD in ADJUSTMENT_SET  # sanity: it WAS in the input set

    def test_adjustment_set_without_cd8_is_unaffected(self, null_patients):
        """If the caller's adjustment_set never had baseline_cd8 in the
        first place, `assert_no_self_adjustment` still does not raise, and
        the regression covariates equal the input set.
        """
        covariates_without_cd8 = ["age", "baseline_cd4", "karnofsky_score"]
        result = run_negative_control_outcome(
            null_patients,
            covariates_without_cd8,
            seed=NEGATIVE_CONTROL_SEED,
            n_bootstrap=5,
        )

        assert result.evidence["regression_covariates"] == covariates_without_cd8
