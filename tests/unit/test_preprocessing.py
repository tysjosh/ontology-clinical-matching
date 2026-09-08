"""Unit tests for `src/cohorts/preprocessing.py`.

Builds a synthetic `SplitCohorts` (same hand-built-event style as
`test_splitting.py`) with deliberately different `baseline_cd4`
distributions in dev vs. val/test, and confirms `FittedPreprocessing`:

- `.fit(DevelopmentSet(dev))` computes parameters matching a hand-computed
  mean/stdev from ONLY the dev patients;
- `.fit()` called with a bare `PatientSet` (e.g. `split_cohorts.val`
  unwrapped) raises `DevelopmentSetRequiredError`;
- `.transform(val)` and `.transform(test)` apply the SAME dev-fitted
  parameters, matching what dev-fit parameters would produce and differing
  from what val/test-fit parameters would have produced;
- `.transform()` before `.fit()` raises `PreprocessingNotFittedError`.

**Validates: Requirements 6.5, 6.6 (Correctness Property 2)**
"""

import statistics
from datetime import datetime

import pytest

from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from cohorts.preprocessing import (
    DevelopmentSet,
    DevelopmentSetRequiredError,
    FittedPreprocessing,
    PreprocessingNotFittedError,
)
from cohorts.sealed_reference import SealedReference, placeholder_estimator
from cohorts.splitting import PatientSet, SplitCohorts, stratified_patient_split
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


def _make_patient_record(patient_id: str, *, baseline_cd4: float) -> PatientRecord:
    """A minimal, hand-built `PatientRecord` with a controllable `baseline_cd4`
    static covariate -- the field these tests standardize.
    """
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=baseline_cd4),
        _event(patient_id, concept_id="age", value=34.0, unit="years"),
    ]
    timeline = build_patient_timeline(events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=0,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME,
            event_indicator=False,
            censoring_indicator=True,
            event_type=None,
        ),
    )


def _patient_set(baseline_cd4_values: list[float], *, prefix: str) -> PatientSet:
    patients = tuple(
        _make_patient_record(f"{prefix}{i:03d}", baseline_cd4=value)
        for i, value in enumerate(baseline_cd4_values)
    )
    return PatientSet(patients)


# Deliberately different underlying distributions: dev is centered around
# 300, val/test are centered around 600 -- far enough apart that fitting on
# val/test instead of dev would produce clearly different standardized
# values, proving val/test values were not used in fitting.
DEV_CD4_VALUES = [280.0, 300.0, 310.0, 320.0, 290.0]
VAL_CD4_VALUES = [590.0, 600.0, 610.0]
TEST_CD4_VALUES = [580.0, 620.0, 605.0]


@pytest.fixture
def split_cohorts() -> SplitCohorts:
    return SplitCohorts(
        dev=_patient_set(DEV_CD4_VALUES, prefix="DEV"),
        val=_patient_set(VAL_CD4_VALUES, prefix="VAL"),
        test=_patient_set(TEST_CD4_VALUES, prefix="TEST"),
    )


def test_fit_computes_parameters_from_dev_patients_only(split_cohorts: SplitCohorts):
    preprocessing = FittedPreprocessing(fields=("baseline_cd4",))
    preprocessing.fit(DevelopmentSet(split_cohorts.dev))

    expected_mean = statistics.fmean(DEV_CD4_VALUES)
    expected_stdev = statistics.stdev(DEV_CD4_VALUES)

    params = preprocessing.fitted_params
    assert params.means["baseline_cd4"] == pytest.approx(expected_mean)
    assert params.stdevs["baseline_cd4"] == pytest.approx(expected_stdev)


def test_fit_rejects_bare_patient_set(split_cohorts: SplitCohorts):
    preprocessing = FittedPreprocessing(fields=("baseline_cd4",))

    with pytest.raises(DevelopmentSetRequiredError):
        preprocessing.fit(split_cohorts.val)

    with pytest.raises(DevelopmentSetRequiredError):
        preprocessing.fit(split_cohorts.dev)  # unwrapped dev is still rejected

    with pytest.raises(DevelopmentSetRequiredError):
        preprocessing.fit(None)


def test_transform_before_fit_raises():
    preprocessing = FittedPreprocessing(fields=("baseline_cd4",))
    dummy_split = _patient_set([300.0], prefix="X")

    with pytest.raises(PreprocessingNotFittedError):
        preprocessing.transform(dummy_split)


def test_transform_applies_dev_fitted_parameters_to_val_and_test(split_cohorts: SplitCohorts):
    preprocessing = FittedPreprocessing(fields=("baseline_cd4",))
    preprocessing.fit(DevelopmentSet(split_cohorts.dev))

    dev_mean = statistics.fmean(DEV_CD4_VALUES)
    dev_stdev = statistics.stdev(DEV_CD4_VALUES)

    transformed_val = preprocessing.transform(split_cohorts.val)
    transformed_test = preprocessing.transform(split_cohorts.test)

    expected_val = [(v - dev_mean) / dev_stdev for v in VAL_CD4_VALUES]
    expected_test = [(v - dev_mean) / dev_stdev for v in TEST_CD4_VALUES]

    actual_val = [
        patient.timeline.static_covariates["baseline_cd4"] for patient in transformed_val
    ]
    actual_test = [
        patient.timeline.static_covariates["baseline_cd4"] for patient in transformed_test
    ]

    assert actual_val == pytest.approx(expected_val)
    assert actual_test == pytest.approx(expected_test)

    # Prove val/test values were NOT used in fitting: standardizing val/test
    # with parameters fit on val/test itself would give very different
    # (near-zero-mean) results than what dev-fitted parameters produce here.
    val_only_mean = statistics.fmean(VAL_CD4_VALUES)
    val_only_stdev = statistics.stdev(VAL_CD4_VALUES)
    val_fit_on_itself = [(v - val_only_mean) / val_only_stdev for v in VAL_CD4_VALUES]
    assert actual_val != pytest.approx(val_fit_on_itself)


def test_transform_does_not_mutate_original_patients(split_cohorts: SplitCohorts):
    preprocessing = FittedPreprocessing(fields=("baseline_cd4",))
    preprocessing.fit(DevelopmentSet(split_cohorts.dev))

    original_values = [
        patient.timeline.static_covariates["baseline_cd4"] for patient in split_cohorts.val
    ]
    preprocessing.transform(split_cohorts.val)

    unchanged_values = [
        patient.timeline.static_covariates["baseline_cd4"] for patient in split_cohorts.val
    ]
    assert original_values == unchanged_values == VAL_CD4_VALUES


def test_field_with_zero_stdev_standardizes_to_zero():
    preprocessing = FittedPreprocessing(fields=("baseline_cd4",))
    identical_dev = _patient_set([100.0], prefix="DEV")  # single value -> stdev 0.0
    preprocessing.fit(DevelopmentSet(identical_dev))

    val_split = _patient_set([100.0, 200.0], prefix="VAL")
    transformed = preprocessing.transform(val_split)

    assert all(
        patient.timeline.static_covariates["baseline_cd4"] == 0.0 for patient in transformed
    )


def test_field_absent_from_development_data_is_left_untouched():
    preprocessing = FittedPreprocessing(fields=("nonexistent_field",))
    dev_split = _patient_set([100.0], prefix="DEV")
    preprocessing.fit(DevelopmentSet(dev_split))

    val_split = _patient_set([200.0], prefix="VAL")
    transformed = preprocessing.transform(val_split)

    # "nonexistent_field" never appears in static_covariates for any
    # patient, so nothing is added/changed; baseline_cd4 remains untouched.
    (patient,) = transformed.patients
    assert "nonexistent_field" not in patient.timeline.static_covariates
    assert patient.timeline.static_covariates["baseline_cd4"] == 200.0


def test_integration_with_stratified_patient_split():
    """End-to-end sanity check: fit on the real `SplitCohorts.dev` produced by
    `stratified_patient_split`, transform `.val`/`.test`, and confirm the
    same dev-fitted parameters are used for both.
    """
    patients = tuple(
        _make_patient_record(f"P{i:03d}", baseline_cd4=300.0 + i)
        for i in range(60)
    )
    cohort = RandomizedCohort(patients=patients)
    sealed = SealedReference(cohort, estimator=placeholder_estimator)
    split_cohorts = stratified_patient_split(cohort, sealed, strata=[], seed=42)

    preprocessing = FittedPreprocessing(fields=("baseline_cd4",))
    preprocessing.fit(DevelopmentSet(split_cohorts.dev))

    dev_values = [
        patient.timeline.static_covariates["baseline_cd4"] for patient in split_cohorts.dev
    ]
    expected_mean = statistics.fmean(dev_values)
    expected_stdev = statistics.stdev(dev_values)

    transformed_val = preprocessing.transform(split_cohorts.val)
    transformed_test = preprocessing.transform(split_cohorts.test)

    for patient, original in zip(transformed_val, split_cohorts.val, strict=True):
        original_value = original.timeline.static_covariates["baseline_cd4"]
        expected = (original_value - expected_mean) / expected_stdev
        assert patient.timeline.static_covariates["baseline_cd4"] == pytest.approx(expected)

    for patient, original in zip(transformed_test, split_cohorts.test, strict=True):
        original_value = original.timeline.static_covariates["baseline_cd4"]
        expected = (original_value - expected_mean) / expected_stdev
        assert patient.timeline.static_covariates["baseline_cd4"] == pytest.approx(expected)
