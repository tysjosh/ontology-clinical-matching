"""Unit tests for `src/cohorts/splitting.py`.

Builds a synthetic `RandomizedCohort` (same hand-built-event style as
`test_cohort_construction.py`/`test_sealed_reference.py`) spanning both
treatment arms, both `primary_event_indicator` values, and both
`prior_art_status` values, seals a `SealedReference` from it (per
`test_sealed_reference.py`'s pattern), and confirms `stratified_patient_split`:

- produces a true partition (no `patient_id` in more than one split);
- covers the whole cohort (`len(dev) + len(val) + len(test) == len(cohort)`);
- respects per-stratum proportions within a reasonable tolerance;
- rejects a missing/wrong-typed `sealed_reference` with `TypeError`;
- is reproducible across repeated calls with the same seed.

**Validates: Requirements 6.2, 6.7**
"""

from datetime import datetime

import pytest

from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from cohorts.sealed_reference import SealedReference, placeholder_estimator
from cohorts.splitting import (
    PatientSet,
    SplitCohorts,
    stratified_patient_split,
)
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
    event_indicator: bool,
    prior_art_exposure_days: float | None,
) -> PatientRecord:
    """A minimal, hand-built `PatientRecord` with a controllable
    `prior_art_exposure_days` static covariate, `treatment_arm`, and
    `endpoint_result.event_indicator` -- exactly the three fields
    `DEFAULT_STRATA` stratifies on.
    """
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=350.0),
        _event(patient_id, concept_id="age", value=34.0, unit="years"),
    ]
    if prior_art_exposure_days is not None:
        events.append(
            _event(
                patient_id,
                concept_id="prior_art_exposure_days",
                value=prior_art_exposure_days,
                unit="days",
            )
        )
    timeline = build_patient_timeline(events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME,
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="death" if event_indicator else None,
        ),
    )


def _build_synthetic_cohort(n_per_group: int = 8) -> RandomizedCohort:
    """A cohort spanning every combination of treatment_arm x
    primary_event_indicator x prior_art_status, with `n_per_group` patients
    in each of the 2x2x2 = 8 combinations (48 patients total by default) --
    enough per stratum to check proportions meaningfully.
    """
    patients: list[PatientRecord] = []
    counter = 0
    for treatment_arm in (0, 1):
        for event_indicator in (False, True):
            for prior_art_exposure_days in (0.0, 120.0):  # naive, experienced
                for _ in range(n_per_group):
                    counter += 1
                    patient_id = f"P{counter:04d}"
                    patients.append(
                        _make_patient_record(
                            patient_id,
                            treatment_arm=treatment_arm,
                            event_indicator=event_indicator,
                            prior_art_exposure_days=prior_art_exposure_days,
                        )
                    )
    return RandomizedCohort(patients=tuple(sorted(patients, key=lambda p: p.patient_id)))


@pytest.fixture
def synthetic_cohort() -> RandomizedCohort:
    return _build_synthetic_cohort()


@pytest.fixture
def sealed_reference(synthetic_cohort: RandomizedCohort) -> SealedReference:
    return SealedReference(synthetic_cohort, estimator=placeholder_estimator)


def test_split_is_a_true_partition_with_no_overlap(
    synthetic_cohort: RandomizedCohort, sealed_reference: SealedReference
):
    result = stratified_patient_split(synthetic_cohort, sealed_reference, seed=42)

    assert isinstance(result, SplitCohorts)
    assert isinstance(result.dev, PatientSet)
    assert isinstance(result.val, PatientSet)
    assert isinstance(result.test, PatientSet)

    dev_ids = set(result.dev.patient_ids)
    val_ids = set(result.val.patient_ids)
    test_ids = set(result.test.patient_ids)

    assert dev_ids.isdisjoint(val_ids)
    assert dev_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Every patient_id appears exactly once across dev/val/test combined.
    all_split_ids = list(result.dev.patient_ids) + list(result.val.patient_ids) + list(
        result.test.patient_ids
    )
    assert len(all_split_ids) == len(set(all_split_ids))


def test_split_covers_the_whole_cohort(
    synthetic_cohort: RandomizedCohort, sealed_reference: SealedReference
):
    result = stratified_patient_split(synthetic_cohort, sealed_reference, seed=42)

    total_split_size = len(result.dev) + len(result.val) + len(result.test)
    assert total_split_size == len(synthetic_cohort)

    combined_ids = set(result.dev.patient_ids) | set(result.val.patient_ids) | set(
        result.test.patient_ids
    )
    assert combined_ids == set(synthetic_cohort.patient_ids)


def test_strata_proportions_are_reasonably_respected(
    synthetic_cohort: RandomizedCohort, sealed_reference: SealedReference
):
    """Each of the 8 treatment_arm x event_indicator x prior_art_status strata
    (8 patients each) should land roughly 60/20/20 across dev/val/test, within
    a generous tolerance appropriate for an 8-patient stratum.
    """
    result = stratified_patient_split(
        synthetic_cohort, sealed_reference, fractions=(0.6, 0.2, 0.2), seed=42
    )

    by_id = {patient.patient_id: patient for patient in synthetic_cohort}

    def stratum_key(patient_id: str) -> tuple:
        patient = by_id[patient_id]
        exposure = patient.timeline.static_covariates.get("prior_art_exposure_days")
        prior_art_status = "experienced" if (exposure or 0) > 0 else "naive"
        return (patient.treatment_arm, patient.endpoint_result.event_indicator, prior_art_status)

    from collections import Counter

    dev_strata = Counter(stratum_key(pid) for pid in result.dev.patient_ids)
    val_strata = Counter(stratum_key(pid) for pid in result.val.patient_ids)
    test_strata = Counter(stratum_key(pid) for pid in result.test.patient_ids)

    all_keys = set(dev_strata) | set(val_strata) | set(test_strata)
    for key in all_keys:
        stratum_total = dev_strata[key] + val_strata[key] + test_strata[key]
        assert stratum_total == 8  # n_per_group default

        # Largest-remainder rounding on an 8-patient stratum with (0.6,0.2,0.2)
        # targets should give something close to (5, 2, 1) or (4/5, 1/2, 1/2)-ish.
        assert dev_strata[key] >= 4
        assert val_strata[key] >= 1
        assert test_strata[key] >= 1


def test_calling_without_a_sealed_reference_raises_type_error(
    synthetic_cohort: RandomizedCohort,
):
    with pytest.raises(TypeError):
        stratified_patient_split(synthetic_cohort, None, seed=42)

    with pytest.raises(TypeError):
        # A bare RandomizedCohort is not a SealedReference either.
        stratified_patient_split(synthetic_cohort, synthetic_cohort, seed=42)


def test_repeated_calls_with_same_seed_are_reproducible(
    synthetic_cohort: RandomizedCohort, sealed_reference: SealedReference
):
    result_1 = stratified_patient_split(synthetic_cohort, sealed_reference, seed=7)
    result_2 = stratified_patient_split(synthetic_cohort, sealed_reference, seed=7)

    assert result_1.dev.patient_ids == result_2.dev.patient_ids
    assert result_1.val.patient_ids == result_2.val.patient_ids
    assert result_1.test.patient_ids == result_2.test.patient_ids


def test_different_seeds_can_produce_different_splits(
    synthetic_cohort: RandomizedCohort, sealed_reference: SealedReference
):
    result_1 = stratified_patient_split(synthetic_cohort, sealed_reference, seed=7)
    result_2 = stratified_patient_split(synthetic_cohort, sealed_reference, seed=999)

    assert result_1.dev.patient_ids != result_2.dev.patient_ids


def test_fractions_not_summing_to_one_raises_value_error(
    synthetic_cohort: RandomizedCohort, sealed_reference: SealedReference
):
    with pytest.raises(ValueError):
        stratified_patient_split(
            synthetic_cohort, sealed_reference, fractions=(0.5, 0.3, 0.3), seed=42
        )


def test_unknown_stratum_field_raises_value_error(
    synthetic_cohort: RandomizedCohort, sealed_reference: SealedReference
):
    with pytest.raises(ValueError):
        stratified_patient_split(
            synthetic_cohort, sealed_reference, strata=["not_a_real_field"], seed=42
        )


def test_prior_art_status_unknown_when_covariate_absent():
    """A patient with no prior_art_exposure_days covariate at all falls into
    its own "unknown" stratum rather than erroring or being silently merged.
    """
    patient_with_missing_covariate = _make_patient_record(
        "P_missing",
        treatment_arm=0,
        event_indicator=False,
        prior_art_exposure_days=None,
    )
    cohort = RandomizedCohort(
        patients=(
            patient_with_missing_covariate,
            _make_patient_record(
                "P_naive", treatment_arm=0, event_indicator=False, prior_art_exposure_days=0.0
            ),
        )
    )
    sealed = SealedReference(cohort, estimator=placeholder_estimator)

    # Should not raise -- the "unknown" patient gets its own 1-patient stratum.
    result = stratified_patient_split(cohort, sealed, seed=1)
    combined_ids = set(result.dev.patient_ids) | set(result.val.patient_ids) | set(
        result.test.patient_ids
    )
    assert combined_ids == {"P_missing", "P_naive"}
