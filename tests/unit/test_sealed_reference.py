"""Unit tests for `src/cohorts/sealed_reference.py` and
`src/infra/reference_guard.py`.

Builds a small synthetic `RandomizedCohort` (same hand-built-event style as
`tests/unit/test_cohort_construction.py`) and confirms `SealedReference`'s
access-control contract from `design.md` "Correctness Property 1: Sealed
reference isolation" (`decisions.md` Section 8):

- `.unseal()` raises `ReferenceAccessError` when called with no token,
  `None`, or a wrong-typed value;
- `.unseal()` succeeds and returns the sealed `InternalReference` when
  called with a validly-minted `LockedEvaluationToken`.

**Validates: Requirements 6.6, 4.6 (Correctness Property 1)**
"""

from datetime import datetime

import pytest

from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from cohorts.sealed_reference import (
    InternalReference,
    LockedEvaluationToken,
    SealedReference,
    placeholder_estimator,
    read_sealed_artifact,
    unadjusted_cox_reference_estimator,
)
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from infra.reference_guard import ReferenceAccessError, assert_reference_unlocked
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


def _make_patient_record(patient_id: str, treatment_arm: int) -> PatientRecord:
    """A minimal, hand-built `PatientRecord` -- enough to populate a
    `RandomizedCohort` without needing the full eligibility/endpoint
    machinery exercised in `test_cohort_construction.py`.
    """
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=350.0),
        _event(patient_id, concept_id="age", value=34.0, unit="years"),
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
def randomized_cohort() -> RandomizedCohort:
    return RandomizedCohort(
        patients=(
            _make_patient_record("p_arm0", treatment_arm=0),
            _make_patient_record("p_arm1", treatment_arm=1),
        )
    )


@pytest.fixture
def sealed_reference(randomized_cohort: RandomizedCohort) -> SealedReference:
    return SealedReference(randomized_cohort, estimator=placeholder_estimator)


def test_sealed_reference_is_constructed_from_randomized_cohort(
    randomized_cohort: RandomizedCohort,
):
    """`SealedReference` computes its `InternalReference` eagerly at
    construction time, over the complete `RandomizedCohort`.
    """
    sealed = SealedReference(randomized_cohort, estimator=placeholder_estimator)
    token = LockedEvaluationToken._mint(issued_by="test")

    internal_reference = sealed.unseal(token)

    assert isinstance(internal_reference, InternalReference)
    assert internal_reference.cohort_size == 2
    assert internal_reference.metadata["arm_0_count"] == 1
    assert internal_reference.metadata["arm_1_count"] == 1


def test_unseal_raises_without_a_token(sealed_reference: SealedReference):
    with pytest.raises(ReferenceAccessError):
        sealed_reference.unseal(None)


def test_unseal_raises_with_a_wrong_typed_argument(sealed_reference: SealedReference):
    with pytest.raises(ReferenceAccessError):
        sealed_reference.unseal("not-a-token")

    with pytest.raises(ReferenceAccessError):
        sealed_reference.unseal(12345)


def test_unseal_succeeds_with_a_validly_minted_token(sealed_reference: SealedReference):
    token = LockedEvaluationToken._mint(issued_by="scripts/check_go_no_go.py")

    internal_reference = sealed_reference.unseal(token)

    assert isinstance(internal_reference, InternalReference)


def test_unadjusted_cox_reference_estimator_produces_a_real_log_hazard_ratio(
    randomized_cohort: RandomizedCohort,
):
    """`unadjusted_cox_reference_estimator` (the real, non-placeholder
    estimator wired in for real-data runs) produces an actual numeric
    log-hazard-ratio value, not `None`."""
    sealed = SealedReference(randomized_cohort, estimator=unadjusted_cox_reference_estimator)
    token = LockedEvaluationToken._mint(issued_by="test")

    internal_reference = sealed.unseal(token)

    assert isinstance(internal_reference.value, float)
    assert internal_reference.metadata["estimator"] == "unadjusted_cox_reference_estimator"
    assert internal_reference.metadata["scale"] == "log_hazard_ratio"


def test_write_sealed_artifact_then_read_sealed_artifact_round_trips(
    randomized_cohort: RandomizedCohort, tmp_path
):
    sealed = SealedReference(randomized_cohort, estimator=unadjusted_cox_reference_estimator)
    artifact_path = tmp_path / "reference" / "internal_randomized_reference.sealed.json"

    written_path = sealed.write_sealed_artifact(artifact_path)
    assert written_path == artifact_path
    assert artifact_path.is_file()

    token = LockedEvaluationToken._mint(issued_by="test")
    reconstructed = read_sealed_artifact(artifact_path, unlock_token=token)

    original = sealed.unseal(token)
    assert reconstructed.value == original.value
    assert reconstructed.cohort_size == original.cohort_size
    assert reconstructed.metadata == original.metadata


def test_write_sealed_artifact_uses_owner_only_permissions(
    randomized_cohort: RandomizedCohort, tmp_path
):
    import os
    import stat

    sealed = SealedReference(randomized_cohort, estimator=placeholder_estimator)
    artifact_path = tmp_path / "internal_randomized_reference.sealed.json"

    sealed.write_sealed_artifact(artifact_path)

    mode = stat.S_IMODE(os.stat(artifact_path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_read_sealed_artifact_refuses_without_a_token_before_opening_the_file(tmp_path):
    """`read_sealed_artifact` must refuse before opening `path` at all --
    confirmed by pointing it at a path that does not even exist and
    checking `ReferenceAccessError` is still raised (not `FileNotFoundError`)."""
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(ReferenceAccessError):
        read_sealed_artifact(missing_path, unlock_token=None)


def test_read_sealed_artifact_raises_file_not_found_with_a_valid_token_but_missing_file(
    tmp_path,
):
    missing_path = tmp_path / "does_not_exist.json"
    token = LockedEvaluationToken._mint(issued_by="test")

    with pytest.raises(FileNotFoundError):
        read_sealed_artifact(missing_path, unlock_token=token)


def test_locked_evaluation_token_requires_non_empty_issued_by():
    with pytest.raises(ValueError):
        LockedEvaluationToken(issued_by="", issued_at=datetime.now())


def test_assert_reference_unlocked_matches_unseal_behavior(sealed_reference: SealedReference):
    """`SealedReference.unseal()` delegates to
    `infra.reference_guard.assert_reference_unlocked` rather than
    duplicating the check -- confirm the standalone function agrees.
    """
    with pytest.raises(ReferenceAccessError):
        assert_reference_unlocked(None)

    valid_token = LockedEvaluationToken._mint(issued_by="test")
    assert_reference_unlocked(valid_token)  # returns silently, no exception
