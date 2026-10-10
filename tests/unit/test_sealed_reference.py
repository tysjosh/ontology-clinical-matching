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

import dataclasses
import json
import struct
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_type_hints

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from causal.baselines import HORIZON_WEEKS
from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from cohorts.preprocessing import DevelopmentSet, FittedPreprocessing
from cohorts.sealed_reference import (
    BOUND_TOKEN_FIELDS,
    RESEAL_INSTRUCTIONS,
    SEALED_FORMAT_VERSION,
    InternalReference,
    LockedEvaluationToken,
    SealedFormatVersionError,
    SealedReference,
    TokenFieldError,
    extended_randomized_reference_estimator,
    placeholder_estimator,
    read_sealed_artifact,
    unadjusted_cox_reference_estimator,
)
from cohorts.selection import observationalize
from cohorts.splitting import PatientSet, stratified_patient_split
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from infra.consumption_ledger import consumed_ledger_path
from infra.reference_guard import ReferenceAccessError, assert_reference_unlocked
from infra.seeding import derive_seed
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult
from tests.token_fixtures import SYNTHETIC_BINDINGS, synthetic_token

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
    token = synthetic_token(issued_by="test")

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
    token = synthetic_token(issued_by="scripts/check_go_no_go.py")

    internal_reference = sealed_reference.unseal(token)

    assert isinstance(internal_reference, InternalReference)


def test_unadjusted_cox_reference_estimator_produces_a_real_log_hazard_ratio(
    randomized_cohort: RandomizedCohort,
):
    """`unadjusted_cox_reference_estimator` (the real, non-placeholder
    estimator wired in for real-data runs) produces an actual numeric
    log-hazard-ratio value, not `None`."""
    sealed = SealedReference(randomized_cohort, estimator=unadjusted_cox_reference_estimator)
    token = synthetic_token(issued_by="test")

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

    token = synthetic_token(issued_by="test")
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
    token = synthetic_token(issued_by="test")

    with pytest.raises(FileNotFoundError):
        read_sealed_artifact(missing_path, unlock_token=token)


def test_locked_evaluation_token_requires_non_empty_issued_by():
    """The original empty-`issued_by` rejection, unchanged in substance.

    `TokenFieldError` subclasses `ValueError`, so this assertion is
    identical to the pre-task-7 one: the nine-field rejection *subsumes*
    the two-field one rather than replacing it (`design.md` Section 4.6).
    """
    with pytest.raises(ValueError):
        LockedEvaluationToken(
            issued_by="",
            issued_at=datetime.now(),
            **SYNTHETIC_BINDINGS,
        )


def test_assert_reference_unlocked_matches_unseal_behavior(sealed_reference: SealedReference):
    """`SealedReference.unseal()` delegates to
    `infra.reference_guard.assert_reference_unlocked` rather than
    duplicating the check -- confirm the standalone function agrees.
    """
    with pytest.raises(ReferenceAccessError):
        assert_reference_unlocked(None)

    valid_token = synthetic_token(issued_by="test")
    assert_reference_unlocked(valid_token)  # returns silently, no exception


# ---------------------------------------------------------------------------
# Version-2 sealed reference (task 6): the extended estimator, the recorded
# format version, and the refusal of old-format artifacts.
#
# **Validates: Requirements 14.1, 14.2, 14.6, 14.7 (Correctness Property 25)**
# ---------------------------------------------------------------------------


def _extended_patient_record(
    patient_id: str,
    *,
    treatment_arm: int,
    event_time_weeks: float,
    event_indicator: bool,
    baseline_cd4: float = 350.0,
    karnofsky_score: float = 90.0,
    age: float = 34.0,
    symptomatic_status: bool = False,
    prior_opportunistic_infection: bool = False,
    prior_art_exposure_days: float = 0.0,
) -> PatientRecord:
    """A `PatientRecord` rich enough for the *extended* estimator.

    `extended_randomized_reference_estimator` evaluates a per-arm
    Kaplan-Meier curve on `causal.baselines._elapsed_weeks`' time scale,
    which requires a `modality="treatment"` event (time zero) -- the
    module-level `_make_patient_record` above deliberately carries only the
    two baseline events the *Cox* path needs, so this builder adds the
    treatment event and the six selection covariates
    `cohorts.selection.observationalize` reads. Same hand-built-event style
    as `tests/unit/test_baselines.py`.
    """
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
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=build_patient_timeline(events),
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME + timedelta(weeks=event_time_weeks),
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="aids_defining_event" if event_indicator else None,
        ),
    )


def _extended_cohort(n_patients: int = 80) -> RandomizedCohort:
    """A deterministic synthetic `RandomizedCohort` with differing arm-wise
    96-week risks (arm 0 events cluster before the horizon, arm 1 events
    after it), so the two `risk` quantities and their difference are
    non-degenerate and any recomputation over a strict subset is visibly a
    different number.

    Deliberately arithmetic rather than random: this test asserts *which
    cohort* the reference was computed over, so the cohort itself must not
    move between runs.
    """
    patients = []
    for index in range(n_patients):
        arm = index % 2
        # Arm 0: events every other patient, spread over 20-90 weeks (before
        # the 96-week horizon). Arm 1: events pushed past the horizon.
        if arm == 0:
            event_indicator = index % 4 == 0
            event_time_weeks = 20.0 + (index % 8) * 9.0
            # Task 7.8: a few arm-0 patients are followed (censored) past
            # every arm-1 event. Without them every arm-1 patient outlives
            # every arm-0 patient, the Cox partial likelihood is monotone
            # with no finite maximum, and the converged Newton fit correctly
            # refuses; the old 300-step descent returned an arbitrary number.
            if index % 8 == 6:
                event_time_weeks = 140.0
        else:
            event_indicator = index % 6 == 1
            event_time_weeks = 100.0 + (index % 5) * 7.0
        patients.append(
            _extended_patient_record(
                f"p{index:03d}",
                treatment_arm=arm,
                event_time_weeks=event_time_weeks,
                event_indicator=event_indicator,
                baseline_cd4=200.0 + (index % 11) * 25.0,
                karnofsky_score=70.0 + (index % 4) * 10.0,
                age=25.0 + (index % 30),
                symptomatic_status=index % 3 == 0,
                prior_opportunistic_infection=index % 7 == 0,
                prior_art_exposure_days=float((index % 5) * 120),
            )
        )
    return RandomizedCohort(patients=tuple(patients))


def test_extended_estimator_carries_the_four_scale_tagged_quantities():
    """Requirement 14.1: the sealed reference carries the randomized Cox
    log-hazard-ratio, the randomized 96-week risk for each arm, and the
    randomized 96-week risk difference -- each tagged with its scale.
    """
    cohort = _extended_cohort()

    reference = extended_randomized_reference_estimator(cohort)

    assert reference.format_version == SEALED_FORMAT_VERSION == 2
    assert set(reference.quantities) == {
        "log_hazard_ratio",
        "risk_arm0",
        "risk_arm1",
        "risk_difference",
    }

    log_hr = reference.quantities["log_hazard_ratio"]
    assert log_hr.scale == "log_hazard_ratio"
    # A Cox coefficient is a whole-follow-up quantity, not a horizon one.
    assert log_hr.horizon_weeks is None

    for name in ("risk_arm0", "risk_arm1"):
        quantity = reference.quantities[name]
        assert quantity.scale == "risk"
        assert quantity.horizon_weeks == HORIZON_WEEKS == 96.0
        assert 0.0 <= quantity.value <= 1.0

    risk_difference = reference.quantities["risk_difference"]
    assert risk_difference.scale == "risk_difference"
    assert risk_difference.horizon_weeks == HORIZON_WEEKS
    assert risk_difference.value == pytest.approx(
        reference.quantities["risk_arm1"].value - reference.quantities["risk_arm0"].value
    )
    # Non-degenerate: the two arms really do have different 96-week risks in
    # this fixture, so nothing below can pass by both sides being zero.
    assert abs(risk_difference.value) > 1e-6


def test_extended_estimator_value_is_byte_identical_to_the_v1_estimator():
    """`design.md` Section 4.5: "`value` keeps its existing meaning so v1
    consumers and existing tests are unaffected."

    Compared at the bit level (`struct.pack`), not just with `==`, so a
    silently re-derived-but-nearly-equal log-hazard ratio would fail.
    """
    cohort = _extended_cohort()

    v1 = unadjusted_cox_reference_estimator(cohort)
    v2 = extended_randomized_reference_estimator(cohort)

    assert isinstance(v2.value, float)
    assert struct.pack("<d", v2.value) == struct.pack("<d", v1.value)
    # The same number is also reachable by name, on its declared scale.
    assert struct.pack("<d", v2.quantities["log_hazard_ratio"].value) == struct.pack(
        "<d", v1.value
    )
    assert v2.metadata["scale"] == v1.metadata["scale"] == "log_hazard_ratio"
    # ... and survives the artifact round trip unchanged, byte for byte.
    assert json.dumps(v2.value) == json.dumps(v1.value)


def test_v1_estimators_carry_no_quantities():
    """The two version-1-shaped estimators are left unchanged: they keep
    reporting only `value`, so no existing consumer starts seeing a
    `quantities` mapping it never asked for.
    """
    cohort = _extended_cohort()

    assert unadjusted_cox_reference_estimator(cohort).quantities == {}
    assert placeholder_estimator(cohort).quantities == {}


def test_extended_quantities_are_computed_pre_split_and_pre_observationalization():
    """Requirement 14.2: all three quantities are computed on the complete
    eligible arm-0/arm-1 cohort, before splitting and before
    observationalization (`decisions.md` Section 9's frozen order).

    Asserted two ways. Structurally: `SealedReference.__init__` accepts only
    a `RandomizedCohort`, so a post-split `PatientSet` cannot reach the
    estimator at all. Numerically: the sealed quantities equal a
    complete-cohort recomputation and differ from what the *same* estimator
    produces on the real post-split and post-observationalization patient
    sets -- so they cannot have been computed after either step.
    """
    cohort = _extended_cohort()
    token = synthetic_token(issued_by="test")

    sealed = SealedReference(cohort, estimator=extended_randomized_reference_estimator)
    reference = sealed.unseal(token)

    # Computed over the complete cohort: size and per-arm counts are the
    # whole cohort's, and the values match a direct complete-cohort call.
    complete = extended_randomized_reference_estimator(cohort)
    assert reference.cohort_size == len(cohort)
    assert reference.metadata["arm_0_count"] == len(cohort.by_arm(0))
    assert reference.metadata["arm_1_count"] == len(cohort.by_arm(1))
    for name in complete.quantities:
        assert reference.quantities[name].value == complete.quantities[name].value

    # Structural: both the sealing constructor and the estimator declare the
    # pre-split cohort type, and the estimator genuinely cannot consume a
    # post-split `PatientSet` -- it reads `by_arm`, which only the pre-split
    # `RandomizedCohort` has.
    assert (
        get_type_hints(SealedReference.__init__)["cohort"] is RandomizedCohort
    )
    assert (
        get_type_hints(extended_randomized_reference_estimator)["cohort"] is RandomizedCohort
    )
    assert not hasattr(PatientSet(patients=cohort.patients), "by_arm")
    with pytest.raises(AttributeError):
        extended_randomized_reference_estimator(
            PatientSet(patients=cohort.patients)  # type: ignore[arg-type]
        )

    # Pre-SPLIT: run the real Stage-5 split and recompute on Development_Split.
    splits = stratified_patient_split(cohort, sealed, seed=101)
    assert 0 < len(splits.dev) < len(cohort)
    post_split = extended_randomized_reference_estimator(
        RandomizedCohort(patients=splits.dev.patients)
    )
    assert post_split.cohort_size < reference.cohort_size
    assert post_split.quantities["risk_difference"].value != (
        reference.quantities["risk_difference"].value
    )

    # Pre-OBSERVATIONALIZATION: run the real Stage-7 selection mechanism on
    # that split and recompute on the retained patients.
    fitted = FittedPreprocessing().fit(DevelopmentSet(splits.dev))
    observationalized = observationalize(
        splits.dev, fitted, seed=derive_seed(101, "development")
    )
    assert 0 < len(observationalized.retained) < len(splits.dev)
    post_selection = extended_randomized_reference_estimator(
        RandomizedCohort(patients=observationalized.retained.patients)
    )
    assert post_selection.cohort_size < post_split.cohort_size
    assert post_selection.quantities["risk_difference"].value != (
        reference.quantities["risk_difference"].value
    )


def test_write_sealed_artifact_records_the_format_version_and_quantities():
    """Requirement 14.6: the written artifact records the sealed-artifact
    format version. Read back as raw JSON (not through
    `read_sealed_artifact`) so the assertion is about the bytes on disk.
    """
    with tempfile.TemporaryDirectory() as directory:
        artifact_path = Path(directory) / "internal_randomized_reference.v2.sealed.json"
        sealed = SealedReference(
            _extended_cohort(), estimator=extended_randomized_reference_estimator
        )
        sealed.write_sealed_artifact(artifact_path)

        payload = json.loads(artifact_path.read_text())

    assert payload["format_version"] == 2
    assert set(payload["quantities"]) == {
        "log_hazard_ratio",
        "risk_arm0",
        "risk_arm1",
        "risk_difference",
    }
    assert payload["quantities"]["risk_difference"]["scale"] == "risk_difference"
    assert payload["quantities"]["risk_difference"]["horizon_weeks"] == 96.0
    # The version-1 keys are still there, unchanged in meaning.
    assert set(payload) >= {"value", "cohort_size", "computed_at", "metadata"}


def test_v2_artifact_round_trips_including_quantities():
    with tempfile.TemporaryDirectory() as directory:
        artifact_path = Path(directory) / "internal_randomized_reference.v2.sealed.json"
        sealed = SealedReference(
            _extended_cohort(), estimator=extended_randomized_reference_estimator
        )
        sealed.write_sealed_artifact(artifact_path)

        token = synthetic_token(issued_by="test")
        reconstructed = read_sealed_artifact(artifact_path, unlock_token=token)
        original = sealed.unseal(token)

    assert reconstructed.format_version == 2
    assert set(reconstructed.quantities) == set(original.quantities)
    for name, quantity in original.quantities.items():
        assert reconstructed.quantities[name] == quantity


def _write_artifact_with_version(directory: Path, version: int | None) -> Path:
    """Write a real sealed artifact, then rewrite its JSON with `version` in
    the `format_version` slot (or with the slot removed entirely, for
    `None` -- exactly the shape a genuine version-1 artifact has, since the
    field did not exist when it was written).
    """
    artifact_path = directory / "internal_randomized_reference.sealed.json"
    sealed = SealedReference(
        _extended_cohort(n_patients=20), estimator=extended_randomized_reference_estimator
    )
    sealed.write_sealed_artifact(artifact_path)

    payload = json.loads(artifact_path.read_text())
    if version is None:
        payload.pop("format_version")
    else:
        payload["format_version"] = version
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return artifact_path


@settings(max_examples=25, deadline=None)
@given(version=st.one_of(st.none(), st.integers(min_value=-3, max_value=6)))
@example(version=None)  # absent -> treated as version 1
@example(version=0)
@example(version=1)
@example(version=2)
@example(version=3)
def test_property_25_old_format_sealed_artifacts_are_refused(version: int | None):
    """**Correctness Property 25: Old-format sealed artifacts are refused.**

    *For any* sealed artifact whose recorded format version is absent or
    below 2, loading raises `SealedFormatVersionError` naming the documented
    re-seal step, and loading succeeds only at version 2 or above.

    **Validates: Requirements 14.7**
    """
    token = synthetic_token(issued_by="test")

    with tempfile.TemporaryDirectory() as directory:
        artifact_path = _write_artifact_with_version(Path(directory), version)

        effective_version = 1 if version is None else version
        if effective_version < SEALED_FORMAT_VERSION:
            with pytest.raises(SealedFormatVersionError) as excinfo:
                read_sealed_artifact(artifact_path, unlock_token=token)
            message = str(excinfo.value)
            # Names the documented re-seal step, verbatim.
            assert RESEAL_INSTRUCTIONS in message
            assert "--stage seal-reference --reseal" in message
            assert "internal_randomized_reference.v2.sealed.json" in message
        else:
            loaded = read_sealed_artifact(artifact_path, unlock_token=token)
            assert loaded.format_version == effective_version


def test_format_version_is_checked_after_authorization_and_before_payload_parsing():
    """`design.md` Section 2.6 tension 4: the version field is read *after*
    `assert_reference_unlocked` and *before* parsing the payload.

    Both halves are observable. Without a token an old-format artifact still
    raises `ReferenceAccessError` (authorization first, not
    `SealedFormatVersionError`). With a token, an old-format artifact whose
    payload is otherwise unparseable still raises `SealedFormatVersionError`
    rather than a `KeyError` from payload parsing.
    """
    token = synthetic_token(issued_by="test")

    with tempfile.TemporaryDirectory() as directory:
        artifact_path = Path(directory) / "v1.sealed.json"
        # A version-1 artifact whose remaining keys are absent: parsing it
        # would raise KeyError, so the version gate must fire first.
        artifact_path.write_text(json.dumps({"unparseable": True}))

        with pytest.raises(ReferenceAccessError):
            read_sealed_artifact(artifact_path, unlock_token=None)

        with pytest.raises(SealedFormatVersionError):
            read_sealed_artifact(artifact_path, unlock_token=token)


def test_non_integer_format_version_is_refused():
    token = synthetic_token(issued_by="test")

    with tempfile.TemporaryDirectory() as directory:
        artifact_path = Path(directory) / "bogus.sealed.json"
        artifact_path.write_text(json.dumps({"format_version": "2", "value": 0.0}))

        with pytest.raises(SealedFormatVersionError):
            read_sealed_artifact(artifact_path, unlock_token=token)


# ---------------------------------------------------------------------------
# The nine-field `LockedEvaluationToken` (task 7): every binding required
# with no default, every offender named in one error, and the consumption
# ledger consulted at mint time.
#
# Scope note: Property 26 ("Minting names every offender") quantifies over
# offending subsets of the nine bound fields *and* the thirteen checklist
# items, and belongs to task 20's `tests/unit/test_go_no_go.py`, which owns
# the checklist half. What is asserted here is the token half only: the
# dataclass and `_mint` contracts, exercised by example over every field.
#
# **Validates: Requirements 13.3, 13.4**
# ---------------------------------------------------------------------------

TOKEN_STRING_FIELDS: tuple[str, ...] = (
    "issued_by",
    "resolved_config_hash",
    "dataset_checksum",
    "split_hash",
    "git_commit",
    "statistical_plan_version",
    "evaluation_id",
)


def _valid_token_kwargs() -> dict:
    """Every constructor argument for a valid token, as a plain mapping.

    Used by the tests below to remove or corrupt exactly one binding at a
    time. Built from `SYNTHETIC_BINDINGS` so it cannot drift from the shared
    factory.
    """
    return {
        "issued_by": "tests/unit/test_sealed_reference.py",
        "issued_at": datetime.now(UTC),
        **SYNTHETIC_BINDINGS,
    }


def test_all_nine_bound_fields_are_declared_without_defaults():
    """Requirement 13.3's "immutable fields" plus `design.md` Section 4.6's
    "all required, no defaults", checked against the dataclass itself
    rather than against a hand-maintained list.
    """
    declared = {f.name: f for f in dataclasses.fields(LockedEvaluationToken)}

    assert tuple(declared) == BOUND_TOKEN_FIELDS
    assert len(BOUND_TOKEN_FIELDS) == 9
    for name, field_spec in declared.items():
        assert field_spec.default is dataclasses.MISSING, f"{name} has a default"
        assert field_spec.default_factory is dataclasses.MISSING, f"{name} has a factory default"


@pytest.mark.parametrize("omitted", BOUND_TOKEN_FIELDS)
def test_no_partially_bound_token_can_be_constructed(omitted: str):
    """Omitting any single binding is a `TypeError` from Python itself.

    This is the structural half of "no partially bound token can exist": it
    is not a validation message but an unsatisfiable call signature, so the
    failure happens before `__post_init__` runs and cannot be bypassed by
    catching a `ValueError`.
    """
    kwargs = _valid_token_kwargs()
    del kwargs[omitted]

    with pytest.raises(TypeError, match=omitted):
        LockedEvaluationToken(**kwargs)


def test_mint_requires_every_binding_as_a_keyword():
    """`_mint("who")` alone no longer produces a token.

    The seven new parameters are keyword-only, so a minter cannot transpose
    two same-shaped hex strings positionally either.
    """
    with pytest.raises(TypeError):
        LockedEvaluationToken._mint("scripts/check_go_no_go.py")


@pytest.mark.parametrize("field_name", BOUND_TOKEN_FIELDS)
def test_an_absent_binding_is_rejected_by_name(field_name: str):
    """`None` in any slot raises `TokenFieldError` naming that slot."""
    kwargs = _valid_token_kwargs()
    kwargs[field_name] = None

    with pytest.raises(TokenFieldError) as excinfo:
        LockedEvaluationToken(**kwargs)

    assert field_name in str(excinfo.value)
    assert "absent" in str(excinfo.value)


@pytest.mark.parametrize("field_name", TOKEN_STRING_FIELDS)
def test_an_empty_binding_is_rejected_by_name(field_name: str):
    """An empty or whitespace-only string is "empty", not "present"."""
    for empty_value in ("", "   "):
        kwargs = _valid_token_kwargs()
        kwargs[field_name] = empty_value

        with pytest.raises(TokenFieldError) as excinfo:
            LockedEvaluationToken(**kwargs)

        assert field_name in str(excinfo.value)
        assert "empty" in str(excinfo.value)


def test_a_multi_field_violation_names_every_offender_in_one_error():
    """Requirement 13.4: "SHALL name every offending field" -- all of them,
    from a single raise, not the first one met.

    Four offenders of three different kinds at once (empty, absent, empty
    tuple), and the message must name all four and none of the five valid
    fields.
    """
    kwargs = _valid_token_kwargs()
    kwargs["issued_by"] = ""
    kwargs["dataset_checksum"] = None
    kwargs["split_hash"] = "   "
    kwargs["checkpoint_hashes"] = ()

    with pytest.raises(TokenFieldError) as excinfo:
        LockedEvaluationToken(**kwargs)

    message = str(excinfo.value)
    offenders = {"issued_by", "dataset_checksum", "split_hash", "checkpoint_hashes"}
    for name in offenders:
        assert name in message, f"{name} not named in {message!r}"
    for name in set(BOUND_TOKEN_FIELDS) - offenders:
        assert name not in message, f"valid field {name} named as an offender"
    assert "4 offending field(s)" in message


def test_offenders_are_named_in_declaration_order():
    """Two runs with the same offending set produce the same message, so a
    failure is diffable rather than set-ordering-dependent.
    """
    kwargs = _valid_token_kwargs()
    kwargs["evaluation_id"] = ""
    kwargs["issued_by"] = ""

    with pytest.raises(TokenFieldError) as excinfo:
        LockedEvaluationToken(**kwargs)

    message = str(excinfo.value)
    assert message.index("issued_by") < message.index("evaluation_id")


def test_token_field_error_subsumes_the_original_value_error():
    """The pre-task-7 rejection raised a bare `ValueError`; callers that
    caught it must keep working.
    """
    assert issubclass(TokenFieldError, ValueError)


@pytest.mark.parametrize(
    "bad_value",
    [
        (),
        [("a", "b")],
        (("only_one_element",),),
        (("name", ""),),
        (("", "sha"),),
        ("not-a-tuple-of-pairs",),
    ],
    ids=[
        "empty",
        "list-not-tuple",
        "one-element-pair",
        "empty-sha",
        "empty-logical-name",
        "bare-string-entry",
    ],
)
def test_malformed_checkpoint_hashes_are_rejected(bad_value):
    """`checkpoint_hashes` must be a non-empty tuple of `(logical_name,
    sha256)` pairs of non-empty strings (`design.md` Section 4.6).

    The empty case matters on its own: an evaluation that binds no
    checkpoint hash has nothing to recompute under Requirement 13.5.
    """
    kwargs = _valid_token_kwargs()
    kwargs["checkpoint_hashes"] = bad_value

    with pytest.raises(TokenFieldError, match="checkpoint_hashes"):
        LockedEvaluationToken(**kwargs)


def test_unsorted_checkpoint_hashes_are_rejected_and_mint_sorts_for_you():
    """The bound checkpoint set has one canonical ordering.

    Directly constructing an unsorted tuple is refused (the dataclass is
    frozen, so it cannot be repaired in place); `_mint` sorts its argument,
    so the designated minter never has to care what order it discovered the
    checkpoint files in.
    """
    unsorted_pairs = (
        ("intervention_aware", "sha-b"),
        ("baseline_gru", "sha-a"),
    )

    kwargs = _valid_token_kwargs()
    kwargs["checkpoint_hashes"] = unsorted_pairs
    with pytest.raises(TokenFieldError, match="not sorted"):
        LockedEvaluationToken(**kwargs)

    minted = synthetic_token(checkpoint_hashes=unsorted_pairs)
    assert minted.checkpoint_hashes == tuple(sorted(unsorted_pairs))


def test_mint_refuses_an_evaluation_id_already_recorded_as_consumed(tmp_path):
    """Requirement 13.4's "duplicated": uniqueness lives in the append-only
    consumption ledger, not in a token field, because the token is frozen
    and serializable.
    """
    ledger = consumed_ledger_path(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "evaluation_id": "eval-already-run",
                "consumed_at": datetime.now(UTC).isoformat(),
                "result_directory": "results/locked_evaluation/eval-already-run",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TokenFieldError) as excinfo:
        synthetic_token(results_root=tmp_path, evaluation_id="eval-already-run")
    assert "evaluation_id" in str(excinfo.value)
    assert "consumed" in str(excinfo.value)

    # A different id under the same ledger mints fine -- the refusal is
    # about this identifier, not about the ledger existing.
    assert synthetic_token(results_root=tmp_path, evaluation_id="eval-fresh") is not None


def test_mint_reports_a_field_offense_and_a_duplicate_together(tmp_path):
    """Neither kind of offense hides the other: a run with both a malformed
    binding and a reused `evaluation_id` learns about both at once.
    """
    ledger = consumed_ledger_path(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"evaluation_id": "eval-already-run"}) + "\n", encoding="utf-8")

    with pytest.raises(TokenFieldError) as excinfo:
        synthetic_token(
            results_root=tmp_path,
            evaluation_id="eval-already-run",
            git_commit="",
        )

    message = str(excinfo.value)
    assert "git_commit" in message
    assert "evaluation_id" in message
    assert "2 offending field(s)" in message


def test_a_fully_bound_minted_token_still_passes_the_unchanged_guard():
    """The whole point of adding fields to the same class: the guard was not
    touched, and an `isinstance` check against a nine-field class is
    strictly stronger than one against a two-field class.
    """
    token = synthetic_token()

    assert isinstance(token, LockedEvaluationToken)
    assert_reference_unlocked(token)  # returns silently
    assert token.resolved_config_hash == SYNTHETIC_BINDINGS["resolved_config_hash"]
    assert token.evaluation_id == SYNTHETIC_BINDINGS["evaluation_id"]


def test_the_token_is_still_frozen():
    """`@dataclass(frozen=True)` is part of the contract, not incidental:
    every field is a *binding*, and a mutable binding authorizes nothing.
    """
    token = synthetic_token()

    with pytest.raises(dataclasses.FrozenInstanceError):
        token.evaluation_id = "something-else"
