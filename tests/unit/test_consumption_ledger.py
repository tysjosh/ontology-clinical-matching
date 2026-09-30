"""The locked-evaluation consumption and reference-access ledgers
(`src/infra/consumption_ledger.py`), over a `tmp_path` results root.

Covers the three things task 7.1 has to be true:

- a duplicated `evaluation_id` is refused --- at mint time (already wired
  in task 7) and again by `append_consumed`, the last point a second record
  could still be written;
- that refusal happens before any sealed file is opened, evidenced by the
  sealed artifact staying byte-identical *and* no reference-access record
  existing (an opened sealed reference always leaves one, so the absence of
  a record is the absence of an open);
- exactly one access record is appended per successful unseal.

**Validates: Requirements 13.6, 14.5, 12.4**
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cohorts.cohort_construction import RandomizedCohort
from cohorts.sealed_reference import (
    SEALED_FORMAT_VERSION,
    InternalReference,
    SealedReference,
    TokenFieldError,
    read_sealed_artifact,
)
from infra.consumption_ledger import (
    RESULTS_ROOT_ENV_VAR,
    ConsumptionLedgerError,
    DuplicateConsumptionError,
    ReferenceAccessLedgerError,
    ResultsRootNotConfiguredError,
    append_consumed,
    append_reference_access,
    consumed_evaluation_ids,
    consumed_ledger_path,
    is_consumed,
    reference_access_ledger_path,
    reference_access_records,
    reference_access_timestamp,
)
from infra.reference_guard import ReferenceAccessError
from tests.token_fixtures import synthetic_token


def _reference(cohort: RandomizedCohort) -> InternalReference:
    """A `ReferenceEstimator` that ignores the cohort.

    These tests are about access accounting, not about the estimate: what
    the sealed value *is* never enters an assertion, so an empty
    `RandomizedCohort` plus a fixed value keeps the setup to one line and
    keeps the patient-record machinery (exercised in
    `test_sealed_reference.py` and `test_cohort_construction.py`) out of
    it.
    """
    return InternalReference(
        value=0.5,
        cohort_size=len(cohort),
        computed_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"estimator": "test_consumption_ledger._reference"},
        format_version=SEALED_FORMAT_VERSION,
    )


@pytest.fixture
def sealed() -> SealedReference:
    return SealedReference(RandomizedCohort(patients=()), estimator=_reference)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------
# consumed.jsonl
# --------------------------------------------------------------------------


def test_append_consumed_records_the_timestamp_and_result_directory(tmp_path: Path):
    consumed_at = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    result_directory = tmp_path / "locked_evaluation" / "eval-1"

    ledger = append_consumed(
        "eval-1",
        tmp_path,
        result_directory=result_directory,
        consumed_at=consumed_at,
    )

    assert ledger == consumed_ledger_path(tmp_path)
    assert _lines(ledger) == [
        {
            "evaluation_id": "eval-1",
            "consumed_at": consumed_at.isoformat(),
            "result_directory": str(result_directory),
        }
    ]
    assert is_consumed("eval-1", tmp_path)
    assert not is_consumed("eval-2", tmp_path)


def test_append_consumed_is_append_only_across_distinct_ids(tmp_path: Path):
    append_consumed("eval-1", tmp_path, result_directory="results/locked_evaluation/eval-1")
    first_line = consumed_ledger_path(tmp_path).read_text(encoding="utf-8")

    append_consumed("eval-2", tmp_path, result_directory="results/locked_evaluation/eval-2")

    contents = consumed_ledger_path(tmp_path).read_text(encoding="utf-8")
    # The first record's bytes are still the file's prefix: nothing was
    # rewritten, only appended.
    assert contents.startswith(first_line)
    assert consumed_evaluation_ids(tmp_path) == ("eval-1", "eval-2")


def test_append_consumed_refuses_a_duplicate_and_leaves_the_ledger_byte_identical(
    tmp_path: Path,
):
    append_consumed("eval-1", tmp_path, result_directory="results/locked_evaluation/eval-1")
    ledger = consumed_ledger_path(tmp_path)
    before = ledger.read_bytes()

    with pytest.raises(DuplicateConsumptionError, match="eval-1"):
        append_consumed(
            "eval-1", tmp_path, result_directory="results/locked_evaluation/eval-1-again"
        )

    assert ledger.read_bytes() == before
    assert consumed_evaluation_ids(tmp_path) == ("eval-1",)


def test_append_consumed_rejects_an_empty_evaluation_id(tmp_path: Path):
    with pytest.raises(ConsumptionLedgerError):
        append_consumed("   ", tmp_path, result_directory=tmp_path)

    assert not consumed_ledger_path(tmp_path).exists()


def test_a_duplicated_evaluation_id_is_refused_before_any_sealed_file_is_opened(
    tmp_path: Path, sealed: SealedReference
):
    """Requirement 13.6: a consumed `evaluation_id` is refused with every
    existing artifact and the consumption record left byte-identical, and
    without the sealed reference being opened.

    The only route to the sealed reference is a `LockedEvaluationToken`, and
    minting one for a consumed id is refused --- so the refusal necessarily
    precedes any open. That is asserted positively rather than by
    inspection: the sealed artifact's bytes and mtime are unchanged, and no
    reference-access record exists anywhere, which is the observable
    signature of "nothing opened the sealed reference" now that every
    authorized open appends one.
    """
    artifact = sealed.write_sealed_artifact(tmp_path / "reference" / "sealed.json")
    append_consumed("eval-1", tmp_path, result_directory=tmp_path / "eval-1")

    artifact_before = artifact.read_bytes()
    artifact_mtime_before = artifact.stat().st_mtime_ns
    ledger_before = consumed_ledger_path(tmp_path).read_bytes()

    with pytest.raises(TokenFieldError, match="already recorded as consumed") as excinfo:
        synthetic_token(evaluation_id="eval-1", results_root=tmp_path)

    assert "evaluation_id" in str(excinfo.value)

    assert artifact.read_bytes() == artifact_before
    assert artifact.stat().st_mtime_ns == artifact_mtime_before
    assert consumed_ledger_path(tmp_path).read_bytes() == ledger_before
    assert not reference_access_ledger_path(tmp_path).exists()


# --------------------------------------------------------------------------
# reference_access.jsonl
# --------------------------------------------------------------------------


def test_each_successful_unseal_appends_exactly_one_access_record(
    tmp_path: Path, sealed: SealedReference
):
    token = synthetic_token(evaluation_id="eval-unseal")

    sealed.unseal(token, results_root=tmp_path)

    records = reference_access_records(tmp_path)
    assert len(records) == 1
    assert records[0][0] == "eval-unseal"

    # A second authorized read is a second access, not a de-duplicated
    # repeat: this ledger counts accesses.
    sealed.unseal(token, results_root=tmp_path)
    assert [evaluation_id for evaluation_id, _ in reference_access_records(tmp_path)] == [
        "eval-unseal",
        "eval-unseal",
    ]


def test_a_refused_unseal_appends_nothing(tmp_path: Path, sealed: SealedReference):
    """Only an *authorized* access is recorded: the guard runs first, so an
    unauthorized attempt leaves no record (and no ledger file at all)."""
    with pytest.raises(ReferenceAccessError):
        sealed.unseal(None, results_root=tmp_path)

    with pytest.raises(ReferenceAccessError):
        read_sealed_artifact(tmp_path / "sealed.json", unlock_token=None, results_root=tmp_path)

    assert not reference_access_ledger_path(tmp_path).exists()


def test_read_sealed_artifact_appends_exactly_one_access_record(
    tmp_path: Path, sealed: SealedReference
):
    artifact = sealed.write_sealed_artifact(tmp_path / "reference" / "sealed.json")
    token = synthetic_token(evaluation_id="eval-read")

    loaded = read_sealed_artifact(artifact, unlock_token=token, results_root=tmp_path)

    assert loaded.value == 0.5
    assert [evaluation_id for evaluation_id, _ in reference_access_records(tmp_path)] == [
        "eval-read"
    ]


def test_read_sealed_artifact_records_the_access_even_when_the_read_then_fails(
    tmp_path: Path,
):
    """The record is appended after authorization and before the file is
    opened, so an authorized attempt that fails on a missing file still
    leaves a trace --- recording only successful parses would let a reader
    learn whether the artifact exists and leave no audit record."""
    token = synthetic_token(evaluation_id="eval-missing")

    with pytest.raises(FileNotFoundError):
        read_sealed_artifact(
            tmp_path / "does_not_exist.json", unlock_token=token, results_root=tmp_path
        )

    assert [evaluation_id for evaluation_id, _ in reference_access_records(tmp_path)] == [
        "eval-missing"
    ]


def test_reference_access_timestamp_reports_the_first_access(tmp_path: Path):
    first = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    append_reference_access("eval-1", tmp_path, accessed_at=first)
    append_reference_access("eval-1", tmp_path, accessed_at=first + timedelta(hours=3))
    append_reference_access("eval-2", tmp_path, accessed_at=first + timedelta(hours=4))

    assert reference_access_timestamp("eval-1", tmp_path) == first
    assert reference_access_timestamp("eval-2", tmp_path) == first + timedelta(hours=4)
    # No access recorded is a real answer, not an error: the audit report's
    # field is `datetime | None`.
    assert reference_access_timestamp("eval-never", tmp_path) is None
    assert reference_access_timestamp("eval-1", tmp_path / "empty_root") is None


def test_an_unrecordable_access_is_refused_rather_than_performed(
    tmp_path: Path, sealed: SealedReference, monkeypatch: pytest.MonkeyPatch
):
    """Requirement 14.5: with no results root resolvable there is nowhere to
    record the access, so `unseal` raises instead of returning the sealed
    reference. An unlogged access is the failure this refusal prevents."""
    monkeypatch.delenv(RESULTS_ROOT_ENV_VAR, raising=False)
    token = synthetic_token(evaluation_id="eval-unrecordable")

    with pytest.raises(ResultsRootNotConfiguredError, match=RESULTS_ROOT_ENV_VAR):
        sealed.unseal(token)

    with pytest.raises(ResultsRootNotConfiguredError):
        read_sealed_artifact(tmp_path / "sealed.json", unlock_token=token)


def test_the_environment_supplies_the_root_and_an_explicit_argument_overrides_it(
    tmp_path: Path, sealed: SealedReference, monkeypatch: pytest.MonkeyPatch
):
    environment_root = tmp_path / "from_environment"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv(RESULTS_ROOT_ENV_VAR, str(environment_root))
    token = synthetic_token(evaluation_id="eval-root")

    sealed.unseal(token)
    sealed.unseal(token, results_root=explicit_root)

    assert len(reference_access_records(environment_root)) == 1
    assert len(reference_access_records(explicit_root)) == 1


def test_a_malformed_access_ledger_is_an_error_not_an_empty_read(tmp_path: Path):
    ledger = reference_access_ledger_path(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)

    ledger.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ReferenceAccessLedgerError, match="not valid JSON"):
        reference_access_records(tmp_path)

    ledger.write_text(json.dumps({"evaluation_id": "eval-1"}) + "\n", encoding="utf-8")
    with pytest.raises(ReferenceAccessLedgerError, match="accessed_at"):
        reference_access_records(tmp_path)
