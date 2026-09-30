"""The locked-evaluation consumption and reference-access ledgers
(`neural-experiment-pipeline/design.md` Section 2.6 tension 5;
Requirements 13.4, 13.6, 14.5, 12.4).

Consumption tracking deliberately lives *outside* `LockedEvaluationToken`:
the token is a frozen dataclass that may be serialized to
`token_verification.json` and re-read, so "has this evaluation already
happened?" cannot be a mutable field on it. Design Section 2.6 tension 5
places the record in an append-only JSON-lines file:

    <results_root>/locked_evaluation/consumed.jsonl

one record per consumed `evaluation_id`, carrying the consumption timestamp
and the result directory the run committed to.

What this module holds
-----------------------

Two append-only JSON-lines ledgers, both under
`<results_root>/locked_evaluation/`, and the readers over them:

`consumed.jsonl` (Requirements 13.4, 13.6)
    `append_consumed(...)` writes one record per consumed `evaluation_id`
    (`{evaluation_id, consumed_at, result_directory}`) and is called by the
    locked-evaluation runner's atomic commit, after the result directory is
    in place. `is_consumed(evaluation_id, results_root)` is the single
    shared reader both consumption checks use:
    `cohorts.sealed_reference.LockedEvaluationToken._mint` calls it so a
    duplicated `evaluation_id` is refused at mint time (Requirement 13.4),
    and the runner calls it again *before* opening the test split or the
    sealed reference (Requirement 13.6). `append_consumed` re-checks it a
    third time and refuses to write a duplicate, so the append-only file
    cannot grow a second record for an id that already has one even if a
    caller skipped the earlier gates.

`reference_access.jsonl` (Requirements 14.5, 12.4)
    `append_reference_access(...)` writes one `{evaluation_id, accessed_at}`
    record per successful unseal. Its two callers are
    `cohorts.sealed_reference.SealedReference.unseal` and
    `cohorts.sealed_reference.read_sealed_artifact`, each appending
    immediately after its existing `assert_reference_unlocked` call.
    `reference_access_timestamp(evaluation_id, results_root)` is the reader
    that supplies the audit report's reference-access timestamp
    (Requirement 12.4); `reference_access_records` exposes the whole ledger
    for an auditor who wants every access, not just the first.

Unlike `consumed.jsonl`, repeated `evaluation_id`s in
`reference_access.jsonl` are legitimate: one locked evaluation unseals the
reference through `unseal` and may additionally read the persisted artifact
through `read_sealed_artifact`, and each of those is an access that
Requirement 14.5 wants recorded. Consumption is the one-shot event;
accesses within that one shot are many.

Where the reference-access ledger's root comes from
---------------------------------------------------

`SealedReference.unseal(token)` and `read_sealed_artifact(path, token)`
had no `results_root` parameter, so they did not know which ledger to
append to. Three candidate answers, and why this module resolves it the
way it does:

*Derive it from the artifact path* was rejected outright: `unseal` has no
path at all (it returns an in-memory reference computed at construction
time), and for `read_sealed_artifact` the walk-up from
`results/<experiment_id>/reference/internal_randomized_reference.sealed.json`
to `results/` is three `.parent`s of pure convention that silently
mis-targets the ledger the moment an artifact is read from anywhere else.

*Bind it to the token* was rejected too, though it is the tempting one ---
`_mint` already takes a `results_root`, and the token already carries the
`evaluation_id` the record needs. But `LockedEvaluationToken` is the frozen
carrier of exactly the nine bindings Requirement 13.3 enumerates, all
required with no defaults; a tenth field for "where to write the audit
log" is not a binding, and giving it a default would put a defaulted field
on the one class whose whole point is that it has none. A token is also
serializable to `token_verification.json` and re-readable, which would make
a baked-in filesystem path go stale the moment a results tree moved.

So the root is resolved by `resolve_results_root`, in this order:

1. an explicit `results_root=` keyword, which both call sites accept and
   which the locked-evaluation runner passes (it knows its own root);
2. the `CWPWM_RESULTS_ROOT` environment variable, which is how a process
   --- including a subprocess entry point, which is why this is an
   environment variable rather than a module-level global --- declares its
   results root once instead of threading it through every call;
3. nothing: `ResultsRootNotConfiguredError`.

Step 3 is the important one. The failure is *loud and refusing*, never a
skipped append: `unseal` raises instead of returning the reference, so a
process that cannot record an access does not get one. An access that
happened but was not logged is precisely what Requirement 14.5 exists to
prevent, and a silently-skipped append would produce exactly that while
looking like success. The pattern of "keyword wins, environment fills in,
absence is an error" follows the existing precedent in
`infra.checklist_verifiers` (`NESTED_VERIFICATION_ENV_VAR`), which likewise
uses an environment variable for process-scoped state a call signature
cannot carry.

`results_root` here
--------------------

`results_root` in this module is the **top-level** results directory
(`results/`), not the per-experiment `results/<experiment_id>` directory
that `infra.readiness`'s `results_root` refers to. A locked evaluation is
scoped to an `evaluation_id`, not to an experiment id, and uniqueness has
to hold across every experiment that ever minted a token --- a
per-experiment ledger could not detect a reused `evaluation_id` at all.

Reading style mirrors `infra.readiness.load_ledger_entries`: a missing file
is an empty ledger (a fresh results root has consumed nothing), while a
*present but malformed* file is an error rather than a silent empty read.
Silently treating an unparseable ledger as empty would turn a corrupted
file into "nothing has been consumed", which is exactly the wrong default
for a one-shot gate.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Path of the consumption ledger relative to the top-level results root
#: (`design.md` Section 2.6 tension 5).
CONSUMED_LEDGER_RELPATH: Path = Path("locked_evaluation") / "consumed.jsonl"

#: Path of the reference-access ledger relative to the top-level results
#: root (`design.md` Section 2.6 tension 5; Requirements 14.5, 12.4). Same
#: directory as `consumed.jsonl` deliberately: one locked evaluation's
#: one-shot record and its access trail belong to the same audit surface.
REFERENCE_ACCESS_LEDGER_RELPATH: Path = Path("locked_evaluation") / "reference_access.jsonl"

#: Environment variable naming the top-level results root for a process
#: that does not pass `results_root=` explicitly --- see the module
#: docstring, "Where the reference-access ledger's root comes from".
RESULTS_ROOT_ENV_VAR: str = "CWPWM_RESULTS_ROOT"


class ConsumptionLedgerError(ValueError):
    """The consumption ledger exists but cannot be read as one.

    An engineering failure, not a domain outcome (`design.md` Error
    Handling): a ledger that cannot be parsed cannot be used to answer
    "was this `evaluation_id` already consumed?", and answering "no" by
    default would let a one-shot evaluation run twice.
    """


class DuplicateConsumptionError(ConsumptionLedgerError):
    """`append_consumed` was asked to record an `evaluation_id` the ledger
    already holds (Requirements 13.4, 13.6).

    A `ConsumptionLedgerError` subclass so a caller that already catches
    the ledger's error type keeps catching this, while a caller that wants
    to distinguish "the ledger is unreadable" from "this evaluation already
    ran" can.
    """


class ReferenceAccessLedgerError(ValueError):
    """The reference-access ledger exists but cannot be read as one.

    Separate from `ConsumptionLedgerError` because the two ledgers answer
    different questions and fail for different reasons: an unreadable
    consumption ledger means "we cannot tell whether this evaluation
    already ran", while an unreadable access ledger means "we cannot tell
    when the sealed reference was opened". Collapsing them would make a
    traceback ambiguous about which audit surface is broken.
    """


class ResultsRootNotConfiguredError(ReferenceAccessLedgerError):
    """A reference access could not be recorded because no results root was
    resolvable --- no `results_root=` argument and no `CWPWM_RESULTS_ROOT`
    in the environment (Requirement 14.5).

    Raised *instead of* returning the sealed reference. The alternative ---
    skipping the append and handing the value over anyway --- would produce
    exactly the unlogged access Requirement 14.5 exists to prevent, while
    looking like a successful unseal. See the module docstring, "Where the
    reference-access ledger's root comes from".
    """


def consumed_ledger_path(results_root: str | Path) -> Path:
    """Path of the consumption ledger inside a top-level results root.

    Args:
        results_root: the top-level results directory (`results/`), not a
            per-experiment `results/<experiment_id>` directory --- see the
            module docstring.

    Returns:
        `<results_root>/locked_evaluation/consumed.jsonl`.
    """
    return Path(results_root) / CONSUMED_LEDGER_RELPATH


def consumed_evaluation_ids(results_root: str | Path) -> tuple[str, ...]:
    """Every `evaluation_id` recorded as consumed, in append order.

    Args:
        results_root: the top-level results directory (`results/`).

    Returns:
        The recorded ids in file order, or `()` when the ledger file does
        not exist. Duplicates are preserved rather than collapsed: this is
        a verbatim read of an append-only file, and a caller auditing the
        ledger should be able to see that a duplicate was ever written.

    Raises:
        ConsumptionLedgerError: if the file exists but any non-blank line
            is not a JSON object carrying a non-empty string
            `evaluation_id`.
    """
    path = consumed_ledger_path(results_root)
    records = _read_ledger_objects(
        path, ledger_name="consumption ledger", error_type=ConsumptionLedgerError
    )
    return tuple(
        _require_evaluation_id(
            record,
            path=path,
            line_number=line_number,
            ledger_name="consumption",
            error_type=ConsumptionLedgerError,
        )
        for line_number, record in records
    )


def is_consumed(evaluation_id: str, results_root: str | Path) -> bool:
    """Whether `evaluation_id` is already recorded as consumed.

    The shared reader for both consumption checks Requirement 13.4/13.6
    call for: `LockedEvaluationToken._mint` calls it so a reused
    `evaluation_id` is refused at mint time, `append_consumed` calls it so
    the ledger cannot grow a second record for an id it already holds, and
    task 21's locked-evaluation runner calls it again *before* opening the
    test split or the sealed reference.

    Args:
        evaluation_id: the candidate evaluation identifier.
        results_root: the top-level results directory (`results/`).

    Returns:
        `True` if the ledger records `evaluation_id`; `False` if it does
        not, including when the ledger file does not exist yet.

    Raises:
        ConsumptionLedgerError: if the ledger exists but is malformed (see
            `consumed_evaluation_ids`).
    """
    return evaluation_id in consumed_evaluation_ids(results_root)


def append_consumed(
    evaluation_id: str,
    results_root: str | Path,
    *,
    result_directory: str | Path,
    consumed_at: datetime | None = None,
) -> Path:
    """Record `evaluation_id` as consumed, append-only (Requirement 13.10).

    The write half of the ledger `is_consumed` reads. Called by the
    locked-evaluation runner's atomic commit, *after* the result directory
    has been moved into place: `design.md` Section 3.10 step 10 orders the
    commit as "`os.rename` the staging directory into place, then append to
    `consumed.jsonl`", so a crash between the two leaves an unrecorded
    result directory (visible, re-checkable) rather than a consumption
    record pointing at a directory that never appeared.

    The record is one JSON object on one line:
    `{"evaluation_id": ..., "consumed_at": <ISO-8601 UTC>,
    "result_directory": ...}` --- the three fields `design.md` Section 2.6
    tension 5 names. Existing lines are never rewritten, and the file is
    opened in append mode, so a reader that already read line *n* keeps
    seeing byte-identical bytes there.

    Args:
        evaluation_id: the identifier being consumed. Must be a non-empty
            string.
        results_root: the top-level results directory (`results/`).
        result_directory: the directory the run committed its result
            artifacts to (e.g.
            `results/locked_evaluation/<evaluation_id>/`). Recorded as a
            string exactly as given --- not resolved to an absolute path,
            so a relative path stays comparable across machines.
        consumed_at: consumption timestamp; defaults to `datetime.now(UTC)`.
            An explicit value is accepted so the runner can record the
            instant of its commit rather than the instant of this call, and
            so tests can pin it.

    Returns:
        The ledger path written to.

    Raises:
        ConsumptionLedgerError: if `evaluation_id` is not a non-empty
            string, or if the existing ledger cannot be read (an
            unreadable ledger cannot be appended to safely --- it cannot
            answer whether this id is already in it).
        DuplicateConsumptionError: if `evaluation_id` is already recorded.
            Nothing is written; the ledger stays byte-identical
            (Requirement 13.6).
    """
    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        raise ConsumptionLedgerError(
            f"Refusing to record consumption for evaluation_id={evaluation_id!r}: "
            "a consumption record must carry a non-empty string evaluation_id, "
            "otherwise the ledger cannot answer whether an evaluation already ran."
        )

    path = consumed_ledger_path(results_root)
    # Reads (and therefore validates) the whole existing ledger before
    # writing. The duplicate check is deliberately *here* as well as in
    # `LockedEvaluationToken._mint` and the runner's pre-flight gate: those
    # two run long before the commit, and this is the last point at which a
    # second record can still be refused.
    if evaluation_id in consumed_evaluation_ids(results_root):
        raise DuplicateConsumptionError(
            f"{path} already records evaluation_id={evaluation_id!r} as consumed. "
            "A locked evaluation is one-shot: its identifier is single-use across "
            "every token ever minted, so this append is refused and the ledger is "
            "left byte-identical (Requirement 13.6)."
        )

    _append_ledger_record(
        path,
        {
            "evaluation_id": evaluation_id,
            "consumed_at": (consumed_at or datetime.now(UTC)).isoformat(),
            "result_directory": str(result_directory),
        },
    )
    return path


def resolve_results_root(results_root: str | Path | None = None) -> Path:
    """The top-level results root to write audit records under.

    Resolution order --- explicit argument, then `CWPWM_RESULTS_ROOT`, then
    an error. See the module docstring ("Where the reference-access
    ledger's root comes from") for why the third case raises rather than
    skipping the record.

    Args:
        results_root: an explicit top-level results directory, or `None` to
            fall back to the environment.

    Returns:
        The resolved root as a `Path`. Not created here --- the writers
        create the ledger's parent directory when they append.

    Raises:
        ResultsRootNotConfiguredError: if `results_root` is `None` (or
            blank) and `CWPWM_RESULTS_ROOT` is unset or blank.
    """
    if results_root is not None and str(results_root).strip():
        return Path(results_root)

    from_environment = os.environ.get(RESULTS_ROOT_ENV_VAR, "")
    if from_environment.strip():
        return Path(from_environment)

    raise ResultsRootNotConfiguredError(
        "Cannot record a reference access: no results root is configured, so "
        f"there is nowhere to append {REFERENCE_ACCESS_LEDGER_RELPATH}. Pass "
        "results_root= explicitly, or set the environment variable "
        f"{RESULTS_ROOT_ENV_VAR} to the top-level results directory. The "
        "access is refused rather than performed unrecorded: Requirement 14.5 "
        "requires every read of the sealed reference to record its "
        "reference-access timestamp."
    )


def reference_access_ledger_path(results_root: str | Path) -> Path:
    """Path of the reference-access ledger inside a top-level results root.

    Args:
        results_root: the top-level results directory (`results/`).

    Returns:
        `<results_root>/locked_evaluation/reference_access.jsonl`.
    """
    return Path(results_root) / REFERENCE_ACCESS_LEDGER_RELPATH


def append_reference_access(
    evaluation_id: str,
    results_root: str | Path | None = None,
    *,
    accessed_at: datetime | None = None,
) -> Path:
    """Record one access to the sealed internal randomized reference
    (Requirement 14.5).

    Called by `cohorts.sealed_reference.SealedReference.unseal` and
    `cohorts.sealed_reference.read_sealed_artifact`, each immediately after
    its `assert_reference_unlocked` call --- so only an *authorized* access
    is ever recorded, and every authorized access is.

    The record is `{"evaluation_id": ..., "accessed_at": <ISO-8601 UTC>}`,
    appended verbatim. Repeated ids are legitimate here (see the module
    docstring): this ledger counts accesses, not evaluations, so nothing is
    de-duplicated and nothing is refused as a duplicate.

    Args:
        evaluation_id: the accessing token's `evaluation_id`.
        results_root: the top-level results directory, or `None` to resolve
            it from `CWPWM_RESULTS_ROOT` (see `resolve_results_root`).
        accessed_at: access timestamp; defaults to `datetime.now(UTC)`.

    Returns:
        The ledger path written to.

    Raises:
        ReferenceAccessLedgerError: if `evaluation_id` is not a non-empty
            string.
        ResultsRootNotConfiguredError: if no root can be resolved --- the
            caller must then refuse the access rather than proceed
            unrecorded.
    """
    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        raise ReferenceAccessLedgerError(
            f"Refusing to record a reference access for evaluation_id="
            f"{evaluation_id!r}: an access record must name the evaluation it "
            "belongs to with a non-empty string, or the audit report has no "
            "reference-access timestamp to report (Requirements 14.5, 12.4)."
        )

    path = reference_access_ledger_path(resolve_results_root(results_root))
    _append_ledger_record(
        path,
        {
            "evaluation_id": evaluation_id,
            "accessed_at": (accessed_at or datetime.now(UTC)).isoformat(),
        },
    )
    return path


def reference_access_records(
    results_root: str | Path,
) -> tuple[tuple[str, datetime], ...]:
    """Every recorded reference access, in append order.

    Args:
        results_root: the top-level results directory (`results/`).

    Returns:
        `(evaluation_id, accessed_at)` pairs in file order, or `()` when the
        ledger file does not exist. Repeated ids are preserved: this is a
        verbatim read of an append-only file.

    Raises:
        ReferenceAccessLedgerError: if the file exists but any non-blank
            line is not a JSON object carrying a non-empty string
            `evaluation_id` and an ISO-8601 `accessed_at`.
    """
    path = reference_access_ledger_path(results_root)
    records = _read_ledger_objects(
        path, ledger_name="reference-access ledger", error_type=ReferenceAccessLedgerError
    )

    accesses: list[tuple[str, datetime]] = []
    for line_number, record in records:
        evaluation_id = _require_evaluation_id(
            record,
            path=path,
            line_number=line_number,
            ledger_name="reference-access",
            error_type=ReferenceAccessLedgerError,
        )
        raw_accessed_at = record.get("accessed_at")
        if not isinstance(raw_accessed_at, str) or not raw_accessed_at:
            raise ReferenceAccessLedgerError(
                f"{path}: line {line_number} carries accessed_at="
                f"{raw_accessed_at!r}, which is not a non-empty ISO-8601 string. "
                "The reference-access timestamp is what the audit report reports "
                "(Requirement 12.4); it cannot be absent."
            )
        try:
            accessed_at = datetime.fromisoformat(raw_accessed_at)
        except ValueError as exc:
            raise ReferenceAccessLedgerError(
                f"{path}: line {line_number} carries accessed_at="
                f"{raw_accessed_at!r}, which is not parseable as ISO-8601 ({exc})."
            ) from exc
        accesses.append((evaluation_id, accessed_at))
    return tuple(accesses)


def reference_access_timestamp(evaluation_id: str, results_root: str | Path) -> datetime | None:
    """The reference-access timestamp the audit report carries for
    `evaluation_id` (Requirement 12.4).

    Returns the *first* recorded access, not the last: the audit field
    answers "when was the sealed reference opened for this evaluation",
    and the earliest record is that moment. Later records for the same id
    are further reads within the same one-shot evaluation (e.g. `unseal`
    followed by `read_sealed_artifact`), which do not move the moment the
    seal was broken. `reference_access_records` exposes all of them for an
    auditor who wants the full trail.

    Args:
        evaluation_id: the locked evaluation whose access is being
            reported.
        results_root: the top-level results directory (`results/`).

    Returns:
        The earliest recorded access time for `evaluation_id`, or `None`
        when the ledger records no access for it (including when the ledger
        file does not exist). `None` is a real answer --- "this estimate did
        not come from a locked evaluation, or its reference was never
        opened" --- which `design.md` Section 4.7 models as the audit
        report's `reference_access_timestamp: datetime | None`.

    Raises:
        ReferenceAccessLedgerError: if the ledger exists but is malformed.
    """
    for recorded_id, accessed_at in reference_access_records(results_root):
        if recorded_id == evaluation_id:
            return accessed_at
    return None


def _append_ledger_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as one line to `path`, creating parents.

    Shared by both writers so "append-only" means the same thing in both
    ledgers: append mode (`"a"`), one `json.dumps` with `sort_keys=True` so
    a record's field order does not depend on dict insertion order, exactly
    one trailing newline, and a flush so a caller that reads the ledger back
    immediately (the runner's own post-commit verification, a test) sees the
    line. No existing byte is ever rewritten.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _read_ledger_objects(
    path: Path, *, ledger_name: str, error_type: type[ValueError]
) -> tuple[tuple[int, dict[str, Any]], ...]:
    """Parse an append-only JSON-lines ledger into `(line_number, record)`.

    Shared by both ledgers' readers, so a missing file means the same thing
    in both (an empty ledger --- a fresh results root has consumed nothing
    and accessed nothing) and a malformed present file means the same thing
    in both (`error_type`, never a silent empty read). Blank lines are
    skipped: they carry no record and a trailing newline is normal.

    `line_number` travels with each record so the caller's field-level
    errors can name the offending line without re-reading the file.
    """
    if not path.exists():
        return ()

    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise error_type(
                    f"{path}: line {line_number} is not valid JSON ({exc}). The "
                    f"{ledger_name} is append-only JSON lines; a malformed line "
                    "cannot be read as 'nothing is recorded'."
                ) from exc
            if not isinstance(record, dict):
                raise error_type(
                    f"{path}: line {line_number} is a {type(record).__name__}, not "
                    f"a JSON object; every {ledger_name} record must be an object "
                    "with an 'evaluation_id' field."
                )
            records.append((line_number, record))
    return tuple(records)


def _require_evaluation_id(
    record: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    ledger_name: str,
    error_type: type[ValueError],
) -> str:
    """Return `record["evaluation_id"]`, or raise naming the offending line.

    Both ledgers key on `evaluation_id`, so both validate it identically:
    a record whose id is absent, non-string, or empty is unusable, and
    treating it as absent would make the ledger under-report.
    """
    evaluation_id = record.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise error_type(
            f"{path}: line {line_number} carries evaluation_id="
            f"{evaluation_id!r}, which is not a non-empty string; the "
            f"{ledger_name} ledger keys on it."
        )
    return evaluation_id
