"""The readiness ledger: the mechanical enforcement point for Requirement 18.3.

Requirement 18.3 -- "WHILE any development-stage seed, ablation arm, or
sensitivity variant remains unexecuted from its frozen configuration, keep
the Test_Split closed and the Sealed_Reference_Store sealed" -- is a
*temporal* condition. A convention cannot enforce it. `design.md` Section
2.6 tension 7 makes it enforceable with two pieces:

1. this ledger, an append-only record of what has actually been executed,
   written by each runner *only on successful completion*; and
2. the single choke point that already exists -- the only way to reach the
   test split's outcomes or the sealed reference is a
   `LockedEvaluationToken`, and the only minter is
   `scripts/check_go_no_go.py`.

Requirement 18.3 then reduces to: the minter refuses unless this ledger is
complete. `evaluate_readiness` is the function it asks, and
`ReadinessDecision.missing_items` is the machine-readable answer it prints
(task 20 wires that call site; this module is only the evidence layer).

Requirement 18.7 -- "executes twice from an unchanged Resolved_Config ...
SHALL produce identical results" -- is why `determinism_verified` exists as
a separate set. A run id lands there when a second execution of the same
`ResolvedConfig` reproduced the first execution's artifact hash. Readiness
requires *every* executed run id to be in that set, so "executed
deterministically" is checked rather than asserted.

What "complete" means
---------------------

Four conditions, all of which must hold (`design.md` Section 2.6 tension 7):

- the executed seed set covers the five frozen seeds `REQUIRED_SEEDS`;
- the executed ablation-run set covers the twelve *expanded* ablation run
  ids (see "The twelve expanded run ids" below);
- the executed sensitivity set covers the eight frozen variant names of
  `evaluation.robustness.SENSITIVITY_VARIANTS`; and
- every executed run id also appears in `determinism_verified`.

"Covers" is superset, not equality: recording an extra exploratory run does
not make the ledger unready. An extra run id does, however, have to be
determinism-verified like any other -- an unverified extra is still an
unverified execution.

Why the required sets are resolved lazily
-----------------------------------------

This module must import in an interpreter where torch is not installed
(`design.md` Section 2.4; `tests/unit/test_torch_isolation.py` names
`infra.readiness`), and it must import *before* task 13 lands. Both
required-set sources are therefore imported inside function bodies rather
than at module scope: `evaluation.ablation_matrix` /
`evaluation.robustness` are themselves torch-free but pull in the cohort
and causal layers, and `expand_ablations` does not exist yet.

The twelve expanded run ids
---------------------------

`decisions.md` Section 17 freezes ten ablation *entries*; `design.md`
Section 2.6 tension 1 expands them to twelve *runs*, because
`witness_removed` becomes one run per witness named in its
`config_diff["applies_to"]`, keyed `f"{entry.name}::{witness}"`. Task 13
adds `evaluation.ablation_matrix.expand_ablations()` as the authoritative
implementation of that rule.

`required_ablation_run_ids()` prefers `expand_ablations()` and falls back
to `_expand_frozen_ablations()` -- the same expansion rule applied to the
same frozen `ABLATIONS` list -- while that function is absent. The
fallback is deliberately a re-derivation from the frozen list rather than a
hardcoded tuple of twelve strings, so it cannot drift away from a rename in
`ABLATIONS`, and the moment task 13 lands this module picks up the real
function with no edit here. Task 13 adds the assertion that the two agree
(`evaluate_readiness`'s required set == `{r.run_id for r in
expand_ablations()}`); `RESOLVED_FROM_EXPAND_ABLATIONS` /
`RESOLVED_FROM_FROZEN_LIST` let a caller or test see which path was taken.

Either way the result is validated to hold exactly
`EXPECTED_ABLATION_RUN_COUNT` ids, so an expansion that silently stops
producing twelve runs raises `ReadinessConfigurationError` rather than
lowering the bar readiness has to clear.

On-disk form
------------

`results/<experiment_id>/readiness_ledger.json`. `results_root` throughout
this module means that per-experiment directory (the one
`infra.registry` computes as `results/<experiment_id>`), and the ledger
lives directly inside it::

    {
      "format_version": 1,
      "entries": [
        {"kind": "seed", "run_id": "101", "artifact_hash": "ab12...",
         "recorded_at": "2025-..."},
        {"kind": "ablation_run", "run_id": "witness_removed::temporal", ...},
        {"kind": "determinism", "run_id": "witness_removed::temporal", ...}
      ]
    }

Entries are append-only: `append_ledger_entry` never removes or rewrites a
prior entry. Re-appending an identical `(kind, run_id, artifact_hash)`
triple is an idempotent no-op, so a resumed runner does not duplicate its
own records. Appending the *same* `(kind, run_id)` with a *different*
`artifact_hash` raises `LedgerConflictError`: that is precisely the
non-determinism Requirement 18.7 forbids, and recording it silently would
let the ledger claim readiness for a run whose artifact changed underneath
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen values
# ---------------------------------------------------------------------------

#: The five frozen seeds every development-stage arm is executed at
#: (`decisions.md` Section 25; `design.md` Section 2.6 tension 7).
REQUIRED_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

#: Number of ablation *runs* the ten frozen `ABLATIONS` entries expand to
#: (`design.md` Section 2.6 tension 1). Not a re-derivation of the frozen
#: list -- an arity check on whatever expansion is in force.
EXPECTED_ABLATION_RUN_COUNT = 12

#: Number of frozen sensitivity variants (`decisions.md` Section 20).
EXPECTED_SENSITIVITY_VARIANT_COUNT = 8

#: The one ablation entry that expands to more than one run.
WITNESS_REMOVED_ABLATION_NAME = "witness_removed"

#: Separator in an expanded sub-run id, `f"{entry.name}::{witness}"`
#: (`design.md` Section 2.6 tension 1).
SUB_RUN_SEPARATOR = "::"

# ---------------------------------------------------------------------------
# Ledger file shape
# ---------------------------------------------------------------------------

#: Ledger filename inside `results/<experiment_id>/`.
READINESS_LEDGER_FILENAME = "readiness_ledger.json"

#: Version stamped into every ledger this module writes.
LEDGER_FORMAT_VERSION = 1

SEED_KIND = "seed"
ABLATION_RUN_KIND = "ablation_run"
SENSITIVITY_VARIANT_KIND = "sensitivity_variant"
DETERMINISM_KIND = "determinism"

#: Every accepted `kind` for `append_ledger_entry`.
LEDGER_ENTRY_KINDS: tuple[str, ...] = (
    SEED_KIND,
    ABLATION_RUN_KIND,
    SENSITIVITY_VARIANT_KIND,
    DETERMINISM_KIND,
)

# ---------------------------------------------------------------------------
# Missing-item vocabulary
#
# `missing_items` is consumed by a script that prints it and by tests that
# assert on it, so the prefixes are named constants with formatting helpers
# rather than string literals spread across call sites.
# ---------------------------------------------------------------------------

MISSING_SEED_PREFIX = "missing_seed:"
MISSING_ABLATION_RUN_PREFIX = "missing_ablation_run:"
MISSING_SENSITIVITY_VARIANT_PREFIX = "missing_sensitivity_variant:"
DETERMINISM_UNVERIFIED_PREFIX = "determinism_unverified:"

#: Values of `ReadinessDecision.required_runs_source`.
RESOLVED_FROM_EXPAND_ABLATIONS = "evaluation.ablation_matrix.expand_ablations"
RESOLVED_FROM_FROZEN_LIST = "evaluation.ablation_matrix.ABLATIONS (fallback expansion)"


class ReadinessConfigurationError(Exception):
    """Raised when the required sets themselves cannot be resolved.

    A wrong-arity expansion (not twelve ablation runs, not eight
    sensitivity variants) means the bar readiness must clear is unknown, so
    this halts rather than evaluating against a truncated requirement.
    """


class ReadinessLedgerError(Exception):
    """Raised when a ledger file on disk cannot be read as a ledger."""


class LedgerConflictError(ReadinessLedgerError):
    """Raised when a `(kind, run_id)` is re-recorded with a different hash.

    Two executions of the same run producing different artifact hashes is
    the non-determinism Requirement 18.7 forbids; see the module docstring.
    """


def missing_seed_item(seed: int) -> str:
    """Name a missing frozen seed for `ReadinessDecision.missing_items`."""
    return f"{MISSING_SEED_PREFIX}{seed}"


def missing_ablation_run_item(run_id: str) -> str:
    """Name a missing expanded ablation run."""
    return f"{MISSING_ABLATION_RUN_PREFIX}{run_id}"


def missing_sensitivity_variant_item(name: str) -> str:
    """Name a missing frozen sensitivity variant."""
    return f"{MISSING_SENSITIVITY_VARIANT_PREFIX}{name}"


def determinism_unverified_item(run_id: str) -> str:
    """Name an executed run whose determinism was never verified."""
    return f"{DETERMINISM_UNVERIFIED_PREFIX}{run_id}"


# ---------------------------------------------------------------------------
# Required sets (lazily resolved -- see module docstring)
# ---------------------------------------------------------------------------


def _expand_frozen_ablations() -> tuple[str, ...]:
    """Expand the ten frozen `ABLATIONS` entries to run ids.

    The fallback used while `expand_ablations()` (task 13) does not exist.
    Applies the rule of `design.md` Section 2.6 tension 1 verbatim: one run
    per entry, except `witness_removed`, which yields one run per witness
    named in its `config_diff["applies_to"]`, keyed
    `f"{entry.name}::{witness}"`.
    """
    from evaluation.ablation_matrix import ABLATIONS

    run_ids: list[str] = []
    for entry in ABLATIONS:
        if entry.name != WITNESS_REMOVED_ABLATION_NAME:
            run_ids.append(entry.name)
            continue
        applies_to = entry.config_diff.get("applies_to")
        if not applies_to:
            raise ReadinessConfigurationError(
                f"ablation {entry.name!r} carries no non-empty "
                f"config_diff['applies_to'], so its per-witness expansion "
                f"(design.md Section 2.6 tension 1) cannot be derived."
            )
        run_ids.extend(f"{entry.name}{SUB_RUN_SEPARATOR}{witness}" for witness in applies_to)
    return tuple(run_ids)


def resolve_required_ablation_runs() -> tuple[tuple[str, ...], str]:
    """Return `(required_run_ids, source)` for the twelve expanded runs.

    Prefers `evaluation.ablation_matrix.expand_ablations()` and falls back
    to `_expand_frozen_ablations()` while that function is absent (see the
    module docstring, "The twelve expanded run ids"). The returned
    `source` is `RESOLVED_FROM_EXPAND_ABLATIONS` or
    `RESOLVED_FROM_FROZEN_LIST`.

    Raises:
        ReadinessConfigurationError: if the expansion does not hold exactly
            `EXPECTED_ABLATION_RUN_COUNT` distinct run ids.
    """
    try:
        from evaluation.ablation_matrix import expand_ablations
    except ImportError:
        run_ids = _expand_frozen_ablations()
        source = RESOLVED_FROM_FROZEN_LIST
    else:
        run_ids = tuple(run.run_id for run in expand_ablations())
        source = RESOLVED_FROM_EXPAND_ABLATIONS

    duplicates = sorted({rid for rid in run_ids if run_ids.count(rid) > 1})
    if duplicates:
        raise ReadinessConfigurationError(
            f"the ablation expansion from {source} produced duplicate run ids "
            f"{duplicates}; run ids are result-directory keys and must be unique."
        )
    if len(run_ids) != EXPECTED_ABLATION_RUN_COUNT:
        raise ReadinessConfigurationError(
            f"the ablation expansion from {source} produced {len(run_ids)} run "
            f"id(s), not the {EXPECTED_ABLATION_RUN_COUNT} frozen by design.md "
            f"Section 2.6 tension 1: {list(run_ids)}. Readiness cannot be "
            f"evaluated against an expansion of unknown arity."
        )
    return run_ids, source


def required_ablation_run_ids() -> tuple[str, ...]:
    """The twelve expanded ablation run ids readiness requires."""
    run_ids, _ = resolve_required_ablation_runs()
    return run_ids


def required_sensitivity_variant_names() -> tuple[str, ...]:
    """The eight frozen sensitivity variant names readiness requires.

    Raises:
        ReadinessConfigurationError: if the frozen list does not hold
            exactly `EXPECTED_SENSITIVITY_VARIANT_COUNT` distinct names.
    """
    from evaluation.robustness import SENSITIVITY_VARIANTS

    names = tuple(variant.name for variant in SENSITIVITY_VARIANTS)
    if len(set(names)) != len(names):
        raise ReadinessConfigurationError(
            f"evaluation.robustness.SENSITIVITY_VARIANTS holds duplicate names: "
            f"{sorted({n for n in names if names.count(n) > 1})}."
        )
    if len(names) != EXPECTED_SENSITIVITY_VARIANT_COUNT:
        raise ReadinessConfigurationError(
            f"evaluation.robustness.SENSITIVITY_VARIANTS holds {len(names)} "
            f"variant(s), not the {EXPECTED_SENSITIVITY_VARIANT_COUNT} frozen by "
            f"decisions.md Section 20: {list(names)}."
        )
    return names


# ---------------------------------------------------------------------------
# The ledger and the decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadinessLedger:
    """What has actually been executed (`design.md` Section 2.6 tension 7).

    Attributes:
        seeds_executed: master seeds whose runs completed successfully.
        ablation_runs_executed: expanded ablation run ids that completed.
        sensitivity_variants_executed: sensitivity variant names that
            completed.
        determinism_verified: run ids re-executed from an unchanged
            `ResolvedConfig` whose artifact hashes matched the first
            execution (Requirement 18.7).
    """

    seeds_executed: tuple[int, ...] = ()
    ablation_runs_executed: tuple[str, ...] = ()
    sensitivity_variants_executed: tuple[str, ...] = ()
    determinism_verified: tuple[str, ...] = ()

    def executed_run_ids(self) -> tuple[str, ...]:
        """Every executed run id, ablation and sensitivity, in ledger order.

        Seeds are excluded: they are integers naming a *repetition* of a
        run rather than a run id, and their determinism is covered by the
        run entries recorded at each seed.
        """
        seen: list[str] = []
        for run_id in (*self.ablation_runs_executed, *self.sensitivity_variants_executed):
            if run_id not in seen:
                seen.append(run_id)
        return tuple(seen)


#: A ledger with nothing recorded -- what `load_ledger` returns for a
#: results root that has never had a runner complete inside it.
EMPTY_LEDGER = ReadinessLedger()


@dataclass(frozen=True)
class ReadinessDecision:
    """The answer `scripts/check_go_no_go.py` gates minting on.

    Attributes:
        ready: `True` only when all four conditions of the module docstring
            hold.
        missing_items: every missing seed, ablation run, sensitivity
            variant, and determinism verification, named via
            `missing_seed_item` / `missing_ablation_run_item` /
            `missing_sensitivity_variant_item` /
            `determinism_unverified_item`. Empty exactly when `ready`.
            Ordered deterministically: seeds in `REQUIRED_SEEDS` order,
            runs and variants in their frozen order, then determinism
            items in executed order.
        required_runs_source: which expansion supplied the required
            ablation-run set, for the report at the gate.
    """

    ready: bool
    missing_items: tuple[str, ...]
    required_runs_source: str


def evaluate_readiness(ledger: ReadinessLedger) -> ReadinessDecision:
    """Decide whether the development stage is complete (Requirements 18.3, 18.7).

    Args:
        ledger: the executed-work record, typically from
            `load_ledger(results_root)`.

    Returns:
        A `ReadinessDecision`. `ready` is `True` only when the executed
        seed set covers `REQUIRED_SEEDS`, the executed ablation-run set
        covers the twelve expanded run ids, the executed sensitivity set
        covers the eight frozen variant names, and every executed run id
        appears in `ledger.determinism_verified`. Otherwise `ready` is
        `False` and `missing_items` names each missing seed, run, variant,
        and determinism verification.

    Raises:
        ReadinessConfigurationError: if the required sets cannot be
            resolved (see `resolve_required_ablation_runs`).
    """
    required_runs, source = resolve_required_ablation_runs()
    required_variants = required_sensitivity_variant_names()

    executed_seeds = set(ledger.seeds_executed)
    executed_runs = set(ledger.ablation_runs_executed)
    executed_variants = set(ledger.sensitivity_variants_executed)
    verified = set(ledger.determinism_verified)

    missing: list[str] = []
    missing.extend(missing_seed_item(s) for s in REQUIRED_SEEDS if s not in executed_seeds)
    missing.extend(missing_ablation_run_item(r) for r in required_runs if r not in executed_runs)
    missing.extend(
        missing_sensitivity_variant_item(v)
        for v in required_variants
        if v not in executed_variants
    )
    missing.extend(
        determinism_unverified_item(run_id)
        for run_id in ledger.executed_run_ids()
        if run_id not in verified
    )

    return ReadinessDecision(
        ready=not missing,
        missing_items=tuple(missing),
        required_runs_source=source,
    )


# ---------------------------------------------------------------------------
# On-disk ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    """One append-only ledger record.

    Attributes:
        kind: one of `LEDGER_ENTRY_KINDS`.
        run_id: the run's id. For `kind="seed"` this is the seed rendered
            as a decimal string, so a single entry shape covers all four
            kinds; `load_ledger` parses it back to `int`.
        artifact_hash: hash of the result artifact this record attests to.
        recorded_at: UTC ISO-8601 timestamp of the append.
    """

    kind: str
    run_id: str
    artifact_hash: str
    recorded_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "artifact_hash": self.artifact_hash,
            "recorded_at": self.recorded_at,
        }


def ledger_path(results_root: str | Path) -> Path:
    """Path of the ledger inside a per-experiment results root.

    `results_root` is `results/<experiment_id>`; the ledger is
    `results/<experiment_id>/readiness_ledger.json`.
    """
    return Path(results_root) / READINESS_LEDGER_FILENAME


def _parse_entry(raw: Any, path: Path, index: int) -> LedgerEntry:
    if not isinstance(raw, dict):
        raise ReadinessLedgerError(f"{path}: entry {index} is not a mapping: {raw!r}")
    missing_fields = sorted(
        field for field in ("kind", "run_id", "artifact_hash") if field not in raw
    )
    if missing_fields:
        raise ReadinessLedgerError(
            f"{path}: entry {index} is missing field(s): {', '.join(missing_fields)}"
        )
    kind = raw["kind"]
    if kind not in LEDGER_ENTRY_KINDS:
        raise ReadinessLedgerError(
            f"{path}: entry {index} has unknown kind {kind!r}; expected one of "
            f"{', '.join(LEDGER_ENTRY_KINDS)}"
        )
    return LedgerEntry(
        kind=kind,
        run_id=str(raw["run_id"]),
        artifact_hash=str(raw["artifact_hash"]),
        recorded_at=str(raw.get("recorded_at", "")),
    )


def load_ledger_entries(results_root: str | Path) -> tuple[LedgerEntry, ...]:
    """Read every ledger entry in append order.

    Returns an empty tuple when no ledger file exists yet -- an absent
    ledger is a results root where nothing has completed, not an error.

    Raises:
        ReadinessLedgerError: if the file is not valid JSON, is not a
            mapping with an `entries` list, or holds a malformed entry.
    """
    path = ledger_path(results_root)
    if not path.exists():
        return ()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReadinessLedgerError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(raw, dict) or "entries" not in raw:
        raise ReadinessLedgerError(
            f"{path}: top-level content must be a mapping with an 'entries' key"
        )
    entries = raw["entries"]
    if not isinstance(entries, list):
        raise ReadinessLedgerError(f"{path}: 'entries' must be a list")

    return tuple(_parse_entry(entry, path, index) for index, entry in enumerate(entries))


def ledger_from_entries(entries: tuple[LedgerEntry, ...]) -> ReadinessLedger:
    """Fold ledger entries into a `ReadinessLedger`.

    Each kind's run ids are de-duplicated, keeping first-append order.
    `kind="seed"` run ids are parsed as integers.

    Raises:
        ReadinessLedgerError: if a `kind="seed"` entry's `run_id` is not an
            integer.
    """
    seeds: list[int] = []
    ablation_runs: list[str] = []
    sensitivity_variants: list[str] = []
    determinism: list[str] = []

    for entry in entries:
        if entry.kind == SEED_KIND:
            try:
                seed = int(entry.run_id)
            except ValueError as exc:
                raise ReadinessLedgerError(
                    f"ledger entry of kind {SEED_KIND!r} has non-integer run_id "
                    f"{entry.run_id!r}; a seed entry records the master seed."
                ) from exc
            if seed not in seeds:
                seeds.append(seed)
        elif entry.kind == ABLATION_RUN_KIND:
            if entry.run_id not in ablation_runs:
                ablation_runs.append(entry.run_id)
        elif entry.kind == SENSITIVITY_VARIANT_KIND:
            if entry.run_id not in sensitivity_variants:
                sensitivity_variants.append(entry.run_id)
        else:  # DETERMINISM_KIND -- the only remaining validated kind.
            if entry.run_id not in determinism:
                determinism.append(entry.run_id)

    return ReadinessLedger(
        seeds_executed=tuple(seeds),
        ablation_runs_executed=tuple(ablation_runs),
        sensitivity_variants_executed=tuple(sensitivity_variants),
        determinism_verified=tuple(determinism),
    )


def load_ledger(results_root: str | Path) -> ReadinessLedger:
    """Read `results/<experiment_id>/readiness_ledger.json` as a ledger.

    Returns `EMPTY_LEDGER` when the file does not exist, so
    `evaluate_readiness(load_ledger(results_root))` on a fresh results root
    reports not-ready with everything named rather than failing.
    """
    return ledger_from_entries(load_ledger_entries(results_root))


def append_ledger_entry(
    results_root: str | Path,
    kind: str,
    run_id: str | int,
    artifact_hash: str,
) -> LedgerEntry:
    """Append one completion record to the ledger.

    Called by a runner *after* the run it describes completed successfully
    (`design.md` Section 2.6 tension 7). Creates the results root and the
    ledger file if absent. Prior entries are never removed or rewritten.

    Args:
        results_root: the per-experiment results directory,
            `results/<experiment_id>`.
        kind: one of `LEDGER_ENTRY_KINDS`.
        run_id: the run id, or the master seed for `kind="seed"`.
        artifact_hash: hash of the result artifact this record attests to.

    Returns:
        The `LedgerEntry` now recorded -- the newly appended one, or the
        already-present identical one.

    Raises:
        ValueError: if `kind` is not a known kind, or `run_id` /
            `artifact_hash` is empty.
        LedgerConflictError: if this `(kind, run_id)` is already recorded
            with a different `artifact_hash`.
        ReadinessLedgerError: if an existing ledger file cannot be read.
    """
    if kind not in LEDGER_ENTRY_KINDS:
        raise ValueError(
            f"unknown readiness ledger kind {kind!r}; expected one of "
            f"{', '.join(LEDGER_ENTRY_KINDS)}"
        )
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise ValueError("run_id must be a non-empty identifier")
    normalized_hash = str(artifact_hash).strip()
    if not normalized_hash:
        raise ValueError(
            f"artifact_hash must be non-empty: a ledger entry for "
            f"{kind}={normalized_run_id!r} without an artifact hash cannot support "
            f"the determinism check of Requirement 18.7"
        )

    existing = load_ledger_entries(results_root)
    for entry in existing:
        if entry.kind != kind or entry.run_id != normalized_run_id:
            continue
        if entry.artifact_hash == normalized_hash:
            return entry  # Idempotent re-append by a resumed runner.
        raise LedgerConflictError(
            f"{ledger_path(results_root)}: {kind} {normalized_run_id!r} is already "
            f"recorded with artifact_hash {entry.artifact_hash!r}, and this append "
            f"carries {normalized_hash!r}. Two executions of the same run produced "
            f"different artifacts, which is the non-determinism Requirement 18.7 "
            f"forbids; the conflict is raised rather than recorded."
        )

    appended = LedgerEntry(
        kind=kind,
        run_id=normalized_run_id,
        artifact_hash=normalized_hash,
        recorded_at=datetime.now(UTC).isoformat(),
    )

    path = ledger_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": LEDGER_FORMAT_VERSION,
        "entries": [entry.to_dict() for entry in (*existing, appended)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return appended
