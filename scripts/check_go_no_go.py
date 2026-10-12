#!/usr/bin/env python
"""Go/no-go checklist gate and `LockedEvaluationToken` minting (task 16;
`decisions.md` Section 21; `design.md` "Correctness Property 1: Sealed
reference isolation").

`design.md` Section 2.1 describes Stage 15 as "a hard checkpoint
implemented as a script (`scripts/check_go_no_go.py`) that reads a
checklist config and exits non-zero if any item is unresolved." This is
that script. `cohorts.sealed_reference.LockedEvaluationToken._mint()` is
documented as having its "intended sole real caller" be this script, "once
every go/no-go checklist item has passed" -- this module is the first real
(non-test) call site for `_mint()`.

Attestations, and evidence against them (task 2.5)
--------------------------------------------------

The twelve checklist items are human-attested booleans: each is marked
`true`/`false` in `configs/go_no_go_checklist.yaml` by whatever process
performed that check -- a reviewer ticking a box, or an earlier automated
stage writing its own result into the file. Gating on those booleans alone
left one hole: an attestation could be simply wrong, and nothing noticed.

`infra.checklist_verifiers` closes it. Each item has a verifier returning
`verified`, `failed`, or `unverifiable`, and this script:

- **reports** every verifier result next to its attested boolean, so the
  evidence behind (or missing from) each attestation is visible at the
  gate rather than assumed; and
- **refuses to mint** when a machine-verifiable item is attested `true`
  but its verifier returns `failed`. An attestation may be the only
  evidence for an item (three of the twelve are process claims no machine
  can check -- see `checklist_verifiers.ATTESTATION_ONLY_REASONS`), but it
  may never contradict evidence that exists.

A verifier returning `unverifiable` never blocks: absence of evidence is
not evidence against, and the attestation remains the signal for those
items by design.

Token bindings (task 20)
-------------------------

`cohorts.sealed_reference.LockedEvaluationToken` binds nine fields
(`neural-experiment-pipeline` Requirement 13.3): `issued_by`, `issued_at`,
and seven artifact bindings this script is the designated minter for.
Task 7 strengthened the token; this task wires the real values in:

- `--resolved-config-hash`, `--dataset-checksum`, `--split-hash`,
  `--git-commit` (auto-resolved via `infra.manifest.resolve_git_provenance`
  unless overridden), `--statistical-plan-version` (defaults to
  `evaluation.statistics.STATISTICAL_PLAN_VERSION`), and `--evaluation-id`
  (auto-generated via `uuid.uuid4().hex` unless overridden, so a caller
  never has to invent a fresh identifier by hand) are each a single
  caller-supplied string.
- `--checkpoint-hash NAME=SHA256` (repeatable, at least one required) --
  one `(logical_name, sha256)` pair per frozen checkpoint the locked
  evaluation will load.

Every one of these seven fields must resolve to a non-empty value before
`_mint` is called; an absent one is refused by name alongside every other
offender in the same message (Requirement 13.4), never a placeholder.
There is no code path left in this script that mints with a sentinel
value -- a caller that omits a required binding gets a refusal, not a
token artifact with `UNWIRED_BINDING_PLACEHOLDER` written into it.

Readiness gating (task 20; Requirement 18.3, 18.4)
----------------------------------------------------

The thirteenth checklist item, `neural_readiness_verified`, is never read
as a human-attested boolean (`configs/go_no_go_checklist.yaml`'s own key
of that name is ignored outright): it is always recomputed from
`infra.readiness.evaluate_readiness(infra.readiness.load_ledger(results_root))`.
When the ledger is not ready, minting is refused and every missing seed,
ablation run, and sensitivity variant is named -- exactly the same
`missing_items` strings `evaluate_readiness` itself produces, so this
gate and the ledger's own evidence can never disagree about what is
outstanding.

`--results-root` supplies both the consumption ledger the mint checks for
a duplicated `evaluation_id` (Requirement 13.4) and the readiness ledger
read here; its default is `checklist_verifiers.DEFAULT_RESULTS_ROOT`, the
same `results/` tree the verifiers already read.

The eleven-name list quoted in the original task description undercounts
by one -- counted individually, the parenthetical actually lists twelve
items. All twelve are encoded below, in
`configs/go_no_go_checklist.yaml`, and in `CHECKLIST_ITEMS`.

Usage::

    python scripts/check_go_no_go.py --checklist configs/go_no_go_checklist.yaml
    python scripts/check_go_no_go.py --checklist path/to/checklist.yaml \\
        --token-output results/locked_evaluation_token.json
    python scripts/check_go_no_go.py --results-root results/run_abc123
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import yaml

from cohorts.sealed_reference import LockedEvaluationToken, TokenFieldError
from infra.checklist_verifiers import (
    VerificationContext,
    VerifierResult,
    contradicted_items,
    verify_all,
)
from infra.manifest import resolve_git_provenance
from infra.readiness import evaluate_readiness, load_ledger

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py` and `scripts/run_paper.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default checklist config path.
DEFAULT_CHECKLIST_PATH = REPO_ROOT / "configs" / "go_no_go_checklist.yaml"

#: The twelve human-attested go/no-go checklist item names (`decisions.md`
#: Section 21; task 16's parenthetical list, counted individually). This is
#: the authoritative set: `load_checklist()` requires every one of these
#: keys to be present, and `check_go_no_go()` only ever reports unmet items
#: drawn from this set.
CHECKLIST_ITEMS: tuple[str, ...] = (
    "frozen_protocol",
    "frozen_causal_diagram",
    "frozen_adjustment_set",
    "estimand_alignment_checked",
    "reference_inaccessible_to_model_selection",
    "leakage_tests_passed",
    "splits_cross_fitting_verified",
    "thresholds_frozen",
    "target_population_defined",
    "statistical_procedures_locked",
    "privacy_checks_passed",
    "one_command_reproduction_verified",
)

#: The thirteenth checklist item (task 20; Requirement 18.3, 18.4).
#: Deliberately NOT a member of `CHECKLIST_ITEMS`: that tuple is the set of
#: *human-attested* booleans `load_checklist` reads from the YAML file,
#: while this item is always computed, never read as an attestation (see
#: module docstring). Keeping it out of `CHECKLIST_ITEMS` means
#: `load_checklist`'s "unknown item" check continues to reject it if a
#: caller's checklist file tries to set it, and `check_go_no_go()`'s
#: human-attested unmet-item logic never has to special-case it.
NEURAL_READINESS_ITEM: str = "neural_readiness_verified"


class ChecklistError(Exception):
    """Raised when a checklist config cannot be loaded or is malformed
    (missing/unknown items, non-boolean values, non-mapping top level).
    """


def load_checklist(path: Path) -> dict[str, bool]:
    """Load a checklist status config and return `{item_name: bool}`.

    Args:
        path: path to a YAML file with an `items:` mapping from each of
            `CHECKLIST_ITEMS` to a boolean (see
            `configs/go_no_go_checklist.yaml` for the template).

    Returns:
        A dict covering every name in `CHECKLIST_ITEMS`.

    Raises:
        ChecklistError: if the file is missing, not valid YAML, not a
            mapping, missing the `items` key, missing any required item,
            contains an unknown item, or contains a non-boolean value.
    """
    if not path.exists():
        raise ChecklistError(f"checklist file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ChecklistError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or "items" not in raw:
        raise ChecklistError(f"{path}: top-level content must be a mapping with an 'items' key")

    items = raw["items"]
    if not isinstance(items, dict):
        raise ChecklistError(f"{path}: 'items' must be a mapping")

    # `NEURAL_READINESS_ITEM` may be present (the shipped template carries
    # it, so the checklist's thirteen-key shape stays uniform) but its
    # value is accepted here and ignored everywhere else -- this function
    # never returns it, and no caller reads it as an attestation (see
    # module docstring).
    known = set(CHECKLIST_ITEMS) | {NEURAL_READINESS_ITEM}
    unknown = sorted(set(items) - known)
    if unknown:
        raise ChecklistError(f"{path}: unknown checklist item(s): {', '.join(unknown)}")

    missing = sorted(set(CHECKLIST_ITEMS) - set(items))
    if missing:
        raise ChecklistError(f"{path}: missing checklist item(s): {', '.join(missing)}")

    non_boolean = sorted(name for name, value in items.items() if not isinstance(value, bool))
    if non_boolean:
        raise ChecklistError(
            f"{path}: checklist item(s) must be boolean true/false: {', '.join(non_boolean)}"
        )

    return {name: items[name] for name in CHECKLIST_ITEMS}


def check_go_no_go(checklist: dict[str, bool]) -> tuple[bool, list[str]]:
    """Evaluate a loaded checklist.

    Args:
        checklist: a `{item_name: bool}` mapping, typically from
            `load_checklist()`.

    Returns:
        `(all_passed, unmet_items)` where `all_passed` is `True` only if
        every one of `CHECKLIST_ITEMS` is `True` in `checklist`, and
        `unmet_items` lists (in `CHECKLIST_ITEMS` order) every item that is
        `False` or absent.
    """
    unmet = [name for name in CHECKLIST_ITEMS if not checklist.get(name, False)]
    return (len(unmet) == 0, unmet)


def check_neural_readiness(results_root: Path) -> tuple[bool, tuple[str, ...]]:
    """Compute the thirteenth checklist item from the readiness ledger
    (task 20; Requirement 18.3, 18.4).

    Unlike `check_go_no_go`, this never reads an attestation: it is always
    `infra.readiness.evaluate_readiness(infra.readiness.load_ledger(results_root))`,
    so a human cannot mark readiness `true` by editing the YAML file (see
    module docstring and `load_checklist`'s handling of
    `NEURAL_READINESS_ITEM`).

    Returns:
        `(ready, missing_items)` -- `missing_items` is
        `ReadinessDecision.missing_items` verbatim (every missing seed,
        ablation run, sensitivity variant, and determinism verification),
        so this gate's refusal message and the ledger's own evidence can
        never disagree about what is outstanding.
    """
    decision = evaluate_readiness(load_ledger(results_root))
    return decision.ready, decision.missing_items


#: The seven artifact bindings this script is the designated minter for
#: (Requirement 13.3), in `LockedEvaluationToken`'s declaration order
#: minus `issued_by`/`issued_at` (which this script supplies itself, not
#: from caller input).
BINDABLE_FIELDS: tuple[str, ...] = (
    "resolved_config_hash",
    "dataset_checksum",
    "split_hash",
    "checkpoint_hashes",
    "git_commit",
    "statistical_plan_version",
    "evaluation_id",
)


def collect_bindable_fields(args: argparse.Namespace) -> tuple[dict[str, object], list[str]]:
    """Collect the seven bindable fields from CLI args, auto-resolving
    `git_commit` and `evaluation_id` when not explicitly overridden
    (task 20; Requirement 13.3, 13.4).

    Returns:
        `(bindings, offenders)`. `bindings` holds whatever could be
        resolved (possibly a partial mapping); `offenders` names every
        field that could not be resolved to a non-empty value, each with
        a human-readable reason, in `BINDABLE_FIELDS` order. An empty
        `offenders` list means every field in `bindings` is ready to pass
        to `_mint`.
    """
    bindings: dict[str, object] = {}
    offenders: list[str] = []

    if args.resolved_config_hash:
        bindings["resolved_config_hash"] = args.resolved_config_hash
    else:
        offenders.append("resolved_config_hash: not supplied (--resolved-config-hash)")

    if args.dataset_checksum:
        bindings["dataset_checksum"] = args.dataset_checksum
    else:
        offenders.append("dataset_checksum: not supplied (--dataset-checksum)")

    if args.split_hash:
        bindings["split_hash"] = args.split_hash
    else:
        offenders.append("split_hash: not supplied (--split-hash)")

    if args.checkpoint_hash:
        parsed_checkpoints: list[tuple[str, str]] = []
        malformed_checkpoints: list[str] = []
        for raw in args.checkpoint_hash:
            name, sep, sha = raw.partition("=")
            if not sep or not name.strip() or not sha.strip():
                malformed_checkpoints.append(raw)
                continue
            parsed_checkpoints.append((name.strip(), sha.strip()))
        if malformed_checkpoints:
            offenders.append(
                "checkpoint_hashes: malformed --checkpoint-hash value(s) "
                f"(expected NAME=SHA256): {', '.join(malformed_checkpoints)}"
            )
        else:
            bindings["checkpoint_hashes"] = tuple(parsed_checkpoints)
    else:
        offenders.append("checkpoint_hashes: not supplied (at least one --checkpoint-hash)")

    if args.git_commit:
        bindings["git_commit"] = args.git_commit
    else:
        provenance = resolve_git_provenance(REPO_ROOT)
        if provenance.available and not provenance.dirty:
            bindings["git_commit"] = provenance.commit
        elif provenance.available and provenance.dirty:
            offenders.append(
                "git_commit: working tree is dirty (uncommitted paths: "
                f"{', '.join(provenance.uncommitted_paths)}); commit before minting "
                "(Requirement 15.3 -- a dirty tree is refused, not overridable)"
            )
        else:
            offenders.append(f"git_commit: unavailable ({provenance.unavailable_reason})")

    if args.statistical_plan_version:
        bindings["statistical_plan_version"] = args.statistical_plan_version
    else:
        from evaluation.statistics import STATISTICAL_PLAN_VERSION

        bindings["statistical_plan_version"] = STATISTICAL_PLAN_VERSION

    if args.evaluation_id:
        bindings["evaluation_id"] = args.evaluation_id
    else:
        bindings["evaluation_id"] = uuid.uuid4().hex

    return bindings, offenders


def format_verification_report(
    checklist: dict[str, bool], results: dict[str, VerifierResult]
) -> list[str]:
    """Render one report line per checklist item: attestation vs. evidence.

    Lines are returned (not printed) so the caller controls the stream and so
    the format is testable without capturing stdout. Order follows
    `CHECKLIST_ITEMS`.
    """
    name_width = max(len(name) for name in CHECKLIST_ITEMS)
    lines: list[str] = []
    for name in CHECKLIST_ITEMS:
        attested = "true" if checklist.get(name, False) else "false"
        result = results.get(name)
        outcome = result.outcome.value if result is not None else "not-run"
        evidence = result.evidence if result is not None else "verifier not invoked"
        lines.append(
            f"  {name:<{name_width}}  attested={attested:<5}  "
            f"verifier={outcome:<12}  {evidence}"
        )
    return lines


def _serialize_token(token: LockedEvaluationToken) -> dict[str, object]:
    """Serialize every field the token binds, not just the two it used to.

    A token artifact that recorded only `issued_by`/`issued_at` while the
    token bound seven more fields would be actively misleading: task 21's
    runner reads this artifact to know which hashes to recompute
    (Requirement 13.5), and a binding absent from the file is a binding
    that cannot be checked.
    """
    return {
        "issued_by": token.issued_by,
        "issued_at": token.issued_at.isoformat(),
        "resolved_config_hash": token.resolved_config_hash,
        "dataset_checksum": token.dataset_checksum,
        "split_hash": token.split_hash,
        # A list of two-element lists: JSON has no tuple, and the runner
        # re-sorts on read anyway.
        "checkpoint_hashes": [list(pair) for pair in token.checkpoint_hashes],
        "git_commit": token.git_commit,
        "statistical_plan_version": token.statistical_plan_version,
        "evaluation_id": token.evaluation_id,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the go/no-go checklist (decisions.md Section 21) and, if "
            "every item passes, mint a LockedEvaluationToken."
        )
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=DEFAULT_CHECKLIST_PATH,
        help=f"Path to the checklist status YAML (default: {DEFAULT_CHECKLIST_PATH}).",
    )
    parser.add_argument(
        "--token-output",
        type=Path,
        default=None,
        help=(
            "If given and every checklist item passes, write the minted "
            "LockedEvaluationToken as JSON to this path."
        ),
    )
    parser.add_argument(
        "--decisions-path",
        type=Path,
        default=None,
        help=(
            "Path to decisions.md, the evidence source for the protocol-freeze "
            "verifiers (default: the in-repo spec copy)."
        ),
    )
    parser.add_argument(
        "--target-trial-path",
        type=Path,
        default=None,
        help=(
            "Path to the target-trial config checked against decisions.md "
            "(default: configs/target_trials/main.yaml)."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Directory scanned by scripts/scan_exports.py for the "
            "privacy_checks_passed verifier, and read for the readiness "
            "ledger and consumption ledger (default: results/)."
        ),
    )
    # The seven bindable fields (task 20; Requirement 13.3).
    parser.add_argument(
        "--resolved-config-hash",
        default=None,
        help="infra.registry.canonical_json hash of this run's ResolvedConfig. Required.",
    )
    parser.add_argument(
        "--dataset-checksum",
        default=None,
        help="The frozen dataset contract's checksum. Required.",
    )
    parser.add_argument(
        "--split-hash",
        default=None,
        help="Hash of the frozen dev/val/test assignment. Required.",
    )
    parser.add_argument(
        "--checkpoint-hash",
        action="append",
        default=None,
        metavar="NAME=SHA256",
        help=(
            "One (logical_name, sha256) checkpoint binding. Repeatable; at least "
            "one is required."
        ),
    )
    parser.add_argument(
        "--git-commit",
        default=None,
        help=(
            "Override the auto-resolved git commit (default: resolved via "
            "infra.manifest.resolve_git_provenance, refusing if the tree is dirty "
            "or the commit is unavailable)."
        ),
    )
    parser.add_argument(
        "--statistical-plan-version",
        default=None,
        help=(
            "Override the frozen statistical-plan version string (default: "
            "evaluation.statistics.STATISTICAL_PLAN_VERSION)."
        ),
    )
    parser.add_argument(
        "--evaluation-id",
        default=None,
        help=(
            "Override the auto-generated evaluation id (default: a fresh "
            "uuid.uuid4().hex)."
        ),
    )
    return parser


def build_verification_context(args: argparse.Namespace) -> VerificationContext:
    """Build the verifier context from parsed CLI arguments.

    Only the paths the caller actually overrode are passed through, so every
    other evidence source keeps `VerificationContext`'s in-repo default.
    """
    overrides: dict[str, Path] = {}
    if args.decisions_path is not None:
        overrides["decisions_path"] = args.decisions_path
    if args.target_trial_path is not None:
        overrides["target_trial_path"] = args.target_trial_path
    if args.results_root is not None:
        overrides["results_root"] = args.results_root
    return VerificationContext(**overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        checklist = load_checklist(args.checklist)
    except ChecklistError as exc:
        print(f"check_go_no_go failed: {exc}", file=sys.stderr)
        return 1

    context = build_verification_context(args)
    results = verify_all(context)

    print("check_go_no_go: attested checklist vs. machine-verified evidence:")
    for line in format_verification_report(checklist, results):
        print(line)

    all_passed, unmet_items = check_go_no_go(checklist)
    contradictions = contradicted_items(checklist, results)

    if not all_passed or contradictions:
        print(
            "check_go_no_go: refusing to mint a LockedEvaluationToken "
            f"({len(unmet_items)} item(s) unmet, {len(contradictions)} "
            "attestation(s) contradicted by evidence):",
            file=sys.stderr,
        )
        for name in unmet_items:
            print(f"  - {name}: unresolved", file=sys.stderr)
        for name in contradictions:
            print(
                f"  - {name}: attested true but its verifier returned failed -- "
                f"{results[name].evidence}",
                file=sys.stderr,
            )
        return 1

    # Task 20 / Requirement 18.3, 18.4: the thirteenth, always-computed
    # item. Checked after the twelve human-attested items so the report
    # above (which only covers those twelve) has already printed, and
    # before any bindable field is collected -- a not-ready ledger refuses
    # minting regardless of whether the seven bindings would have resolved.
    ready, missing_readiness_items = check_neural_readiness(context.results_root)
    if not ready:
        print(
            "check_go_no_go: refusing to mint a LockedEvaluationToken: the "
            f"readiness ledger at {context.results_root} is not ready "
            f"({len(missing_readiness_items)} item(s) missing):",
            file=sys.stderr,
        )
        for item in missing_readiness_items:
            print(f"  - {item}", file=sys.stderr)
        return 1

    # Task 20 / Requirement 13.3, 13.4: collect the seven real artifact
    # bindings. Any that could not be resolved is refused by name here,
    # before `_mint` is ever called -- `_mint`'s own field validation is a
    # second, independent check (e.g. a malformed `--evaluation-id`), not
    # the only one.
    bindings, binding_offenders = collect_bindable_fields(args)
    if binding_offenders:
        print(
            "check_go_no_go: refusing to mint a LockedEvaluationToken: "
            f"{len(binding_offenders)} bindable field offense(s):",
            file=sys.stderr,
        )
        for offense in binding_offenders:
            print(f"  - {offense}", file=sys.stderr)
        return 1

    try:
        token = LockedEvaluationToken._mint(
            issued_by="scripts/check_go_no_go.py",
            resolved_config_hash=bindings["resolved_config_hash"],
            dataset_checksum=bindings["dataset_checksum"],
            split_hash=bindings["split_hash"],
            checkpoint_hashes=bindings["checkpoint_hashes"],
            git_commit=bindings["git_commit"],
            statistical_plan_version=bindings["statistical_plan_version"],
            evaluation_id=bindings["evaluation_id"],
            results_root=context.results_root,
        )
    except TokenFieldError as exc:
        print(f"check_go_no_go: refusing to mint a LockedEvaluationToken: {exc}", file=sys.stderr)
        return 1

    if args.token_output is not None:
        args.token_output.parent.mkdir(parents=True, exist_ok=True)
        args.token_output.write_text(
            json.dumps(_serialize_token(token), indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            "check_go_no_go: all checklist items passed. LockedEvaluationToken "
            f"minted and written to {args.token_output}"
        )
    else:
        print("check_go_no_go: all checklist items passed. LockedEvaluationToken minted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
