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

Token bindings are NOT wired yet (task 20 owns that)
-----------------------------------------------------

`cohorts.sealed_reference.LockedEvaluationToken` now binds nine fields
(`neural-experiment-pipeline` Requirement 13.3): `issued_by`, `issued_at`,
and seven artifact bindings this script is the designated minter for. Task
7 strengthened the token; **task 20 wires the real values into this
script** -- computing the resolved-config hash, dataset checksum, split
hash, per-checkpoint sha256s, git commit, statistical-plan version, and a
fresh evaluation id, and refusing to mint if any of them is unavailable.

Until then, this script mints with `UNWIRED_BINDING_PLACEHOLDER` in every
one of the seven slots. This is deliberately *not* a plausible-looking
value: it is a single, loudly-labeled sentinel string, so a token artifact
produced today cannot be mistaken for one bound to real artifacts, and
`scripts/run_locked_evaluation.py` (task 21) recomputing the bound hashes
per Requirement 13.5 will mismatch on the first field and refuse. Nothing
in this file pretends to know the real values.

`--results-root` supplies the consumption ledger the mint checks for a
duplicated `evaluation_id` (Requirement 13.4); its default is
`checklist_verifiers.DEFAULT_RESULTS_ROOT`, the same `results/` tree the
verifiers already read.

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
from pathlib import Path

import yaml

from cohorts.sealed_reference import LockedEvaluationToken
from infra.checklist_verifiers import (
    VerificationContext,
    VerifierResult,
    contradicted_items,
    verify_all,
)

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py` and `scripts/run_paper.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default checklist config path.
DEFAULT_CHECKLIST_PATH = REPO_ROOT / "configs" / "go_no_go_checklist.yaml"

#: Stand-in for each of the seven artifact bindings Requirement 13.3
#: requires, until task 20 computes them for real (see module docstring).
#: One sentinel for every slot, loudly labeled, so a token minted today is
#: recognizable on sight as unbound and fails task 21's Requirement-13.5
#: hash recomputation immediately.
UNWIRED_BINDING_PLACEHOLDER: str = "UNWIRED-PLACEHOLDER-see-task-20"

#: The twelve frozen go/no-go checklist item names (`decisions.md` Section
#: 21; task 16's parenthetical list, counted individually). This is the
#: authoritative set: `load_checklist()` requires every one of these keys
#: to be present, and `check_go_no_go()` only ever reports unmet items
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

    known = set(CHECKLIST_ITEMS)
    unknown = sorted(set(items) - known)
    if unknown:
        raise ChecklistError(f"{path}: unknown checklist item(s): {', '.join(unknown)}")

    missing = sorted(known - set(items))
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
            "privacy_checks_passed verifier (default: results/)."
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

    # TASK 20: replace every UNWIRED_BINDING_PLACEHOLDER below with the real
    # recomputed value, and refuse to mint (naming the field) when one is
    # unavailable. See the module docstring: nothing here claims to know
    # these values yet, and a token minted with the sentinel is rejected by
    # task 21's Requirement-13.5 hash recomputation.
    token = LockedEvaluationToken._mint(
        issued_by="scripts/check_go_no_go.py",
        resolved_config_hash=UNWIRED_BINDING_PLACEHOLDER,
        dataset_checksum=UNWIRED_BINDING_PLACEHOLDER,
        split_hash=UNWIRED_BINDING_PLACEHOLDER,
        checkpoint_hashes=((UNWIRED_BINDING_PLACEHOLDER, UNWIRED_BINDING_PLACEHOLDER),),
        git_commit=UNWIRED_BINDING_PLACEHOLDER,
        statistical_plan_version=UNWIRED_BINDING_PLACEHOLDER,
        evaluation_id=UNWIRED_BINDING_PLACEHOLDER,
        results_root=context.results_root,
    )

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
