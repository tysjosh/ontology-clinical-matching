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

Scoping decision: gating logic only, not the underlying checks
------------------------------------------------------------------

The twelve checklist items below are each, in a real run, the kind of
thing that would be independently verified by inspecting actual pipeline
artifacts -- e.g. "leakage tests passed" would check `pytest` results for
`tests/unit/test_leakage_audit.py`; "thresholds frozen" would check that
`decisions.md` Sections 12/16's thresholds have not been edited since
freeze; "splits/cross-fitting verified" would check `src/cohorts/splitting.py`
and `src/causal/propensity.py`'s fold-overlap guarantees. Building real
verification logic for all twelve items from scratch is a much larger
integration effort than this single task -- most of the actual checking
machinery already exists in earlier tasks' own modules (`leakage_audit.py`,
`splitting.py`, `estimand_alignment.py`, the privacy scanners, etc.), and
wiring every one of the twelve items to its own real check belongs to
those modules (or to a future consolidation task), not to this gate.

So `check_go_no_go.py` takes the proportionate path: it reads a
**checklist status config** (a YAML file, see
`configs/go_no_go_checklist.yaml`) where each of the twelve items has
already been marked `true`/`false` by whatever process performed that
check -- a human reviewer ticking a box, or an earlier automated stage
writing its own boolean result into this file. This script's job is
strictly the **gating logic**: read the checklist, verify 100% completion,
and either mint a token (all items `true`) or exit non-zero listing every
unmet item (any item `false` or missing). It does not re-implement, re-run,
or independently re-derive any of the twelve underlying verifications.

The eleven-name list quoted in the task description undercounts by one --
counted individually, the parenthetical actually lists twelve items. All
twelve are encoded below, in `configs/go_no_go_checklist.yaml`, and in
`CHECKLIST_ITEMS`.

Usage::

    python scripts/check_go_no_go.py --checklist configs/go_no_go_checklist.yaml
    python scripts/check_go_no_go.py --checklist path/to/checklist.yaml \\
        --token-output results/locked_evaluation_token.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from cohorts.sealed_reference import LockedEvaluationToken

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py` and `scripts/run_paper.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default checklist config path.
DEFAULT_CHECKLIST_PATH = REPO_ROOT / "configs" / "go_no_go_checklist.yaml"

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
        raise ChecklistError(
            f"{path}: top-level content must be a mapping with an 'items' key"
        )

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


def _serialize_token(token: LockedEvaluationToken) -> dict[str, str]:
    return {
        "issued_by": token.issued_by,
        "issued_at": token.issued_at.isoformat(),
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        checklist = load_checklist(args.checklist)
    except ChecklistError as exc:
        print(f"check_go_no_go failed: {exc}", file=sys.stderr)
        return 1

    all_passed, unmet_items = check_go_no_go(checklist)

    if not all_passed:
        print(
            f"check_go_no_go: {len(unmet_items)} checklist item(s) unmet -- "
            "LockedEvaluationToken NOT minted:",
            file=sys.stderr,
        )
        for name in unmet_items:
            print(f"  - {name}", file=sys.stderr)
        return 1

    token = LockedEvaluationToken._mint(issued_by="scripts/check_go_no_go.py")

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
