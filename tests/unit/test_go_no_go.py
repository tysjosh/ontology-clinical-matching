"""Property-based tests for `scripts/check_go_no_go.py` (task 20;
`design.md` Property 26: "Minting names every offender").

Property 26, verbatim: "*For any* non-empty subset of the token's bound
fields made absent, empty, or duplicated, and *for any* non-empty subset
of checklist items made unresolved, absent, or non-boolean, the
Token_Minter declines, names exactly that subset, exits non-zero, and
writes no token artifact."

Two generated dimensions, exercised independently (each test holds the
other dimension fixed at "everything resolved") so a failing example
isolates to one offense category:

- a non-empty subset of `check_go_no_go.CHECKLIST_ITEMS` (the twelve
  human-attested items) set to `False` -- `check_go_no_go()` must name
  exactly that subset as unmet;
- a non-empty subset of `check_go_no_go.BINDABLE_FIELDS` (the seven
  artifact bindings this script collects via CLI args) left unresolved
  (not supplied, and for `checkpoint_hashes` specifically, supplied but
  malformed) -- `collect_bindable_fields()` must name exactly that subset
  as offenders.

`tests/unit/test_check_go_no_go.py` already covers the readiness-ledger
gate (Requirement 18.3, 18.4) and the end-to-end `main()` refusal paths
with concrete examples, including the required case of an all-true
checklist blocked by a not-ready ledger; this module focuses on the
combinatorial "every subset, named exactly" guarantee itself.

**Validates: Requirements 13.4, 13.13, 15.3, 15.5**
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_go_no_go  # noqa: E402  (import after sys.path manipulation, by design)

_MAX_EXAMPLES = 50


# ---------------------------------------------------------------------------
# Checklist-item dimension: unresolved human-attested items.
# ---------------------------------------------------------------------------


def _all_true() -> dict[str, bool]:
    return dict.fromkeys(check_go_no_go.CHECKLIST_ITEMS, True)


@given(
    unresolved=st.sets(
        st.sampled_from(check_go_no_go.CHECKLIST_ITEMS), min_size=1
    )
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_26_unresolved_checklist_items_are_named_exactly(unresolved: set[str]) -> None:
    checklist = _all_true()
    for name in unresolved:
        checklist[name] = False

    all_passed, unmet = check_go_no_go.check_go_no_go(checklist)

    assert all_passed is False
    assert set(unmet) == unresolved


@given(
    absent=st.sets(
        st.sampled_from(check_go_no_go.CHECKLIST_ITEMS), min_size=1
    )
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_26_absent_checklist_items_are_named_exactly(absent: set[str]) -> None:
    """An item absent from the mapping entirely (not merely `False`) is
    treated identically to an unresolved one -- `.get(name, False)`."""
    checklist = _all_true()
    for name in absent:
        del checklist[name]

    all_passed, unmet = check_go_no_go.check_go_no_go(checklist)

    assert all_passed is False
    assert set(unmet) == absent


# ---------------------------------------------------------------------------
# Bindable-field dimension: absent/malformed artifact bindings.
# ---------------------------------------------------------------------------


def _complete_bindable_args() -> dict[str, object]:
    """An `argparse.Namespace`-shaped dict with every bindable field
    resolvable, as a plain dict so individual keys can be dropped per
    generated example."""
    return {
        "resolved_config_hash": "test-resolved-config-hash",
        "dataset_checksum": "test-dataset-checksum",
        "split_hash": "test-split-hash",
        "checkpoint_hash": ["intervention_aware=test-checkpoint-sha256"],
        "git_commit": "test-git-commit",
        "statistical_plan_version": "test-statistical-plan-version",
        "evaluation_id": "test-evaluation-id",
    }


def _namespace(values: dict[str, object]):
    import argparse

    return argparse.Namespace(**values)


#: Maps each `BINDABLE_FIELDS` name to the CLI-arg attribute name
#: `collect_bindable_fields` reads it from -- `checkpoint_hashes`'s source
#: attribute is `checkpoint_hash` (singular, since it's a repeatable
#: `--checkpoint-hash` flag), every other field's source attribute matches
#: its bound-field name.
_SOURCE_ATTR_BY_BINDABLE_FIELD: dict[str, str] = {
    "resolved_config_hash": "resolved_config_hash",
    "dataset_checksum": "dataset_checksum",
    "split_hash": "split_hash",
    "checkpoint_hashes": "checkpoint_hash",
    "git_commit": "git_commit",
    "statistical_plan_version": "statistical_plan_version",
    "evaluation_id": "evaluation_id",
}


@given(
    unresolved=st.sets(
        st.sampled_from(check_go_no_go.BINDABLE_FIELDS), min_size=1
    )
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_26_absent_bindable_fields_are_named_exactly(unresolved: set[str]) -> None:
    """Every bindable field left unresolved (its source CLI attribute set
    to `None`) is named as an offender -- except `statistical_plan_version`
    and `evaluation_id`, which auto-resolve to a real value rather than
    offending when omitted (see `collect_bindable_fields`'s docstring: only
    `resolved_config_hash`, `dataset_checksum`, `split_hash`, and
    `checkpoint_hashes` are hard-required caller inputs; `git_commit` also
    auto-resolves but is excluded from this generated set since its
    offense path requires a real git repository, covered by
    `tests/unit/test_check_go_no_go.py::test_main_refuses_to_mint_when_the_working_tree_is_dirty`
    instead)."""
    always_offending = unresolved - {
        "statistical_plan_version",
        "evaluation_id",
        "git_commit",
    }
    if not always_offending:
        return  # nothing in this generated subset can actually offend; skip

    values = _complete_bindable_args()
    for field_name in always_offending:
        values[_SOURCE_ATTR_BY_BINDABLE_FIELD[field_name]] = None

    args = _namespace(values)
    bindings, offenders = check_go_no_go.collect_bindable_fields(args)

    offending_field_names = {offense.split(":")[0] for offense in offenders}
    assert offending_field_names == always_offending


def test_property_26_malformed_checkpoint_hash_offends_even_when_supplied() -> None:
    """`checkpoint_hashes` offends not only when absent but also when every
    supplied `--checkpoint-hash` value is malformed (no `NAME=SHA256`
    separator)."""
    values = _complete_bindable_args()
    values["checkpoint_hash"] = ["not-a-valid-pair"]

    args = _namespace(values)
    bindings, offenders = check_go_no_go.collect_bindable_fields(args)

    offending_field_names = {offense.split(":")[0] for offense in offenders}
    assert offending_field_names == {"checkpoint_hashes"}


# ---------------------------------------------------------------------------
# Required case: a not-ready ledger blocks minting even with every
# human-attested item true.
# ---------------------------------------------------------------------------


def test_a_not_ready_ledger_blocks_minting_with_all_twelve_items_true(tmp_path) -> None:
    checklist_path = tmp_path / "checklist.yaml"
    import yaml

    checklist_path.write_text(yaml.safe_dump({"items": _all_true()}))

    exit_code = check_go_no_go.main(
        [
            "--checklist",
            str(checklist_path),
            "--results-root",
            str(tmp_path / "results"),  # never populated: not ready
            "--resolved-config-hash",
            "test-resolved-config-hash",
            "--dataset-checksum",
            "test-dataset-checksum",
            "--split-hash",
            "test-split-hash",
            "--checkpoint-hash",
            "intervention_aware=test-checkpoint-sha256",
            "--git-commit",
            "test-git-commit",
        ]
    )

    assert exit_code != 0
    assert not (tmp_path / "token.json").exists()
