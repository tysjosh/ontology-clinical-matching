"""The readiness ledger (task 3; `design.md` Property 10).

Property 10, verbatim: "*For any* readiness ledger missing at least one of
the five frozen seeds, the twelve expanded ablation runs, or the eight
frozen sensitivity variants, `evaluate_readiness` returns not-ready with the
missing items named, and the Token_Minter declines to mint; readiness is
true only when all three sets are complete and every executed run appears in
the determinism-verified set."

The minter half of that sentence belongs to task 20 (which wires
`check_go_no_go.py` onto `evaluate_readiness`); this module covers the
evidence half -- the decision function and its on-disk ledger.

The generated cases drop a random subset of the 5 seeds, the 12 runs, and
the 8 variants and assert the decision names *exactly* the dropped items.
Asserting exact equality rather than containment is what makes the test
meaningful: a decision function that named every required item on every
not-ready ledger would satisfy a containment assertion while telling a
reviewer nothing about what is actually outstanding.

**Validates: Requirements 18.3, 18.4, 18.7**
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from infra.readiness import (
    ABLATION_RUN_KIND,
    DETERMINISM_KIND,
    EMPTY_LEDGER,
    EXPECTED_ABLATION_RUN_COUNT,
    EXPECTED_SENSITIVITY_VARIANT_COUNT,
    LEDGER_ENTRY_KINDS,
    LEDGER_FORMAT_VERSION,
    READINESS_LEDGER_FILENAME,
    REQUIRED_SEEDS,
    RESOLVED_FROM_EXPAND_ABLATIONS,
    RESOLVED_FROM_FROZEN_LIST,
    SEED_KIND,
    SENSITIVITY_VARIANT_KIND,
    SUB_RUN_SEPARATOR,
    WITNESS_REMOVED_ABLATION_NAME,
    LedgerConflictError,
    ReadinessLedger,
    ReadinessLedgerError,
    append_ledger_entry,
    determinism_unverified_item,
    evaluate_readiness,
    ledger_path,
    load_ledger,
    load_ledger_entries,
    missing_ablation_run_item,
    missing_seed_item,
    missing_sensitivity_variant_item,
    required_ablation_run_ids,
    required_sensitivity_variant_names,
)

_MAX_EXAMPLES = 50

REQUIRED_RUNS = required_ablation_run_ids()
REQUIRED_VARIANTS = required_sensitivity_variant_names()


def complete_ledger() -> ReadinessLedger:
    """The one ledger shape that is ready: all three sets, all verified."""
    return ReadinessLedger(
        seeds_executed=REQUIRED_SEEDS,
        ablation_runs_executed=REQUIRED_RUNS,
        sensitivity_variants_executed=REQUIRED_VARIANTS,
        determinism_verified=(*REQUIRED_RUNS, *REQUIRED_VARIANTS),
    )


# ---------------------------------------------------------------------------
# The required sets themselves
# ---------------------------------------------------------------------------


def test_required_sets_have_the_frozen_arities():
    assert REQUIRED_SEEDS == (101, 202, 303, 404, 505)
    assert len(REQUIRED_RUNS) == EXPECTED_ABLATION_RUN_COUNT == 12
    assert len(REQUIRED_VARIANTS) == EXPECTED_SENSITIVITY_VARIANT_COUNT == 8


def test_the_witness_removed_entry_expands_to_three_sub_runs():
    """`design.md` Section 2.6 tension 1: ten entries, twelve runs.

    Holds whether the ids came from task 13's `expand_ablations()` or from
    this module's fallback expansion of the frozen `ABLATIONS` list.
    """
    sub_runs = [
        run_id
        for run_id in REQUIRED_RUNS
        if run_id.startswith(f"{WITNESS_REMOVED_ABLATION_NAME}{SUB_RUN_SEPARATOR}")
    ]

    assert len(sub_runs) == 3, sub_runs
    assert WITNESS_REMOVED_ABLATION_NAME not in REQUIRED_RUNS


def test_the_required_run_set_matches_expand_ablations_once_it_exists():
    """Task 13's assertion, stated here so it starts holding automatically."""
    decision = evaluate_readiness(EMPTY_LEDGER)

    try:
        from evaluation.ablation_matrix import expand_ablations
    except ImportError:
        assert decision.required_runs_source == RESOLVED_FROM_FROZEN_LIST
        pytest.skip(
            "evaluation.ablation_matrix.expand_ablations does not exist yet "
            "(task 13); readiness is using its documented fallback expansion "
            "of the frozen ABLATIONS list."
        )

    assert decision.required_runs_source == RESOLVED_FROM_EXPAND_ABLATIONS
    assert set(REQUIRED_RUNS) == {run.run_id for run in expand_ablations()}


def test_required_variants_are_the_eight_frozen_names():
    from evaluation.robustness import SENSITIVITY_VARIANTS

    assert REQUIRED_VARIANTS == tuple(v.name for v in SENSITIVITY_VARIANTS)


# ---------------------------------------------------------------------------
# Property 10
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(
    dropped_seeds=st.sets(st.sampled_from(REQUIRED_SEEDS)),
    dropped_runs=st.sets(st.sampled_from(REQUIRED_RUNS)),
    dropped_variants=st.sets(st.sampled_from(REQUIRED_VARIANTS)),
)
def test_property_10_missing_items_are_exactly_the_missing_work(
    dropped_seeds, dropped_runs, dropped_variants
):
    """Not-ready with exactly the missing seeds, runs, and variants named.

    The empty-drop case is included on purpose: it is the ready case, and
    generating it alongside the others keeps both branches under the same
    property rather than only in a hand-written example.
    """
    executed_runs = tuple(r for r in REQUIRED_RUNS if r not in dropped_runs)
    executed_variants = tuple(v for v in REQUIRED_VARIANTS if v not in dropped_variants)
    ledger = ReadinessLedger(
        seeds_executed=tuple(s for s in REQUIRED_SEEDS if s not in dropped_seeds),
        ablation_runs_executed=executed_runs,
        sensitivity_variants_executed=executed_variants,
        # Everything executed is determinism-verified, so this case isolates
        # incomplete *execution* from unverified determinism.
        determinism_verified=(*executed_runs, *executed_variants),
    )

    decision = evaluate_readiness(ledger)

    expected = {
        *(missing_seed_item(s) for s in dropped_seeds),
        *(missing_ablation_run_item(r) for r in dropped_runs),
        *(missing_sensitivity_variant_item(v) for v in dropped_variants),
    }
    assert set(decision.missing_items) == expected
    assert len(decision.missing_items) == len(expected), "missing_items holds duplicates"
    assert decision.ready is (not expected)


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(unverified=st.sets(st.sampled_from((*REQUIRED_RUNS, *REQUIRED_VARIANTS)), min_size=1))
def test_property_10_completeness_without_determinism_is_not_ready(unverified):
    """Requirement 18.7: complete execution is not sufficient.

    Every seed, run, and variant is present; only the determinism-verified
    set is short. The decision must still be not-ready, naming exactly the
    unverified run ids.
    """
    all_runs = (*REQUIRED_RUNS, *REQUIRED_VARIANTS)
    ledger = ReadinessLedger(
        seeds_executed=REQUIRED_SEEDS,
        ablation_runs_executed=REQUIRED_RUNS,
        sensitivity_variants_executed=REQUIRED_VARIANTS,
        determinism_verified=tuple(r for r in all_runs if r not in unverified),
    )

    decision = evaluate_readiness(ledger)

    assert decision.ready is False
    assert set(decision.missing_items) == {determinism_unverified_item(r) for r in unverified}


def test_a_complete_and_verified_ledger_is_ready():
    decision = evaluate_readiness(complete_ledger())

    assert decision.ready is True
    assert decision.missing_items == ()


def test_an_empty_ledger_names_every_required_item():
    decision = evaluate_readiness(EMPTY_LEDGER)

    assert decision.ready is False
    assert len(decision.missing_items) == 5 + 12 + 8


def test_extra_runs_do_not_break_readiness_but_must_be_verified():
    """"Covers" is superset, not equality -- but an extra is still a run."""
    base = complete_ledger()
    with_unverified_extra = ReadinessLedger(
        seeds_executed=(*base.seeds_executed, 999),
        ablation_runs_executed=(*base.ablation_runs_executed, "exploratory_run"),
        sensitivity_variants_executed=base.sensitivity_variants_executed,
        determinism_verified=base.determinism_verified,
    )

    decision = evaluate_readiness(with_unverified_extra)

    assert decision.ready is False
    assert decision.missing_items == (determinism_unverified_item("exploratory_run"),)

    verified_extra = ReadinessLedger(
        seeds_executed=with_unverified_extra.seeds_executed,
        ablation_runs_executed=with_unverified_extra.ablation_runs_executed,
        sensitivity_variants_executed=with_unverified_extra.sensitivity_variants_executed,
        determinism_verified=(*base.determinism_verified, "exploratory_run"),
    )

    assert evaluate_readiness(verified_extra).ready is True


def test_missing_item_names_identify_the_offender_by_name():
    """A reviewer reading the gate's refusal can see which work is outstanding."""
    ledger = ReadinessLedger(
        seeds_executed=(101, 202, 303, 404),
        ablation_runs_executed=REQUIRED_RUNS[:-1],
        sensitivity_variants_executed=REQUIRED_VARIANTS[:-1],
        determinism_verified=(*REQUIRED_RUNS[:-1], *REQUIRED_VARIANTS[:-1]),
    )

    decision = evaluate_readiness(ledger)

    assert decision.missing_items == (
        missing_seed_item(505),
        missing_ablation_run_item(REQUIRED_RUNS[-1]),
        missing_sensitivity_variant_item(REQUIRED_VARIANTS[-1]),
    )
    assert "505" in decision.missing_items[0]
    assert REQUIRED_RUNS[-1] in decision.missing_items[1]
    assert REQUIRED_VARIANTS[-1] in decision.missing_items[2]


# ---------------------------------------------------------------------------
# On-disk ledger
# ---------------------------------------------------------------------------


def test_load_ledger_on_a_fresh_results_root_is_empty_not_an_error(tmp_path):
    """A results root where nothing has completed reports not-ready."""
    assert load_ledger(tmp_path) == EMPTY_LEDGER
    assert evaluate_readiness(load_ledger(tmp_path)).ready is False


def test_append_writes_the_ledger_at_the_designed_path(tmp_path):
    results_root = tmp_path / "results" / "abc123def4567890"

    append_ledger_entry(results_root, ABLATION_RUN_KIND, "no_overlap_gate", "hash-1")

    path = results_root / READINESS_LEDGER_FILENAME
    assert path.exists()
    assert ledger_path(results_root) == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == LEDGER_FORMAT_VERSION
    assert payload["entries"] == [
        {
            "kind": ABLATION_RUN_KIND,
            "run_id": "no_overlap_gate",
            "artifact_hash": "hash-1",
            "recorded_at": payload["entries"][0]["recorded_at"],
        }
    ]
    assert payload["entries"][0]["recorded_at"] != ""


def test_appending_every_kind_round_trips_into_a_ready_ledger(tmp_path):
    """The full runner sequence: seeds, runs, variants, then determinism."""
    for seed in REQUIRED_SEEDS:
        append_ledger_entry(tmp_path, SEED_KIND, seed, f"seed-hash-{seed}")
    for run_id in REQUIRED_RUNS:
        append_ledger_entry(tmp_path, ABLATION_RUN_KIND, run_id, f"run-hash-{run_id}")
    for name in REQUIRED_VARIANTS:
        append_ledger_entry(tmp_path, SENSITIVITY_VARIANT_KIND, name, f"var-hash-{name}")

    partway = load_ledger(tmp_path)
    assert partway.seeds_executed == REQUIRED_SEEDS
    assert evaluate_readiness(partway).ready is False

    for run_id in (*REQUIRED_RUNS, *REQUIRED_VARIANTS):
        append_ledger_entry(tmp_path, DETERMINISM_KIND, run_id, f"verified-{run_id}")

    assert evaluate_readiness(load_ledger(tmp_path)).ready is True


def test_appending_is_append_only_and_idempotent(tmp_path):
    append_ledger_entry(tmp_path, ABLATION_RUN_KIND, "no_overlap_gate", "hash-1")
    append_ledger_entry(tmp_path, ABLATION_RUN_KIND, "no_uncertainty_gate", "hash-2")
    repeated = append_ledger_entry(tmp_path, ABLATION_RUN_KIND, "no_overlap_gate", "hash-1")

    entries = load_ledger_entries(tmp_path)

    assert [e.run_id for e in entries] == ["no_overlap_gate", "no_uncertainty_gate"]
    assert repeated.artifact_hash == "hash-1"


def test_the_same_run_recorded_with_a_different_hash_is_refused(tmp_path):
    """Requirement 18.7's failure mode is raised, not silently recorded."""
    append_ledger_entry(tmp_path, ABLATION_RUN_KIND, "no_overlap_gate", "hash-1")

    with pytest.raises(LedgerConflictError) as excinfo:
        append_ledger_entry(tmp_path, ABLATION_RUN_KIND, "no_overlap_gate", "hash-2")

    message = str(excinfo.value)
    assert "no_overlap_gate" in message
    assert "hash-1" in message and "hash-2" in message
    assert [e.artifact_hash for e in load_ledger_entries(tmp_path)] == ["hash-1"]


@pytest.mark.parametrize(
    ("kind", "run_id", "artifact_hash"),
    [
        ("not_a_kind", "no_overlap_gate", "hash-1"),
        (ABLATION_RUN_KIND, "  ", "hash-1"),
        (ABLATION_RUN_KIND, "no_overlap_gate", ""),
    ],
)
def test_append_rejects_malformed_records_and_writes_nothing(
    tmp_path, kind, run_id, artifact_hash
):
    with pytest.raises(ValueError):
        append_ledger_entry(tmp_path, kind, run_id, artifact_hash)

    assert not ledger_path(tmp_path).exists()


def test_every_kind_name_is_accepted_by_append(tmp_path):
    for index, kind in enumerate(LEDGER_ENTRY_KINDS):
        run_id = str(101 + index) if kind == SEED_KIND else f"run_{index}"
        append_ledger_entry(tmp_path, kind, run_id, f"hash-{index}")

    assert len(load_ledger_entries(tmp_path)) == len(LEDGER_ENTRY_KINDS)


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '["entries"]',
        '{"format_version": 1}',
        '{"format_version": 1, "entries": {}}',
        '{"format_version": 1, "entries": ["a string, not a mapping"]}',
        '{"format_version": 1, "entries": [{"kind": "seed", "run_id": "101"}]}',
        '{"format_version": 1, "entries": [{"kind": "nope", "run_id": "x", '
        '"artifact_hash": "h"}]}',
    ],
)
def test_a_malformed_ledger_file_raises_rather_than_reading_as_empty(tmp_path, content):
    """An unreadable ledger must not silently look like "nothing executed"."""
    ledger_path(tmp_path).write_text(content, encoding="utf-8")

    with pytest.raises(ReadinessLedgerError):
        load_ledger(tmp_path)


def test_a_non_integer_seed_entry_is_rejected(tmp_path):
    ledger_path(tmp_path).write_text(
        json.dumps(
            {
                "format_version": LEDGER_FORMAT_VERSION,
                "entries": [
                    {
                        "kind": SEED_KIND,
                        "run_id": "one-hundred-one",
                        "artifact_hash": "h",
                        "recorded_at": "2025-01-01T00:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReadinessLedgerError):
        load_ledger(tmp_path)
