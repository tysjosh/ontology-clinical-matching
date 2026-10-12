"""Unit tests for the frozen ablation matrix (`decisions.md` Section 17;
Requirement 12) and its task-13 expansion into twelve executed runs
(Requirement 7.1, 7.8; `design.md` Section 2.6 tension 1).

**Validates: Requirements 7.1, 7.8, 6.1, 18.3**
(`.kiro/specs/neural-experiment-pipeline/requirements.md`)
"""

from __future__ import annotations

from evaluation.ablation_matrix import ABLATIONS, AblationRun, expand_ablations
from evaluation.robustness import SENSITIVITY_VARIANTS
from models.variants import ABLATION_ENTRY_NAMES, variant_for_ablation, variant_for_sensitivity

EXPECTED_NAMES = [
    "no_treatment_input",
    "treatment_as_ordinary_feature",
    "explicit_intervention_channel",
    "intervention_channel_without_protocol",
    "no_overlap_gate",
    "no_uncertainty_gate",
    "no_clinical_constraints",
    "witness_removed",
    "predictive_accuracy_only",
    "placebo_intervention_channel",
]


def test_exactly_ten_ablations():
    """`decisions.md` Section 17 freezes exactly ten primary ablations."""
    assert len(ABLATIONS) == 10


def test_ablation_names_are_unique():
    assert len({a.name for a in ABLATIONS}) == 10


def test_ablation_names_are_stable():
    """Names must match the frozen set exactly -- an accidental rename of any
    entry (e.g. by a future refactor) must fail this test rather than pass
    silently.
    """
    assert [a.name for a in ABLATIONS] == EXPECTED_NAMES


def test_every_ablation_has_a_nonempty_description_and_config_diff():
    for ablation in ABLATIONS:
        assert isinstance(ablation.description, str)
        assert ablation.description.strip() != ""
        assert isinstance(ablation.config_diff, dict)
        assert len(ablation.config_diff) > 0


# ---------------------------------------------------------------------------
# expand_ablations: ten entries -> twelve runs (design.md Section 2.6
# tension 1; Requirement 7.1, 7.8)
# ---------------------------------------------------------------------------


def test_ablations_list_stays_at_exactly_ten_entries_after_expansion_exists():
    """`expand_ablations` must never cause `ABLATIONS` itself to be
    mutated, reordered, or resized -- the frozen entry count is a separate
    invariant from the expanded run count."""
    assert len(ABLATIONS) == 10
    assert [a.name for a in ABLATIONS] == EXPECTED_NAMES


def test_expand_ablations_produces_exactly_twelve_runs():
    runs = expand_ablations()
    assert len(runs) == 12


def test_exactly_three_runs_share_source_entry_witness_removed():
    runs = expand_ablations()
    witness_runs = [r for r in runs if r.source_entry == "witness_removed"]
    assert len(witness_runs) == 3
    assert {r.sub_run_key for r in witness_runs} == {"temporal", "causal", "clinical"}


def test_witness_removed_sub_run_ids_use_the_double_colon_separator():
    runs = expand_ablations()
    witness_run_ids = {r.run_id for r in runs if r.source_entry == "witness_removed"}
    assert witness_run_ids == {
        "witness_removed::temporal",
        "witness_removed::causal",
        "witness_removed::clinical",
    }


def test_non_witness_runs_have_run_id_equal_to_source_entry():
    runs = expand_ablations()
    for run in runs:
        if run.source_entry != "witness_removed":
            assert run.run_id == run.source_entry
            assert run.sub_run_key is None


def test_six_runs_retrain_and_six_runs_reuse_predictions():
    runs = expand_ablations()
    retrain = [r for r in runs if r.execution_mode == "retrain"]
    reuse = [r for r in runs if r.execution_mode == "reuse_predictions"]
    assert len(retrain) == 6
    assert len(reuse) == 6


def test_every_run_id_is_unique():
    runs = expand_ablations()
    run_ids = [r.run_id for r in runs]
    assert len(run_ids) == len(set(run_ids))


def test_every_run_is_an_ablation_run_instance_with_a_variant():
    runs = expand_ablations()
    for run in runs:
        assert isinstance(run, AblationRun)
        assert run.variant is not None


# ---------------------------------------------------------------------------
# variant_for_ablation / variant_for_sensitivity bridges (design.md
# Section 2.6 tension 2; Requirement 6.1)
# ---------------------------------------------------------------------------


def test_ablation_entry_names_mapping_key_set_matches_ablations():
    assert ABLATION_ENTRY_NAMES == {a.name for a in ABLATIONS}


def test_variant_for_ablation_covers_every_expanded_run():
    """Every run `expand_ablations()` produces must resolve through
    `variant_for_ablation` without raising -- the mapping in
    `models.variants` must be total over the twelve expanded keys, not
    just the ten entry names."""
    for run in expand_ablations():
        resolved = variant_for_ablation(run)
        assert resolved is run.variant


def test_variant_for_sensitivity_covers_every_frozen_sensitivity_name():
    names = {v.name for v in SENSITIVITY_VARIANTS}
    assert len(names) == 8
    for name in names:
        # Must not raise.
        variant_for_sensitivity(name)


def test_variant_for_sensitivity_rejects_unknown_name():
    import pytest

    with pytest.raises(KeyError):
        variant_for_sensitivity("not_a_real_sensitivity_variant")


def test_sensitivity_variant_names_mapping_key_set_matches_sensitivity_variants():
    from models.variants import _SENSITIVITY_VARIANTS  # noqa: PLC0415 -- test-only introspection

    assert set(_SENSITIVITY_VARIANTS) == {v.name for v in SENSITIVITY_VARIANTS}


# ---------------------------------------------------------------------------
# Readiness-ledger agreement (design.md Section 2.6 tension 7)
# ---------------------------------------------------------------------------


def test_variant_collision_detection_import():
    """Guard against the import below silently failing to find the names
    this test module needs; a dedicated section follows."""
    from evaluation.ablation_matrix import (  # noqa: F401
        COLLISION_PREDICTION_TOLERANCE,
        CollisionReport,
        PredictionCollision,
        VariantCollisionError,
        detect_variant_collisions,
    )


def test_evaluate_readiness_required_ablation_run_set_matches_expand_ablations():
    """`infra.readiness.required_ablation_run_ids()` must, now that
    `expand_ablations` exists, resolve from it (not its frozen-list
    fallback) and agree exactly."""
    from infra.readiness import RESOLVED_FROM_EXPAND_ABLATIONS, resolve_required_ablation_runs

    required_runs, source = resolve_required_ablation_runs()

    assert source == RESOLVED_FROM_EXPAND_ABLATIONS
    assert set(required_runs) == {r.run_id for r in expand_ablations()}
