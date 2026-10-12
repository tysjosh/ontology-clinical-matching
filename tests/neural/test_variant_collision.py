"""Torch-dependent half of Property 11 (task 14.3; Requirement 6.4, 7.13).

"All twelve variants built through `build_model` on one fixed batch
produce pairwise output differences above 1e-9 wherever
`model_affecting_fields` differ." The pure-Python detection-logic half of
Property 11 lives in `tests/unit/test_variant_collision.py`.

**Validates: Requirements 6.4, 7.13**
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from evaluation.ablation_matrix import (  # noqa: E402
    COLLISION_PREDICTION_TOLERANCE,
    expand_ablations,
)
from models.dataset import PatientDatasetBuilder  # noqa: E402
from models.torch_intervention_aware import outcome_from_hazards  # noqa: E402
from models.torch_training import factual_outputs  # noqa: E402
from models.variants import build_model, model_affecting_fields  # noqa: E402

DIMENSIONS = PatientDatasetBuilder().dimensions

ALL_RUNS = expand_ablations()


def _predicted_risks(variant, dimensions, batch, seed: int) -> torch.Tensor:
    model = build_model(variant, dimensions, seed)
    with torch.no_grad():
        outputs = factual_outputs(model, batch)
        return outcome_from_hazards(outputs.hazards)


@pytest.fixture(scope="module")
def val_batch():
    from tests.neural.test_dataset_builder import _cohort

    records = _cohort(20, prefix="V", seed=17)
    builder = PatientDatasetBuilder().fit(records)
    return builder.build(records, "val")


#: Of `models.variants.MODEL_AFFECTING_FIELD_NAMES`, only these three
#: actually change what `build_model` constructs (architecture, input
#: content). `causal_adjustment` and `balance_penalty_enabled` change
#: only the *training loss* (`models.torch_training.compute_loss_terms`)
#: -- nothing about construction, `forward`, or `rollout_both_arms` reads
#: either field. Two freshly *constructed but untrained* models that
#: differ only in those two loss-weighting fields are therefore bitwise
#: identical at construction time by necessity, not by defect: they
#: diverge only after training under their different objectives, which
#: is what the production `detect_variant_collisions` call site actually
#: checks (trained checkpoints' predictions), not fresh `build_model`
#: output. Any pair sharing these three fields is excluded from the
#: construction-time check below for that documented reason, confirmed
#: explicitly by `test_predictive_accuracy_only_collides_with_the_complete_model_before_training`.
_ARCHITECTURE_FIELD_NAMES = ("model_family", "treatment_representation", "protocol_channel")


def _architecture_fields(variant) -> dict[str, object]:
    return {name: getattr(variant, name) for name in _ARCHITECTURE_FIELD_NAMES}


def test_every_pair_of_model_affecting_different_runs_produces_differing_predictions(val_batch):
    """For every pair of the twelve runs whose `model_affecting_fields`
    differ in a way that reaches construction (i.e. their architecture
    fields also differ -- see the module-level note above), built at the
    same seed on the same batch, at least one patient's predicted risk
    differs by more than `COLLISION_PREDICTION_TOLERANCE`."""
    seed = 101
    predictions_by_run = {
        run.run_id: _predicted_risks(run.variant, DIMENSIONS, val_batch, seed)
        for run in ALL_RUNS
    }

    checked_pairs = 0
    for i, run_a in enumerate(ALL_RUNS):
        for run_b in ALL_RUNS[i + 1 :]:
            if model_affecting_fields(run_a.variant) == model_affecting_fields(run_b.variant):
                continue
            if _architecture_fields(run_a.variant) == _architecture_fields(run_b.variant):
                # Differ only in causal_adjustment/balance_penalty_enabled
                # (loss-only fields) -- a legitimate, documented
                # construction-time collision, not checked here.
                continue
            checked_pairs += 1
            diff = (predictions_by_run[run_a.run_id] - predictions_by_run[run_b.run_id]).abs()
            max_diff = float(diff.max())
            assert max_diff > COLLISION_PREDICTION_TOLERANCE, (
                f"{run_a.run_id} vs {run_b.run_id} collide: max_diff={max_diff}"
            )

    # Sanity: there really were model-affecting-different pairs checked
    # (otherwise this test would vacuously pass).
    assert checked_pairs > 0


def test_predictive_accuracy_only_collides_with_the_complete_model_before_training(val_batch):
    """Documents, rather than hides, the one expected construction-time
    collision: `causal_adjustment`/`balance_penalty_enabled` affect only
    the training loss, so untrained predictions are bitwise identical.
    Production collision detection operates on trained-model predictions
    (`scripts/run_ablations.py`'s `predictions_by_run`), where this pair
    is expected to diverge after training under their different
    objectives -- this test only confirms the pre-training premise."""
    seed = 101
    complete_run = next(r for r in ALL_RUNS if r.run_id == "explicit_intervention_channel")
    predictive_only_run = next(r for r in ALL_RUNS if r.run_id == "predictive_accuracy_only")

    risks_complete = _predicted_risks(complete_run.variant, DIMENSIONS, val_batch, seed)
    risks_predictive_only = _predicted_risks(
        predictive_only_run.variant, DIMENSIONS, val_batch, seed
    )
    assert torch.equal(risks_complete, risks_predictive_only)


def test_runs_sharing_model_affecting_fields_produce_identical_predictions(val_batch):
    """The converse sanity check: reuse-mode runs that share every
    model-affecting field (e.g. no_overlap_gate vs no_uncertainty_gate,
    both the complete intervention-aware model) are built identically and
    so legitimately produce identical predictions at a fixed seed -- this
    is expected, not a defect, since they differ only in post-hoc
    gating."""
    seed = 101
    no_overlap_gate = next(r for r in ALL_RUNS if r.run_id == "no_overlap_gate")
    no_uncertainty_gate = next(r for r in ALL_RUNS if r.run_id == "no_uncertainty_gate")
    assert model_affecting_fields(no_overlap_gate.variant) == model_affecting_fields(
        no_uncertainty_gate.variant
    )

    risks_a = _predicted_risks(no_overlap_gate.variant, DIMENSIONS, val_batch, seed)
    risks_b = _predicted_risks(no_uncertainty_gate.variant, DIMENSIONS, val_batch, seed)
    assert torch.equal(risks_a, risks_b)
