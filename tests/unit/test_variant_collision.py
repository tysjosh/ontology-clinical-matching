"""Unit tests for `evaluation.ablation_matrix.detect_variant_collisions`
(task 14.3; Requirement 6.4, 7.13, 18.3; Property 11).

Property 11, verbatim: "*For any* pair of runs at the same seed whose
`model_affecting_fields` differ, their Validation_Split patient-level
predictions differ by more than 1e-9 for at least one patient, and the
six retrained runs yield six distinct primary-metric values; when either
conjunct fails, the collision detector names the colliding runs and the
ablation table is marked invalid rather than published."

The torch-dependent half of Property 11 ("all twelve variants built
through `build_model` on one fixed batch produce pairwise output
differences above 1e-9 wherever `model_affecting_fields` differ") lives
in `tests/neural/test_variant_collision.py`, behind
`pytest.importorskip("torch")`, since it requires constructing real
torch models. This module covers the pure-Python detection logic itself
with hand-built prediction/metric dicts.

**Validates: Requirements 6.4, 7.13, 18.3**
"""

from __future__ import annotations

from evaluation.ablation_matrix import (
    COLLISION_PREDICTION_TOLERANCE,
    detect_variant_collisions,
    expand_ablations,
)

ALL_RUNS = expand_ablations()
RETRAIN_RUNS = tuple(run for run in ALL_RUNS if run.execution_mode == "retrain")
RETRAIN_IDS = tuple(run.run_id for run in RETRAIN_RUNS)


def _run(run_id: str):
    return next(run for run in ALL_RUNS if run.run_id == run_id)


def _distinct_metrics(seed: int = 101) -> dict[str, dict[int, float]]:
    """Six distinct metric values for the six retrain-mode runs, at `seed`."""
    return {run_id: {seed: float(index)} for index, run_id in enumerate(RETRAIN_IDS)}


def _no_predictions() -> dict[str, dict[int, dict[str, float]]]:
    return {}


# ---------------------------------------------------------------------------
# No collision: distinct predictions, distinct metrics
# ---------------------------------------------------------------------------


def test_no_collision_when_predictions_differ_and_metrics_are_distinct():
    predictions = {
        "no_treatment_input": {101: {"P1": 0.1, "P2": 0.2}},
        "treatment_as_ordinary_feature": {101: {"P1": 0.9, "P2": 0.8}},
    }
    report = detect_variant_collisions(
        [_run("no_treatment_input"), _run("treatment_as_ordinary_feature")],
        predictions,
        _distinct_metrics(),
    )

    assert report.has_collision is False
    assert report.prediction_collisions == ()
    assert report.insufficient_distinct_metrics is False
    assert report.distinct_metric_count == 6


# ---------------------------------------------------------------------------
# Prediction collision
# ---------------------------------------------------------------------------


def test_identical_predictions_for_model_affecting_different_runs_is_flagged():
    """Deliberately identical prediction sets for two model-affecting-
    different runs are flagged with both run names."""
    identical = {"P1": 0.42, "P2": 0.73, "P3": 0.11}
    predictions = {
        "no_treatment_input": {101: dict(identical)},
        "treatment_as_ordinary_feature": {101: dict(identical)},
    }

    report = detect_variant_collisions(
        [_run("no_treatment_input"), _run("treatment_as_ordinary_feature")],
        predictions,
        _distinct_metrics(),
    )

    assert report.has_collision is True
    assert len(report.prediction_collisions) == 1
    collision = report.prediction_collisions[0]
    assert {collision.run_id_a, collision.run_id_b} == {
        "no_treatment_input",
        "treatment_as_ordinary_feature",
    }
    assert collision.max_absolute_difference < COLLISION_PREDICTION_TOLERANCE


def test_predictions_within_tolerance_but_not_identical_still_collide():
    predictions = {
        "no_treatment_input": {101: {"P1": 0.5, "P2": 0.5}},
        "treatment_as_ordinary_feature": {
            101: {"P1": 0.5 + 1e-12, "P2": 0.5 - 1e-12}
        },
    }
    report = detect_variant_collisions(
        [_run("no_treatment_input"), _run("treatment_as_ordinary_feature")],
        predictions,
        _distinct_metrics(),
    )
    assert report.has_collision is True


def test_predictions_differing_above_tolerance_do_not_collide():
    predictions = {
        "no_treatment_input": {101: {"P1": 0.5, "P2": 0.5}},
        "treatment_as_ordinary_feature": {101: {"P1": 0.5 + 1e-6, "P2": 0.5}},
    }
    report = detect_variant_collisions(
        [_run("no_treatment_input"), _run("treatment_as_ordinary_feature")],
        predictions,
        _distinct_metrics(),
    )
    assert report.prediction_collisions == ()


def test_runs_sharing_model_affecting_fields_are_never_flagged_even_if_identical():
    """Two reuse-mode runs (same model, different gate) legitimately share
    identical predictions by construction -- not a collision."""
    identical = {"P1": 0.3, "P2": 0.6}
    predictions = {
        "no_overlap_gate": {101: dict(identical)},
        "no_uncertainty_gate": {101: dict(identical)},
    }
    report = detect_variant_collisions(
        [_run("no_overlap_gate"), _run("no_uncertainty_gate")],
        predictions,
        _distinct_metrics(),
    )
    assert report.prediction_collisions == ()
    assert report.has_collision is False


def test_runs_with_no_shared_patients_at_a_seed_are_not_compared():
    predictions = {
        "no_treatment_input": {101: {"P1": 0.4}},
        "treatment_as_ordinary_feature": {101: {"P2": 0.4}},  # disjoint patient set
    }
    report = detect_variant_collisions(
        [_run("no_treatment_input"), _run("treatment_as_ordinary_feature")],
        predictions,
        _distinct_metrics(),
    )
    assert report.prediction_collisions == ()


# ---------------------------------------------------------------------------
# Fewer-than-six-distinct-metrics
# ---------------------------------------------------------------------------


def test_five_identical_retrained_metric_values_fire_the_fewer_than_six_branch():
    metrics = {run_id: {101: 0.75} for run_id in RETRAIN_IDS}  # all six identical -> 1 distinct

    report = detect_variant_collisions(list(RETRAIN_RUNS), _no_predictions(), metrics)

    assert report.insufficient_distinct_metrics is True
    assert report.distinct_metric_count == 1
    assert report.has_collision is True


def test_exactly_five_distinct_values_among_six_runs_is_insufficient():
    metrics = {run_id: {101: float(i)} for i, run_id in enumerate(RETRAIN_IDS)}
    # Make two runs share a value -> 5 distinct values among 6 runs.
    metrics[RETRAIN_IDS[1]] = metrics[RETRAIN_IDS[0]]

    report = detect_variant_collisions(list(RETRAIN_RUNS), _no_predictions(), metrics)

    assert report.distinct_metric_count == 5
    assert report.insufficient_distinct_metrics is True


def test_six_distinct_values_is_sufficient():
    metrics = _distinct_metrics()
    report = detect_variant_collisions(list(RETRAIN_RUNS), _no_predictions(), metrics)

    assert report.distinct_metric_count == 6
    assert report.insufficient_distinct_metrics is False


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


def test_collision_report_to_dict_is_json_serializable():
    import json

    identical = {"P1": 0.5}
    predictions = {
        "no_treatment_input": {101: dict(identical)},
        "treatment_as_ordinary_feature": {101: dict(identical)},
    }
    report = detect_variant_collisions(
        [_run("no_treatment_input"), _run("treatment_as_ordinary_feature")],
        predictions,
        _distinct_metrics(),
    )

    serialized = json.dumps(report.to_dict())
    assert "no_treatment_input" in serialized
