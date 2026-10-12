"""Unit tests for `evaluation.ablation_reuse` (task 14.1; Requirement
7.6, 7.7, 7.8, 7.15; Property 28).

**Validates: Requirements 7.6, 7.7, 7.8, 7.15**
"""

from __future__ import annotations

import json

import pytest

from evaluation.ablation_matrix import expand_ablations
from evaluation.ablation_reuse import (
    REUSE_RUN_IDS,
    CompleteModelPredictions,
    PatientPrediction,
    ReusePredictionSourceError,
    load_complete_model_predictions,
    run_all_reuse_ablations,
    run_reuse_ablation,
)

CHECKPOINT_HASH = "a" * 64


def _patient(
    patient_id: str,
    *,
    predicted_risk: float = 0.4,
    propensity_in_common_support: bool = True,
    ensemble_uncertainty_ok: bool = True,
    protocol_valid: bool = True,
    no_hard_violation: bool = True,
) -> PatientPrediction:
    supported = (
        propensity_in_common_support
        and ensemble_uncertainty_ok
        and protocol_valid
        and no_hard_violation
    )
    return PatientPrediction(
        patient_id=patient_id,
        predicted_risk=predicted_risk,
        propensity_in_common_support=propensity_in_common_support,
        ensemble_uncertainty_ok=ensemble_uncertainty_ok,
        protocol_valid=protocol_valid,
        no_hard_violation=no_hard_violation,
        supported=supported,
    )


def _source(patients: tuple[PatientPrediction, ...]) -> CompleteModelPredictions:
    return CompleteModelPredictions(
        split_name="val", checkpoint_sha256=CHECKPOINT_HASH, patients=patients
    )


# ---------------------------------------------------------------------------
# Reuse run id set matches evaluation.ablation_matrix's reuse-mode subset
# ---------------------------------------------------------------------------


def test_reuse_run_ids_match_expand_ablations_reuse_mode_subset():
    expected = {
        run.run_id for run in expand_ablations() if run.execution_mode == "reuse_predictions"
    }
    assert REUSE_RUN_IDS == expected
    assert len(REUSE_RUN_IDS) == 6


# ---------------------------------------------------------------------------
# Property 28: reused predictions reproduced byte-identically, no training
# ---------------------------------------------------------------------------


def test_predicted_risks_are_reproduced_unmodified():
    patients = (_patient("P1", predicted_risk=0.37), _patient("P2", predicted_risk=0.81))
    result = run_reuse_ablation("no_overlap_gate", _source(patients), CHECKPOINT_HASH)

    assert result.predicted_risks == {"P1": 0.37, "P2": 0.81}


@pytest.mark.parametrize(
    ("run_id", "forced_field"),
    [
        ("no_overlap_gate", "propensity_in_common_support"),
        ("no_uncertainty_gate", "ensemble_uncertainty_ok"),
        ("no_clinical_constraints", "protocol_valid"),
        ("witness_removed::clinical", "protocol_valid"),
        ("witness_removed::temporal", "no_hard_violation"),
    ]
)
def test_bypassing_the_named_condition_rescues_a_patient_blocked_only_by_it(run_id, forced_field):
    """Each run forces exactly its named condition; a patient failing
    *only* that condition becomes supported again, and the independently
    computed differing-decision count matches."""
    blocked_only_by_target = _patient("P1", **{forced_field: False})
    unaffected = _patient("P2")  # already supported; unaffected by any bypass

    source = _source((blocked_only_by_target, unaffected))
    result = run_reuse_ablation(run_id, source, CHECKPOINT_HASH)

    assert result.bypassed_supported["P1"] is True
    assert result.bypassed_supported["P2"] is True
    assert result.differing_patient_ids == ("P1",)


def test_witness_removed_causal_has_no_per_patient_effect():
    """The causal witness evaluates cohort-level conditions
    PatientSupportGate never reads, so this run changes zero patients."""
    patients = (
        _patient("P1", propensity_in_common_support=False),
        _patient("P2", protocol_valid=False),
        _patient("P3"),
    )
    result = run_reuse_ablation("witness_removed::causal", _source(patients), CHECKPOINT_HASH)

    assert result.forced_condition is None
    assert result.differing_patient_ids == ()
    # Bypassed decisions equal the original (unmodified) decisions exactly.
    assert result.bypassed_supported["P1"] is False
    assert result.bypassed_supported["P2"] is False
    assert result.bypassed_supported["P3"] is True


def test_a_patient_blocked_by_two_conditions_is_not_rescued_by_bypassing_only_one():
    patient = _patient("P1", propensity_in_common_support=False, protocol_valid=False)
    result = run_reuse_ablation("no_overlap_gate", _source((patient,)), CHECKPOINT_HASH)

    # Still blocked by protocol_valid=False even with overlap bypassed.
    assert result.bypassed_supported["P1"] is False
    assert result.differing_patient_ids == ()


def test_run_reuse_ablation_rejects_an_unknown_run_id():
    with pytest.raises(ValueError, match="not a reuse-mode run"):
        run_reuse_ablation("not_a_real_run", _source((_patient("P1"),)), CHECKPOINT_HASH)


def test_no_training_performed_result_is_pure_function_of_source():
    """Calling the same run twice on the same source yields byte-identical
    (structurally identical) results -- there is no stochastic element,
    confirming no training/randomness occurred."""
    patients = (_patient("P1"), _patient("P2", propensity_in_common_support=False))
    source = _source(patients)

    result_a = run_reuse_ablation("no_overlap_gate", source, CHECKPOINT_HASH)
    result_b = run_reuse_ablation("no_overlap_gate", source, CHECKPOINT_HASH)

    assert result_a.to_dict() == result_b.to_dict()


# ---------------------------------------------------------------------------
# run_all_reuse_ablations
# ---------------------------------------------------------------------------


def test_run_all_reuse_ablations_covers_every_run_id_by_default():
    patients = (_patient("P1"),)
    results = run_all_reuse_ablations(_source(patients), CHECKPOINT_HASH)

    assert set(results.keys()) == REUSE_RUN_IDS


# ---------------------------------------------------------------------------
# Checkpoint-hash verification (Requirement 7.15)
# ---------------------------------------------------------------------------


def test_run_reuse_ablation_rejects_a_mismatched_checkpoint_hash():
    source = _source((_patient("P1"),))
    with pytest.raises(ReusePredictionSourceError, match="does not match"):
        run_reuse_ablation("no_overlap_gate", source, expected_checkpoint_sha256="b" * 64)


def test_load_complete_model_predictions_rejects_a_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(ReusePredictionSourceError, match="missing"):
        load_complete_model_predictions(missing_path, CHECKPOINT_HASH)


def test_load_complete_model_predictions_rejects_an_unreadable_file(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ReusePredictionSourceError, match="unreadable"):
        load_complete_model_predictions(bad_path, CHECKPOINT_HASH)


def test_load_complete_model_predictions_rejects_a_hash_mismatch(tmp_path):
    source = _source((_patient("P1"),))
    path = source.write(tmp_path / "val_patient_predictions.json")

    with pytest.raises(ReusePredictionSourceError, match="mismatch"):
        load_complete_model_predictions(path, expected_checkpoint_sha256="c" * 64)


def test_load_complete_model_predictions_round_trips_through_write(tmp_path):
    original_patients = (
        _patient("P1", predicted_risk=0.22),
        _patient("P2", predicted_risk=0.91, protocol_valid=False),
    )
    source = _source(original_patients)
    path = source.write(tmp_path / "val_patient_predictions.json")

    loaded = load_complete_model_predictions(path, CHECKPOINT_HASH)

    assert loaded.checkpoint_sha256 == CHECKPOINT_HASH
    assert loaded.split_name == "val"
    assert {p.patient_id: p.predicted_risk for p in loaded.patients} == {
        "P1": 0.22,
        "P2": 0.91,
    }


def test_a_malformed_patient_record_raises_reuse_prediction_source_error(tmp_path):
    path = tmp_path / "val_patient_predictions.json"
    path.write_text(
        json.dumps(
            {
                "split_name": "val",
                "checkpoint_sha256": CHECKPOINT_HASH,
                "patients": [{"patient_id": "P1"}],  # missing required fields
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReusePredictionSourceError):
        load_complete_model_predictions(path, CHECKPOINT_HASH)


def test_a_mismatched_source_leaves_other_already_written_results_unaffected(tmp_path):
    """Requirement 7.15: a mismatched/missing source for one run writes no
    metrics for that run and leaves results already written for other
    runs unchanged. Simulated here by writing one run's result file
    first, then confirming a second run's failure does not touch it."""
    good_source = _source((_patient("P1"),))
    good_path = good_source.write(tmp_path / "val_patient_predictions.json")

    other_run_result_path = tmp_path / "run_no_uncertainty_gate_seed101.json"
    other_run_result_path.write_text(json.dumps({"already": "written"}), encoding="utf-8")
    original_bytes = other_run_result_path.read_bytes()

    loaded = load_complete_model_predictions(good_path, CHECKPOINT_HASH)
    with pytest.raises(ReusePredictionSourceError):
        run_reuse_ablation("no_overlap_gate", loaded, expected_checkpoint_sha256="wrong" * 10)

    assert other_run_result_path.read_bytes() == original_bytes
