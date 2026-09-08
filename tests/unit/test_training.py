"""Unit tests for `src/models/training.py` (Requirement 8.3).

The explicitly required test is the checkpoint save/load round-trip
(confirming exact, not approximate, parameter recovery). The remaining
tests are smoke tests for `EarlyStopping`, `compute_class_weights`, and
the censoring-aware loss function.

**Validates: Requirements 8.3**
"""

from __future__ import annotations

import math

import pytest

from models.sequence_baseline import SequenceGRU
from models.training import (
    EarlyStopping,
    censoring_aware_loss,
    compute_class_weights,
    derive_training_seed,
    discrete_time_censored_nll,
    load_checkpoint,
    make_checkpoint_metadata,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Checkpoint round-trip (explicitly required test)
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_exact_with_synthetic_params(tmp_path):
    """Save/load a synthetic flat-list-of-floats checkpoint and confirm the
    loaded parameters exactly (bit-for-bit) match the originals, and that
    metadata round-trips correctly."""
    original_params = [0.1, -2.5, 3.333333333333333, 0.0, 1e-9, -1e9]
    metadata = {"epoch": 7, "seed": 12345, "note": "synthetic"}
    checkpoint_path = tmp_path / "checkpoint.json"

    save_checkpoint(original_params, checkpoint_path, metadata)
    loaded_params, loaded_metadata = load_checkpoint(checkpoint_path)

    assert loaded_params == original_params
    for original, loaded in zip(original_params, loaded_params, strict=True):
        assert original == loaded  # exact equality, not approximate
    assert loaded_metadata == metadata


def test_checkpoint_round_trip_with_sequence_gru_flat_parameters(tmp_path):
    """Save/load `SequenceGRU`'s actual flat parameter vector and confirm
    it round-trips exactly, then confirm loading it back into a fresh
    model reproduces identical forward-pass output."""
    model = SequenceGRU.initialize(baseline_dim=4, hidden_size=5, seed=3)
    original_params = model.flat_parameters()
    checkpoint_path = tmp_path / "sequence_gru_checkpoint.json"

    save_checkpoint(original_params, checkpoint_path, metadata={"epoch": 1, "seed": 3})
    loaded_params, loaded_metadata = load_checkpoint(checkpoint_path)

    assert loaded_params == original_params
    assert loaded_metadata == {"epoch": 1, "seed": 3}

    baseline_covariates = (350.0, 900.0, 80.0, 35.0)
    sequence_input = ((350.0, 900.0, 1.0), (330.0, 880.0, 1.0), (310.0, 860.0, 1.0))
    before = model.forward(baseline_covariates, sequence_input)

    restored_model = SequenceGRU.initialize(baseline_dim=4, hidden_size=5, seed=999)
    restored_model.load_flat_parameters(loaded_params)
    after = restored_model.forward(baseline_covariates, sequence_input)

    assert before == after


def test_checkpoint_defaults_metadata_to_empty_dict(tmp_path):
    checkpoint_path = tmp_path / "no_metadata.json"
    save_checkpoint([1.0, 2.0], checkpoint_path)
    _, metadata = load_checkpoint(checkpoint_path)
    assert metadata == {}


def test_checkpoint_metadata_includes_derived_seed(tmp_path):
    checkpoint_path = tmp_path / "with_seed_metadata.json"
    metadata = make_checkpoint_metadata(epoch=2, master_seed=42)

    save_checkpoint([0.0], checkpoint_path, metadata)
    _, loaded_metadata = load_checkpoint(checkpoint_path)

    assert loaded_metadata["epoch"] == 2
    assert loaded_metadata["master_seed"] == 42
    assert loaded_metadata["seed"] == derive_training_seed(42)
    assert "timestamp" in loaded_metadata


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


def test_early_stopping_does_not_stop_while_improving():
    stopper = EarlyStopping(patience=3)
    for loss in (1.0, 0.8, 0.6, 0.4, 0.2):
        assert stopper.should_stop(loss) is False


def test_early_stopping_signals_stop_after_patience_non_improving_epochs():
    stopper = EarlyStopping(patience=2)
    assert stopper.should_stop(1.0) is False  # improvement (from inf)
    assert stopper.should_stop(1.0) is False  # non-improving epoch 1
    assert stopper.should_stop(1.0) is True  # non-improving epoch 2 -> stop


def test_early_stopping_resets_counter_on_improvement():
    stopper = EarlyStopping(patience=2)
    assert stopper.should_stop(1.0) is False
    assert stopper.should_stop(1.0) is False  # 1 non-improving epoch
    assert stopper.should_stop(0.5) is False  # improvement resets counter
    assert stopper.should_stop(0.5) is False  # 1 non-improving epoch
    assert stopper.should_stop(0.5) is True  # 2 non-improving epochs -> stop


def test_early_stopping_rejects_non_positive_patience():
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


def test_compute_class_weights_hand_checked_example():
    # 8 samples: 6 of class 0, 2 of class 1. Balanced weight formula:
    # weight[c] = n_samples / (n_classes * count[c])
    labels = [0, 0, 0, 0, 0, 0, 1, 1]
    weights = compute_class_weights(labels)

    assert weights[0] == pytest.approx(8 / (2 * 6))
    assert weights[1] == pytest.approx(8 / (2 * 2))


def test_compute_class_weights_balanced_classes_all_weight_one():
    labels = [0, 0, 1, 1]
    weights = compute_class_weights(labels)
    assert weights == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}


def test_compute_class_weights_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_class_weights([])


# ---------------------------------------------------------------------------
# Censoring-aware loss
# ---------------------------------------------------------------------------


def test_discrete_time_censored_nll_event_case_finite_nonnegative():
    hazards = [0.1, 0.2, 0.6]
    loss = discrete_time_censored_nll(hazards, event_time_index=2)
    assert math.isfinite(loss)
    assert loss >= 0.0


def test_discrete_time_censored_nll_censored_case_finite_nonnegative():
    hazards = [0.1, 0.2, 0.15]
    loss = discrete_time_censored_nll(hazards, event_time_index=None)
    assert math.isfinite(loss)
    assert loss >= 0.0


def test_discrete_time_censored_nll_rejects_out_of_range_event_index():
    with pytest.raises(ValueError):
        discrete_time_censored_nll([0.1, 0.2], event_time_index=5)


def test_censoring_aware_loss_runs_on_synthetic_batch_and_is_finite_nonnegative():
    batch_hazards = [[0.1, 0.2, 0.6], [0.05, 0.1, 0.1]]
    batch_event_time_indices = [2, None]

    loss = censoring_aware_loss(batch_hazards, batch_event_time_indices)

    assert math.isfinite(loss)
    assert loss >= 0.0


def test_censoring_aware_loss_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        censoring_aware_loss([[0.1]], [])
