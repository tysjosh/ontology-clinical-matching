"""Unit tests for `src/models/sequence_baseline.py`.

Two checks (Requirement 8.2):

1. Shape correctness -- `SequenceGRU.forward()` on a synthetic 3-landmark
   patient sequence produces hidden states, next-state predictions, and
   hazard predictions of the documented shapes.
2. Training-loop smoke test -- `SequenceGRU.train_step()` run repeatedly on
   a small synthetic batch demonstrably reduces loss from the first
   iteration to the last (a basic sanity check that training does
   something, not a convergence guarantee).

**Validates: Requirements 8.2**
"""

from models.sequence_baseline import (
    INPUT_SIZE,
    N_LANDMARKS,
    STATE_DIM,
    SequenceGRU,
    SequenceTrainingSample,
)

HIDDEN_SIZE = 6
BASELINE_DIM = 4


def _make_sample(seed_offset: int = 0) -> SequenceTrainingSample:
    """One synthetic patient's 3-landmark training sample."""
    base = float(seed_offset)
    baseline_covariates = (350.0 + base, 900.0 + base, 80.0, 35.0)
    # [cd4, cd8, treatment_arm] at baseline, week 20, week 96.
    sequence_input = (
        (350.0 + base, 900.0 + base, 1.0),
        (330.0 + base, 880.0 + base, 1.0),
        (310.0 + base, 860.0 + base, 1.0),
    )
    target_next_states = (
        (330.0 + base, 880.0 + base),
        (310.0 + base, 860.0 + base),
        (0.0, 0.0),  # placeholder, excluded via mask
    )
    target_hazards = (0.0, 0.0, 1.0)
    mask = (True, True, False)
    return SequenceTrainingSample(
        baseline_covariates=baseline_covariates,
        sequence_input=sequence_input,
        target_next_states=target_next_states,
        target_hazards=target_hazards,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------


def test_forward_output_shapes():
    model = SequenceGRU.initialize(baseline_dim=BASELINE_DIM, hidden_size=HIDDEN_SIZE, seed=0)
    sample = _make_sample()

    hidden_states, next_state_predictions, hazard_predictions = model.forward(
        sample.baseline_covariates, sample.sequence_input
    )

    assert len(sample.sequence_input) == N_LANDMARKS
    assert len(sample.sequence_input[0]) == INPUT_SIZE

    # One hidden state per timestep, each of width `hidden_size`.
    assert len(hidden_states) == N_LANDMARKS
    for h in hidden_states:
        assert len(h) == HIDDEN_SIZE
        assert all(isinstance(v, float) for v in h)

    # One [cd4, cd8] prediction per timestep, width STATE_DIM.
    assert len(next_state_predictions) == N_LANDMARKS
    for prediction in next_state_predictions:
        assert len(prediction) == STATE_DIM

    # One scalar hazard probability per timestep, each in [0, 1].
    assert len(hazard_predictions) == N_LANDMARKS
    for hazard in hazard_predictions:
        assert 0.0 <= hazard <= 1.0


def test_forward_rejects_wrong_baseline_dim():
    model = SequenceGRU.initialize(baseline_dim=BASELINE_DIM, hidden_size=HIDDEN_SIZE, seed=0)
    sample = _make_sample()

    import pytest

    with pytest.raises(ValueError):
        model.forward(sample.baseline_covariates[:-1], sample.sequence_input)


def test_flat_parameter_round_trip_preserves_forward_output():
    """`load_flat_parameters(flat_parameters())` is a no-op on `forward()`
    output -- confirms the (de)serialization used by `train_step`'s
    numerical-gradient loop is shape/order-consistent."""
    model = SequenceGRU.initialize(baseline_dim=BASELINE_DIM, hidden_size=HIDDEN_SIZE, seed=1)
    sample = _make_sample()

    before = model.forward(sample.baseline_covariates, sample.sequence_input)
    model.load_flat_parameters(model.flat_parameters())
    after = model.forward(sample.baseline_covariates, sample.sequence_input)

    assert before == after


# ---------------------------------------------------------------------------
# Training-loop smoke test
# ---------------------------------------------------------------------------


def test_train_step_reduces_loss_over_iterations():
    model = SequenceGRU.initialize(baseline_dim=BASELINE_DIM, hidden_size=4, seed=42)
    batch = [_make_sample(seed_offset=0), _make_sample(seed_offset=5)]

    losses = []
    n_iterations = 8
    for _ in range(n_iterations):
        loss_before_step = model.train_step(batch, learning_rate=0.1)
        losses.append(loss_before_step)

    # Final loss (measured just before the last step) should be lower than
    # the very first iteration's loss -- a basic sanity check, not a
    # convergence guarantee.
    assert losses[-1] < losses[0]

    # Confirm the loss after all updates is also lower than the initial
    # loss, corroborating the per-iteration trend above.
    final_loss = model.loss(batch)
    assert final_loss < losses[0]
