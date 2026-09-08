"""Unit tests for `src/models/intervention_aware.py`.

Covers:

1. The capacity-matching check (Correctness Property 7; Requirement 9.8):
   swapping `InterventionEncoder` for `PlaceboInterventionEncoder` of the
   same configured width does not change `len(z_t)`.
2. `PlaceboInterventionEncoder` caching: same seed returns the same vector
   across multiple calls (never redrawn); different seed returns a
   different vector.
3. `GRUTransition.rollout()` shape correctness across a 3-timestep
   sequence (mirrors `tests/unit/test_sequence_baseline.py`'s pattern).

**Validates: Requirements 9.2, 9.3, 9.8; Correctness Property 7**
"""

from models.intervention_aware import (
    GRUTransition,
    InterventionEncoder,
    PlaceboInterventionEncoder,
    ProtocolEncoder,
    StateEncoder,
    concatenate_channels,
    intervention_features_from,
)

STATE_WIDTH = 5
INTERVENTION_WIDTH = 3
PROTOCOL_WIDTH = 2

BASELINE_COVARIATES = (350.0, 900.0, 80.0, 35.0)
PROTOCOL_FEATURES = (1.0, 1.0)


# ---------------------------------------------------------------------------
# Capacity-matching check (Correctness Property 7)
# ---------------------------------------------------------------------------


def test_placebo_channel_preserves_concatenation_width():
    """Disabling the intervention channel (swapping in
    `PlaceboInterventionEncoder`) must not shrink `z_t` -- `len(z_t)` is
    identical whether the real `InterventionEncoder` or the placebo is
    used, as long as both are configured to the same output width."""
    state_encoder = StateEncoder.initialize(output_size=STATE_WIDTH, seed=0)
    intervention_encoder = InterventionEncoder.initialize(output_size=INTERVENTION_WIDTH, seed=1)
    placebo_encoder = PlaceboInterventionEncoder.initialize(
        output_size=INTERVENTION_WIDTH, seed=2
    )
    protocol_encoder = ProtocolEncoder.initialize(output_size=PROTOCOL_WIDTH, seed=3)

    z_state = state_encoder.encode(BASELINE_COVARIATES)
    z_protocol = protocol_encoder.encode(PROTOCOL_FEATURES)

    intervention_features = intervention_features_from(treatment_arm=1.0)
    z_intervention_real = intervention_encoder.encode(intervention_features)
    z_intervention_placebo = placebo_encoder.encode(intervention_features)

    z_t_real = concatenate_channels(z_state, z_intervention_real, z_protocol)
    z_t_placebo = concatenate_channels(z_state, z_intervention_placebo, z_protocol)

    expected_width = STATE_WIDTH + INTERVENTION_WIDTH + PROTOCOL_WIDTH
    assert len(z_t_real) == expected_width
    assert len(z_t_placebo) == expected_width
    assert len(z_t_real) == len(z_t_placebo)

    # The placebo vector is a *different* vector from the real encoder's
    # output (it is disabling information content) but the same *width*.
    assert len(z_intervention_placebo) == len(z_intervention_real) == INTERVENTION_WIDTH


# ---------------------------------------------------------------------------
# PlaceboInterventionEncoder caching behavior
# ---------------------------------------------------------------------------


def test_placebo_encoder_returns_cached_vector_across_calls():
    """Same seed: every `encode()` call returns the identical vector
    (redrawn once at construction, never per call)."""
    placebo_encoder = PlaceboInterventionEncoder.initialize(output_size=INTERVENTION_WIDTH, seed=7)

    first_call = placebo_encoder.encode()
    second_call = placebo_encoder.encode([1.0, 2.0, 3.0])  # input ignored entirely
    third_call = placebo_encoder.encode()

    assert first_call == second_call == third_call


def test_placebo_encoder_returned_vector_is_not_the_cached_original():
    """`encode()` returns a copy, so callers mutating the returned list
    cannot corrupt the cached vector for subsequent calls."""
    placebo_encoder = PlaceboInterventionEncoder.initialize(output_size=INTERVENTION_WIDTH, seed=7)

    returned = placebo_encoder.encode()
    returned[0] = 999.0

    assert placebo_encoder.encode()[0] != 999.0


def test_placebo_encoder_different_seed_yields_different_vector():
    """A different seed draws a different fixed vector."""
    encoder_seed_a = PlaceboInterventionEncoder.initialize(output_size=INTERVENTION_WIDTH, seed=1)
    encoder_seed_b = PlaceboInterventionEncoder.initialize(output_size=INTERVENTION_WIDTH, seed=2)

    assert encoder_seed_a.encode() != encoder_seed_b.encode()


# ---------------------------------------------------------------------------
# GRUTransition.rollout() shape correctness
# ---------------------------------------------------------------------------


def test_gru_transition_rollout_output_shapes():
    hidden_size = 6
    state_encoder = StateEncoder.initialize(output_size=STATE_WIDTH, seed=0)
    intervention_encoder = InterventionEncoder.initialize(output_size=INTERVENTION_WIDTH, seed=1)
    protocol_encoder = ProtocolEncoder.initialize(output_size=PROTOCOL_WIDTH, seed=3)

    z_state = state_encoder.encode(BASELINE_COVARIATES)
    z_protocol = protocol_encoder.encode(PROTOCOL_FEATURES)
    intervention_features = intervention_features_from(treatment_arm=1.0)
    z_intervention = intervention_encoder.encode(intervention_features)

    z_t = concatenate_channels(z_state, z_intervention, z_protocol)
    input_size = len(z_t)

    # Baseline, week 20, week 96 -- the same z_t repeated (ITT: treatment
    # and baseline covariates are constant across landmarks in this test).
    z_sequence = [z_t, z_t, z_t]

    gru_transition = GRUTransition.initialize(
        input_size=input_size, hidden_size=hidden_size, seed=42
    )
    hidden_states = gru_transition.rollout(z_sequence)

    assert len(hidden_states) == len(z_sequence) == 3
    for h in hidden_states:
        assert len(h) == hidden_size
        assert all(isinstance(v, float) for v in h)


def test_gru_transition_rollout_defaults_to_zero_initial_hidden_state():
    """With no `initial_hidden_state` supplied, the rollout starts from a
    zero vector of width `hidden_size` -- confirmed indirectly by checking
    the rollout runs and produces the documented shape."""
    hidden_size = 4
    input_size = 3
    gru_transition = GRUTransition.initialize(
        input_size=input_size, hidden_size=hidden_size, seed=0
    )
    z_sequence = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    hidden_states = gru_transition.rollout(z_sequence)

    assert len(hidden_states) == 3
    assert all(len(h) == hidden_size for h in hidden_states)


def test_gru_transition_rollout_accepts_explicit_initial_hidden_state():
    hidden_size = 4
    input_size = 3
    gru_transition = GRUTransition.initialize(
        input_size=input_size, hidden_size=hidden_size, seed=0
    )
    z_sequence = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    zero_start = [0.0] * hidden_size
    default_start = gru_transition.rollout(z_sequence)
    explicit_zero_start = gru_transition.rollout(z_sequence, initial_hidden_state=zero_start)

    assert default_start == explicit_zero_start
