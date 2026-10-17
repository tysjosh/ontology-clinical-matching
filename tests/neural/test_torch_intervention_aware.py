"""`models.torch_intervention_aware` as task 9.2 builds it (Requirement 3).

Scope note. This module verifies what task 9.2 implements, at hand-built
examples: the frozen structure and widths, the arm-specific hazard heads, the
hazard-derived 96-week outcome and its boundary convention, `eval()`-mode
determinism, autograd reachability, and every construction/argument error
path.

Deliberately left to task 9.3 (`tests/neural/test_intervention_channels.py`),
so this file does not pre-empt it:

- **Correctness Property 13** as a generated property over every
  `InterventionMode` x `ProtocolMode` combination. Covered here only as three
  single-point width checks (`LEARNED`/`ZEROED`/`PLACEBO` at
  `ProtocolMode.ACTIVE`, plus one `ProtocolMode.ZEROED` case), not the full
  cross product.
- **Correctness Property 14** over `hypothesis`-generated batches, including
  the ablated-batch half. Covered here only at one fixed cohort, as the
  object-identity assertions the structure has to make true.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from causal.baselines import HORIZON_WEEKS  # noqa: E402
from causal.propensity import COVARIATE_FIELDS  # noqa: E402
from models.dataset import (  # noqa: E402
    BIN_WIDTH_WEEKS,
    LANDMARK_COUNT,
    SURVIVAL_BIN_COUNT,
    DatasetDimensions,
    PatientDatasetBuilder,
    survival_bin_index,
)
from models.intervention_aware import (  # noqa: E402
    INTERVENTION_INPUT_DIM,
    PROTOCOL_INPUT_DIM,
)
from models.torch_intervention_aware import (  # noqa: E402
    HIDDEN_SIZE,
    OUTCOME_HORIZON_BIN_COUNT,
    SHARED_INPUT_FIELDS,
    STATE_PREDICTION_WIDTH,
    BranchOutput,
    InterventionMode,
    ProtocolMode,
    RolloutPair,
    TorchBranchOutput,
    TorchInterventionAwareModel,
    outcome_from_hazards,
)
from tests.neural.test_dataset_builder import _cohort  # noqa: E402

pytestmark = pytest.mark.neural

_PLACEBO = (0.25, -0.5, 0.75)


def _batch(n: int = 8):
    records = _cohort(n)
    builder = PatientDatasetBuilder().fit(records)
    return builder.build(records, "dev")


def _model(seed: int = 0, **kwargs) -> TorchInterventionAwareModel:
    """A seeded model on the builder's reported dimensions, in `eval()` mode.

    `PatientDatasetBuilder().dimensions` is readable before `fit`: every width
    is a frozen or imported constant, never a property of the data.
    """
    torch.manual_seed(seed)
    model = TorchInterventionAwareModel(PatientDatasetBuilder().dimensions, **kwargs)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Structure and frozen widths (Requirements 3.1, 3.2)
# ---------------------------------------------------------------------------


def test_module_owns_exactly_the_frozen_structure():
    """Requirement 3.1's five structural elements, and nothing else: three
    channel encoders, the GRU transition cell, two treatment-specific hazard
    heads, and one shared CD4/CD8 state head."""
    model = _model()

    assert sorted(dict(model.named_children())) == [
        "gru_cell",
        "hazard_head_arm0",
        "hazard_head_arm1",
        "intervention_encoder",
        "protocol_encoder",
        "state_encoder",
        "state_head",
    ]
    assert isinstance(model.gru_cell, nn.GRUCell)
    assert isinstance(model.hazard_head_arm0, nn.Linear)
    assert isinstance(model.hazard_head_arm1, nn.Linear)
    assert model.hazard_head_arm0 is not model.hazard_head_arm1
    assert model.state_head.out_features == STATE_PREDICTION_WIDTH


def test_widths_come_from_the_imported_frozen_constants():
    """Requirement 3.2: channel widths and hidden width are the imported
    `models.intervention_aware` / `decisions.md` Section 25 values, not local
    literals."""
    model = _model()

    assert HIDDEN_SIZE == 16
    assert model.hidden_size == HIDDEN_SIZE
    assert model.gru_cell.hidden_size == HIDDEN_SIZE
    # The state channel reads the Section 13 adjustment set (decisions.md
    # Section 26 amendment of 2026-10-16), so its width is that block's.
    assert model.state_encoder.in_features == len(COVARIATE_FIELDS)
    assert model.intervention_encoder.in_features == INTERVENTION_INPUT_DIM
    assert model.protocol_encoder.in_features == PROTOCOL_INPUT_DIM
    assert model.state_channel_width == len(COVARIATE_FIELDS)
    assert model.intervention_channel_width == INTERVENTION_INPUT_DIM
    assert model.protocol_channel_width == PROTOCOL_INPUT_DIM
    assert model.representation_width == (
        len(COVARIATE_FIELDS) + INTERVENTION_INPUT_DIM + PROTOCOL_INPUT_DIM
    )
    assert model.gru_cell.input_size == model.representation_width
    assert model.hazard_head_arm0.out_features == SURVIVAL_BIN_COUNT
    assert model.hazard_head_arm1.out_features == SURVIVAL_BIN_COUNT


def test_no_dropout_or_batchnorm_anywhere():
    """The determinism guarantee is structural, as in `TorchSequenceGRU`."""
    model = _model()
    for module in model.modules():
        assert not isinstance(module, nn.Dropout | nn.BatchNorm1d)


def test_design_section_3_3_branch_output_name_is_the_same_class():
    assert BranchOutput is TorchBranchOutput


# ---------------------------------------------------------------------------
# Capacity-matched channel widths, single-point (Requirement 3.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intervention_mode", "placebo_vector"),
    [
        (InterventionMode.LEARNED, None),
        (InterventionMode.ZEROED, None),
        (InterventionMode.PLACEBO, _PLACEBO),
    ],
)
def test_representation_width_is_mode_independent(intervention_mode, placebo_vector):
    """A zeroed or placebo channel occupies the same width as the learned one
    -- ablations change channel content, never capacity. Task 9.3 quantifies
    this over the full mode cross product (Property 13)."""
    batch = _batch()
    complete = _model()
    ablated = _model(intervention_mode=intervention_mode, placebo_vector=placebo_vector)

    assert ablated.representation_width == complete.representation_width
    assert (
        ablated.encode_representation(batch, 1).shape
        == complete.encode_representation(batch, 1).shape
    )
    # Same parameter shapes too: the encoder is not narrowed or dropped.
    assert [tuple(p.shape) for p in ablated.parameters()] == [
        tuple(p.shape) for p in complete.parameters()
    ]


def test_zeroed_protocol_mode_keeps_the_protocol_width():
    batch = _batch()
    model = _model(protocol_mode=ProtocolMode.ZEROED)

    shared = model.shared_inputs(batch)
    assert shared.protocol.shape == batch.protocol.shape
    assert torch.equal(shared.protocol, torch.zeros_like(batch.protocol))
    assert model.representation_width == _model().representation_width


# ---------------------------------------------------------------------------
# Forward output shapes and determinism (Requirements 3.1, 3.4)
# ---------------------------------------------------------------------------


def test_forward_output_shapes():
    batch = _batch(7)
    model = _model()
    output = model(batch, 1)

    n_rows = len(batch.patient_ids)
    assert output.arm == 1
    assert output.representation.shape == (n_rows, model.representation_width)
    assert output.hidden_states.shape == (n_rows, LANDMARK_COUNT, HIDDEN_SIZE)
    assert output.state_pred.shape == (n_rows, LANDMARK_COUNT, STATE_PREDICTION_WIDTH)
    assert output.hazards.shape == (n_rows, SURVIVAL_BIN_COUNT)
    assert output.hazard_logits.shape == (n_rows, SURVIVAL_BIN_COUNT)
    assert output.outcome_96w.shape == (n_rows,)
    assert output.factual_mask.shape == (n_rows,)
    assert output.factual_mask.dtype is torch.bool
    assert torch.all((output.hazards > 0.0) & (output.hazards < 1.0))
    assert torch.isfinite(output.outcome_96w).all()


def test_eval_mode_forward_is_bitwise_reproducible():
    batch = _batch()
    model = _model()

    first = model(batch, 0)
    second = model(batch, 0)

    for field in ("representation", "hidden_states", "state_pred", "hazards", "outcome_96w"):
        assert torch.equal(getattr(first, field), getattr(second, field))


def test_factual_mask_marks_exactly_the_patients_observed_on_that_arm():
    batch = _batch()
    model = _model()
    arm_0, arm_1 = model.rollout_both_arms(batch)

    assert torch.equal(arm_1.factual_mask, batch.treatment_arm == 1.0)
    assert torch.equal(arm_0.factual_mask, batch.treatment_arm == 0.0)
    # Exactly one of the two rollouts is factual for each patient.
    assert torch.all(arm_0.factual_mask ^ arm_1.factual_mask)


def test_hazard_heads_are_treatment_specific():
    """The arm selects the hazard head, so only that head receives gradient
    from its own rollout (Requirement 3.1's "treatment-specific hazard
    heads")."""
    batch = _batch()
    model = _model()

    model(batch, 1).outcome_96w.sum().backward()

    assert model.hazard_head_arm1.weight.grad is not None
    assert model.hazard_head_arm0.weight.grad is None
    # Autograd reaches the encoders and the recurrence too.
    assert model.intervention_encoder.weight.grad is not None
    assert model.state_encoder.weight.grad is not None
    assert model.gru_cell.weight_ih.grad is not None


# ---------------------------------------------------------------------------
# Counterfactual rollout: identity of every non-intervention input (3.4, 3.6)
# ---------------------------------------------------------------------------


def test_rollout_both_arms_returns_both_arms():
    batch = _batch()
    model = _model()
    pair = model.rollout_both_arms(batch)

    assert isinstance(pair, RolloutPair)
    assert isinstance(pair, tuple)
    arm_0, arm_1 = pair
    assert (arm_0.arm, arm_1.arm) == (0, 1)
    assert arm_0 is pair.arm_0 and arm_1 is pair.arm_1


def test_rollout_both_arms_shares_every_non_intervention_input_by_identity():
    """Requirement 3.6 by object identity, not by value: the two rollouts get
    literally the same shared-input object and the same tensor in every field
    of it, and differ only in the intervention-channel input. Task 9.3
    quantifies this over generated batches (Property 14)."""
    batch = _batch()
    model = _model()
    arm_0, arm_1 = model.rollout_both_arms(batch)

    assert arm_0.shared_inputs is arm_1.shared_inputs
    for field in SHARED_INPUT_FIELDS:
        assert getattr(arm_0.shared_inputs, field) is getattr(arm_1.shared_inputs, field)
    assert arm_0.intervention_input is not arm_1.intervention_input
    assert not torch.equal(arm_0.intervention_input, arm_1.intervention_input)


def test_shared_inputs_pass_the_batch_tensors_through_untouched():
    batch = _batch()
    model = _model()
    shared = model.shared_inputs(batch)

    assert shared.baseline is batch.baseline
    assert shared.state is batch.baseline  # the state channel reads the adjustment set
    assert shared.protocol is batch.protocol
    assert shared.cd4_mask is batch.cd4_mask
    assert shared.cd8_mask is batch.cd8_mask


def test_learned_intervention_input_differs_only_in_the_treatment_column():
    batch = _batch()
    model = _model()
    before = batch.intervention.clone()

    arm_0_input = model.intervention_input(batch, 0)
    arm_1_input = model.intervention_input(batch, 1)

    assert torch.equal(arm_0_input[:, 0], torch.zeros_like(arm_0_input[:, 0]))
    assert torch.equal(arm_1_input[:, 0], torch.ones_like(arm_1_input[:, 0]))
    assert torch.equal(arm_0_input[:, 1:], batch.intervention[:, 1:])
    assert torch.equal(arm_1_input[:, 1:], batch.intervention[:, 1:])
    # The batch is never mutated in place.
    assert torch.equal(batch.intervention, before)


def test_zeroed_intervention_channel_makes_the_rolled_out_state_arm_independent():
    """What the ablation removes, and what it does not.

    With a zeroed channel the two arms' representations and hidden states are
    identical -- treatment no longer reaches the state the GRU rolls forward.
    The hazards still differ, because the arm also selects which
    treatment-specific hazard head reads that state, and those heads are
    structural (`decisions.md` Section 14.3), not part of what this mode
    ablates. The two channel tensors remain distinct objects of the same
    width."""
    batch = _batch()
    model = _model(intervention_mode=InterventionMode.ZEROED)
    arm_0, arm_1 = model.rollout_both_arms(batch)

    assert arm_0.intervention_input is not arm_1.intervention_input
    assert torch.equal(arm_0.intervention_input, arm_1.intervention_input)
    assert torch.equal(arm_0.representation, arm_1.representation)
    assert torch.equal(arm_0.hidden_states, arm_1.hidden_states)
    assert torch.equal(arm_0.state_pred, arm_1.state_pred)
    assert not torch.equal(arm_0.hazards, arm_1.hazards)


def test_learned_intervention_channel_moves_the_rolled_out_state():
    """The contrast to the ablation above: with the real channel, the arm
    changes the representation and hence the state trajectory itself."""
    batch = _batch()
    model = _model()
    arm_0, arm_1 = model.rollout_both_arms(batch)

    assert not torch.equal(arm_0.representation, arm_1.representation)
    assert not torch.equal(arm_0.hidden_states, arm_1.hidden_states)


def test_placebo_channel_is_the_frozen_vector_for_every_patient():
    batch = _batch()
    model = _model(intervention_mode=InterventionMode.PLACEBO, placebo_vector=_PLACEBO)

    channel = model.intervention_input(batch, 1)

    assert channel.shape == (len(batch.patient_ids), INTERVENTION_INPUT_DIM)
    expected = torch.tensor(_PLACEBO, dtype=channel.dtype)
    for row in channel:
        assert torch.equal(row, expected)
    # Frozen, not learned, and preserved in the checkpoint.
    assert "placebo_vector" in model.state_dict()
    assert all(name != "placebo_vector" for name, _ in model.named_parameters())


# ---------------------------------------------------------------------------
# The hazard-derived 96-week outcome (Requirement 3.5)
# ---------------------------------------------------------------------------


def test_outcome_bin_count_is_derived_from_the_frozen_bin_definition():
    """`OUTCOME_HORIZON_BIN_COUNT` is `survival_bin_index(96.0)`, so the
    accumulated bins are exactly those spanning `[0, 96)`; the bin that
    *starts* at week 96 is excluded (see the module's boundary note)."""
    assert OUTCOME_HORIZON_BIN_COUNT == survival_bin_index(HORIZON_WEEKS)
    assert OUTCOME_HORIZON_BIN_COUNT * BIN_WIDTH_WEEKS == HORIZON_WEEKS
    # Last included bin ends at week 96; the first excluded bin starts there.
    assert survival_bin_index(HORIZON_WEEKS - 1e-3) == OUTCOME_HORIZON_BIN_COUNT - 1
    assert survival_bin_index(HORIZON_WEEKS) == OUTCOME_HORIZON_BIN_COUNT
    assert OUTCOME_HORIZON_BIN_COUNT < SURVIVAL_BIN_COUNT


def test_outcome_is_one_minus_the_product_of_survival_probabilities():
    hazards = torch.rand(5, SURVIVAL_BIN_COUNT) * 0.4 + 0.05

    derived = outcome_from_hazards(hazards)

    expected = 1.0 - torch.prod(1.0 - hazards[:, :OUTCOME_HORIZON_BIN_COUNT], dim=1)
    assert torch.allclose(derived, expected)
    # Bins beyond the horizon cannot move the 96-week outcome.
    perturbed = hazards.clone()
    perturbed[:, OUTCOME_HORIZON_BIN_COUNT:] = 0.9
    assert torch.allclose(outcome_from_hazards(perturbed), derived)
    assert torch.all((derived >= 0.0) & (derived <= 1.0))


def test_model_outcome_equals_the_derivation_from_its_own_hazards():
    batch = _batch()
    model = _model()
    output = model(batch, 1)

    assert torch.equal(output.outcome_96w, outcome_from_hazards(output.hazards))


def test_outcome_edge_cases_at_zero_and_certain_hazards():
    zero_hazards = torch.zeros(2, SURVIVAL_BIN_COUNT)
    assert torch.equal(outcome_from_hazards(zero_hazards), torch.zeros(2))

    certain = torch.zeros(2, SURVIVAL_BIN_COUNT)
    certain[:, 0] = 1.0
    assert torch.equal(outcome_from_hazards(certain), torch.ones(2))


def test_outcome_from_hazards_rejects_bad_shapes_and_bin_counts():
    with pytest.raises(ValueError, match="2-D"):
        outcome_from_hazards(torch.zeros(SURVIVAL_BIN_COUNT))
    with pytest.raises(ValueError, match="n_bins"):
        outcome_from_hazards(torch.zeros(2, SURVIVAL_BIN_COUNT), n_bins=0)
    with pytest.raises(ValueError, match="n_bins"):
        outcome_from_hazards(torch.zeros(2, SURVIVAL_BIN_COUNT), n_bins=SURVIVAL_BIN_COUNT + 1)


def test_no_submodule_may_produce_an_outcome_independently_of_the_hazards():
    """Requirement 3.5 structurally: the only outcome route is
    `outcome_from_hazards`, and adding a head that bypasses it fails at the
    structural check rather than silently producing a second answer."""
    model = _model()
    # No submodule maps the hidden state to a single scalar outcome.
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            assert module.out_features in (
                len(COVARIATE_FIELDS),
                INTERVENTION_INPUT_DIM,
                PROTOCOL_INPUT_DIM,
                STATE_PREDICTION_WIDTH,
                SURVIVAL_BIN_COUNT,
            ), f"{name} produces an unexpected output width"

    model.outcome_head = nn.Linear(HIDDEN_SIZE, 1)
    with pytest.raises(AssertionError, match="Requirement 3.5"):
        model._assert_no_outcome_head()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_forward_rejects_an_arm_outside_zero_and_one():
    batch = _batch()
    model = _model()
    with pytest.raises(ValueError, match="arm must be 0 or 1"):
        model(batch, 2)


def test_placebo_mode_requires_a_correctly_sized_vector():
    dimensions = PatientDatasetBuilder().dimensions

    with pytest.raises(ValueError, match="requires placebo_vector"):
        TorchInterventionAwareModel(dimensions, intervention_mode=InterventionMode.PLACEBO)
    with pytest.raises(ValueError, match="must match the intervention channel"):
        TorchInterventionAwareModel(
            dimensions,
            intervention_mode=InterventionMode.PLACEBO,
            placebo_vector=(0.1, 0.2),
        )
    with pytest.raises(ValueError, match="silently ignored"):
        TorchInterventionAwareModel(dimensions, placebo_vector=_PLACEBO)


def test_construction_rejects_dimensions_that_disagree_with_the_frozen_widths():
    dimensions = PatientDatasetBuilder().dimensions
    drifted = DatasetDimensions(
        baseline_covariate_width=dimensions.baseline_covariate_width,
        state_width=dimensions.state_width + 1,
        intervention_width=dimensions.intervention_width,
        protocol_width=dimensions.protocol_width,
        landmark_count=dimensions.landmark_count,
        survival_bin_count=dimensions.survival_bin_count,
    )

    with pytest.raises(ValueError, match="state_width"):
        TorchInterventionAwareModel(drifted)


def test_forward_rejects_a_batch_whose_widths_disagree_with_the_model():
    batch = _batch()
    model = _model()
    narrowed = torch.zeros(len(batch.patient_ids), PROTOCOL_INPUT_DIM - 1)
    object.__setattr__(batch, "protocol", narrowed)

    with pytest.raises(ValueError, match="batch.protocol"):
        model(batch, 1)


def test_every_adjustment_set_column_reaches_the_counterfactual_rollout():
    """decisions.md Section 26 amendment of 2026-10-16: the state channel
    reads the full Section 13 adjustment set, so each column, perturbed
    alone, moves both arms' hazards -- no confounder is silently dropped."""
    import dataclasses

    batch = _batch()
    model = _model()
    with torch.no_grad():
        reference = model.rollout_both_arms(batch)
        for column in range(batch.baseline.shape[1]):
            baseline = batch.baseline.clone()
            baseline[:, column] += 1.0
            perturbed = model.rollout_both_arms(dataclasses.replace(batch, baseline=baseline))
            for arm in ("arm_0", "arm_1"):
                assert not torch.equal(
                    getattr(perturbed, arm).hazard_logits, getattr(reference, arm).hazard_logits
                ), f"{COVARIATE_FIELDS[column]} does not reach {arm}"
