"""Correctness Properties 13 and 14 for `models.torch_intervention_aware`
(task 9.3).

Generated counterpart to `tests/neural/test_torch_intervention_aware.py`.
That module pins task 9.2's structure at one hand-built 8-patient cohort;
this one quantifies the two named properties over the **full**
`InterventionMode` x `ProtocolMode` cross product (six combinations,
enumerated by `itertools.product` so a seventh cannot be silently omitted)
and over `hypothesis`-generated cohorts:

- **Property 13** -- capacity-matched channel width invariance. Every mode
  combination yields the same per-channel widths, the same concatenated
  representation width `len(COVARIATE_FIELDS) + INTERVENTION_INPUT_DIM +
  PROTOCOL_INPUT_DIM` (14 + 3 + 2), and the same parameter shapes as the
  complete model, so an ablation changes channel *content* and never
  capacity.
- **Property 14** -- both halves. (a) Within one `rollout_both_arms` call the
  two arms receive the same `SharedRolloutInputs` object and the same tensor
  in every one of its fields, and the intervention-channel input is the only
  tensor that differs. (b) Across models, an ablated model's rollout inputs
  differ from the complete model's in exactly the named channel -- zeroed
  protocol, or zeroed/placebo intervention -- and in no other tensor.

Three things this module deliberately does not assert, each a property of
the frozen architecture rather than of the test:

- **A zeroed or placebo channel does not make the two arms' hazards equal.**
  It makes their representation, hidden states, and state predictions
  identical, which is what the ablation is for; the arm still selects which
  treatment-specific hazard head reads the hidden state, and those heads are
  structural (`decisions.md` Section 14.3). Asserting equal hazards would be
  asserting the ablation removes a structural element it does not touch.
- **`InterventionMode.ZEROED` does not always change the channel's
  *values*.** ACTG175 records neither dose nor adherence, so
  `models.dataset._intervention_features` emits
  `[treatment_arm, 0.0, 0.0]`, and the learned arm-0 channel is therefore
  already all zeros. The ablated channel is arm-independent while the learned
  channel differs between arms in column 0, so an ablated channel can
  coincide with at most one of the two arms -- which is the form the
  "differs from the complete batch" half is asserted in.
- **Equality of the non-ablated shared tensors is asserted by object
  identity, not by value**, in both halves: they pass through from the
  `TensorBatch` untouched, so identity is the available and stronger claim
  (`design.md` Section 3.3).

`_make_patient_record` and `_cohort` are imported from
`tests/neural/test_dataset_builder.py` rather than restated, so generated
records follow the one definition of the loader's landmark-placement
convention.

**Validates: Requirements 3.3, 3.4, 3.6**
"""

from __future__ import annotations

import dataclasses
from itertools import product

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from causal.propensity import COVARIATE_FIELDS  # noqa: E402
from cohorts.cohort_construction import PatientRecord  # noqa: E402
from models.dataset import (  # noqa: E402
    LANDMARK_COUNT,
    PatientDatasetBuilder,
    TensorBatch,
)
from models.intervention_aware import (  # noqa: E402
    INTERVENTION_INPUT_DIM,
    PROTOCOL_INPUT_DIM,
)
from models.torch_intervention_aware import (  # noqa: E402
    SHARED_INPUT_FIELDS,
    InterventionMode,
    ProtocolMode,
    TorchInterventionAwareModel,
)
from tests.neural.test_dataset_builder import _cohort, _make_patient_record  # noqa: E402

pytestmark = pytest.mark.neural

#: The complete model: the combination every ablation is compared against.
_COMPLETE_MODES: tuple[InterventionMode, ProtocolMode] = (
    InterventionMode.LEARNED,
    ProtocolMode.ACTIVE,
)

#: Property 13 and 14(b) quantify over *every* mode combination, so the
#: parameterization is the cross product itself rather than a hand-listed
#: subset. `test_the_mode_cross_product_is_exhaustive` pins the count.
_MODE_COMBINATIONS: tuple[tuple[InterventionMode, ProtocolMode], ...] = tuple(
    product(InterventionMode, ProtocolMode)
)
_MODE_IDS: tuple[str, ...] = tuple(
    f"{intervention.name}-{protocol.name}" for intervention, protocol in _MODE_COMBINATIONS
)

#: The state channel's width: the adjustment set's encoded width
#: (`len(COVARIATE_FIELDS)`, 14), since the `decisions.md` Section 26
#: amendment of 2026-10-16 made the state channel read `TensorBatch.baseline`.
_STATE_CHANNEL_WIDTH: int = len(COVARIATE_FIELDS)

#: The representation width, written as the sum of the channel widths
#: rather than as a literal (14 + 3 + 2).
_REPRESENTATION_WIDTH: int = _STATE_CHANNEL_WIDTH + INTERVENTION_INPUT_DIM + PROTOCOL_INPUT_DIM

#: Cohort properties build up to six records of ~14 events each, fit the
#: builder, and run four rollouts per example, across six parameterizations.
#: Modest budget, `deadline=None`, per the suite's convention.
_COHORT_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

#: A landmark measurement, or `None` for a missing one.
_landmark_value = st.one_of(
    st.none(),
    st.floats(min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
)

#: The frozen placebo channel is "a fixed random vector matching the
#: intervention channel's dimensionality" (`decisions.md` Section 14.3), so
#: it is generated rather than fixed -- including the values that collide
#: with a learned arm's channel, which the module docstring covers.
_placebo_vectors = st.tuples(
    *(
        st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
        for _ in range(INTERVENTION_INPUT_DIM)
    )
)


@st.composite
def _cohorts(draw, min_size: int = 1, max_size: int = 6) -> list[PatientRecord]:
    """1-6 unique `PatientRecord`s with randomized values and missingness.

    Deliberately smaller than `tests/neural/test_dataset.py`'s 1-50 cohorts:
    nothing these properties assert scales with cohort size beyond the row
    count of the emitted tensors, and each example here also constructs two
    models and runs four rollouts.

    Identifiers are a permutation of `P0..P{size-1}`, so uniqueness is
    structural rather than enforced by a filter that discards examples.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    patient_ids = draw(st.permutations([f"P{number}" for number in range(size)]))
    return [
        _make_patient_record(
            patient_id,
            treatment_arm=draw(st.sampled_from((0, 1))),
            baseline_cd4=draw(st.floats(min_value=1.0, max_value=1500.0)),
            baseline_cd8=draw(st.floats(min_value=1.0, max_value=3000.0)),
            karnofsky_score=draw(st.floats(min_value=0.0, max_value=100.0)),
            age=draw(st.floats(min_value=18.0, max_value=90.0)),
            event_time_weeks=draw(
                st.floats(min_value=0.0, max_value=260.0, allow_nan=False, allow_infinity=False)
            ),
            event_indicator=draw(st.booleans()),
            cd4_by_landmark=tuple(
                draw(st.tuples(*(_landmark_value for _ in range(LANDMARK_COUNT))))
            ),
            cd8_by_landmark=tuple(
                draw(st.tuples(*(_landmark_value for _ in range(LANDMARK_COUNT))))
            ),
            eligible=draw(st.booleans()),
        )
        for patient_id in patient_ids
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch(records: list[PatientRecord]) -> TensorBatch:
    builder = PatientDatasetBuilder().fit(records)
    return builder.build(records, "dev")


def _build_model(
    intervention_mode: InterventionMode,
    protocol_mode: ProtocolMode,
    *,
    placebo_vector: tuple[float, ...] = (0.25, -0.5, 0.75),
    seed: int = 0,
) -> TorchInterventionAwareModel:
    """A seeded `eval()`-mode model for one mode combination.

    `placebo_vector` is forwarded only under `InterventionMode.PLACEBO`,
    which is what makes a single parameterization over the cross product
    possible: the constructor requires the vector in that mode and rejects it
    in the other two, deliberately, so a supplied vector is never silently
    ignored. Both models a comparison builds are seeded identically, and
    parameter shapes are mode-independent (Property 13), so their weights are
    equal and any observed difference can only come from channel content.
    """
    torch.manual_seed(seed)
    model = TorchInterventionAwareModel(
        PatientDatasetBuilder().dimensions,
        intervention_mode=intervention_mode,
        protocol_mode=protocol_mode,
        placebo_vector=(
            placebo_vector if intervention_mode is InterventionMode.PLACEBO else None
        ),
    )
    model.eval()
    return model


def _named_channels(
    intervention_mode: InterventionMode, protocol_mode: ProtocolMode
) -> frozenset[str]:
    """The channels a mode combination ablates -- Property 14(b)'s "named
    channel". Empty for the complete model."""
    named = set()
    if intervention_mode is not InterventionMode.LEARNED:
        named.add("intervention")
    if protocol_mode is not ProtocolMode.ACTIVE:
        named.add("protocol")
    return frozenset(named)


def _expected_intervention_input(
    batch: TensorBatch,
    intervention_mode: InterventionMode,
    arm: int,
    placebo_vector: tuple[float, ...],
) -> torch.Tensor:
    """The intervention channel each mode is *defined* to carry.

    Written from the batch and the mode definitions of `decisions.md` Section
    14.3, never by calling `model.intervention_input`, so "differs in exactly
    the named channel" is checked against the specified content rather than
    against the implementation's own answer.
    """
    n_rows = batch.intervention.shape[0]
    dtype = batch.intervention.dtype
    if intervention_mode is InterventionMode.ZEROED:
        return torch.zeros_like(batch.intervention)
    if intervention_mode is InterventionMode.PLACEBO:
        row = torch.tensor(placebo_vector, dtype=dtype)
        return row.unsqueeze(0).expand(n_rows, -1).clone()
    arm_column = torch.full((n_rows, 1), float(arm), dtype=dtype)
    return torch.cat([arm_column, batch.intervention[:, 1:]], dim=1)


def _expected_protocol(batch: TensorBatch, protocol_mode: ProtocolMode) -> torch.Tensor:
    """The protocol channel each mode is defined to carry."""
    if protocol_mode is ProtocolMode.ACTIVE:
        return batch.protocol
    return torch.zeros_like(batch.protocol)


# ---------------------------------------------------------------------------
# The parameterization itself
# ---------------------------------------------------------------------------


def test_the_mode_cross_product_is_exhaustive():
    """Properties 13 and 14(b) say "for any combination", so the count is
    pinned: adding a fourth `InterventionMode` or a third `ProtocolMode`
    fails here, which is the prompt to check that
    `_expected_intervention_input` and `_named_channels` describe the new
    mode before the parameterization silently starts covering it."""
    assert set(InterventionMode) == {
        InterventionMode.LEARNED,
        InterventionMode.ZEROED,
        InterventionMode.PLACEBO,
    }
    assert set(ProtocolMode) == {ProtocolMode.ACTIVE, ProtocolMode.ZEROED}
    assert len(_MODE_COMBINATIONS) == len(InterventionMode) * len(ProtocolMode) == 6
    assert _COMPLETE_MODES in _MODE_COMBINATIONS
    assert _named_channels(*_COMPLETE_MODES) == frozenset()


def test_shared_input_fields_names_every_field_a_rollout_receives():
    """What makes Property 14's identity enumeration complete.

    `SHARED_INPUT_FIELDS` is asserted to be *exactly* the fields of the
    object the rollouts share, so a tensor added to `SharedRolloutInputs`
    later cannot escape the identity assertions by not being listed.
    """
    batch = _batch(_cohort(4))
    arm_0, _ = _build_model(*_COMPLETE_MODES).rollout_both_arms(batch)

    assert tuple(field.name for field in dataclasses.fields(arm_0.shared_inputs)) == (
        SHARED_INPUT_FIELDS
    )
    assert "intervention" not in SHARED_INPUT_FIELDS
    assert "intervention_input" not in SHARED_INPUT_FIELDS


# ---------------------------------------------------------------------------
# Property 13: capacity-matched channel width invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("intervention_mode", "protocol_mode"), _MODE_COMBINATIONS, ids=_MODE_IDS)
@_COHORT_SETTINGS
@given(records=_cohorts(), placebo_vector=_placebo_vectors)
def test_property_13_every_mode_combination_has_the_complete_width_and_capacity(
    intervention_mode: InterventionMode,
    protocol_mode: ProtocolMode,
    records: list[PatientRecord],
    placebo_vector: tuple[float, ...],
) -> None:
    """**Correctness Property 13**, Requirement 3.3.

    For any `InterventionMode` x `ProtocolMode` combination and any generated
    cohort, the three channel widths, the concatenated representation width,
    the GRU input width, and every parameter shape equal the complete
    model's. So a zeroed or placebo channel changes information content and
    never capacity -- the condition under which an ablation's loss in
    performance is attributable to the missing signal rather than to a
    smaller model.
    """
    batch = _batch(records)
    complete = _build_model(*_COMPLETE_MODES)
    ablated = _build_model(intervention_mode, protocol_mode, placebo_vector=placebo_vector)

    assert ablated.state_channel_width == complete.state_channel_width == _STATE_CHANNEL_WIDTH
    assert (
        ablated.intervention_channel_width
        == complete.intervention_channel_width
        == INTERVENTION_INPUT_DIM
    )
    assert ablated.protocol_channel_width == complete.protocol_channel_width == PROTOCOL_INPUT_DIM
    assert ablated.representation_width == complete.representation_width == _REPRESENTATION_WIDTH
    assert ablated.gru_cell.input_size == complete.gru_cell.input_size == _REPRESENTATION_WIDTH

    # The emitted representation, not only the advertised width.
    n_rows = len(batch.patient_ids)
    for arm in (0, 1):
        ablated_z = ablated.encode_representation(batch, arm)
        assert ablated_z.shape == (n_rows, _REPRESENTATION_WIDTH)
        assert ablated_z.shape == complete.encode_representation(batch, arm).shape
        assert torch.isfinite(ablated_z).all()

    # Capacity: the same parameters of the same shapes, in the same order.
    # The placebo buffer is not a parameter, so it cannot pad this out.
    assert [(name, tuple(p.shape)) for name, p in ablated.named_parameters()] == [
        (name, tuple(p.shape)) for name, p in complete.named_parameters()
    ]
    assert sum(p.numel() for p in ablated.parameters()) == sum(
        p.numel() for p in complete.parameters()
    )


# ---------------------------------------------------------------------------
# Property 14(a): the two arms differ only in the intervention channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("intervention_mode", "protocol_mode"), _MODE_COMBINATIONS, ids=_MODE_IDS)
@_COHORT_SETTINGS
@given(records=_cohorts(), placebo_vector=_placebo_vectors)
def test_property_14_arm_rollouts_differ_only_in_the_intervention_channel(
    intervention_mode: InterventionMode,
    protocol_mode: ProtocolMode,
    records: list[PatientRecord],
    placebo_vector: tuple[float, ...],
) -> None:
    """**Correctness Property 14** (first half), Requirements 3.4 and 3.6.

    For any generated batch and any mode combination: the arm-0 and arm-1
    rollouts receive the *same object* for their shared inputs and for every
    tensor inside it -- baseline, state, protocol, and both masks -- and the
    intervention-channel input is a distinct object carrying exactly the
    content its mode defines. Every patient is rolled out under both arms,
    and exactly one of the two rollouts is the factual one per patient.

    Under `LEARNED` the two channels differ in column 0 (the arm) and agree
    everywhere else. Under `ZEROED`/`PLACEBO` they are equal by value and
    still distinct objects, which is the ablation's content: the arm no
    longer reaches the rolled-out state. The hazards still differ in those
    modes, for the structural reason the module docstring gives, so nothing
    here asserts otherwise.
    """
    batch = _batch(records)
    model = _build_model(intervention_mode, protocol_mode, placebo_vector=placebo_vector)
    arm_0, arm_1 = model.rollout_both_arms(batch)

    # Requirement 3.6 by identity: one shared object, one tensor per field.
    assert arm_0.shared_inputs is arm_1.shared_inputs
    for field in SHARED_INPUT_FIELDS:
        assert getattr(arm_0.shared_inputs, field) is getattr(arm_1.shared_inputs, field), field
    assert torch.equal(arm_0.shared_inputs.protocol, _expected_protocol(batch, protocol_mode))

    # The intervention channel: a fresh tensor per arm, in every mode.
    assert arm_0.intervention_input is not arm_1.intervention_input
    for arm_output in (arm_0, arm_1):
        assert torch.equal(
            arm_output.intervention_input,
            _expected_intervention_input(
                batch, intervention_mode, arm_output.arm, placebo_vector
            ),
        )
        assert arm_output.intervention_input.shape == (
            len(batch.patient_ids),
            INTERVENTION_INPUT_DIM,
        )
    if intervention_mode is InterventionMode.LEARNED:
        assert not torch.equal(arm_0.intervention_input, arm_1.intervention_input)
        assert torch.equal(arm_0.intervention_input[:, 1:], arm_1.intervention_input[:, 1:])
    else:
        assert torch.equal(arm_0.intervention_input, arm_1.intervention_input)

    # Requirement 3.4: both arms, for every evaluated patient.
    n_rows = len(batch.patient_ids)
    for arm_output in (arm_0, arm_1):
        assert arm_output.hidden_states.shape[0] == n_rows
        assert arm_output.state_pred.shape[0] == n_rows
        assert arm_output.hazards.shape[0] == n_rows
        assert arm_output.outcome_96w.shape == (n_rows,)
    assert torch.all(arm_0.factual_mask ^ arm_1.factual_mask)
    assert (arm_0.arm, arm_1.arm) == (0, 1)


# ---------------------------------------------------------------------------
# Property 14(b): an ablation changes exactly the named channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("intervention_mode", "protocol_mode"), _MODE_COMBINATIONS, ids=_MODE_IDS)
@_COHORT_SETTINGS
@given(records=_cohorts(), placebo_vector=_placebo_vectors)
def test_property_14_ablation_changes_exactly_the_named_channel(
    intervention_mode: InterventionMode,
    protocol_mode: ProtocolMode,
    records: list[PatientRecord],
    placebo_vector: tuple[float, ...],
) -> None:
    """**Correctness Property 14** (second half), Requirement 3.3.

    Complete and ablated models are built from the same seed, so their
    weights are identical and the only thing an ablation can change is
    channel content. On the same batch, and for both arms: every
    non-ablated input is *the same object* the complete model received, and
    each ablated channel carries exactly its mode's defined content.

    The "differs" direction is asserted where it is a theorem rather than a
    coincidence of the data: a zeroed protocol channel always differs,
    because `is_time_zero_valid` is `1.0` for every row `build` emits; an
    ablated intervention channel is arm-independent while the learned one
    differs between arms, so it can coincide with at most one arm and must
    differ for the other. See the module docstring on why a per-arm
    "differs" assertion would be false for arm 0 under `ZEROED`.
    """
    batch = _batch(records)
    complete = _build_model(*_COMPLETE_MODES)
    ablated = _build_model(intervention_mode, protocol_mode, placebo_vector=placebo_vector)
    named = _named_channels(intervention_mode, protocol_mode)

    # Same weights, so a difference can only come from channel content.
    for (name, left), (_, right) in zip(
        complete.named_parameters(), ablated.named_parameters(), strict=True
    ):
        assert torch.equal(left, right), name

    complete_pair = complete.rollout_both_arms(batch)
    ablated_pair = ablated.rollout_both_arms(batch)

    for complete_arm, ablated_arm in zip(complete_pair, ablated_pair, strict=True):
        assert complete_arm.arm == ablated_arm.arm
        for field in SHARED_INPUT_FIELDS:
            left = getattr(complete_arm.shared_inputs, field)
            right = getattr(ablated_arm.shared_inputs, field)
            if field in named:
                assert right.shape == left.shape and right.dtype == left.dtype
                assert torch.equal(right, _expected_protocol(batch, protocol_mode))
                assert not torch.equal(right, left), field
            else:
                assert right is left, field

        expected_channel = _expected_intervention_input(
            batch, intervention_mode, ablated_arm.arm, placebo_vector
        )
        assert torch.equal(ablated_arm.intervention_input, expected_channel)
        assert ablated_arm.intervention_input.shape == complete_arm.intervention_input.shape
        if "intervention" not in named:
            assert torch.equal(ablated_arm.intervention_input, complete_arm.intervention_input)

    if "intervention" in named:
        assert any(
            not torch.equal(ablated_arm.intervention_input, complete_arm.intervention_input)
            for complete_arm, ablated_arm in zip(complete_pair, ablated_pair, strict=True)
        )
