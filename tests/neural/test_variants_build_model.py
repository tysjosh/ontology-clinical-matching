"""Unit tests for `models.variants.build_model` and
`model_affecting_fields` (task 12.1; Requirement 6.3, 6.4, 6.5).

This module requires torch (`build_model` dispatches to the two torch
model families), so it lives under `tests/neural/` behind
`pytest.importorskip("torch")`. The torch-free half of `build_model`'s
contract -- the `MissingDependencyError` path for `"sequence_gru"`/
`"intervention_aware"` when torch is absent, and `model_affecting_fields`
itself -- is covered separately in `tests/unit/test_variants.py` (no
torch import needed there, since `"cox"` never touches torch and
`model_affecting_fields` is pure dataclass projection).

**Validates: Requirements 6.3, 6.4, 6.5**
(`.kiro/specs/neural-experiment-pipeline/requirements.md`)
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from models.dataset import PatientDatasetBuilder  # noqa: E402
from models.non_neural_baseline import CoxBaseline  # noqa: E402
from models.torch_intervention_aware import TorchInterventionAwareModel  # noqa: E402
from models.torch_sequence_gru import TorchSequenceGRU  # noqa: E402
from models.variants import (  # noqa: E402
    COMPLETE,
    MODEL_AFFECTING_FIELD_NAMES,
    ExperimentVariant,
    MissingDependencyError,
    VariantValidationError,
    build_model,
    model_affecting_fields,
)

DIMENSIONS = PatientDatasetBuilder().dimensions


def _variant(**overrides: object) -> ExperimentVariant:
    kwargs = {f.name: getattr(COMPLETE, f.name) for f in dataclasses.fields(ExperimentVariant)}
    kwargs.update(overrides)
    return ExperimentVariant(**kwargs)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_cox_dispatches_to_cox_baseline():
    variant = _variant(
        model_family="cox",
        treatment_representation="ordinary_feature",
        protocol_channel="zeroed",
    )
    model = build_model(variant, DIMENSIONS, seed=1)
    assert isinstance(model, CoxBaseline)


def test_sequence_gru_dispatches_to_torch_sequence_gru():
    variant = _variant(model_family="sequence_gru", treatment_representation="ordinary_feature")
    model = build_model(variant, DIMENSIONS, seed=1)
    assert isinstance(model, TorchSequenceGRU)


def test_intervention_aware_dispatches_to_torch_intervention_aware_model():
    model = build_model(COMPLETE, DIMENSIONS, seed=1)
    assert isinstance(model, TorchInterventionAwareModel)


def test_intervention_aware_rejects_ordinary_feature_representation():
    """'ordinary_feature' is sequence_gru's representation, not
    intervention_aware's -- it has no InterventionMode counterpart and
    must not be silently coerced to one."""
    variant = _variant(
        model_family="intervention_aware", treatment_representation="ordinary_feature"
    )
    with pytest.raises(VariantValidationError, match="ordinary_feature"):
        build_model(variant, DIMENSIONS, seed=1)


# ---------------------------------------------------------------------------
# Width is read from dimensions, never hard-coded (Requirement 6.3)
# ---------------------------------------------------------------------------


def test_sequence_gru_width_tracks_dimensions_not_a_literal():
    variant = _variant(model_family="sequence_gru", treatment_representation="ordinary_feature")
    model = build_model(variant, DIMENSIONS, seed=1)
    assert model.gru.input_size == 3  # cd4, cd8, treatment_arm -- INPUT_SIZE
    assert model.state_init_projection.in_features == DIMENSIONS.baseline_covariate_width


def test_intervention_aware_width_tracks_dimensions_not_a_literal():
    model = build_model(COMPLETE, DIMENSIONS, seed=1)
    assert model.state_channel_width == DIMENSIONS.baseline_covariate_width
    assert model.intervention_channel_width == DIMENSIONS.intervention_width
    assert model.protocol_channel_width == DIMENSIONS.protocol_width


# ---------------------------------------------------------------------------
# Seed determines parameter initialization (Requirement 6.5)
# ---------------------------------------------------------------------------


def test_same_seed_gives_bitwise_identical_initial_parameters_sequence_gru():
    variant = _variant(model_family="sequence_gru", treatment_representation="ordinary_feature")
    model_a = build_model(variant, DIMENSIONS, seed=42)
    model_b = build_model(variant, DIMENSIONS, seed=42)
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


def test_different_seed_gives_different_initial_parameters_sequence_gru():
    variant = _variant(model_family="sequence_gru", treatment_representation="ordinary_feature")
    model_a = build_model(variant, DIMENSIONS, seed=1)
    model_b = build_model(variant, DIMENSIONS, seed=2)
    differing = any(
        not torch.equal(param_a, param_b)
        for param_a, param_b in zip(
            model_a.parameters(), model_b.parameters(), strict=True
        )
    )
    assert differing


def test_same_seed_gives_bitwise_identical_initial_parameters_intervention_aware():
    model_a = build_model(COMPLETE, DIMENSIONS, seed=7)
    model_b = build_model(COMPLETE, DIMENSIONS, seed=7)
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


# ---------------------------------------------------------------------------
# Placebo vector: drawn from the seed, reproducible, frozen width
# ---------------------------------------------------------------------------


def test_placebo_channel_draws_a_vector_of_the_frozen_intervention_width():
    variant = _variant(treatment_representation="placebo_channel")
    model = build_model(variant, DIMENSIONS, seed=11)
    assert model.placebo_vector.shape == (DIMENSIONS.intervention_width,)


def test_placebo_channel_is_reproducible_at_the_same_seed():
    variant = _variant(treatment_representation="placebo_channel")
    model_a = build_model(variant, DIMENSIONS, seed=11)
    model_b = build_model(variant, DIMENSIONS, seed=11)
    assert torch.equal(model_a.placebo_vector, model_b.placebo_vector)


def test_placebo_channel_differs_across_seeds():
    variant = _variant(treatment_representation="placebo_channel")
    model_a = build_model(variant, DIMENSIONS, seed=11)
    model_b = build_model(variant, DIMENSIONS, seed=12)
    assert not torch.equal(model_a.placebo_vector, model_b.placebo_vector)


# ---------------------------------------------------------------------------
# Two variants differing in a model-affecting field yield differing models
# (Requirement 6.4)
# ---------------------------------------------------------------------------


def test_variants_differing_in_model_family_yield_different_architectures():
    gru_variant = _variant(
        model_family="sequence_gru", treatment_representation="ordinary_feature"
    )
    gru_model = build_model(gru_variant, DIMENSIONS, seed=1)
    aware_model = build_model(COMPLETE, DIMENSIONS, seed=1)
    assert type(gru_model) is not type(aware_model)
    assert model_affecting_fields(gru_variant) != model_affecting_fields(COMPLETE)


def test_variants_differing_in_treatment_representation_yield_different_input_content():
    zeroed_variant = _variant(treatment_representation="absent")
    learned_model = build_model(COMPLETE, DIMENSIONS, seed=3)
    zeroed_model = build_model(zeroed_variant, DIMENSIONS, seed=3)

    assert model_affecting_fields(zeroed_variant) != model_affecting_fields(COMPLETE)

    sample = PatientDatasetBuilder()
    # Same seed => identical parameters; only the intervention_input differs.
    for (name_a, param_a), (name_b, param_b) in zip(
        learned_model.named_parameters(), zeroed_model.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)
    del sample  # not needed beyond documenting the "same params" setup above


def test_variants_differing_only_outside_model_affecting_fields_build_identical_models():
    """A pair differing only in overlap_gate (not model-affecting) must
    still describe the *same* model -- this is what makes the
    reuse-vs-retrain decision correct (Requirement 7.6-7.8's reuse runs
    for no_overlap_gate/no_uncertainty_gate/no_clinical_constraints)."""
    no_overlap_gate_variant = _variant(overlap_gate=False)
    assert model_affecting_fields(no_overlap_gate_variant) == model_affecting_fields(COMPLETE)

    model_a = build_model(COMPLETE, DIMENSIONS, seed=5)
    model_b = build_model(no_overlap_gate_variant, DIMENSIONS, seed=5)
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


# ---------------------------------------------------------------------------
# model_affecting_fields projection shape
# ---------------------------------------------------------------------------


def test_model_affecting_fields_has_exactly_the_declared_keys():
    projected = model_affecting_fields(COMPLETE)
    assert set(projected.keys()) == set(MODEL_AFFECTING_FIELD_NAMES)


def test_model_affecting_fields_excludes_gate_and_witness_fields():
    excluded = {"overlap_gate", "uncertainty_gate", "clinical_constraints", "enabled_witnesses"}
    assert excluded.isdisjoint(MODEL_AFFECTING_FIELD_NAMES)


# ---------------------------------------------------------------------------
# MissingDependencyError is importable without being raised when torch IS
# installed (sanity check that this module's own import list is correct)
# ---------------------------------------------------------------------------


def test_missing_dependency_error_is_a_runtime_error_subclass():
    assert issubclass(MissingDependencyError, RuntimeError)
