"""Unit tests for `models.variants.ExperimentVariant` (task 12; Requirement
6.1, 6.2; `design.md` Section 3.5, Section 4.2).

This module must import and run with torch absent -- it is one of the
`TORCH_FREE_MODULES` named in `tests/unit/test_torch_isolation.py` -- so no
test here may import anything from `models.torch_*` or `models.dataset`.

**Validates: Requirements 6.1, 6.2**
(`.kiro/specs/neural-experiment-pipeline/requirements.md`)
"""

from __future__ import annotations

import dataclasses

import pytest

from models.variants import (
    COMPLETE,
    MODEL_AFFECTING_FIELD_NAMES,
    ExperimentVariant,
    MissingDependencyError,
    VariantValidationError,
    build_model,
    model_affecting_fields,
)


def _complete_kwargs(**overrides: object) -> dict[str, object]:
    """`COMPLETE`'s fields as a kwargs dict, with `overrides` applied.

    Lets each test construct a variant that differs from the known-valid
    `COMPLETE` instance in exactly one respect.
    """
    kwargs = {f.name: getattr(COMPLETE, f.name) for f in dataclasses.fields(ExperimentVariant)}
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# COMPLETE itself
# ---------------------------------------------------------------------------


def test_complete_variant_constructs_without_error():
    assert COMPLETE.name == "complete"
    assert COMPLETE.model_family == "intervention_aware"
    assert COMPLETE.treatment_representation == "explicit_channel"
    assert COMPLETE.protocol_channel == "active"
    assert COMPLETE.overlap_gate is True
    assert COMPLETE.uncertainty_gate is True
    assert COMPLETE.clinical_constraints is True
    assert COMPLETE.enabled_witnesses == frozenset({"temporal", "causal", "clinical"})
    assert COMPLETE.causal_adjustment is True
    assert COMPLETE.balance_penalty_enabled is True


def test_experiment_variant_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        COMPLETE.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Per-field out-of-set values
# ---------------------------------------------------------------------------


def test_empty_name_is_rejected():
    with pytest.raises(VariantValidationError, match="name"):
        ExperimentVariant(**_complete_kwargs(name=""))


def test_whitespace_only_name_is_rejected():
    with pytest.raises(VariantValidationError, match="name"):
        ExperimentVariant(**_complete_kwargs(name="   "))


def test_non_string_name_is_rejected():
    with pytest.raises(VariantValidationError, match="name"):
        ExperimentVariant(**_complete_kwargs(name=123))


@pytest.mark.parametrize("bad_value", ["gradient_boosted_tree", "", None, 1])
def test_out_of_set_model_family_is_rejected(bad_value):
    with pytest.raises(VariantValidationError, match="model_family"):
        ExperimentVariant(**_complete_kwargs(model_family=bad_value))


@pytest.mark.parametrize("bad_value", ["learned_embedding", "", None])
def test_out_of_set_treatment_representation_is_rejected(bad_value):
    with pytest.raises(VariantValidationError, match="treatment_representation"):
        # Pair with sequence_gru so this doesn't also trip the impossible-combo
        # check and obscure which message is being asserted.
        ExperimentVariant(
            **_complete_kwargs(
                model_family="cox",
                protocol_channel="zeroed",
                treatment_representation=bad_value,
            )
        )


@pytest.mark.parametrize("bad_value", ["partial", "", None])
def test_out_of_set_protocol_channel_is_rejected(bad_value):
    with pytest.raises(VariantValidationError, match="protocol_channel"):
        ExperimentVariant(**_complete_kwargs(protocol_channel=bad_value))


@pytest.mark.parametrize(
    "field_name",
    [
        "overlap_gate",
        "uncertainty_gate",
        "clinical_constraints",
        "causal_adjustment",
        "balance_penalty_enabled",
    ],
)
def test_non_bool_gate_fields_are_rejected(field_name):
    with pytest.raises(VariantValidationError, match=field_name):
        ExperimentVariant(**_complete_kwargs(**{field_name: "yes"}))


def test_enabled_witnesses_must_be_a_frozenset():
    with pytest.raises(VariantValidationError, match="enabled_witnesses"):
        ExperimentVariant(**_complete_kwargs(enabled_witnesses=["temporal", "causal"]))


def test_enabled_witnesses_rejects_unknown_witness_name():
    with pytest.raises(VariantValidationError, match="enabled_witnesses"):
        ExperimentVariant(
            **_complete_kwargs(enabled_witnesses=frozenset({"temporal", "statistical"}))
        )


def test_enabled_witnesses_may_be_empty():
    """An empty witness set is valid -- it is exactly what the
    'all three witnesses removed' corner of the witness-removed ablation
    would describe, not an out-of-set value."""
    variant = ExperimentVariant(**_complete_kwargs(enabled_witnesses=frozenset()))
    assert variant.enabled_witnesses == frozenset()


# ---------------------------------------------------------------------------
# The two structurally impossible combinations (design.md Section 3.5)
# ---------------------------------------------------------------------------


def test_sequence_gru_with_explicit_channel_is_rejected():
    with pytest.raises(VariantValidationError, match="sequence_gru"):
        ExperimentVariant(
            **_complete_kwargs(
                model_family="sequence_gru",
                treatment_representation="explicit_channel",
                protocol_channel="zeroed",
            )
        )


def test_sequence_gru_with_ordinary_feature_is_allowed():
    """The impossible combination is specifically explicit_channel; ordinary
    feature treatment is exactly what the sequence-GRU baseline uses."""
    variant = ExperimentVariant(
        **_complete_kwargs(
            model_family="sequence_gru",
            treatment_representation="ordinary_feature",
            protocol_channel="zeroed",
        )
    )
    assert variant.model_family == "sequence_gru"
    assert variant.treatment_representation == "ordinary_feature"


def test_cox_with_active_protocol_channel_is_rejected():
    with pytest.raises(VariantValidationError, match="cox"):
        ExperimentVariant(
            **_complete_kwargs(
                model_family="cox",
                treatment_representation="ordinary_feature",
                protocol_channel="active",
            )
        )


def test_cox_with_zeroed_protocol_channel_is_allowed():
    variant = ExperimentVariant(
        **_complete_kwargs(
            model_family="cox",
            treatment_representation="ordinary_feature",
            protocol_channel="zeroed",
        )
    )
    assert variant.model_family == "cox"
    assert variant.protocol_channel == "zeroed"


def test_multiple_problems_are_named_in_one_error():
    """Both an out-of-set field and an impossible combination violated at
    once must appear together in a single raised error, not require two
    separate construction attempts to discover."""
    with pytest.raises(VariantValidationError) as excinfo:
        ExperimentVariant(
            **_complete_kwargs(
                name="",
                model_family="sequence_gru",
                treatment_representation="explicit_channel",
            )
        )
    message = str(excinfo.value)
    assert "name" in message
    assert "sequence_gru" in message


# ---------------------------------------------------------------------------
# Torch-free import (design.md Section 2.4)
# ---------------------------------------------------------------------------


def test_build_model_cox_requires_no_torch():
    """`"cox"` is the one model_family build_model must be able to
    construct even when torch is entirely absent -- confirmed here by
    simply calling it in this torch-free test module (no
    `importorskip("torch")` guard anywhere in this file)."""
    from models.non_neural_baseline import CoxBaseline

    variant = ExperimentVariant(
        **{
            **_complete_kwargs(),
            "model_family": "cox",
            "treatment_representation": "ordinary_feature",
            "protocol_channel": "zeroed",
        }
    )
    model = build_model(variant, dimensions=None, seed=1)
    assert isinstance(model, CoxBaseline)


@pytest.mark.parametrize("model_family", ["sequence_gru", "intervention_aware"])
def test_build_model_raises_missing_dependency_error_without_torch(monkeypatch, model_family):
    """Simulates a torch-free environment by making `import torch` fail,
    regardless of whether torch happens to be installed in the test
    runner. `build_model` must raise `MissingDependencyError`, naming the
    `neural` extra, rather than letting an `ImportError` propagate raw or
    (worse) silently falling back to a different model family.
    """
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"simulated absence of torch for {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    kwargs = _complete_kwargs(model_family=model_family)
    if model_family == "sequence_gru":
        kwargs["treatment_representation"] = "ordinary_feature"
    variant = ExperimentVariant(**kwargs)

    with pytest.raises(MissingDependencyError, match="neural"):
        build_model(variant, dimensions=None, seed=1)


def test_model_affecting_fields_is_pure_dataclass_projection_no_torch_needed():
    projected = model_affecting_fields(COMPLETE)
    assert set(projected.keys()) == set(MODEL_AFFECTING_FIELD_NAMES)
    assert projected["model_family"] == "intervention_aware"


def test_module_imports_without_torch(monkeypatch):
    """`models.variants` must be importable with torch absent from
    `sys.modules`/unavailable, since `evaluation.ablation_matrix` and
    `evaluation.robustness` depend on it and both must stay torch-free at
    import time (`design.md` Section 2.4). The authoritative subprocess
    check lives in `tests/unit/test_torch_isolation.py`; this is a cheap
    same-process sanity check that the module itself performs no top-level
    torch import.
    """
    import ast
    import pathlib

    source_path = pathlib.Path(__file__).parent.parent.parent / "src" / "models" / "variants.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    # Only the module's direct top-level statements -- deliberately not
    # `ast.walk`, which would also descend into function bodies (e.g.
    # `build_model`'s deferred `import torch`, which is correct and must
    # NOT be flagged here).
    top_level_imports = [
        node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    for node in top_level_imports:
        names = (
            [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
        )
        for name in names:
            assert name is not None and not name.startswith("torch"), (
                f"models/variants.py must not import torch at module scope, found {name!r}"
            )
