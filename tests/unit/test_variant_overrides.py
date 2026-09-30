"""Unit tests for the variant-override fields added to
`src/infra/config.py`'s `ExperimentConfig` (`design.md` Section 2.6
tension 3).

Three things are being pinned here:

1. Every new field defaults to the primary/frozen setting, so a config
   that names none of them describes the primary analysis exactly.
2. An out-of-set literal, or a key the schema does not declare, is a
   loud validation error rather than a silently-absorbed extra --
   `extra="forbid"` is what makes a misspelled sensitivity knob fail at
   load time instead of quietly executing the primary analysis under a
   variant's name.
3. `canonical_dict()` serializes the defaults, so two runs differing in
   any single override get different `experiment_id` values. This is the
   property that makes the eight sensitivity variants separately
   identifiable at all.

**Validates: Requirements 8.1, 8.2, 9.9, 18.1**
(`.kiro/specs/neural-experiment-pipeline/requirements.md`)
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from infra.config import (
    ExperimentConfig,
    ExperimentVariantSpec,
    ResolvedConfig,
    VariantOverrides,
    load_config,
)
from infra.registry import compute_experiment_id

FIXTURES = Path(__file__).parent.parent / "fixtures" / "config"


def _resolved() -> ResolvedConfig:
    return load_config(
        FIXTURES / "valid_dataset.yaml",
        FIXTURES / "valid_target_trial.yaml",
        FIXTURES / "valid_experiment.yaml",
    )


def _with_experiment(resolved: ResolvedConfig, **updates: object) -> ResolvedConfig:
    """A copy of `resolved` whose experiment config carries `updates`."""
    return ResolvedConfig(
        dataset=resolved.dataset,
        target_trial=resolved.target_trial,
        experiment=resolved.experiment.model_copy(update=updates),
    )


COMPLETE_VARIANT = ExperimentVariantSpec(
    name="complete",
    model_family="intervention_aware",
    treatment_representation="explicit_channel",
    protocol_channel="active",
    overlap_gate=True,
    uncertainty_gate=True,
    clinical_constraints=True,
    enabled_witnesses=("causal", "clinical", "temporal"),
    causal_adjustment=True,
    balance_penalty_enabled=True,
)


# --------------------------------------------------------------------------
# 1. Defaults equal the primary/frozen settings
# --------------------------------------------------------------------------


def test_variant_overrides_default_to_the_primary_settings():
    """`decisions.md` Section 26's re-identification amendment records each
    of these eight defaults; they are the primary analysis."""
    overrides = VariantOverrides()

    assert overrides.propensity_model == "logistic"
    assert overrides.weight_clipping == "primary_1_99"
    assert overrides.support_thresholds == "primary_0.10_0.90"
    assert overrides.missingness_stress == "none"
    assert overrides.landmark_masking == "none"
    assert overrides.comparator_orientation == "primary"
    assert overrides.endpoint_horizon_weeks == 96.0
    assert overrides.selection_severity == "primary"


def test_experiment_config_defaults_to_primary_variant_no_model_and_1000_bootstrap():
    config = ExperimentConfig(experiment_name="x", seed=1)

    assert config.variant == VariantOverrides()
    assert config.model_variant is None
    assert config.negative_control_n_bootstrap == 1000


def test_existing_config_file_loads_with_the_primary_defaults():
    """A config file written before these fields existed still loads, and
    loads as the primary analysis."""
    resolved = _resolved()

    assert resolved.experiment.variant == VariantOverrides()
    assert resolved.experiment.model_variant is None
    assert resolved.experiment.negative_control_n_bootstrap == 1000


def test_variant_overrides_are_frozen():
    overrides = VariantOverrides()

    with pytest.raises(ValidationError):
        overrides.selection_severity = "stronger"


# --------------------------------------------------------------------------
# 2. Out-of-set literals and unknown keys are rejected
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("propensity_model", "random_forest"),
        ("weight_clipping", "primary_2_98"),
        ("support_thresholds", "primary_0.15_0.85"),
        ("missingness_stress", "decreased"),
        ("landmark_masking", "week_96_removed"),
        ("comparator_orientation", "flipped"),
        ("selection_severity", "weaker"),
    ],
)
def test_out_of_set_variant_override_value_is_rejected(field, bad_value):
    with pytest.raises(ValidationError) as excinfo:
        VariantOverrides(**{field: bad_value})

    assert field in str(excinfo.value)


def test_unknown_variant_override_key_is_rejected():
    with pytest.raises(ValidationError) as excinfo:
        VariantOverrides(propensity_modle="logistic")

    assert "propensity_modle" in str(excinfo.value)


def test_unknown_experiment_config_key_is_rejected():
    with pytest.raises(ValidationError) as excinfo:
        ExperimentConfig(
            experiment_name="x", seed=1, negative_control_n_bootstraps=500
        )

    assert "negative_control_n_bootstraps" in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_family", "transformer"),
        ("treatment_representation", "implicit"),
        ("protocol_channel", "partial"),
        ("enabled_witnesses", ("temporal", "economic")),
    ],
)
def test_out_of_set_experiment_variant_spec_value_is_rejected(field, bad_value):
    fields = COMPLETE_VARIANT.model_dump()
    fields[field] = bad_value

    with pytest.raises(ValidationError) as excinfo:
        ExperimentVariantSpec(**fields)

    assert field in str(excinfo.value)


def test_empty_experiment_variant_name_is_rejected():
    fields = COMPLETE_VARIANT.model_dump()
    fields["name"] = "   "

    with pytest.raises(ValidationError, match="non-empty"):
        ExperimentVariantSpec(**fields)


def test_enabled_witnesses_normalize_to_sorted_unique():
    """Two configs naming the same witness set in different order describe
    the same variant, so they must serialize -- and therefore hash --
    identically."""
    a = ExperimentVariantSpec(
        **{
            **COMPLETE_VARIANT.model_dump(),
            "enabled_witnesses": ("temporal", "causal", "clinical"),
        }
    )
    b = ExperimentVariantSpec(
        **{
            **COMPLETE_VARIANT.model_dump(),
            "enabled_witnesses": ("clinical", "temporal", "causal", "temporal"),
        }
    )

    assert b.enabled_witnesses == ("causal", "clinical", "temporal")
    assert b.model_dump(mode="json") == a.model_dump(mode="json")


# --------------------------------------------------------------------------
# 3. canonical_dict() serializes the defaults, so overrides re-identify
# --------------------------------------------------------------------------


def test_canonical_dict_serializes_the_defaulted_fields():
    canonical = _resolved().canonical_dict()

    experiment = canonical["experiment"]
    assert experiment["negative_control_n_bootstrap"] == 1000
    assert experiment["model_variant"] is None
    assert experiment["variant"] == {
        "propensity_model": "logistic",
        "weight_clipping": "primary_1_99",
        "support_thresholds": "primary_0.10_0.90",
        "missingness_stress": "none",
        "landmark_masking": "none",
        "comparator_orientation": "primary",
        "endpoint_horizon_weeks": 96.0,
        "selection_severity": "primary",
    }


@pytest.mark.parametrize(
    ("field", "non_default"),
    [
        ("propensity_model", "boosted"),
        ("weight_clipping", "narrow_0p5_99p5"),
        ("support_thresholds", "strict_0.20_0.80"),
        ("missingness_stress", "increased"),
        ("landmark_masking", "week_20_removed"),
        ("comparator_orientation", "reversed"),
        ("endpoint_horizon_weeks", 48.0),
        ("selection_severity", "stronger"),
    ],
)
def test_each_variant_override_changes_the_experiment_id(field, non_default):
    base = _resolved()
    varied = _with_experiment(
        base, variant=VariantOverrides(**{field: non_default})
    )

    assert compute_experiment_id(base) != compute_experiment_id(varied)


def test_model_variant_changes_the_experiment_id():
    base = _resolved()
    with_variant = _with_experiment(base, model_variant=COMPLETE_VARIANT)

    assert compute_experiment_id(base) != compute_experiment_id(with_variant)


def test_two_model_variants_differing_in_one_field_get_different_ids():
    base = _resolved()
    a = _with_experiment(base, model_variant=COMPLETE_VARIANT)
    b = _with_experiment(
        base,
        model_variant=COMPLETE_VARIANT.model_copy(
            update={"balance_penalty_enabled": False}
        ),
    )

    assert compute_experiment_id(a) != compute_experiment_id(b)


def test_negative_control_bootstrap_count_changes_the_experiment_id():
    base = _resolved()
    varied = _with_experiment(base, negative_control_n_bootstrap=200)

    assert compute_experiment_id(base) != compute_experiment_id(varied)


def test_experiment_id_is_unchanged_by_restating_the_defaults_explicitly():
    """Identity is a function of the resolved values, not of which keys the
    author happened to type."""
    base = _resolved()
    explicit = _with_experiment(
        base,
        variant=VariantOverrides(
            propensity_model="logistic",
            weight_clipping="primary_1_99",
            support_thresholds="primary_0.10_0.90",
            missingness_stress="none",
            landmark_masking="none",
            comparator_orientation="primary",
            endpoint_horizon_weeks=96.0,
            selection_severity="primary",
        ),
        model_variant=None,
        negative_control_n_bootstrap=1000,
    )

    assert compute_experiment_id(base) == compute_experiment_id(explicit)


def test_variant_overrides_can_be_named_in_a_yaml_experiment_config(tmp_path):
    """The overrides are reachable from a config file, not only from
    Python -- that is the path `SensitivityRunner.apply_variant` needs."""
    experiment_path = tmp_path / "sensitivity_experiment.yaml"
    experiment_path.write_text(
        "experiment_name: stronger_selection_severity\n"
        "seed: 12345\n"
        "variant:\n"
        "  selection_severity: stronger\n"
    )

    resolved = load_config(
        FIXTURES / "valid_dataset.yaml",
        FIXTURES / "valid_target_trial.yaml",
        experiment_path,
    )

    assert resolved.experiment.variant.selection_severity == "stronger"
    # Untouched knobs stay primary.
    assert resolved.experiment.variant.propensity_model == "logistic"
