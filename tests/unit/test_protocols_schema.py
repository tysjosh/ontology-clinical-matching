"""Unit tests for `src/protocols/schema.py::TargetTrialSpec`.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from protocols.schema import TargetTrialSpec

MAIN_SPEC_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"
)

REQUIRED_TOP_LEVEL_FIELDS = [
    "population",
    "intervention",
    "comparator",
    "time_zero",
    "follow_up",
    "outcome",
    "censoring",
    "estimand",
    "analysis",
    "causal_diagram",
    "adjustment_set",
    "assumptions",
]


def _load_main_spec_dict() -> dict:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_frozen_actg175_spec_loads_successfully():
    """The frozen `configs/target_trials/main.yaml` spec validates end to end."""
    raw = _load_main_spec_dict()

    spec = TargetTrialSpec.model_validate(raw)

    assert spec.version == "1.0.0"
    assert spec.intervention.concept_id == "zidovudine+didanosine"
    assert spec.comparator.concept_id == "zidovudine"
    assert spec.estimand.primary == "marginal_hazard_ratio_itt"
    assert spec.estimand.external_reference_hr == 0.50
    assert spec.analysis.method == "ipcw_aipw"
    assert spec.adjustment_set == raw["adjustment_set"]
    assert set(spec.causal_diagram.nodes) == {"X", "A", "L", "Y", "S", "C"}


def test_assumptions_include_all_six_named_fields():
    """Requirement 3.6: consistency, conditional exchangeability, positivity,
    no-interference, missing-data, and censoring assumptions are all present
    and non-empty.
    """
    raw = _load_main_spec_dict()
    spec = TargetTrialSpec.model_validate(raw)

    assert spec.assumptions.consistency
    assert spec.assumptions.conditional_exchangeability
    assert spec.assumptions.positivity
    assert spec.assumptions.no_interference
    assert spec.assumptions.missing_data
    assert spec.assumptions.censoring


def test_adjustment_set_is_frozen_list_of_strings():
    """Requirement 3.7: the adjustment set is present and frozen (list[str])."""
    raw = _load_main_spec_dict()
    spec = TargetTrialSpec.model_validate(raw)

    assert isinstance(spec.adjustment_set, list)
    assert all(isinstance(item, str) for item in spec.adjustment_set)
    assert "baseline_cd4" in spec.adjustment_set
    assert "baseline_cd8" in spec.adjustment_set


def test_spec_instance_is_immutable():
    """Requirement 3.3: a complete specification is immutable after construction.

    Amendments require constructing a new `TargetTrialSpec` with a new
    `version`, never in-place mutation.
    """
    raw = _load_main_spec_dict()
    spec = TargetTrialSpec.model_validate(raw)

    with pytest.raises(ValidationError):
        spec.version = "2.0.0"


@pytest.mark.parametrize("missing_field", REQUIRED_TOP_LEVEL_FIELDS)
def test_spec_missing_required_field_fails_with_field_level_error(missing_field):
    """Requirement 3.1-3.2: every required top-level field is mandatory, and a
    spec missing any one of them fails with a field-level `pydantic.ValidationError`
    naming that field.
    """
    raw = _load_main_spec_dict()
    incomplete = {k: v for k, v in raw.items() if k != missing_field}

    with pytest.raises(ValidationError) as exc_info:
        TargetTrialSpec.model_validate(incomplete)

    errors = exc_info.value.errors()
    assert any(error["loc"] == (missing_field,) for error in errors)


def test_spec_rejects_unknown_top_level_key():
    """`extra="forbid"` (Requirement 3.2): an unrecognized top-level key is
    rejected rather than silently dropped or ignored.
    """
    raw = _load_main_spec_dict()
    raw_with_extra = {**raw, "unexpected_extra_field": "should_not_be_allowed"}

    with pytest.raises(ValidationError) as exc_info:
        TargetTrialSpec.model_validate(raw_with_extra)

    errors = exc_info.value.errors()
    assert any(error["type"] == "extra_forbidden" for error in errors)


def test_nested_model_rejects_unknown_key():
    """`extra="forbid"` also applies inside nested sub-models (e.g. `outcome`)."""
    raw = _load_main_spec_dict()
    invalid = {**raw, "outcome": {**raw["outcome"], "unexpected_key": "nope"}}

    with pytest.raises(ValidationError) as exc_info:
        TargetTrialSpec.model_validate(invalid)

    errors = exc_info.value.errors()
    assert any(error["type"] == "extra_forbidden" for error in errors)


def test_assumptions_missing_named_assumption_fails():
    """Requirement 3.6: each of the six named assumptions is individually required."""
    raw = _load_main_spec_dict()
    invalid_assumptions = {k: v for k, v in raw["assumptions"].items() if k != "positivity"}
    invalid = {**raw, "assumptions": invalid_assumptions}

    with pytest.raises(ValidationError) as exc_info:
        TargetTrialSpec.model_validate(invalid)

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("assumptions", "positivity") for error in errors)
