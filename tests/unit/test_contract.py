"""Unit tests for `src/data/contract.py`.

**Validates: Requirements 5.1, 5.2**
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from data.contract import DatasetContract, MissingContractError, assert_contract_present

FROZEN_CONTRACT_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "datasets" / "actg175_v3_2.yaml"
)


def _load_frozen_contract_dict() -> dict:
    with FROZEN_CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_frozen_actg175_contract_loads_successfully():
    """The frozen `configs/datasets/actg175_v3_2.yaml` contract validates
    end to end and produces a valid `DatasetContract`.
    """
    raw = _load_frozen_contract_dict()

    contract = DatasetContract.model_validate(raw)

    assert contract.name == "LongCART::ACTG175"
    assert contract.version == "3.2"
    assert contract.population.size == 2139
    assert contract.treatment == "zidovudine+didanosine"
    assert contract.comparator == "zidovudine"
    assert contract.endpoint == "composite_time_to_event"
    assert "baseline_covariates" in contract.available_modalities
    assert contract.reference_effect_source
    assert len(contract.checksum_sha256) == 64


def test_assert_contract_present_loads_the_frozen_yaml():
    """`assert_contract_present` on the real frozen contract path succeeds
    and returns a validated `DatasetContract`.
    """
    contract = assert_contract_present(FROZEN_CONTRACT_PATH)

    assert isinstance(contract, DatasetContract)
    assert contract.name == "LongCART::ACTG175"
    assert contract.version == "3.2"


def test_assert_contract_present_raises_on_nonexistent_path(tmp_path):
    """Requirement 5.2: the pipeline refuses to run without the dataset
    contract. A nonexistent contract path raises `MissingContractError`.
    """
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(MissingContractError, match="not found"):
        assert_contract_present(missing_path)


def test_assert_contract_present_raises_on_malformed_yaml(tmp_path):
    """A file that is not valid YAML raises `MissingContractError`, not a
    raw `yaml.YAMLError`.
    """
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed\n")

    with pytest.raises(MissingContractError):
        assert_contract_present(bad_yaml)


def test_assert_contract_present_raises_on_non_mapping_yaml(tmp_path):
    """A YAML file whose top level is not a mapping (e.g. a list) raises
    `MissingContractError`.
    """
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- one\n- two\n")

    with pytest.raises(MissingContractError, match="mapping"):
        assert_contract_present(list_yaml)


def test_assert_contract_present_raises_on_incomplete_contract(tmp_path):
    """A contract missing required fields fails schema validation and raises
    `MissingContractError` (not a raw `pydantic.ValidationError`).
    """
    raw = _load_frozen_contract_dict()
    del raw["checksum_sha256"]
    incomplete_path = tmp_path / "incomplete.yaml"
    incomplete_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(MissingContractError):
        assert_contract_present(incomplete_path)


def test_contract_rejects_unknown_top_level_key():
    """`extra="forbid"`: an unrecognized top-level key is rejected rather
    than silently dropped or ignored.
    """
    raw = _load_frozen_contract_dict()
    raw_with_extra = {**raw, "unexpected_extra_field": "should_not_be_allowed"}

    with pytest.raises(ValidationError) as exc_info:
        DatasetContract.model_validate(raw_with_extra)

    errors = exc_info.value.errors()
    assert any(error["type"] == "extra_forbidden" for error in errors)


@pytest.mark.parametrize(
    "missing_field",
    [
        "name",
        "version",
        "population",
        "treatment",
        "comparator",
        "endpoint",
        "study_period",
        "available_modalities",
        "reference_effect_source",
        "checksum_sha256",
    ],
)
def test_contract_missing_required_field_fails_validation(missing_field):
    """Every required top-level field is mandatory; a contract missing any
    one of them fails with a field-level `pydantic.ValidationError`.
    """
    raw = _load_frozen_contract_dict()
    incomplete = {k: v for k, v in raw.items() if k != missing_field}

    with pytest.raises(ValidationError) as exc_info:
        DatasetContract.model_validate(incomplete)

    errors = exc_info.value.errors()
    assert any(error["loc"] == (missing_field,) for error in errors)


def test_contract_instance_is_immutable():
    """A loaded `DatasetContract` cannot be mutated in place."""
    raw = _load_frozen_contract_dict()
    contract = DatasetContract.model_validate(raw)

    with pytest.raises(ValidationError):
        contract.version = "4.0"
