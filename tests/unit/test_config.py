"""Unit tests for `src/infra/config.py`.

**Validates: Requirements 1.1**
"""

from pathlib import Path

import pytest

from infra.config import ConfigError, load_config

FIXTURES = Path(__file__).parent.parent / "fixtures" / "config"


def _paths(dataset: str, target_trial: str, experiment: str) -> tuple[Path, Path, Path]:
    return (
        FIXTURES / dataset,
        FIXTURES / target_trial,
        FIXTURES / experiment,
    )


def test_valid_config_merges_into_resolved_config():
    dataset_path, target_trial_path, experiment_path = _paths(
        "valid_dataset.yaml", "valid_target_trial.yaml", "valid_experiment.yaml"
    )

    resolved = load_config(dataset_path, target_trial_path, experiment_path)

    assert resolved.dataset.name == "LongCART::ACTG175"
    assert resolved.dataset.version == "3.2"
    assert resolved.target_trial.estimand["primary"] == "marginal_hazard_ratio_itt"
    assert resolved.experiment.experiment_name == "paper_primary"
    assert resolved.experiment.seed == 12345
    assert resolved.experiment.witness_policy == "ALL_REQUIRED"


def test_canonical_dict_is_deterministic_across_repeated_calls():
    dataset_path, target_trial_path, experiment_path = _paths(
        "valid_dataset.yaml", "valid_target_trial.yaml", "valid_experiment.yaml"
    )

    resolved_a = load_config(dataset_path, target_trial_path, experiment_path)
    resolved_b = load_config(dataset_path, target_trial_path, experiment_path)

    assert resolved_a.canonical_dict() == resolved_b.canonical_dict()


def test_unknown_key_in_dataset_contract_raises_config_error():
    dataset_path, target_trial_path, experiment_path = _paths(
        "invalid_dataset_unknown_key.yaml", "valid_target_trial.yaml", "valid_experiment.yaml"
    )

    with pytest.raises(ConfigError, match="dataset contract"):
        load_config(dataset_path, target_trial_path, experiment_path)


def test_missing_required_field_in_experiment_config_raises_config_error():
    dataset_path, target_trial_path, experiment_path = _paths(
        "valid_dataset.yaml", "valid_target_trial.yaml", "invalid_experiment_missing_field.yaml"
    )

    with pytest.raises(ConfigError, match="experiment config"):
        load_config(dataset_path, target_trial_path, experiment_path)


def test_two_of_three_witness_policy_rejected_when_not_exploratory():
    dataset_path, target_trial_path, experiment_path = _paths(
        "valid_dataset.yaml",
        "valid_target_trial.yaml",
        "invalid_experiment_two_of_three_not_exploratory.yaml",
    )

    with pytest.raises(ConfigError, match="exploratory"):
        load_config(dataset_path, target_trial_path, experiment_path)


def test_missing_config_file_raises_config_error(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ConfigError, match="not found"):
        load_config(missing, missing, missing)


def test_malformed_yaml_raises_config_error(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed\n")

    with pytest.raises(ConfigError):
        load_config(bad_yaml, bad_yaml, bad_yaml)


def test_non_mapping_top_level_yaml_raises_config_error(tmp_path):
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- one\n- two\n")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(list_yaml, list_yaml, list_yaml)
