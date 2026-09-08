"""Unit tests for `src/infra/registry.py`.

**Validates: Requirements 1.5, 1.6, 1.7**
"""

from pathlib import Path

import pytest

from infra.config import load_config
from infra.registry import (
    DuplicateExperimentError,
    ExperimentRegistry,
    compute_experiment_id,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "config"


def _load(experiment_filename: str = "valid_experiment.yaml"):
    return load_config(
        FIXTURES / "valid_dataset.yaml",
        FIXTURES / "valid_target_trial.yaml",
        FIXTURES / experiment_filename,
    )


def test_experiment_id_is_deterministic_across_separate_load_config_calls():
    resolved_a = _load()
    resolved_b = _load()

    assert compute_experiment_id(resolved_a) == compute_experiment_id(resolved_b)


def test_experiment_id_changes_when_seed_changes(tmp_path):
    base = _load()

    changed_path = tmp_path / "different_seed_experiment.yaml"
    changed_path.write_text(
        (FIXTURES / "valid_experiment.yaml").read_text().replace("12345", "54321")
    )
    changed = load_config(
        FIXTURES / "valid_dataset.yaml",
        FIXTURES / "valid_target_trial.yaml",
        changed_path,
    )

    assert base.experiment.seed != changed.experiment.seed
    assert compute_experiment_id(base) != compute_experiment_id(changed)


def test_experiment_id_is_16_hex_characters():
    resolved = _load()
    experiment_id = compute_experiment_id(resolved)

    assert len(experiment_id) == 16
    int(experiment_id, 16)  # raises ValueError if not valid hex


def test_register_appends_entry_to_registry_jsonl(tmp_path):
    resolved = _load()
    registry_path = tmp_path / "registry.jsonl"
    results_root = tmp_path / "results"
    registry = ExperimentRegistry(registry_path=registry_path, results_root=results_root)

    experiment_id = registry.register(resolved, git_commit="abc123", dirty=False)

    assert registry_path.exists()
    records = registry.read_all()
    assert len(records) == 1
    assert records[0]["experiment_id"] == experiment_id
    assert records[0]["seed"] == resolved.experiment.seed
    assert records[0]["dataset_version"] == resolved.dataset.version
    assert records[0]["witness_policy"] == resolved.experiment.witness_policy
    assert records[0]["git_commit"] == "abc123"
    assert records[0]["dirty"] is False


def test_register_is_append_only_across_distinct_experiments(tmp_path):
    resolved_primary = _load()
    registry_path = tmp_path / "registry.jsonl"
    results_root = tmp_path / "results"
    registry = ExperimentRegistry(registry_path=registry_path, results_root=results_root)

    registry.register(resolved_primary, git_commit="commit1", dirty=False)

    # A different seed produces a different experiment_id, so this second
    # registration must succeed and be appended as a second line, not replace
    # the first.
    changed_path = tmp_path / "different_seed_experiment.yaml"
    changed_path.write_text(
        (FIXTURES / "valid_experiment.yaml").read_text().replace("12345", "99999")
    )
    resolved_other = load_config(
        FIXTURES / "valid_dataset.yaml",
        FIXTURES / "valid_target_trial.yaml",
        changed_path,
    )
    registry.register(resolved_other, git_commit="commit2", dirty=True)

    records = registry.read_all()
    assert len(records) == 2
    assert records[0]["experiment_id"] != records[1]["experiment_id"]
    assert records[1]["dirty"] is True


def test_register_raises_duplicate_experiment_error_on_existing_nonempty_results_dir(
    tmp_path,
):
    resolved = _load()
    registry_path = tmp_path / "registry.jsonl"
    results_root = tmp_path / "results"
    registry = ExperimentRegistry(registry_path=registry_path, results_root=results_root)

    experiment_id = compute_experiment_id(resolved)
    existing_results_dir = results_root / experiment_id
    existing_results_dir.mkdir(parents=True)
    (existing_results_dir / "manifest.json").write_text("{}")

    with pytest.raises(DuplicateExperimentError, match=experiment_id):
        registry.register(resolved, git_commit="abc123", dirty=False)

    # The registry file must not have been written to on a rejected registration.
    assert not registry_path.exists()


def test_register_allows_reregistration_when_results_dir_is_empty(tmp_path):
    resolved = _load()
    registry_path = tmp_path / "registry.jsonl"
    results_root = tmp_path / "results"
    registry = ExperimentRegistry(registry_path=registry_path, results_root=results_root)

    experiment_id = compute_experiment_id(resolved)
    # An empty (but existing) results directory should not block registration.
    (results_root / experiment_id).mkdir(parents=True)

    returned_id = registry.register(resolved, git_commit="abc123", dirty=False)

    assert returned_id == experiment_id
    assert len(registry.read_all()) == 1
