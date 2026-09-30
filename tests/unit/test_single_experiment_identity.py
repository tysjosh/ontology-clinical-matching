"""Unit tests for the reconciliation of `scripts/run_causal_analysis.py`
onto `infra.registry.compute_experiment_id` (`design.md` Section 2.6
tension 6).

The script used to mint its own identifier from a local content hash over
`(dataset_checksum, trial_spec_version, master_seed,
negative_control_n_bootstrap)` and raise its own
`DuplicateExperimentError`. Two identity schemes existed, and only one of
them was the definition of record. These tests pin the three facts that
make the deletion of the local scheme safe:

1. Every input the deleted function covered is reachable through
   `ResolvedConfig`, and therefore hashed.
2. The one input that was *not* previously in the config --
   `negative_control_n_bootstrap` -- now changes the identifier, so two
   runs differing only in the bootstrap count no longer collide.
3. A repeat run of an unchanged config is refused by the registry, not by
   the script, so the duplicate check has exactly one raise site.

These run without the real ACTG175 data: the identity question is about
the config, not the cohort. The end-to-end behaviour on real data is
covered by `tests/integration/test_run_causal_analysis.py`.

**Validates: Requirements 9.9, 18.1**
(`.kiro/specs/neural-experiment-pipeline/requirements.md`)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from infra.config import ResolvedConfig
from infra.registry import (
    DuplicateExperimentError,
    ExperimentRegistry,
    compute_experiment_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REAL_DATASET_CONFIG = REPO_ROOT / "configs" / "datasets" / "actg175_v3_2.yaml"
REAL_TRIAL_CONFIG = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
REAL_EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiments" / "real_data_primary.yaml"


def _resolve(master_seed: int, negative_control_n_bootstrap: int) -> ResolvedConfig:
    """Compose a run's config exactly as the script does."""
    from run_causal_analysis import _resolve_config_for_run

    return _resolve_config_for_run(
        REAL_DATASET_CONFIG,
        REAL_TRIAL_CONFIG,
        REAL_EXPERIMENT_CONFIG,
        master_seed,
        negative_control_n_bootstrap,
    )


def test_the_local_identity_scheme_is_gone():
    """The reconciliation is only real if the second scheme was deleted, not
    merely bypassed."""
    import run_causal_analysis

    assert not hasattr(run_causal_analysis, "_compute_content_experiment_id")
    source = Path(run_causal_analysis.__file__).read_text()
    # The script must not raise DuplicateExperimentError itself; it inherits
    # the registry's raise site. It still catches it at the CLI boundary.
    assert "raise DuplicateExperimentError" not in source


def test_every_input_the_deleted_function_covered_is_inside_the_hash():
    resolved = _resolve(master_seed=4242, negative_control_n_bootstrap=37)
    canonical = resolved.canonical_dict()

    # dataset checksum -- via the frozen dataset contract
    assert canonical["dataset"]["checksum_sha256"] == resolved.dataset.checksum_sha256
    assert len(resolved.dataset.checksum_sha256) == 64
    # trial-spec version -- via the target-trial config
    assert canonical["target_trial"]["version"] == resolved.target_trial.version
    assert resolved.target_trial.version != "unversioned"
    # master seed -- via ExperimentConfig.seed, overridden from the argument
    assert canonical["experiment"]["seed"] == 4242
    # bootstrap count -- via the new ExperimentConfig field
    assert canonical["experiment"]["negative_control_n_bootstrap"] == 37


def test_two_runs_differing_only_in_bootstrap_count_get_different_ids():
    a = _resolve(master_seed=27182, negative_control_n_bootstrap=200)
    b = _resolve(master_seed=27182, negative_control_n_bootstrap=1000)

    assert a.experiment.seed == b.experiment.seed
    assert compute_experiment_id(a) != compute_experiment_id(b)


def test_two_runs_differing_only_in_master_seed_get_different_ids():
    a = _resolve(master_seed=27182, negative_control_n_bootstrap=200)
    b = _resolve(master_seed=31415, negative_control_n_bootstrap=200)

    assert compute_experiment_id(a) != compute_experiment_id(b)


def test_an_unchanged_config_reproduces_its_identifier():
    a = _resolve(master_seed=27182, negative_control_n_bootstrap=200)
    b = _resolve(master_seed=27182, negative_control_n_bootstrap=200)

    assert compute_experiment_id(a) == compute_experiment_id(b)


def test_registry_refuses_a_repeat_run_of_an_unchanged_config(tmp_path):
    """The second run of an unchanged config must be refused before it can
    overwrite the first run's artifacts -- and the refusal must come from
    the registry."""
    resolved = _resolve(master_seed=27182, negative_control_n_bootstrap=200)
    registry = ExperimentRegistry(
        registry_path=tmp_path / "registry.jsonl", results_root=tmp_path
    )

    experiment_id = registry.register(resolved, git_commit="abc123", dirty=False)
    # Stand in for the first run's artifacts.
    experiment_dir = tmp_path / experiment_id
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "manifest.json").write_text("{}")

    with pytest.raises(DuplicateExperimentError, match=experiment_id):
        registry.register(resolved, git_commit="abc123", dirty=False)

    # The refused run left the first run's artifact untouched.
    assert (experiment_dir / "manifest.json").read_text() == "{}"


def test_a_run_differing_only_in_bootstrap_count_is_not_blocked(tmp_path):
    """The flip side of duplicate detection: two genuinely different runs
    must both be registrable in the same results root."""
    first = _resolve(master_seed=27182, negative_control_n_bootstrap=200)
    second = _resolve(master_seed=27182, negative_control_n_bootstrap=1000)
    registry = ExperimentRegistry(
        registry_path=tmp_path / "registry.jsonl", results_root=tmp_path
    )

    first_id = registry.register(first, git_commit="abc123", dirty=False)
    (tmp_path / first_id).mkdir(parents=True)
    (tmp_path / first_id / "manifest.json").write_text("{}")

    second_id = registry.register(second, git_commit="abc123", dirty=False)

    assert second_id != first_id
    assert len(registry.read_all()) == 2
