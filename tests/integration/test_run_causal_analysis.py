"""Integration test for `scripts/run_causal_analysis.py` (real-data
causal-analysis wiring, follow-up to task 18).

Calls `run_causal_analysis()` directly (the library function, not the CLI
subprocess -- this test needs to inspect the returned summary dict and the
written artifacts, so a direct call is more useful here than the
subprocess pattern `tests/integration/test_real_data_pipeline_stage.py`
uses for the simpler `run_stage.py` stages) and confirms:

- all six causal baselines produce a finite numeric estimate;
- overlap diagnostics, both negative controls, and all three witnesses
  run without error;
- real artifacts (manifest, registry entry, cohort-flow table, sealed
  reference) are written to `--results-root`, using a temporary directory
  so the real `results/` tree is never touched by this test;
- the sealed reference artifact is NOT readable without a valid
  `LockedEvaluationToken` (Correctness Property 1 still holds even though
  this script writes the artifact with its real value);
- re-running with the same `master_seed` reproduces identical results.

Runtime note: each call to `run_causal_analysis` refits several models
(cross-fitted propensity, six causal baselines, two bootstrap-based
negative controls). A small `negative_control_n_bootstrap` is used
throughout this test module (the negative-control classification logic
itself does not depend on the replicate count) to keep total runtime
reasonable; most assertions share a single `module`-scoped run via the
`analysis_summary` fixture rather than triggering a fresh run each test.

Skipped (not failed) when the real raw data is not present, matching
`tests/integration/test_real_data_pipeline_stage.py`'s convention.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"
TRIAL_CONFIG_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

#: Small bootstrap replicate count for this test module only -- keeps each
#: call to `run_causal_analysis` fast; the classification LOGIC being
#: tested (pass/fail/uncertain, artifact writing, determinism) does not
#: depend on the replicate count being large.
_TEST_N_BOOTSTRAP = 10

pytestmark = pytest.mark.skipif(
    not REAL_DATA_PATH.is_dir(),
    reason=(
        f"Real ACTG175 data not present at {REAL_DATA_PATH} (gitignored; see "
        "docs/data_access.md Section 2)."
    ),
)

from run_causal_analysis import run_causal_analysis  # noqa: E402


@pytest.fixture(scope="module")
def analysis_summary(tmp_path_factory):
    results_root = tmp_path_factory.mktemp("causal_analysis_results")
    return run_causal_analysis(
        real_data_path=REAL_DATA_PATH,
        trial_config_path=TRIAL_CONFIG_PATH,
        results_root=results_root,
        master_seed=13579,
        negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
    )


def test_produces_all_six_causal_estimates(analysis_summary):
    expected_names = {
        "unadjusted_cox",
        "adjusted_cox",
        "ps_weighted_marginal_cox",
        "outcome_regression",
        "ipcw_ipw",
        "ipcw_aipw",
    }
    assert set(analysis_summary["causal_estimates"]) == expected_names
    for name, estimate in analysis_summary["causal_estimates"].items():
        assert isinstance(estimate, float), name
        assert math.isfinite(estimate), name


def test_runs_negative_controls_and_witnesses(analysis_summary):
    assert analysis_summary["negative_control_exposure"] in {"pass", "fail", "uncertain"}
    assert analysis_summary["negative_control_outcome"] in {"pass", "fail", "uncertain"}
    assert set(analysis_summary["witness_outcomes"]) == {"temporal", "causal", "clinical"}
    for status in analysis_summary["witness_outcomes"].values():
        assert status in {"pass", "fail", "uncertain"}
    assert isinstance(analysis_summary["aggregate_decision_passed"], bool)


def test_writes_real_artifacts_to_results_root(analysis_summary):
    manifest_path = Path(analysis_summary["manifest_path"])
    reference_path = Path(analysis_summary["reference_path"])
    experiment_dir = manifest_path.parent
    flow_table_path = experiment_dir / "cohort" / "flow_table.json"
    registry_path = experiment_dir.parent / "registry.jsonl"

    assert manifest_path.is_file()
    assert reference_path.is_file()
    assert flow_table_path.is_file()
    assert registry_path.is_file()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["experiment_id"] == analysis_summary["experiment_id"]
    assert manifest["stage"] == "causal_analysis"
    assert "causal_baselines" in manifest["metrics"]
    assert len(manifest["metrics"]["causal_baselines"]) == 6

    flow_table = json.loads(flow_table_path.read_text())
    assert flow_table["flow_table"]["analyzed_population_count"] > 0

    registry_lines = registry_path.read_text().splitlines()
    assert len(registry_lines) >= 1
    last_entry = json.loads(registry_lines[-1])
    assert last_entry["experiment_id"] == analysis_summary["experiment_id"]


def test_sealed_reference_artifact_is_not_readable_without_a_token(analysis_summary):
    from infra.reference_guard import ReferenceAccessError

    reference_path = Path(analysis_summary["reference_path"])
    assert reference_path.is_file()

    from cohorts.sealed_reference import read_sealed_artifact

    with pytest.raises(ReferenceAccessError):
        read_sealed_artifact(reference_path, unlock_token=None)


def test_sealed_reference_artifact_has_owner_only_permissions(analysis_summary):
    import os
    import stat

    reference_path = Path(analysis_summary["reference_path"])
    mode = stat.S_IMODE(os.stat(reference_path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_is_deterministic_given_the_same_master_seed(tmp_path):
    summary_1 = run_causal_analysis(
        real_data_path=REAL_DATA_PATH,
        trial_config_path=TRIAL_CONFIG_PATH,
        results_root=tmp_path / "run1",
        master_seed=555,
        negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
    )
    summary_2 = run_causal_analysis(
        real_data_path=REAL_DATA_PATH,
        trial_config_path=TRIAL_CONFIG_PATH,
        results_root=tmp_path / "run2",
        master_seed=555,
        negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
    )

    assert summary_1["causal_estimates"] == summary_2["causal_estimates"]
    assert summary_1["witness_outcomes"] == summary_2["witness_outcomes"]
    assert summary_1["aggregate_decision_passed"] == summary_2["aggregate_decision_passed"]
    # Same inputs (dataset checksum, trial spec version, seed, bootstrap
    # count) -> same content-hash experiment_id, even though the two runs
    # used different results_root directories.
    assert summary_1["experiment_id"] == summary_2["experiment_id"]


def test_rerunning_with_identical_inputs_refuses_to_overwrite(tmp_path):
    """Two runs with identical (dataset checksum, trial spec version,
    master_seed, negative_control_n_bootstrap) hash to the same
    experiment_id; the second run must refuse rather than silently
    overwriting the first run's artifacts."""
    from infra.registry import DuplicateExperimentError

    run_causal_analysis(
        real_data_path=REAL_DATA_PATH,
        trial_config_path=TRIAL_CONFIG_PATH,
        results_root=tmp_path,
        master_seed=31415,
        negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
    )

    with pytest.raises(DuplicateExperimentError):
        run_causal_analysis(
            real_data_path=REAL_DATA_PATH,
            trial_config_path=TRIAL_CONFIG_PATH,
            results_root=tmp_path,
            master_seed=31415,
            negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
        )


def test_different_bootstrap_count_produces_a_different_experiment_id(tmp_path):
    """A run differing only in negative_control_n_bootstrap must get a
    different experiment_id -- it must NOT collide with (and therefore
    must not be blocked by) a prior run that used a different count."""
    summary_1 = run_causal_analysis(
        real_data_path=REAL_DATA_PATH,
        trial_config_path=TRIAL_CONFIG_PATH,
        results_root=tmp_path,
        master_seed=27182,
        negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
    )
    summary_2 = run_causal_analysis(
        real_data_path=REAL_DATA_PATH,
        trial_config_path=TRIAL_CONFIG_PATH,
        results_root=tmp_path,
        master_seed=27182,
        negative_control_n_bootstrap=_TEST_N_BOOTSTRAP + 1,
    )

    assert summary_1["experiment_id"] != summary_2["experiment_id"]
