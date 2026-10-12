"""Integration test for `scripts/run_ablations.py`'s real-data CLI path
(task 14; real-data pipeline wiring, follow-up to task 18).

Invokes the CLI script as a subprocess (matching
`tests/integration/test_real_data_pipeline_stage.py`'s pattern) and
confirms `--target-trial-config ... --master-seed ... --seeds ...`
(defaulting to the real `data/raw/actg175_v3_2/` path) runs the six
retrain-mode frozen ablation arms against the real ACTG175 data end to
end, writing one `run_<run_id>_seed<seed>.json` per `(run, seed)` plus one
`retrained_arms_summary.json`.

`--seeds` is passed explicitly with a single seed (rather than relying on
the five-frozen-seed default) so this test trains each of the six arms
once rather than five times, keeping it fast enough for routine CI
execution; the full five-seed run is what a real experiment invocation
uses, not what this test needs to prove the CLI path works end to end.

Skipped (not failed) when the real raw data is not present, matching the
other real-data integration tests' convention.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_ABLATIONS_SCRIPT = REPO_ROOT / "scripts" / "run_ablations.py"
TRIAL_CONFIG_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"

pytestmark = pytest.mark.skipif(
    not REAL_DATA_PATH.is_dir(),
    reason=(
        f"Real ACTG175 data not present at {REAL_DATA_PATH} (gitignored; see "
        "docs/data_access.md Section 2)."
    ),
)


def _run_ablations_cli(
    results_root: Path, master_seed: int = 909090, seeds: tuple[int, ...] = (101,)
) -> subprocess.CompletedProcess[str]:
    # NOTE: a VariantCollisionError here would be a genuine finding (the
    # real-data retrained arms producing fewer than six distinct
    # concordance values, or two architecturally different arms agreeing
    # to within 1e-9) rather than a test artifact, since this invocation
    # uses the full frozen training configuration, not a shortened one.
    return subprocess.run(
        [
            sys.executable,
            str(RUN_ABLATIONS_SCRIPT),
            "--target-trial-config",
            str(TRIAL_CONFIG_PATH),
            "--master-seed",
            str(master_seed),
            "--results-root",
            str(results_root),
            "--seeds",
            *(str(seed) for seed in seeds),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_run_ablations_cli_runs_the_six_retrain_arms_against_real_data(tmp_path):
    result = _run_ablations_cli(tmp_path)

    assert (
        result.returncode == 0
    ), f"run_ablations.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "loaded real ACTG175 data:" in result.stdout
    assert "ran 6 retrain-mode ablation arms at 1 seed(s)" in result.stdout
    assert "run_ablations.py succeeded" in result.stdout

    result_files = sorted(tmp_path.glob("*/ablations/run_*_seed*.json"))
    assert len(result_files) == 6

    run_ids = set()
    for result_file in result_files:
        payload = json.loads(result_file.read_text())
        assert "run_id" in payload
        assert "succeeded" in payload
        assert payload["succeeded"] is True, payload.get("failure_reason")
        run_ids.add(payload["run_id"])
    assert len(run_ids) == 6

    summary_files = sorted(tmp_path.glob("*/ablations/retrained_arms_summary.json"))
    assert len(summary_files) == 1
    summary = json.loads(summary_files[0].read_text())
    assert summary["table_incomplete"] is False
    assert len(summary["runs"]) == 6


def test_run_ablations_cli_refuses_cleanly_on_missing_real_data(tmp_path):
    missing_path = tmp_path / "does_not_exist"

    result = subprocess.run(
        [
            sys.executable,
            str(RUN_ABLATIONS_SCRIPT),
            "--target-trial-config",
            str(TRIAL_CONFIG_PATH),
            "--master-seed",
            "1",
            "--real-data-path",
            str(missing_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "real ACTG175 data not found" in result.stderr
