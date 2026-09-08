"""Integration test for `scripts/run_ablations.py`'s real-data CLI path
(real-data pipeline wiring, follow-up to task 18).

Invokes the CLI script as a subprocess (matching
`tests/integration/test_real_data_pipeline_stage.py`'s pattern) and
confirms `--target-trial-config ... --master-seed ...` (defaulting to the
real `data/raw/actg175_v3_2/` path) runs all ten frozen ablations against
the real ACTG175 data end to end, writing one `result.json` per ablation.

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
    results_root: Path, master_seed: int = 909090
) -> subprocess.CompletedProcess[str]:
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
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_run_ablations_cli_runs_all_ten_against_real_data(tmp_path):
    result = _run_ablations_cli(tmp_path)

    assert (
        result.returncode == 0
    ), f"run_ablations.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "loaded real ACTG175 data:" in result.stdout
    assert "ran 10 ablations" in result.stdout
    assert "run_ablations.py succeeded" in result.stdout

    result_files = sorted(tmp_path.glob("*/ablations/*/result.json"))
    assert len(result_files) == 10

    names = set()
    for result_file in result_files:
        payload = json.loads(result_file.read_text())
        assert "ablation_name" in payload
        assert "concordance_index" in payload
        assert "config_diff" in payload
        names.add(payload["ablation_name"])
    assert len(names) == 10


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
