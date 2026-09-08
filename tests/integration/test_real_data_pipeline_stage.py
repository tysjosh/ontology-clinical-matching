"""Integration test for `scripts/run_stage.py --stage real_data_pipeline`
(real-data pipeline wiring, follow-up to task 18).

Invokes the CLI script as a subprocess (matching
`tests/integration/test_full_pipeline_stage.py`'s pattern for the
`full_pipeline` stage) and confirms the `real_data_pipeline` stage -- cohort
construction through the predictive baseline on the real, packaged
ACTG175 v3.2 data -- runs end to end with a zero exit code and prints the
expected summary lines.

Skipped (not failed) when the real raw data is not present at
`data/raw/actg175_v3_2/` -- that directory is gitignored
(`docs/data_access.md` Section 2) and is not expected to exist in every
environment this suite runs in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_STAGE_SCRIPT = REPO_ROOT / "scripts" / "run_stage.py"
REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"

pytestmark = pytest.mark.skipif(
    not REAL_DATA_PATH.is_dir(),
    reason=(
        f"Real ACTG175 data not present at {REAL_DATA_PATH} (gitignored; see "
        "docs/data_access.md Section 2). This test only runs in an environment "
        "where the real raw data has been obtained and placed there."
    ),
)


def _run_real_data_pipeline_stage() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_STAGE_SCRIPT), "--stage", "real_data_pipeline"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_real_data_pipeline_stage_runs_end_to_end_successfully():
    result = _run_real_data_pipeline_stage()

    assert (
        result.returncode == 0
    ), f"real_data_pipeline stage failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    assert "loaded real ACTG175 data:" in result.stdout
    assert "cohort constructed:" in result.stdout
    assert "leakage audit:" in result.stdout
    assert "cox baseline fit: converged" in result.stdout
    assert "internal randomized reference sealed:" in result.stdout
    assert "real_data_pipeline stage succeeded" in result.stdout


def test_real_data_pipeline_stage_is_deterministic_across_runs():
    first = _run_real_data_pipeline_stage()
    second = _run_real_data_pipeline_stage()

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout


def test_real_data_pipeline_stage_refuses_cleanly_on_missing_data(tmp_path):
    missing_path = tmp_path / "does_not_exist"

    result = subprocess.run(
        [
            sys.executable,
            str(RUN_STAGE_SCRIPT),
            "--stage",
            "real_data_pipeline",
            "--real-data-path",
            str(missing_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    assert "real ACTG175 data not found" in result.stderr
