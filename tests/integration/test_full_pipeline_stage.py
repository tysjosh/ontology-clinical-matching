"""Integration test for `scripts/run_stage.py --stage full_pipeline` (task 18;
Requirement 1.2, 1.3, 15.5; Correctness Properties 1-8).

Invokes the CLI script as a subprocess (matching `tests/e2e/test_smoke.py`'s
pattern for the `smoke` stage) and confirms the `full_pipeline` stage -- cohort
construction through the predictive baseline on a tiny synthetic cohort --
runs end to end with a zero exit code and prints the expected summary lines.

**Validates: Requirements 1.2, 1.3, 15.5**
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_STAGE_SCRIPT = REPO_ROOT / "scripts" / "run_stage.py"


def _run_full_pipeline_stage() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_STAGE_SCRIPT), "--stage", "full_pipeline"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_full_pipeline_stage_runs_end_to_end_successfully():
    result = _run_full_pipeline_stage()

    assert (
        result.returncode == 0
    ), f"full_pipeline stage failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    assert "cohort constructed:" in result.stdout
    assert "leakage audit:" in result.stdout
    assert "cox baseline fit: converged" in result.stdout
    assert "full_pipeline stage succeeded" in result.stdout


def test_full_pipeline_stage_is_deterministic_across_runs():
    first = _run_full_pipeline_stage()
    second = _run_full_pipeline_stage()

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout
