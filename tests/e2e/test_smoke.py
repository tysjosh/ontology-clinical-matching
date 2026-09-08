"""End-to-end smoke test for `scripts/run_stage.py --stage smoke`.

Invokes the CLI script as a subprocess (a true end-to-end test of the
documented one-command entry point, Requirement 1.4) against a temporary
results root, and asserts the run succeeds and produces a valid manifest
under `results/<experiment_id>/manifest.json`.

**Validates: Requirements 1.2, 1.3, 1.4**
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_STAGE_SCRIPT = REPO_ROOT / "scripts" / "run_stage.py"


def _run_smoke_stage(results_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RUN_STAGE_SCRIPT),
            "--stage",
            "smoke",
            "--results-root",
            str(results_root),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _find_manifest(results_root: Path) -> Path:
    manifests = sorted(results_root.glob("*/manifest.json"))
    assert manifests, f"no manifest.json found under {results_root}"
    assert len(manifests) == 1, f"expected exactly one manifest, found {manifests}"
    return manifests[0]


def test_smoke_command_runs_end_to_end_and_produces_a_valid_manifest(tmp_path):
    results_root = tmp_path / "results"

    result = _run_smoke_stage(results_root)

    assert (
        result.returncode == 0
    ), f"smoke stage failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    manifest_path = _find_manifest(results_root)
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    required_keys = {
        "experiment_id",
        "stage",
        "input_hashes",
        "cohort_definition",
        "feature_list",
        "treatment_definition",
        "outcome_definition",
        "model_config",
        "checkpoint_paths",
        "predictions",
        "metrics",
        "environment_versions",
        "nondeterminism_invocations",
    }
    assert required_keys.issubset(manifest.keys())
    assert manifest["stage"] == "smoke"
    assert manifest["experiment_id"] == manifest_path.parent.name


def test_smoke_command_registers_the_experiment_in_the_registry(tmp_path):
    results_root = tmp_path / "results"

    result = _run_smoke_stage(results_root)

    assert result.returncode == 0
    registry_path = results_root / "registry.jsonl"
    assert registry_path.exists()

    records = [json.loads(line) for line in registry_path.read_text().splitlines() if line]
    assert len(records) == 1
    manifest_path = _find_manifest(results_root)
    manifest = json.loads(manifest_path.read_text())
    assert records[0]["experiment_id"] == manifest["experiment_id"]


def test_rerunning_smoke_with_same_results_root_fails_without_overwriting(tmp_path):
    results_root = tmp_path / "results"

    first = _run_smoke_stage(results_root)
    assert first.returncode == 0
    manifest_path = _find_manifest(results_root)
    original_manifest_contents = manifest_path.read_text()

    second = _run_smoke_stage(results_root)

    assert second.returncode != 0
    assert "duplicate" in second.stderr.lower() or "already has a non-empty" in second.stderr
    # The original manifest must not have been touched by the rejected rerun.
    assert manifest_path.read_text() == original_manifest_contents
