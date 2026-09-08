"""End-to-end test for `scripts/run_paper.py` (task 15.4; Requirements 15.5-15.6).

Invokes the CLI script as a subprocess (mirroring `tests/e2e/test_smoke.py`'s
subprocess-invocation pattern) against a fixed synthetic saved-predictions
JSON file and a target-trial YAML, and asserts:

- the reproduction stage exits 0, regenerates tables/figures/model-card
  files with valid JSON content under a temp output directory, and never
  creates or reads any sealed-reference-named file anywhere under a
  scratch results tree the test controls;
- `--stage locked-evaluation` (with no valid token) always exits non-zero
  with a clear message and creates no output files at all.

**Validates: Requirements 15.5, 15.6**
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_PAPER_SCRIPT = REPO_ROOT / "scripts" / "run_paper.py"
# `tests/fixtures/config/valid_target_trial.yaml` only satisfies the
# lightweight `infra.config.TargetTrialConfig` schema (used at config-load
# time), not the authoritative, richly-typed `protocols.schema.TargetTrialSpec`
# this script validates against -- so this test uses the frozen, complete
# `configs/target_trials/main.yaml` instead (see task 15.4 design guidance).
TRIAL_CONFIG = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

# One row per required column group (factual, causal recovery, selective
# prediction, consistency) -- see reporting.tables.REQUIRED_COLUMNS. Values
# are arbitrary fixed synthetic numbers, not real ACTG175 results.
_SYNTHETIC_RESULTS_TABLE_ROWS = [
    {
        "label": "synthetic_baseline",
        "n": 250,
        "concordance_index": 0.71,
        "auroc_96wk": 0.68,
        "auprc_96wk": 0.55,
        "ipcw_brier_score": 0.19,
        "calibration_slope": 0.98,
        "calibration_intercept": 0.01,
        "cd4_cd8_mae": 12.4,
        "estimated_hr": 0.61,
        "e_log_hr": 0.11,
        "effect_direction_agreement": 1,
        "ci_overlap": True,
        "risk_difference_error": 0.04,
        "retained_coverage": 0.85,
        "selective_risk": 0.12,
        "aurcc": 0.09,
        "incorrect_release_rate": 0.03,
        "unnecessary_abstention_rate": 0.05,
        "error_at_matched_coverage": 0.06,
        "temporal_violations": 0,
        "protocol_violations": 0,
        "impossible_transition_rate": 0.0,
        "witness_detected_failures": 0,
    }
]

_SYNTHETIC_EXPERIMENT_JSON = {
    "results_table_rows": _SYNTHETIC_RESULTS_TABLE_ROWS,
    "treatment_effect_estimate": {
        "point_estimate": -0.62,
        "ci_low": -0.81,
        "ci_high": -0.40,
        "n_replicates": 1000,
        "n_successful": 1000,
    },
    "reference_value": -0.69,
    "reference_label": "Internal reference",
    "ablation_results": {
        "no_treatment_input": 0.55,
        "explicit_intervention_channel": 0.71,
    },
}

_SEALED_REFERENCE_NAME_PATTERNS = ("*sealed*", "*internal_randomized_reference*")


def _write_experiment_json(path: Path) -> Path:
    experiment_path = path / "synthetic_experiment.json"
    experiment_path.write_text(json.dumps(_SYNTHETIC_EXPERIMENT_JSON, indent=2), encoding="utf-8")
    return experiment_path


def _run_run_paper(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_PAPER_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _find_sealed_reference_files(root: Path) -> list[Path]:
    found: list[Path] = []
    if not root.exists():
        return found
    for pattern in _SEALED_REFERENCE_NAME_PATTERNS:
        found.extend(root.rglob(pattern))
    return found


def test_reproduction_stage_regenerates_tables_figures_and_model_card(tmp_path):
    scratch_results_root = tmp_path / "results"
    scratch_results_root.mkdir()
    experiment_path = _write_experiment_json(tmp_path)
    output_dir = scratch_results_root / "paper_reproduction" / "reporting"

    result = _run_run_paper(
        [
            "--trial",
            str(TRIAL_CONFIG),
            "--experiment",
            str(experiment_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result.returncode == 0, (
        f"run_paper.py reproduction stage failed:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )

    main_results_table_path = output_dir / "tables" / "main_results_table.json"
    assert main_results_table_path.exists()
    main_results_table = json.loads(main_results_table_path.read_text())
    assert main_results_table == _SYNTHETIC_RESULTS_TABLE_ROWS

    architecture_diagram_path = output_dir / "figures" / "architecture_diagram.json"
    assert architecture_diagram_path.exists()
    architecture_diagram = json.loads(architecture_diagram_path.read_text())
    assert "mermaid" in architecture_diagram

    treatment_effect_plot_path = (
        output_dir / "figures" / "treatment_effect_vs_reference_plot.json"
    )
    assert treatment_effect_plot_path.exists()
    treatment_effect_plot = json.loads(treatment_effect_plot_path.read_text())
    assert treatment_effect_plot["model_point_estimate"] == -0.62
    assert treatment_effect_plot["reference_value"] == -0.69

    ablation_bar_chart_path = output_dir / "figures" / "ablation_bar_chart.json"
    assert ablation_bar_chart_path.exists()
    ablation_bar_chart = json.loads(ablation_bar_chart_path.read_text())
    assert ablation_bar_chart["labels"] == ["no_treatment_input", "explicit_intervention_channel"]

    model_card_json_path = output_dir / "model_card.json"
    assert model_card_json_path.exists()
    model_card = json.loads(model_card_json_path.read_text())
    assert "intended_use" in model_card

    model_card_md_path = output_dir / "model_card.md"
    assert model_card_md_path.exists()
    assert "Model Card" in model_card_md_path.read_text()

    provenance_path = output_dir / "provenance.json"
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text())
    # trial_path is redacted to a repo-relative path (never the raw
    # absolute path, which would leak the local username -- see
    # run_paper.py's _redact_path_for_provenance).
    assert provenance["trial_path"] == "configs/target_trials/main.yaml"
    assert not Path(provenance["trial_path"]).is_absolute()
    assert not Path(provenance["experiment_path"]).is_absolute()

    # No sealed-reference-named file was created (or, by construction of
    # this test, read) anywhere under the scratch results tree this test
    # controls.
    assert _find_sealed_reference_files(scratch_results_root) == []


def test_locked_evaluation_stage_is_refused_and_creates_no_output(tmp_path):
    scratch_results_root = tmp_path / "results"
    scratch_results_root.mkdir()
    experiment_path = _write_experiment_json(tmp_path)
    output_dir = scratch_results_root / "paper_reproduction" / "reporting"

    result = _run_run_paper(
        [
            "--trial",
            str(TRIAL_CONFIG),
            "--experiment",
            str(experiment_path),
            "--output-dir",
            str(output_dir),
            "--stage",
            "locked-evaluation",
        ]
    )

    assert result.returncode != 0
    assert "locked-evaluation" in result.stderr.lower()
    assert "LockedEvaluationToken" in result.stderr

    # No output files at all -- the refusal happens before any file access.
    assert not output_dir.exists()
    assert _find_sealed_reference_files(scratch_results_root) == []
