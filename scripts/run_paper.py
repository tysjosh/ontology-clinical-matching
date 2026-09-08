#!/usr/bin/env python
"""One-command paper-reproduction entry point (task 15.4; Requirements 15.5-15.6).

`scripts/run_stage.py` runs one pipeline *stage*. This script is a
different, narrower thing: it regenerates the reporting artifacts (main
results table, principal figures, model card) from **already-saved**
prediction/metric data. It does not re-run the causal/model pipeline
itself -- it is a thin caller of `src.reporting.tables`,
`src.reporting.figures`, and `src.reporting.model_card` (Requirement 15.5:
"A single documented command SHALL regenerate paper tables and figures
from saved privacy-safe predictions or aggregates").

Usage::

    python scripts/run_paper.py \\
        --trial configs/target_trials/main.yaml \\
        --experiment path/to/saved_predictions.json \\
        --output-dir results/paper_reproduction/reporting

The saved-experiment JSON schema
---------------------------------

`--experiment` points at a caller-prepared JSON file with the following
top-level keys, all optional except `results_table_rows`::

    {
      "results_table_rows": [
        {
          // every column in reporting.tables.REQUIRED_COLUMNS, plus any
          // extra columns (e.g. "label", "n") the caller wants passed
          // through unchanged.
          "concordance_index": 0.71, "auroc_96wk": 0.68, ...
        },
        ...
      ],
      "treatment_effect_estimate": {
        "point_estimate": -0.62, "ci_low": -0.81, "ci_high": -0.40,
        "n_replicates": 1000, "n_successful": 1000
      },
      "reference_value": -0.69,
      "reference_label": "Internal reference",
      "ablation_results": {
        "no_treatment_input": 0.55, "treatment_as_ordinary_feature": 0.68
      }
    }

- `results_table_rows` (required): rows forwarded unchanged to
  `reporting.tables.build_main_results_table()`.
- `treatment_effect_estimate` + `reference_value` (optional, used
  together): forwarded to
  `reporting.figures.build_treatment_effect_vs_reference_plot()` as a
  reconstructed `uncertainty.bootstrap.BootstrapCI` plus the scalar
  reference value. `reference_value` is a plain caller-supplied number --
  reading it from this file never touches
  `reference/internal_randomized_reference.sealed.json`.
- `ablation_results` (optional): forwarded unchanged to
  `reporting.figures.build_ablation_bar_chart()`.

Figures deliberately out of scope for this entry point
----------------------------------------------------------

`reporting.figures.build_cohort_flow_diagram()`,
`build_propensity_overlap_plot()`, and `build_risk_coverage_curve_plot()`
each require rich pipeline objects (patient timelines and treatment-arm
assignments, a `causal.overlap.OverlapReport`, or raw
prediction/label/confidence arrays) that are not part of the minimal saved
prediction/metric schema documented above -- they are each an artifact of
an *earlier* pipeline stage (cohort construction, overlap diagnostics,
selective-prediction evaluation) and belong to that stage's own artifact
output, not to a "regenerate from saved predictions" reproduction command.
This script regenerates only the figures that consume plain saved
scalar/mapping data: `build_architecture_diagram()` (no data required),
`build_treatment_effect_vs_reference_plot()`, and
`build_ablation_bar_chart()`.

Locked-evaluation gating (Requirement 15.6)
------------------------------------------------

Requirement 15.6: "The reproduction command SHALL not require access to
the historical reference until the locked external-evaluation stage."
`--stage` defaults to `"reproduction"`, which never touches
`reference/internal_randomized_reference.sealed.json` and never imports
`cohorts.sealed_reference`.

`--stage locked-evaluation` is accepted by the parser (so the CLI surface
documents the gate rather than hiding it), but this script always refuses
it: unsealing the sealed reference requires a valid, minted
`cohorts.sealed_reference.LockedEvaluationToken`
(`infra.reference_guard.assert_reference_unlocked`), and the only intended
minting path is `scripts/check_go_no_go.py` (task 16), which does not
exist yet in this repository. This script deliberately does **not**
attempt to deserialize a `LockedEvaluationToken` from a caller-supplied
`--unlock-token-path` file: `LockedEvaluationToken` is a plain frozen
dataclass, and constructing one from arbitrary file contents would let any
caller forge a "valid" token, defeating the entire point of gating token
issuance behind `_mint()`. So today, `--stage locked-evaluation` always
exits non-zero with a clear message, before opening any file at all
(including `--unlock-token-path` itself and any
`*.sealed.json`-named path). This is the correct, intentional behavior --
not a bug -- until task 16's `check_go_no_go.py` exists to mint a real
token.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from protocols.schema import TargetTrialSpec
from reporting.figures import (
    build_ablation_bar_chart,
    build_architecture_diagram,
    build_treatment_effect_vs_reference_plot,
)
from reporting.model_card import build_model_card, to_markdown
from reporting.tables import build_main_results_table
from uncertainty.bootstrap import BootstrapCI

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default output directory (`design.md` Section 2.3's
#: `results/<experiment_id>/reporting/` convention, generalized here since
#: this command does not compute a real `experiment_id` from a resolved
#: config -- it only reads already-saved reporting inputs).
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "paper_reproduction" / "reporting"

#: `--stage` values this CLI accepts. `"reproduction"` is the only one that
#: actually runs; `"locked-evaluation"` is always refused (see module
#: docstring, "Locked-evaluation gating").
_STAGE_CHOICES = ("reproduction", "locked-evaluation")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"top-level content of {path} must be a mapping")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"top-level content of {path} must be a JSON object")
    return data


def _redact_path_for_provenance(path: Path) -> str:
    """Record a provenance-safe representation of `path`: relative to
    `REPO_ROOT` when possible, or just the filename otherwise -- never the
    raw absolute path.

    `provenance.json` is a reporting artifact meant to leave protected
    storage (`docs/data_access.md` Section 6's permitted-derived-artifacts
    list includes "environment/version metadata"; an absolute developer-
    machine path is not on that list). An absolute path such as
    `/Users/<username>/...` leaks the local username into a public
    artifact and trips `scripts/scan_exports.py`'s unauthorized-filesystem-
    path check (`src/infra/privacy_heuristics.py`'s
    `_HOME_DIR_PATH_REGEX`) -- the exact same heuristic
    `infra.manifest.PrivacyScanner` already enforces for manifests
    (Requirement 1.11, 2.2-2.4, 2.7). This function makes `run_paper.py`'s
    reporting output satisfy that same rule structurally, rather than
    relying on every caller to pass an already-relative path.

    Args:
        path: an absolute or relative filesystem path (e.g. `--trial` or
            `--experiment`'s resolved value).

    Returns:
        `str(path.relative_to(REPO_ROOT))` if `path` is under `REPO_ROOT`;
        otherwise `path.name` (just the filename, with no directory
        component at all, so nothing about the caller's machine/directory
        layout is recorded when the input lives outside the repository,
        e.g. a test's `tmp_path`).
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _refuse_locked_evaluation() -> int:
    """Refuse `--stage locked-evaluation` before touching any file.

    See module docstring ("Locked-evaluation gating"). This function opens
    nothing -- not `--trial`, not `--experiment`, not `--unlock-token-path`,
    and certainly not any `reference/internal_randomized_reference.sealed.json`-named
    path -- the refusal happens purely from the value of `--stage` itself.
    """
    print(
        "run_paper.py: refusing --stage locked-evaluation. Unsealing the "
        "internal randomized reference requires a valid, minted "
        "LockedEvaluationToken (cohorts.sealed_reference.LockedEvaluationToken), "
        "and the only intended minting path is scripts/check_go_no_go.py "
        "(task 16), which does not exist yet in this repository. This command "
        "does not attempt to deserialize or construct a token from any "
        "caller-supplied file, and has not opened reference/"
        "internal_randomized_reference.sealed.json or any file at all. "
        "The locked-evaluation stage is intentionally unreachable via this "
        "CLI until task 16 exists -- this is by design, not a bug.",
        file=sys.stderr,
    )
    return 1


def run_reproduction(
    trial_path: Path, experiment_path: Path, output_dir: Path
) -> int:
    """Regenerate the main results table, in-scope figures, and model card
    from already-saved prediction/metric data (Requirement 15.5).

    Returns 0 on success. On any failure, prints a clear message to stderr
    and returns a non-zero exit code without raising.
    """
    try:
        trial_raw = _load_yaml(trial_path)
    except ValueError as exc:
        print(f"run_paper.py failed: could not load --trial: {exc}", file=sys.stderr)
        return 1
    try:
        trial_spec = TargetTrialSpec.model_validate(trial_raw)
    except ValidationError as exc:
        print(
            f"run_paper.py failed: --trial {trial_path} is not a valid TargetTrialSpec:\n{exc}",
            file=sys.stderr,
        )
        return 1

    try:
        experiment_data = _load_json(experiment_path)
    except ValueError as exc:
        print(f"run_paper.py failed: could not load --experiment: {exc}", file=sys.stderr)
        return 1

    if "results_table_rows" not in experiment_data:
        print(
            "run_paper.py failed: --experiment JSON is missing required key "
            "'results_table_rows' (see scripts/run_paper.py's module docstring "
            "for the documented schema).",
            file=sys.stderr,
        )
        return 1

    try:
        results_table = build_main_results_table(experiment_data["results_table_rows"])
    except ValueError as exc:
        print(f"run_paper.py failed: invalid results_table_rows: {exc}", file=sys.stderr)
        return 1

    figures: dict[str, Any] = {"architecture_diagram": build_architecture_diagram()}

    estimate_data = experiment_data.get("treatment_effect_estimate")
    reference_value = experiment_data.get("reference_value")
    if estimate_data is not None and reference_value is not None:
        try:
            estimate = BootstrapCI(**estimate_data)
        except TypeError as exc:
            print(
                f"run_paper.py failed: invalid treatment_effect_estimate: {exc}",
                file=sys.stderr,
            )
            return 1
        figures["treatment_effect_vs_reference_plot"] = build_treatment_effect_vs_reference_plot(
            estimate,
            reference_value,
            reference_label=experiment_data.get("reference_label", "Internal reference"),
        )

    ablation_results = experiment_data.get("ablation_results")
    if ablation_results is not None:
        try:
            figures["ablation_bar_chart"] = build_ablation_bar_chart(ablation_results)
        except ValueError as exc:
            print(f"run_paper.py failed: invalid ablation_results: {exc}", file=sys.stderr)
            return 1

    model_card = build_model_card()
    model_card_markdown = to_markdown(model_card)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (tables_dir / "main_results_table.json").write_text(
        json.dumps(results_table, indent=2, sort_keys=True), encoding="utf-8"
    )
    for figure_name, figure_data in figures.items():
        (figures_dir / f"{figure_name}.json").write_text(
            json.dumps(figure_data, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output_dir / "model_card.json").write_text(
        json.dumps(model_card, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "model_card.md").write_text(model_card_markdown, encoding="utf-8")
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "trial_spec_version": trial_spec.version,
                "trial_path": _redact_path_for_provenance(trial_path),
                "experiment_path": _redact_path_for_provenance(experiment_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"run_paper.py succeeded: reporting artifacts written to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate paper tables, in-scope figures, and the model card from "
            "already-saved predictions/metrics (Requirement 15.5). Does not "
            "re-run the causal/model pipeline."
        )
    )
    parser.add_argument(
        "--trial",
        type=Path,
        required=True,
        help="Path to a target-trial-spec YAML (e.g. configs/target_trials/main.yaml).",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help=(
            "Path to a saved predictions/metrics JSON file (see this module's "
            "docstring for the documented schema)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Where to write regenerated tables/figures/model card "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=_STAGE_CHOICES,
        default="reproduction",
        help=(
            "'reproduction' (default) regenerates artifacts from saved data only. "
            "'locked-evaluation' is always refused today -- see module docstring."
        ),
    )
    parser.add_argument(
        "--unlock-token-path",
        type=Path,
        default=None,
        help=(
            "Reserved for the locked-evaluation stage once scripts/check_go_no_go.py "
            "(task 16) exists to mint a real LockedEvaluationToken. Currently unused: "
            "--stage locked-evaluation is always refused before any file, including "
            "this one, is opened."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Refusal must happen first, before any file access at all (Requirement
    # 15.6 / task 15.4's explicit refusal-ordering requirement).
    if args.stage == "locked-evaluation":
        return _refuse_locked_evaluation()

    return run_reproduction(args.trial, args.experiment, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
