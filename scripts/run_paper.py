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

Locked-evaluation gating (Requirement 15.6; task 22, Requirement 13.12)
------------------------------------------------------------------------

Requirement 15.6: "The reproduction command SHALL not require access to
the historical reference until the locked external-evaluation stage."
`--stage` defaults to `"reproduction"`, which never touches
`reference/internal_randomized_reference.sealed.json` and never imports
`cohorts.sealed_reference`.

`--stage locked-evaluation` used to always refuse (back when
`scripts/check_go_no_go.py`, task 16, did not exist). Task 22 changes
that refusal to a delegation: `--stage locked-evaluation` now calls
`run_locked_evaluation.main(remaining_argv)` -- `scripts/run_locked_evaluation.py`'s
own CLI entry point (task 21) -- and propagates its exit status unchanged
(Requirement 13.12). `remaining_argv` is whatever argv this script's own
parser did not recognize (via `parse_known_args`), so a caller invokes
this stage with exactly `run_locked_evaluation.py`'s own flags
(`--token-artifact`, `--results-root`, `--resolved-config-hash`,
`--dataset-checksum`, `--split-hash`, `--checkpoint`) rather than a
second, parallel flag set this script would have to keep in sync.
`--trial`/`--experiment` (required for `--stage reproduction`) are not
required for `--stage locked-evaluation`: this script does not itself
open any file for that stage -- `run_locked_evaluation.main()` does its
own argument parsing and file access entirely on its own terms, including
its own rejected-option check (Requirement 13.2), which runs before this
delegation does anything beyond identifying the stage.

This script emits no message stating that `check_go_no_go.py` is absent
(Requirement 13.12's explicit prohibition) -- that message existed only
because task 16 had not landed yet; it no longer applies, and the
delegation path never constructs or inspects any such message.
`_redact_path_for_provenance` is unaffected by this change and continues
to apply to every path this script itself records in `provenance.json`
for the `reproduction` stage.
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

# ---------------------------------------------------------------------------
# Task 23.2 (Requirement 20.6, 20.7): generate every LaTeX-reported number
# and table from the same saved artifacts `run_reproduction` above already
# reads, never hand-typed into `paper/*.tex`. See
# `build_manuscript_artifacts` below for the entry point and
# `_SUPPRESSED_STATUS_TEXT` for the suppression-rendering rule.
# ---------------------------------------------------------------------------

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default output directory (`design.md` Section 2.3's
#: `results/<experiment_id>/reporting/` convention, generalized here since
#: this command does not compute a real `experiment_id` from a resolved
#: config -- it only reads already-saved reporting inputs).
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "paper_reproduction" / "reporting"

#: Default manuscript-generated-artifacts directory (task 23.2;
#: Requirement 20.6) -- the fixed `paper/generated/` path task 23.1's
#: skeleton already names as "written by run_paper.py from saved
#: artifacts, never hand-edited". Distinct from `DEFAULT_OUTPUT_DIR`
#: (the `results/.../reporting/` tree): this is the manuscript package's
#: own write target, consumed by `paper/main.tex`'s `\input`s.
DEFAULT_MANUSCRIPT_DIR = REPO_ROOT / "paper" / "generated"

#: Rendered in place of a numeric value, in both `numbers.tex` and the
#: generated tables, for any row whose `estimate_status` key equals
#: `"suppressed"` (Requirement 20.7; the same `estimate_status` field
#: `witnesses.audit_report.AuditReport` already carries).
_SUPPRESSED_STATUS_TEXT = "suppressed"

#: `--stage` values this CLI accepts. `"reproduction"` regenerates
#: artifacts from already-saved data; `"locked-evaluation"` delegates to
#: `scripts/run_locked_evaluation.py`'s own CLI (see module docstring,
#: "Locked-evaluation gating").
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


def _delegate_locked_evaluation(remaining_argv: list[str]) -> int:
    """Delegate `--stage locked-evaluation` to
    `scripts/run_locked_evaluation.py`'s own CLI entry point (task 21) and
    propagate its exit status unchanged (Requirement 13.12).

    See module docstring ("Locked-evaluation gating"). This function opens
    no file itself -- it imports `run_locked_evaluation` (a sibling module
    in this same `scripts/` directory) and hands it `remaining_argv`
    verbatim; every file access, including `run_locked_evaluation`'s own
    rejected-option check (Requirement 13.2), happens entirely inside that
    module's own `main()`, on that module's own terms.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import run_locked_evaluation

    return run_locked_evaluation.main(remaining_argv)


def run_reproduction(
    trial_path: Path,
    experiment_path: Path,
    output_dir: Path,
    manuscript_dir: Path = DEFAULT_MANUSCRIPT_DIR,
) -> int:
    """Regenerate the main results table, in-scope figures, model card,
    and manuscript-generated artifacts from already-saved
    prediction/metric data (Requirement 15.5, 20.6, 20.7).

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

    # Task 23.2 (Requirement 20.6, 20.7): write every manuscript-generated
    # artifact (numbers.tex, the four generated tables, and the figure
    # PDFs) from the exact same already-saved experiment data and
    # already-built `figures` dict this run already validated and used
    # above -- never a second, independently-read copy.
    build_manuscript_artifacts(experiment_data, results_table, manuscript_dir, figures)

    print(f"run_paper.py succeeded: reporting artifacts written to {output_dir}")
    return 0


def _latex_escape(text: str) -> str:
    """Escape the handful of LaTeX-special characters that can appear in
    a caller-supplied label/column name, so a generated `.tex` file is
    never broken by an unescaped `_`, `%`, `&`, `#`, or `$`."""
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _latex_macro_name(*parts: str) -> str:
    """Build a valid LaTeX `\\newcommand` macro name (letters only, no
    digits or underscores -- `\\newcommand` names must be purely
    alphabetic) from one or more caller-supplied identifier parts.

    Digits are spelled out (`0`->`zero`, ..., `9`->`nine`) rather than
    dropped, so two macro names that would otherwise only differ by a
    digit (e.g. row index `0` vs `1`) stay distinct.
    """
    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }
    joined = "".join(parts)
    out = []
    for char in joined:
        if char.isalpha():
            out.append(char)
        elif char in digit_words:
            out.append(digit_words[char].capitalize())
        # every other character (., -, space, etc.) is dropped, not
        # substituted, matching LaTeX's own macro-name restriction.
    return "paperNum" + "".join(out)


def _render_value_or_suppressed(row: dict[str, Any], column: str) -> str:
    """`str(row[column])`, or `_SUPPRESSED_STATUS_TEXT` if `row` carries
    `estimate_status == "suppressed"` (Requirement 20.7)."""
    if row.get("estimate_status") == "suppressed":
        return _SUPPRESSED_STATUS_TEXT
    value = row.get(column)
    if isinstance(value, float) and value != value:  # NaN
        return "n/a"
    return str(value)


def write_numbers_tex(
    results_table: list[dict[str, Any]], manuscript_dir: Path
) -> Path:
    """Write `paper/generated/numbers.tex`: one `\\newcommand` per
    reported number in `results_table` (Requirement 20.6).

    Every row's `label` (or its index, if no `label` key is present) is
    combined with every one of `reporting.tables.REQUIRED_COLUMNS` to
    build one macro name per cell -- e.g. row `label="complete"`'s
    `concordance_index` becomes `\\paperNumCompleteConcordanceindex`.
    Non-required, caller-supplied extra columns (e.g. `"n"`) are included
    too, so nothing in a results-table row is left untraceable to a
    macro. A row carrying `estimate_status == "suppressed"` renders every
    one of its macros as the literal text `"suppressed"` rather than a
    fabricated or stale number (Requirement 20.7).

    Returns the written path.
    """
    from reporting.tables import REQUIRED_COLUMNS

    manuscript_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/run_paper.py from saved result artifacts.",
        "% Never hand-edited (Requirement 20.6).",
    ]
    for row_index, row in enumerate(results_table):
        row_label = str(row.get("label", row_index))
        columns = list(dict.fromkeys((*REQUIRED_COLUMNS, *row.keys())))
        for column in columns:
            if column in ("label", "estimate_status"):
                continue
            macro_name = _latex_macro_name(row_label, column)
            rendered = _render_value_or_suppressed(row, column)
            lines.append(f"\\newcommand{{\\{macro_name}}}{{{_latex_escape(rendered)}}}")

    target = manuscript_dir / "numbers.tex"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _write_latex_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    caption: str,
    label: str,
    target: Path,
) -> Path:
    """Write one generated LaTeX `table` environment from already-built
    rows -- the shared body `write_manuscript_tables` calls once per
    table (Requirement 20.6: tables drawn from saved artifacts, never
    hand-edited)."""
    header = " & ".join(_latex_escape(column) for column in columns)
    body_lines = []
    for row in rows:
        cells = [_render_value_or_suppressed(row, column) for column in columns]
        body_lines.append(" & ".join(_latex_escape(cell) for cell in cells) + r" \\")

    lines = [
        "% Generated by scripts/run_paper.py from saved result artifacts.",
        "% Never hand-edited (Requirement 20.6).",
        r"\begin{table}[t]",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\begin{center}",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_manuscript_tables(
    experiment_data: dict[str, Any], results_table: list[dict[str, Any]], manuscript_dir: Path
) -> dict[str, Path]:
    """Write `table_main_results.tex`, `table_ablations.tex`,
    `table_sensitivity.tex`, and `table_replication.tex` from already-
    saved artifacts (Requirement 20.6).

    `table_main_results.tex` is always written, from `results_table`
    (every row, every `reporting.tables.REQUIRED_COLUMNS` column).
    `table_ablations.tex`/`table_sensitivity.tex`/`table_replication.tex`
    are written only when `experiment_data` carries the corresponding
    optional key (`"ablation_results"`/`"sensitivity_variant_outcomes"`/
    `"replication_summary"`); when absent, a placeholder table stating
    the data is not yet available is written instead, so `\\input`-ing
    the file from `main.tex` never fails with a missing-file error.

    Returns `{table_name: written_path}`.
    """
    from reporting.tables import REQUIRED_COLUMNS

    written: dict[str, Path] = {}

    main_results_columns = ["label", *REQUIRED_COLUMNS]
    written["table_main_results"] = _write_latex_table(
        results_table,
        [c for c in main_results_columns if any(c in row for row in results_table)] or ["label"],
        "Main results",
        "tab:main_results",
        manuscript_dir / "table_main_results.tex",
    )

    ablation_results = experiment_data.get("ablation_results")
    if ablation_results:
        ablation_rows = [
            {"label": name, "value": value} for name, value in ablation_results.items()
        ]
        written["table_ablations"] = _write_latex_table(
            ablation_rows,
            ["label", "value"],
            "Ablation results",
            "tab:ablations",
            manuscript_dir / "table_ablations.tex",
        )
    else:
        written["table_ablations"] = _write_placeholder_table(
            manuscript_dir / "table_ablations.tex", "Ablation results", "tab:ablations"
        )

    sensitivity_outcomes = experiment_data.get("sensitivity_variant_outcomes") or (
        experiment_data.get("causal_analysis_summary", {}).get("sensitivity_variant_outcomes")
        if isinstance(experiment_data.get("causal_analysis_summary"), dict)
        else None
    )
    if sensitivity_outcomes:
        sensitivity_rows = [
            {
                "label": outcome.get("name", str(i)),
                "status": outcome.get("status"),
                "estimate": outcome.get("estimate"),
            }
            for i, outcome in enumerate(sensitivity_outcomes)
        ]
        written["table_sensitivity"] = _write_latex_table(
            sensitivity_rows,
            ["label", "status", "estimate"],
            "Sensitivity-variant results",
            "tab:sensitivity",
            manuscript_dir / "table_sensitivity.tex",
        )
    else:
        written["table_sensitivity"] = _write_placeholder_table(
            manuscript_dir / "table_sensitivity.tex",
            "Sensitivity-variant results",
            "tab:sensitivity",
        )

    replication_summary = experiment_data.get("replication_summary")
    if replication_summary:
        replication_rows = [
            {"label": str(result.get("seed", i)), "estimate": result.get("primary_estimate")}
            for i, result in enumerate(replication_summary.get("seed_results", []))
        ]
        written["table_replication"] = _write_latex_table(
            replication_rows,
            ["label", "estimate"],
            "Cross-seed replication results",
            "tab:replication",
            manuscript_dir / "table_replication.tex",
        )
    else:
        written["table_replication"] = _write_placeholder_table(
            manuscript_dir / "table_replication.tex",
            "Cross-seed replication results",
            "tab:replication",
        )

    return written


def _write_placeholder_table(target: Path, caption: str, label: str) -> Path:
    """A generated (never hand-edited) placeholder table body, written
    when the corresponding saved artifact is not yet available -- keeps
    `main.tex`'s `\\input` resolvable without a missing-file build error,
    while never fabricating a number."""
    lines = [
        "% Generated by scripts/run_paper.py: no saved artifact was",
        "% available for this table yet. Never hand-edited.",
        r"\begin{table}[t]",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\begin{center}",
        "(not yet available)",
        r"\end{center}",
        r"\end{table}",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _write_standalone_tikz_pdf(tikz_body: str, target_tex: Path) -> Path | None:
    """Compile one `standalone`-class TikZ/pgfplots document to a PDF via
    `tectonic` (Requirement 20.6's "figure PDFs"), writing the `.tex`
    source alongside the compiled PDF either way.

    Returns the written PDF path, or `None` (never raising) if `tectonic`
    is not on `PATH` -- a missing LaTeX engine is an environment
    precondition, not a reason to fail the whole reproduction run; the
    `.tex` source is still written so the figure is reproducible once an
    engine is available.
    """
    import shutil
    import subprocess

    target_tex.parent.mkdir(parents=True, exist_ok=True)
    document = (
        "% Generated by scripts/run_paper.py from saved result artifacts.\n"
        "% Never hand-edited (Requirement 20.6).\n"
        "\\documentclass{standalone}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\begin{document}\n"
        f"{tikz_body}\n"
        "\\end{document}\n"
    )
    target_tex.write_text(document, encoding="utf-8")

    if shutil.which("tectonic") is None:
        return None
    try:
        subprocess.run(
            ["tectonic", "--keep-logs", target_tex.name],
            cwd=target_tex.parent,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    pdf_path = target_tex.with_suffix(".pdf")
    return pdf_path if pdf_path.is_file() else None


def _tikz_bar_chart(labels: list[str], values: list[float], ylabel: str) -> str:
    coordinates = " ".join(
        f"({_latex_escape(label)},{value})" for label, value in zip(labels, values, strict=True)
    )
    symbolic_coords = ",".join(_latex_escape(label) for label in labels)
    return (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[ybar, symbolic x coords={" + symbolic_coords + "}, "
        "xtick=data, x tick label style={rotate=45,anchor=east}, "
        f"ylabel={{{_latex_escape(ylabel)}}}]\n"
        f"\\addplot coordinates {{{coordinates}}};\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}"
    )


def write_manuscript_figures(figures: dict[str, Any], manuscript_dir: Path) -> dict[str, Path]:
    """Write figure PDFs from already-built `reporting.figures` output
    (task 23.2; Requirement 20.6).

    Args:
        figures: the same `figures` dict `run_reproduction` already
            builds via `reporting.figures.*` and writes as JSON -- this
            function renders a PDF rendering of each figure it knows how
            to, from that identical already-computed data, never
            recomputing anything.
        manuscript_dir: `paper/generated/`.

    Returns:
        `{figure_name: written_pdf_path}` for every figure a PDF was
        successfully compiled for (a figure this function does not know
        how to render, or a `tectonic`-unavailable/failed compile, is
        simply absent from the returned mapping rather than raising).
    """
    written: dict[str, Path] = {}

    ablation_bar_chart = figures.get("ablation_bar_chart")
    if ablation_bar_chart and "labels" in ablation_bar_chart and "values" in ablation_bar_chart:
        tikz_body = _tikz_bar_chart(
            ablation_bar_chart["labels"], ablation_bar_chart["values"], "value"
        )
        pdf_path = _write_standalone_tikz_pdf(
            tikz_body, manuscript_dir / "figure_ablation_bar_chart.tex"
        )
        if pdf_path is not None:
            written["figure_ablation_bar_chart"] = pdf_path

    treatment_effect_plot = figures.get("treatment_effect_vs_reference_plot")
    if treatment_effect_plot is not None:
        labels = ["model", "reference"]
        values = [
            treatment_effect_plot.get("model_point_estimate", 0.0),
            treatment_effect_plot.get("reference_value", 0.0),
        ]
        tikz_body = _tikz_bar_chart(labels, values, "point estimate")
        pdf_path = _write_standalone_tikz_pdf(
            tikz_body, manuscript_dir / "figure_treatment_effect_vs_reference_plot.tex"
        )
        if pdf_path is not None:
            written["figure_treatment_effect_vs_reference_plot"] = pdf_path

    return written


def build_manuscript_artifacts(
    experiment_data: dict[str, Any],
    results_table: list[dict[str, Any]],
    manuscript_dir: Path,
    figures: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write every manuscript-generated artifact (task 23.2;
    Requirement 20.6, 20.7): `numbers.tex`, the four generated tables,
    and (when `figures` is supplied and `tectonic` is available) the
    figure PDFs. Called by `run_reproduction` on every successful run.

    Returns `{artifact_name: written_path}`.
    """
    written = {"numbers": write_numbers_tex(results_table, manuscript_dir)}
    written.update(write_manuscript_tables(experiment_data, results_table, manuscript_dir))
    if figures is not None:
        written.update(write_manuscript_figures(figures, manuscript_dir))
    return written


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
        default=None,
        help=(
            "Path to a target-trial-spec YAML (e.g. configs/target_trials/main.yaml). "
            "Required for '--stage reproduction' (the default); unused and not "
            "required for '--stage locked-evaluation'."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help=(
            "Path to a saved predictions/metrics JSON file (see this module's "
            "docstring for the documented schema). Required for '--stage "
            "reproduction' (the default); unused and not required for "
            "'--stage locked-evaluation'."
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
        "--manuscript-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_DIR,
        help=(
            "Where to write generated manuscript artifacts (numbers.tex and "
            f"the four generated tables; task 23.2; default: {DEFAULT_MANUSCRIPT_DIR})."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=_STAGE_CHOICES,
        default="reproduction",
        help=(
            "'reproduction' (default) regenerates artifacts from saved data only. "
            "'locked-evaluation' delegates to scripts/run_locked_evaluation.py's "
            "own CLI -- see module docstring. Pass that script's own flags "
            "(--token-artifact, --results-root, --resolved-config-hash, "
            "--dataset-checksum, --split-hash, --checkpoint) after --stage "
            "locked-evaluation; this parser does not recognize them itself and "
            "forwards them unchanged."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining_argv = parser.parse_known_args(argv)

    # The stage check happens first, before this script opens any file at
    # all (Requirement 15.6 / task 15.4's explicit refusal-ordering
    # requirement, now a delegation-ordering requirement under task 22):
    # `--stage locked-evaluation` hands off immediately, without this
    # script itself ever touching --trial/--experiment.
    if args.stage == "locked-evaluation":
        return _delegate_locked_evaluation(remaining_argv)

    if args.trial is None or args.experiment is None:
        parser.error("--trial and --experiment are required for --stage reproduction")

    return run_reproduction(args.trial, args.experiment, args.output_dir, args.manuscript_dir)


if __name__ == "__main__":
    sys.exit(main())
