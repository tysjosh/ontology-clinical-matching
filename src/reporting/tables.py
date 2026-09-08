"""Public results tables.

Provides `build_main_results_table()` (Requirement 15.1), the thin
aggregation/presentation layer over the four frozen metric groups defined in
`decisions.md` Section 18 (factual prediction, causal recovery, selective
prediction, consistency), and `apply_min_cell_size()` (task 2.2, Requirement
2.5-2.6) that any table reporting a subgroup or aggregated cell must call
before being written anywhere outside protected storage.
"""

from __future__ import annotations

from typing import Any

from reporting.min_cell_size import Row, suppress_small_cells

__all__ = ["apply_min_cell_size", "build_main_results_table", "REQUIRED_COLUMNS"]


# Frozen column schema for the main results table (decisions.md Section 18).
# Each group below corresponds exactly to one of the four frozen metric
# groups. Column names are the documented, consistent names every caller
# must use when assembling a row of already-computed metric values.

# Factual prediction (6 columns).
_FACTUAL_COLUMNS: tuple[str, ...] = (
    "concordance_index",
    "auroc_96wk",
    "auprc_96wk",
    "ipcw_brier_score",
    "calibration_slope",
    "calibration_intercept",
    "cd4_cd8_mae",
)

# Causal recovery (5 columns).
_CAUSAL_RECOVERY_COLUMNS: tuple[str, ...] = (
    "estimated_hr",
    "e_log_hr",
    "effect_direction_agreement",
    "ci_overlap",
    "risk_difference_error",
)

# Selective prediction (6 columns). `risk_coverage_curve` itself is a list of
# points, not a scalar, and does not fit a flat table row cleanly -- it is
# reported as a companion figure (task 15.1's job), not a table column. Only
# its scalar summary, AURCC (area under the risk-coverage curve), is a
# frozen column here.
_SELECTIVE_PREDICTION_COLUMNS: tuple[str, ...] = (
    "retained_coverage",
    "selective_risk",
    "aurcc",
    "incorrect_release_rate",
    "unnecessary_abstention_rate",
    "error_at_matched_coverage",
)

# Consistency (4 columns).
_CONSISTENCY_COLUMNS: tuple[str, ...] = (
    "temporal_violations",
    "protocol_violations",
    "impossible_transition_rate",
    "witness_detected_failures",
)

#: Every required column across the four frozen metric groups. Single
#: source of truth for both `build_main_results_table()` and any test or
#: downstream consumer that needs to check a row for completeness.
REQUIRED_COLUMNS: tuple[str, ...] = (
    *_FACTUAL_COLUMNS,
    *_CAUSAL_RECOVERY_COLUMNS,
    *_SELECTIVE_PREDICTION_COLUMNS,
    *_CONSISTENCY_COLUMNS,
)


def build_main_results_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble the main results table from already-computed metric values.

    This is a thin aggregation/presentation layer, not a metric-computing
    function (Requirement 15.1). Each element of `rows` is one
    already-assembled per-model-or-config dict -- the caller is expected to
    have run the metric functions in `src/evaluation/metrics.py` and
    `src/witnesses/` and to pass their results in. This function only
    validates that every row carries the full frozen column schema and
    returns the rows unchanged, ready to be written to a CSV/reporting
    sink (after `apply_min_cell_size()`, if the row reports a subgroup or
    aggregated cell).

    Frozen column schema (decisions.md Section 18), by metric group:

    - Factual prediction: `concordance_index`, `auroc_96wk`, `auprc_96wk`,
      `ipcw_brier_score`, `calibration_slope`, `calibration_intercept`,
      `cd4_cd8_mae`.
    - Causal recovery: `estimated_hr`, `e_log_hr`,
      `effect_direction_agreement`, `ci_overlap`, `risk_difference_error`.
    - Selective prediction: `retained_coverage`, `selective_risk`, `aurcc`,
      `incorrect_release_rate`, `unnecessary_abstention_rate`,
      `error_at_matched_coverage`. The risk-coverage curve itself is a
      companion figure/artifact, not a table column -- only its scalar
      summary (`aurcc`) is included here.
    - Consistency: `temporal_violations`, `protocol_violations`,
      `impossible_transition_rate`, `witness_detected_failures`.

    Args:
        rows: one dict per model/config being compared, each containing at
            least every column in `REQUIRED_COLUMNS`. Extra columns (e.g.
            a row label or `n`) are passed through untouched.

    Returns:
        The same rows, validated and passed through unchanged, in the
        original order.

    Raises:
        ValueError: if any row is missing one or more required columns.
            The error names every missing column for that row.
    """
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(
                f"Row {index} is missing required column(s): {', '.join(missing)}"
            )
        validated.append(row)
    return validated


def apply_min_cell_size(
    table: list[Row],
    min_cell_size: int,
    count_field: str = "n",
    aggregate_label_field: str | None = None,
    suppression_value: Any = "suppressed",
) -> list[Row]:
    """Enforce the minimum-cell-size policy on a table before it is reported.

    Thin wrapper around `reporting.min_cell_size.suppress_small_cells()`
    (Requirement 2.5: "Public tables ... SHALL ... enforce the dataset's
    minimum-cell-size policy"). Kept as a separate, table-specific entry
    point so `build_main_results_table()` (task 15) has an obvious,
    documented call site rather than needing to know about the shared
    `min_cell_size` module directly.

    Args:
        table: rows produced by a table-building function, each containing
            at least `count_field`.
        min_cell_size: the configured minimum reportable cell size.
        count_field: the column holding each row's underlying count. Table
            builders that report per-subgroup counts should name this
            column consistently (e.g. `"n"`) across every table.
        aggregate_label_field: if given, below-threshold rows are pooled
            into a single labelled "other" row instead of being masked in
            place. See `suppress_small_cells()` for the exact semantics.
        suppression_value: placeholder written into fields that cannot be
            disclosed.

    Returns:
        A new list of rows with every cell at or above `min_cell_size`,
        or masked/aggregated in its place.
    """
    return suppress_small_cells(
        table,
        count_field=count_field,
        min_cell_size=min_cell_size,
        suppression_value=suppression_value,
        aggregate_label_field=aggregate_label_field,
    )
