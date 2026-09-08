"""Minimum-cell-size enforcement for public reporting artifacts.

Requirement 2.5 requires public tables and figures to use aggregate results
and enforce the dataset's minimum-cell-size policy: no public artifact may
disclose a row, subgroup, or bucket whose underlying patient count falls
below the configured threshold, because a sufficiently small count is itself
re-identifying (`docs/data_access.md` Section 6, "May leave protected
storage" / "Must remain in protected storage"). Requirement 2.6 additionally
requires that patient-level predictions remain in protected storage; the
suppression/aggregation this module performs is what keeps every row this
module *emits* at or above `min_cell_size`, never at the individual-patient
level.

This module is the single implementation of the minimum-cell-size rule.
`src/reporting/tables.py`'s `apply_min_cell_size()` and
`src/reporting/figures.py`'s `apply_min_cell_size_to_figure_data()` both call
into `suppress_small_cells()` here rather than reimplementing the threshold
check, so the two call sites can never drift out of sync — the same pattern
already used for `src/infra/privacy_heuristics.py`, shared by the manifest
privacy scanner and the export scanner.

Scope note: `decisions.md` does not freeze a specific numeric
`min_cell_size` threshold for ACTG175 reporting (Requirement 2.5 only
requires that *a* configured threshold be enforced); the threshold itself is
supplied by the caller (e.g. from the dataset contract or experiment config
once task 15 wires this up), not hardcoded here.

This module intentionally operates on plain `list[dict]` rows rather than
`pandas.DataFrame`. `pyproject.toml` has no table/dataframe library as a
project dependency yet, and this task does not need one — introducing
`pandas` here, ahead of the full table-building implementation, would add an
unnecessary dependency for a self-contained policy check. Task 15
(`build_main_results_table()`) may adopt `pandas.DataFrame` for the table
representation if warranted by the rest of that module's needs; if it does,
it can convert to/from `list[dict]` at the boundary and continue to reuse
`suppress_small_cells()` unchanged.
"""

from __future__ import annotations

from typing import Any

#: A row is a plain mapping of column name -> value, matching how the rest
#: of this scaffold (and, later, `build_main_results_table()`) represents
#: one reportable table row.
Row = dict[str, Any]

#: Default replacement value used when a small cell's fields are masked
#: rather than aggregated (see `aggregate_label_field` below).
DEFAULT_SUPPRESSION_VALUE = "suppressed"

#: Default label used for the pooled "other" bucket when small cells are
#: aggregated rather than masked in place.
DEFAULT_AGGREGATE_LABEL = "other (suppressed)"

#: Sentinel field added to every row this function produces from a
#: below-threshold input row (whether masked or aggregated), so downstream
#: consumers can identify which rows were affected without re-deriving
#: counts themselves.
SUPPRESSED_FLAG_FIELD = "suppressed"


def suppress_small_cells(
    table_rows: list[Row],
    count_field: str,
    min_cell_size: int,
    suppression_value: Any = DEFAULT_SUPPRESSION_VALUE,
    aggregate_label_field: str | None = None,
    aggregate_label: str = DEFAULT_AGGREGATE_LABEL,
) -> list[Row]:
    """Suppress or aggregate any row whose count is below `min_cell_size`.

    Rows with `row[count_field] >= min_cell_size` are returned unchanged
    (copied, not mutated in place). Rows with a count below `min_cell_size`
    (or a missing/`None` count) are handled one of two ways:

    - **Aggregation** (`aggregate_label_field` is given): every below-threshold
      row is pooled into a single output row labelled `aggregate_label` under
      `aggregate_label_field`. Numeric fields (other than the label field)
      are summed across the pooled rows so the underlying totals are not
      lost; every other field is replaced with `suppression_value`, since it
      cannot be meaningfully combined across rows. The pooled row is tagged
      `suppressed=True`.
    - **Masking** (`aggregate_label_field` is omitted, the default): each
      below-threshold row is kept in the output — so its presence is not
      silently dropped from the table — but every field's value is replaced
      with `suppression_value`, and the row is tagged `suppressed=True`.

    Args:
        table_rows: rows to check, each a mapping containing at least
            `count_field`.
        count_field: the key in each row holding the cell's underlying
            patient/record count.
        min_cell_size: the minimum count a cell must have to be disclosed
            as-is. Must be non-negative.
        suppression_value: the placeholder written into fields that cannot
            be disclosed (either masked fields, in the masking mode, or
            non-numeric fields of the pooled row, in the aggregation mode).
        aggregate_label_field: if given, enables aggregation mode and names
            the field that receives `aggregate_label` in the pooled row. If
            omitted, masking mode is used instead.
        aggregate_label: the label written into `aggregate_label_field` for
            the pooled "other" row, when aggregation mode is used.

    Returns:
        A new list of rows with the minimum-cell-size policy applied. Row
        order among kept (at-or-above-threshold) rows is preserved; any
        aggregated/masked rows follow them.

    Raises:
        ValueError: if `min_cell_size` is negative.
        KeyError: if a row is missing `count_field`.
    """
    if min_cell_size < 0:
        raise ValueError(f"min_cell_size must be >= 0, got {min_cell_size}")

    kept_rows: list[Row] = []
    small_rows: list[Row] = []
    for row in table_rows:
        if count_field not in row:
            raise KeyError(f"Row is missing required count_field '{count_field}': {row!r}")
        count = row[count_field]
        if count is None or count < min_cell_size:
            small_rows.append(row)
        else:
            kept_rows.append(dict(row))

    if not small_rows:
        return kept_rows

    if aggregate_label_field is not None:
        kept_rows.append(
            _aggregate_small_rows(
                small_rows,
                aggregate_label_field=aggregate_label_field,
                aggregate_label=aggregate_label,
                suppression_value=suppression_value,
            )
        )
    else:
        for row in small_rows:
            kept_rows.append(_mask_row(row, suppression_value=suppression_value))

    return kept_rows


def _is_summable_numeric(value: Any) -> bool:
    # bool is a subclass of int in Python; treat it as non-numeric here so
    # boolean flag columns are masked rather than nonsensically summed.
    return isinstance(value, int | float) and not isinstance(value, bool)


def _mask_row(row: Row, suppression_value: Any) -> Row:
    masked: Row = {key: suppression_value for key in row}
    masked[SUPPRESSED_FLAG_FIELD] = True
    return masked


def _aggregate_small_rows(
    small_rows: list[Row],
    aggregate_label_field: str,
    aggregate_label: str,
    suppression_value: Any,
) -> Row:
    all_fields = {key for row in small_rows for key in row}
    numeric_fields = {
        key
        for key in all_fields
        if key != aggregate_label_field
        and all(_is_summable_numeric(row.get(key)) for row in small_rows if key in row)
        and any(key in row for row in small_rows)
    }

    aggregated: Row = {aggregate_label_field: aggregate_label}
    for field in numeric_fields:
        aggregated[field] = sum(row.get(field, 0) or 0 for row in small_rows)
    for field in all_fields - numeric_fields - {aggregate_label_field}:
        aggregated[field] = suppression_value
    aggregated[SUPPRESSED_FLAG_FIELD] = True
    return aggregated
