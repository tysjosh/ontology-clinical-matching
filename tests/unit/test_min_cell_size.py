"""Unit tests for `src/reporting/min_cell_size.py` and its `tables.py` /
`figures.py` wrappers.

**Validates: Requirements 2.5, 2.6**
"""

from __future__ import annotations

import pytest

from reporting.figures import apply_min_cell_size_to_figure_data
from reporting.min_cell_size import suppress_small_cells
from reporting.tables import apply_min_cell_size


def _sample_table() -> list[dict]:
    return [
        {"subgroup": "prior_art_naive", "n": 150, "hr": 0.55},
        {"subgroup": "prior_art_experienced", "n": 120, "hr": 0.48},
        {"subgroup": "rare_subgroup_x", "n": 4, "hr": 0.30},
    ]


class TestSuppressSmallCells:
    def test_small_subgroup_is_masked_by_default(self):
        table = _sample_table()

        result = suppress_small_cells(table, count_field="n", min_cell_size=100)

        # Large subgroups pass through unchanged.
        naive = next(r for r in result if r.get("subgroup") == "prior_art_naive")
        assert naive["n"] == 150
        assert naive["hr"] == 0.55

        # The below-threshold row is retained (not silently dropped) but
        # every field is masked and flagged.
        small = [r for r in result if r.get("suppressed") is True]
        assert len(small) == 1
        assert small[0]["subgroup"] == "suppressed"
        assert small[0]["n"] == "suppressed"
        assert small[0]["hr"] == "suppressed"

    def test_small_subgroup_is_aggregated_when_requested(self):
        table = _sample_table()

        result = suppress_small_cells(
            table,
            count_field="n",
            min_cell_size=100,
            aggregate_label_field="subgroup",
            aggregate_label="other (suppressed)",
        )

        labels = {row["subgroup"] for row in result}
        assert "rare_subgroup_x" not in labels
        assert "other (suppressed)" in labels

        pooled = next(r for r in result if r["subgroup"] == "other (suppressed)")
        assert pooled["n"] == 4  # numeric field summed across pooled rows
        assert pooled["suppressed"] is True

    def test_no_rows_below_threshold_returns_table_unchanged(self):
        table = [{"subgroup": "a", "n": 200}, {"subgroup": "b", "n": 150}]

        result = suppress_small_cells(table, count_field="n", min_cell_size=100)

        assert result == table

    def test_missing_count_field_raises_key_error(self):
        table = [{"subgroup": "a"}]

        with pytest.raises(KeyError):
            suppress_small_cells(table, count_field="n", min_cell_size=100)

    def test_negative_min_cell_size_raises_value_error(self):
        with pytest.raises(ValueError):
            suppress_small_cells([], count_field="n", min_cell_size=-1)

    def test_none_count_is_treated_as_below_threshold(self):
        table = [{"subgroup": "a", "n": None}]

        result = suppress_small_cells(table, count_field="n", min_cell_size=100)

        assert result[0]["suppressed"] is True


class TestTablesWrapper:
    def test_apply_min_cell_size_suppresses_small_subgroup_in_generated_table(self):
        table = _sample_table()

        result = apply_min_cell_size(table, min_cell_size=100)

        kept_subgroups = {row["subgroup"] for row in result if not row.get("suppressed")}
        assert kept_subgroups == {"prior_art_naive", "prior_art_experienced"}

        suppressed_rows = [row for row in result if row.get("suppressed")]
        assert len(suppressed_rows) == 1
        assert suppressed_rows[0]["subgroup"] == "suppressed"


class TestFiguresWrapper:
    def test_apply_min_cell_size_to_figure_data_suppresses_small_subgroup(self):
        figure_data = _sample_table()

        result = apply_min_cell_size_to_figure_data(figure_data, min_cell_size=100)

        suppressed_rows = [row for row in result if row.get("suppressed")]
        assert len(suppressed_rows) == 1
        assert suppressed_rows[0]["hr"] == "suppressed"
