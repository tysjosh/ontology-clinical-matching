"""Unit tests for `src/reporting/tables.py`.

**Validates: Requirements 15.1**
"""

import pytest

from reporting.tables import REQUIRED_COLUMNS, build_main_results_table


def _synthetic_row(**overrides) -> dict:
    """A plausible, fully-populated main-results-table row."""
    row = {
        # Factual prediction.
        "concordance_index": 0.71,
        "auroc_96wk": 0.68,
        "auprc_96wk": 0.42,
        "ipcw_brier_score": 0.11,
        "calibration_slope": 0.95,
        "calibration_intercept": 0.02,
        "cd4_cd8_mae": 0.08,
        # Causal recovery.
        "estimated_hr": 0.55,
        "e_log_hr": 0.10,
        "effect_direction_agreement": True,
        "ci_overlap": True,
        "risk_difference_error": 0.03,
        # Selective prediction.
        "retained_coverage": 0.80,
        "selective_risk": 0.09,
        "aurcc": 0.06,
        "incorrect_release_rate": 0.02,
        "unnecessary_abstention_rate": 0.05,
        "error_at_matched_coverage": 0.07,
        # Consistency.
        "temporal_violations": 0,
        "protocol_violations": 1,
        "impossible_transition_rate": 0.0,
        "witness_detected_failures": 2,
    }
    row.update(overrides)
    return row


def test_all_required_columns_present_in_output():
    row = _synthetic_row(model="baseline_gru", n=340)

    result = build_main_results_table([row])

    assert len(result) == 1
    for column in REQUIRED_COLUMNS:
        assert column in result[0], f"missing column: {column}"
    # Extra, non-schema columns pass through untouched.
    assert result[0]["model"] == "baseline_gru"
    assert result[0]["n"] == 340


def test_multiple_rows_pass_through_in_order():
    rows = [
        _synthetic_row(model="baseline_gru"),
        _synthetic_row(model="intervention_aware", estimated_hr=0.60),
    ]

    result = build_main_results_table(rows)

    assert [row["model"] for row in result] == ["baseline_gru", "intervention_aware"]
    for row in result:
        for column in REQUIRED_COLUMNS:
            assert column in row


def test_missing_required_column_raises_value_error_naming_it():
    row = _synthetic_row()
    del row["ipcw_brier_score"]

    with pytest.raises(ValueError, match="ipcw_brier_score"):
        build_main_results_table([row])


def test_missing_multiple_required_columns_names_all_of_them():
    row = _synthetic_row()
    del row["aurcc"]
    del row["temporal_violations"]

    with pytest.raises(ValueError) as excinfo:
        build_main_results_table([row])

    assert "aurcc" in str(excinfo.value)
    assert "temporal_violations" in str(excinfo.value)
