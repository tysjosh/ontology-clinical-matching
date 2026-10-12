"""Unit tests for `src/reporting/tables.py`.

**Validates: Requirements 15.1, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7**
"""

import pytest

from evaluation.statistics import PRIMARY_COMPARISONS
from reporting.tables import (
    FROZEN_VALID_NEGATIVE_RESULT,
    HEADLINE_COMPARISON_NAMES,
    REQUIRED_COLUMNS,
    build_blocked_outcome_report,
    build_executed_results_report,
    build_headline_comparisons_table,
    build_main_results_table,
    build_negative_result_report,
)


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


# ---------------------------------------------------------------------------
# build_headline_comparisons_table (task 22.1; Requirement 17.1-17.7)
# ---------------------------------------------------------------------------


def _simple_comparison_entry(**overrides) -> dict:
    entry = {
        "split": "test",
        "seed_set": (101, 202, 303, 404, 505),
        "point_estimate": 0.12,
        "ci_low": 0.02,
        "ci_high": 0.22,
    }
    entry.update(overrides)
    return entry


def _full_comparisons_payload() -> dict:
    comparisons = {
        name: _simple_comparison_entry(p_value=0.01 * (i + 1))
        for i, name in enumerate(PRIMARY_COMPARISONS)
    }
    comparisons["intervention_encoder_vs_capacity_matched_placebo"] = _simple_comparison_entry()
    comparisons["complete_vs_no_protocol_and_no_treatment_input"] = {
        "split": "test",
        "seed_set": (101, 202, 303, 404, 505),
        "sub_comparisons": {
            "vs_no_protocol": _simple_comparison_entry(),
            "vs_no_treatment_input": _simple_comparison_entry(),
        },
    }
    return comparisons


def test_all_five_headline_comparisons_present_in_output():
    result = build_headline_comparisons_table(_full_comparisons_payload())

    names = [row["comparison"] for row in result]
    assert names == list(HEADLINE_COMPARISON_NAMES)
    assert len(result) == 5


def test_only_the_three_frozen_comparisons_are_corrected():
    result = build_headline_comparisons_table(_full_comparisons_payload())

    corrected_names = {row["comparison"] for row in result if row["corrected"]}
    uncorrected_names = {row["comparison"] for row in result if not row["corrected"]}

    assert corrected_names == set(PRIMARY_COMPARISONS)
    assert uncorrected_names == {
        "intervention_encoder_vs_capacity_matched_placebo",
        "complete_vs_no_protocol_and_no_treatment_input",
    }
    for row in result:
        if row["corrected"]:
            assert "holm_adjusted_p_value" in row
            assert "raw_p_value" in row
        else:
            assert row["correction_label"] == "uncorrected"
            assert "holm_adjusted_p_value" not in row


def test_split_seed_set_and_interval_populated_for_every_comparison():
    result = build_headline_comparisons_table(_full_comparisons_payload())

    for row in result:
        assert row["split"] == "test"
        assert row["seed_set"] == (101, 202, 303, 404, 505)
        if row["comparison"] == "complete_vs_no_protocol_and_no_treatment_input":
            for sub in row["sub_comparisons"].values():
                assert {"point_estimate", "ci_low", "ci_high"} <= set(sub)
        else:
            assert {"point_estimate", "ci_low", "ci_high"} <= set(row)


def test_missing_comparison_name_raises_value_error():
    comparisons = _full_comparisons_payload()
    del comparisons["gated_vs_ungated"]

    with pytest.raises(ValueError, match="gated_vs_ungated"):
        build_headline_comparisons_table(comparisons)


def test_unknown_comparison_name_raises_value_error():
    comparisons = _full_comparisons_payload()
    comparisons["not_a_real_comparison"] = _simple_comparison_entry()

    with pytest.raises(ValueError, match="not_a_real_comparison"):
        build_headline_comparisons_table(comparisons)


def test_frozen_comparison_missing_p_value_raises_value_error():
    comparisons = _full_comparisons_payload()
    del comparisons["baseline_vs_intervention_aware"]["p_value"]

    with pytest.raises(ValueError, match="p_value"):
        build_headline_comparisons_table(comparisons)


def test_compound_entry_missing_a_sub_comparison_raises_value_error():
    comparisons = _full_comparisons_payload()
    del comparisons["complete_vs_no_protocol_and_no_treatment_input"]["sub_comparisons"][
        "vs_no_protocol"
    ]

    with pytest.raises(ValueError, match="vs_no_protocol"):
        build_headline_comparisons_table(comparisons)


def test_holm_correction_is_actually_applied_to_the_frozen_three():
    # p-values: baseline_vs_intervention_aware=0.01, gated_vs_ungated=0.03,
    # intervention_aware_vs_ipcw_aipw=0.04 -- same hand-computed example as
    # tests/unit/test_statistics.py's holm_correction test.
    comparisons = _full_comparisons_payload()
    comparisons["baseline_vs_intervention_aware"]["p_value"] = 0.01
    comparisons["intervention_aware_vs_ipcw_aipw"]["p_value"] = 0.04
    comparisons["gated_vs_ungated"]["p_value"] = 0.03

    result = build_headline_comparisons_table(comparisons)
    by_name = {row["comparison"]: row for row in result}

    assert by_name["baseline_vs_intervention_aware"]["holm_adjusted_p_value"] == pytest.approx(
        0.03
    )
    assert by_name["gated_vs_ungated"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert by_name["intervention_aware_vs_ipcw_aipw"]["holm_adjusted_p_value"] == pytest.approx(
        0.06
    )


# ---------------------------------------------------------------------------
# build_blocked_outcome_report / build_negative_result_report /
# build_executed_results_report (task 22.2; Requirement 19.1, 19.2, 19.3)
# ---------------------------------------------------------------------------

def test_blocked_outcome_report_carries_every_failing_condition_and_both_counts():
    report = build_blocked_outcome_report(
        released=False,
        failing_conditions=["overlap_valid: fail", "causal_pass: fail (status='uncertain')"],
        supported_count=187,
        abstained_count=23,
    )

    assert report["blocked"] is True
    assert report["failing_conditions"] == [
        "overlap_valid: fail",
        "causal_pass: fail (status='uncertain')",
    ]
    assert report["supported_count"] == 187
    assert report["abstained_count"] == 23


def test_blocked_outcome_report_rejects_a_passing_decision():
    with pytest.raises(ValueError):
        build_blocked_outcome_report(
            released=True, failing_conditions=[], supported_count=10, abstained_count=0
        )


def test_negative_result_report_emits_frozen_text_only_for_the_dissociation_case():
    dissociated = build_negative_result_report(factual_improved=True, recovery_improved=False)
    assert dissociated["dissociation_observed"] is True
    assert dissociated["frozen_negative_result"] == FROZEN_VALID_NEGATIVE_RESULT

    not_dissociated_cases = [
        (True, True),
        (False, False),
        (False, True),
    ]
    for factual_improved, recovery_improved in not_dissociated_cases:
        report = build_negative_result_report(factual_improved, recovery_improved)
        assert report["dissociation_observed"] is False
        assert "frozen_negative_result" not in report


def test_negative_result_text_matches_decisions_md_section_21_verbatim():
    assert FROZEN_VALID_NEGATIVE_RESULT == (
        "Better factual trajectory prediction did not produce better "
        "randomized treatment-effect recovery, while overlap-aware "
        "abstention prevented unsupported causal claims."
    )


def test_executed_results_report_includes_every_arm_including_contradicting_ones():
    arms = [
        {"name": "no_treatment_input", "estimate": 0.05, "contradicts_advantage": False},
        {"name": "placebo_intervention_channel", "estimate": -0.02, "contradicts_advantage": True},
        {"name": "seed_303", "estimate": 0.01, "contradicts_advantage": True},
    ]

    report = build_executed_results_report(arms)

    assert report["n_arms"] == 3
    assert report["arms"] == arms
    contradicting = [arm for arm in report["arms"] if arm.get("contradicts_advantage")]
    assert len(contradicting) == 2


def test_executed_results_report_preserves_order_and_count_for_empty_input():
    report = build_executed_results_report([])
    assert report == {"arms": [], "n_arms": 0}
