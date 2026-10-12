"""Public results tables.

Provides `build_main_results_table()` (Requirement 15.1), the thin
aggregation/presentation layer over the four frozen metric groups defined in
`decisions.md` Section 18 (factual prediction, causal recovery, selective
prediction, consistency); `build_headline_comparisons_table()` (task 22.1,
Requirement 17.1-17.7), the named-entries table for the five headline
experimental comparisons; and `apply_min_cell_size()` (task 2.2, Requirement
2.5-2.6) that any table reporting a subgroup or aggregated cell must call
before being written anywhere outside protected storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evaluation.statistics import PRIMARY_COMPARISONS, holm_correction
from reporting.min_cell_size import Row, suppress_small_cells

__all__ = [
    "apply_min_cell_size",
    "build_blocked_outcome_report",
    "build_executed_results_report",
    "build_headline_comparisons_table",
    "build_main_results_table",
    "build_negative_result_report",
    "FROZEN_VALID_NEGATIVE_RESULT",
    "HEADLINE_COMPARISON_NAMES",
    "REQUIRED_COLUMNS",
]


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


#: The five headline experimental comparisons (Requirement 17.1-17.5;
#: `design.md` Section 3.15). The first three are exactly
#: `evaluation.statistics.PRIMARY_COMPARISONS` -- the frozen Holm family
#: (`decisions.md` Section 19) -- reusing those names directly rather than
#: inventing a second, parallel naming scheme for the same three
#: comparisons. The fourth (intervention encoder vs. the capacity-matched
#: placebo channel, Requirement 17.4) and fifth (the complete model vs.
#: both the no-protocol and no-treatment-input variants, Requirement
#: 17.5) are reported without correction and labeled as such (Requirement
#: 17.6). The fifth is one named table entry bundling two sub-comparisons
#: (`vs_no_protocol` and `vs_no_treatment_input`) under a single
#: uncorrected row, per task 22.1's "report the last two" (i.e. exactly
#: two non-frozen named entries, the second of which carries two values).
HEADLINE_COMPARISON_NAMES: tuple[str, ...] = (
    *PRIMARY_COMPARISONS,
    "intervention_encoder_vs_capacity_matched_placebo",
    "complete_vs_no_protocol_and_no_treatment_input",
)

#: Required per-comparison metadata for every entry of
#: `build_headline_comparisons_table()`'s input, besides the point
#: estimate/interval itself (Requirement 17.7: "report the evaluated
#: split, the seed set, and the interval alongside every point
#: difference").
_HEADLINE_COMPARISON_METADATA_KEYS: tuple[str, ...] = ("split", "seed_set")

#: Required point-estimate/interval keys for a simple (non-compound)
#: headline-comparison entry.
_HEADLINE_COMPARISON_VALUE_KEYS: tuple[str, ...] = ("point_estimate", "ci_low", "ci_high")

#: The compound fifth entry's two sub-comparison keys (see
#: `HEADLINE_COMPARISON_NAMES`'s docstring comment above).
_NO_PROTOCOL_AND_NO_TREATMENT_INPUT_SUB_KEYS: tuple[str, ...] = (
    "vs_no_protocol",
    "vs_no_treatment_input",
)

#: The compound fifth entry's own name, singled out so
#: `build_headline_comparisons_table()` can apply its different (nested)
#: value shape without a positional index into `HEADLINE_COMPARISON_NAMES`.
_COMPOUND_COMPARISON_NAME = "complete_vs_no_protocol_and_no_treatment_input"


def build_headline_comparisons_table(
    comparisons: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Assemble the headline-comparisons table from already-computed
    per-comparison results (Requirement 17.1-17.7; task 22.1).

    Like `build_main_results_table()`, this is a thin presentation layer:
    it validates the input and reapplies
    `evaluation.statistics.holm_correction()` to the three frozen primary
    comparisons' already-computed raw p-values (Requirement 17.6) rather
    than computing any paired-bootstrap statistic itself -- that is
    `evaluation.statistics.paired_bootstrap_test`/`paired_bootstrap_p_value`'s
    job, run by the caller (e.g. `scripts/run_locked_evaluation.py`) over
    the identical test patients (Requirement 17.6's "paired bootstrap
    comparison on identical patients").

    Args:
        comparisons: `{comparison_name: entry}`, keyed by exactly
            `HEADLINE_COMPARISON_NAMES`. For every name except
            `"complete_vs_no_protocol_and_no_treatment_input"`, `entry`
            must contain `"split"`, `"seed_set"`, `"point_estimate"`,
            `"ci_low"`, `"ci_high"`, and -- for the three names in
            `evaluation.statistics.PRIMARY_COMPARISONS` only -- a raw
            `"p_value"` to Holm-correct. The compound fifth entry must
            instead contain `"split"`, `"seed_set"`, and a nested
            `"sub_comparisons"` dict with exactly the two keys
            `"vs_no_protocol"` and `"vs_no_treatment_input"`, each itself
            a `{"point_estimate", "ci_low", "ci_high"}` mapping.

    Returns:
        One row per name in `HEADLINE_COMPARISON_NAMES`, in that order.
        Every row carries `"comparison"`, `"split"`, `"seed_set"`, and
        `"corrected"` (`True` for the three frozen comparisons, `False`
        otherwise). Frozen rows additionally carry `"raw_p_value"` and
        `"holm_adjusted_p_value"`; non-frozen simple rows carry
        `"correction_label": "uncorrected"` alongside their point
        estimate/interval; the compound row carries `"correction_label":
        "uncorrected"` and a `"sub_comparisons"` dict passed through
        unchanged.

    Raises:
        ValueError: if `comparisons`'s key set is not exactly
            `HEADLINE_COMPARISON_NAMES`, if any entry is missing a
            required key, or if the compound entry's `sub_comparisons`
            does not have exactly the two documented sub-keys.
    """
    missing = sorted(set(HEADLINE_COMPARISON_NAMES) - set(comparisons))
    extra = sorted(set(comparisons) - set(HEADLINE_COMPARISON_NAMES))
    if missing or extra:
        raise ValueError(
            f"comparisons must cover exactly the five headline comparisons "
            f"{HEADLINE_COMPARISON_NAMES!r}; missing={missing!r}, extra={extra!r}"
        )

    for name, entry in comparisons.items():
        missing_metadata = [key for key in _HEADLINE_COMPARISON_METADATA_KEYS if key not in entry]
        if missing_metadata:
            raise ValueError(
                f"headline comparison {name!r} is missing required metadata "
                f"key(s): {', '.join(missing_metadata)}"
            )
        if name == _COMPOUND_COMPARISON_NAME:
            if "sub_comparisons" not in entry:
                raise ValueError(
                    f"headline comparison {name!r} is missing required key: "
                    f"sub_comparisons"
                )
            sub = entry["sub_comparisons"]
            sub_missing = sorted(set(_NO_PROTOCOL_AND_NO_TREATMENT_INPUT_SUB_KEYS) - set(sub))
            sub_extra = sorted(set(sub) - set(_NO_PROTOCOL_AND_NO_TREATMENT_INPUT_SUB_KEYS))
            if sub_missing or sub_extra:
                raise ValueError(
                    f"headline comparison {name!r}'s sub_comparisons must cover "
                    f"exactly {_NO_PROTOCOL_AND_NO_TREATMENT_INPUT_SUB_KEYS!r}; "
                    f"missing={sub_missing!r}, extra={sub_extra!r}"
                )
            for sub_name, sub_entry in sub.items():
                sub_missing_values = [
                    key for key in _HEADLINE_COMPARISON_VALUE_KEYS if key not in sub_entry
                ]
                if sub_missing_values:
                    raise ValueError(
                        f"headline comparison {name!r}'s sub-comparison "
                        f"{sub_name!r} is missing required key(s): "
                        f"{', '.join(sub_missing_values)}"
                    )
        else:
            missing_values = [key for key in _HEADLINE_COMPARISON_VALUE_KEYS if key not in entry]
            if missing_values:
                raise ValueError(
                    f"headline comparison {name!r} is missing required key(s): "
                    f"{', '.join(missing_values)}"
                )
            if name in PRIMARY_COMPARISONS and "p_value" not in entry:
                raise ValueError(
                    f"headline comparison {name!r} is one of the frozen primary "
                    f"comparisons and must carry a raw 'p_value' key to Holm-correct"
                )

    raw_p_values = {name: comparisons[name]["p_value"] for name in PRIMARY_COMPARISONS}
    holm_adjusted = holm_correction(raw_p_values)

    rows: list[dict[str, Any]] = []
    for name in HEADLINE_COMPARISON_NAMES:
        entry = comparisons[name]
        row: dict[str, Any] = {
            "comparison": name,
            "split": entry["split"],
            "seed_set": entry["seed_set"],
            "corrected": name in PRIMARY_COMPARISONS,
        }
        if name in PRIMARY_COMPARISONS:
            row["point_estimate"] = entry["point_estimate"]
            row["ci_low"] = entry["ci_low"]
            row["ci_high"] = entry["ci_high"]
            row["raw_p_value"] = entry["p_value"]
            row["holm_adjusted_p_value"] = holm_adjusted[name]
        elif name == _COMPOUND_COMPARISON_NAME:
            row["correction_label"] = "uncorrected"
            row["sub_comparisons"] = entry["sub_comparisons"]
        else:
            row["point_estimate"] = entry["point_estimate"]
            row["ci_low"] = entry["ci_low"]
            row["ci_high"] = entry["ci_high"]
            row["correction_label"] = "uncorrected"
        rows.append(row)
    return rows


#: The frozen valid negative result (`decisions.md` Section 21), quoted
#: verbatim -- do not paraphrase, reorder, or edit (task 22.2, Requirement
#: 19.2). Mirrors `src/reporting/model_card.py::PERMITTED_CLAIMS`/
#: `PROHIBITED_USES`'s own "copied verbatim" convention for frozen
#: `decisions.md` text.
FROZEN_VALID_NEGATIVE_RESULT: str = (
    "Better factual trajectory prediction did not produce better "
    "randomized treatment-effect recovery, while overlap-aware "
    "abstention prevented unsupported causal claims."
)


def build_blocked_outcome_report(
    released: bool,
    failing_conditions: Sequence[str],
    supported_count: int,
    abstained_count: int,
) -> dict[str, Any]:
    """Report a gate-blocked primary estimate honestly (Requirement 19.1).

    A thin presentation layer, like every other function in this module:
    `released`/`failing_conditions` are already-computed
    (`evaluation.release_gate.ReleaseDecision.released`/`.reasons`, or an
    equivalent already-serialized summary such as
    `scripts/run_causal_analysis.py`'s `release_decision_reasons`), and
    this function only reshapes them into a reportable dict -- it does
    not re-evaluate the gate, and it deliberately takes the two plain
    values rather than the `ReleaseDecision` object itself so a caller
    holding only an already-serialized summary (no live dataclass
    instance) can still call it.

    Args:
        released: the release gate's own `released` boolean.
        failing_conditions: every failing condition named
            (`ReleaseDecision.reasons`, or its serialized equivalent);
            empty when `released` is `True`.
        supported_count: the supported (retained) population count for
            the blocked estimate.
        abstained_count: the abstained (not retained) population count
            for the blocked estimate.

    Returns:
        A dict with `"blocked": True`, every failing condition named in
        `"failing_conditions"` (unchanged), and the two population
        counts.

    Raises:
        ValueError: if `released` is `True` -- this function reports a
            blocked outcome specifically; a passing decision has nothing
            to block-report.
    """
    if released:
        raise ValueError(
            "build_blocked_outcome_report requires a blocked (released=False) "
            "decision; this decision passed."
        )
    return {
        "blocked": True,
        "failing_conditions": list(failing_conditions),
        "supported_count": supported_count,
        "abstained_count": abstained_count,
    }


def build_negative_result_report(
    factual_improved: bool, recovery_improved: bool
) -> dict[str, Any]:
    """Report the frozen valid negative result when factual performance
    improves without a matching improvement in treatment-effect recovery
    (Requirement 19.2; `decisions.md` Section 21).

    Args:
        factual_improved: whether factual (predictive) performance
            improved for the configuration being reported (e.g. the
            intervention-aware model vs. the baseline GRU).
        recovery_improved: whether treatment-effect recovery improved for
            the same comparison.

    Returns:
        A dict with `"dissociation_observed"` (`True` only when
        `factual_improved and not recovery_improved`) and, only in that
        case, `"frozen_negative_result"` set to
        `FROZEN_VALID_NEGATIVE_RESULT` verbatim. When the dissociation did
        not occur, the dict carries `"dissociation_observed": False` and
        no frozen-result text -- this function never emits
        `FROZEN_VALID_NEGATIVE_RESULT` for a case it does not describe.
    """
    dissociation_observed = factual_improved and not recovery_improved
    report: dict[str, Any] = {"dissociation_observed": dissociation_observed}
    if dissociation_observed:
        report["frozen_negative_result"] = FROZEN_VALID_NEGATIVE_RESULT
    return report


def build_executed_results_report(
    arms: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Report every executed ablation arm, sensitivity variant, and seed,
    including arms whose results contradict the intervention-aware
    model's advantage (Requirement 19.3).

    A thin pass-through, like `build_main_results_table`: this function
    does not decide which arms "contradict" anything -- that judgment
    (e.g. an ablation arm whose point estimate favors the baseline over
    the intervention-aware model) is the caller's, since it depends on
    comparing each arm's value against the primary estimate's direction,
    which varies per run. This function's only job is to guarantee every
    supplied arm is reported, unfiltered -- there is no code path here
    that drops, hides, or reorders an entry, which is the Requirement
    19.3 property under test (a caller cannot silently filter out
    contradicting arms by passing fewer entries without that showing up
    as a shorter `arms` output here, since this function changes nothing
    about `len`).

    Args:
        arms: one dict per executed ablation arm, sensitivity variant, or
            seed result -- each caller-assembled, with at least a `"name"`
            key naming the arm/variant/seed; a `"contradicts_advantage"`
            key is optional and, if present, passed through unchanged
            rather than inspected.

    Returns:
        `{"arms": arms_as_given, "n_arms": len(arms)}` -- every arm
        passed through unchanged and in the given order.
    """
    arms_list = list(arms)
    return {"arms": arms_list, "n_arms": len(arms_list)}


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
