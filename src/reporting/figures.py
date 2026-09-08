"""Principal paper figures (task 15.1, Requirements 15.2, 15.3).

No plotting library is a dependency of this project (`pyproject.toml`
lists only `pydantic`/`PyYAML`; there is deliberately no `matplotlib` or
equivalent). Adding one is a real dependency decision out of scope for
this task. Every figure builder below therefore follows the exact
precedent already set by `cohorts.cohort_flow.build_cohort_flow_table()`
(task 6.6): it returns a flat, JSON-serializable data structure that a
future task (once a plotting library is approved) can render, plus —
where the figure is fundamentally a flowchart/diagram — a Mermaid
`flowchart TD` source string. Figures that are fundamentally a data plot
(scatter/line/bar) instead return plain structured plot data (parallel
lists such as `{"x": [...], "y": [...]}`), the natural shape a plotting
call would consume. No image-rendering, no pixels, no plotting-library
imports anywhere in this module — this is the same deliberate
simplification already used for `cohort_flow.py`'s diagram, applied
consistently to all six principal figures (Requirement 15.2):

1. `build_architecture_diagram` — static Mermaid flowchart, no dynamic data.
2. `build_cohort_flow_diagram` — thin wrapper re-exporting task 6.6's
   `cohort_flow.build_cohort_flow_table()` unchanged.
3. `build_treatment_effect_vs_reference_plot` — structured plot data
   (point estimate + CI vs. a reference value).
4. `build_propensity_overlap_plot` — structured plot data (per-arm
   propensity summary stats).
5. `build_risk_coverage_curve_plot` — structured plot data (coverage/risk
   parallel lists).
6. `build_ablation_bar_chart` — structured plot data (bar labels/values).

Plus `apply_min_cell_size_to_figure_data()` (task 2.2, Requirement
2.5-2.6), unchanged, that any figure-data builder operating on
subgroup-level rows must call before the data is plotted or exported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cohorts.cohort_flow import build_cohort_flow_table
from data.timelines import PatientTimeline as DataPatientTimeline
from evaluation.ablation_matrix import ABLATIONS
from evaluation.metrics import risk_coverage_curve
from protocols.schema import TargetTrialSpec
from reporting.min_cell_size import Row, suppress_small_cells
from uncertainty.bootstrap import BootstrapCI

__all__ = [
    "apply_min_cell_size_to_figure_data",
    "build_architecture_diagram",
    "build_cohort_flow_diagram",
    "build_treatment_effect_vs_reference_plot",
    "build_propensity_overlap_plot",
    "build_risk_coverage_curve_plot",
    "build_ablation_bar_chart",
]


def apply_min_cell_size_to_figure_data(
    figure_data: list[Row],
    min_cell_size: int,
    count_field: str = "n",
    aggregate_label_field: str | None = None,
    suppression_value: Any = "suppressed",
) -> list[Row]:
    """Enforce the minimum-cell-size policy on figure data before plotting.

    Any figure that aggregates over subgroups (e.g. a subgroup forest plot,
    a per-stratum overlap plot) must not plot a point/bar/series drawn from
    fewer than `min_cell_size` patients (Requirement 2.5). This wraps the
    same shared implementation used by `reporting.tables.apply_min_cell_size()`
    (`reporting.min_cell_size.suppress_small_cells()`), so table and figure
    enforcement can never drift apart.

    Args:
        figure_data: rows describing one plotted unit each (e.g. one
            subgroup or bucket), each containing at least `count_field`.
        min_cell_size: the configured minimum reportable cell size.
        count_field: the column holding each row's underlying count.
        aggregate_label_field: if given, below-threshold rows are pooled
            into a single labelled "other" row instead of being masked in
            place. See `suppress_small_cells()` for the exact semantics.
        suppression_value: placeholder written into fields that cannot be
            disclosed.

    Returns:
        A new list of rows safe to pass to a plotting routine.
    """
    return suppress_small_cells(
        figure_data,
        count_field=count_field,
        min_cell_size=min_cell_size,
        suppression_value=suppression_value,
        aggregate_label_field=aggregate_label_field,
    )


def build_architecture_diagram() -> dict[str, Any]:
    """Figure 1: system architecture (Requirement 15.2).

    Static Mermaid `flowchart TD` source describing the high-level
    architecture already built in `models.intervention_aware`
    (`StateEncoder`, `InterventionEncoder`, `ProtocolEncoder` ->
    concatenation -> `GRUTransition`) and `models.branch_heads`
    (`branch_1`/`branch_0` heads) / `models.aggregation`
    (`estimate_ate`/`estimate_cate`). No dynamic data is involved, so
    unlike the other five figures this always returns the same content.

    Returns:
        `{"mermaid": "flowchart TD\\n..."}`.
    """
    lines = [
        "flowchart TD",
        '    S["StateEncoder<br/>(baseline covariates)"]',
        '    I["InterventionEncoder<br/>(dose/route/frequency/adherence)"]',
        '    P["ProtocolEncoder<br/>(protocol features)"]',
        '    Z["Concatenation<br/>z_t = [z_state; z_intervention; z_protocol]"]',
        '    G["GRUTransition<br/>(multi-step rollout: baseline -> week-20 -> week-96)"]',
        '    B1["branch_1 head<br/>(event_hazard, next_cd4, next_cd8)"]',
        '    B0["branch_0 head<br/>(event_hazard, next_cd4, next_cd8)"]',
        '    A["Aggregation<br/>(estimate_ate / estimate_cate)"]',
        "    S --> Z",
        "    I --> Z",
        "    P --> Z",
        "    Z --> G",
        "    G --> B1",
        "    G --> B0",
        "    B1 --> A",
        "    B0 --> A",
    ]
    return {"mermaid": "\n".join(lines)}


def build_cohort_flow_diagram(
    timelines: Sequence[DataPatientTimeline] | Mapping[str, DataPatientTimeline],
    treatment_arms: Mapping[str, int],
    spec: TargetTrialSpec,
) -> dict[str, Any]:
    """Figure 2: cohort flow (Requirement 15.2, 15.3; reuses task 6.6).

    Thin wrapper re-exporting `cohorts.cohort_flow.build_cohort_flow_table()`
    unchanged -- that function already implements the cohort-flow diagram
    (initial population -> eligibility exclusions -> arm assignment ->
    censoring -> analyzed population, Requirement 15.3) as JSON plus
    Mermaid source. No reimplementation here.

    Returns:
        The unmodified `{"flow_table": {...}, "mermaid": "..."}` from
        `build_cohort_flow_table()`.
    """
    return build_cohort_flow_table(timelines, treatment_arms, spec)


def build_treatment_effect_vs_reference_plot(
    model_estimate: BootstrapCI,
    reference_value: float,
    reference_label: str = "Internal reference",
) -> dict[str, Any]:
    """Figure 3: treatment-effect-vs-reference comparison (Requirement 15.2).

    Structured plot data for a point-estimate-with-CI vs. a single
    reference value (e.g. the model's estimated treatment effect against
    `uncertainty.bootstrap.patient_level_bootstrap`'s internal-reference
    point estimate) -- the shape a bar/point-with-errorbar plot consumes.

    Args:
        model_estimate: the model's bootstrap point estimate + CI.
        reference_value: the single scalar reference value to compare
            against (e.g. the internal reference's point estimate).
        reference_label: a human-readable label for the reference series.

    Returns:
        `{"model_point_estimate": ..., "model_ci_low": ..., "model_ci_high": ...,
        "reference_value": ..., "reference_label": ...}`.
    """
    return {
        "model_point_estimate": model_estimate.point_estimate,
        "model_ci_low": model_estimate.ci_low,
        "model_ci_high": model_estimate.ci_high,
        "reference_value": reference_value,
        "reference_label": reference_label,
    }


def build_propensity_overlap_plot(overlap_report: Any) -> dict[str, Any]:
    """Figure 4: propensity overlap (Requirement 15.2).

    Structured plot data extracting per-arm propensity summary stats from
    `causal.overlap.OverlapReport.arm_propensity_summary` (each arm's `n`,
    `minimum`, `maximum`, `mean`, `median`) into a plottable structure
    (e.g. for an overlapping-histogram or box-plot-style rendering).

    Args:
        overlap_report: a `causal.overlap.OverlapReport` (typed `Any` here
            to avoid importing `causal.overlap`, mirroring `overlap.py`'s
            own documented decoupling convention -- see that module's
            docstring, "Interface / decoupling from `causal.propensity`").

    Returns:
        `{"arms": {arm: {"n": ..., "minimum": ..., "maximum": ..., "mean": ...,
        "median": ...}, ...}}`.
    """
    arms: dict[Any, dict[str, float]] = {}
    for arm, summary in overlap_report.arm_propensity_summary.items():
        arms[arm] = {
            "n": summary.n,
            "minimum": summary.minimum,
            "maximum": summary.maximum,
            "mean": summary.mean,
            "median": summary.median,
        }
    return {"arms": arms}


def build_risk_coverage_curve_plot(
    predictions: Sequence[float],
    labels: Sequence[bool],
    confidence_scores: Sequence[float],
) -> dict[str, Any]:
    """Figure 5: selective-risk/discrepancy (risk-coverage) curve (Requirement 15.2).

    Calls `evaluation.metrics.risk_coverage_curve()` (task 10.5) and
    unzips its `(coverage, risk)` pairs into two parallel lists -- the
    natural shape for a line-plot call.

    Returns:
        `{"coverage": [...], "risk": [...]}`, in increasing-coverage
        order. Both lists are empty if there are no patients.
    """
    curve = risk_coverage_curve(predictions, labels, confidence_scores)
    coverage = [point[0] for point in curve]
    risk = [point[1] for point in curve]
    return {"coverage": coverage, "risk": risk}


def build_ablation_bar_chart(ablation_results: Mapping[str, float]) -> dict[str, Any]:
    """Figure 6: principal ablations bar chart (Requirement 15.2).

    Structured plot data (`{"labels": [...], "values": [...]}`) from a
    `{ablation_name: metric_value}` mapping (e.g. per-ablation concordance
    from task 12.1's harness output, `scripts/run_ablations.py`).

    Labels are validated against `evaluation.ablation_matrix.ABLATIONS`'
    frozen names: any name in `ablation_results` that is not one of the
    ten frozen ablation names raises `ValueError`, since a silently
    accepted typo'd/renamed ablation would be indistinguishable from a
    real result in the rendered chart. `ablation_results` is not required
    to cover all ten ablations (a partial run is still plottable);
    ordering follows `ablation_results`' insertion order.

    Args:
        ablation_results: `{ablation_name: metric_value}`.

    Returns:
        `{"labels": [...], "values": [...]}`, parallel lists in the same
        order as `ablation_results`.

    Raises:
        ValueError: if any key in `ablation_results` is not one of
            `ABLATIONS`' frozen names.
    """
    known_names = {ablation.name for ablation in ABLATIONS}
    unknown_names = [name for name in ablation_results if name not in known_names]
    if unknown_names:
        raise ValueError(
            f"unknown ablation name(s) not in ABLATIONS: {unknown_names!r}"
        )
    return {
        "labels": list(ablation_results.keys()),
        "values": list(ablation_results.values()),
    }
