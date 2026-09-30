"""Smoke tests for `src/reporting/figures.py` (task 15.1).

Each of the six principal-figure builders is called once on small
synthetic/dummy data and checked to run without error and return a
non-empty dict with the expected top-level keys. This mirrors the task's
own framing: a smoke test, not exhaustive correctness checking.

**Validates: Requirements 15.2, 15.3**
"""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from causal.overlap import ArmPropensitySummary, CommonSupportSummary, OverlapReport
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.schema import TargetTrialSpec
from reporting.figures import (
    build_ablation_bar_chart,
    build_architecture_diagram,
    build_cohort_flow_diagram,
    build_propensity_overlap_plot,
    build_risk_coverage_curve_plot,
    build_treatment_effect_vs_reference_plot,
)
from uncertainty.bootstrap import BootstrapCI

MAIN_SPEC_PATH = Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


def test_build_architecture_diagram_runs_without_error():
    result = build_architecture_diagram()

    assert result
    assert "mermaid" in result
    assert "flowchart" in result["mermaid"]


def test_build_cohort_flow_diagram_runs_without_error(target_trial_spec):
    event = ClinicalEvent(
        patient_id="p1",
        encounter_id=None,
        concept_id="hiv_positive",
        value=True,
        unit=None,
        modality="baseline",
        observed_at=datetime(2020, 1, 15, 9, 0, 0),
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    timelines = [build_patient_timeline([event])]
    treatment_arms = {"p1": 0}

    result = build_cohort_flow_diagram(timelines, treatment_arms, target_trial_spec)

    assert result
    assert "flow_table" in result
    assert "mermaid" in result


def test_build_treatment_effect_vs_reference_plot_runs_without_error():
    model_estimate = BootstrapCI(
        point_estimate=-0.05,
        ci_low=-0.10,
        ci_high=0.00,
        n_replicates=1000,
        n_successful=1000,
    )

    result = build_treatment_effect_vs_reference_plot(
        model_estimate, reference_value=-0.04, reference_label="Internal reference"
    )

    assert result
    assert set(result.keys()) == {
        "model_point_estimate",
        "model_ci_low",
        "model_ci_high",
        "reference_value",
        "reference_label",
    }


def test_build_propensity_overlap_plot_runs_without_error():
    overlap_report = OverlapReport(
        arm_counts={0: 100, 1: 100},
        arm_propensity_summary={
            0: ArmPropensitySummary(n=100, minimum=0.05, maximum=0.6, mean=0.3, median=0.3),
            1: ArmPropensitySummary(n=100, minimum=0.4, maximum=0.95, mean=0.7, median=0.7),
        },
        common_support=CommonSupportSummary(
            primary_count=180,
            primary_fraction=0.9,
            sensitivity_count=190,
            sensitivity_fraction=0.95,
        ),
        extreme_weight_count=20,
        extreme_weight_fraction=0.1,
        ess_by_arm={0: 90.0, 1: 90.0},
        smd_by_covariate={"age": 0.05},
        arm_counts_sufficient=True,
        ess_sufficient=True,
        smd_sufficient=True,
        no_unresolved_positivity_warning=False,
        support_valid=False,
    )

    result = build_propensity_overlap_plot(overlap_report)

    assert result
    assert "arms" in result
    assert set(result["arms"].keys()) == {0, 1}
    assert result["arms"][0]["n"] == 100


def test_build_risk_coverage_curve_plot_runs_without_error():
    predictions = [0.1, 0.4, 0.6, 0.9]
    labels = [False, False, True, True]
    confidence_scores = [0.9, 0.6, 0.7, 0.95]

    result = build_risk_coverage_curve_plot(predictions, labels, confidence_scores)

    assert result
    assert "coverage" in result
    assert "risk" in result
    assert len(result["coverage"]) == len(result["risk"]) == len(predictions)


def test_build_ablation_bar_chart_runs_without_error():
    ablation_results = {"no_treatment_input": 0.62, "treatment_as_ordinary_feature": 0.68}

    result = build_ablation_bar_chart(ablation_results)

    assert result
    assert result["labels"] == ["no_treatment_input", "treatment_as_ordinary_feature"]
    assert result["values"] == [0.62, 0.68]


def test_build_ablation_bar_chart_rejects_unknown_ablation_name():
    with pytest.raises(ValueError):
        build_ablation_bar_chart({"not_a_real_ablation": 0.5})
