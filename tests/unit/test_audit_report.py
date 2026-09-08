"""Unit tests for `src/witnesses/audit_report.py::build_audit_report`.

Constructs minimal-but-real instances of every input type (a loaded
`TargetTrialSpec`, a `BootstrapCI`, an `OverlapReport`, several
`WitnessResult`s, a `SupportDecision`, a `ProtocolViolation`, and a
provenance dict) and asserts every field required by Requirement 11.10
(protocol version, population, treatment, estimate and interval, overlap
diagnostics, witness outcomes, support restrictions/abstention reasons,
violations, and provenance) is present and non-empty in the resulting
`AuditReport` and its `.to_dict()` output.

**Validates: Requirements 11.10**
"""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from causal.overlap import ArmPropensitySummary, CommonSupportSummary, OverlapReport
from protocols.consistency_checker import ProtocolViolation
from protocols.schema import TargetTrialSpec
from uncertainty.bootstrap import BootstrapCI
from uncertainty.support_gate import SupportDecision
from witnesses.audit_report import AuditReport, build_audit_report
from witnesses.temporal import WitnessResult

MAIN_SPEC_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"
)


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


@pytest.fixture
def overlap_report() -> OverlapReport:
    return OverlapReport(
        arm_counts={0: 120, 1: 118},
        arm_propensity_summary={
            0: ArmPropensitySummary(n=120, minimum=0.12, maximum=0.87, mean=0.5, median=0.5),
            1: ArmPropensitySummary(n=118, minimum=0.11, maximum=0.88, mean=0.51, median=0.5),
        },
        common_support=CommonSupportSummary(
            primary_count=230, primary_fraction=0.96, sensitivity_count=235, sensitivity_fraction=0.99
        ),
        extreme_weight_count=8,
        extreme_weight_fraction=0.03,
        ess_by_arm={0: 110.0, 1: 108.0},
        smd_by_covariate={"age": 0.05, "baseline_cd4": 0.08},
        arm_counts_sufficient=True,
        ess_sufficient=True,
        smd_sufficient=True,
        no_unresolved_positivity_warning=True,
        support_valid=True,
    )


@pytest.fixture
def bootstrap_ci() -> BootstrapCI:
    return BootstrapCI(
        point_estimate=-0.12, ci_low=-0.20, ci_high=-0.04, n_replicates=1000, n_successful=1000
    )


@pytest.fixture
def witness_results() -> list[WitnessResult]:
    return [
        WitnessResult(witness="temporal", status="pass", evidence={"note": "ok"}),
        WitnessResult(witness="causal", status="pass", evidence={"note": "ok"}),
        WitnessResult(witness="clinical", status="pass", evidence={"note": "ok"}),
    ]


@pytest.fixture
def abstentions() -> list[SupportDecision]:
    return [
        SupportDecision(supported=True, reasons=["propensity_in_common_support: pass"]),
        SupportDecision(
            supported=False,
            reasons=[
                "propensity_in_common_support: fail (0.97 not in [0.1, 0.9])",
                "ensemble_uncertainty_ok: pass",
                "protocol_valid: pass",
                "no_hard_violation: pass",
            ],
        ),
    ]


@pytest.fixture
def violations() -> list[ProtocolViolation]:
    return [
        ProtocolViolation(
            patient_id="P001",
            rule="ineligible",
            timestamp=datetime(2020, 1, 10, 8, 0, 0),
            evidence={"reason": "baseline_cd4 out of range"},
        )
    ]


@pytest.fixture
def provenance() -> dict:
    return {
        "experiment_id": "abc123def4567890",
        "seed": 12345,
        "commit": "deadbeef",
        "timestamp": "2020-01-15T09:00:00",
    }


@pytest.fixture
def report(
    target_trial_spec, overlap_report, bootstrap_ci, witness_results, abstentions, violations, provenance
) -> AuditReport:
    return build_audit_report(
        protocol=target_trial_spec,
        population={"description": "ACTG175 eligible arm-0/arm-1 cohort", "n": 238},
        treatment={"intervention": "zidovudine+didanosine", "comparator": "zidovudine"},
        estimate=bootstrap_ci,
        overlap=overlap_report,
        witness_results=witness_results,
        abstentions=abstentions,
        violations=violations,
        provenance=provenance,
    )


def test_audit_report_contains_protocol_version(report, target_trial_spec):
    assert report.protocol_version == target_trial_spec.version
    assert report.protocol_version


def test_audit_report_contains_population_and_treatment(report):
    assert report.population
    assert report.treatment
    assert report.population["n"] == 238
    assert report.treatment["comparator"] == "zidovudine"


def test_audit_report_contains_estimate_and_interval(report, bootstrap_ci):
    assert report.estimate == bootstrap_ci.point_estimate
    assert report.estimate_ci_low == bootstrap_ci.ci_low
    assert report.estimate_ci_high == bootstrap_ci.ci_high


def test_audit_report_contains_overlap_diagnostics(report):
    assert report.overlap_diagnostics
    assert report.overlap_diagnostics["arm_counts"] == {0: 120, 1: 118}
    assert report.overlap_diagnostics["support_valid"] is True


def test_audit_report_contains_witness_outcomes(report):
    assert report.witness_outcomes == {"temporal": "pass", "causal": "pass", "clinical": "pass"}


def test_audit_report_contains_abstention_reasons(report):
    # Only the unsupported SupportDecision's reasons should be surfaced.
    assert report.abstention_reasons
    assert any("propensity_in_common_support: fail" in reason for reason in report.abstention_reasons)


def test_audit_report_contains_violations(report):
    assert report.violations
    assert report.violations[0]["patient_id"] == "P001"
    assert report.violations[0]["rule"] == "ineligible"


def test_audit_report_contains_provenance(report, provenance):
    assert report.provenance == provenance


def test_audit_report_to_dict_contains_all_required_fields(report):
    """Requirement 11.10: protocol version, population, treatment, estimate
    and interval, overlap diagnostics, witness outcomes, support
    restrictions, violations, and provenance must all be present and
    non-empty."""
    as_dict = report.to_dict()

    required_nonempty_fields = [
        "protocol_version",
        "population",
        "treatment",
        "overlap_diagnostics",
        "witness_outcomes",
        "abstention_reasons",
        "violations",
        "provenance",
    ]
    for field_name in required_nonempty_fields:
        assert field_name in as_dict, f"missing field: {field_name}"
        assert as_dict[field_name], f"field is empty: {field_name}"

    # estimate and interval: present and non-None (may legitimately be 0.0).
    for field_name in ["estimate", "estimate_ci_low", "estimate_ci_high"]:
        assert field_name in as_dict
        assert as_dict[field_name] is not None
