"""Unit tests for `scripts/run_locked_evaluation.py::run_locked_evaluation`
(task 21.1; Requirement 13.7, 13.8, 13.9, 13.1, 14.3, 14.4, 14.5).

Builds a small synthetic cohort, a sealed reference over it, and a
synthetic, fully-bound `LockedEvaluationToken` (`tests/token_fixtures.py`),
then confirms `run_locked_evaluation`:

- opens the test split and unseals the reference only via the token
  guards (never bypassing them);
- compares recovery estimates against the sealed reference on matching
  scales, raising `ScaleMismatchError` on a mismatch;
- runs the paired bootstrap and Holm correction across exactly the three
  frozen comparisons;
- records the already-evaluated witness results and release decision
  unchanged;
- recomputes every loaded checkpoint's hash.

`scripts/` is not an installed package, so this test adds it to
`sys.path` directly, mirroring `tests/unit/test_scan_exports.py`.

**Validates: Requirements 13.1, 13.7, 13.8, 13.9, 14.3, 14.4, 14.5**
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_locked_evaluation  # noqa: E402  (import after sys.path manipulation, by design)

from causal.baselines import CausalEstimate  # noqa: E402
from cohorts.cohort_construction import PatientRecord, RandomizedCohort  # noqa: E402
from cohorts.sealed_reference import (  # noqa: E402
    ScaleMismatchError,
    SealedReference,
    extended_randomized_reference_estimator,
)
from data.event_model import ClinicalEvent, MissingnessState  # noqa: E402
from data.timelines import build_patient_timeline  # noqa: E402
from evaluation.release_gate import ReleaseDecision  # noqa: E402
from protocols.eligibility_dsl import EligibilityResult  # noqa: E402
from protocols.endpoints import EndpointResult  # noqa: E402
from tests.token_fixtures import synthetic_token  # noqa: E402
from witnesses.temporal import WitnessResult  # noqa: E402

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


def _event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="baseline_cd4",
        value=300.0,
        unit="cells/mm3",
        modality="baseline",
        observed_at=ASSIGNMENT_TIME,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


def _patient(patient_id: str, treatment_arm: int, event_indicator: bool) -> PatientRecord:
    from datetime import timedelta

    treatment_concept = "zidovudine" if treatment_arm == 0 else "zidovudine+didanosine"
    baseline_event = _event(patient_id)
    treatment_event = _event(
        patient_id,
        concept_id=treatment_concept,
        value=1.0,
        unit=None,
        modality="treatment",
        observed_at=ASSIGNMENT_TIME,
    )
    timeline = build_patient_timeline([baseline_event, treatment_event])
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME + timedelta(weeks=50.0),
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="death" if event_indicator else None,
        ),
    )


@pytest.fixture
def synthetic_cohort() -> RandomizedCohort:
    patients = tuple(
        _patient(f"P{i:03d}", treatment_arm=i % 2, event_indicator=(i % 3 == 0))
        for i in range(20)
    )
    return RandomizedCohort(patients=patients)


@pytest.fixture
def sealed_reference(synthetic_cohort) -> SealedReference:
    return SealedReference(synthetic_cohort, estimator=extended_randomized_reference_estimator)


@dataclass
class _FakePipelineResult:
    """Minimal stand-in for `cohorts.pipeline.ObservationalizationPipelineResult`:
    `require_test_split` only ever reads `.test` after its token guard
    passes, and `run_locked_evaluation` only reads `.sealed_reference` --
    nothing else of the real dataclass is needed for this test."""

    sealed_reference: SealedReference
    test: object


@pytest.fixture
def pipeline_result(sealed_reference) -> _FakePipelineResult:
    return _FakePipelineResult(sealed_reference=sealed_reference, test=SimpleNamespace())


@pytest.fixture
def token(tmp_path):
    return synthetic_token(results_root=tmp_path / "results")


@pytest.fixture(scope="module")
def target_trial_spec():
    import yaml

    from protocols.schema import TargetTrialSpec

    main_spec_path = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
    with main_spec_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


ALL_PASS_WITNESSES = [
    WitnessResult(witness="temporal", status="pass", evidence={}),
    WitnessResult(witness="causal", status="pass", evidence={}),
    WitnessResult(witness="clinical", status="pass", evidence={}),
]


def _release_decision(released: bool = True) -> ReleaseDecision:
    return ReleaseDecision(
        released=released,
        overlap_valid=released,
        balance_valid=released,
        uncertainty_valid=released,
        temporal_pass=released,
        causal_pass=released,
        clinical_pass=released,
        reasons=() if released else ("overlap_valid: fail",),
    )


def _default_comparison_values() -> dict[str, tuple[list[float], list[float]]]:
    from evaluation.statistics import PRIMARY_COMPARISONS

    values_a = [0.6, 0.5, 0.7, 0.4, 0.55]
    values_b = [0.5, 0.5, 0.5, 0.5, 0.5]
    return {name: (values_a, values_b) for name in PRIMARY_COMPARISONS}


def _write_checkpoint(path: Path, content: bytes = b"checkpoint-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_run_locked_evaluation_computes_recovery_errors_on_matching_scales(
    token, pipeline_result, target_trial_spec, tmp_path
):
    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")

    recovery_estimates = {
        "risk_difference": CausalEstimate(
            name="ipcw_aipw", estimate=0.05, se=None, is_primary=True, scale="risk_difference"
        )
    }

    result = run_locked_evaluation.run_locked_evaluation(
        token,
        pipeline_result,
        target_trial_spec=target_trial_spec,
        recovery_estimates=recovery_estimates,
        comparison_values=_default_comparison_values(),
        witness_results=ALL_PASS_WITNESSES,
        release_decision=_release_decision(),
        checkpoint_paths={"intervention_aware": checkpoint_path},
        results_root=tmp_path / "results",
        bootstrap_seed=0,
    )

    assert "risk_difference" in result.recovery_errors
    assert result.recovery_errors["risk_difference"] >= 0.0


def test_run_locked_evaluation_raises_scale_mismatch_error(
    token, pipeline_result, target_trial_spec, tmp_path
):
    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")

    recovery_estimates = {
        # "risk_difference" sealed-reference quantity has scale
        # "risk_difference"; supplying a log-hazard-ratio-scaled estimate
        # against it must raise.
        "risk_difference": CausalEstimate(
            name="unadjusted_cox",
            estimate=0.05,
            se=None,
            is_primary=False,
            scale="log_hazard_ratio",
        )
    }

    with pytest.raises(ScaleMismatchError):
        run_locked_evaluation.run_locked_evaluation(
            token,
            pipeline_result,
            target_trial_spec=target_trial_spec,
            recovery_estimates=recovery_estimates,
            comparison_values=_default_comparison_values(),
            witness_results=ALL_PASS_WITNESSES,
            release_decision=_release_decision(),
            checkpoint_paths={"intervention_aware": checkpoint_path},
            results_root=tmp_path / "results",
            bootstrap_seed=0,
        )


def test_run_locked_evaluation_runs_the_paired_bootstrap_and_holm_correction_over_exactly_three(
    token, pipeline_result, target_trial_spec, tmp_path
):
    from evaluation.statistics import PRIMARY_COMPARISONS

    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")

    result = run_locked_evaluation.run_locked_evaluation(
        token,
        pipeline_result,
        target_trial_spec=target_trial_spec,
        recovery_estimates={},
        comparison_values=_default_comparison_values(),
        witness_results=ALL_PASS_WITNESSES,
        release_decision=_release_decision(),
        checkpoint_paths={"intervention_aware": checkpoint_path},
        results_root=tmp_path / "results",
        bootstrap_seed=0,
    )

    assert set(result.comparison_results) == set(PRIMARY_COMPARISONS)
    assert set(result.holm_adjusted_p_values) == set(PRIMARY_COMPARISONS)
    for name in PRIMARY_COMPARISONS:
        ci = result.comparison_results[name]
        assert ci.n_replicates == 1000
        assert 0.0 <= result.holm_adjusted_p_values[name] <= 1.0


def test_run_locked_evaluation_rejects_a_comparison_values_set_missing_a_frozen_comparison(
    token, pipeline_result, tmp_path
):
    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")
    incomplete = _default_comparison_values()
    del incomplete["gated_vs_ungated"]

    with pytest.raises(ValueError, match="gated_vs_ungated"):
        run_locked_evaluation.run_locked_evaluation(
            token,
            pipeline_result,
            target_trial_spec=target_trial_spec,
            recovery_estimates={},
            comparison_values=incomplete,
            witness_results=ALL_PASS_WITNESSES,
            release_decision=_release_decision(),
            checkpoint_paths={"intervention_aware": checkpoint_path},
            results_root=tmp_path / "results",
            bootstrap_seed=0,
        )


def test_run_locked_evaluation_records_witness_results_and_release_decision_unchanged(
    token, pipeline_result, target_trial_spec, tmp_path
):
    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")
    decision = _release_decision(released=False)

    result = run_locked_evaluation.run_locked_evaluation(
        token,
        pipeline_result,
        target_trial_spec=target_trial_spec,
        recovery_estimates={},
        comparison_values=_default_comparison_values(),
        witness_results=ALL_PASS_WITNESSES,
        release_decision=decision,
        checkpoint_paths={"intervention_aware": checkpoint_path},
        results_root=tmp_path / "results",
        bootstrap_seed=0,
    )

    assert result.witness_results == ALL_PASS_WITNESSES
    assert result.release_decision is decision


def test_run_locked_evaluation_recomputes_every_checkpoint_hash(
    token, pipeline_result, target_trial_spec, tmp_path
):
    import hashlib

    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")
    expected_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    result = run_locked_evaluation.run_locked_evaluation(
        token,
        pipeline_result,
        target_trial_spec=target_trial_spec,
        recovery_estimates={},
        comparison_values=_default_comparison_values(),
        witness_results=ALL_PASS_WITNESSES,
        release_decision=_release_decision(),
        checkpoint_paths={"intervention_aware": checkpoint_path},
        results_root=tmp_path / "results",
        bootstrap_seed=0,
    )

    assert result.checkpoint_hashes_after == {"intervention_aware": expected_hash}


def test_run_locked_evaluation_records_a_reference_access_timestamp(
    token, pipeline_result, target_trial_spec, tmp_path
):
    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")
    results_root = tmp_path / "results"

    result = run_locked_evaluation.run_locked_evaluation(
        token,
        pipeline_result,
        target_trial_spec=target_trial_spec,
        recovery_estimates={},
        comparison_values=_default_comparison_values(),
        witness_results=ALL_PASS_WITNESSES,
        release_decision=_release_decision(),
        checkpoint_paths={"intervention_aware": checkpoint_path},
        results_root=results_root,
        bootstrap_seed=0,
    )

    assert result.reference_access_timestamp is not None

    from infra.consumption_ledger import reference_access_records

    records = reference_access_records(results_root)
    assert len(records) == 1
    assert records[0][0] == token.evaluation_id


def test_run_locked_evaluation_builds_an_audit_report_for_every_candidate_estimate(
    token, pipeline_result, target_trial_spec, tmp_path
):
    """Requirement 12.1, 12.2, 12.4: one audit report per recovery
    estimate plus one per frozen comparison, each carrying the
    locked-evaluation id and the reference-access timestamp."""
    from evaluation.statistics import PRIMARY_COMPARISONS

    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")
    recovery_estimates = {
        "risk_difference": CausalEstimate(
            name="ipcw_aipw", estimate=0.05, se=None, is_primary=True, scale="risk_difference"
        )
    }

    result = run_locked_evaluation.run_locked_evaluation(
        token,
        pipeline_result,
        target_trial_spec=target_trial_spec,
        recovery_estimates=recovery_estimates,
        comparison_values=_default_comparison_values(),
        witness_results=ALL_PASS_WITNESSES,
        release_decision=_release_decision(),
        checkpoint_paths={"intervention_aware": checkpoint_path},
        results_root=tmp_path / "results",
        bootstrap_seed=0,
    )

    expected_claim_ids = {"recovery_risk_difference"} | {
        f"comparison_{name}" for name in PRIMARY_COMPARISONS
    }
    assert set(result.audit_reports) == expected_claim_ids
    for report in result.audit_reports.values():
        assert report.locked_evaluation_id == token.evaluation_id
        assert report.reference_access_timestamp is not None
        assert report.estimate_status == "released"


def test_run_locked_evaluation_builds_suppressed_audit_reports_when_the_gate_blocks(
    token, pipeline_result, target_trial_spec, tmp_path
):
    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")

    result = run_locked_evaluation.run_locked_evaluation(
        token,
        pipeline_result,
        target_trial_spec=target_trial_spec,
        recovery_estimates={},
        comparison_values=_default_comparison_values(),
        witness_results=ALL_PASS_WITNESSES,
        release_decision=_release_decision(released=False),
        checkpoint_paths={"intervention_aware": checkpoint_path},
        results_root=tmp_path / "results",
        bootstrap_seed=0,
    )

    assert len(result.audit_reports) > 0
    for report in result.audit_reports.values():
        assert report.estimate_status == "suppressed"
        assert report.release_reasons


def test_run_locked_evaluation_refuses_without_a_valid_token(
    pipeline_result, target_trial_spec, tmp_path
):
    from infra.reference_guard import ReferenceAccessError

    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")

    with pytest.raises(ReferenceAccessError):
        run_locked_evaluation.run_locked_evaluation(
            None,
            pipeline_result,
            target_trial_spec=target_trial_spec,
            recovery_estimates={},
            comparison_values=_default_comparison_values(),
            witness_results=ALL_PASS_WITNESSES,
            release_decision=_release_decision(),
            checkpoint_paths={"intervention_aware": checkpoint_path},
            results_root=tmp_path / "results",
            bootstrap_seed=0,
        )
