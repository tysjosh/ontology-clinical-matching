"""Unit/property tests for `src/witnesses/audit_io.py` (task 17.1;
Requirement 12.5, 12.6, 12.7, 19.5; `design.md` Property 16: "Suppressed
estimates carry no numeric value into public artifacts").

For `hypothesis`-generated unreleased candidate estimates with randomized
numeric point estimates, confirms:

- the numeric value's string form appears nowhere in the serialized text
  of the public artifact;
- the public artifact's `estimate` is `null`, `estimate_status` is
  `"suppressed"`, and `release_reasons` is non-empty;
- the protected artifact retains the real numeric value; and
- `scan_exports.scan_directory` -- the real scanner, used as the oracle,
  per the module docstring's "a future scanner rule change is
  automatically applied to audit artifacts" -- reports zero violations
  over the public directory.

**Validates: Requirements 12.5, 12.6, 12.7, 19.5**
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causal.overlap import ArmPropensitySummary, CommonSupportSummary, OverlapReport
from evaluation.release_gate import ReleaseDecision
from protocols.schema import TargetTrialSpec
from uncertainty.bootstrap import BootstrapCI
from witnesses.audit_io import audit_artifact_paths, write_audit_artifacts
from witnesses.audit_report import build_audit_report
from witnesses.temporal import WitnessResult

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scan_exports  # noqa: E402  (import after sys.path manipulation, by design)

MAIN_SPEC_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

_MAX_EXAMPLES = 25


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    import yaml

    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


def _overlap_report() -> OverlapReport:
    return OverlapReport(
        arm_counts={0: 120, 1: 118},
        arm_propensity_summary={
            0: ArmPropensitySummary(n=120, minimum=0.12, maximum=0.87, mean=0.5, median=0.5),
            1: ArmPropensitySummary(n=118, minimum=0.11, maximum=0.88, mean=0.51, median=0.5),
        },
        common_support=CommonSupportSummary(
            primary_count=230,
            primary_fraction=0.96,
            sensitivity_count=235,
            sensitivity_fraction=0.99,
        ),
        extreme_weight_count=8,
        extreme_weight_fraction=0.03,
        ess_by_arm={0: 110.0, 1: 108.0},
        smd_by_covariate={"age": 0.05, "baseline_cd4": 0.08},
        arm_counts_sufficient=True,
        ess_sufficient=True,
        smd_sufficient=True,
        no_unresolved_positivity_warning=True,
        support_valid=False,  # blocked: overlap_valid fails
    )


def _blocked_release_decision() -> ReleaseDecision:
    return ReleaseDecision(
        released=False,
        overlap_valid=False,
        balance_valid=True,
        uncertainty_valid=True,
        temporal_pass=True,
        causal_pass=True,
        clinical_pass=True,
        reasons=("overlap_valid: fail (OverlapReport.support_valid is False)",),
    )


def _build_suppressed_report(target_trial_spec, point_estimate: float):
    """Build a suppressed `AuditReport` whose real estimate is
    `point_estimate` -- a value this test must confirm never reaches the
    public artifact's serialized text."""
    estimate = BootstrapCI(
        point_estimate=point_estimate,
        ci_low=point_estimate - 0.05,
        ci_high=point_estimate + 0.05,
        n_replicates=1000,
        n_successful=1000,
    )
    witness_results = [
        WitnessResult(witness="temporal", status="pass", evidence={}),
        WitnessResult(witness="causal", status="pass", evidence={}),
        WitnessResult(witness="clinical", status="pass", evidence={}),
    ]
    return build_audit_report(
        protocol=target_trial_spec,
        population={"description": "synthetic", "n": 238},
        treatment={"intervention": "zidovudine+didanosine", "comparator": "zidovudine"},
        estimate=estimate,
        overlap=_overlap_report(),
        witness_results=witness_results,
        abstentions=[],
        violations=[],
        provenance={"experiment_id": "abc123def4567890"},
        release_decision=_blocked_release_decision(),
        estimate_status="suppressed",
        dataset_checksum="deadbeef" * 8,
        split_hash="split-hash-abc123",
        resolved_config_hash="config-hash-def456",
        checkpoint_hashes={"intervention_aware": "ckpt-hash-1"},
        seed=12345,
        sensitivity_variant=None,
        supported_count=230,
        abstained_count=8,
    )


#: Point estimates far from zero and from each other, with enough decimal
#: precision that their string form is distinctive and unlikely to collide
#: with any other number already present in the report (claim id, hash
#: fragment, counts, etc.) by coincidence.
_point_estimate = st.floats(
    min_value=-999.0, max_value=999.0, allow_nan=False, allow_infinity=False
).map(lambda x: round(x, 6)).filter(lambda x: x != 0.0)


@given(point_estimate=_point_estimate)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_16_suppressed_estimate_value_never_appears_in_public_artifact_text(
    target_trial_spec, point_estimate: float
) -> None:
    # A fresh `tempfile.TemporaryDirectory` per generated example, rather
    # than the function-scoped `tmp_path` fixture (which `hypothesis`
    # flags as not reset between examples -- `HealthCheck.function_scoped_fixture`).
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = _build_suppressed_report(target_trial_spec, point_estimate)
        experiment_dir = Path(tmp_dir) / "exp"
        claim_id = "primary_seed12345"

        public_path, protected_path = write_audit_artifacts(report, experiment_dir, claim_id)

        public_text = public_path.read_text(encoding="utf-8")
        numeric_strings = {
            repr(point_estimate),
            str(point_estimate),
            json.dumps(point_estimate),
        }
        for numeric_string in numeric_strings:
            assert numeric_string not in public_text, (
                f"point estimate {point_estimate!r}'s string form {numeric_string!r} "
                f"leaked into the public audit artifact: {public_text}"
            )

        public_payload = json.loads(public_text)
        assert public_payload["estimate"] is None
        assert public_payload["estimate_ci_low"] is None
        assert public_payload["estimate_ci_high"] is None
        assert public_payload["estimate_status"] == "suppressed"
        assert public_payload["release_reasons"]

        protected_payload = json.loads(protected_path.read_text(encoding="utf-8"))
        assert protected_payload["estimate"] == point_estimate
        assert protected_payload["estimate_ci_low"] == point_estimate - 0.05
        assert protected_payload["estimate_ci_high"] == point_estimate + 0.05


@given(point_estimate=_point_estimate)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_16_export_scanner_reports_zero_violations_over_public_directory(
    target_trial_spec, point_estimate: float
) -> None:
    """Uses the real `scan_exports.scan_directory` as the oracle (not a
    reimplementation), per the module docstring: a future scanner rule
    change is automatically applied to audit artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = _build_suppressed_report(target_trial_spec, point_estimate)
        experiment_dir = Path(tmp_dir) / "exp"
        claim_id = "primary_seed12345"

        write_audit_artifacts(report, experiment_dir, claim_id)
        public_dir = experiment_dir / "audit" / "public"

        violations = scan_exports.scan_directory(public_dir)

        assert violations == [], f"expected zero export violations, got: {violations}"


def test_released_report_public_and_protected_artifacts_are_identical(
    tmp_path, target_trial_spec
) -> None:
    """A `"released"` report has nothing to redact: its public and
    protected copies carry the identical estimate."""
    estimate = BootstrapCI(
        point_estimate=-0.12, ci_low=-0.20, ci_high=-0.04, n_replicates=1000, n_successful=1000
    )
    witness_results = [
        WitnessResult(witness="temporal", status="pass", evidence={}),
        WitnessResult(witness="causal", status="pass", evidence={}),
        WitnessResult(witness="clinical", status="pass", evidence={}),
    ]
    passing_overlap = OverlapReport(
        arm_counts={0: 120, 1: 118},
        arm_propensity_summary={
            0: ArmPropensitySummary(n=120, minimum=0.12, maximum=0.87, mean=0.5, median=0.5),
            1: ArmPropensitySummary(n=118, minimum=0.11, maximum=0.88, mean=0.51, median=0.5),
        },
        common_support=CommonSupportSummary(
            primary_count=230,
            primary_fraction=0.96,
            sensitivity_count=235,
            sensitivity_fraction=0.99,
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
    released_decision = ReleaseDecision(
        released=True,
        overlap_valid=True,
        balance_valid=True,
        uncertainty_valid=True,
        temporal_pass=True,
        causal_pass=True,
        clinical_pass=True,
        reasons=(),
    )
    report = build_audit_report(
        protocol=target_trial_spec,
        population={"description": "synthetic", "n": 238},
        treatment={"intervention": "zidovudine+didanosine", "comparator": "zidovudine"},
        estimate=estimate,
        overlap=passing_overlap,
        witness_results=witness_results,
        abstentions=[],
        violations=[],
        provenance={"experiment_id": "abc123def4567890"},
        release_decision=released_decision,
        estimate_status="released",
        dataset_checksum="deadbeef" * 8,
        split_hash="split-hash-abc123",
        resolved_config_hash="config-hash-def456",
        checkpoint_hashes={"intervention_aware": "ckpt-hash-1"},
        seed=12345,
        sensitivity_variant=None,
        supported_count=230,
        abstained_count=8,
    )

    experiment_dir = tmp_path / "exp"
    public_path, protected_path = write_audit_artifacts(report, experiment_dir, "primary_seed12345")

    public_payload = json.loads(public_path.read_text(encoding="utf-8"))
    protected_payload = json.loads(protected_path.read_text(encoding="utf-8"))

    assert public_payload["estimate"] == protected_payload["estimate"] == -0.12


def test_protected_artifact_is_written_at_owner_read_write_only_permissions(
    tmp_path, target_trial_spec
) -> None:
    import stat

    report = _build_suppressed_report(target_trial_spec, 1.5)
    experiment_dir = tmp_path / "exp"

    _, protected_path = write_audit_artifacts(report, experiment_dir, "primary_seed12345")

    mode = stat.S_IMODE(protected_path.stat().st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_audit_artifact_paths_matches_what_write_audit_artifacts_writes(tmp_path):
    experiment_dir = tmp_path / "exp"
    expected_public, expected_protected = audit_artifact_paths(experiment_dir, "claim-1")

    assert expected_public == experiment_dir / "audit" / "public" / "claim-1.json"
    assert expected_protected == experiment_dir / "audit" / "protected" / "claim-1.json"
