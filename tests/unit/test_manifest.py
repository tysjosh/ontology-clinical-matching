"""Unit tests for `src/infra/manifest.py`.

**Validates: Requirements 1.8, 1.9, 1.10, 1.11, 2.2, 2.3, 2.4, 2.7, 2.8**
"""

import json

import pytest

from infra.manifest import ManifestPrivacyError, ManifestWriter, PrivacyScanner
from infra.nondeterminism import mark_nondeterministic, reset_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    # The nondeterminism registry is process-global module state; reset
    # before and after every test so recorded invocations never leak across
    # test cases (mirrors tests/unit/test_nondeterminism.py's fixture).
    reset_registry()
    yield
    reset_registry()


def _build_synthetic_writer(tmp_path, allowed_path_prefixes=()):
    if not allowed_path_prefixes:
        allowed_path_prefixes = (str(tmp_path),)
    writer = ManifestWriter(
        experiment_id="abc123deadbeef00",
        stage="cohort_construction",
        allowed_path_prefixes=allowed_path_prefixes,
    )
    writer.add_input_hash("actg175_raw", "e" * 64)
    writer.add_input_hash("target_trial_spec", "f" * 64)
    writer.set_cohort_definition(
        {
            "population": "arm0_arm1_eligible",
            "eligibility_rules": ["hiv_positive", "baseline_cd4_in_range"],
        }
    )
    writer.set_feature_list(["age", "weight", "baseline_cd4", "karnofsky_score"])
    writer.set_treatment_definition({"arm_1": "zidovudine+didanosine", "arm_0": "zidovudine"})
    writer.set_outcome_definition({"type": "composite_time_to_event"})
    writer.set_model_config({"model": "cox_baseline", "adjustment_set_size": 13})
    writer.add_checkpoint_path(str(tmp_path / "checkpoints" / "model.pt"))
    writer.set_predictions({"summary_path": str(tmp_path / "predictions" / "summary.json")})
    writer.set_metrics({"concordance": 0.71, "auroc_96wk": 0.68})
    return writer


class TestManifestCompleteness:
    def test_manifest_contains_all_required_keys(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)

        manifest = writer.build()

        required_keys = {
            "input_hashes",
            "cohort_definition",
            "feature_list",
            "treatment_definition",
            "outcome_definition",
            "model_config",
            "checkpoint_paths",
            "predictions",
            "metrics",
            "environment_versions",
            "nondeterminism_invocations",
        }
        assert required_keys.issubset(manifest.keys())
        assert manifest["input_hashes"]["actg175_raw"] == "e" * 64
        assert manifest["cohort_definition"]["population"] == "arm0_arm1_eligible"
        assert "baseline_cd4" in manifest["feature_list"]
        assert manifest["treatment_definition"]["arm_0"] == "zidovudine"
        assert manifest["outcome_definition"]["type"] == "composite_time_to_event"
        assert manifest["model_config"]["model"] == "cox_baseline"
        assert manifest["checkpoint_paths"]
        assert manifest["predictions"]["summary_path"]
        assert manifest["metrics"]["concordance"] == 0.71
        # Environment versions must be non-empty in any real Python
        # environment (at minimum, pytest and this project's own deps).
        assert len(manifest["environment_versions"]) > 0

    def test_manifest_includes_recorded_nondeterminism_invocations(self, tmp_path):
        @mark_nondeterministic("cudnn backward pass")
        def flaky_op():
            return None

        flaky_op()

        writer = _build_synthetic_writer(tmp_path)
        manifest = writer.build()

        assert len(manifest["nondeterminism_invocations"]) == 1
        assert manifest["nondeterminism_invocations"][0]["reason"] == "cudnn backward pass"

    def test_clean_manifest_writes_successfully(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        target = tmp_path / "manifest.json"

        written_path = writer.write(target)

        assert written_path == target
        assert target.exists()
        on_disk = json.loads(target.read_text())
        assert on_disk["cohort_definition"]["population"] == "arm0_arm1_eligible"


class TestPrivacyScanner:
    def test_scan_flags_injected_ssn_like_direct_identifier(self):
        manifest = {"notes": "patient contact: 123-45-6789"}

        violations = PrivacyScanner().scan(manifest)

        assert any(v.category == "direct_identifier" for v in violations)

    def test_scan_flags_direct_identifier_field_name(self):
        manifest = {"cohort_definition": {"name": "Jane Doe"}}

        violations = PrivacyScanner().scan(manifest)

        assert any(v.category == "direct_identifier" for v in violations)

    def test_scan_flags_quasi_identifier_field_names(self):
        manifest = {"patient_record": {"zip_code": "94110"}}

        violations = PrivacyScanner().scan(manifest)

        assert any(v.category == "quasi_identifier" for v in violations)

    def test_scan_flags_combined_dob_and_zip_as_quasi_identifier(self):
        manifest = {"patient_record": {"date_of_birth": "1980-01-01", "zip_code": "94110"}}

        violations = PrivacyScanner().scan(manifest)

        combined = [v for v in violations if "combination" in v.message]
        assert combined

    def test_scan_flags_injected_aws_key_like_secret(self):
        manifest = {"config": {"note": "AKIAABCDEFGHIJKLMNOP"}}

        violations = PrivacyScanner().scan(manifest)

        assert any(v.category == "secret" for v in violations)

    def test_scan_flags_secret_field_name(self):
        manifest = {"config": {"api_key": "some-value-here"}}

        violations = PrivacyScanner().scan(manifest)

        assert any(v.category == "secret" for v in violations)

    def test_scan_flags_unauthorized_absolute_path_outside_allowed_prefix(self):
        manifest = {"checkpoint_paths": ["/Users/someone/secret_project/model.pt"]}

        violations = PrivacyScanner(allowed_path_prefixes=("/tmp/results",)).scan(manifest)

        assert any(v.category == "unauthorized_path" for v in violations)

    def test_scan_allows_absolute_path_under_allowed_prefix(self, tmp_path):
        allowed_root = tmp_path / "results"
        allowed_root.mkdir()
        checkpoint = allowed_root / "model.pt"
        manifest = {"checkpoint_paths": [str(checkpoint)]}

        violations = PrivacyScanner(allowed_path_prefixes=(str(allowed_root),)).scan(manifest)

        assert not any(v.category == "unauthorized_path" for v in violations)

    def test_scan_returns_no_violations_for_clean_manifest(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        manifest = writer.build()

        violations = PrivacyScanner(allowed_path_prefixes=(str(tmp_path),)).scan(manifest)

        assert violations == []

    def test_violation_evidence_never_contains_raw_value(self):
        manifest = {"notes": "SSN is 123-45-6789"}

        violations = PrivacyScanner().scan(manifest)

        for violation in violations:
            assert "123-45-6789" not in violation.evidence


class TestManifestWriterPrivacyBlocking:
    def test_write_raises_and_does_not_write_on_injected_ssn(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.set_cohort_definition({"population": "arm0_arm1_eligible", "notes": "123-45-6789"})
        target = tmp_path / "manifest.json"

        with pytest.raises(ManifestPrivacyError) as exc_info:
            writer.write(target)

        assert not target.exists()
        assert any(v.category == "direct_identifier" for v in exc_info.value.violations)

    def test_write_raises_and_does_not_write_on_injected_secret(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.set_model_config({"model": "cox_baseline", "api_key": "AKIAABCDEFGHIJKLMNOP"})
        target = tmp_path / "manifest.json"

        with pytest.raises(ManifestPrivacyError) as exc_info:
            writer.write(target)

        assert not target.exists()
        assert any(v.category == "secret" for v in exc_info.value.violations)

    def test_write_raises_and_does_not_write_on_unauthorized_path(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.add_checkpoint_path("/Users/someone/outside_repo/model.pt")
        target = tmp_path / "manifest.json"

        with pytest.raises(ManifestPrivacyError) as exc_info:
            writer.write(target)

        assert not target.exists()
        assert any(v.category == "unauthorized_path" for v in exc_info.value.violations)

    def test_write_succeeds_for_clean_manifest(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        target = tmp_path / "manifest.json"

        writer.write(target)

        assert target.exists()
