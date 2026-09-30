"""Unit tests for `src/infra/manifest.py`.

**Validates: Requirements 1.8, 1.9, 1.10, 1.11, 2.2, 2.3, 2.4, 2.7, 2.8, 5.7, 15.2, 15.4, 15.5**
"""

import json

import pytest

from infra.manifest import (
    DEVICE_UNSPECIFIED,
    GIT_COMMIT_UNAVAILABLE,
    GIT_STATE_CLEAN,
    GIT_STATE_DIRTY,
    GitProvenance,
    ManifestPrivacyError,
    ManifestWriter,
    PrivacyScanner,
    resolve_git_provenance,
)
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
        # Stubbed by default so no test in this module depends on the state of
        # the real working tree, and so no test spawns a `git` subprocess it
        # does not need. The git-provenance tests below override this.
        git_provenance_resolver=lambda: GitProvenance(commit="0" * 40),
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


class TestGitProvenanceRecording:
    """Git provenance is recorded on every manifest write.

    **Validates: Requirements 15.2, 15.4, 15.5, 5.7**

    Every case below injects a stubbed resolver, so none of these tests
    depends on -- or can be perturbed by -- the state of the real working
    tree.
    """

    @staticmethod
    def _stub(provenance):
        def resolver():
            return provenance

        return resolver

    def test_clean_tree_records_commit_and_clean_state(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = self._stub(GitProvenance(commit="d43bd1d" + "a" * 33))

        provenance = writer.build()["git_provenance"]

        assert provenance["state"] == GIT_STATE_CLEAN
        assert provenance["commit"] == "d43bd1d" + "a" * 33
        assert provenance["dirty"] is False
        assert provenance["uncommitted_paths"] == []
        assert provenance["unavailable_reason"] is None

    def test_dirty_tree_records_dirty_state_and_uncommitted_paths(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = self._stub(
            GitProvenance(
                commit="b" * 40,
                dirty=True,
                uncommitted_paths=("src/infra/manifest.py", "tests/unit/test_manifest.py"),
            )
        )

        provenance = writer.build()["git_provenance"]

        assert provenance["state"] == GIT_STATE_DIRTY
        assert provenance["commit"] == "b" * 40
        assert provenance["dirty"] is True
        assert provenance["uncommitted_paths"] == [
            "src/infra/manifest.py",
            "tests/unit/test_manifest.py",
        ]

    def test_unavailable_commit_is_recorded_explicitly_not_omitted(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = self._stub(
            GitProvenance.unavailable("git executable not found on PATH")
        )

        manifest = writer.build()

        # Requirement 15.5: the field is present and explicitly marked, never
        # absent and never null.
        assert "git_provenance" in manifest
        provenance = manifest["git_provenance"]
        assert provenance["state"] == GIT_COMMIT_UNAVAILABLE
        assert provenance["commit"] == GIT_COMMIT_UNAVAILABLE
        assert provenance["unavailable_reason"] == "git executable not found on PATH"

    def test_resolver_that_raises_is_recorded_as_unavailable(self, tmp_path):
        def exploding_resolver():
            raise RuntimeError("git subsystem is on fire")

        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = exploding_resolver

        provenance = writer.build()["git_provenance"]

        assert provenance["state"] == GIT_COMMIT_UNAVAILABLE
        assert "RuntimeError" in provenance["unavailable_reason"]

    def test_resolver_returning_wrong_type_is_recorded_as_unavailable(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = lambda: "d43bd1d"

        provenance = writer.build()["git_provenance"]

        assert provenance["state"] == GIT_COMMIT_UNAVAILABLE
        assert "not a GitProvenance" in provenance["unavailable_reason"]

    def test_provenance_is_recorded_on_write_not_only_on_build(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = self._stub(GitProvenance(commit="c" * 40))
        target = tmp_path / "manifest.json"

        writer.write(target)

        on_disk = json.loads(target.read_text())
        assert on_disk["git_provenance"]["commit"] == "c" * 40
        assert on_disk["git_provenance"]["state"] == GIT_STATE_CLEAN


class TestArtifactCommitBinding:
    """Every written checkpoint path and result directory is bound to the commit.

    **Validates: Requirements 15.4, 15.5**
    """

    def test_checkpoints_and_result_directories_are_bound_to_the_commit(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.add_checkpoint_path(str(tmp_path / "checkpoints" / "ensemble_seed101.pt"))
        writer.add_result_directory(str(tmp_path / "abc123deadbeef00"))
        writer.git_provenance_resolver = lambda: GitProvenance(commit="d" * 40)

        bindings = writer.build()["artifact_commit_bindings"]

        # Two checkpoints (one from the shared fixture) and one result directory.
        assert len(bindings) == 3
        assert {b["artifact_kind"] for b in bindings} == {"checkpoint", "result_directory"}
        assert all(b["git_commit"] == "d" * 40 for b in bindings)
        assert all(b["git_state"] == GIT_STATE_CLEAN for b in bindings)
        bound_artifacts = {b["artifact"] for b in bindings}
        assert str(tmp_path / "checkpoints" / "ensemble_seed101.pt") in bound_artifacts
        assert str(tmp_path / "abc123deadbeef00") in bound_artifacts

    def test_bindings_carry_the_dirty_state_alongside_the_commit(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.add_result_directory(str(tmp_path / "abc123deadbeef00"))
        writer.git_provenance_resolver = lambda: GitProvenance(
            commit="e" * 40, dirty=True, uncommitted_paths=("src/infra/manifest.py",)
        )

        bindings = writer.build()["artifact_commit_bindings"]

        assert bindings
        assert all(b["git_state"] == GIT_STATE_DIRTY for b in bindings)

    def test_binding_is_present_with_unavailable_marker_when_commit_unresolvable(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.add_result_directory(str(tmp_path / "abc123deadbeef00"))
        writer.git_provenance_resolver = lambda: GitProvenance.unavailable("not a git repository")

        bindings = writer.build()["artifact_commit_bindings"]

        assert bindings
        assert all(b["git_commit"] == GIT_COMMIT_UNAVAILABLE for b in bindings)
        assert all(b["git_state"] == GIT_COMMIT_UNAVAILABLE for b in bindings)

    def test_bindings_are_empty_when_no_artifact_was_recorded(self, tmp_path):
        writer = ManifestWriter(
            experiment_id="abc123deadbeef00",
            stage="cohort_construction",
            git_provenance_resolver=lambda: GitProvenance(commit="f" * 40),
        )

        assert writer.build()["artifact_commit_bindings"] == []


class TestGitProvenanceValidation:
    """`GitProvenance` refuses states that would serialize dishonestly.

    **Validates: Requirements 15.2, 15.5**
    """

    def test_empty_commit_string_is_refused(self):
        with pytest.raises(ValueError, match="non-empty identifier"):
            GitProvenance(commit="   ")

    def test_unavailable_state_requires_a_reason(self):
        with pytest.raises(ValueError, match="unavailable_reason"):
            GitProvenance(commit=None)

    def test_unavailable_state_cannot_claim_cleanliness_knowledge(self):
        with pytest.raises(ValueError, match="cleanliness"):
            GitProvenance(commit=None, unavailable_reason="no repo", dirty=True)

    def test_unavailable_reason_is_reduced_to_one_short_line(self):
        provenance = GitProvenance.unavailable("fatal: not a git repository\nusage: git ...\n")

        assert provenance.unavailable_reason == "fatal: not a git repository"

    def test_states_are_exactly_clean_dirty_or_unavailable(self):
        assert GitProvenance(commit="a" * 40).state == GIT_STATE_CLEAN
        assert GitProvenance(commit="a" * 40, dirty=True).state == GIT_STATE_DIRTY
        assert GitProvenance.unavailable("no git").state == GIT_COMMIT_UNAVAILABLE


class TestDefaultResolver:
    """The production resolver behaves on a real and on a non-repository path.

    **Validates: Requirements 15.2, 15.5**
    """

    def test_resolver_reports_unavailable_outside_a_repository(self, tmp_path):
        # `tmp_path` on macOS lives under /private/var, outside this project's
        # repository, so `git rev-parse HEAD` there must fail rather than
        # silently picking up an unrelated parent repository.
        provenance = resolve_git_provenance(repo_root=tmp_path)

        if provenance.available:
            pytest.skip("temporary directory unexpectedly sits inside a git repository")
        assert provenance.state == GIT_COMMIT_UNAVAILABLE
        assert provenance.unavailable_reason

    def test_resolver_reports_a_commit_for_this_repository(self):
        provenance = resolve_git_provenance()

        # This project is a git repository as of task 2; the working tree may
        # legitimately be clean or dirty, but the commit must resolve and the
        # state must be one of the two available states.
        assert provenance.available
        assert provenance.state in {GIT_STATE_CLEAN, GIT_STATE_DIRTY}
        assert len(provenance.commit) >= 7
        assert provenance.dirty == bool(provenance.uncommitted_paths)


class TestDeviceRecording:
    """The device a run executed on is recorded in that run's manifest.

    **Validates: Requirement 5.7**
    """

    def test_set_device_is_recorded(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = lambda: GitProvenance(commit="a" * 40)
        writer.set_device("cuda")

        assert writer.build()["device"] == "cuda"

    def test_unset_device_records_the_explicit_sentinel(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = lambda: GitProvenance(commit="a" * 40)

        manifest = writer.build()

        assert "device" in manifest
        assert manifest["device"] == DEVICE_UNSPECIFIED

    def test_empty_device_is_refused(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)

        with pytest.raises(ValueError, match="non-empty"):
            writer.set_device("  ")


class TestProvenanceSurvivesPrivacyScan:
    """A recorded provenance block does not itself trip the privacy scanner.

    **Validates: Requirements 2.4, 15.2**
    """

    def test_commit_sha_and_relative_uncommitted_paths_are_clean(self, tmp_path):
        writer = _build_synthetic_writer(tmp_path)
        writer.git_provenance_resolver = lambda: GitProvenance(
            commit="d43bd1d9f0a1b2c3d4e5f60718293a4b5c6d7e8f",
            dirty=True,
            uncommitted_paths=("src/infra/manifest.py", "tests/unit/test_manifest.py"),
        )
        target = tmp_path / "manifest.json"

        writer.write(target)

        assert target.exists()
        assert PrivacyScanner(allowed_path_prefixes=(str(tmp_path),)).scan(writer.build()) == []
