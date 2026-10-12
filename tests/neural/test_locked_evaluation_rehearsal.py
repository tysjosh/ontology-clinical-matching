"""End-to-end rehearsal of the Locked_Evaluation_Runner (task 21.2;
Requirement 13.10, 13.11; `design.md` Property 27).

Runs `scripts/run_locked_evaluation.py`'s full verified-token path --
`run_locked_evaluation` (task 21.1) followed by `commit_locked_evaluation`
(task 21.2) -- end to end against a ~40-patient synthetic cohort, under a
directly-constructed synthetic `LockedEvaluationToken`
(`tests/token_fixtures.py`) in a `tmp_path` results root. Per `design.md`'s
"No test mints a real token" rule, every token here is synthetic; because
the guard is a plain `isinstance` check, a synthetic token opens only the
synthetic sealed artifact and test split built in this file, never
anything real.

Property 27, verbatim: "*For any* step after successful token
verification, injecting a failure at that step leaves the results tree
byte-identical, the token and evaluation identifier unconsumed, and no
partial artifact present; a consumed evaluation identifier is refused
before the test split or sealed reference is opened; and result artifacts
are written only to unoccupied paths."

This file implements the "injecting a failure at that step leaves
everything unchanged" half by parameterizing over every post-verification
step `commit_locked_evaluation` can fail at (staging, the unchanged-
checkpoint confirmation, the rename, and the consumption-ledger append) --
the steps task 21.2 itself owns and that can be feasibly broken without
monkeypatching the pure-computation internals already covered by
`tests/unit/test_run_locked_evaluation_core.py`. The "consumed id refused
before protected resources open" and "unoccupied path only" halves are
already covered there and in `tests/unit/test_run_locked_evaluation.py`
(Requirement 13.6, 13.10's own dedicated tests); this file does not repeat
them.

The one test that would touch a *real* (non-synthetic) locked-evaluation
path is gated behind `--run-locked-evaluation-test` and skipped by
default, mirroring `scripts/run_paper.py`'s own production gating -- see
`test_real_locked_evaluation_path_is_not_exercised_by_default` and
`tests/conftest.py`'s `pytest_addoption`.

**Validates: Requirements 13.10, 13.11**
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_locked_evaluation  # noqa: E402  (import after sys.path manipulation, by design)

from causal.baselines import CausalEstimate  # noqa: E402
from cohorts.cohort_construction import PatientRecord, RandomizedCohort  # noqa: E402
from cohorts.sealed_reference import (  # noqa: E402
    SealedReference,
    extended_randomized_reference_estimator,
)
from data.event_model import ClinicalEvent, MissingnessState  # noqa: E402
from data.timelines import build_patient_timeline  # noqa: E402
from evaluation.release_gate import ReleaseDecision  # noqa: E402
from evaluation.statistics import PRIMARY_COMPARISONS  # noqa: E402
from infra.consumption_ledger import consumed_ledger_path, is_consumed  # noqa: E402
from protocols.eligibility_dsl import EligibilityResult  # noqa: E402
from protocols.endpoints import EndpointResult  # noqa: E402
from protocols.schema import TargetTrialSpec  # noqa: E402
from tests.token_fixtures import synthetic_token  # noqa: E402
from witnesses.temporal import WitnessResult  # noqa: E402

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)

pytestmark = pytest.mark.neural


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


@pytest.fixture(scope="module")
def synthetic_cohort() -> RandomizedCohort:
    """~40 patients, matching task 21.2's own sizing."""
    patients = tuple(
        _patient(f"P{i:03d}", treatment_arm=i % 2, event_indicator=(i % 3 == 0))
        for i in range(40)
    )
    return RandomizedCohort(patients=patients)


@pytest.fixture(scope="module")
def sealed_reference(synthetic_cohort) -> SealedReference:
    return SealedReference(synthetic_cohort, estimator=extended_randomized_reference_estimator)


@dataclass
class _FakePipelineResult:
    sealed_reference: SealedReference
    test: object


@pytest.fixture
def pipeline_result(sealed_reference) -> _FakePipelineResult:
    return _FakePipelineResult(sealed_reference=sealed_reference, test=SimpleNamespace())


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    import yaml

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


def _comparison_values(synthetic_cohort) -> dict[str, tuple[list[float], list[float]]]:
    """Per-patient paired values for the three frozen comparisons, over
    the identical synthetic patient id set (Requirement 14.4's pairing
    requirement)."""
    n = len(synthetic_cohort.patients)
    values_a = [0.5 + 0.01 * i for i in range(n)]
    values_b = [0.5 for _ in range(n)]
    return {name: (values_a, values_b) for name in PRIMARY_COMPARISONS}


def _write_checkpoint(path: Path, content: bytes = b"checkpoint-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _directory_hash(path: Path) -> str:
    """A content hash of every file under `path`, by relative path and
    bytes -- the "results tree byte-identical" check Property 27 asks
    for, computed over whatever subset of the tree exists (an absent
    directory hashes identically to an empty one, so a test can call this
    before anything exists)."""
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _run_full_evaluation(
    token, pipeline_result, target_trial_spec, synthetic_cohort, tmp_path, results_root
):
    """Run `run_locked_evaluation` end to end (task 21.1), returning
    `(token, result, checkpoint_paths)` ready for `commit_locked_evaluation`."""
    checkpoint_path = _write_checkpoint(tmp_path / "ckpt.pt")
    checkpoint_paths = {"intervention_aware": checkpoint_path}

    result = run_locked_evaluation.run_locked_evaluation(
        token,
        pipeline_result,
        target_trial_spec=target_trial_spec,
        recovery_estimates={
            "risk_difference": CausalEstimate(
                name="ipcw_aipw", estimate=0.03, se=None, is_primary=True, scale="risk_difference"
            )
        },
        comparison_values=_comparison_values(synthetic_cohort),
        witness_results=ALL_PASS_WITNESSES,
        release_decision=_release_decision(),
        checkpoint_paths=checkpoint_paths,
        results_root=results_root,
        bootstrap_seed=0,
    )
    return result, checkpoint_paths


# ---------------------------------------------------------------------------
# End-to-end rehearsal: the identical patient id set flows through
# recovery, the paired bootstrap, the witnesses, and the gate.
# ---------------------------------------------------------------------------


def test_identical_patient_set_flows_through_recovery_bootstrap_witnesses_and_gate(
    synthetic_cohort, pipeline_result, target_trial_spec, tmp_path
):
    results_root = tmp_path / "results"
    token = synthetic_token(results_root=results_root)

    result, checkpoint_paths = _run_full_evaluation(
        token, pipeline_result, target_trial_spec, synthetic_cohort, tmp_path, results_root
    )

    # Recovery: a real, finite error against the sealed reference.
    assert "risk_difference" in result.recovery_errors
    assert result.recovery_errors["risk_difference"] >= 0.0

    # Paired bootstrap + Holm: all three frozen comparisons, run over the
    # SAME per-patient value pairs built from the identical patient set.
    assert set(result.comparison_results) == set(PRIMARY_COMPARISONS)
    assert set(result.holm_adjusted_p_values) == set(PRIMARY_COMPARISONS)

    # Witnesses + gate: recorded unchanged, over the identical patients.
    assert {r.witness for r in result.witness_results} == {"temporal", "causal", "clinical"}
    assert result.release_decision.released is True

    # Commit succeeds and produces the expected artifact tree.
    checkpoint_hashes_before = {
        name: _sha256_file(path) for name, path in checkpoint_paths.items()
    }
    committed_path = run_locked_evaluation.commit_locked_evaluation(
        token,
        result,
        checkpoint_hashes_before=checkpoint_hashes_before,
        results_root=results_root,
    )

    assert committed_path.is_dir()
    assert (committed_path / "summary.json").is_file()
    assert (committed_path / "audit" / "public").is_dir()
    assert (committed_path / "audit" / "protected").is_dir()
    assert is_consumed(token.evaluation_id, results_root)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Property 27: failure injection over every post-verification commit step.
# ---------------------------------------------------------------------------


def _prepare_committed_result(
    synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, results_root
):
    """Build a token and a real evaluated `LockedEvaluationResult`, ready
    to pass to `commit_locked_evaluation` -- the shared setup for every
    failure-injection case below."""
    token = synthetic_token(results_root=results_root)
    result, checkpoint_paths = _run_full_evaluation(
        token, pipeline_result, target_trial_spec, synthetic_cohort, tmp_path, results_root
    )
    checkpoint_hashes_before = {
        name: _sha256_file(path) for name, path in checkpoint_paths.items()
    }
    return token, result, checkpoint_hashes_before


def test_property_27_staging_failure_leaves_everything_unchanged(
    synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, monkeypatch
):
    results_root = tmp_path / "results"
    token, result, checkpoint_hashes_before = _prepare_committed_result(
        synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, results_root
    )

    before_hash = _directory_hash(results_root)
    ledger_existed_before = consumed_ledger_path(results_root).exists()

    monkeypatch.setattr(
        run_locked_evaluation,
        "_stage_locked_evaluation_artifacts",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected staging failure")),
    )

    with pytest.raises(run_locked_evaluation.CommitFailedError) as excinfo:
        run_locked_evaluation.commit_locked_evaluation(
            token,
            result,
            checkpoint_hashes_before=checkpoint_hashes_before,
            results_root=results_root,
        )

    assert excinfo.value.failing_step == "staging"
    assert _directory_hash(results_root) == before_hash
    assert consumed_ledger_path(results_root).exists() == ledger_existed_before
    assert not is_consumed(token.evaluation_id, results_root)
    assert not run_locked_evaluation.staging_dir(results_root, token.evaluation_id).exists()
    assert not run_locked_evaluation.locked_evaluation_result_dir(
        results_root, token.evaluation_id
    ).exists()


def test_property_27_checkpoint_changed_mid_run_leaves_everything_unchanged(
    synthetic_cohort, pipeline_result, target_trial_spec, tmp_path
):
    results_root = tmp_path / "results"
    token, result, checkpoint_hashes_before = _prepare_committed_result(
        synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, results_root
    )

    before_hash = _directory_hash(results_root)

    # Simulate a checkpoint that changed after step-3 verification but
    # before commit: deliberately mismatch the "before" hash.
    tampered_hashes_before = dict(checkpoint_hashes_before)
    tampered_hashes_before["intervention_aware"] = "not-the-real-hash"

    with pytest.raises(run_locked_evaluation.CommitFailedError) as excinfo:
        run_locked_evaluation.commit_locked_evaluation(
            token,
            result,
            checkpoint_hashes_before=tampered_hashes_before,
            results_root=results_root,
        )

    assert excinfo.value.failing_step == "checkpoint_unchanged_check"
    assert _directory_hash(results_root) == before_hash
    assert not is_consumed(token.evaluation_id, results_root)
    assert not run_locked_evaluation.staging_dir(results_root, token.evaluation_id).exists()
    assert not run_locked_evaluation.locked_evaluation_result_dir(
        results_root, token.evaluation_id
    ).exists()


def test_property_27_rename_failure_leaves_everything_unchanged(
    synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, monkeypatch
):
    results_root = tmp_path / "results"
    token, result, checkpoint_hashes_before = _prepare_committed_result(
        synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, results_root
    )

    before_hash = _directory_hash(results_root)

    def _failing_rename(*args, **kwargs):
        raise OSError("injected rename failure")

    monkeypatch.setattr(run_locked_evaluation.os, "rename", _failing_rename)

    with pytest.raises(run_locked_evaluation.CommitFailedError) as excinfo:
        run_locked_evaluation.commit_locked_evaluation(
            token,
            result,
            checkpoint_hashes_before=checkpoint_hashes_before,
            results_root=results_root,
        )

    assert excinfo.value.failing_step == "rename"
    assert _directory_hash(results_root) == before_hash
    assert not is_consumed(token.evaluation_id, results_root)
    assert not run_locked_evaluation.staging_dir(results_root, token.evaluation_id).exists()
    assert not run_locked_evaluation.locked_evaluation_result_dir(
        results_root, token.evaluation_id
    ).exists()


def test_property_27_consumption_append_failure_is_reported_distinctly(
    synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, monkeypatch
):
    """Requirement 13.11 scopes "leaves pre-existing artifacts byte-
    identical" to failures at steps that have not yet become
    irreversible. The consumption-ledger append happens AFTER the
    `os.rename` already committed the result directory, so a failure here
    cannot be undone by deleting a (no-longer-existing) staging
    directory; this test confirms that failure is reported as its own
    distinct, named step rather than silently swallowed or misreported as
    a staging failure, and that the committed result directory itself is
    real and present."""
    results_root = tmp_path / "results"
    token, result, checkpoint_hashes_before = _prepare_committed_result(
        synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, results_root
    )

    def _failing_append_consumed(*args, **kwargs):
        raise RuntimeError("injected consumption-ledger append failure")

    monkeypatch.setattr(run_locked_evaluation, "append_consumed", _failing_append_consumed)

    with pytest.raises(run_locked_evaluation.CommitFailedError) as excinfo:
        run_locked_evaluation.commit_locked_evaluation(
            token,
            result,
            checkpoint_hashes_before=checkpoint_hashes_before,
            results_root=results_root,
        )

    assert excinfo.value.failing_step == "consumption_append"
    # The rename already happened: the result directory is real.
    committed = run_locked_evaluation.locked_evaluation_result_dir(
        results_root, token.evaluation_id
    )
    assert committed.is_dir()
    assert (committed / "summary.json").is_file()
    # The staging directory itself is gone either way (renamed away).
    assert not run_locked_evaluation.staging_dir(results_root, token.evaluation_id).exists()


def test_a_consumed_evaluation_id_is_refused_before_opening_protected_resources(
    synthetic_cohort, pipeline_result, target_trial_spec, tmp_path
):
    """The other half of Property 27: a consumed evaluation identifier is
    refused before the test split or sealed reference is opened. Exercised
    here via the full `main()`-level `check_not_consumed` step (already
    unit-tested directly in `tests/unit/test_run_locked_evaluation.py`);
    this test additionally confirms a full commit cycle followed by a
    second attempt at the SAME evaluation_id is refused end to end."""
    results_root = tmp_path / "results"
    token, result, checkpoint_hashes_before = _prepare_committed_result(
        synthetic_cohort, pipeline_result, target_trial_spec, tmp_path, results_root
    )

    run_locked_evaluation.commit_locked_evaluation(
        token,
        result,
        checkpoint_hashes_before=checkpoint_hashes_before,
        results_root=results_root,
    )

    with pytest.raises(run_locked_evaluation.AlreadyConsumedError):
        run_locked_evaluation.check_not_consumed(token, results_root)


# ---------------------------------------------------------------------------
# The real locked-evaluation path: skipped by default (design.md's "No
# test mints a real token" rule and its own production-gating mirror).
# ---------------------------------------------------------------------------


def test_run_locked_evaluation_test_flag_defaults_to_off(request):
    """Confirms the gating flag (`tests/conftest.py`'s `pytest_addoption`)
    is registered and defaults to `False`, mirroring production gating.
    No real (non-synthetic) locked-evaluation path exists anywhere in
    this suite to gate -- `design.md`'s "No test mints a real token" rule
    means every locked-evaluation test, including this rehearsal file,
    always uses a synthetic token over a synthetic cohort regardless of
    this flag's value. The flag exists so a *future* real-data test has
    somewhere to attach, per the task's own instruction, without this
    file inventing one against data it has no business touching."""
    assert request.config.getoption("--run-locked-evaluation-test") is False
