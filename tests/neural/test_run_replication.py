"""Integration test for `scripts/run_replication.py`'s `run_seed`/
`aggregate` (task 18/18.1; Requirement 9.1-9.9, 18.1, 18.2, 18.3).

Builds a small synthetic cohort (mirroring
`tests/integration/test_run_ablations.py`'s established fixture pattern)
and confirms `run_seed(...)` runs the full development-stage pipeline
once at a given seed -- split/observationalization, both neural models'
five-replica ensembles, the three witnesses, the release gate, and a
content-hashed per-seed result directory -- and that `aggregate(...)`
combines several seeds' results without ever selecting a best/worst
seed.

**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.7, 9.8, 9.9**
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from data.event_model import ClinicalEvent, MissingnessState  # noqa: E402
from data.timelines import build_patient_timeline  # noqa: E402
from protocols.schema import TargetTrialSpec  # noqa: E402

pytestmark = pytest.mark.neural

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_replication import CROSS_SEED_STABILITY_CRITERION, aggregate, run_seed  # noqa: E402

from models.torch_training import TrainingConfig  # noqa: E402

#: Short training configuration for test speed (mirrors
#: `tests/integration/test_run_ablations.py`'s `SHORT_CONFIG`).
SHORT_CONFIG = TrainingConfig(max_epochs=3, patience=2)

MAIN_SPEC_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = ASSIGNMENT_TIME + timedelta(weeks=20)
WEEK_96 = ASSIGNMENT_TIME + timedelta(weeks=96)


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


def _event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="cd4_count",
        value=350.0,
        unit="cells/mm3",
        modality="lab",
        observed_at=WEEK_20,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


def _make_patient_events(
    patient_id: str,
    *,
    treatment_arm: int,
    baseline_cd4: float,
    baseline_cd8: float,
    karnofsky_score: float,
    age: float,
    symptomatic_status: bool,
    prior_opportunistic_infection: bool,
    prior_art_exposure_days: float,
    death_time: datetime | None,
    week_20_cd4: float,
    week_96_cd4: float,
    censoring_time_weeks: float = 150.0,
) -> list[ClinicalEvent]:
    treatment_concept = "zidovudine" if treatment_arm == 0 else "zidovudine+didanosine"
    events = [
        _event(
            patient_id,
            concept_id="hiv_positive",
            value=True,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="baseline_cd4",
            value=baseline_cd4,
            unit="cells/mm3",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="baseline_cd8",
            value=baseline_cd8,
            unit="cells/mm3",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="age",
            value=age,
            unit="years",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="karnofsky_score",
            value=karnofsky_score,
            unit="score",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="symptomatic_status",
            value=symptomatic_status,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="prior_opportunistic_infection",
            value=prior_opportunistic_infection,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="prior_art_exposure_days",
            value=prior_art_exposure_days,
            unit="days",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id=treatment_concept,
            value=1.0,
            unit=None,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="cd4_count",
            value=week_20_cd4,
            unit="cells/mm3",
            modality="lab",
            observed_at=WEEK_20,
        ),
        _event(
            patient_id,
            concept_id="cd4_count",
            value=week_96_cd4,
            unit="cells/mm3",
            modality="lab",
            observed_at=WEEK_96,
        ),
    ]
    if death_time is not None:
        events.append(
            _event(
                patient_id,
                concept_id="death",
                value=1.0,
                unit=None,
                modality="endpoint",
                observed_at=death_time,
            )
        )
    else:
        events.append(
            _event(
                patient_id,
                concept_id="administrative_censoring",
                value=1.0,
                unit=None,
                modality="censoring",
                observed_at=ASSIGNMENT_TIME + timedelta(weeks=censoring_time_weeks),
            )
        )
    return events


def _build_small_synthetic_inputs(n_per_arm: int = 15):
    """A small synthetic cohort with a genuine event/censoring mix
    (mirrors `tests/integration/test_run_ablations.py`'s
    `_build_small_synthetic_inputs` exactly)."""
    timelines = []
    treatment_arms: dict[str, int] = {}
    counter = 0
    for treatment_arm in (0, 1):
        for i in range(n_per_arm):
            counter += 1
            patient_id = f"P{counter:04d}"
            had_event = i % 2 == 0
            death_time = (
                ASSIGNMENT_TIME + timedelta(weeks=30.0 + i * 3.0) if had_event else None
            )
            censoring_time_weeks = 140.0 + i * 2.0
            baseline_cd4 = 200.0 + (i * 6) % 300
            events = _make_patient_events(
                patient_id,
                treatment_arm=treatment_arm,
                baseline_cd4=baseline_cd4,
                baseline_cd8=850.0 + (i * 7) % 300,
                karnofsky_score=70.0 + (i % 30),
                age=18.0 + (i % 50),
                symptomatic_status=(i % 3 == 0),
                prior_opportunistic_infection=(i % 5 == 0),
                prior_art_exposure_days=float((i % 4) * 30),
                death_time=death_time,
                week_20_cd4=baseline_cd4 - 10.0,
                week_96_cd4=baseline_cd4 - (40.0 if had_event else 5.0),
                censoring_time_weeks=censoring_time_weeks,
            )
            timelines.append(build_patient_timeline(events))
            treatment_arms[patient_id] = treatment_arm
    return timelines, treatment_arms


@pytest.fixture(scope="module")
def synthetic_inputs():
    # Larger than tests/integration/test_run_ablations.py's n_per_arm=15:
    # run_seed's propensity cross-fitting (even at TEST_PROPENSITY_N_FOLDS)
    # needs enough per-fold training patients for the logistic fit to
    # converge, which run_ablations.py's harness never exercises (it has
    # no propensity model at all). Since the common-support trim (decisions.md
    # Section 26 amendment of 2026-10-12, item B), the cohort must also be
    # large enough that the 14-column propensity fit does not overfit to
    # extreme scores and trim the estimation population below what the
    # outcome regression can be fit on.
    return _build_small_synthetic_inputs(n_per_arm=100)


#: 2-fold rather than the frozen 5-fold cross-fitting: at this fixture's
#: size (~15 patients/arm before observationalization retention), 5-fold
#: cross-fitting can leave a training fold too small per arm for the
#: logistic propensity fit to converge within its iteration cap. The
#: production caller always uses the frozen 5 folds (`propensity_n_folds=None`
#: resolves to it); this override exists only for this test's cohort size.
TEST_PROPENSITY_N_FOLDS = 2


# ---------------------------------------------------------------------------
# run_seed
# ---------------------------------------------------------------------------


def test_run_seed_produces_a_complete_seed_result(synthetic_inputs, target_trial_spec, tmp_path):
    timelines, treatment_arms = synthetic_inputs

    result = run_seed(
        101,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path,
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )

    assert result.seed == 101
    assert result.scale == "risk_difference"
    assert set(dict(result.witness_status)) == {"temporal", "causal", "clinical"}
    for status in dict(result.witness_status).values():
        assert status in {"pass", "fail", "uncertain"}
    assert result.release_decision is not None
    # Ten checkpoints: five GRU replicas + five intervention-aware replicas.
    assert len(result.checkpoint_hashes) == 10
    assert result.resolved_config_hash
    assert result.split_hash
    assert result.experiment_id


def test_run_seed_writes_a_content_hashed_per_seed_directory(
    synthetic_inputs, target_trial_spec, tmp_path
):
    timelines, treatment_arms = synthetic_inputs

    run_seed(
        202,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path,
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )

    seed_dir = tmp_path / "replication" / "seeds" / "202"
    assert seed_dir.is_dir()
    assert (seed_dir / "val_patient_predictions.json").is_file()
    assert (seed_dir / "checkpoints" / "sequence_gru").is_dir()
    assert (seed_dir / "checkpoints" / "intervention_aware").is_dir()


def test_rerunning_the_same_seed_from_an_unchanged_config_reproduces_identical_hashes(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 9.9: rerunning a seed from an unchanged `ResolvedConfig`
    reproduces the same `experiment_id`, `resolved_config_hash`, and
    `split_hash` (two separate, unoccupied result directories so this
    test is about the computed identifiers, not about the duplicate-
    directory refusal `test_rerunning_an_already_populated_seed_directory_refuses`
    covers)."""
    timelines, treatment_arms = synthetic_inputs

    result_1 = run_seed(
        606,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path / "first",
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )
    result_2 = run_seed(
        606,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path / "second",
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )

    assert result_1.experiment_id == result_2.experiment_id
    assert result_1.resolved_config_hash == result_2.resolved_config_hash
    assert result_1.split_hash == result_2.split_hash


def test_rerunning_an_already_populated_seed_directory_refuses(
    synthetic_inputs, target_trial_spec, tmp_path
):
    from infra.registry import DuplicateExperimentError

    timelines, treatment_arms = synthetic_inputs

    run_seed(
        303,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path,
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )

    with pytest.raises(DuplicateExperimentError):
        run_seed(
            303,
            timelines,
            treatment_arms,
            target_trial_spec,
            tmp_path,
            training_config=SHORT_CONFIG,
            negative_control_n_bootstrap=10,
        )


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_retains_every_seed_result_in_every_branch(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 9.5, 9.8: no best/worst-seed selection; every seed's
    value survives aggregation."""
    timelines, treatment_arms = synthetic_inputs

    seed_results = [
        run_seed(
            seed,
            timelines,
            treatment_arms,
            target_trial_spec,
            tmp_path / f"seed_{seed}",
            training_config=SHORT_CONFIG,
            negative_control_n_bootstrap=10,
            propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
        )
        for seed in (404, 505)
    ]

    summary = aggregate(seed_results, CROSS_SEED_STABILITY_CRITERION)

    assert len(summary.seed_results) == 2
    assert {r.seed for r in summary.seed_results} == {404, 505}
    assert summary.stability_criterion == CROSS_SEED_STABILITY_CRITERION.criterion_description
    assert summary.stability_determination in {"stable", "directionally_unstable", "undetermined"}


def test_aggregate_reports_the_criterion_verbatim_alongside_the_determination():
    """Requirement 9.7."""
    from run_replication import SeedResult

    from evaluation.release_gate import ReleaseDecision

    def _dummy_seed_result(seed: int, estimate: float) -> SeedResult:
        return SeedResult(
            seed=seed,
            experiment_id=f"exp{seed}",
            resolved_config_hash=f"hash{seed}",
            split_hash=f"split{seed}",
            checkpoint_hashes=(),
            primary_estimate=estimate,
            ci_low=None,
            ci_high=None,
            scale="risk_difference",
            witness_status=(("temporal", "pass"), ("causal", "pass"), ("clinical", "pass")),
            release_decision=ReleaseDecision(
                released=True,
                overlap_valid=True,
                balance_valid=True,
                uncertainty_valid=True,
                temporal_pass=True,
                causal_pass=True,
                clinical_pass=True,
                reasons=(),
            ),
            factual_metrics={},
            selective_metrics={},
        )

    seed_results = [_dummy_seed_result(seed, -0.1) for seed in (101, 202, 303, 404, 505)]
    summary = aggregate(seed_results, CROSS_SEED_STABILITY_CRITERION)

    assert summary.stability_determination == "stable"
    assert summary.direction_agreement_count == 5
    assert summary.stability_criterion == CROSS_SEED_STABILITY_CRITERION.criterion_description


def test_aggregate_with_a_three_two_split_is_directionally_unstable():
    """A 3-of-5 split does not meet the frozen 4-of-5 criterion."""
    from run_replication import SeedResult

    from evaluation.release_gate import ReleaseDecision

    def _dummy_seed_result(seed: int, estimate: float) -> SeedResult:
        return SeedResult(
            seed=seed,
            experiment_id=f"exp{seed}",
            resolved_config_hash=f"hash{seed}",
            split_hash=f"split{seed}",
            checkpoint_hashes=(),
            primary_estimate=estimate,
            ci_low=None,
            ci_high=None,
            scale="risk_difference",
            witness_status=(("temporal", "pass"), ("causal", "pass"), ("clinical", "pass")),
            release_decision=ReleaseDecision(
                released=True,
                overlap_valid=True,
                balance_valid=True,
                uncertainty_valid=True,
                temporal_pass=True,
                causal_pass=True,
                clinical_pass=True,
                reasons=(),
            ),
            factual_metrics={},
            selective_metrics={},
        )

    estimates = [-0.1, -0.1, -0.1, 0.1, 0.1]
    seeds = (101, 202, 303, 404, 505)
    seed_results = [_dummy_seed_result(s, e) for s, e in zip(seeds, estimates, strict=True)]
    summary = aggregate(seed_results, CROSS_SEED_STABILITY_CRITERION)

    assert summary.stability_determination == "directionally_unstable"
    assert len(summary.seed_results) == 5


def test_a_contradicting_required_sensitivity_variant_blocks_stable():
    """Even with 5-of-5 seed agreement, a contradicting required
    sensitivity variant forces `directionally_unstable`, not `stable`
    (the criterion's conjunction clause)."""
    from run_replication import SeedResult

    from evaluation.release_gate import ReleaseDecision

    def _dummy_seed_result(seed: int, estimate: float) -> SeedResult:
        return SeedResult(
            seed=seed,
            experiment_id=f"exp{seed}",
            resolved_config_hash=f"hash{seed}",
            split_hash=f"split{seed}",
            checkpoint_hashes=(),
            primary_estimate=estimate,
            ci_low=None,
            ci_high=None,
            scale="risk_difference",
            witness_status=(("temporal", "pass"), ("causal", "pass"), ("clinical", "pass")),
            release_decision=ReleaseDecision(
                released=True,
                overlap_valid=True,
                balance_valid=True,
                uncertainty_valid=True,
                temporal_pass=True,
                causal_pass=True,
                clinical_pass=True,
                reasons=(),
            ),
            factual_metrics={},
            selective_metrics={},
        )

    seed_results = [_dummy_seed_result(seed, -0.1) for seed in (101, 202, 303, 404, 505)]

    all_stable_summary = aggregate(
        seed_results,
        CROSS_SEED_STABILITY_CRITERION,
        sensitivity_overall_stabilities=["stable"] * 7 + ["stable"],
    )
    assert all_stable_summary.stability_determination == "stable"

    contradicting_summary = aggregate(
        seed_results,
        CROSS_SEED_STABILITY_CRITERION,
        sensitivity_overall_stabilities=["stable"] * 7 + ["unstable"],
    )
    assert contradicting_summary.stability_determination == "directionally_unstable"
    assert len(contradicting_summary.seed_results) == 5


def test_write_replication_summary_writes_the_aggregate_to_disk(tmp_path):
    import json as _json

    from run_replication import SeedResult, write_replication_summary

    from evaluation.release_gate import ReleaseDecision

    seed_result = SeedResult(
        seed=101,
        experiment_id="exp101",
        resolved_config_hash="hash101",
        split_hash="split101",
        checkpoint_hashes=(),
        primary_estimate=-0.1,
        ci_low=None,
        ci_high=None,
        scale="risk_difference",
        witness_status=(("temporal", "pass"), ("causal", "pass"), ("clinical", "pass")),
        release_decision=ReleaseDecision(
            released=True,
            overlap_valid=True,
            balance_valid=True,
            uncertainty_valid=True,
            temporal_pass=True,
            causal_pass=True,
            clinical_pass=True,
            reasons=(),
        ),
        factual_metrics={},
        selective_metrics={},
    )
    summary = aggregate([seed_result], CROSS_SEED_STABILITY_CRITERION)

    written_path = write_replication_summary(summary, tmp_path)

    assert written_path == tmp_path / "replication" / "replication_summary.json"
    payload = _json.loads(written_path.read_text())
    assert payload["stability_criterion"] == CROSS_SEED_STABILITY_CRITERION.criterion_description
    assert len(payload["seed_results"]) == 1


def test_no_aggregate_function_signature_accepts_a_seed_subset():
    """Requirement 9.5: `aggregate`'s only seed-bearing parameter is the
    full `seed_results` sequence -- there is no `best_of`/`exclude`/
    similar parameter."""
    import inspect

    signature = inspect.signature(aggregate)
    param_names = set(signature.parameters)
    assert param_names == {"seed_results", "criterion", "sensitivity_overall_stabilities"}


# ---------------------------------------------------------------------------
# Task 18.2: record_seed_determinism / readiness-ledger wiring
# ---------------------------------------------------------------------------


def test_record_seed_determinism_appends_to_the_ledger_on_matching_hashes(
    synthetic_inputs, target_trial_spec, tmp_path
):
    from run_replication import record_seed_determinism

    from infra.readiness import DETERMINISM_KIND, load_ledger_entries

    timelines, treatment_arms = synthetic_inputs

    first = run_seed(
        606,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path / "first",
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )
    second = run_seed(
        606,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path / "second",
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )

    record_seed_determinism(first, second, tmp_path / "first")

    entries = load_ledger_entries(tmp_path / "first" / "replication")
    determinism_entries = [e for e in entries if e.kind == DETERMINISM_KIND]
    assert len(determinism_entries) == 1
    assert determinism_entries[0].run_id == "606"


def test_record_seed_determinism_raises_on_a_seed_mismatch(
    synthetic_inputs, target_trial_spec, tmp_path
):
    from run_replication import record_seed_determinism

    timelines, treatment_arms = synthetic_inputs

    result_a = run_seed(
        808,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path / "a",
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )
    result_b = run_seed(
        909,
        timelines,
        treatment_arms,
        target_trial_spec,
        tmp_path / "b",
        training_config=SHORT_CONFIG,
        negative_control_n_bootstrap=10,
        propensity_n_folds=TEST_PROPENSITY_N_FOLDS,
    )

    with pytest.raises(ValueError):
        record_seed_determinism(result_a, result_b, tmp_path / "a")


def test_record_seed_determinism_raises_non_determinism_error_on_a_hash_mismatch():
    from run_replication import NonDeterminismError, SeedResult, record_seed_determinism

    from evaluation.release_gate import ReleaseDecision

    def _dummy(resolved_config_hash: str) -> SeedResult:
        return SeedResult(
            seed=1,
            experiment_id="exp1",
            resolved_config_hash=resolved_config_hash,
            split_hash="split1",
            checkpoint_hashes=(("a", "hash-a"),),
            primary_estimate=0.0,
            ci_low=None,
            ci_high=None,
            scale="risk_difference",
            witness_status=(),
            release_decision=ReleaseDecision(
                released=False,
                overlap_valid=False,
                balance_valid=False,
                uncertainty_valid=False,
                temporal_pass=False,
                causal_pass=False,
                clinical_pass=False,
                reasons=(),
            ),
            factual_metrics={},
            selective_metrics={},
        )

    with pytest.raises(NonDeterminismError):
        record_seed_determinism(_dummy("hash-1"), _dummy("hash-2"), Path("/tmp/does-not-matter"))
