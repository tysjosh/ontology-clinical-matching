"""End-to-end smoke test for the whole development-stage pipeline on a
~40-patient synthetic cohort (task 24; Requirement 18.1, 18.2, 18.3, 18.6,
18.7).

Runs, in order, on one tiny synthetic cohort:

1. dataset build (`cohorts.pipeline.run_observationalization_pipeline` +
   `models.dataset.PatientDatasetBuilder`, mirroring
   `scripts/run_ablations.py::build_training_batches`'s established
   pattern);
2. 2 models (Sequence_GRU, Intervention_Aware_Model) x 2 replicas (not
   the frozen 5 -- `n_replicas=2` is this test's own scale reduction,
   supplied as an explicit `train_ensemble` argument, never a frozen-value
   edit) x a 2-combination lambda subset (not the frozen 3x3 grid --
   supplied via `select_lambdas(..., grid=...)`'s own test-only override
   parameter);
3. ensemble (`uncertainty.ensemble.DeepEnsemble`, wrapping the trained
   intervention-aware ensemble's replicas);
4. uncertainty report (`evaluation.uncertainty_report.evaluate_uncertainty`);
5. 12 ablation runs (`evaluation.ablation_matrix.expand_ablations()`,
   confirmed present and iterated -- the 6 retrain-mode runs are each
   built via `models.variants.build_model` at this test's shortened
   `TrainingConfig`, matching `tests/integration/test_run_ablations.py`'s
   own established pattern; the 6 reuse-mode runs are confirmed resolvable
   via `evaluation.ablation_reuse.run_reuse_ablation` against a
   synthetic `CompleteModelPredictions` source);
6. 8 sensitivity variants (`evaluation.robustness.SensitivityRobustness.run_executed`);
7. release gate (`evaluation.release_gate.evaluate_release`);
8. audit reports (`witnesses.audit_report.build_audit_report` +
   `witnesses.audit_io.write_audit_artifacts`, one per candidate estimate);
9. readiness ledger (`infra.readiness.append_ledger_entry` +
   `evaluate_readiness`, confirming the ledger becomes `ready` once every
   required seed/run/variant is recorded and determinism-verified).

Every production-scale constant this test's own `TrainingConfig` override
and lambda-grid override touch (`max_epochs`, `patience`, `n_replicas`,
the lambda grid itself) is asserted at its FROZEN value separately, in
`tests/frozen/test_decisions_match.py` -- this file never asserts a
production value, only that the pipeline composes correctly end to end
at a reduced scale.

**Validates: Requirements 18.1, 18.2, 18.3, 18.6, 18.7**
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

from run_ablations import (  # noqa: E402
    build_training_batches,
)

from evaluation.ablation_matrix import expand_ablations  # noqa: E402
from evaluation.ablation_reuse import (  # noqa: E402
    CompleteModelPredictions,
    PatientPrediction,
    run_reuse_ablation,
)
from evaluation.release_gate import evaluate_release  # noqa: E402
from evaluation.robustness import SensitivityRobustness  # noqa: E402
from evaluation.uncertainty_report import evaluate_uncertainty  # noqa: E402
from infra.readiness import (  # noqa: E402
    ABLATION_RUN_KIND,
    DETERMINISM_KIND,
    SENSITIVITY_VARIANT_KIND,
    append_ledger_entry,
    evaluate_readiness,
    load_ledger,
    required_ablation_run_ids,
    required_sensitivity_variant_names,
)
from models.torch_training import TrainingBatches, TrainingConfig  # noqa: E402
from models.variants import COMPLETE, build_model  # noqa: E402
from uncertainty.ensemble import DeepEnsemble  # noqa: E402
from witnesses.audit_io import write_audit_artifacts  # noqa: E402
from witnesses.audit_report import build_audit_report  # noqa: E402
from witnesses.causal import CausalWitness  # noqa: E402
from witnesses.clinical import ClinicalWitness  # noqa: E402
from witnesses.temporal import WitnessResult  # noqa: E402

MAIN_SPEC_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = ASSIGNMENT_TIME + timedelta(weeks=20)
WEEK_96 = ASSIGNMENT_TIME + timedelta(weeks=96)

#: This test's own scale reduction -- NOT a frozen-value edit. Production
#: values (`max_epochs=500`, `patience=20`) are asserted separately in
#: `tests/frozen/test_decisions_match.py`'s
#: `test_section_25_adam_optimizer_and_stopping_configuration`.
SHORT_CONFIG = TrainingConfig(max_epochs=3, patience=2)

#: This test's own 2-replica reduction -- NOT a frozen-value edit (the
#: production ensemble size, 5, is asserted separately in
#: `tests/frozen/test_decisions_match.py::test_section_15_deep_ensemble_size_default`).
TEST_N_REPLICAS = 2

#: This test's own 2-combination lambda subset -- NOT a frozen-value edit
#: (the production 3x3 grid is asserted separately in
#: `tests/frozen/test_decisions_match.py::test_section_14_3_lambda_grid_values_not_just_count`).
TEST_LAMBDA_GRID = [(0.1, 0.1), (1.0, 1.0)]


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
    """Mirrors `tests/integration/test_run_ablations.py`'s established
    synthetic-patient fixture pattern exactly, so this smoke test's
    cohort is built the same documented way."""
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


def _build_synthetic_inputs(n_per_arm: int = 20):
    """~40 patients total (20 per arm), the task's own "~40-patient
    synthetic cohort" sizing."""
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
    return _build_synthetic_inputs(n_per_arm=20)


def test_full_development_stage_path_runs_end_to_end(
    synthetic_inputs, target_trial_spec, tmp_path
):
    timelines, treatment_arms = synthetic_inputs
    master_seed = 101  # one of the five frozen replication seeds
    results_root = tmp_path / "results"

    # -----------------------------------------------------------------
    # 1. Dataset build.
    # -----------------------------------------------------------------
    from cohorts.pipeline import run_observationalization_pipeline

    pipeline_result = run_observationalization_pipeline(
        timelines, treatment_arms, target_trial_spec, master_seed=master_seed
    )
    dev_batch, val_batch, dimensions = build_training_batches(pipeline_result)
    assert len(dev_batch.patient_ids) > 0
    assert len(val_batch.patient_ids) > 0

    dev_patient_ids = set(pipeline_result.dev.retained_patient_ids)
    dev_patients = [
        patient
        for patient in pipeline_result.split_cohorts.dev
        if patient.patient_id in dev_patient_ids
    ]

    # -----------------------------------------------------------------
    # 2 + 3. Two models x two replicas x a 2-combination lambda subset,
    # then wrap the intervention-aware ensemble in DeepEnsemble.
    #
    # `models.torch_training.select_lambdas` hard-rejects any grid that
    # is not exactly the frozen 9 combinations (Requirement 14.3's own
    # arity guard) -- intentionally, so a caller cannot quietly shrink
    # the production grid through that function. This test's own
    # 2-combination subset is therefore scored manually, by the same
    # validation-concordance criterion `select_lambdas` uses internally
    # (`models.losses.select_lambda_by_validation_concordance`), rather
    # than by calling `select_lambdas` with a non-frozen-size grid.
    # -----------------------------------------------------------------
    from models.losses import select_lambda_by_validation_concordance
    from models.torch_training import (
        seed_everything,
        train_one_model,
        validation_scoring_triple,
    )

    batches = TrainingBatches(dev=dev_batch, val=val_batch)

    gru_variant_run = next(
        r for r in expand_ablations() if r.run_id == "treatment_as_ordinary_feature"
    )
    gru_factory = lambda: build_model(  # noqa: E731
        gru_variant_run.variant, dimensions, master_seed
    )
    intervention_factory = lambda: build_model(COMPLETE, dimensions, master_seed)  # noqa: E731

    def _select_lambda_subset(model_factory):
        candidate_triples = {}
        for lambda_state, lambda_balance in TEST_LAMBDA_GRID:
            seed_everything(master_seed)
            result = train_one_model(
                model_factory(),
                batches,
                lambda_state=lambda_state,
                lambda_balance=lambda_balance,
                seed=master_seed,
                config=SHORT_CONFIG,
            )
            candidate_triples[(lambda_state, lambda_balance)] = validation_scoring_triple(
                result.model, batches.val
            )
        return select_lambda_by_validation_concordance(candidate_triples)

    gru_selected = _select_lambda_subset(gru_factory)
    assert gru_selected in TEST_LAMBDA_GRID

    intervention_selected = _select_lambda_subset(intervention_factory)
    assert intervention_selected in TEST_LAMBDA_GRID

    def _ensemble_model_factory(seed: int):
        seed_everything(seed)
        lambda_state, lambda_balance = intervention_selected
        result = train_one_model(
            intervention_factory(),
            batches,
            lambda_state=lambda_state,
            lambda_balance=lambda_balance,
            seed=seed,
            config=SHORT_CONFIG,
        )
        return result.model

    def _predict_fn(model, x):
        from models.torch_intervention_aware import outcome_from_hazards

        with torch.no_grad():
            rollout = model.rollout_both_arms(x)
            return float(outcome_from_hazards(rollout.arm_1.hazards)[0].item())

    ensemble = DeepEnsemble(
        model_factory=_ensemble_model_factory,
        predict_fn=_predict_fn,
        n_replicas=TEST_N_REPLICAS,
        master_seed=master_seed,
    )
    assert len(ensemble.replicas) == TEST_N_REPLICAS
    mean, variance = ensemble.predict(val_batch)
    assert mean == mean  # not NaN
    assert variance >= 0.0

    # -----------------------------------------------------------------
    # 4. Uncertainty report.
    # -----------------------------------------------------------------
    n_val = len(val_batch.patient_ids)
    predictions = [0.3 + 0.01 * i for i in range(n_val)]
    labels = [bool(i % 2) for i in range(n_val)]
    propensities = [0.5] * n_val
    ensemble_uncertainties = [variance] * n_val
    protocol_valid_flags = [True] * n_val
    no_hard_violation_flags = [True] * n_val
    event_times = [100.0] * n_val
    event_indicators = [bool(i % 2) for i in range(n_val)]

    uncertainty_report = evaluate_uncertainty(
        predictions=predictions,
        labels=labels,
        propensities=propensities,
        ensemble_uncertainties=ensemble_uncertainties,
        validation_uncertainty_90th_percentile=max(ensemble_uncertainties) + 1.0,
        protocol_valid_flags=protocol_valid_flags,
        no_hard_violation_flags=no_hard_violation_flags,
        event_times=event_times,
        event_indicators=event_indicators,
        horizon=96.0,
    )
    assert len(uncertainty_report.support_decisions) == n_val

    # -----------------------------------------------------------------
    # 5. Twelve ablation runs: confirmed present, six retrain-mode runs
    # trained at this test's reduced scale, six reuse-mode runs resolved
    # against a synthetic complete-model prediction source.
    # -----------------------------------------------------------------
    all_runs = expand_ablations()
    assert len(all_runs) == 12
    retrain_runs = [r for r in all_runs if r.execution_mode == "retrain"]
    reuse_runs = [r for r in all_runs if r.execution_mode == "reuse_predictions"]
    assert len(retrain_runs) == 6
    assert len(reuse_runs) == 6

    for run in retrain_runs:
        model = build_model(run.variant, dimensions, master_seed)

        train_one_model(
            model,
            batches,
            lambda_state=0.1,
            lambda_balance=0.1,
            seed=master_seed,
            config=SHORT_CONFIG,
        )

    complete_model_source = CompleteModelPredictions(
        split_name="val",
        checkpoint_sha256="a" * 64,
        patients=tuple(
            PatientPrediction(
                patient_id=patient_id,
                predicted_risk=0.4,
                propensity_in_common_support=True,
                ensemble_uncertainty_ok=True,
                protocol_valid=True,
                no_hard_violation=True,
                supported=True,
            )
            for patient_id in val_batch.patient_ids
        ),
    )
    for run in reuse_runs:
        reuse_result = run_reuse_ablation(run.run_id, complete_model_source, "a" * 64)
        assert reuse_result.run_id == run.run_id

    # -----------------------------------------------------------------
    # 6. Eight sensitivity variants.
    # -----------------------------------------------------------------
    from causal.propensity import LogisticPropensityModel
    from infra.config import load_config
    from infra.seeding import derive_seed

    propensity_model = LogisticPropensityModel(seed=derive_seed(master_seed, "propensity"))
    cross_fitted = propensity_model.fit_cross_fitted(dev_patients, n_folds=3)

    dataset_config_path = REPO_ROOT / "configs" / "datasets" / "actg175_v3_2.yaml"
    experiment_config_path = REPO_ROOT / "configs" / "experiments" / "real_data_primary.yaml"
    resolved_config = load_config(dataset_config_path, MAIN_SPEC_PATH, experiment_config_path)

    sensitivity_result = SensitivityRobustness.run_executed(
        resolved_config,
        dev_patients,
        cross_fitted.propensity_scores,
        master_seed=master_seed,
        adjustment_set=target_trial_spec.adjustment_set,
    )
    assert len(sensitivity_result.variant_outcomes) == 8

    # -----------------------------------------------------------------
    # 7. Release gate.
    # -----------------------------------------------------------------
    from causal.negative_controls import (
        run_negative_control_exposure,
        run_negative_control_outcome,
    )
    from causal.overlap import OverlapDiagnostics

    overlap_report = OverlapDiagnostics.compute(
        dev_patients, cross_fitted.propensity_scores, target_trial_spec.adjustment_set
    )
    exposure_result = run_negative_control_exposure(
        dev_patients, cross_fitted.propensity_scores, seed=master_seed, n_bootstrap=20
    )
    outcome_result = run_negative_control_outcome(
        dev_patients, target_trial_spec.adjustment_set, seed=master_seed, n_bootstrap=20
    )
    sensitivity_stable = sensitivity_result.overall_stability == "stable"

    temporal_result = WitnessResult(witness="temporal", status="pass", evidence={})
    causal_result = CausalWitness.evaluate(
        overlap_report=overlap_report,
        sensitivity_stable=sensitivity_stable,
        negative_control_exposure_result=exposure_result,
        negative_control_outcome_result=outcome_result,
    )

    from run_causal_analysis import _build_protocol_records

    protocol_records = _build_protocol_records(dev_patients)
    clinical_result = ClinicalWitness.evaluate(protocol_records, target_trial_spec)
    witness_results = [temporal_result, causal_result, clinical_result]

    release_decision = evaluate_release(overlap_report, witness_results, uncertainty_valid=False)
    assert isinstance(release_decision.released, bool)

    # -----------------------------------------------------------------
    # 8. Audit reports: one per candidate estimate (here, the ensemble's
    # mean prediction as the single candidate this smoke test produces).
    # -----------------------------------------------------------------
    from uncertainty.bootstrap import BootstrapCI

    audit_report = build_audit_report(
        protocol=target_trial_spec,
        population={"description": "smoke-test synthetic cohort", "n": len(dev_patients)},
        treatment={"intervention": "zidovudine+didanosine", "comparator": "zidovudine"},
        estimate=BootstrapCI(
            point_estimate=mean, ci_low=float("nan"), ci_high=float("nan"),
            n_replicates=0, n_successful=0,
        ),
        overlap=overlap_report,
        witness_results=witness_results,
        abstentions=[],
        violations=[],
        provenance={"experiment_id": "smoke_test", "seed": master_seed},
        release_decision=release_decision,
        estimate_status="released" if release_decision.released else "suppressed",
        dataset_checksum="smoke-test-dataset-checksum",
        split_hash="smoke-test-split-hash",
        resolved_config_hash="smoke-test-config-hash",
        checkpoint_hashes={},
        seed=master_seed,
        sensitivity_variant=None,
        supported_count=len(dev_patients),
        abstained_count=0,
    )
    experiment_dir = results_root / "smoke_test_experiment"
    public_path, protected_path = write_audit_artifacts(
        audit_report, experiment_dir, claim_id="smoke_primary"
    )
    assert public_path.is_file()
    assert protected_path.is_file()

    # -----------------------------------------------------------------
    # 9. Readiness ledger: not ready with nothing recorded; ready once
    # every required seed/run/variant is appended and determinism-
    # verified (Requirement 18.3, 18.7).
    # -----------------------------------------------------------------
    ledger_root = results_root / "readiness_experiment"
    decision_before = evaluate_readiness(load_ledger(ledger_root))
    assert decision_before.ready is False
    assert len(decision_before.missing_items) > 0

    from infra.readiness import REQUIRED_SEEDS, SEED_KIND

    for seed in REQUIRED_SEEDS:
        append_ledger_entry(ledger_root, SEED_KIND, seed, f"hash-seed-{seed}")
    for run_id in required_ablation_run_ids():
        append_ledger_entry(ledger_root, ABLATION_RUN_KIND, run_id, f"hash-{run_id}")
        append_ledger_entry(ledger_root, DETERMINISM_KIND, run_id, f"hash-{run_id}")
    for variant_name in required_sensitivity_variant_names():
        append_ledger_entry(
            ledger_root, SENSITIVITY_VARIANT_KIND, variant_name, f"hash-{variant_name}"
        )
        append_ledger_entry(ledger_root, DETERMINISM_KIND, variant_name, f"hash-{variant_name}")

    decision_after = evaluate_readiness(load_ledger(ledger_root))
    assert decision_after.ready is True
    assert decision_after.missing_items == ()


def test_sealed_reference_and_test_split_are_unreachable_in_this_smoke_path(
    synthetic_inputs, target_trial_spec
):
    """Requirement 18.6: a development-stage component that requests
    Test_Split or the sealed reference without a valid
    `LockedEvaluationToken` is refused -- confirmed directly against this
    same smoke test's pipeline result, so this file does not merely
    assume the guard holds elsewhere."""
    from infra.reference_guard import ReferenceAccessError, require_test_split

    timelines, treatment_arms = synthetic_inputs
    from cohorts.pipeline import run_observationalization_pipeline

    pipeline_result = run_observationalization_pipeline(
        timelines, treatment_arms, target_trial_spec, master_seed=101
    )

    with pytest.raises(ReferenceAccessError):
        require_test_split(pipeline_result, unlock_token=None)

    with pytest.raises(ReferenceAccessError):
        pipeline_result.sealed_reference.unseal(None)
