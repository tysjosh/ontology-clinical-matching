#!/usr/bin/env python
"""Full causal-analysis pipeline on the development split: propensity,
all six causal baselines, overlap diagnostics, negative controls, and the
three witnesses -- persisting real experiment artifacts to
`results/<experiment_id>/` (follow-up to task 18's real-data wiring).

`scripts/run_stage.py --stage real_data_pipeline` stops after fitting the
predictive Cox baseline; it does not run any causal estimator, negative
control, overlap diagnostic, or witness, and it writes nothing to disk
(`results/` stays empty). This script picks up immediately after that
stage's cohort construction/observationalization/leakage-audit sequence
and runs everything task 18's scope deferred:

1. Cross-fit the primary logistic propensity model on the development
   split's retained (observationalized) patients
   (`causal.propensity.LogisticPropensityModel.fit_cross_fitted`).
2. Run all six frozen causal baselines
   (`causal.baselines.run_all_baselines`) using those cross-fitted
   propensity scores.
3. Compute overlap/support diagnostics
   (`causal.overlap.OverlapDiagnostics.compute`).
4. Run both negative controls
   (`causal.negative_controls.run_negative_control_exposure`/
   `run_negative_control_outcome`).
5. Evaluate the three witnesses (temporal, causal, clinical) and aggregate
   them under the frozen `ALL_REQUIRED` policy
   (`witnesses.aggregation.aggregate`).
6. Register the experiment through `infra.registry.ExperimentRegistry`,
   which is the sole owner of run identity: it computes
   `compute_experiment_id(resolved_config)` and is the only place a
   duplicate run is refused. This script composes that `ResolvedConfig`
   from the three frozen YAML sources (see `_resolve_config_for_run`) and
   mints no identifier of its own. Then write a
   privacy-scanned `manifest.json` (`infra.manifest.ManifestWriter`), the
   cohort-flow table (`cohorts.cohort_flow.build_cohort_flow_table`), and
   persist the sealed internal randomized reference to
   `reference/internal_randomized_reference.sealed.json`
   (`cohorts.sealed_reference.SealedReference.write_sealed_artifact`) --
   never unsealed here; this script never mints or holds a
   `LockedEvaluationToken`.

What this script deliberately still does NOT do
----------------------------------------------------

- It does not run the neural intervention-aware model or the sequence-GRU
  baseline (`src/models/intervention_aware.py`,
  `src/models/sequence_baseline.py`) -- those require a training loop this
  script does not implement; only the six *causal* baselines
  (`causal.baselines`) and the *predictive* Cox baseline
  (`models.non_neural_baseline.CoxBaseline`, already run by
  `run_stage.py --stage real_data_pipeline`) are exercised.
- It does not run the ablation matrix (`scripts/run_ablations.py`) --
  that remains a separately scoped, deferred follow-up. It does run the
  eight frozen sensitivity variants (`src/evaluation/robustness.py`'s
  `SensitivityRobustness.run_executed`, task 15.1), persisting
  `robustness/sensitivity_stability.json` and feeding the real
  `overall_stability` boolean to the causal witness.
- It does not unseal the internal randomized reference, mint a
  `LockedEvaluationToken`, or run `scripts/check_go_no_go.py` (task 16,
  which does not exist in this repository) -- the sealed reference is
  written to disk with its real value, but reading it back before Stage
  17 still raises `ReferenceAccessError` (see
  `cohorts.sealed_reference.read_sealed_artifact`).
- It does not generate the final paper reporting tables/figures
  (`scripts/run_paper.py`) -- this script's manifest/cohort/reference
  artifacts are themselves inputs a future reporting stage would consume,
  not the reporting output itself.

Usage::

    python scripts/run_causal_analysis.py
    python scripts/run_causal_analysis.py --real-data-path data/raw/actg175_v3_2 \
        --results-root /tmp/results --master-seed 20240101
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from causal.baselines import run_all_baselines
from causal.negative_controls import (
    run_negative_control_exposure,
    run_negative_control_outcome,
)
from causal.overlap import OverlapDiagnostics, trim_to_common_support
from causal.propensity import LogisticPropensityModel
from cohorts.cohort_construction import PatientRecord
from cohorts.cohort_flow import build_cohort_flow_table
from cohorts.pipeline import run_observationalization_pipeline
from cohorts.sealed_reference import unadjusted_cox_reference_estimator
from data.loaders.actg175 import Actg175DataNotFoundError, load_actg175
from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
from evaluation.release_gate import evaluate_release
from evaluation.robustness import (
    SensitivityRobustness,
    SensitivityStabilityResult,
    SupportPopulation,
    build_stronger_selection_cohort,
)
from infra.config import ResolvedConfig, load_config
from infra.manifest import ManifestPrivacyError, ManifestWriter, resolve_git_provenance
from infra.readiness import SENSITIVITY_VARIANT_KIND, append_ledger_entry
from infra.registry import DuplicateExperimentError, ExperimentRegistry, canonical_json
from infra.seeding import derive_seed
from protocols.consistency_checker import PatientProtocolRecord
from protocols.eligibility_dsl import PatientTimeline as EligibilityPatientTimeline
from protocols.schema import TargetTrialSpec
from uncertainty.bootstrap import BootstrapCI
from witnesses.aggregation import AggregateDecision, WitnessPolicy, aggregate
from witnesses.audit_io import write_audit_artifacts
from witnesses.audit_report import build_audit_report
from witnesses.causal import CausalWitness
from witnesses.clinical import ClinicalWitness
from witnesses.temporal import TemporalWitness, WitnessResult

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"
DEFAULT_TRIAL_CONFIG = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
DEFAULT_DATASET_CONFIG = REPO_ROOT / "configs" / "datasets" / "actg175_v3_2.yaml"
DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiments" / "real_data_primary.yaml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

#: Number of cross-fitting folds for the primary propensity model
#: (`decisions.md` Section 10: "Five-fold cross-fitting is used inside
#: development+validation for causal nuisance models").
PROPENSITY_N_FOLDS = 5

#: Bootstrap replicate count for both negative controls. Kept modest
#: (documented in `causal.negative_controls`'s module docstring as "modest
#: given pure-Python") since this script runs both controls on every
#: invocation, not only in a dedicated test.
NEGATIVE_CONTROL_N_BOOTSTRAP = 200


def _resolve_config_for_run(
    dataset_config_path: Path,
    trial_config_path: Path,
    experiment_config_path: Path,
    master_seed: int,
    negative_control_n_bootstrap: int,
) -> ResolvedConfig:
    """Compose the `ResolvedConfig` this run's identity is computed from.

    This script used to mint its own identifier from a hand-rolled content
    hash over `(dataset_checksum, trial_spec_version, master_seed,
    negative_control_n_bootstrap)`, which made two identity schemes exist
    at once. `infra.registry.compute_experiment_id` is now the only one
    (`design.md` Section 2.6 tension 6), and every input the deleted
    function covered is hashed through this resolved config instead:

    - **dataset checksum** -- `resolved.dataset.checksum_sha256`, from the
      frozen dataset contract. This is the same value as the loaded
      tables' `checksum_sha256`: `data.loaders.actg175.load_actg175`
      verifies the computed checksum against this contract and raises
      `ChecksumMismatchError` on any disagreement, so a run whose real
      data differs from the contract does not reach identity computation
      at all.
    - **trial-spec version** -- `resolved.target_trial.version`, from the
      same target-trial YAML this script validates into a
      `protocols.schema.TargetTrialSpec` for execution.
    - **master seed** -- `resolved.experiment.seed`, overridden from the
      CLI/caller argument so the argument, not the file, decides the run.
    - **negative-control bootstrap count** --
      `resolved.experiment.negative_control_n_bootstrap`, the field added
      for exactly this purpose.

    The two overrides are applied to the loaded `ExperimentConfig` rather
    than read from the file alone, so `--master-seed` and
    `--negative-control-n-bootstrap` remain the run's real inputs *and*
    are inside the hash.
    """
    resolved = load_config(dataset_config_path, trial_config_path, experiment_config_path)
    return ResolvedConfig(
        dataset=resolved.dataset,
        target_trial=resolved.target_trial,
        experiment=resolved.experiment.model_copy(
            update={
                "seed": master_seed,
                "negative_control_n_bootstrap": negative_control_n_bootstrap,
            }
        ),
    )


def _adapt_to_eligibility_timeline(patient: PatientRecord) -> EligibilityPatientTimeline:
    """Bridge one retained patient's canonical timeline into
    `protocols.eligibility_dsl.PatientTimeline`, mirroring
    `cohorts.cohort_construction._adapt_to_eligibility_timeline` (a private
    helper in that module; reimplemented locally here rather than reaching
    into it, following the same pattern `evaluation.robustness` already
    uses for `_prior_art_status`).
    """
    return EligibilityPatientTimeline(
        baseline_covariates=dict(patient.timeline.static_covariates),
        assignment_time=(
            min(event.observed_at for event in patient.timeline.treatment_events)
            if patient.timeline.treatment_events
            else None
        ),
    )


def _build_protocol_records(patients: list[PatientRecord]) -> list[PatientProtocolRecord]:
    """Build one `PatientProtocolRecord` per retained patient, for the
    clinical witness (`witnesses.clinical.ClinicalWitness.evaluate`)."""
    records = []
    for patient in patients:
        eligibility_timeline = _adapt_to_eligibility_timeline(patient)
        treatment_start_time = (
            min(event.observed_at for event in patient.timeline.treatment_events)
            if patient.timeline.treatment_events
            else None
        )
        records.append(
            PatientProtocolRecord(
                patient_id=patient.patient_id,
                eligibility_timeline=eligibility_timeline,
                assigned_arm=patient.treatment_arm,
                endpoint_record=None,
                treatment_start_time=treatment_start_time,
            )
        )
    return records


def _run_temporal_witness(
    patients: list[PatientRecord],
    spec: TargetTrialSpec,
    held_out_cd4_cd8_mae: float | None = None,
    factual_baseline_cd4_cd8_mae: float | None = None,
) -> WitnessResult:
    """Evaluate the temporal witness on the retained cohort.

    The held-out longitudinal-error inputs (`held_out_cd4_cd8_mae`,
    `factual_baseline_cd4_cd8_mae`) need both trained neural models. This
    script trains neither, so it leaves them `None` and
    `TemporalWitness.evaluate` reports `"uncertain"` rather than guessing.
    `scripts/run_replication.py`, which trains both, passes real values.
    """
    timelines_and_specs = [
        (_adapt_to_eligibility_timeline(patient), spec) for patient in patients
    ]
    assignment_times = [
        min(event.observed_at for event in patient.timeline.treatment_events)
        for patient in patients
        if patient.timeline.treatment_events
    ]
    # event_times/assignment_times must be paired 1:1; only patients with a
    # recorded treatment-start time contribute to this specific check.
    paired_event_times = [
        patient.endpoint_result.event_time
        for patient in patients
        if patient.timeline.treatment_events
    ]
    landmark_timestamps = [
        sorted(
            {event.observed_at for event in patient.timeline.timestamped_observations}
            | {
                min(event.observed_at for event in patient.timeline.treatment_events)
            }
        )
        for patient in patients
        if patient.timeline.treatment_events
    ]

    return TemporalWitness.evaluate(
        timelines_and_specs=timelines_and_specs,
        event_times=paired_event_times,
        assignment_times=assignment_times,
        landmark_timestamps=landmark_timestamps,
        held_out_cd4_cd8_mae=held_out_cd4_cd8_mae,
        factual_baseline_cd4_cd8_mae=factual_baseline_cd4_cd8_mae,
    )


def _split_hash(patient_ids: list[str]) -> str:
    """A deterministic content hash of this run's retained dev-split
    patient-id set, for the audit report's `split_hash` field (task
    17.2). No shared split-hashing helper exists in this codebase yet
    (`infra.registry.canonical_json` hashes a `ResolvedConfig`, not a
    patient-id set) -- this is a small, local, order-independent hash
    (sorted ids, joined, SHA-256) rather than a new shared module, since
    no second caller needs it yet.
    """
    joined = "\n".join(sorted(patient_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _audit_bootstrap_ci(estimate: Any) -> BootstrapCI:
    """Wrap one `causal.baselines.CausalEstimate` as a `BootstrapCI` for
    `build_audit_report` (task 17.2).

    This script does not run `uncertainty.bootstrap.patient_level_bootstrap`
    over any of the six causal baselines -- that is a separately-scoped
    concern (exercised by `causal.negative_controls`'s own inline bootstrap
    and, eventually, the locked-evaluation runner's paired bootstrap). So
    there is no real confidence interval to report here yet; `ci_low`/
    `ci_high` are set to NaN and `n_replicates`/`n_successful` to `0`,
    documented explicitly as a placeholder rather than silently reusing
    the point estimate as a fake zero-width interval (which would read as
    a real, if degenerate, computed CI).
    """
    return BootstrapCI(
        point_estimate=estimate.estimate,
        ci_low=float("nan"),
        ci_high=float("nan"),
        n_replicates=0,
        n_successful=0,
    )


def run_causal_analysis(
    real_data_path: Path,
    trial_config_path: Path,
    results_root: Path,
    master_seed: int,
    negative_control_n_bootstrap: int = NEGATIVE_CONTROL_N_BOOTSTRAP,
    dataset_config_path: Path = DEFAULT_DATASET_CONFIG,
    experiment_config_path: Path = DEFAULT_EXPERIMENT_CONFIG,
) -> dict[str, Any]:
    """Run the full causal-analysis sequence and persist real artifacts.

    Args:
        negative_control_n_bootstrap: bootstrap replicate count for both
            negative controls. Defaults to `NEGATIVE_CONTROL_N_BOOTSTRAP`
            (200); tests may pass a smaller value to keep runtime down,
            since each negative control refits a model once per replicate.
        dataset_config_path: the frozen dataset contract, composed into
            this run's `ResolvedConfig` (see `_resolve_config_for_run`).
        experiment_config_path: the experiment config composed into this
            run's `ResolvedConfig`; its `seed` and
            `negative_control_n_bootstrap` are overridden by the two
            arguments above.

    Returns a summary dict (also what gets printed/logged); raises on any
    failure rather than swallowing it, since this is a library function a
    test can call directly (mirroring `scripts/run_ablations.py`'s
    `run_all_ablations` convention) -- `main()` below is the only place
    that catches exceptions and converts them to a CLI exit code.
    """
    resolved_config = _resolve_config_for_run(
        dataset_config_path,
        trial_config_path,
        experiment_config_path,
        master_seed,
        negative_control_n_bootstrap,
    )

    with trial_config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    spec = TargetTrialSpec.model_validate(raw)

    tables = load_actg175(real_data_path)
    timelines, treatment_arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)

    pipeline_seed = derive_seed(master_seed, "causal_analysis_pipeline")
    result = run_observationalization_pipeline(
        timelines,
        treatment_arms,
        spec,
        pipeline_seed,
        reference_estimator=unadjusted_cox_reference_estimator,
        precomputed_endpoint_results=endpoint_results,
    )

    dev_patient_ids = set(result.dev.retained_patient_ids)
    dev_patients = [
        patient for patient in result.split_cohorts.dev if patient.patient_id in dev_patient_ids
    ]

    # 1. Cross-fit the primary propensity model on the retained dev split.
    propensity_model = LogisticPropensityModel(seed=derive_seed(master_seed, "propensity"))
    cross_fitted = propensity_model.fit_cross_fitted(dev_patients, n_folds=PROPENSITY_N_FOLDS)

    # 1b. Common-support trimming (`decisions.md` Section 26 amendment of
    # 2026-10-12, item B): patients outside the primary band leave the
    # estimation population before anything is estimated or checked.
    # Everything below runs on `analysis_patients`; the count trimmed per
    # arm is reported in the summary and manifest.
    support_trim = trim_to_common_support(dev_patients, cross_fitted.propensity_scores)
    analysis_patients = list(support_trim.retained)

    # 2. Run all six frozen causal baselines.
    causal_estimates = run_all_baselines(analysis_patients, cross_fitted.propensity_scores)

    # 3. Overlap/support diagnostics.
    overlap_report = OverlapDiagnostics.compute(
        analysis_patients, cross_fitted.propensity_scores, spec.adjustment_set
    )

    # 4. Negative controls.
    negative_control_seed = derive_seed(master_seed, "negative_controls")
    exposure_result = run_negative_control_exposure(
        analysis_patients,
        cross_fitted.propensity_scores,
        seed=negative_control_seed,
        n_bootstrap=negative_control_n_bootstrap,
    )
    outcome_result = run_negative_control_outcome(
        analysis_patients,
        spec.adjustment_set,
        seed=negative_control_seed,
        n_bootstrap=negative_control_n_bootstrap,
    )

    # 5. Witnesses.
    temporal_result = _run_temporal_witness(analysis_patients, spec)

    # Task 15.1: execute the eight frozen sensitivity variants and feed
    # the real `overall_stability` boolean to the causal witness.
    # `undetermined` (any variant unexecuted, e.g.
    # `boosted_propensity_model` without the `boosted` extra) maps to
    # `False` -- it can never read as stability (Requirement 8.9).
    #
    # The stronger-selection variant needs its own cohort: the identical
    # pipeline (same seeds, same split) observationalized under the
    # Section 26 stronger mechanism, with propensities refit on it.
    stronger_selection_cohort = build_stronger_selection_cohort(
        timelines,
        treatment_arms,
        spec,
        pipeline_seed,
        propensity_seed=derive_seed(master_seed, "propensity"),
        n_folds=PROPENSITY_N_FOLDS,
        precomputed_endpoint_results=endpoint_results,
    )
    sensitivity_result: SensitivityStabilityResult = SensitivityRobustness.run_executed(
        resolved_config,
        analysis_patients,
        cross_fitted.propensity_scores,
        master_seed=master_seed,
        adjustment_set=spec.adjustment_set,
        stronger_selection_cohort=stronger_selection_cohort,
        support_population=SupportPopulation(
            patients=tuple(dev_patients), propensity_scores=cross_fitted.propensity_scores
        ),
        propensity_n_folds=PROPENSITY_N_FOLDS,
    )
    sensitivity_stable = sensitivity_result.overall_stability == "stable"

    causal_result = CausalWitness.evaluate(
        overlap_report=overlap_report,
        sensitivity_stable=sensitivity_stable,
        negative_control_exposure_result=exposure_result,
        negative_control_outcome_result=outcome_result,
    )
    clinical_result = ClinicalWitness.evaluate(
        _build_protocol_records(analysis_patients), spec
    )
    witness_results = [temporal_result, causal_result, clinical_result]
    aggregate_decision: AggregateDecision = aggregate(witness_results, WitnessPolicy.ALL_REQUIRED)

    # Task 16: the real six-way `ReleaseDecision`, not the three-witness
    # `aggregate_decision` alone -- `uncertainty_valid` has no computed
    # value at this script's scope (task 16.1's ensemble/support-gate
    # machinery is a neural-model concern this script does not touch), so
    # it is conservatively supplied as `False`: a release gate this
    # script cannot actually evaluate must never read as passing.
    release_decision = evaluate_release(overlap_report, witness_results, uncertainty_valid=False)

    # 6. Register the experiment and persist real artifacts.
    #
    # `ExperimentRegistry` owns both halves of run identity: it computes
    # `compute_experiment_id(resolved_config)` -- the single identity
    # function -- and it is the single place that refuses a duplicate,
    # raising `DuplicateExperimentError` when the computed id already has a
    # non-empty results directory. This script has no identity scheme and
    # no duplicate check of its own (`design.md` Section 2.6 tension 6).
    # Registration happens before any artifact is written, so a refused
    # duplicate leaves the prior run's directory untouched.
    registry = ExperimentRegistry(
        registry_path=results_root / "registry.jsonl", results_root=results_root
    )
    provenance = resolve_git_provenance(REPO_ROOT)
    experiment_id = registry.register(
        resolved_config,
        git_commit=provenance.commit,
        dirty=provenance.dirty,
        results_root=results_root,
    )

    experiment_dir = results_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    cohort_flow = build_cohort_flow_table(timelines, treatment_arms, spec)
    cohort_dir = experiment_dir / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    (cohort_dir / "flow_table.json").write_text(
        json.dumps(cohort_flow, indent=2, sort_keys=True, default=str)
    )

    reference_path = experiment_dir / "reference" / "internal_randomized_reference.sealed.json"
    result.sealed_reference.write_sealed_artifact(reference_path)

    # Task 15.1: persist the machine-readable sensitivity-stability result
    # (`design.md` Section 4.3) and append each executed variant name to
    # the readiness ledger (Requirement 18.3). A variant's own
    # `resolved_config_hash` (from `apply_variant`, computed per-variant
    # inside `_execute_variant`) is what each ledger entry's
    # `artifact_hash` attests to -- `None` for an unexecuted variant,
    # which is recorded as the literal string `"unexecuted"` rather than
    # skipped, so a reader of the ledger can see that the variant ran
    # (attempted) even though it did not produce a usable hash.
    robustness_dir = experiment_dir / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    (robustness_dir / "sensitivity_stability.json").write_text(
        json.dumps(sensitivity_result.to_dict(), indent=2, sort_keys=True, default=str)
    )
    for outcome in sensitivity_result.variant_outcomes:
        append_ledger_entry(
            experiment_dir,
            SENSITIVITY_VARIANT_KIND,
            outcome.name,
            outcome.resolved_config_hash or "unexecuted",
        )

    manifest = ManifestWriter(
        experiment_id=experiment_id,
        stage="causal_analysis",
        allowed_path_prefixes=(str(results_root),),
    )
    manifest.add_input_hash("actg175_raw_tables", tables.checksum_sha256)
    manifest.set_cohort_definition(
        {
            "dev_retained_count": len(dev_patients),
            "arm_0_count": sum(1 for p in dev_patients if p.treatment_arm == 0),
            "arm_1_count": sum(1 for p in dev_patients if p.treatment_arm == 1),
            "common_support_trim": support_trim.to_dict(),
        }
    )
    manifest.set_feature_list(list(spec.adjustment_set))
    manifest.set_treatment_definition(
        {
            "intervention": spec.intervention.concept_id,
            "comparator": spec.comparator.concept_id,
        }
    )
    manifest.set_outcome_definition({"type": spec.outcome.type})
    manifest.set_model_config({"stage": "causal_analysis", "master_seed": master_seed})
    # Bind this run's result directory to the commit the run executed from
    # (Requirement 15.4). The commit itself is resolved inside `build()`.
    manifest.add_result_directory(str(experiment_dir))
    manifest.set_metrics(
        {
            "causal_baselines": {
                estimate.name: {
                    "estimate": estimate.estimate,
                    "se": estimate.se,
                    "scale": estimate.scale,
                }
                for estimate in causal_estimates
            },
            "overlap_support_valid": overlap_report.support_valid,
            "negative_control_exposure": exposure_result.classification,
            "negative_control_outcome": outcome_result.classification,
            "witness_outcomes": {r.witness: r.status for r in witness_results},
            "aggregate_decision_passed": aggregate_decision.passed,
            "sensitivity_overall_stability": sensitivity_result.overall_stability,
            "sensitivity_unexecuted_variants": list(sensitivity_result.unexecuted_variants),
        }
    )
    manifest_path = experiment_dir / "manifest.json"
    try:
        manifest.write(manifest_path)
    except ManifestPrivacyError as exc:
        raise RuntimeError(f"manifest privacy scan blocked the write: {exc}") from exc

    # Task 17.2: build and write a public/protected audit-artifact pair
    # for every candidate estimate this run produced -- all six causal
    # baselines, released or suppressed alike (Requirement 12.1, 12.2).
    # None of them has a neural checkpoint, so `checkpoint_hashes={}`
    # (explicitly empty, not omitted -- see `build_audit_report`'s
    # docstring).
    split_hash = _split_hash([patient.patient_id for patient in dev_patients])
    # The estimation population is the trimmed one; trimmed patients are
    # counted as abstained (no patient-level gate otherwise runs here).
    supported_count = len(analysis_patients)
    abstained_count = len(support_trim.trimmed_patient_ids)
    for estimate in causal_estimates:
        audit_report = build_audit_report(
            protocol=spec,
            population={"description": "ACTG175 dev-split retained cohort", "n": supported_count},
            treatment={
                "intervention": spec.intervention.concept_id,
                "comparator": spec.comparator.concept_id,
            },
            estimate=_audit_bootstrap_ci(estimate),
            overlap=overlap_report,
            witness_results=witness_results,
            abstentions=[],
            violations=[],
            provenance={
                "experiment_id": experiment_id,
                "seed": master_seed,
                "commit": provenance.commit,
            },
            release_decision=release_decision,
            estimate_status="released" if release_decision.released else "suppressed",
            dataset_checksum=tables.checksum_sha256,
            split_hash=split_hash,
            resolved_config_hash=canonical_json(resolved_config),
            checkpoint_hashes={},
            seed=master_seed,
            sensitivity_variant=None,
            supported_count=supported_count,
            abstained_count=abstained_count,
        )
        write_audit_artifacts(audit_report, experiment_dir, claim_id=estimate.name)

    # Task 17.2: the sensitivity runner's own candidate estimates --
    # `SensitivityRobustness.run_executed`'s eight variants, including
    # the one that failed to execute (`boosted_propensity_model` without
    # xgboost), get an audit report too. An unexecuted variant's estimate
    # is `None`, and `estimate_status` is forced to `"suppressed"`
    # regardless of `release_decision` -- there is no numeric estimate to
    # release in the first place.
    for outcome in sensitivity_result.variant_outcomes:
        sensitivity_estimate = (
            BootstrapCI(
                point_estimate=outcome.estimate,
                ci_low=outcome.ci_low if outcome.ci_low is not None else float("nan"),
                ci_high=outcome.ci_high if outcome.ci_high is not None else float("nan"),
                n_replicates=0,
                n_successful=0,
            )
            if outcome.status == "executed"
            else None
        )
        sensitivity_audit_report = build_audit_report(
            protocol=spec,
            population={"description": "ACTG175 dev-split retained cohort", "n": supported_count},
            treatment={
                "intervention": spec.intervention.concept_id,
                "comparator": spec.comparator.concept_id,
            },
            estimate=sensitivity_estimate,
            overlap=overlap_report,
            witness_results=witness_results,
            abstentions=[],
            violations=[],
            provenance={
                "experiment_id": experiment_id,
                "seed": master_seed,
                "commit": provenance.commit,
            },
            release_decision=release_decision,
            estimate_status=(
                "suppressed"
                if outcome.status == "unexecuted"
                else ("released" if release_decision.released else "suppressed")
            ),
            dataset_checksum=tables.checksum_sha256,
            split_hash=split_hash,
            resolved_config_hash=outcome.resolved_config_hash or canonical_json(resolved_config),
            checkpoint_hashes={},
            seed=master_seed,
            sensitivity_variant=outcome.name,
            supported_count=supported_count,
            abstained_count=abstained_count,
        )
        write_audit_artifacts(
            sensitivity_audit_report, experiment_dir, claim_id=f"sensitivity_{outcome.name}"
        )

    temporal_violation_count = (
        temporal_result.evidence["baseline_precedes_treatment_and_no_future_leakage"][
            "violation_count"
        ]
        + temporal_result.evidence["outcomes_after_time_zero"]["violation_count"]
        + temporal_result.evidence["chronological_ordering_preserved"]["violation_count"]
    )
    protocol_violation_count = clinical_result.evidence["violation_count"]
    witness_detected_failure_count = sum(
        1 for r in witness_results if r.status in ("fail", "uncertain")
    )

    return {
        "experiment_id": experiment_id,
        "dev_retained_count": len(dev_patients),
        "common_support_trim": support_trim.to_dict(),
        "analysis_population_count": len(analysis_patients),
        "causal_estimates": {e.name: e.estimate for e in causal_estimates},
        "overlap_support_valid": overlap_report.support_valid,
        "negative_control_exposure": exposure_result.classification,
        "negative_control_outcome": outcome_result.classification,
        "witness_outcomes": {r.witness: r.status for r in witness_results},
        "aggregate_decision_passed": aggregate_decision.passed,
        "manifest_path": str(manifest_path),
        "reference_path": str(reference_path),
        "temporal_violation_count": temporal_violation_count,
        "protocol_violation_count": protocol_violation_count,
        "witness_detected_failure_count": witness_detected_failure_count,
        "sensitivity_overall_stability": sensitivity_result.overall_stability,
        "sensitivity_unexecuted_variants": list(sensitivity_result.unexecuted_variants),
        # Task 22.2 (Requirement 19.1, 19.3): the real six-way release
        # decision's failing-condition names (empty when it passed) and
        # every sensitivity variant's full per-variant result (name,
        # status, estimate, direction, etc.), including variants whose
        # direction contradicts the primary estimate's -- `release_gate.py`
        # and `evaluation.robustness` already compute both; this just
        # surfaces them on the summary rather than only the aggregated
        # booleans/names this dict already carried.
        "release_decision_released": release_decision.released,
        "release_decision_reasons": list(release_decision.reasons),
        "supported_count": supported_count,
        "abstained_count": abstained_count,
        "sensitivity_variant_outcomes": [
            outcome.to_dict() for outcome in sensitivity_result.variant_outcomes
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full causal-analysis pipeline (propensity, six causal "
            "baselines, overlap diagnostics, negative controls, three "
            "witnesses) on the real ACTG175 development split, and persist "
            "real experiment artifacts under --results-root."
        )
    )
    parser.add_argument("--real-data-path", type=Path, default=DEFAULT_REAL_DATA_PATH)
    parser.add_argument("--trial-config", type=Path, default=DEFAULT_TRIAL_CONFIG)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help=(
            "Path to the frozen dataset-contract YAML composed into this run's "
            "ResolvedConfig, which is what the experiment_id is hashed from "
            f"(default: {DEFAULT_DATASET_CONFIG.name})."
        ),
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help=(
            "Path to the experiment YAML composed into this run's ResolvedConfig; "
            "its seed and negative_control_n_bootstrap are overridden by "
            "--master-seed and --negative-control-n-bootstrap "
            f"(default: {DEFAULT_EXPERIMENT_CONFIG.name})."
        ),
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--master-seed", type=int, default=20240101)
    parser.add_argument(
        "--negative-control-n-bootstrap",
        type=int,
        default=NEGATIVE_CONTROL_N_BOOTSTRAP,
        help=(
            "Bootstrap replicate count for both negative controls "
            f"(default: {NEGATIVE_CONTROL_N_BOOTSTRAP})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_causal_analysis(
            args.real_data_path,
            args.trial_config,
            args.results_root,
            args.master_seed,
            negative_control_n_bootstrap=args.negative_control_n_bootstrap,
            dataset_config_path=args.dataset_config,
            experiment_config_path=args.experiment_config,
        )
    except Actg175DataNotFoundError as exc:
        print(f"run_causal_analysis.py failed: real ACTG175 data not found: {exc}", file=sys.stderr)
        return 1
    except DuplicateExperimentError as exc:
        print(f"run_causal_analysis.py failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"run_causal_analysis.py failed: {exc}", file=sys.stderr)
        return 1

    print(f"dev retained: {summary['dev_retained_count']} patients")
    trim = summary["common_support_trim"]
    print(
        f"common-support trim {trim['band']}: removed {len(trim['trimmed_patient_ids'])} "
        f"(by arm {trim['trimmed_count_by_arm']}); "
        f"estimation population {summary['analysis_population_count']}"
    )
    print(f"causal estimates: {summary['causal_estimates']}")
    print(f"overlap support_valid: {summary['overlap_support_valid']}")
    print(f"negative_control_exposure: {summary['negative_control_exposure']}")
    print(f"negative_control_outcome: {summary['negative_control_outcome']}")
    print(f"witness_outcomes: {summary['witness_outcomes']}")
    print(f"aggregate_decision_passed: {summary['aggregate_decision_passed']}")
    print(f"sensitivity_overall_stability: {summary['sensitivity_overall_stability']}")
    print(f"sensitivity_unexecuted_variants: {summary['sensitivity_unexecuted_variants']}")
    print(f"manifest written to: {summary['manifest_path']}")
    print(f"sealed reference written to: {summary['reference_path']}")
    print("run_causal_analysis.py succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
