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
6. Register the experiment (`infra.registry.ExperimentRegistry`), write a
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
- It does not run the ablation matrix (`scripts/run_ablations.py`) or the
  robustness/sensitivity variants (`src/evaluation/robustness.py`) --
  those remain separately scoped, deferred follow-ups.
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
from causal.overlap import OverlapDiagnostics
from causal.propensity import LogisticPropensityModel
from cohorts.cohort_construction import PatientRecord
from cohorts.cohort_flow import build_cohort_flow_table
from cohorts.pipeline import run_observationalization_pipeline
from cohorts.sealed_reference import unadjusted_cox_reference_estimator
from data.loaders.actg175 import Actg175DataNotFoundError, load_actg175
from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
from infra.manifest import ManifestPrivacyError, ManifestWriter
from infra.registry import DuplicateExperimentError, ExperimentRegistry
from infra.seeding import derive_seed
from protocols.consistency_checker import PatientProtocolRecord
from protocols.eligibility_dsl import PatientTimeline as EligibilityPatientTimeline
from protocols.schema import TargetTrialSpec
from witnesses.aggregation import AggregateDecision, WitnessPolicy, aggregate
from witnesses.causal import CausalWitness
from witnesses.clinical import ClinicalWitness
from witnesses.temporal import TemporalWitness, WitnessResult

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"
DEFAULT_TRIAL_CONFIG = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
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

#: Length, in hex characters, of the truncated content-hash experiment ID
#: (matches `infra.registry`'s `_EXPERIMENT_ID_LENGTH` convention).
_EXPERIMENT_ID_LENGTH = 16


def _compute_content_experiment_id(
    dataset_checksum: str,
    trial_spec_version: str,
    master_seed: int,
    negative_control_n_bootstrap: int,
) -> str:
    """A real, collision-resistant experiment ID derived from every input
    that actually changes this run's output -- not just `master_seed`.

    Unlike this script's earlier `f"causal_analysis_{master_seed}"` (which
    collided across any two runs sharing a seed even if the underlying
    data, trial spec, or bootstrap configuration differed -- silently
    overwriting the first run's artifacts and appending a second,
    indistinguishable registry record), this hashes every input that
    changes the run's actual output: the real raw-data checksum, the
    locked trial-spec version, the master seed, and the negative-control
    bootstrap count. Two runs that hash identically are, by construction,
    runs that would have produced identical results anyway -- this is not
    `infra.registry.compute_experiment_id`'s full
    `sha256(canonical_json(resolved_config))` over a real `ResolvedConfig`
    (no such config is composed at this call site; see
    `scripts/run_ablations.py`'s `_compute_harness_experiment_id` for the
    same documented simplification), but it closes the specific collision
    risk a seed-only identifier has: two runs against different data (a
    changed/re-ingested dataset) or a different trial-spec version, that
    happen to share a `master_seed`, now get different, non-colliding IDs.

    Returns:
        `sha256(...)`'s first `_EXPERIMENT_ID_LENGTH` hex characters.
    """
    canonical = (
        f"dataset_checksum={dataset_checksum}|"
        f"trial_spec_version={trial_spec_version}|"
        f"master_seed={master_seed}|"
        f"negative_control_n_bootstrap={negative_control_n_bootstrap}"
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"causal_analysis_{digest[:_EXPERIMENT_ID_LENGTH]}"


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


def _run_temporal_witness(patients: list[PatientRecord], spec: TargetTrialSpec) -> WitnessResult:
    """Evaluate the temporal witness on the retained cohort.

    Held-out longitudinal-error inputs (`held_out_cd4_cd8_mae`,
    `factual_baseline_cd4_cd8_mae`) are not available -- this script does
    not train the sequence-GRU baseline (see module docstring) -- so that
    sub-condition is left `None`, and `TemporalWitness.evaluate` correctly
    classifies the overall result as `"uncertain"` rather than guessing a
    value for a check this script cannot actually perform.
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
        held_out_cd4_cd8_mae=None,
        factual_baseline_cd4_cd8_mae=None,
    )


def run_causal_analysis(
    real_data_path: Path,
    trial_config_path: Path,
    results_root: Path,
    master_seed: int,
    negative_control_n_bootstrap: int = NEGATIVE_CONTROL_N_BOOTSTRAP,
) -> dict[str, Any]:
    """Run the full causal-analysis sequence and persist real artifacts.

    Args:
        negative_control_n_bootstrap: bootstrap replicate count for both
            negative controls. Defaults to `NEGATIVE_CONTROL_N_BOOTSTRAP`
            (200); tests may pass a smaller value to keep runtime down,
            since each negative control refits a model once per replicate.

    Returns a summary dict (also what gets printed/logged); raises on any
    failure rather than swallowing it, since this is a library function a
    test can call directly (mirroring `scripts/run_ablations.py`'s
    `run_all_ablations` convention) -- `main()` below is the only place
    that catches exceptions and converts them to a CLI exit code.
    """
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

    # 2. Run all six frozen causal baselines.
    causal_estimates = run_all_baselines(dev_patients, cross_fitted.propensity_scores)

    # 3. Overlap/support diagnostics.
    overlap_report = OverlapDiagnostics.compute(
        dev_patients, cross_fitted.propensity_scores, spec.adjustment_set
    )

    # 4. Negative controls.
    negative_control_seed = derive_seed(master_seed, "negative_controls")
    exposure_result = run_negative_control_exposure(
        dev_patients,
        cross_fitted.propensity_scores,
        seed=negative_control_seed,
        n_bootstrap=negative_control_n_bootstrap,
    )
    outcome_result = run_negative_control_outcome(
        dev_patients,
        spec.adjustment_set,
        seed=negative_control_seed,
        n_bootstrap=negative_control_n_bootstrap,
    )

    # 5. Witnesses.
    temporal_result = _run_temporal_witness(dev_patients, spec)
    causal_result = CausalWitness.evaluate(
        overlap_report=overlap_report,
        # No sensitivity-variant machinery exists yet (task 13.1 is a
        # separately-scoped follow-up; see `witnesses.causal`'s module
        # docstring) -- conservatively treated as NOT stable so this
        # script never silently reports a pass this codebase cannot back
        # up. A future caller wiring in real sensitivity variants should
        # pass the real computed boolean instead of this constant.
        sensitivity_stable=False,
        negative_control_exposure_result=exposure_result,
        negative_control_outcome_result=outcome_result,
    )
    clinical_result = ClinicalWitness.evaluate(
        _build_protocol_records(dev_patients), spec
    )
    witness_results = [temporal_result, causal_result, clinical_result]
    aggregate_decision: AggregateDecision = aggregate(witness_results, WitnessPolicy.ALL_REQUIRED)

    # 6. Persist real experiment artifacts.
    experiment_id = _compute_content_experiment_id(
        dataset_checksum=tables.checksum_sha256,
        trial_spec_version=spec.version,
        master_seed=master_seed,
        negative_control_n_bootstrap=negative_control_n_bootstrap,
    )
    experiment_dir = results_root / experiment_id
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise DuplicateExperimentError(
            f"experiment_id {experiment_id!r} already has a non-empty results "
            f"directory at {experiment_dir}; refusing to overwrite a prior "
            "run's artifacts. Use a different master_seed, or remove that "
            "directory first if this rerun is intentional."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    cohort_flow = build_cohort_flow_table(timelines, treatment_arms, spec)
    cohort_dir = experiment_dir / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    (cohort_dir / "flow_table.json").write_text(
        json.dumps(cohort_flow, indent=2, sort_keys=True, default=str)
    )

    reference_path = experiment_dir / "reference" / "internal_randomized_reference.sealed.json"
    result.sealed_reference.write_sealed_artifact(reference_path)

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
        }
    )
    manifest_path = experiment_dir / "manifest.json"
    try:
        manifest.write(manifest_path)
    except ManifestPrivacyError as exc:
        raise RuntimeError(f"manifest privacy scan blocked the write: {exc}") from exc

    registry = ExperimentRegistry(
        registry_path=results_root / "registry.jsonl", results_root=results_root
    )
    # This script computes its own simplified experiment_id (see above) --
    # not `infra.registry.compute_experiment_id`'s full resolved-config
    # hash, since no `ResolvedConfig` exists at this call site (mirrors
    # `scripts/run_ablations.py`'s documented simplification). Register
    # directly via a manual append rather than `ExperimentRegistry.register()`
    # (which requires a `ResolvedConfig`), to avoid fabricating one.
    registry.registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_entry = {
        "experiment_id": experiment_id,
        "dataset_version": "3.2",
        "master_seed": master_seed,
        "stage": "causal_analysis",
        "witness_policy": "ALL_REQUIRED",
        "aggregate_decision_passed": aggregate_decision.passed,
    }
    with registry.registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(registry_entry, sort_keys=True))
        handle.write("\n")

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
    print(f"causal estimates: {summary['causal_estimates']}")
    print(f"overlap support_valid: {summary['overlap_support_valid']}")
    print(f"negative_control_exposure: {summary['negative_control_exposure']}")
    print(f"negative_control_outcome: {summary['negative_control_outcome']}")
    print(f"witness_outcomes: {summary['witness_outcomes']}")
    print(f"aggregate_decision_passed: {summary['aggregate_decision_passed']}")
    print(f"manifest written to: {summary['manifest_path']}")
    print(f"sealed reference written to: {summary['reference_path']}")
    print("run_causal_analysis.py succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
