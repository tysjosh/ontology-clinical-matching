#!/usr/bin/env python
"""Run the six model-affecting ("retrain") frozen primary ablation arms
through `models.variants.build_model` and the frozen training harness
(task 14; Requirement 7.2, 7.3, 7.4, 7.5, 7.9, 7.10, 7.14, 7.16).

Replaces the previous harness, which fit one placeholder `CoxBaseline` for
all ten `AblationConfig` entries regardless of what each one was supposed
to express architecturally. That placeholder is gone: every retrained arm
below is built through `models.variants.build_model(variant, dimensions,
seed)`, the same single construction entry point task 12.1 introduced, so
each arm's model is the real thing `decisions.md` Section 17 and
`design.md` Section 3.6 describe -- not a stand-in.

Scope: six of twelve expanded runs
-------------------------------------------------------------------------

`evaluation.ablation_matrix.expand_ablations()` (task 13) produces twelve
runs from the ten frozen entries. Six have `execution_mode="retrain"` --
these are the ones this script executes, since they are the arms that
differ from the complete model in architecture or input content and
therefore require training from initialization. The other six
(`no_overlap_gate`, `no_uncertainty_gate`, `no_clinical_constraints`, and
the three `witness_removed` sub-runs) have `execution_mode=
"reuse_predictions"` -- they reuse the complete model's saved predictions
and bypass only a gate/constraint/witness, which is task 14.1's job, not
this script's. `run_retrained_ablations` below filters
`expand_ablations()` down to the `"retrain"` subset explicitly, rather
than silently ignoring the other six.

What "retrain" means here (Requirement 7.10)
-------------------------------------------------------------------------

For each of the six retrain runs, this script:

1. selects `(lambda_state, lambda_balance)` once per run, via
   `models.torch_training.select_lambdas` on the first seed in the
   configured seed sequence -- the same "select once, train every seeded
   replica at the selected pair" convention
   `models.torch_training.train_ensemble` already uses for the ensemble's
   five replicas, applied here across the six arms instead;
2. trains that arm from a fresh `build_model(run.variant, dimensions,
   seed)` construction, under the frozen training configuration
   (`models.torch_training.FROZEN_TRAINING_CONFIG`), once for each
   configured seed -- by default the five frozen seeds 101, 202, 303,
   404, 505 (`infra.readiness.REQUIRED_SEEDS`);
3. writes that seed's checkpoint and a per-run-per-seed result record
   under `results/<experiment_id>/ablations/`.

No complete-model checkpoint is ever loaded or reused for a retrain run
(Requirement 7.10's "SHALL reuse no complete-model checkpoint for that
arm") -- every retrain run constructs its model from scratch through
`build_model`.

How each arm's variant already encodes its required contrast
-------------------------------------------------------------------------

This script does not special-case any arm's input construction -- every
contrast Requirement 7.2-7.5 and 7.9 names is already encoded in the
`ExperimentVariant` `models.variants.variant_for_ablation` resolves for
that run, and realized by `build_model` / `TorchInterventionAwareModel` /
`TorchSequenceGRU` (task 12.1, 9.1-9.3 of the parent spec):

- `no_treatment_input`: `treatment_representation="absent"` ->
  `InterventionMode.ZEROED` zeros the entire intervention channel
  (`[treatment_arm, dose_or_default, adherence_or_default]`) at every
  landmark for every patient, while every other channel (state,
  protocol) is read from the batch unchanged (Requirement 7.2). ACTG175
  never populates dose/adherence (they are always 0.0 already), so
  zeroing the whole channel is exactly zeroing the treatment element.
- `intervention_channel_without_protocol`: `protocol_channel="zeroed"` ->
  `ProtocolMode.ZEROED` supplies an all-zero width-2 protocol input at
  every landmark while the intervention channel is read unchanged
  (Requirement 7.4).
- `placebo_intervention_channel`: `treatment_representation=
  "placebo_channel"` -> `build_model` draws the placebo vector from
  `derive_seed(seed, "placebo_intervention_channel")` internally and
  passes it to `TorchInterventionAwareModel` as a non-trainable buffer
  (Requirement 7.5). This script reads `model.placebo_vector` back off
  the trained model and records it in that seed's result, so a rerun at
  the same seed is checkable against the recorded vector.
- `treatment_as_ordinary_feature`: `model_family="sequence_gru"` ->
  `TorchSequenceGRU`, which has no intervention channel at all and reads
  treatment as an ordinary input column, at the frozen hidden width 16
  shared with every `intervention_aware`-family arm (Requirement 7.3,
  7.14).
- `predictive_accuracy_only`: `balance_penalty_enabled=False`,
  `causal_adjustment=False` -> this script passes `lambda_balance=0.0`
  for this one run instead of running lambda selection for it, which
  realizes "disable the causal adjustment" (the MMD balance term is the
  only causal-adjustment-shaped term in the frozen objective) while
  leaving the factual survival/state terms' weighting exactly as
  selected for every other arm (Requirement 7.9).

Failure handling (Requirement 7.16)
-------------------------------------------------------------------------

A `(run, seed)` training attempt that raises is recorded with
`succeeded=False` and a `failure_reason` string naming the exception,
rather than aborting the whole harness. `table_incomplete` is set `True`
if any attempt failed; task 14.2's `ablation_table.json` reads this flag
and excludes failed runs, marking the table incomplete rather than
invalid -- "incomplete" (recoverable by rerunning one arm) is a distinct
state from "invalid" (task 14.3's collision detector's output).

Fitting/evaluation population
-------------------------------------------------------------------------

Development and validation `TensorBatch`es are built from each split's
*retained* (post-observationalization) patients --
`pipeline_result.dev.retained_patient_ids` /
`pipeline_result.val.retained_patient_ids` filtered against
`pipeline_result.split_cohorts.{dev,val}` -- matching
`scripts/run_causal_analysis.py`'s established convention for which
population a model is fit against. Standardization is fit on the
retained development population only
(`models.dataset.PatientDatasetBuilder.fit`).

Primary factual metric (task 14.2; Requirement 7.11, 7.12)
-------------------------------------------------------------------------

`decisions.md` Section 18's factual-prediction group lists several
metrics; the one `AblationRecord`/`AblationTable` below treats as *the*
"frozen primary factual metric" for ranking ablation arms is
time-dependent concordance (`evaluation.metrics.concordance_index`) --
the same metric this harness's task-12.1 predecessor used, and the one
every causal/predictive baseline in this codebase (`CoxBaseline`,
`select_lambdas`'s own lambda-selection criterion) already treats as the
model-comparison metric of record.

Computed on the retained Validation_Split population only
(`build_training_batches`'s `val_batch`), via each seed's trained model's
factual-arm 96-week risk (`models.torch_training.factual_outputs` +
`models.torch_intervention_aware.outcome_from_hazards`) -- never on any
Development_Split patient, satisfying Requirement 7.11's "exclude every
Development_Split patient used for fitting." One value is reported per
seed, plus the five-seed (or however many `seeds` were configured) mean.

What this script still does NOT do
-------------------------------------------------------------------------

This script does not execute the six reuse-prediction arms (task 14.1,
`evaluation.ablation_reuse`) and does not run variant-collision detection
(task 14.3, `evaluation.ablation_matrix.detect_variant_collisions`) --
both are separate modules a future top-level ablation runner composes
with this one's output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cohorts.cohort_construction import PatientRecord
from cohorts.pipeline import ObservationalizationPipelineResult, run_observationalization_pipeline
from data.loaders.actg175 import Actg175DataNotFoundError, load_actg175
from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
from data.timelines import PatientTimeline as DataPatientTimeline
from evaluation.ablation_matrix import (
    AblationRun,
    VariantCollisionError,
    detect_variant_collisions,
    expand_ablations,
)
from evaluation.metrics import concordance_index
from infra.readiness import REQUIRED_SEEDS
from models.dataset import DatasetDimensions, PatientDatasetBuilder, TensorBatch

# Imported at module scope -- despite the rest of this module's torch
# imports being deferred into function bodies to match `models.variants`'
# convention -- so that `models.torch_training.train_one_model` is
# reachable (and monkeypatchable) as `run_ablations.train_one_model`,
# the same "patch the harness module's own name" pattern
# `tests/integration/test_run_ablations.py` uses to inject a training
# failure for the Requirement 7.16 test. This script is a `scripts/`
# entry point, not a library module other code imports without torch
# (unlike `models.variants`), so there is no torch-isolation guarantee to
# preserve here -- `scripts.run_ablations` is not named in
# `tests/unit/test_torch_isolation.py`'s `TORCH_FREE_MODULES`.
from models.torch_training import (  # noqa: E402
    FROZEN_TRAINING_CONFIG,
    TrainingBatches,
    TrainingConfig,
    checkpoint_sha256,
    factual_outputs,
    select_lambdas,
    train_one_model,
)
from models.variants import ExperimentVariant, build_model
from protocols.endpoints import EndpointResult
from protocols.schema import TargetTrialSpec
from uncertainty.bootstrap import BootstrapCI
from witnesses.audit_io import write_audit_artifacts
from witnesses.audit_report import build_audit_report

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default results root (`results/<experiment_id>/ablations/<run_id>/...`).
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

#: Fixed label folded into the harness's simplified experiment identifier
#: (see `scripts/run_stage.py`/`infra.registry` for the real
#: `compute_experiment_id`, not used here -- see the historical note in
#: `_compute_harness_experiment_id`). "_v2" distinguishes this task-14
#: harness's identifiers from the task-12.1 harness's, since the two are
#: not interchangeable: this harness's runs are a strict subset (the six
#: retrain arms only) and its result-file contents differ entirely.
_HARNESS_LABEL = "ablation_harness_v2"

#: The real, packaged raw ACTG175 v3.2 data directory (`docs/data_access.md`
#: Section 2's approved storage location: `data/raw/`, gitignored). Used
#: only when `--real-data-path` is supplied to `main()`.
DEFAULT_REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"


def _not_evaluated_overlap_report() -> Any:
    """A placeholder `OverlapReport` for `build_audit_report`'s `overlap`
    argument, documenting that this harness computes no overlap
    diagnostics of its own (task 14's scope is model construction and
    training, not causal overlap/balance). Every boolean is `False` so
    `evaluate_release`'s six-way conjunction can never read this as a
    pass by omission.
    """
    from causal.overlap import CommonSupportSummary, OverlapReport

    return OverlapReport(
        arm_counts={},
        arm_propensity_summary={},
        common_support=CommonSupportSummary(
            primary_count=0, primary_fraction=0.0, sensitivity_count=0, sensitivity_fraction=0.0
        ),
        extreme_weight_count=0,
        extreme_weight_fraction=0.0,
        ess_by_arm={},
        smd_by_covariate={},
        arm_counts_sufficient=False,
        ess_sufficient=False,
        smd_sufficient=False,
        no_unresolved_positivity_warning=False,
        support_valid=False,
    )


def _not_evaluated_witness_results() -> list[Any]:
    """Placeholder `WitnessResult`s for `build_audit_report`'s
    `witness_results` argument -- this harness evaluates no witness
    itself (that is `scripts/run_causal_analysis.py`'s job). `"uncertain"`
    status, never `"pass"`, so `evaluate_release` cannot read any of the
    three witness conditions as satisfied.
    """
    from witnesses.temporal import WitnessResult

    reason = {"note": "not evaluated by scripts/run_ablations.py; see this harness's audit note"}
    return [
        WitnessResult(witness="temporal", status="uncertain", evidence=reason),
        WitnessResult(witness="causal", status="uncertain", evidence=reason),
        WitnessResult(witness="clinical", status="uncertain", evidence=reason),
    ]


def _write_ablation_audit_artifacts(
    run: AblationRun,
    seed: int,
    seed_record: dict[str, Any],
    experiment_dir: Path,
    spec: TargetTrialSpec,
    dataset_checksum: str,
    split_hash: str,
) -> None:
    """Build and write one `(run, seed)` ablation arm's public/protected
    audit-artifact pair (task 17.2; Requirement 12.1, 12.2).

    This harness computes no overlap diagnostics, no witnesses, and no
    `ReleaseDecision` of its own -- see `_not_evaluated_overlap_report`/
    `_not_evaluated_witness_results`. `evaluate_release` over those
    placeholders is therefore always `released=False`, so every ablation
    arm's audit report is `estimate_status="suppressed"` unconditionally;
    this is a statement about this harness's scope ("no release gate is
    computed here"), not a claim that any arm actually failed release.
    A failed `(run, seed)` training attempt (Requirement 7.16) has no
    numeric estimate at all -- `estimate=None`.
    """
    from evaluation.release_gate import evaluate_release

    overlap_report = _not_evaluated_overlap_report()
    witness_results = _not_evaluated_witness_results()
    release_decision = evaluate_release(overlap_report, witness_results, uncertainty_valid=False)

    concordance = seed_record.get("concordance_index")
    checkpoint_hash = seed_record.get("checkpoint_sha256")
    estimate = (
        BootstrapCI(
            point_estimate=concordance,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_replicates=0,
            n_successful=0,
        )
        if seed_record.get("succeeded") and concordance is not None
        else None
    )

    audit_report = build_audit_report(
        protocol=spec,
        population={"description": f"ablation arm {run.run_id}", "n": 0},
        treatment={"ablation_run_id": run.run_id, "source_entry": run.source_entry},
        estimate=estimate,
        overlap=overlap_report,
        witness_results=witness_results,
        abstentions=[],
        violations=[],
        provenance={"run_id": run.run_id, "seed": seed},
        release_decision=release_decision,
        estimate_status="suppressed",
        dataset_checksum=dataset_checksum,
        split_hash=split_hash,
        resolved_config_hash=f"{run.run_id}:{run.sub_run_key or ''}",
        checkpoint_hashes={run.run_id: checkpoint_hash} if checkpoint_hash else {},
        seed=seed,
        sensitivity_variant=None,
        supported_count=0,
        abstained_count=0,
    )
    write_audit_artifacts(
        audit_report, experiment_dir, claim_id=f"ablation_{run.run_id}_seed{seed}"
    )


def _compute_harness_experiment_id(master_seed: int) -> str:
    """A simple, deterministic identifier for one harness run.

    `sha256(f"{_HARNESS_LABEL}:{master_seed}")[:16]` -- deterministic given
    `master_seed` alone, distinct from `infra.registry.compute_experiment_id`
    (this harness has no `ResolvedConfig` to hash; see module docstring).
    """
    digest = hashlib.sha256(f"{_HARNESS_LABEL}:{master_seed}".encode()).hexdigest()
    return digest[:16]


def _harness_dataset_checksum(treatment_arms: Mapping[str, int]) -> str:
    """A content hash of this harness's input `treatment_arms` mapping,
    standing in for a real dataset checksum this harness never loads
    (see `run_retrained_ablations`'s inline note)."""
    joined = "\n".join(f"{pid}:{arm}" for pid, arm in sorted(treatment_arms.items()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _harness_split_hash(patient_ids: Sequence[str]) -> str:
    """A content hash of a retained patient-id set, standing in for a
    real split hash (see `run_retrained_ablations`'s inline note)."""
    joined = "\n".join(sorted(patient_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dataset construction (retained dev/val populations)
# ---------------------------------------------------------------------------


def _retained_patients(
    split_patients: Sequence[PatientRecord], retained_patient_ids: Sequence[str]
) -> list[PatientRecord]:
    retained_ids = set(retained_patient_ids)
    return [patient for patient in split_patients if patient.patient_id in retained_ids]


def build_training_batches(
    pipeline_result: ObservationalizationPipelineResult,
) -> tuple[TensorBatch, TensorBatch, DatasetDimensions]:
    """Build the dev/val `TensorBatch`es every retrained arm trains against.

    Both batches are built from each split's *retained*
    (post-observationalization) population, matching
    `scripts/run_causal_analysis.py`'s established "retained dev split"
    convention. Standardization is fit on the retained development
    population only (`models.dataset.PatientDatasetBuilder.fit`).

    Returns:
        `(dev_batch, val_batch, dimensions)`. Every retrained arm in one
        harness run shares this one `(dev_batch, val_batch)` pair, so
        every arm is compared on identical populations.

    Raises:
        ValueError: if either split's retained population is empty.
    """
    dev_patients = _retained_patients(
        pipeline_result.split_cohorts.dev.patients, pipeline_result.dev.retained_patient_ids
    )
    val_patients = _retained_patients(
        pipeline_result.split_cohorts.val.patients, pipeline_result.val.retained_patient_ids
    )
    if not dev_patients:
        raise ValueError(
            "the development split's retained (post-observationalization) "
            "population is empty; cannot fit a model with no training data"
        )
    if not val_patients:
        raise ValueError(
            "the validation split's retained (post-observationalization) "
            "population is empty; cannot early-stop/select lambdas with no "
            "validation data"
        )

    builder = PatientDatasetBuilder().fit(dev_patients)
    dev_batch = builder.build(dev_patients, "dev")
    val_batch = builder.build(val_patients, "val")
    return dev_batch, val_batch, builder.dimensions


# ---------------------------------------------------------------------------
# Per-(run, seed) training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationMetricResult:
    """One trained model's primary factual metric on the retained
    Validation_Split population, plus the per-patient predictions and
    the propensity/support-gate flags task 14.1/14.3 consume downstream.

    Attributes:
        concordance_index: `evaluation.metrics.concordance_index` over
            `val_batch`'s retained patients, using the factual-arm
            96-week risk as the risk score.
        patient_ids: `val_batch.patient_ids`, in the same order as
            `predicted_risks` -- the Validation_Split identifier set this
            metric was computed over (Property 29).
        predicted_risks: per-patient factual-arm 96-week risk, in
            `patient_ids` order -- the patient-level predictions task
            14.3's collision detector compares across runs.
    """

    concordance_index: float
    patient_ids: tuple[str, ...]
    predicted_risks: tuple[float, ...]


def compute_validation_metric(model: Any, val_batch: TensorBatch) -> ValidationMetricResult:
    """Compute the frozen primary factual metric (time-dependent
    concordance) for `model` on `val_batch`'s retained population.

    Args:
        model: a trained `TorchSequenceGRU` or `TorchInterventionAwareModel`
            (in `eval()` mode, as `train_one_model` already leaves it).
        val_batch: the retained Validation_Split `TensorBatch` (never a
            Development_Split or test batch -- Requirement 7.11).

    Returns:
        A `ValidationMetricResult`.
    """
    import torch

    from models.torch_intervention_aware import outcome_from_hazards

    with torch.no_grad():
        outputs = factual_outputs(model, val_batch)
        risk = outcome_from_hazards(outputs.hazards)

    predicted_risks = tuple(float(v) for v in risk.cpu().tolist())
    event_times = tuple(float(v) for v in val_batch.event_time_weeks.cpu().tolist())
    event_indicators = tuple(bool(v) for v in val_batch.event_indicator.cpu().tolist())

    concordance = concordance_index(predicted_risks, event_times, event_indicators)

    return ValidationMetricResult(
        concordance_index=concordance,
        patient_ids=val_batch.patient_ids,
        predicted_risks=predicted_risks,
    )


@dataclass(frozen=True)
class RetrainedArmSeedResult:
    """One retrained ablation arm's outcome at one seed.

    Attributes:
        run_id: the expanded ablation run id this result belongs to.
        seed: the seed this attempt trained at.
        succeeded: whether training completed without raising.
        failure_reason: `None` on success; otherwise
            `f"{type(exc).__name__}: {exc}"` for the exception that
            stopped training (Requirement 7.16).
        checkpoint_path: path to the written checkpoint, or `None` on
            failure.
        checkpoint_sha256: hex SHA-256 of the checkpoint file, or `None`
            on failure.
        lambda_state: the `lambda_state` this attempt trained at.
        lambda_balance: the `lambda_balance` this attempt trained at --
            `0.0` for `predictive_accuracy_only`, selected for every
            other arm.
        placebo_vector: the drawn placebo vector, as a plain tuple, for
            the `placebo_intervention_channel` run; `None` for every
            other run.
        best_epoch: the checkpointed (validation-loss-minimizing) epoch.
        best_val_loss: that epoch's validation loss.
        concordance_index: the frozen primary factual metric
            (time-dependent concordance) on the retained
            Validation_Split population (Requirement 7.11), or `None` on
            failure.
        val_patient_ids: the Validation_Split patient id set this
            metric was computed over, or `None` on failure.
        val_predicted_risks: per-patient factual-arm 96-week risk, in
            `val_patient_ids` order, or `None` on failure -- the
            patient-level predictions task 14.3's collision detector
            compares across runs.
    """

    run_id: str
    seed: int
    succeeded: bool
    failure_reason: str | None
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    lambda_state: float | None
    lambda_balance: float | None
    placebo_vector: tuple[float, ...] | None
    best_epoch: int | None
    best_val_loss: float | None
    concordance_index: float | None = None
    val_patient_ids: tuple[str, ...] | None = None
    val_predicted_risks: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_run_lambdas(
    variant: ExperimentVariant,
    dimensions: DatasetDimensions,
    dev_batch: TensorBatch,
    val_batch: TensorBatch,
    selection_seed: int,
    training_config: Any,
) -> tuple[float, float]:
    """`(lambda_state, lambda_balance)` for `variant`, selected once.

    Runs the standard frozen 3x3 grid search exactly as every other arm
    does, then, for `predictive_accuracy_only`
    (`balance_penalty_enabled=False`), overrides the selected
    `lambda_balance` to `0.0` before returning (Requirement 7.9 --
    "disable the causal adjustment"). The balance term is excluded from
    that one arm's objective by zeroing its weight, not by running a
    different, smaller grid search restricted to `lambda_balance=0.0` --
    the grid itself stays the one frozen 3x3 grid for every arm.
    """
    batches = TrainingBatches(dev=dev_batch, val=val_batch)
    model_factory = lambda: build_model(variant, dimensions, selection_seed)  # noqa: E731
    selection = select_lambdas(model_factory, batches, selection_seed, config=training_config)
    lambda_state, lambda_balance = selection.selected
    if not variant.balance_penalty_enabled:
        lambda_balance = 0.0
    return lambda_state, lambda_balance


def train_retrained_arm_at_seed(
    run: AblationRun,
    dimensions: DatasetDimensions,
    dev_batch: TensorBatch,
    val_batch: TensorBatch,
    seed: int,
    lambda_state: float,
    lambda_balance: float,
    checkpoint_dir: Path,
    training_config: Any,
) -> RetrainedArmSeedResult:
    """Train `run`'s model from initialization at `seed`, under
    `training_config` (the frozen training configuration by default,
    Requirement 7.10).

    Never raises: any exception during construction or training is caught
    and returned as a failed `RetrainedArmSeedResult` (Requirement 7.16).
    """
    try:
        model = build_model(run.variant, dimensions, seed)
        checkpoint_path = checkpoint_dir / f"seed_{seed}.pt"
        result = train_one_model(
            model,
            TrainingBatches(dev=dev_batch, val=val_batch),
            lambda_state=lambda_state,
            lambda_balance=lambda_balance,
            seed=seed,
            config=training_config,
            checkpoint_path=checkpoint_path,
        )

        placebo_vector: tuple[float, ...] | None = None
        if run.variant.treatment_representation == "placebo_channel":
            placebo_vector = tuple(float(v) for v in result.model.placebo_vector.tolist())

        validation_metric = compute_validation_metric(result.model, val_batch)

        return RetrainedArmSeedResult(
            run_id=run.run_id,
            seed=seed,
            succeeded=True,
            failure_reason=None,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256(checkpoint_path),
            lambda_state=lambda_state,
            lambda_balance=lambda_balance,
            placebo_vector=placebo_vector,
            best_epoch=result.best_epoch,
            best_val_loss=result.best_val_loss,
            concordance_index=validation_metric.concordance_index,
            val_patient_ids=validation_metric.patient_ids,
            val_predicted_risks=validation_metric.predicted_risks,
        )
    except Exception as exc:  # noqa: BLE001 -- failures are data, see Requirement 7.16
        return RetrainedArmSeedResult(
            run_id=run.run_id,
            seed=seed,
            succeeded=False,
            failure_reason=f"{type(exc).__name__}: {exc}",
            checkpoint_path=None,
            checkpoint_sha256=None,
            lambda_state=lambda_state,
            lambda_balance=lambda_balance,
            placebo_vector=None,
            best_epoch=None,
            best_val_loss=None,
            concordance_index=None,
            val_patient_ids=None,
            val_predicted_risks=None,
        )


# ---------------------------------------------------------------------------
# Top-level harness
# ---------------------------------------------------------------------------


def run_retrained_ablations(
    raw_timelines: Sequence[DataPatientTimeline] | Mapping[str, DataPatientTimeline],
    treatment_arms: Mapping[str, int],
    spec: TargetTrialSpec,
    master_seed: int,
    results_root: Path,
    precomputed_endpoint_results: Mapping[str, EndpointResult] | None = None,
    seeds: Sequence[int] = REQUIRED_SEEDS,
    training_config: TrainingConfig = FROZEN_TRAINING_CONFIG,
    raise_on_collision: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run the six retrain-mode ablation arms through `build_model` and the
    frozen training harness, at every seed in `seeds`.

    Runs the frozen Section-9 pipeline exactly once
    (`cohorts.pipeline.run_observationalization_pipeline`) with the given
    `master_seed`, so every arm trains against the identical dev/val
    retained populations. For each of the six retrain-mode runs from
    `evaluation.ablation_matrix.expand_ablations()`: selects
    `(lambda_state, lambda_balance)` once (at `seeds[0]`), then trains
    that arm from a fresh `build_model` construction at every seed in
    `seeds`, writing a checkpoint and a result file per `(run, seed)`.

    Args:
        raw_timelines: candidate patients' `data.timelines.PatientTimeline`s.
        treatment_arms: `{patient_id: arm}` raw treatment-arm assignments.
        spec: the target-trial specification.
        master_seed: the single top-level seed the pipeline (splits,
            observationalization) is derived from.
        results_root: root directory under which
            `<experiment_id>/ablations/...` is written.
        precomputed_endpoint_results: forwarded unchanged to
            `run_observationalization_pipeline`.
        seeds: the seeds each retrain-mode arm is trained at. Defaults to
            the five frozen seeds (`infra.readiness.REQUIRED_SEEDS`);
            overridable for tests that need a shorter sequence.
        training_config: the optimizer/stopping configuration every
            retrain attempt uses. Defaults to the frozen
            `models.torch_training.FROZEN_TRAINING_CONFIG`
            (Requirement 7.10); overridable only so a test can shorten
            `max_epochs` -- production callers must leave this at the
            default.
        raise_on_collision: whether to raise `VariantCollisionError` when
            `evaluation.ablation_matrix.detect_variant_collisions` finds
            a collision (Requirement 7.13). Defaults to `True`
            (production behavior). `collision_report.json` and
            `ablation_table.json`'s `table_invalid` flag are written
            either way -- this parameter only controls whether a
            detected collision also raises. The one legitimate reason to
            set it `False` is a short/under-powered training budget on a
            small test cohort, where real (not synthetic) model outputs
            can genuinely coincide by small-sample chance; production
            callers must always leave this at the default.

    Returns:
        `{run_id: {"seeds": {seed: RetrainedArmSeedResult.to_dict()}}}`,
        one entry per retrain-mode run.

    Raises:
        VariantCollisionError: if `raise_on_collision` is `True` and a
            collision is detected.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")

    experiment_id = _compute_harness_experiment_id(master_seed)
    ablations_root = Path(results_root) / experiment_id / "ablations"

    pipeline_result = run_observationalization_pipeline(
        raw_timelines,
        treatment_arms,
        spec,
        master_seed,
        precomputed_endpoint_results=precomputed_endpoint_results,
    )
    dev_batch, val_batch, dimensions = build_training_batches(pipeline_result)

    # Task 17.2's audit reports need a `dataset_checksum`/`split_hash`,
    # but this harness (unlike `scripts/run_causal_analysis.py`) is
    # handed `raw_timelines` directly rather than loading them itself
    # through `data.loaders.actg175.load_actg175` against the frozen
    # dataset contract -- there is no real checksum available at this
    # scope. `_harness_dataset_checksum`/`_harness_split_hash` are
    # explicit, documented placeholders (a content hash of what this
    # harness actually has: the input patient-id/treatment-arm mapping,
    # and the retained dev-split patient-id set), not a substitute for
    # the real dataset checksum a caller that loaded real data would
    # have.
    dataset_checksum = _harness_dataset_checksum(treatment_arms)
    split_hash = _harness_split_hash(dev_batch.patient_ids)

    retrain_runs = tuple(run for run in expand_ablations() if run.execution_mode == "retrain")

    results: dict[str, dict[str, Any]] = {}
    table_incomplete = False

    for run in retrain_runs:
        lambda_state, lambda_balance = _select_run_lambdas(
            run.variant,
            dimensions,
            dev_batch,
            val_batch,
            selection_seed=seeds[0],
            training_config=training_config,
        )

        checkpoint_dir = ablations_root / run.run_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        per_seed: dict[str, Any] = {}
        for seed in seeds:
            seed_result = train_retrained_arm_at_seed(
                run,
                dimensions,
                dev_batch,
                val_batch,
                seed,
                lambda_state,
                lambda_balance,
                checkpoint_dir,
                training_config=training_config,
            )
            if not seed_result.succeeded:
                table_incomplete = True

            record = seed_result.to_dict()
            per_seed[str(seed)] = record

            result_path = ablations_root / f"run_{run.run_id}_seed{seed}.json"
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)

            # Task 17.2: one public/protected audit-artifact pair per
            # (run, seed) candidate estimate, suppressed or not
            # (Requirement 12.1, 12.2) -- see
            # `_write_ablation_audit_artifacts`'s docstring for why every
            # one of them is unconditionally `"suppressed"` at this
            # harness's current scope.
            _write_ablation_audit_artifacts(
                run,
                seed,
                record,
                Path(results_root) / experiment_id,
                spec,
                dataset_checksum,
                split_hash,
            )

        results[run.run_id] = {
            "run_id": run.run_id,
            "source_entry": run.source_entry,
            "sub_run_key": run.sub_run_key,
            "execution_mode": run.execution_mode,
            "lambda_state": lambda_state,
            "lambda_balance": lambda_balance,
            "seeds": per_seed,
        }

    summary = {
        "experiment_id": experiment_id,
        "master_seed": master_seed,
        "seeds": list(seeds),
        "table_incomplete": table_incomplete,
        "runs": results,
    }
    summary_path = ablations_root / "retrained_arms_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    collision_report = _detect_collisions_among_retrained_runs(retrain_runs, results)
    if collision_report.has_collision:
        collision_path = ablations_root / "collision_report.json"
        with collision_path.open("w", encoding="utf-8") as handle:
            json.dump(collision_report.to_dict(), handle, indent=2, sort_keys=True)

    ablation_table = build_ablation_table(
        results, table_incomplete, table_invalid=collision_report.has_collision
    )
    table_path = ablations_root / "ablation_table.json"
    with table_path.open("w", encoding="utf-8") as handle:
        json.dump(ablation_table, handle, indent=2, sort_keys=True)

    if collision_report.has_collision and raise_on_collision:
        raise VariantCollisionError(
            "variant collision detected among the retrained ablation arms "
            f"(prediction_collisions={collision_report.prediction_collisions!r}, "
            f"insufficient_distinct_metrics={collision_report.insufficient_distinct_metrics!r}, "
            f"distinct_metric_count={collision_report.distinct_metric_count!r}); "
            f"see {ablations_root / 'collision_report.json'}"
        )

    return results


def _detect_collisions_among_retrained_runs(
    retrain_runs: Sequence[AblationRun], results: dict[str, dict[str, Any]]
):
    """Build `predictions_by_run`/`metrics_by_run` from `results` and call
    `evaluation.ablation_matrix.detect_variant_collisions` (Requirement
    7.13, 6.4).

    Only succeeded `(run, seed)` attempts contribute a prediction/metric
    entry -- a failed attempt (Requirement 7.16) has no predictions to
    compare and is simply absent from both mappings at that seed, not
    treated as a `0.0`/missing-value collision candidate.
    """
    predictions_by_run: dict[str, dict[int, dict[str, float]]] = {}
    metrics_by_run: dict[str, dict[int, float]] = {}

    for run_id, payload in results.items():
        for seed_str, seed_record in payload["seeds"].items():
            if not seed_record.get("succeeded"):
                continue
            seed = int(seed_str)
            patient_ids = seed_record.get("val_patient_ids")
            predicted_risks = seed_record.get("val_predicted_risks")
            concordance = seed_record.get("concordance_index")
            if patient_ids is not None and predicted_risks is not None:
                predictions_by_run.setdefault(run_id, {})[seed] = dict(
                    zip(patient_ids, predicted_risks, strict=True)
                )
            if concordance is not None:
                metrics_by_run.setdefault(run_id, {})[seed] = concordance

    return detect_variant_collisions(list(retrain_runs), predictions_by_run, metrics_by_run)


def build_ablation_table(
    results: dict[str, dict[str, Any]], table_incomplete: bool, table_invalid: bool = False
) -> dict[str, Any]:
    """Aggregate `run_retrained_ablations`'s per-run results into
    `ablation_table.json`'s shape (Requirement 7.11, 7.12).

    For each run: the run id, the `ExperimentVariant` fields, the
    checkpoint hash for every succeeded seed, the evaluated split
    (always `"val"` for this metric), every seed's concordance-index
    value, and the mean across succeeded seeds. A run with zero
    succeeded seeds reports a `nan` mean rather than raising.

    Args:
        results: `run_retrained_ablations`'s return value.
        table_incomplete: whether any `(run, seed)` attempt failed
            (Requirement 7.16) -- recorded on the table itself so a
            consumer does not have to re-derive it from the per-run
            entries.
        table_invalid: whether `evaluation.ablation_matrix.detect_variant_collisions`
            found a collision (Requirement 7.13). `invalid` and
            `incomplete` are independent, non-exclusive conditions: a
            table can be both (some arms failed to train AND the arms
            that did complete collided).

    Returns:
        A JSON-serializable dict with `table_incomplete`, `table_invalid`,
        and `runs`, one entry per run id.
    """
    rows: dict[str, Any] = {}
    for run_id, payload in results.items():
        per_seed_concordance: dict[str, float | None] = {}
        checkpoint_hashes: dict[str, str | None] = {}
        succeeded_values: list[float] = []

        for seed_str, seed_record in payload["seeds"].items():
            concordance = seed_record.get("concordance_index")
            per_seed_concordance[seed_str] = concordance
            checkpoint_hashes[seed_str] = seed_record.get("checkpoint_sha256")
            if seed_record.get("succeeded") and concordance is not None:
                succeeded_values.append(concordance)

        mean_concordance = (
            sum(succeeded_values) / len(succeeded_values)
            if succeeded_values
            else float("nan")
        )

        rows[run_id] = {
            "run_id": run_id,
            "source_entry": payload["source_entry"],
            "sub_run_key": payload["sub_run_key"],
            "execution_mode": payload["execution_mode"],
            "evaluated_split": "val",
            "per_seed_concordance_index": per_seed_concordance,
            "mean_concordance_index": mean_concordance,
            "checkpoint_sha256_by_seed": checkpoint_hashes,
        }

    return {
        "table_incomplete": table_incomplete,
        "table_invalid": table_invalid,
        "runs": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six retrain-mode frozen primary ablation arms "
            "(evaluation.ablation_matrix.expand_ablations(), execution_mode='retrain') "
            "through models.variants.build_model and the frozen training harness, "
            "against the real, packaged ACTG175 v3.2 raw data by default. The "
            "integration test (tests/integration/test_run_ablations.py) exercises "
            "run_retrained_ablations() directly against a synthetic cohort instead, "
            "for fast, data-independent testing."
        )
    )
    parser.add_argument(
        "--target-trial-config",
        type=Path,
        required=True,
        help="Path to a target-trial-spec YAML config (see configs/target_trials/main.yaml).",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        required=True,
        help="The single top-level seed the pipeline (splits, observationalization) uses.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root directory results are written under (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--real-data-path",
        type=Path,
        default=DEFAULT_REAL_DATA_PATH,
        help=(
            "Directory containing the real raw ACTG175 v3.2 table CSVs "
            f"(default: {DEFAULT_REAL_DATA_PATH}). If this path does not exist, "
            "the command refuses with a clear error rather than falling back to "
            "a synthetic cohort."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(REQUIRED_SEEDS),
        help=f"Seeds each retrain-mode arm is trained at (default: {list(REQUIRED_SEEDS)}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load the real ACTG175 data, adapt it into pipeline
    inputs, and run the six retrain-mode frozen ablation arms against it.

    Loads real raw tables via `data.loaders.actg175.load_actg175`
    (checksum-verified against the frozen dataset contract) and converts
    them via `data.loaders.actg175_adapter.raw_tables_to_pipeline_inputs`.

    Returns 0 on success. On any exception -- including the raw data being
    absent at `--real-data-path` -- prints the exception message to stderr
    and returns 1 without raising. The substantive logic lives in
    `run_retrained_ablations`, which is exercised directly (against a
    synthetic cohort, for fast tests) by
    `tests/integration/test_run_ablations.py`.
    """
    import yaml

    from protocols.schema import TargetTrialSpec

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        with args.target_trial_config.open("r", encoding="utf-8") as handle:
            raw_spec = yaml.safe_load(handle)
        spec = TargetTrialSpec.model_validate(raw_spec)

        tables = load_actg175(args.real_data_path)
        timelines, treatment_arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)
        print(
            f"loaded real ACTG175 data: {len(timelines)} arm-0/1 patients "
            f"(checksum {tables.checksum_sha256[:12]}...)"
        )

        results = run_retrained_ablations(
            timelines,
            treatment_arms,
            spec,
            args.master_seed,
            args.results_root,
            precomputed_endpoint_results=endpoint_results,
            seeds=tuple(args.seeds),
        )
    except Actg175DataNotFoundError as exc:
        print(
            f"run_ablations.py failed: real ACTG175 data not found: {exc}. "
            "See docs/data_access.md Section 2 for the approved storage location "
            "(data/raw/actg175_v3_2/) and Section 5 for the data-use agreement.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary, see docstring
        print(f"run_ablations.py failed: {exc}", file=sys.stderr)
        return 1

    experiment_id = _compute_harness_experiment_id(args.master_seed)
    print(
        f"ran {len(results)} retrain-mode ablation arms at {len(args.seeds)} seed(s); "
        f"results under {args.results_root / experiment_id / 'ablations'}"
    )
    print("run_ablations.py succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
