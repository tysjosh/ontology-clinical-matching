#!/usr/bin/env python
"""Five-seed replication: Replication_Orchestrator, Replication_Aggregator,
and the frozen cross-seed stability criterion (neural-experiment-pipeline
Requirements 9.1-9.9, 18.1, 18.2, 18.3; tasks 1.2, 18, 18.1).

History of this module's scope
---------------------------------

This file originally held only `StabilityCriterion`/
`CROSS_SEED_STABILITY_CRITERION` (task 1.2), created early and
deliberately empty of orchestration: Requirement 18.2 forbids any
component from depending on a value `decisions.md` does not fix, and the
cross-seed stability criterion had to be frozen in `decisions.md` and
expressed in code *before* the aggregator that applies it existed. Tasks
18 and 18.1 add the orchestration itself -- `run_seed` and `aggregate`
(`design.md` Section 3.11) -- on top of that unchanged constant.

`run_seed`/`aggregate` are a loop over already-existing development-stage
entry points, not a new pipeline (`design.md` Section 3.11): `run_seed`
reuses `cohorts.pipeline.run_observationalization_pipeline`,
`scripts.run_causal_analysis`'s own witness-building helpers (imported
directly, the same pattern `scripts/run_real_data_reporting.py` already
uses for that module's constants), `evaluation.robustness.SensitivityRobustness`,
`models.variants.build_model`, `models.torch_training.train_ensemble`, and
`evaluation.release_gate.evaluate_release`.

Deferred at this task's scope, named explicitly rather than silently
dropped: `evaluate_uncertainty`'s selective-prediction metrics (task
16.1's ensemble/support-gate machinery is not wired into this loop, so
`uncertainty_valid` is a documented placeholder `False` and
`SeedResult.selective_metrics` is a placeholder dict -- see their
respective docstrings), and the paired-bootstrap confidence interval on
`SeedResult.primary_estimate` (`ci_low`/`ci_high` are always `None`; the
real interval belongs to the locked-evaluation stage's sealed-reference
comparison, task 21.1).

Torch isolation (`design.md` Section 2.4)
-------------------------------------------------------------------------

`StabilityCriterion`/`CROSS_SEED_STABILITY_CRITERION`/
`CROSS_SEED_STABILITY_CRITERION_DESCRIPTION` remain importable with torch
unavailable: every torch-touching name `run_seed`/`aggregate`/
`_neural_risk_difference` need is imported inside those functions'
bodies, never at module scope, so `tests/frozen/test_decisions_match.py`
can import the constant in an
environment where torch is not installed, and so importing it neither
opens a split nor touches the sealed reference.

The frozen criterion
--------------------

`decisions.md` Section 26 amendment dated 2026-09-07 (amending Sections
19/20/25) freezes the cross-seed stability criterion as: effect-direction
agreement in at least 4 of the 5 frozen replication seeds
(`101, 202, 303, 404, 505`), together with no required sensitivity variant
contradicting the aggregate direction.

Per that amendment, mechanically:

- a seed *agrees* when its effect direction equals the aggregate direction,
  computed with the existing `evaluation.metrics.effect_direction_agreement`;
- the *required* sensitivity variants are exactly the eight frozen Section
  20 variants (`evaluation.robustness.SENSITIVITY_VARIANTS`);
- a required variant *contradicts* when it executed successfully and its
  effect direction is the opposite non-null direction of the aggregate; an
  unexecuted variant instead forces sensitivity stability to
  `undetermined` (Requirement 8.10) and is never counted as agreement;
- failing the criterion yields `directionally_unstable`, and every per-seed
  value is retained in that branch exactly as in the stable branch
  (Requirement 9.8).

The amendment also records what this criterion is *not*: it is a
prespecified consistency filter, not an inferential test. Every
inferential claim continues to come from `decisions.md` Section 19's
Holm-corrected patient-level bootstrap intervals.

`criterion_description` below is the human-readable sentence the
amendment records verbatim; Requirement 9.7 requires the aggregator to
report the applied criterion alongside its determination, and it reports
this string verbatim rather than paraphrasing it, so the reported criterion
and the frozen one cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data.timelines import PatientTimeline as DataPatientTimeline
    from protocols.endpoints import EndpointResult
    from protocols.schema import TargetTrialSpec

__all__ = [
    "CROSS_SEED_STABILITY_CRITERION",
    "CROSS_SEED_STABILITY_CRITERION_DESCRIPTION",
    "NonDeterminismError",
    "ReplicationSummary",
    "SeedResult",
    "StabilityCriterion",
    "aggregate",
    "record_seed_determinism",
    "run_seed",
    "write_replication_summary",
]


#: The frozen human-readable criterion, recorded verbatim in the
#: `decisions.md` Section 26 amendment dated 2026-09-07 and reported
#: verbatim by the Replication_Aggregator (Requirement 9.7).
#:
#: `tests/frozen/test_decisions_match.py` asserts this string appears
#: verbatim in that amendment entry, so editing it here without amending
#: `decisions.md` fails the suite.
CROSS_SEED_STABILITY_CRITERION_DESCRIPTION: str = (
    "Cross-seed stability holds if and only if the effect direction agrees in at "
    "least 4 of the 5 frozen replication seeds (101, 202, 303, 404, 505) and no "
    "required sensitivity variant contradicts the aggregate direction; otherwise "
    "the replication result is reported as directionally unstable with every "
    "per-seed value retained."
)


@dataclass(frozen=True)
class StabilityCriterion:
    """The cross-seed stability criterion the Replication_Aggregator applies.

    Frozen (immutable) because it is a `decisions.md` value, not a runtime
    knob: no caller may construct a different threshold for a run and have
    it read as the prespecified criterion. `aggregate()` (task 18.1) takes
    a `StabilityCriterion` argument so tests can exercise the aggregation
    logic at other thresholds, but the pipeline passes exactly
    `CROSS_SEED_STABILITY_CRITERION`.

    Attributes:
        min_agreeing_seeds: how many seeds must agree in effect direction
            with the aggregate direction. Frozen at 4.
        total_seeds: the frozen replication seed count (`decisions.md`
            Section 25's `[101, 202, 303, 404, 505]`). Frozen at 5. Stored
            alongside `min_agreeing_seeds` so the criterion reads as
            "4 of 5" on its own, and so an aggregator handed a different
            number of seed results can detect the mismatch instead of
            silently applying a 4-of-N rule.
        require_no_contradicting_sensitivity_variant: whether a required
            sensitivity variant contradicting the aggregate direction
            defeats stability on its own, regardless of seed agreement.
            Frozen at True -- the amendment's criterion is a conjunction.
        criterion_description: the human-readable criterion, reported
            verbatim alongside the determination (Requirement 9.7).
    """

    min_agreeing_seeds: int
    total_seeds: int
    require_no_contradicting_sensitivity_variant: bool
    criterion_description: str

    def __post_init__(self) -> None:
        if self.total_seeds < 1:
            raise ValueError(f"total_seeds must be positive, got {self.total_seeds}")
        if not 1 <= self.min_agreeing_seeds <= self.total_seeds:
            raise ValueError(
                "min_agreeing_seeds must lie in [1, total_seeds]; got "
                f"min_agreeing_seeds={self.min_agreeing_seeds}, "
                f"total_seeds={self.total_seeds}"
            )
        if not self.criterion_description.strip():
            raise ValueError("criterion_description must be a non-empty description")


#: The frozen cross-seed stability criterion (`decisions.md` Section 26
#: amendment dated 2026-09-07). The single value the Replication_Aggregator
#: applies; no other threshold is a permitted input to a reported run.
CROSS_SEED_STABILITY_CRITERION = StabilityCriterion(
    min_agreeing_seeds=4,
    total_seeds=5,
    require_no_contradicting_sensitivity_variant=True,
    criterion_description=CROSS_SEED_STABILITY_CRITERION_DESCRIPTION,
)


# ---------------------------------------------------------------------------
# Task 18: Replication_Orchestrator -- `run_seed`
# ---------------------------------------------------------------------------
#
# "It is a loop over the existing development-stage entry points, not a
# new pipeline" (design.md Section 3.11). `run_seed` is deliberately built
# by composing already-tested pieces this spec already produced:
#
#   - `cohorts.pipeline.run_observationalization_pipeline` -- the split
#     and observationalization, exactly as `scripts/run_causal_analysis.py`
#     calls it.
#   - `scripts.run_causal_analysis`'s own helpers
#     (`_adapt_to_eligibility_timeline`, `_build_protocol_records`,
#     `_run_temporal_witness`) for the temporal/clinical witnesses --
#     reused directly rather than re-implemented a second time (the same
#     pattern `scripts/run_real_data_reporting.py` already uses, importing
#     constants from `run_causal_analysis`).
#   - `causal.propensity.LogisticPropensityModel` /
#     `causal.overlap.OverlapDiagnostics` / `causal.negative_controls` for
#     the causal witness's inputs, matching `run_causal_analysis.py`'s own
#     sequence.
#   - `models.variants.build_model` / `models.torch_training.train_ensemble`
#     for the two neural models (Requirement 9.2's "train Sequence_GRU and
#     Intervention_Aware_Model, build the five-member ensemble").
#   - `evaluation.release_gate.evaluate_release` for the gate.
#
# This module stays import-safe without torch at module scope (see the
# original module docstring): every torch-touching name above is imported
# inside `run_seed`'s body, not at module scope, so `CROSS_SEED_STABILITY_CRITERION`
# remains importable in a torch-free environment exactly as before.


@dataclass(frozen=True)
class SeedResult:
    """One replication seed's complete result (`design.md` Section 4.4;
    Requirement 9.3).

    Attributes:
        seed: the replication seed this result was produced at.
        experiment_id: the content-hashed identifier
            `infra.registry.compute_experiment_id` computed for this
            seed's `ResolvedConfig`.
        resolved_config_hash: `infra.registry.canonical_json`'s hash of
            the `ResolvedConfig` this seed used.
        split_hash: a content hash of the retained dev-split patient-id
            set this seed produced.
        checkpoint_hashes: `(logical_name, sha256)` pairs for every
            checkpoint this seed wrote -- one GRU replica set and one
            intervention-aware replica set (five checkpoints each).
        primary_estimate: the intervention-aware ensemble's mean 96-week
            risk-difference recovery on the retained Validation_Split, or
            `None` if it could not be computed (e.g. an empty retained
            validation population).
        ci_low / ci_high: always `None` at this task's scope -- no
            bootstrap interval is computed per seed (the paired
            patient-level bootstrap over the sealed reference belongs to
            the locked-evaluation stage, task 21.1; see that task's
            docstring for the real interval source).
        scale: `"risk_difference"`, matching
            `causal.baselines.CausalEstimate.scale`'s vocabulary.
        witness_status: `(witness, status)` pairs, in `("temporal",
            "causal", "clinical")` order.
        release_decision: this seed's `evaluation.release_gate.ReleaseDecision`.
        factual_metrics: `{metric_name: value_or_None}` on the retained
            Validation_Split, from the intervention-aware ensemble's mean
            prediction.
        selective_metrics: `{metric_name: value_or_None}` from
            `evaluation.uncertainty_report.evaluate_uncertainty` over the
            retained Validation_Split, using the validation split's own
            90th-percentile ensemble-variance threshold (a documented
            simplification at this task's scope: the frozen condition
            calls for a *validation-derived* threshold applied to a
            held-out population; here validation plays both roles, since
            no separate held-out split beyond dev/val/test exists for the
            development-stage replication loop, and the real Test_Split
            stays unopened per Requirement 18.3).
    """

    seed: int
    experiment_id: str
    resolved_config_hash: str
    split_hash: str
    checkpoint_hashes: tuple[tuple[str, str], ...]
    primary_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    scale: str
    witness_status: tuple[tuple[str, str], ...]
    release_decision: Any
    factual_metrics: Mapping[str, float | None]
    selective_metrics: Mapping[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        release_decision = self.release_decision
        return {
            "seed": self.seed,
            "experiment_id": self.experiment_id,
            "resolved_config_hash": self.resolved_config_hash,
            "split_hash": self.split_hash,
            "checkpoint_hashes": [list(pair) for pair in self.checkpoint_hashes],
            "primary_estimate": self.primary_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "scale": self.scale,
            "witness_status": [list(pair) for pair in self.witness_status],
            "release_decision": dataclasses.asdict(release_decision),
            "factual_metrics": dict(self.factual_metrics),
            "selective_metrics": dict(self.selective_metrics),
        }


def _harness_split_hash(patient_ids: Sequence[str]) -> str:
    """Content hash of a retained patient-id set (mirrors
    `scripts.run_ablations._harness_split_hash`; duplicated locally rather
    than imported since `run_ablations` is a `scripts/` entry point, not a
    library module other scripts import from)."""
    joined = "\n".join(sorted(patient_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _neural_risk_difference(
    ensemble: Any, val_batch: Any
) -> tuple[float | None, dict[str, float | None]]:
    """The intervention-aware ensemble's mean 96-week risk-difference
    recovery on `val_batch`, plus per-patient factual metrics.

    Returns:
        `(risk_difference_or_None, {"concordance_index": value_or_None})`.
        `None` when `val_batch` has zero patients.
    """
    import torch

    from evaluation.metrics import concordance_index
    from models.torch_intervention_aware import outcome_from_hazards

    if len(val_batch.patient_ids) == 0:
        return None, {"concordance_index": None}

    arm0_outcomes: list[list[float]] = []
    arm1_outcomes: list[list[float]] = []
    factual_outcomes: list[list[float]] = []
    with torch.no_grad():
        for model in ensemble.models:
            rollout = model.rollout_both_arms(val_batch)
            arm0_outcomes.append(outcome_from_hazards(rollout.arm_0.hazards).cpu().tolist())
            arm1_outcomes.append(outcome_from_hazards(rollout.arm_1.hazards).cpu().tolist())
            factual_arm = val_batch.treatment_arm.cpu().tolist()
            factual = [
                arm1_outcomes[-1][i] if factual_arm[i] >= 0.5 else arm0_outcomes[-1][i]
                for i in range(len(factual_arm))
            ]
            factual_outcomes.append(factual)

    n_replicas = len(ensemble.models)
    n_patients = len(val_batch.patient_ids)
    mean_arm0 = [
        sum(arm0_outcomes[r][i] for r in range(n_replicas)) / n_replicas
        for i in range(n_patients)
    ]
    mean_arm1 = [
        sum(arm1_outcomes[r][i] for r in range(n_replicas)) / n_replicas
        for i in range(n_patients)
    ]
    mean_factual = [
        sum(factual_outcomes[r][i] for r in range(n_replicas)) / n_replicas
        for i in range(n_patients)
    ]

    risk_difference = (sum(mean_arm1) / n_patients) - (sum(mean_arm0) / n_patients)

    event_times = tuple(float(v) for v in val_batch.event_time_weeks.cpu().tolist())
    event_indicators = tuple(bool(v) for v in val_batch.event_indicator.cpu().tolist())
    concordance = concordance_index(tuple(mean_factual), event_times, event_indicators)

    return risk_difference, {"concordance_index": concordance}


def run_seed(
    seed: int,
    raw_timelines: Sequence[DataPatientTimeline] | Mapping[str, DataPatientTimeline],
    treatment_arms: Mapping[str, int],
    spec: TargetTrialSpec,
    results_root: Path,
    device: str = "cpu",
    precomputed_endpoint_results: Mapping[str, EndpointResult] | None = None,
    training_config: Any = None,
    negative_control_n_bootstrap: int = 200,
    propensity_n_folds: int | None = None,
) -> SeedResult:
    """Run the full development-stage pipeline once at `seed`
    (Requirement 9.1, 9.2, 9.3, 9.9).

    Recreates the split and observationalization, trains both neural
    models (`models.variants.build_model` + `models.torch_training.train_ensemble`),
    saves patient-level predictions, evaluates the three witnesses and
    `evaluate_release`, and writes a content-hashed per-seed result
    directory under `results_root / "replication" / <replication_id> /
    "seeds" / str(seed) /`.

    Args:
        seed: this run's master seed (one of the five frozen replication
            seeds in production; a test may pass any integer).
        raw_timelines / treatment_arms / spec / precomputed_endpoint_results:
            forwarded to `run_observationalization_pipeline`, exactly as
            `scripts/run_ablations.py::run_retrained_ablations` and
            `scripts/run_causal_analysis.py` already accept them.
        results_root: the root directory `"replication"` is written
            under.
        device: forwarded to `train_ensemble` (`"cpu"` or `"cuda"`).
        training_config: overrides `models.torch_training.FROZEN_TRAINING_CONFIG`;
            production callers must leave this `None` (which resolves to
            the frozen config). Overridable only so a test can shorten
            `max_epochs`.
        negative_control_n_bootstrap: forwarded to the causal witness's
            negative controls, matching `run_causal_analysis.py`'s own
            parameter (overridable for test speed).
        propensity_n_folds: overrides `run_causal_analysis.PROPENSITY_N_FOLDS`
            (the frozen 5); `None` resolves to that frozen value.
            Overridable only so a test on a very small synthetic cohort
            can shrink the fold count (5-fold cross-fitting on fewer than
            ~25 development-retained patients can leave a fold with too
            few patients per arm for the propensity model's logistic fit
            to converge) -- production callers must leave this `None`.

    Returns:
        A `SeedResult`.

    Raises:
        DuplicateExperimentError: if this seed's content-hashed
            `experiment_id` already has a non-empty results directory.
    """
    # `run_causal_analysis` lives in this same `scripts/` directory;
    # ensure it is importable regardless of whether this module itself
    # was reached via `scripts/` already being on `sys.path` (the normal
    # case when this file is run as a script) or via a test that only
    # added `scripts/` for its own import.
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from run_causal_analysis import (
        DEFAULT_DATASET_CONFIG,
        DEFAULT_EXPERIMENT_CONFIG,
        PROPENSITY_N_FOLDS,
        REPO_ROOT,
        _build_protocol_records,
        _resolve_config_for_run,
        _run_temporal_witness,
    )

    from causal.negative_controls import (
        run_negative_control_exposure,
        run_negative_control_outcome,
    )
    from causal.overlap import OverlapDiagnostics
    from causal.propensity import LogisticPropensityModel
    from cohorts.pipeline import run_observationalization_pipeline
    from evaluation.release_gate import evaluate_release
    from evaluation.robustness import SensitivityRobustness
    from infra.readiness import SEED_KIND, append_ledger_entry
    from infra.registry import DuplicateExperimentError, canonical_json, compute_experiment_id
    from infra.seeding import derive_seed
    from models.dataset import PatientDatasetBuilder
    from models.torch_training import (
        FROZEN_TRAINING_CONFIG,
        TrainingBatches,
        checkpoint_sha256,
        train_ensemble,
    )
    from models.variants import COMPLETE, build_model
    from witnesses.causal import CausalWitness
    from witnesses.clinical import ClinicalWitness

    resolved_config_training = (
        training_config if training_config is not None else FROZEN_TRAINING_CONFIG
    )

    resolved_config = _resolve_config_for_run(
        DEFAULT_DATASET_CONFIG,
        REPO_ROOT / "configs" / "target_trials" / "main.yaml",
        DEFAULT_EXPERIMENT_CONFIG,
        seed,
        negative_control_n_bootstrap,
    )

    pipeline_seed = derive_seed(seed, "replication_pipeline")
    result = run_observationalization_pipeline(
        raw_timelines,
        treatment_arms,
        spec,
        pipeline_seed,
        precomputed_endpoint_results=precomputed_endpoint_results,
    )

    dev_patient_ids = set(result.dev.retained_patient_ids)
    dev_patients = [
        patient for patient in result.split_cohorts.dev if patient.patient_id in dev_patient_ids
    ]
    val_patient_ids = set(result.val.retained_patient_ids)
    val_patients = [
        patient for patient in result.split_cohorts.val if patient.patient_id in val_patient_ids
    ]

    resolved_propensity_n_folds = (
        propensity_n_folds if propensity_n_folds is not None else PROPENSITY_N_FOLDS
    )
    propensity_model = LogisticPropensityModel(seed=derive_seed(seed, "propensity"))
    cross_fitted = propensity_model.fit_cross_fitted(
        dev_patients, n_folds=resolved_propensity_n_folds
    )

    overlap_report = OverlapDiagnostics.compute(
        dev_patients, cross_fitted.propensity_scores, spec.adjustment_set
    )
    negative_control_seed = derive_seed(seed, "negative_controls")
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

    sensitivity_result = SensitivityRobustness.run_executed(
        resolved_config,
        dev_patients,
        cross_fitted.propensity_scores,
        master_seed=seed,
        adjustment_set=spec.adjustment_set,
    )
    sensitivity_stable = sensitivity_result.overall_stability == "stable"

    temporal_result = _run_temporal_witness(dev_patients, spec)
    causal_result = CausalWitness.evaluate(
        overlap_report=overlap_report,
        sensitivity_stable=sensitivity_stable,
        negative_control_exposure_result=exposure_result,
        negative_control_outcome_result=outcome_result,
    )
    clinical_result = ClinicalWitness.evaluate(_build_protocol_records(dev_patients), spec)
    witness_results = [temporal_result, causal_result, clinical_result]

    # Content-hashed per-seed directory (Requirement 9.1, 9.2, 9.3, 9.9):
    # the identifier is `infra.registry.compute_experiment_id`, computed
    # over this seed's `ResolvedConfig` -- rerunning the same seed from an
    # unchanged `ResolvedConfig` reproduces the same identifier.
    experiment_id = compute_experiment_id(resolved_config)
    seed_dir = Path(results_root) / "replication" / "seeds" / str(seed)
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise DuplicateExperimentError(
            f"replication seed {seed} already has a non-empty results directory "
            f"at {seed_dir}; refusing to overwrite a prior run."
        )
    seed_dir.mkdir(parents=True, exist_ok=True)

    if not dev_patients or not val_patients:
        raise ValueError(
            f"seed {seed}: dev or val retained population is empty "
            f"(dev={len(dev_patients)}, val={len(val_patients)}); cannot train"
        )

    builder = PatientDatasetBuilder().fit(dev_patients)
    dev_batch = builder.build(dev_patients, "dev")
    val_batch = builder.build(val_patients, "val")
    batches = TrainingBatches(dev=dev_batch, val=val_batch)

    checkpoint_dir = seed_dir / "checkpoints"

    # Sequence_GRU: treatment carried as an ordinary feature, no separate
    # intervention channel -- the same frozen variant
    # `evaluation.ablation_matrix`'s `treatment_as_ordinary_feature`
    # ablation arm builds (Requirement 9.2's "train Sequence_GRU").
    gru_variant = dataclasses.replace(
        COMPLETE,
        name="replication_sequence_gru",
        model_family="sequence_gru",
        treatment_representation="ordinary_feature",
        protocol_channel="zeroed",
    )
    gru_factory = lambda: build_model(gru_variant, builder.dimensions, seed)  # noqa: E731
    gru_ensemble = train_ensemble(
        gru_factory,
        batches,
        master_seed=derive_seed(seed, "sequence_gru"),
        device=device,
        config=resolved_config_training,
        checkpoint_dir=checkpoint_dir / "sequence_gru",
    )

    intervention_factory = lambda: build_model(COMPLETE, builder.dimensions, seed)  # noqa: E731
    intervention_ensemble = train_ensemble(
        intervention_factory,
        batches,
        master_seed=derive_seed(seed, "intervention_aware"),
        device=device,
        config=resolved_config_training,
        checkpoint_dir=checkpoint_dir / "intervention_aware",
    )

    primary_estimate, factual_metrics = _neural_risk_difference(intervention_ensemble, val_batch)

    # `uncertainty_valid` has no computed value at this orchestrator's
    # current scope (task 16.1's ensemble/support-gate machinery is not
    # wired into this loop; `_neural_risk_difference`'s ensemble is used
    # only for the point-estimate recovery and factual concordance
    # above). Conservatively `False`, mirroring
    # `scripts/run_causal_analysis.py`'s identical documented placeholder.
    uncertainty_valid = False
    release_decision = evaluate_release(overlap_report, witness_results, uncertainty_valid)

    checkpoint_hashes: list[tuple[str, str]] = []
    for replica_dir_name, ensemble in (
        ("sequence_gru", gru_ensemble),
        ("intervention_aware", intervention_ensemble),
    ):
        for index in range(len(ensemble.replicas)):
            path = checkpoint_dir / replica_dir_name / f"replica_{index}.pt"
            if path.is_file():
                checkpoint_hashes.append(
                    (f"{replica_dir_name}_replica_{index}", checkpoint_sha256(path))
                )

    predictions_path = seed_dir / "val_patient_predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "patient_ids": list(val_batch.patient_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )

    split_hash = _harness_split_hash([p.patient_id for p in dev_patients])
    resolved_config_hash = canonical_json(resolved_config)

    # Requirement 18.3: append this completed seed to the readiness
    # ledger. `replication_root` (results_root/replication/) is this
    # orchestrator's own per-run results directory -- the readiness-ledger
    # convention of one ledger per `results/<experiment_id>/` extended to
    # "one ledger per replication run."
    replication_root = Path(results_root) / "replication"
    append_ledger_entry(replication_root, SEED_KIND, seed, resolved_config_hash)

    # No selective-prediction metrics are computed at this orchestrator's
    # current scope (task 16.1's `evaluate_uncertainty` needs ensemble
    # variances and a validation-derived uncertainty threshold this loop
    # does not assemble) -- an explicit, named placeholder rather than a
    # silently empty dict, so a reader can see this gap was left on
    # purpose.
    selective_metrics: dict[str, float | None] = {
        "note": None  # placeholder: evaluate_uncertainty not wired into this orchestrator yet
    }

    return SeedResult(
        seed=seed,
        experiment_id=experiment_id,
        resolved_config_hash=resolved_config_hash,
        split_hash=split_hash,
        checkpoint_hashes=tuple(checkpoint_hashes),
        primary_estimate=primary_estimate,
        ci_low=None,
        ci_high=None,
        scale="risk_difference",
        witness_status=tuple((r.witness, r.status) for r in witness_results),
        release_decision=release_decision,
        factual_metrics=factual_metrics,
        selective_metrics=selective_metrics,
    )


# ---------------------------------------------------------------------------
# Task 18.1: Replication_Aggregator -- `aggregate`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicationSummary:
    """`aggregate`'s output (`design.md` Section 4.4; Requirement 9.4-9.8).

    Attributes:
        seed_results: all `len(seed_results)` `SeedResult`s, retained in
            full in every branch (Requirement 9.5, 9.8) -- no field here
            selects a subset.
        mean / median / std_dev: summary statistics of
            `seed_results[i].primary_estimate` across every seed whose
            estimate is not `None`. `None` if every seed's estimate is
            `None`.
        value_range: `(min, max)` of the same non-`None` estimates, or
            `None` if none exist.
        direction_agreement_count: how many seeds' `primary_estimate`
            agrees in direction with the aggregate direction (the sign of
            `mean`), via `evaluation.metrics.effect_direction_agreement`.
            `0` if `mean` is `None`.
        stability_criterion: `criterion.criterion_description`, reported
            verbatim alongside `stability_determination` (Requirement
            9.7).
        stability_determination: `"stable"` only if
            `direction_agreement_count >= criterion.min_agreeing_seeds`
            and (when `criterion.require_no_contradicting_sensitivity_variant`)
            no seed's `SensitivityStabilityResult` -- not tracked on
            `SeedResult` at this task's scope, so this condition is
            evaluated via the `sensitivity_overall_stability` sequence
            passed to `aggregate` directly -- is `"unstable"`.
            `"directionally_unstable"` otherwise.
    """

    seed_results: tuple[SeedResult, ...]
    mean: float | None
    median: float | None
    std_dev: float | None
    value_range: tuple[float, float] | None
    direction_agreement_count: int
    stability_criterion: str
    stability_determination: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_results": [result.to_dict() for result in self.seed_results],
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "value_range": list(self.value_range) if self.value_range is not None else None,
            "direction_agreement_count": self.direction_agreement_count,
            "stability_criterion": self.stability_criterion,
            "stability_determination": self.stability_determination,
        }


def record_seed_determinism(
    first: SeedResult, second: SeedResult, results_root: Path
) -> Any:
    """Verify two `run_seed` executions of the same seed are identical, and
    append the result to the readiness ledger's `determinism_verified` set
    (task 18.2; Requirement 18.7).

    "Artifact hashes match the first" (task 18.2's wording) is checked as
    the conjunction of `resolved_config_hash`, `split_hash`, and the full
    `checkpoint_hashes` tuple all agreeing between `first` and `second` --
    anything less would let two runs that trained different models still
    read as "the same run, verified."

    Args:
        first: the first execution's `SeedResult`.
        second: a second execution of `run_seed` at the identical seed
            (and, in production, an unchanged `ResolvedConfig`).
        results_root: the root `append_ledger_entry` writes the ledger
            under (the same `results_root` passed to both `run_seed`
            calls).

    Returns:
        The appended (or already-idempotently-present) `LedgerEntry`.

    Raises:
        ValueError: if `first.seed != second.seed`.
        NonDeterminismError: if any of the three hash comparisons
            disagrees -- two executions of the same seed produced
            different artifacts, which Requirement 18.7 forbids.
    """
    from infra.readiness import DETERMINISM_KIND, append_ledger_entry

    if first.seed != second.seed:
        raise ValueError(
            f"record_seed_determinism compares two executions of the SAME seed; "
            f"got seed={first.seed!r} and seed={second.seed!r}"
        )

    mismatches: list[str] = []
    if first.resolved_config_hash != second.resolved_config_hash:
        mismatches.append("resolved_config_hash")
    if first.split_hash != second.split_hash:
        mismatches.append("split_hash")
    if first.checkpoint_hashes != second.checkpoint_hashes:
        mismatches.append("checkpoint_hashes")

    if mismatches:
        raise NonDeterminismError(
            f"seed {first.seed}: two executions from an unchanged ResolvedConfig "
            f"produced different artifact hash(es) for: {', '.join(mismatches)}. "
            "This is the non-determinism Requirement 18.7 forbids."
        )

    replication_root = Path(results_root) / "replication"
    combined_hash = "|".join(
        (first.resolved_config_hash, first.split_hash, *(h for _, h in first.checkpoint_hashes))
    )
    return append_ledger_entry(
        replication_root, DETERMINISM_KIND, str(first.seed), combined_hash
    )


class NonDeterminismError(Exception):
    """Raised by `record_seed_determinism` when two executions of the same
    seed from an unchanged `ResolvedConfig` produced different artifact
    hashes (Requirement 18.7)."""


def write_replication_summary(summary: ReplicationSummary, results_root: Path) -> Path:
    """Write `aggregate`'s output to
    `results_root / "replication" / "replication_summary.json"`.

    A thin, separate function (not folded into `aggregate` itself) so
    `aggregate`'s own unit tests can exercise the pure aggregation logic
    without touching the filesystem.
    """
    path = Path(results_root) / "replication" / "replication_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True, default=str))
    return path


def aggregate(
    seed_results: Sequence[SeedResult],
    criterion: StabilityCriterion = CROSS_SEED_STABILITY_CRITERION,
    sensitivity_overall_stabilities: Sequence[str] | None = None,
) -> ReplicationSummary:
    """Combine every seed's result into one cross-seed summary
    (Requirement 9.4-9.8).

    No code path here selects a best or worst seed: the only input is
    the full `seed_results` tuple, and `ReplicationSummary.seed_results`
    retains every one of them unchanged, in every branch, including
    `"directionally_unstable"` (Requirement 9.5, 9.8).

    Args:
        seed_results: every seed's `SeedResult`, in any order. Not
            required to be exactly five -- a test may pass fewer; the
            production caller passes exactly the five frozen seeds'
            results, and `criterion.total_seeds` documents that
            expectation without this function enforcing it (a shorter
            sequence stays mechanically well-defined: fewer seeds to
            agree, same arithmetic).
        criterion: the frozen cross-seed stability criterion
            (`CROSS_SEED_STABILITY_CRITERION` by default; a test may pass
            a different one to exercise the aggregation logic at other
            thresholds -- see `StabilityCriterion`'s own docstring).
        sensitivity_overall_stabilities: each seed's
            `evaluation.robustness.SensitivityStabilityResult.overall_stability`
            string (`"stable"` / `"unstable"` / `"undetermined"`), in the
            same order as `seed_results`, or `None` if not supplied (in
            which case the sensitivity-contradiction half of the
            criterion is skipped and stability is determined from seed
            agreement alone -- a documented simplification, since
            `SeedResult` does not itself carry this field at this task's
            scope; see that dataclass's docstring).

    Returns:
        A `ReplicationSummary`.
    """
    from evaluation.metrics import effect_direction_agreement

    estimates = [r.primary_estimate for r in seed_results if r.primary_estimate is not None]

    if not estimates:
        mean = median = std_dev = None
        value_range = None
        direction_agreement_count = 0
    else:
        n = len(estimates)
        mean = sum(estimates) / n
        sorted_estimates = sorted(estimates)
        mid = n // 2
        median = (
            sorted_estimates[mid]
            if n % 2 == 1
            else (sorted_estimates[mid - 1] + sorted_estimates[mid]) / 2.0
        )
        variance = sum((x - mean) ** 2 for x in estimates) / n
        std_dev = variance**0.5
        value_range = (min(estimates), max(estimates))
        direction_agreement_count = sum(
            1
            for r in seed_results
            if r.primary_estimate is not None
            and effect_direction_agreement(r.primary_estimate, mean)
        )

    seeds_agree = (
        mean is not None and direction_agreement_count >= criterion.min_agreeing_seeds
    )

    sensitivity_contradicts = False
    if criterion.require_no_contradicting_sensitivity_variant and sensitivity_overall_stabilities:
        sensitivity_contradicts = any(
            status == "unstable" for status in sensitivity_overall_stabilities
        )

    if mean is None:
        # No seed produced a usable estimate at all -- there is no
        # aggregate direction to agree or disagree with, so this is
        # neither "stable" nor "directionally_unstable" (both presuppose
        # a direction exists); `design.md` Section 4.4's third literal.
        stability_determination = "undetermined"
    elif seeds_agree and not sensitivity_contradicts:
        stability_determination = "stable"
    else:
        stability_determination = "directionally_unstable"

    return ReplicationSummary(
        seed_results=tuple(seed_results),
        mean=mean,
        median=median,
        std_dev=std_dev,
        value_range=value_range,
        direction_agreement_count=direction_agreement_count,
        stability_criterion=criterion.criterion_description,
        stability_determination=stability_determination,
    )


# ---------------------------------------------------------------------------
# CLI entry point: five-seed replication on the real ACTG175 data
# ---------------------------------------------------------------------------

#: Repository root (matches `scripts/run_ablations.py`'s convention).
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Subdirectory a `--verify-determinism` second execution is written to,
#: under `--results-root`. Kept separate because `run_seed` refuses to
#: write into a non-empty per-seed directory.
DETERMINISM_RERUN_DIRNAME = "_determinism_rerun"


def build_parser() -> Any:
    import argparse

    from infra.readiness import REQUIRED_SEEDS
    from models.torch_training import SUPPORTED_DEVICES

    parser = argparse.ArgumentParser(
        description=(
            "Run the five-seed replication (task 18): the full development-stage "
            "pipeline (split, observationalization, both neural ensembles, the "
            "three witnesses, the release gate) once per seed on the real "
            "ACTG175 v3.2 data, then aggregate under the frozen cross-seed "
            "stability criterion and write replication_summary.json."
        )
    )
    parser.add_argument(
        "--target-trial-config",
        type=Path,
        default=_REPO_ROOT / "configs" / "target_trials" / "main.yaml",
        help="Target-trial-spec YAML (default: configs/target_trials/main.yaml).",
    )
    parser.add_argument(
        "--real-data-path",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "actg175_v3_2",
        help="Directory with the real raw ACTG175 v3.2 CSVs (default: data/raw/actg175_v3_2).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO_ROOT / "results",
        help="Root under which replication/ is written (default: results/).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(REQUIRED_SEEDS),
        help=f"Replication seeds (default: the five frozen seeds {list(REQUIRED_SEEDS)}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=sorted(SUPPORTED_DEVICES),
        help="Training device for both ensembles (default: cpu).",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help=(
            "Run every seed a second time into "
            f"<results-root>/{DETERMINISM_RERUN_DIRNAME}/ and record a determinism "
            "entry in the readiness ledger when the artifact hashes match "
            "(task 18.2). Raises NonDeterminismError on any mismatch. Doubles runtime."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load the real data, run `run_seed` for every seed, aggregate, and
    write `replication_summary.json`. Returns 0 on success, 1 on any error
    (message on stderr), matching the other `scripts/run_*.py` entry points.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import yaml

    from data.loaders.actg175 import Actg175DataNotFoundError, load_actg175
    from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
    from models.torch_training import resolve_device
    from protocols.schema import TargetTrialSpec

    args = build_parser().parse_args(argv)

    try:
        # Fail fast on an unavailable device, before any data is loaded.
        device = resolve_device(args.device)

        with args.target_trial_config.open("r", encoding="utf-8") as handle:
            spec = TargetTrialSpec.model_validate(yaml.safe_load(handle))

        tables = load_actg175(args.real_data_path)
        timelines, treatment_arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)
        print(
            f"loaded real ACTG175 data: {len(timelines)} arm-0/1 patients "
            f"(checksum {tables.checksum_sha256[:12]}...)"
        )

        def _run(seed: int, root: Path) -> SeedResult:
            return run_seed(
                seed,
                timelines,
                treatment_arms,
                spec,
                root,
                device=device,
                precomputed_endpoint_results=endpoint_results,
            )

        seed_results: list[SeedResult] = []
        for seed in args.seeds:
            print(f"seed {seed}: running full pipeline on {device}...", flush=True)
            first = _run(seed, args.results_root)
            released = first.release_decision.released
            print(
                f"seed {seed}: primary_estimate={first.primary_estimate} "
                f"released={released} witnesses={dict(first.witness_status)}",
                flush=True,
            )
            if args.verify_determinism:
                print(f"seed {seed}: determinism re-run...", flush=True)
                second = _run(seed, args.results_root / DETERMINISM_RERUN_DIRNAME)
                record_seed_determinism(first, second, args.results_root)
                print(f"seed {seed}: determinism verified (hashes match)", flush=True)
            seed_results.append(first)

        summary = aggregate(seed_results)
        summary_path = write_replication_summary(summary, args.results_root)
    except Actg175DataNotFoundError as exc:
        print(
            f"run_replication.py failed: real ACTG175 data not found: {exc}. "
            "See docs/data_access.md Section 2.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"run_replication.py failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"stability_determination: {summary.stability_determination} "
        f"(direction agreement {summary.direction_agreement_count}/{len(seed_results)})"
    )
    print(f"mean={summary.mean} median={summary.median} std_dev={summary.std_dev}")
    print(f"replication summary written to: {summary_path}")
    print("run_replication.py succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
