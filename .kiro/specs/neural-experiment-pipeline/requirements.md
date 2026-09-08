# Requirements Document

## Neural Experiment Pipeline

**Target:** ICLR 2027 main track (abstract 2026-09-18, paper 2026-09-25)
**Parent spec:** `.kiro/specs/causal-witness-pwm/` (requirements.md, design.md, decisions.md, tasks.md) — complete and implemented
**Scope:** the not-yet-implemented neural and experimental layer required to make the parent spec's central research contribution empirically testable

## Introduction

The parent spec `causal-witness-pwm` is implemented: real ACTG175 v3.2 ingestion, cohort construction, controlled observationalization, six conventional causal estimators, leakage audit, three witnesses with aggregation, overlap diagnostics, negative controls, privacy-safe export, sealed-reference persistence with token-gated access, content-hashed experiment identifiers, and release-gate suppression of unreleased estimates. This spec does not restate, re-derive, or re-specify any of that work.

This spec covers only the missing layer: production PyTorch models trained by real gradient descent, a variant factory that makes ablations swap models instead of re-fitting one model, executed sensitivity analyses, five-seed replication with prespecified stability criteria, ensemble uncertainty and selective-prediction metrics, an explicit cohort release gate, claim-level audit reports for every candidate estimate, a token-bound locked evaluation entry point, an extended sealed reference carrying risk-difference quantities, git provenance, a controlled semi-synthetic failure-mode benchmark, and conversion of the manuscript target to ICLR 2027.

Every modeling value already frozen in `causal-witness-pwm/decisions.md` (notably Sections 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25) is an input to this spec, read from that document. This spec introduces no competing value for anything frozen there. Where a new implementation choice is required that `decisions.md` does not fix, the choice is frozen in `decisions.md` (with a Section 26 amendment-log entry) before code depends on it.

### Scope boundary

Out of scope for this spec: any change to data ingestion, cohort construction, observationalization, the six conventional estimators, witness definitions, overlap rules, or export scanning — those exist and pass their tests. In scope only where this spec's list below names a specific gap.

**Two prerequisite amendments.** Requirement 8's stronger-observational-selection coefficients and Requirement 9's cross-seed stability criterion are values `decisions.md` does not yet fix. Both must be recorded in `decisions.md` with a Section 26 amendment-log entry before the first sensitivity run or replication run executes, per Requirement 18.2.

**Checklist reconciliation.** `configs/go_no_go_checklist.yaml` currently carries 11 items, all `false`. The design phase reconciles that count against this spec's readiness conditions, which add a neural-readiness gate (all five seeds, twelve ablation runs, and eight sensitivity variants executed deterministically from frozen configurations).

### Empirical honesty premise

This spec does not assume the intervention-aware model outperforms the baselines. That is the empirical question the pipeline exists to answer. The most recent real-data run failed the release gate (overlap support fail, causal witness fail, temporal uncertain, aggregate release false), and the pipeline must be able to report exactly that outcome, and the parent spec's frozen valid negative result (`decisions.md` Section 21), without alteration.

## Glossary

- **Patient_Dataset_Builder**: the module (`src/models/dataset.py`) that converts validated `PatientRecord` objects into model-ready tensors, masks, and targets.
- **Sequence_GRU**: the production PyTorch treatment-as-feature baseline (`src/models/torch_sequence_gru.py`) implementing `decisions.md` Section 14.2.
- **Intervention_Aware_Model**: the production PyTorch intervention-aware model (`src/models/torch_intervention_aware.py`) implementing `decisions.md` Section 14.3.
- **Training_Harness**: the module (`src/models/torch_training.py`) that runs gradient-descent training, lambda selection, early stopping, checkpointing, and deep-ensemble construction.
- **Model_Registry**: the existing model/experiment registry in `src/infra/registry.py`.
- **Dependency_Manifest**: the project dependency declaration in `pyproject.toml`.
- **Repository**: the git repository containing this project's source tree.
- **Reference_Guard**: the existing sealed-reference and test-split access guard in `src/infra/reference_guard.py`.
- **Token_Minter**: `scripts/check_go_no_go.py`, which validates the go/no-go checklist and mints a Locked_Evaluation_Token.
- **Export_Scanner**: `scripts/scan_exports.py`, the privacy-safe export scan.
- **Reporting_Component**: the reporting entry points (`scripts/run_real_data_reporting.py`, `scripts/run_paper.py`) that render tables, figures, and claim text from saved artifacts.
- **Neural_Experiment_Pipeline**: the complete set of components introduced by this spec.
- **Estimate_Runner**: any pipeline entry point that produces a candidate treatment-effect estimate (`scripts/run_causal_analysis.py`, Ablation_Runner, Sensitivity_Runner, Replication_Orchestrator, Locked_Evaluation_Runner).
- **Experiment_Variant**: the validated frozen dataclass describing one experimental configuration (model family, treatment representation, protocol channel, gates, constraints, enabled witnesses, causal adjustment).
- **Variant_Factory**: the single `build_model(variant, dimensions, seed)` entry point that constructs the model implied by an Experiment_Variant.
- **Ablation_Runner**: the orchestration entry point (`scripts/run_ablations.py`) that executes the ten frozen ablations of `decisions.md` Section 17.
- **Sensitivity_Runner**: `src/evaluation/robustness.py::SensitivityRobustness` together with its variant-application function.
- **Resolved_Config**: a fully materialized run configuration produced by applying a sensitivity config difference to a base configuration.
- **Sensitivity_Stability_Result**: the machine-readable record of executed sensitivity outcomes consumed by the Causal_Witness.
- **Replication_Orchestrator**: the component that runs the complete development-stage pipeline once per replication seed.
- **Replication_Aggregator**: the component that combines per-seed results into a single cross-seed summary.
- **Uncertainty_Evaluator**: the component computing ensemble-based factual, calibration, and selective-prediction metrics of `decisions.md` Section 18.
- **Cohort_Release_Gate**: the component evaluating the complete cohort-level release formula of `decisions.md` Section 16.
- **Audit_Report_Builder**: `src/witnesses/audit_report.py::build_audit_report` and its callers.
- **Locked_Evaluation_Runner**: the separate script (`scripts/run_locked_evaluation.py`) that evaluates frozen checkpoints on the test split under a consumed token.
- **Locked_Evaluation_Token**: the token minted by `scripts/check_go_no_go.py` that authorizes exactly one locked evaluation.
- **Sealed_Reference_Store**: `src/cohorts/sealed_reference.py`, the token-gated sealed randomized-reference artifact.
- **Provenance_Recorder**: `src/infra/manifest.py`'s manifest writer together with the git-commit resolution used to bind artifacts to source state.
- **Synthetic_Benchmark**: the controlled semi-synthetic data generator with known ground-truth counterfactual effects.
- **Manuscript_Package**: the ICLR 2027 LaTeX sources, style files, and build entry point.
- **Development_Split / Validation_Split / Test_Split**: the three patient-level splits frozen in `decisions.md` Section 10.
- **Frozen_Decisions**: `.kiro/specs/causal-witness-pwm/decisions.md`.

## Requirements

### Requirement 1: Tensor dataset construction

**User Story:** As a researcher, I want validated patient records converted into model-ready tensors, so that PyTorch models train on the same cohort the causal estimators use.

#### Acceptance Criteria

1. WHEN Patient_Dataset_Builder receives a non-empty list of N `PatientRecord` objects for one split, THE Patient_Dataset_Builder SHALL emit a baseline-covariate tensor with N rows and one column per baseline covariate in the frozen baseline-covariate ordering, a CD4 observation tensor and a CD8 observation tensor each with N rows and exactly 3 columns corresponding to landmark weeks 0, 20, and 96 in that order, one missingness-mask tensor per observation tensor with the same N-by-3 shape, a treatment-arm tensor with N rows, and a protocol-feature tensor with N rows and the protocol-channel width frozen in Frozen_Decisions Section 25, such that row index i of every emitted tensor refers to the same patient identifier.
2. WHEN Patient_Dataset_Builder builds survival targets, THE Patient_Dataset_Builder SHALL assign each patient exactly one integer discrete-hazard bin index in the inclusive range 0 to 22, under the bin definition frozen in Frozen_Decisions Section 25 (23 bins: 8-week-wide half-open bins covering elapsed weeks 0 through 176, plus one final open bin covering elapsed week 176 and above, so any event time at or beyond 176 elapsed weeks receives index 22), and SHALL emit a censoring mask whose entry is set for exactly those patients whose `endpoint_result.censoring_indicator` is set and whose censoring time falls before the start of the assigned bin.
3. WHEN Patient_Dataset_Builder computes elapsed time for any patient, THE Patient_Dataset_Builder SHALL obtain that value by calling the existing elapsed-week helper of `causal.baselines` rather than computing elapsed weeks independently, so that the elapsed weeks used for bin assignment equal the value that helper returns for the same `PatientRecord`.
4. WHEN Patient_Dataset_Builder fits any preprocessing transformation, THE Patient_Dataset_Builder SHALL compute the transformation's parameters using only Development_Split rows and only entries whose missingness mask is not set to missing, SHALL apply those same parameter values without recomputation to the Validation_Split and the Test_Split, and SHALL produce identical parameter values on repeated builds from the same Development_Split input.
5. IF a CD4 or CD8 measurement is absent for a landmark timepoint, THEN THE Patient_Dataset_Builder SHALL set that entry's missingness mask to missing, SHALL set that entry's observation-tensor value to a single fixed finite placeholder value used identically for every missing entry and every build, SHALL emit observation tensors containing no non-finite values, and SHALL give that entry zero weight in the state-loss contribution.
6. WHEN Patient_Dataset_Builder emits a tensor batch, THE Patient_Dataset_Builder SHALL record patient identifiers in ascending `patient_id` order and SHALL produce that same identifier sequence and the same tensor row ordering on repeated builds of the same split from the same input records.
7. THE Patient_Dataset_Builder SHALL expose, as integer-valued data readable by Variant_Factory without hard-coded widths, the baseline-covariate width, the state-channel width, the intervention-channel width, the protocol-channel width, the landmark count (3), and the survival-bin count (23), each equal to the corresponding trailing dimension of the tensors emitted per criterion 1 and the target range in criterion 2.
8. IF the received list of `PatientRecord` objects is empty or contains more than one record with the same `patient_id`, THEN THE Patient_Dataset_Builder SHALL raise an error indicating whether the input was empty or which identifier was duplicated, and SHALL emit no tensors for that split.
9. IF a CD4 or CD8 landmark column is absent for every patient in the Development_Split, THEN THE Patient_Dataset_Builder SHALL set that column's missingness mask to missing for every patient in the Development_Split, the Validation_Split, and the Test_Split, SHALL skip fitting a preprocessing transformation for that column, SHALL fill that column with the placeholder value of criterion 5, and SHALL complete the build without raising an error.
10. IF a patient's `endpoint_result.event_time` is absent, non-finite, or yields negative elapsed weeks, THEN THE Patient_Dataset_Builder SHALL raise an error identifying that patient and the reason the event time was rejected, and SHALL emit no tensors for that split rather than a partially populated batch.

### Requirement 2: Production sequence-GRU baseline

**User Story:** As a researcher, I want a real treatment-as-feature GRU baseline, so that the intervention-aware contrast measures representation and not numerical-gradient artifacts.

#### Acceptance Criteria

1. THE Sequence_GRU SHALL initialize the recurrent hidden state from the baseline-covariate tensor.
2. THE Sequence_GRU SHALL consume the treatment-arm value as an ordinary input feature at each landmark timestep, without a separate intervention encoder.
3. THE Sequence_GRU SHALL produce a CD4/CD8 state-prediction output at each landmark timestep and a discrete-hazard output over the bins frozen in Frozen_Decisions Section 25.
4. THE Sequence_GRU SHALL use the hidden width frozen in Frozen_Decisions Section 25.
5. WHEN Sequence_GRU computes gradients, THE Sequence_GRU SHALL use PyTorch automatic differentiation.
6. WHEN a Sequence_GRU forward pass runs twice on identical inputs in evaluation mode, THE Sequence_GRU SHALL return identical outputs.

### Requirement 3: Production intervention-aware model

**User Story:** As a researcher, I want the intervention-aware model to represent interventions explicitly and roll out counterfactuals, so that treatment-effect recovery is measurable per patient.

#### Acceptance Criteria

1. THE Intervention_Aware_Model SHALL implement separate state, intervention, and protocol encoders, a GRU state-transition module, treatment-specific hazard heads, and a CD4/CD8 state-prediction head, as structured in Frozen_Decisions Section 14.3.
2. THE Intervention_Aware_Model SHALL use the encoder channel widths and hidden width frozen in Frozen_Decisions Section 25.
3. THE Intervention_Aware_Model SHALL keep each encoder channel width fixed regardless of whether that channel carries informative values, so that capacity-matched ablations differ only in channel content.
4. WHEN Intervention_Aware_Model performs a counterfactual rollout, THE Intervention_Aware_Model SHALL produce state and hazard trajectories under treatment arm 0 and under treatment arm 1 for every evaluated patient.
5. WHEN Intervention_Aware_Model produces a 96-week outcome, THE Intervention_Aware_Model SHALL derive that outcome from the discrete-hazard sequence rather than from a separate unconstrained head.
6. WHEN Intervention_Aware_Model rolls out arm 0 and arm 1 for one patient, THE Intervention_Aware_Model SHALL hold every non-intervention input identical across the two rollouts.

### Requirement 4: Training harness

**User Story:** As a researcher, I want real gradient-descent training with validation-only selection, so that reported results come from trained models and leak no held-out information.

#### Acceptance Criteria

1. THE Training_Harness SHALL optimize the sum of the discrete-time survival negative log-likelihood, the masked CD4/CD8 Huber loss weighted by the state coefficient, and the treated-versus-comparator representation-balance MMD penalty weighted by the balance coefficient, as frozen in Frozen_Decisions Section 14.3.
2. WHEN Training_Harness computes the state loss, THE Training_Harness SHALL exclude masked-missing observations from the loss value and from the gradient.
3. WHEN Training_Harness selects the state and balance coefficients, THE Training_Harness SHALL evaluate the nine-combination grid frozen in Frozen_Decisions Section 14.3 and SHALL select the combination maximizing validation composite-endpoint concordance on the Validation_Split.
4. THE Training_Harness SHALL apply the optimizer, learning rate, batch policy, gradient-clipping norm, early-stopping patience, and maximum epoch count frozen in Frozen_Decisions Section 25.
5. WHEN Training_Harness selects a checkpoint, THE Training_Harness SHALL select the epoch minimizing validation loss on the Validation_Split and SHALL write the checkpoint in the plain-dictionary format frozen in Frozen_Decisions Section 25.
6. WHEN Training_Harness builds an ensemble, THE Training_Harness SHALL train five replicas whose sub-seeds are derived through `infra.seeding.derive_seed` from the run's top-level seed, as frozen in Frozen_Decisions Sections 15 and 25.
7. THE Training_Harness SHALL read model-selection, early-stopping, checkpoint-selection, and threshold-selection inputs exclusively from the Development_Split and Validation_Split.
8. IF any training or selection step requests the Sealed_Reference_Store or the Test_Split, THEN THE Training_Harness SHALL raise an error and SHALL terminate the run.
9. WHEN Training_Harness finishes a run, THE Training_Harness SHALL record the per-epoch loss curve, the selected coefficients, the selected epoch, the device, and the checkpoint hash in the run manifest.

### Requirement 5: Declared dependencies and device-agnostic execution

**User Story:** As a researcher, I want the environment declared and the code runnable on both CPU and GPU, so that local validation and remote A100 training produce the same computation.

#### Acceptance Criteria

1. THE Dependency_Manifest SHALL declare pinned versions of torch and numpy alongside the existing pinned dependencies.
2. THE Dependency_Manifest SHALL declare the torch version installed in the project environment, so that a fresh install reproduces the same tensor library version.
3. THE Training_Harness SHALL accept a `--device` flag defaulting to `cpu` and accepting `cuda`, as frozen in Frozen_Decisions Section 25.
4. WHEN a training script starts, THE Training_Harness SHALL seed `torch.manual_seed` and the CUDA seed from the same top-level seed used for every other seeded sub-stream, and SHALL enable deterministic algorithms where a deterministic kernel exists for the operations used.
5. WHEN the same seeded configuration runs on `cpu` and on `cuda`, THE Training_Harness SHALL produce loss curves and final weights that agree to within floating-point accumulation tolerance.
6. IF `cuda` is requested and no CUDA device is available, THEN THE Training_Harness SHALL report the unavailable device and SHALL terminate before training.
7. WHEN a run completes, THE Provenance_Recorder SHALL record the device used in that run's manifest.

### Requirement 6: Experiment variant specification and model factory

**User Story:** As a researcher, I want one validated variant description and one factory, so that each experimental arm provably builds a different model.

#### Acceptance Criteria

1. THE Experiment_Variant SHALL be a frozen, validated dataclass carrying model family, treatment representation, protocol channel, overlap gate, uncertainty gate, clinical constraints, enabled witnesses, and causal adjustment.
2. IF an Experiment_Variant field receives a value outside the declared allowed set, THEN THE Experiment_Variant SHALL raise a validation error at construction.
3. THE Variant_Factory SHALL expose exactly one entry point `build_model(variant, dimensions, seed)` that returns the model implied by the supplied Experiment_Variant.
4. WHEN Variant_Factory receives two Experiment_Variant values whose model-affecting fields differ, THE Variant_Factory SHALL return models that differ in architecture, input content, or both.
5. THE Variant_Factory SHALL construct models whose parameter initialization is determined by the supplied seed.
6. WHEN Variant_Factory registers a constructed variant, THE Variant_Factory SHALL record the variant fields and the resulting content-hashed experiment identifier through Model_Registry.

### Requirement 7: Genuine model-swapping ablations

**User Story:** As a reviewer, I want each frozen ablation to change the system under test, so that the ablation table reports real differences rather than one repeated number.

#### Acceptance Criteria

1. WHEN Ablation_Runner starts an ablation session, THE Ablation_Runner SHALL execute exactly twelve runs, consisting of the nine single-run entries of Frozen_Decisions Section 17 plus the three per-witness runs of the witness-removed entry, and SHALL construct each run's arm through Variant_Factory.
2. WHEN Ablation_Runner executes the no-treatment-input run, THE Ablation_Runner SHALL hold every treatment input element at zero at every landmark for every patient, and SHALL leave every non-treatment input element identical to the complete model's inputs for the same patient and landmark.
3. WHEN Ablation_Runner executes the treatment-as-ordinary-feature run, THE Ablation_Runner SHALL build that arm from Sequence_GRU with the treatment indicator carried inside the state input channel and with no separate intervention channel.
4. WHEN Ablation_Runner executes the no-protocol run, THE Ablation_Runner SHALL supply a protocol channel of the frozen protocol width (two elements, Frozen_Decisions Section 25) whose every element is zero at every landmark for every patient, and SHALL leave the intervention channel identical to the complete model's.
5. WHEN Ablation_Runner executes the placebo-intervention run, THE Ablation_Runner SHALL replace the intervention channel with one fixed random vector of the frozen intervention width (three elements, Frozen_Decisions Sections 14.3 and 25), drawn once from that run's seed-derived stream, held constant across every patient and every landmark within that seed, redrawn only when the seed changes, and SHALL record the drawn vector with the run's results so that a re-run at the same seed reproduces the same vector.
6. WHEN Ablation_Runner executes the no-overlap-gate run or the no-uncertainty-gate run, THE Ablation_Runner SHALL reuse the complete model's saved patient-level predictions without modification, SHALL perform no retraining, SHALL bypass only the named gate, and SHALL record the count of Validation_Split patients whose gate decision differs from the complete model's.
7. WHEN Ablation_Runner executes the no-clinical-constraints run, THE Ablation_Runner SHALL reuse the complete model's saved patient-level predictions without modification, SHALL disable the clinical constraints only, SHALL keep the overlap gate, the uncertainty gate, and all three witnesses active, and SHALL record the count of Validation_Split patients whose constraint outcome differs from the complete model's.
8. WHEN Ablation_Runner executes the witness-removed entry, THE Ablation_Runner SHALL expand it into exactly three runs that remove the temporal witness, the causal witness, and the clinical witness respectively while the other two witnesses remain active, SHALL reuse the complete model's saved patient-level predictions without modification in all three runs, SHALL record for each run the count of Validation_Split patients whose witness verdict differs from the complete model's, and SHALL represent the three runs as sub-runs of one frozen entry so that the frozen entry count remains exactly ten.
9. WHEN Ablation_Runner executes the predictive-accuracy-alone run, THE Ablation_Runner SHALL set the representation-balance penalty weight to zero, SHALL disable the causal adjustment, and SHALL leave the factual prediction objective unchanged from the complete model's frozen objective.
10. WHEN a run's Experiment_Variant differs from the complete model in model architecture or in input content, THE Ablation_Runner SHALL retrain that arm from initialization under the frozen training configuration of Frozen_Decisions Section 25, once for each of the five frozen seeds 101, 202, 303, 404, and 505, and SHALL reuse no complete-model checkpoint for that arm.
11. WHEN Ablation_Runner reports metrics for a run, THE Ablation_Runner SHALL compute the frozen primary factual metric of Frozen_Decisions Section 18 on Validation_Split patients only, SHALL exclude every Development_Split patient used for fitting, and SHALL report one metric value per frozen seed together with the mean across those five values.
12. WHEN Ablation_Runner writes results for a run, THE Ablation_Runner SHALL record the run identifier, the Experiment_Variant fields, the checkpoint hash for a retrained run or the reused-prediction source identifier together with the source's checkpoint hash for a reuse run, the evaluated split name, the seed, and the reported metric values.
13. IF, at the same seed, two runs whose Experiment_Variant differs in any model-affecting field produce patient-level predictions that agree for every Validation_Split patient within an absolute difference of 1e-9, or the six retrained runs yield fewer than six distinct primary-metric values, THEN THE Ablation_Runner SHALL report a variant-collision error naming the colliding runs and SHALL mark the ablation table invalid instead of publishing it.
14. WHEN Ablation_Runner executes the explicit-intervention-channel run, THE Ablation_Runner SHALL build that arm from Intervention_Aware_Model with an intervention channel of the frozen intervention width (three elements) and with the same frozen hidden width used by the treatment-as-ordinary-feature arm, so that the two arms differ only in intervention representation.
15. IF a reuse run's saved complete-model prediction source is absent, unreadable, or carries a checkpoint hash that does not match the complete model's recorded checkpoint hash, THEN THE Ablation_Runner SHALL report an error indicating the missing or mismatched prediction source, SHALL write no metrics for that run, and SHALL leave results already written for other runs unchanged.
16. IF a retrained run does not complete training under the frozen training configuration, THEN THE Ablation_Runner SHALL record that run as failed with an error indication naming the run and the failure cause, SHALL exclude that run from the ablation table, and SHALL mark the ablation table incomplete.

### Requirement 8: Executed sensitivity analyses

**User Story:** As a researcher, I want sensitivity variants actually executed, so that causal-witness stability reflects evidence instead of a hardcoded value.

#### Acceptance Criteria

1. THE Sensitivity_Runner SHALL expose `apply_variant(base_config, config_diff)` returning a Resolved_Config in which every field required to execute a run is materialized.
2. IF a config difference names a field absent from the base configuration, THEN THE Sensitivity_Runner SHALL raise a configuration error before execution.
3. THE Sensitivity_Runner SHALL execute all eight sensitivity variants frozen in Frozen_Decisions Section 20 and SHALL report each variant's estimate, interval, and diagnostic status.
4. WHEN Sensitivity_Runner executes the treatment-and-comparator-reversal variant, THE Sensitivity_Runner SHALL reverse the effect interpretation so that the reported direction refers to the original treatment contrast.
5. WHEN Sensitivity_Runner executes the increased-missingness variant, THE Sensitivity_Runner SHALL apply the additional missingness after splitting, using a seed derived from the run's top-level seed and the variant name.
6. WHEN Sensitivity_Runner executes the week-20-removal variant, THE Sensitivity_Runner SHALL set the week-20 observation mask to missing and SHALL retain the week-20 timestep in the sequence.
7. THE Sensitivity_Runner SHALL read the stronger-observational-selection coefficients from a frozen configuration recorded in Frozen_Decisions before any sensitivity run executes.
8. WHEN Sensitivity_Runner completes the frozen variant set, THE Sensitivity_Runner SHALL emit a machine-readable Sensitivity_Stability_Result carrying per-variant estimates, per-variant status, and an overall stability determination.
9. WHEN an Estimate_Runner evaluates the causal witness, THE Estimate_Runner SHALL supply the Sensitivity_Stability_Result produced by Sensitivity_Runner in place of the currently hardcoded stability value in `scripts/run_causal_analysis.py`.
10. IF a sensitivity variant fails to execute, THEN THE Sensitivity_Runner SHALL record that variant as unexecuted with the failure reason and SHALL mark overall stability as undetermined.

### Requirement 9: Five-seed replication and aggregation

**User Story:** As a reviewer, I want results replicated across the frozen seeds and aggregated without seed cherry-picking, so that reported effects are not single-run artifacts.

#### Acceptance Criteria

1. THE Replication_Orchestrator SHALL run the development-stage pipeline once for each seed frozen in Frozen_Decisions Section 25.
2. WHEN Replication_Orchestrator runs one seed, THE Replication_Orchestrator SHALL recreate the split and observationalization, train Sequence_GRU and Intervention_Aware_Model, build the five-member ensemble, save patient-level predictions, evaluate the witnesses, evaluate the Cohort_Release_Gate, and write a content-hashed result directory.
3. WHEN Replication_Orchestrator writes a per-seed result directory, THE Replication_Orchestrator SHALL record the seed, the Resolved_Config hash, the split hash, and every checkpoint hash for that seed.
4. THE Replication_Aggregator SHALL report per-seed estimates with intervals, the mean, median, standard deviation, and range across seeds, effect-direction agreement, witness status by seed, release-gate status by seed, sensitivity stability, and the factual and selective-prediction metrics by seed.
5. THE Replication_Aggregator SHALL treat every seed's result as reportable and SHALL exclude any best-seed or worst-seed selection from the aggregate.
6. THE Replication_Aggregator SHALL apply a stability criterion recorded in Frozen_Decisions before the first replication run executes, requiring effect-direction agreement for at least four of the five seeds together with no required sensitivity variant contradicting the aggregate direction.
7. WHEN Replication_Aggregator evaluates stability, THE Replication_Aggregator SHALL report the applied criterion alongside the determination.
8. IF fewer seeds agree in direction than the recorded criterion requires, THEN THE Replication_Aggregator SHALL report the result as directionally unstable and SHALL retain every per-seed value in the report.
9. WHEN Replication_Orchestrator reruns a seed from an unchanged Resolved_Config, THE Replication_Orchestrator SHALL produce the same content-hashed experiment identifier.

### Requirement 10: Ensemble uncertainty and selective prediction

**User Story:** As a reviewer, I want the uncertainty and abstention metrics computed, so that the claim that abstention improves reliability is checkable at matched coverage.

#### Acceptance Criteria

1. THE Uncertainty_Evaluator SHALL compute CD4 mean absolute error, CD8 mean absolute error, IPCW Brier score, calibration slope, and calibration intercept from ensemble predictions, as listed in Frozen_Decisions Section 18.
2. THE Uncertainty_Evaluator SHALL compute retained coverage, selective risk, the risk-coverage curve, area under the risk-coverage curve, incorrect-release rate, unnecessary-abstention rate, and error at matched coverage.
3. WHEN Uncertainty_Evaluator derives epistemic uncertainty, THE Uncertainty_Evaluator SHALL use the ensemble variance convention frozen in Frozen_Decisions Section 15.
4. WHEN Uncertainty_Evaluator fits calibration, THE Uncertainty_Evaluator SHALL fit on the Validation_Split only.
5. WHEN Uncertainty_Evaluator compares a gated configuration with an ungated configuration, THE Uncertainty_Evaluator SHALL report error at equal retained coverage for both configurations.
6. WHEN Uncertainty_Evaluator determines the patient-level retention condition, THE Uncertainty_Evaluator SHALL apply the condition frozen in Frozen_Decisions Section 16, including the validation-derived uncertainty percentile threshold.
7. IF the supported population is empty for a configuration, THEN THE Uncertainty_Evaluator SHALL report every selective metric as undefined with the empty-population reason and SHALL continue evaluating the remaining configurations.

### Requirement 11: Explicit cohort release gate

**User Story:** As a reviewer, I want the release decision computed by the full frozen formula, so that witness aggregation alone cannot authorize a claim.

#### Acceptance Criteria

1. THE Cohort_Release_Gate SHALL compute the release decision as the conjunction of overlap validity, balance validity, uncertainty validity, temporal-witness pass, causal-witness pass, and clinical-witness pass, exactly as frozen in Frozen_Decisions Section 16.
2. WHEN Cohort_Release_Gate returns a decision, THE Cohort_Release_Gate SHALL return the individual status of each of the six conditions.
3. WHEN a causal-witness result is uncertain, THE Cohort_Release_Gate SHALL treat that result as blocking release, as frozen in Frozen_Decisions Section 16.
4. WHEN an Estimate_Runner reports a release decision, THE Estimate_Runner SHALL report the Cohort_Release_Gate decision rather than the witness-aggregation result alone.
5. WHEN Cohort_Release_Gate blocks release, THE Cohort_Release_Gate SHALL enumerate every failing condition as a machine-readable reason list.
6. WHERE the two-of-three witness policy is reported, THE Reporting_Component SHALL label that policy as exploratory and SHALL retain the all-witnesses policy as the primary gate.

### Requirement 12: Claim-level audit reports

**User Story:** As a reviewer, I want an audit record for every candidate estimate including suppressed ones, so that suppression is visible and traceable.

#### Acceptance Criteria

1. WHEN an Estimate_Runner produces a candidate estimate, THE Estimate_Runner SHALL call Audit_Report_Builder for that estimate.
2. THE Estimate_Runner SHALL call Audit_Report_Builder for suppressed candidate estimates as well as released candidate estimates.
3. THE Audit_Report_Builder SHALL emit a report carrying the release decision, the release reason list, an `estimate_status` field valued `released` or `suppressed`, the dataset checksum, the split hash, the Resolved_Config hash, every checkpoint hash, the seed, the sensitivity-variant name, and the supported and abstained population counts.
4. WHERE an estimate originates from a locked evaluation, THE Audit_Report_Builder SHALL additionally carry the locked evaluation identifier and the reference-access timestamp.
5. WHEN the Cohort_Release_Gate blocks an estimate, THE Audit_Report_Builder SHALL write a public artifact carrying `estimate_status` set to `suppressed`, a null estimate value, and the release reason list.
6. WHERE a protected internal artifact is written, THE Estimate_Runner SHALL retain the numeric estimate in the protected artifact only.
7. WHEN audit artifacts are written for export, THE Export_Scanner SHALL report every public audit artifact as export-safe.
8. IF a required audit field is unavailable, THEN THE Audit_Report_Builder SHALL raise an error identifying the missing field rather than emitting a partial report.

### Requirement 13: Token-bound locked evaluation

**User Story:** As a researcher, I want a separate locked-evaluation entry point that cannot train, so that the test split and sealed reference are opened exactly once under verified provenance.

#### Acceptance Criteria

1. THE Locked_Evaluation_Runner SHALL be an entry point distinct from every training entry point that loads frozen checkpoints in read-only mode, exposes no option to fit, tune, or select any parameter or threshold, and leaves every loaded checkpoint's hash identical before and after the run.
2. IF the Locked_Evaluation_Runner receives an invocation option or reaches a code path that would fit, update, tune, or select any parameter or threshold, THEN THE Locked_Evaluation_Runner SHALL raise an error naming the rejected option or operation, SHALL exit with a non-zero status, and SHALL terminate before opening the test split, the sealed reference, or any result artifact.
3. THE Locked_Evaluation_Token SHALL bind, as immutable fields, the Resolved_Config hash, the dataset checksum, the split hash, one hash per checkpoint, the git commit identifier, the statistical-plan version, and an evaluation identifier that is unique across all previously minted tokens.
4. IF any field required by criterion 3 is absent, empty, or duplicated when minting is attempted, THEN THE Token_Minter SHALL decline to mint a Locked_Evaluation_Token, SHALL name every offending field, SHALL exit with a non-zero status, and SHALL write no token artifact.
5. WHEN Locked_Evaluation_Runner starts, THE Locked_Evaluation_Runner SHALL recompute and compare every hash bound in the token against the artifacts present before opening the test split or the sealed reference, and SHALL terminate with a non-zero status on the first mismatch, naming the mismatched field and writing no result artifact.
6. IF a submitted evaluation identifier or token is already recorded as consumed, THEN THE Locked_Evaluation_Runner SHALL refuse the run before opening the test split or the sealed reference, SHALL exit with a non-zero status, and SHALL leave every existing artifact and consumption record byte-identical.
7. WHEN token verification succeeds, THE Locked_Evaluation_Runner SHALL evaluate the saved test assignment, SHALL unseal the internal randomized reference through Sealed_Reference_Store using the verified token, and SHALL compute treatment-effect recovery over that identical set of test patients.
8. WHEN treatment-effect recovery is computed, THE Locked_Evaluation_Runner SHALL run the paired bootstrap frozen in Frozen_Decisions Section 19 using exactly 1000 patient-level replicates over the identical test patients and SHALL apply Holm correction across exactly the three primary comparisons (baseline GRU vs intervention-aware; intervention-aware vs IPCW-AIPW; complete gated vs ungated) reporting 95% confidence intervals.
9. WHEN the paired bootstrap and Holm correction complete, THE Locked_Evaluation_Runner SHALL evaluate the witnesses and the Cohort_Release_Gate over the identical test patients and SHALL record each witness outcome and the gate outcome as pass or fail.
10. WHEN every step in criteria 7 through 9 completes, THE Locked_Evaluation_Runner SHALL record the token and evaluation identifier as consumed and write the result artifacts as one indivisible commit, and SHALL write result artifacts only to paths that contain no existing artifact, refusing with a non-zero status and changing nothing if any target path is already occupied.
11. IF any step after successful token verification fails, THEN THE Locked_Evaluation_Runner SHALL discard every partial output, SHALL leave the token, the evaluation identifier, and all pre-existing artifacts byte-identical, SHALL report an error naming the failing step, and SHALL exit with a non-zero status.
12. WHEN `scripts/run_paper.py` is invoked with the locked-evaluation stage, THE Reporting_Component SHALL delegate to Locked_Evaluation_Runner, SHALL propagate its exit status unchanged, and SHALL emit no message stating that the go/no-go script is absent.
13. IF the go/no-go checklist contains an item that is unresolved, absent, or non-boolean, THEN THE Token_Minter SHALL decline to mint a Locked_Evaluation_Token, SHALL name every such item, SHALL exit with a non-zero status, and SHALL write no token artifact.

### Requirement 14: Extended sealed reference

**User Story:** As a researcher, I want the sealed reference to carry risk-scale quantities, so that neural risk differences are compared against a matching randomized quantity.

#### Acceptance Criteria

1. THE Sealed_Reference_Store SHALL carry the randomized Cox log-hazard-ratio, the randomized 96-week risk for each arm, and the randomized 96-week risk difference.
2. WHEN the sealed reference is computed, THE Sealed_Reference_Store SHALL compute all three quantities on the complete eligible arm-0/arm-1 cohort before splitting and before observationalization, per the frozen order of operations in Frozen_Decisions Section 9.
3. WHEN Locked_Evaluation_Runner compares a model estimate with a sealed-reference quantity, THE Locked_Evaluation_Runner SHALL compare quantities on the same scale.
4. IF a requested comparison pairs a risk-difference estimate with a log-hazard-ratio reference, THEN THE Locked_Evaluation_Runner SHALL raise a scale-mismatch error.
5. WHEN Sealed_Reference_Store is read, THE Sealed_Reference_Store SHALL require a valid Locked_Evaluation_Token and SHALL record the reference-access timestamp.
6. WHEN the extended sealed reference is written, THE Sealed_Reference_Store SHALL record the sealed-artifact format version in that artifact.
7. IF a loaded sealed artifact carries a format version older than the extended version, THEN THE Sealed_Reference_Store SHALL raise a format-version error naming the documented re-seal step.

### Requirement 15: Git provenance

**User Story:** As a researcher, I want artifacts bound to a verifiable commit, so that a locked evaluation is reproducible from a known source state.

#### Acceptance Criteria

1. THE Repository SHALL track this project's source tree at the project root and SHALL exclude raw patient data and generated artifacts from tracked content, per the existing ignore rules.
2. WHEN Provenance_Recorder writes a manifest, THE Provenance_Recorder SHALL record the git commit identifier and the working-tree cleanliness state.
3. IF the working tree contains uncommitted changes when a Locked_Evaluation_Token is minted, THEN THE Token_Minter SHALL decline to mint the token and SHALL report the uncommitted paths.
4. WHEN a checkpoint or result directory is written, THE Provenance_Recorder SHALL bind that artifact to the recorded commit identifier.
5. IF the git commit identifier is unavailable, THEN THE Provenance_Recorder SHALL record the unavailable state explicitly and THE Token_Minter SHALL decline to mint a token.

### Requirement 16: Controlled semi-synthetic benchmark

**User Story:** As a reviewer, I want a controlled benchmark with known counterfactual effects, so that the pipeline's failure modes are demonstrable beyond one dataset.

#### Acceptance Criteria

1. THE Synthetic_Benchmark SHALL generate patient cohorts with a known ground-truth counterfactual treatment effect.
2. THE Synthetic_Benchmark SHALL generate configurable failure-mode scenarios covering limited overlap, unmeasured confounding, and increased missingness.
3. WHEN Synthetic_Benchmark generates a cohort, THE Synthetic_Benchmark SHALL emit records satisfying the same data contract the real cohort satisfies, so that the identical pipeline evaluates both.
4. WHEN Synthetic_Benchmark runs a scenario with a fixed seed, THE Synthetic_Benchmark SHALL reproduce identical cohorts.
5. WHEN benchmark results are reported, THE Reporting_Component SHALL label synthetic results as failure-mode diagnostics and SHALL report the real-data result as the primary empirical result.
6. WHEN a scenario's ground-truth effect is known, THE Reporting_Component SHALL report the recovery error against that ground truth and the release decision the gate produced for that scenario.

### Requirement 17: Primary experimental comparisons

**User Story:** As a reviewer, I want the five headline comparisons computed under one protocol, so that the paper's claims map to specific tests.

#### Acceptance Criteria

1. THE Reporting_Component SHALL report the comparison of the treatment-as-feature Sequence_GRU with the Intervention_Aware_Model.
2. THE Reporting_Component SHALL report the comparison of the Intervention_Aware_Model with the IPCW-AIPW estimator.
3. THE Reporting_Component SHALL report the comparison of the complete gated configuration with the identically-predicted ungated configuration.
4. THE Reporting_Component SHALL report the comparison of the intervention encoder with the capacity-matched placebo channel.
5. THE Reporting_Component SHALL report the comparison of the complete model with the no-protocol variant and with the no-treatment-input variant.
6. WHEN the three primary model comparisons of Frozen_Decisions Section 19 are reported, THE Reporting_Component SHALL apply paired bootstrap comparison on identical patients and the Holm correction frozen in that section.
7. WHEN any comparison is reported, THE Reporting_Component SHALL report the evaluated split, the seed set, and the interval alongside the point difference.

### Requirement 18: Frozen-decision fidelity and sealed-split isolation

**User Story:** As a reviewer, I want every frozen value read from the decisions document and the test split kept closed until readiness holds, so that the locked evaluation remains a single honest test.

#### Acceptance Criteria

1. THE Neural_Experiment_Pipeline SHALL read every value frozen in Frozen_Decisions from that document and SHALL introduce no alternative value for a frozen field.
2. WHEN a component of the Neural_Experiment_Pipeline requires a value that Frozen_Decisions does not fix, THE Neural_Experiment_Pipeline SHALL depend on that value only after the value is recorded in Frozen_Decisions with a Section 26 amendment-log entry.
3. WHILE any development-stage seed, ablation arm, or sensitivity variant remains unexecuted from its frozen configuration, THE Neural_Experiment_Pipeline SHALL keep the Test_Split closed and the Sealed_Reference_Store sealed.
4. WHILE any go/no-go checklist item remains unresolved, THE Token_Minter SHALL decline to mint a Locked_Evaluation_Token.
5. THE Training_Harness SHALL derive model selection, early stopping, checkpoint selection, and threshold selection from Development_Split and Validation_Split values only.
6. WHEN a development-stage component requests Test_Split or Sealed_Reference_Store data, THE Reference_Guard SHALL raise an access error and THE Neural_Experiment_Pipeline SHALL terminate that run.
7. WHEN the Neural_Experiment_Pipeline executes twice from an unchanged Resolved_Config, THE Neural_Experiment_Pipeline SHALL produce identical results, so that readiness is demonstrated deterministically.

### Requirement 19: Honest reporting of negative and blocked results

**User Story:** As a reviewer, I want negative and gate-blocked outcomes reported as findings, so that the abstention contribution is credible.

#### Acceptance Criteria

1. WHEN the Cohort_Release_Gate blocks the primary estimate, THE Reporting_Component SHALL report the blocked outcome, every failing condition, and the supported and abstained population counts.
2. WHEN factual performance improves without improving treatment-effect recovery, THE Reporting_Component SHALL report that dissociation as the frozen valid negative result of Frozen_Decisions Section 21.
3. THE Reporting_Component SHALL report every executed ablation arm, sensitivity variant, and seed, including arms whose results contradict the intervention-aware model's advantage.
4. THE Reporting_Component SHALL restrict claims to the permitted set of Frozen_Decisions Section 22 and SHALL exclude the prohibited claims of that section.
5. IF a reported estimate lacks a passing release decision, THEN THE Reporting_Component SHALL exclude that estimate's numeric value from the public artifact and SHALL retain the suppression reasons.

### Requirement 20: ICLR 2027 manuscript package

**User Story:** As an author, I want the manuscript targeted at ICLR 2027 under its formatting rules, so that the submission is compliant.

#### Acceptance Criteria

1. THE Manuscript_Package SHALL use the official ICLR 2027 style files.
2. THE Manuscript_Package SHALL limit main text to nine pages, with references and appendices outside that limit.
3. THE Manuscript_Package SHALL exclude author names, affiliations, acknowledgements, and identifying repository links from the submission build, satisfying double-blind review.
4. THE Manuscript_Package SHALL include a statement describing the use of AI assistance.
5. THE Dependency_Manifest SHALL name ICLR 2027 main track as the submission venue, and THE Manuscript_Package SHALL record the same venue in place of the parent spec's NeurIPS 2026 WMHS venue line.
6. WHEN the manuscript is built, THE Manuscript_Package SHALL draw every reported number from saved result artifacts through the existing one-command reproduction entry point.
7. WHERE a figure or table reports a suppressed estimate, THE Manuscript_Package SHALL present the suppression status in place of the numeric value.
