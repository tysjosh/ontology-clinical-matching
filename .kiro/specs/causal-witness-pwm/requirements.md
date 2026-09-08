# Requirements Document

## Causal Witness Patient World Model

**Target:** Nine-page WMHS @ NeurIPS 2026 submission  
**Scope:** Submission-critical empirical system  
**Status:** Implementation-ready after dataset, treatment, comparator, endpoint, and historical reference are filled in  
**Last updated:** September 6, 2026

## 1. Introduction

The Causal Witness Patient World Model (PWM) is a research system for estimating a prespecified cohort-level treatment effect from observational longitudinal clinical data and comparing that estimate with an independently reported historical randomized-trial effect.

The system represents patient state, interventions, target-trial protocol, uncertainty, and evidence provenance separately. It evaluates factual prediction, causal diagnostics, protocol consistency, and selective abstention without treating any one of them as proof of clinical validity.

The submission-critical study is limited to:

- one versioned clinical dataset;
- one treatment and comparator;
- one endpoint type;
- one prespecified target trial;
- one aligned historical randomized reference effect;
- one conventional non-neural outcome model;
- one sequence-model baseline;
- one intervention-aware model;
- one primary causal baseline, augmented inverse-probability weighting;
- one primary uncertainty method;
- propensity-based overlap diagnostics;
- deterministic protocol and clinical-constraint checks;
- bootstrap uncertainty for the cohort-level treatment effect;
- prespecified ablations and robustness checks.

The system is a clinical-research prototype. It is not an autonomous treatment recommender and must not be described as replacing randomized trials.

## 2. Out of scope

The following are deferred unless a tested implementation already exists:

- general-purpose natural-language protocol parsing;
- support for every endpoint family;
- multiple historical trials;
- multiple treatment decisions;
- large clinical foundation-model pretraining;
- imaging or free-text encoders not already available;
- mechanism-of-action or pathway encoders;
- dynamic treatment regimes not required by the chosen trial;
- autonomous clinical or protocol-design agents;
- personalized treatment recommendations;
- regulatory decision support;
- prospective clinical validation;
- synthetic replacement of randomized control arms.

## 3. Non-negotiable execution order

1. Freeze the causal question and target-trial protocol.
2. Freeze the causal model, adjustment set, and estimand-alignment criteria.
3. Construct and audit the leakage-safe cohort.
4. Implement conventional predictive and causal baselines.
5. Implement the intervention-aware model.
6. Fit uncertainty and abstention thresholds using development data only.
7. Freeze the analysis configuration.
8. Run the locked test evaluation.
9. Compare the frozen estimate with the historical randomized reference.
10. Generate the paper artifacts and audit package.

The historical randomized result must not influence model training, threshold selection, hyperparameter selection, ablation selection, or model selection.

## 4. Primary research questions

### RQ1 — Predictive–causal relationship

Does factual patient-outcome or trajectory-prediction performance predict recovery of the aligned historical treatment effect?

### RQ2 — Intervention representation

Does representing treatment as a first-class intervention improve cohort-level treatment-effect estimation relative to treating treatment as an ordinary feature or token?

### RQ3 — Protocol constraints

Does executable target-trial conditioning reduce eligibility, time-zero, treatment-regime, temporal, and endpoint violations?

### RQ4 — Selective estimation

Do overlap- and uncertainty-aware support restrictions reduce treatment-effect discrepancy within a clearly defined supported target population?

### RQ5 — Witness validation

Do temporal, causal-diagnostic, and clinical-constraint witnesses detect failures that factual predictive performance alone misses?

## 5. Primary endpoints

The paper has three confirmatory endpoints.

### 5.1 Treatment-effect discrepancy

\[
D_\tau=\hat\tau_{model}-\hat\tau_{RCT}
\]

The absolute discrepancy may also be reported:

\[
E_\tau=|D_\tau|
\]

### 5.2 Standardized treatment-effect discrepancy

\[
Z_\tau=
\frac{\hat\tau_{model}-\hat\tau_{RCT}}
{\sqrt{SE_{model}^2+SE_{RCT}^2}}
\]

This comparison is valid only after verifying estimand alignment and reasonable independence of the two estimates.

### 5.3 Protocol-violation rate

\[
PVR=
\frac{\#\text{invalid evaluated trajectories or records}}
{\#\text{evaluated trajectories or records}}
\]

The denominator and unit of evaluation must be declared before test evaluation.

### 5.4 Selective treatment-effect discrepancy

Treatment-effect discrepancy is recomputed within prespecified supported populations at defined retained-coverage levels. The supported-population estimand must be reported separately from the original-population estimand.

## 6. Secondary and diagnostic endpoints

Secondary measures may include:

- factual AUROC and AUPRC;
- Brier score;
- calibration slope and intercept;
- factual-outcome conformal coverage and interval width;
- propensity overlap;
- covariate balance;
- effective sample size;
- negative-control results;
- selective risk and retained coverage;
- incorrect-release and unnecessary-abstention rates;
- temporal-consistency violations;
- subgroup estimates with sample sizes;
- descriptive treatment-effect direction agreement;
- descriptive confidence-interval comparison.

With one historical trial, effect-direction agreement is one descriptive observation, not a population-level accuracy metric. Interval overlap must not be described as a formal equivalence test. Relative effect error must not be used when the reference effect is zero or close to zero.

## 7. Requirements

### Requirement 1 — Reproducible experiment foundation

**User story:** As a researcher, I want every experiment to be reproducible and traceable so that every paper result can be regenerated and audited.

#### Acceptance criteria

1. The repository SHALL provide a Python package structure, locked dependency management, configuration loading, structured logging, seed handling, tests, linting, and results-directory conventions.
2. `pytest` SHALL complete without failures before a paper result can be marked final.
3. The test suite SHALL include unit, integration, and end-to-end smoke tests.
4. A smoke experiment SHALL run end-to-end from one documented command.
5. Every run SHALL save configuration, seed, timestamp, source commit, and whether the working tree was clean or dirty.
6. Every run SHALL record operations known to be nondeterministic.
7. The experiment registry SHALL record experiment ID, dataset version, trial specification version, causal-model version, model, seed, split, intervention representation, protocol constraints, overlap gate, uncertainty gate, and witness policy.
8. Duplicate experiment identities SHALL not silently overwrite one another.
9. Every table entry SHALL resolve to exactly one experiment ID or a declared aggregation of experiment IDs.
10. Every completed run SHALL save a privacy-safe artifact manifest containing input-version hashes, cohort definition, feature list, treatment and outcome definitions, model configuration, checkpoints, predictions, metrics, and environment versions.
11. Public manifests SHALL not expose protected data, direct identifiers, quasi-identifiers, secrets, or unauthorized filesystem paths.

### Requirement 2 — Clinical-data privacy and security

**User story:** As a data steward, I want clinical information confined to approved locations and excluded from unsafe artifacts so that the research pipeline complies with applicable data-use and privacy obligations.

#### Acceptance criteria

1. Before data access, the project SHALL document the approved storage location, access rules, data-use agreement, and permitted derived artifacts.
2. Direct identifiers SHALL not enter logs, manifests, filenames, checkpoints, reports, or version control.
3. Raw free text SHALL not enter debugging logs or public artifacts.
4. Secrets and credentials SHALL be loaded through an approved secret mechanism and excluded from version control.
5. Public tables and figures SHALL use aggregate results and enforce the dataset's minimum-cell-size policy.
6. Patient-level predictions SHALL remain in approved protected storage.
7. Exported artifacts SHALL pass an automated identifier and secret scan.
8. Data hashing SHALL use stable internal identifiers or approved dataset-version files without exposing patient identifiers.
9. A privacy review SHALL be completed before artifact release.

### Requirement 3 — Target-trial and causal specification

**User story:** As a causal researcher, I want the causal question, assumptions, and target trial fixed before analysis so that the estimand is unambiguous and data-dependent choices are minimized.

#### Acceptance criteria

1. The target-trial schema SHALL require population, eligibility, intervention, comparator, assignment procedure being emulated, time zero, follow-up, endpoint, censoring, causal contrast, estimand, and analysis method.
2. Invalid or incomplete specifications SHALL fail with specific errors.
3. A complete specification SHALL be versioned and immutable after analysis lock except through a documented amendment.
4. The system SHALL store a causal diagram or equivalent structured causal model.
5. The causal specification SHALL identify treatment, outcome, baseline confounders, time-varying confounders, mediators, colliders, instruments, censoring variables, and prohibited post-treatment variables where applicable.
6. The specification SHALL document consistency, conditional exchangeability, positivity, no-interference, missing-data, and censoring assumptions.
7. The adjustment set SHALL be frozen before test evaluation.
8. Eligibility logic SHALL support only the operators required by the chosen trial, including AND, OR, NOT, thresholds, temporal windows, and explicit missing-value behavior.
9. Unit tests SHALL cover negation, logical scope, threshold boundaries, temporal eligibility, and missing values.
10. Time-zero validation SHALL enforce that eligibility and baseline features precede or coincide with treatment assignment as specified.
11. Post-treatment information SHALL not enter baseline features.
12. The system SHALL implement and test the endpoint type required by the selected trial.
13. The protocol checker SHALL return structured violations containing record or patient reference, rule, timestamp, and evidence.

### Requirement 4 — Estimand alignment with the historical trial

**User story:** As a reviewer, I want evidence that the observational emulation and randomized reference estimate the same causal quantity so that treatment-effect discrepancy is interpretable.

#### Acceptance criteria

1. The system SHALL generate an estimand-alignment report comparing observational and randomized definitions of eligibility, treatment versions, comparator, time zero, follow-up, endpoint, effect scale, adherence strategy, censoring strategy, and target population.
2. Every difference SHALL be classified as aligned, approximately aligned, correctable, or non-alignable.
3. If populations differ and sufficient variables are available, the system SHALL implement an explicit transport or standardization analysis.
4. If material differences cannot be corrected, the historical comparison SHALL be labelled descriptive rather than validation of effect recovery.
5. The system SHALL verify whether the observational data include participants or records derived from the reference trial.
6. Any lack of independence SHALL be disclosed and incorporated into uncertainty interpretation.
7. A clinically meaningful discrepancy or equivalence margin, \(\delta\), SHALL be prespecified when defensible.

### Requirement 5 — Clinical data pipeline

**User story:** As a researcher, I want a leakage-safe, provenance-bearing pipeline from raw records to patient timelines so that the analysis cohort and features are trustworthy.

#### Acceptance criteria

1. The pipeline SHALL require a versioned dataset contract specifying dataset version, population, treatment, comparator, endpoint, study period, available modalities, and reference-effect source.
2. The pipeline SHALL refuse to run without the dataset contract.
3. Loaders SHALL be implemented only for modalities available and required for the selected study.
4. Loader outputs SHALL be typed and schema validated.
5. Every clinical event SHALL include patient key, encounter key where available, normalized concept, value, unit, modality, observation time, validity interval where available, source, missingness state, and provenance.
6. Event ordering SHALL be reproducible.
7. Concept and unit normalization SHALL be deterministic and versioned.
8. Unresolved concepts SHALL be flagged instead of silently discarded.
9. Patient timelines SHALL contain baseline covariates, timestamped observations, treatments, endpoints, censoring events, and missingness masks required by the trial.
10. Leakage auditing SHALL detect post-treatment baseline variables, outcome-derived features, duplicated patients across splits, future observations, and known indirect treatment or outcome encodings.
11. Synthetic leakage fixtures SHALL trigger pipeline failures.
12. Missing-data handling SHALL state its assumptions and SHALL be fitted using development or training folds only.
13. The pipeline SHALL support missingness indicators and, if used, fold-safe multiple imputation.
14. The data-quality report SHALL include cohort size, missingness, treatment prevalence, endpoint prevalence, follow-up, event frequency, subgroup counts, censoring, and treatment switching.

### Requirement 6 — Analysis splits and cross-fitting

**User story:** As a methodologist, I want sample splitting and cross-fitting defined consistently so that nuisance estimation, model selection, causal estimation, and external validation remain separated.

#### Acceptance criteria

1. The analysis plan SHALL define the population on which the primary ATE is estimated.
2. Patient-level separation SHALL be enforced wherever predictive train, validation, and test splits are used.
3. The system SHALL define whether the primary ATE uses a locked test cohort or cross-fitted predictions over the full eligible emulation cohort.
4. Nuisance models SHALL use cross-fitting and SHALL not predict observations used for their own fitting.
5. Calibration and abstention thresholds SHALL be fitted using development or validation predictions only.
6. The test cohort and historical reference SHALL remain unavailable to model and threshold selection.
7. Temporal-shift evaluation, if used, SHALL preserve chronological ordering and patient-level separation.
8. Bootstrap resampling SHALL occur at the patient level or at a higher clustering level when the independence unit requires it.

### Requirement 7 — Cohort and causal-design engine

**User story:** As a causal researcher, I want a target-trial cohort with overlap diagnostics and credible baseline estimators so that the neural model is compared against defensible causal methods.

#### Acceptance criteria

1. Cohort construction SHALL apply eligibility, index time, intervention strategy, comparator, follow-up, censoring, and endpoint extraction from configuration.
2. Repeated construction from the same inputs SHALL reproduce cohort membership and counts.
3. The system SHALL generate a cohort-flow table.
4. Propensity estimation SHALL implement logistic regression and MAY include one gradient-boosted sensitivity model.
5. Nuisance estimation SHALL use the cross-fitting policy in Requirement 6.
6. Overlap diagnostics SHALL include arm-specific propensity distributions, common support, extreme weights, effective sample size, and standardized mean differences.
7. Each patient SHALL receive a support-region flag.
8. The primary conventional causal baseline SHALL be augmented inverse-probability weighting.
9. Outcome regression and inverse-probability weighting SHALL be reported as component baselines when feasible.
10. Weight clipping and truncation rules SHALL be prespecified and included in sensitivity analysis.
11. If treatment or censoring varies over time, the system SHALL use an appropriate longitudinal estimator or explicitly narrow the trial to a baseline-treatment estimand.
12. Negative-control analyses SHALL be implemented only when scientifically defensible controls are available.

### Requirement 8 — Predictive baselines

**User story:** As a researcher, I want conventional factual predictive baselines so that predictive accuracy can be separated from causal validity.

#### Acceptance criteria

1. A non-neural baseline SHALL be selected to match the endpoint type, such as logistic regression, Cox regression, survival forest, or gradient boosting.
2. A sequence baseline SHALL use one frozen architecture and include treatment as an ordinary feature or token.
3. Training SHALL support early stopping, checkpointing, class weighting where appropriate, censoring-aware loss where required, and recorded stochastic seeds.
4. Factual metrics SHALL be prespecified for the endpoint type.
5. Factual metrics SHALL not be described as measures of causal validity.
6. Model-selection criteria SHALL not use the historical randomized effect.

### Requirement 9 — Intervention-aware model

**User story:** As a researcher, I want an explicit intervention-aware outcome or trajectory model so that the role of treatment can be isolated experimentally.

#### Acceptance criteria

1. An intervention SHALL be represented with only the fields available and relevant to the selected trial, which may include concept, dose, unit, route, frequency, initiation, discontinuation, and adherence.
2. The model SHALL encode patient state, intervention, and protocol through separately identifiable channels.
3. The model SHALL define its factual outcome loss, treatment or propensity objective, balancing regularization if used, censoring loss if used, and total optimization objective.
4. The model SHALL produce both potential-outcome predictions, \(\hat Y_i(1)\) and \(\hat Y_i(0)\), while identifying which corresponds to the observed treatment.
5. Only one potential outcome per patient SHALL be described as factually observed.
6. Cohort treatment effects SHALL be computed by a declared standardization or g-computation procedure over a declared target population.
7. Patient-level potential-outcome differences SHALL be labelled model-dependent estimates, not observed individual effects.
8. Intervention-channel ablations SHALL use a capacity-matched placebo or null channel when feasible.
9. Intervention invariance tests SHALL verify that future treatment cannot affect earlier states and that unchanged interventions preserve the appropriate factual computation.
10. If the model does not predict longitudinal next states or multi-step trajectories, the paper SHALL call it an intervention-aware patient outcome model rather than claiming a full patient world model.
11. To retain the world-model claim, the system SHALL implement intervention-conditioned state transitions, multi-step rollouts, and rollout-horizon evaluation.

### Requirement 10 — Uncertainty, support, and abstention

**User story:** As a researcher, I want uncertainty and support diagnostics to restrict unsupported claims without misrepresenting unobservable counterfactual uncertainty.

#### Acceptance criteria

1. The primary predictive-uncertainty method SHALL be prespecified as a deep ensemble, bootstrap ensemble, Monte Carlo dropout, or other defined method.
2. The system SHALL report point predictions and uncertainty.
3. Calibration SHALL be fitted on validation data only.
4. Factual-outcome calibration SHALL include Brier score, calibration slope, calibration intercept, and a reliability diagram where applicable.
5. Conformal prediction MAY be used for factual observed outcomes and SHALL report empirical coverage and interval width.
6. Conformal coverage SHALL not be claimed for individual treatment effects unless a causal conformal method and its assumptions are explicitly implemented.
7. Cohort-level treatment-effect intervals SHALL use patient-level or cluster-level bootstrap or another causally justified interval procedure.
8. Out-of-distribution support SHALL prioritize propensity extremity and one prespecified representation or missingness diagnostic; implementing multiple redundant scores is optional.
9. The system SHALL distinguish a patient-support gate, cohort-estimand gate, and narrative claim gate.
10. Every unsupported result SHALL return machine-readable reasons.
11. Restricting the cohort SHALL redefine the supported target population, and the corresponding estimand SHALL be reported explicitly.
12. Selective-prediction reporting SHALL include retained coverage, selective risk or discrepancy, and incorrect-release and unnecessary-abstention rates where gold labels make them identifiable.

### Requirement 11 — Three-witness validation

**User story:** As a researcher, I want separate temporal, causal-diagnostic, and clinical-constraint witnesses so that no single evaluation dimension is treated as sufficient.

#### Acceptance criteria

1. The temporal witness SHALL evaluate the factual temporal properties implemented by the model, such as outcome prediction, event ordering, active-versus-historical state, or rollout consistency.
2. The causal witness SHALL evaluate overlap, balance, effective sample size, estimator stability, negative controls where available, and sensitivity diagnostics.
3. Insufficient overlap SHALL cause the causal witness to fail or return uncertain even when factual prediction is accurate.
4. Historical RCT effect recovery SHALL not be part of the operational causal witness.
5. The historical effect SHALL be used only in locked final external evaluation after model and policy selection.
6. The clinical-constraint witness SHALL evaluate only an explicit, auditable set of protocol and clinical rules.
7. In the absence of qualified clinician review, the paper SHALL use the term clinical-constraint witness rather than claiming comprehensive clinical plausibility.
8. Witness outputs SHALL be `pass`, `fail`, or `uncertain` with rule-level evidence.
9. The primary aggregation policy SHALL be prespecified before test evaluation.
10. Every released cohort estimate SHALL receive an audit report containing protocol version, population, treatment, estimate and interval, overlap diagnostics, witness outcomes, support restrictions, violations, and provenance.

### Requirement 12 — Ablation framework

**User story:** As a researcher, I want controlled ablations of the principal mechanisms so that the contribution of each component can be isolated.

#### Acceptance criteria

1. The configuration system SHALL independently toggle intervention representation, structured protocol conditioning, overlap gating, uncertainty gating, and each witness.
2. The primary intervention ablation SHALL compare treatment as an ordinary feature with an explicit intervention channel and a capacity-matched null or placebo channel where feasible.
3. The primary protocol ablation SHALL compare no protocol conditioning with structured executable protocol conditioning.
4. Unstructured protocol text MAY be included as a secondary baseline.
5. The primary abstention ablation SHALL compare no restriction, overlap only, uncertainty only, and combined support restriction.
6. Each witness SHALL be removed independently.
7. Shuffled pathway or mechanism controls SHALL be required only if external mechanism or pathway knowledge is used.
8. Correctly aligned external knowledge SHALL outperform shuffled and capacity-matched controls before a knowledge-specific benefit is claimed.
9. All primary ablations SHALL reuse identical splits, seeds, and evaluation procedures.

### Requirement 13 — Robustness and validation

**User story:** As a researcher, I want prespecified stress tests so that the result is not based on one favorable static configuration.

#### Acceptance criteria

1. At least one robustness analysis SHALL be selected before test evaluation.
2. The submission-critical robustness analysis SHALL be either temporal shift, clinically justified subgroup analysis, or missingness stress, chosen based on dataset support.
3. Every subgroup result SHALL include sample size, treatment counts, endpoint counts, overlap, and uncertainty.
4. Small or unsupported subgroup estimates SHALL be suppressed or marked uncertain.
5. Temporal-shift evaluation SHALL train or fit on earlier periods and evaluate on later periods without future leakage.
6. Missingness stress SHALL distinguish random removal from clinically informative missingness.
7. Protocol perturbations SHALL include at least negation, threshold boundaries, time-zero shifts, endpoint-window changes, and treatment/comparator reversal when applicable.
8. Cross-site evaluation SHALL be performed only when an external cohort is legitimately available.
9. Absence of cross-site validation SHALL be reported as a limitation rather than approximated with a random split.
10. Exploratory subgroup and robustness analyses SHALL be labelled exploratory and handled appropriately for multiplicity.

### Requirement 14 — Statistical analysis

**User story:** As a researcher, I want a locked statistical analysis plan so that treatment-effect comparisons and uncertainty are not overstated.

#### Acceptance criteria

1. The analysis plan SHALL specify estimands, primary endpoints, hypotheses, comparison set, uncertainty procedures, and multiplicity handling before test evaluation.
2. Treatment-effect estimates SHALL receive patient-level or appropriate cluster-level bootstrap intervals.
3. The bootstrap procedure SHALL declare whether nuisance models are refitted in each replicate.
4. Model comparisons on the same evaluation population SHALL use paired resampling or an appropriate paired test.
5. Prespecified primary comparisons SHALL receive multiplicity correction.
6. Predictive–causal association SHALL be assessed only across enough independent model or dataset units to support the analysis.
7. With few configurations, the analysis SHALL be descriptive and SHALL not claim a stable correlation.
8. Supported-population estimates SHALL not be presented as estimates for the full eligible population.
9. Sensitivity analysis SHALL include propensity specification, weight clipping, overlap definition, uncertainty threshold, protocol interpretation, and an unmeasured-confounding analysis when defensible.
10. Statistical significance SHALL not substitute for clinical relevance; effect discrepancies SHALL be interpreted against prespecified clinical margins where available.

### Requirement 15 — Reporting and paper artifacts

**User story:** As a researcher, I want one command to regenerate privacy-safe tables, figures, and audit artifacts so that the submission is reproducible and responsibly documented.

#### Acceptance criteria

1. The main results table SHALL include factual performance, causal baseline estimates, intervention-aware estimate, treatment-effect discrepancy, protocol violations, uncertainty, and supported-population coverage.
2. The principal figures SHALL include the system architecture, cohort flow, treatment-effect comparison, propensity overlap, selective-risk or discrepancy curve, and principal ablations.
3. The cohort-flow diagram SHALL show initial population, exclusions, treatment arms, censoring, analyzed population, and support-restricted population.
4. A model card SHALL state intended use, prohibited uses, training population, endpoint, uncertainty behavior, known failure modes, and subgroup findings.
5. A single documented command SHALL regenerate paper tables and figures from saved privacy-safe predictions or aggregates.
6. The reproduction command SHALL not require access to the historical reference until the locked external-evaluation stage.
7. Responsible-use documentation SHALL cover confounding, privacy, historical bias, subgroup performance, unsupported individual recommendations, misuse as an RCT replacement, uncertainty, clinical oversight, and auditability.
8. The paper SHALL distinguish observed outcomes, modelled potential outcomes, cohort estimands, and simulated trajectories.
9. The paper SHALL distinguish validation of causal diagnostics from descriptive agreement with a single historical trial.
10. The final artifact SHALL pass anonymity, page-limit, citation, privacy, and reproducibility checks.

## 8. Release gates

### 8.1 Patient-support gate

A patient is marked supported only if prespecified overlap, missingness, and input-validity criteria pass.

```python
patient_supported = (
    within_propensity_support
    and input_valid
    and missingness_supported
)
```

Patients outside support are not silently discarded. Their counts and characteristics are reported.

### 8.2 Cohort-estimand gate

A cohort estimate is reportable only if:

- the trial specification is valid;
- the estimand-alignment report is complete;
- overlap meets prespecified criteria;
- covariate balance meets prespecified criteria or residual imbalance is disclosed;
- effective sample size is adequate;
- estimator diagnostics pass;
- uncertainty is finite and interpretable;
- the supported target population is explicitly stated.

### 8.3 Narrative claim gate

The paper may describe an estimate as reproducing or approximately agreeing with the historical effect only after locked evaluation and only when estimand alignment is adequate. Otherwise, it must use language such as "descriptive comparison" or "directionally consistent under differing populations."

## 9. Locked external-evaluation protocol

1. Freeze all source code and configurations used for the primary analysis.
2. Record the source commit and clean-working-tree state.
3. Generate the observational treatment-effect estimate and interval.
4. Generate all operational witness outputs without accessing the historical effect.
5. Lock the supported target population and all exclusions.
6. Load the historical reference effect from a separate evaluation-only file.
7. Run estimand-alignment checks.
8. Compute \(D_\tau\), \(E_\tau\), and \(Z_\tau\) where valid.
9. Generate the final comparison figure and audit report.
10. Do not return to model or threshold selection after viewing the comparison without declaring a new exploratory analysis.

## 10. Minimum ablation matrix

| Arm | Treatment representation | Protocol constraints | Overlap restriction | Uncertainty restriction | Witnesses |
|---|---|---|---|---|---|
| B0 | Ordinary feature | No | No | No | No |
| B1 | Explicit intervention | No | No | No | No |
| B2 | Explicit intervention | Structured | No | No | No |
| B3 | Explicit intervention | Structured | Yes | No | Operational |
| B4 | Explicit intervention | Structured | No | Yes | Operational |
| B5 | Explicit intervention | Structured | Yes | Yes | Operational |

The primary comparison is B5 versus B0. B1 through B4 isolate representation, protocol, overlap, and uncertainty contributions.

## 11. Go/no-go checklist

The final locked evaluation SHALL NOT begin until all answers below are yes:

- Is the treatment, comparator, endpoint, time zero, follow-up, and estimand frozen?
- Is the causal diagram or structured causal model frozen?
- Is the adjustment set frozen?
- Is estimand alignment with the historical trial plausible?
- Is the historical effect inaccessible to model selection?
- Has post-treatment leakage testing passed?
- Have patient-level splits and cross-fitting been verified?
- Are overlap, uncertainty, and witness thresholds frozen?
- Is the supported target population defined?
- Are the primary endpoints and statistical procedures locked?
- Have privacy and artifact-export checks passed?
- Can the pipeline reproduce its development results from one command?

If any answer is no, the external comparison remains blocked.

## 12. Required paper limitations

The submission SHALL state that:

- causal validity depends on untestable identification assumptions;
- one historical trial does not establish general treatment-effect accuracy;
- estimand and population differences may limit comparability;
- individual counterfactual outcomes are unobservable;
- automated clinical constraints are incomplete;
- factual calibration does not prove counterfactual calibration;
- support restriction changes the target population;
- retrospective agreement does not establish prospective clinical utility;
- the system is not approved for individual treatment decisions;
- randomized evidence is not replaced by the proposed model.
