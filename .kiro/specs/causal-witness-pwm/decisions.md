# Frozen Decisions

## Causal Witness Patient World Model

**Status:** Frozen. Changes after this point require a documented amendment (see Section 26) and, if made after analysis lock, must be disclosed as a protocol deviation in the paper.
**Applies to:** `requirements.md` in this spec. This document resolves every open design choice, placeholder, and optional-wording clause left unresolved in the requirements.
**Last updated:** September 6, 2026

This document is the single source of truth for dataset, treatment, endpoint, estimand, model, uncertainty, overlap, witness, ablation, robustness, and statistical choices. `design.md` and `tasks.md` are built on top of it and must not reintroduce discretion this document has already closed.

---

## 1. Paper framing

**Working title:** *Causal Witnesses for Patient World Models: Knowing When Not to Simulate Under Limited Overlap*

**Primary research question:** Can an intervention-aware longitudinal patient model recover a historical randomized treatment effect from a deliberately confounded cohort while abstaining when causal support is inadequate?

**Secondary research questions:**

1. Does explicit intervention representation outperform treatment encoded as an ordinary feature?
2. Does good factual prediction imply accurate treatment-effect recovery?
3. Do overlap and uncertainty gates reduce incorrect treatment-effect releases?
4. Which temporal, causal, or clinical witness detects failures missed by predictive accuracy alone?

**Framing constraint:** This is a controlled causal benchmark built from a historical randomized trial, not an independent real-world EHR validation. Every downstream document and every figure/table caption must preserve this framing (see Section 22, permitted/prohibited claims).

---

## 2. Dataset

| Field | Value |
|---|---|
| Dataset | `LongCART::ACTG175`, version 3.2 |
| Packaged population | 2,139 patients before treatment-arm restriction |
| Available modalities | baseline covariates, longitudinal CD4/CD8 measurements, treatment, censoring, clinical-event outcomes |
| Data integrity | SHA-256 checksum computed and frozen at ingestion; pipeline refuses to run on a mismatched checksum |

ACTG 175 is a longitudinal randomized dataset with a published treatment effect. It is not a multimodal EHR dataset, and no claim in the paper may imply otherwise.

---

## 3. Treatment comparison

| Field | Value |
|---|---|
| Intervention | zidovudine + didanosine (ACTG arm 1) |
| Comparator | zidovudine alone (ACTG arm 0) |
| Assignment time | randomization |
| Strategy | intention to treat (ITT) |
| Treatment switching | recorded and described descriptively; ignored in the primary ITT analysis |

Arms 2 and 3 (zalcitabine, and zidovudine+zalcitabine) are excluded before any modeling, with no exceptions. They are not used as a negative-control population (Section 11.3 explains why they were rejected for that role).

---

## 4. Target population

Included patients must:

1. satisfy ACTG 175's original HIV and baseline-CD4 eligibility criteria;
2. belong to treatment arm 0 or arm 1;
3. have valid baseline treatment and covariate records;
4. have nonnegative recorded follow-up time;
5. have sufficient information to construct the primary composite endpoint.

No post hoc exclusion based on observed outcomes is permitted at any stage.

---

## 5. Time zero and follow-up

| Field | Value |
|---|---|
| Time zero | randomized treatment assignment |
| Primary follow-up | complete available trial follow-up |
| Secondary fixed horizon | 96 weeks |
| Censoring | administrative end of follow-up, or last recorded observation |
| Landmark measurements | baseline, week 20, week 96 (where available) |

Post-treatment CD4/CD8 values, treatment discontinuation, and any follow-up-derived variable are permanently excluded from the baseline adjustment set (enforced by the leakage auditor, Requirement 5.10).

---

## 6. Endpoints

**Primary endpoint:** time to the ACTG composite outcome, defined as the first occurrence of:

- ≥50% decline in CD4 count from baseline, or
- an AIDS-defining event, or
- death.

**Secondary endpoints:**

- 96-week composite-event risk (binary);
- week-20 CD4 count (continuous);
- week-96 CD4 count (continuous);
- longitudinal CD4 trajectory (sequence).

The composite time-to-event endpoint carries the causal-validation claim. The CD4/CD8 trajectory endpoints carry the world-model claim (Section 14).

---

## 7. Estimands

**Primary estimand:** marginal hazard ratio, arm 1 vs. arm 0, under ITT.

**Published external reference:** HR = 0.50 (zidovudine+didanosine vs. zidovudine alone). Source: Hammer SM, et al., *NEJM* 1996. PMID [8813038](https://pubmed.ncbi.nlm.nih.gov/8813038/), DOI `10.1056/NEJM199610103351501`.

**Secondary estimand:** marginal 96-week risk difference,

$$
P(Y^{A=1}_{96}=1) - P(Y^{A=0}_{96}=1)
$$

estimated directly from the full randomized (pre-observationalization) cohort.

Patient-level predicted potential-outcome differences are model estimates and must never be reported or described as observed individual treatment effects (per Requirement 9.7).

---

## 8. Reference-effect policy

Two references are locked and used for different purposes:

1. **External published reference** — HR = 0.50 (Section 7). Historical external corroboration only.
2. **Internal randomized reference** — the effect estimated from the complete eligible arm-0/arm-1 dataset, prior to observationalization.

Because the packaged dataset originates from the same trial as the published reference, the published reference is *not* an independent external cohort and must be described as historical corroboration, never as independent validation (closes Requirement 4.4–4.6 for this specific case: the "lack of independence" is disclosed here, permanently).

The internal randomized reference is computed exactly once, on the complete eligible arm-0/arm-1 cohort, **before** any splitting or observationalization step. It is written to a sealed reference artifact and withheld from model development. It may be opened only after model configurations, the selection mechanism, witness rules, and release thresholds have all been frozen (i.e., at the locked external-evaluation stage, `requirements.md` Section 9).

Neither reference may be used for:

- hyperparameter selection;
- model selection;
- early stopping;
- uncertainty-threshold selection;
- observationalization-mechanism tuning (beyond the single documented pilot recalibration in Section 9).

**Clinically meaningful discrepancy margin (δ):** none is defined. Requirement 4.7's "when defensible" clause is resolved as *not defensible here* — a single small-sample historical HR does not support a principled equivalence margin. Report log-HR discrepancy and its confidence interval; do not threshold it into a binary pass/fail against an arbitrary δ.

---

## 9. Controlled observationalization

The randomized cohort is converted into a reproducible pseudo-observational cohort via a fixed, documented, treatment- and covariate-dependent selection mechanism.

### Frozen order of operations

This order is mandatory and must not be reordered:

1. Construct the complete eligible arm-0/arm-1 randomized cohort (Section 4).
2. Compute and seal the internal randomized reference effect (Section 8, item 2) on that complete cohort, before any splitting or selection. Write it to a sealed reference artifact and withhold it from model development.
3. Create the 60%/20%/20% patient-level development/validation/test split (Section 10) using the frozen split seed, stratified as specified there.
4. Fit all preprocessing transformations (standardization/z-scoring of covariates used in the risk score, imputation if any) on the development split only.
5. Calculate the observationalization risk score $r_i$ using those development-fitted transformations, applied to every split.
6. Apply the same frozen selection formula independently within development, validation, and test.
7. Draw each split's selection indicators using deterministic split-specific seeds derived from one master seed and the split name (e.g. `seed_dev = HMAC(master_seed, "dev")`, `seed_val = HMAC(master_seed, "val")`, `seed_test = HMAC(master_seed, "test")`), so no random-number stream is reused across splits.
8. Freeze the retained patient IDs and a cohort manifest for every split.

Validation or test outcomes must never influence the selection coefficients, preprocessing transformations, random seeds, or retention targets. Selection must never be applied to the full cohort before splitting.

**Baseline risk score**, from development-fitted standardized (z-scored) baseline covariates:

$$
r_i = -0.8\,z(\mathrm{CD4}_i) - 0.3\,z(\mathrm{Karnofsky}_i) + 0.3\,z(\mathrm{Age}_i) + 0.6\,\mathrm{Symptomatic}_i + 0.6\,\mathrm{PriorOI}_i + 0.4\,\mathrm{PriorART}_i
$$

**Selection probability:**

$$
P(S_i = 1 \mid A_i, X_i) = \sigma\bigl(1.2 + (2A_i - 1)\,r_i\bigr)
$$

where $S_i = 1$ means the patient is retained in the pseudo-observational cohort.

**Frozen artifacts (all persisted in the artifact manifest, Requirement 1.10):**

- the formula and every coefficient above;
- the development-fitted z-scoring parameters (mean/SD per covariate) used for every split;
- the master seed and the per-split HMAC-derived seeds used to draw $S_i \sim \mathrm{Bernoulli}(P(S_i=1\mid A_i,X_i))$;
- the resulting retained patient ID list per split;
- resulting treatment prevalence in the retained cohort, per split;
- resulting propensity distribution in the retained cohort, per split.

**One-time pilot recalibration:** if this mechanism produces an unusable sample on development data (either arm below the minimum count in Section 12), the coefficients may be recalibrated exactly once, during a documented pilot run on development data only — never on validation or test. The recalibration decision, old coefficients, new coefficients, and reason are recorded in this document's changelog (Section 26) before the mechanism is frozen for real. No further recalibration is permitted after that freeze, and no recalibration may ever be triggered by validation or test results.

This guarantees:

- the test split's pseudo-observational cohort is generated by the identical mechanism as dev/val, with no mechanism recalibration possible after seeing validation or test data;
- the internal randomized reference is computed from the full pre-split, pre-selection cohort, so it is never contaminated by either the split or the observationalization step;
- selection-induced treatment prevalence and propensity distribution are reported per split, not only pooled.

---

## 10. Data splitting

Patient-level split, performed on the complete eligible arm-0/arm-1 cohort, strictly **after** the internal randomized reference is sealed (Section 9, step 2) and **before** any fitted preprocessing and before observationalization:

| Split | Fraction |
|---|---|
| Development | 60% |
| Validation | 20% |
| Held-out test | 20% |

Stratified on: treatment arm, primary-event indicator, prior-antiretroviral-treatment stratum.

All preprocessing (z-scoring, imputation, propensity-model fitting) is fit on the development split only and applied unchanged to validation and test. Five-fold cross-fitting is used inside development+validation for causal nuisance models (propensity, outcome regression components). The test split remains sealed until every configuration and every release-gate threshold is frozen (Section 17's go/no-go checklist, inherited from `requirements.md` Section 11).

The pseudo-observational selection mechanism (Section 9) is applied only after this split, independently within each split, using deterministic split-specific seeds derived from one master seed. The same mathematical selection mechanism and development-fitted coefficients are used in every split; only the random draw differs per split, and no random-number stream is reused across splits.

---

## 11. Causal baselines

**Prespecified baseline estimators:**

1. Unadjusted Kaplan–Meier and unadjusted Cox model.
2. Covariate-adjusted Cox model (adjustment set: Section 6 of `requirements.md`'s design — see Section 13 below for the concrete list).
3. Propensity-score-weighted marginal Cox model.
4. Outcome regression for 96-week risk.
5. IPCW-adjusted inverse-probability weighting.
6. IPCW-AIPW for the 96-week risk difference. This is the primary conventional causal baseline (resolves Requirement 7.8).

**Propensity modeling:**

- primary model: logistic regression;
- sensitivity model: gradient boosting (fixed config: `XGBoost`, `max_depth=3`, `n_estimators=200`, `learning_rate=0.05`, seed-locked) — resolves the "MAY include gradient boosting" clause in Requirement 7.4 to a firm "WILL, as sensitivity analysis only, never primary";
- stabilized weights used throughout;
- primary weight clipping: 1st/99th percentile;
- sensitivity clipping: 0.5th/99.5th and 5th/95th percentile.

### 11.3 Negative controls (revised)

Arms 2/3 (zalcitabine-containing regimens) and hemophilia status are **rejected** as negative controls: both are plausible independent causal factors (arms 2/3 are active antiretroviral therapies; hemophilia is associated with bleeding-related morbidity and healthcare-utilization patterns), so a detected association would not cleanly falsify the pipeline. Using either would risk mistaking a real clinical effect for a pipeline artifact, or vice versa.

Two concrete, prespecified negative-control pairs are frozen instead:

1. **Negative-control exposure — synthetic sham treatment.** A deterministic synthetic label generated via `HMAC-SHA-256(patient_id, negative_control_seed)`, thresholded to reproduce the observed real-treatment prevalence in the development pseudo-observational cohort. It is generated **without reference to** any covariate, outcome, censoring indicator, or the real treatment assignment — it depends only on `patient_id` and the frozen negative-control seed. The primary time-to-event causal estimator (IPCW-AIPW, Section 11) is rerun with this sham label substituted for the real treatment. Because the sham label is generated independently of all clinical information, it cannot have a true causal effect on the composite endpoint; any detected effect indicates estimator instability or a pipeline defect, not confounding.

2. **Negative-control outcome — pre-treatment baseline CD8.** The last CD8 value recorded at or before time zero, used as a pseudo-outcome. Since it is measured before treatment assignment, treatment cannot causally affect it. The real treatment assignment and the full baseline adjustment set (Section 13) are used, but **baseline CD8 is removed from the covariate/adjustment side of this specific regression** (see the double-use safeguard below) so the check is not tautological (i.e., CD8 does not "adjust for itself").

**Frozen artifacts:** the negative-control seed, the sham-label generation rule, the per-patient sham assignments, and the baseline-CD8 extraction rule are all stored in the experiment manifest before test evaluation (Requirement 1.10).

**Double-use safeguard (baseline CD8 as both a covariate and a negative-control outcome):** baseline CD8 is used in two places in this design — as a component of the adjustment set (Section 13) and as the CD4/CD8 longitudinal-state prediction target (Section 14). To prevent it from appearing as both the outcome and a predictor in the *same* regression:

- in the negative-control-outcome regression only, baseline CD8 is excluded from the right-hand-side adjustment set (all other baseline covariates are retained);
- in every other model (propensity model, outcome regression, GRU state-prediction head), baseline CD8 continues to serve its normal role as an adjustment covariate or prediction target and is not otherwise restricted;
- this exclusion rule is scoped narrowly to the negative-control-outcome check and must not be read as removing CD8 from the adjustment set elsewhere.

**Pass / fail / uncertain thresholds:**

- *Negative-control exposure* (sham treatment, hazard ratio scale): **pass** if the estimated hazard ratio's 95% CI contains 1.0 **and** the point estimate falls within [0.90, 1.10]; **fail** if the 95% CI excludes 1.0; **uncertain** if the CI contains 1.0 but the point estimate falls outside [0.90, 1.10] (i.e., wide/unstable interval).
- *Negative-control outcome* (baseline CD8, mean-difference scale): **pass** if the adjusted standardized mean difference is below 0.10 in absolute value **and** the bootstrap CI for the adjusted mean difference contains zero; **fail** if the CI excludes zero; **uncertain** if the CI contains zero but the absolute standardized mean difference is ≥ 0.10.
- The [0.90, 1.10] hazard-ratio band is a prespecified falsification tolerance for detecting pipeline defects, not a claim of clinical equivalence.

Under the primary all-witnesses-required policy (Section 16), a **fail or uncertain** result on either negative control blocks release of the cohort-level estimate.

Both checks are computed on development/validation cross-fitted data during development, and recomputed once at the frozen test evaluation stage.

If treatment or censoring is found to vary in a way that requires a longitudinal (time-varying) estimator, the trial is narrowed to the baseline (ITT, single time-zero) estimand rather than implementing a marginal structural model — resolves Requirement 7.11 by choosing the narrowing branch, since ACTG 175's primary strategy is ITT-at-randomization.

---

## 12. Overlap and support rules

**Primary common-support rule:**

$$
0.10 \le \hat e(X) \le 0.90
$$

**Sensitivity rule:**

$$
0.05 \le \hat e(X) \le 0.95
$$

A causal estimate is support-valid only if all of the following hold:

- both arms retain ≥100 patients;
- effective sample size ≥100 per arm;
- every adjusted baseline covariate has absolute standardized mean difference < 0.10;
- no unresolved positivity warning remains.

These four conditions are the concrete implementation of Requirement 7 and Section 8.2's cohort-estimand gate.

---

## 13. Causal graph and adjustment set

```text
Baseline covariates X ──→ Treatment A ──→ Longitudinal state L ──→ Outcome Y
        │                     │                                  ↑
        └─────────────────────┴──────────────────────────────────┘

X, A ──→ Observational-selection indicator S
A, X, L ──→ Censoring C
```

**Baseline adjustment set (frozen):**

age; weight; sex; race/ethnicity; hemophilia; homosexual activity; intravenous drug-use history; Karnofsky score; prior opportunistic infection; symptomatic status; baseline CD4; baseline CD8; prior antiretroviral exposure and duration.

**Explicitly excluded from baseline adjustment (leakage-audited, Requirement 5.10):**

week-20 CD4/CD8; week-96 CD4/CD8; treatment discontinuation; any future measurement; event-derived variables; censoring information observed after time zero.

---

## 14. Models

### 14.1 Endpoint-specific non-neural baseline

Cox proportional-hazards regression on the composite time-to-event endpoint (resolves Requirement 8.1 — time-to-event endpoint requires Cox, not logistic regression, given the primary endpoint is time-to-composite-event).

### 14.2 Sequence-model baseline

Capacity-matched GRU patient-state model:

- baseline covariates initialize hidden state;
- CD4/CD8 observations form the longitudinal input sequence;
- treatment supplied as an ordinary input feature (undifferentiated from other covariates);
- two output heads: next patient-state (CD4/CD8) regression, and discrete-time event-hazard.

A Transformer architecture is rejected as unnecessary given only three principal clinical timepoints (baseline, week 20, week 96).

### 14.3 Intervention-aware model family and loss

Small intervention-aware recurrent model:

- baseline-state encoder;
- separate structured intervention encoder;
- protocol encoder;
- GRU state-transition module;
- treatment-specific event-hazard heads (one per branch);
- CD4/CD8 state-prediction head.

Representation:

$$
z_t = [z_t^{state}; z_t^{intervention}; z_t^{protocol}]
$$

For every eligible patient, both branches are evaluated: $\hat Y_i(1)$, $\hat Y_i(0)$.

**Frozen training objective:**

$$
\mathcal{L} = \mathcal{L}_{\text{survival}} + \lambda_{\text{state}}\,\mathcal{L}_{\text{CD4/CD8}} + \lambda_{\text{balance}}\,\mathcal{L}_{\text{representation balance}}
$$

- $\mathcal{L}_{\text{survival}}$: discrete-time survival negative log-likelihood;
- $\mathcal{L}_{\text{CD4/CD8}}$: masked Huber loss;
- $\mathcal{L}_{\text{representation balance}}$: MMD penalty between treated and comparator representations.

$\lambda_{\text{state}}$ and $\lambda_{\text{balance}}$ are selected on validation data only, via a small prespecified grid (frozen grid: $\lambda \in \{0.1, 0.3, 1.0\}$ for each, 9 combinations, selected by validation composite-endpoint concordance).

**Placebo/null channel** (resolves "when feasible" in Requirement 9.8): a fixed random vector matching the intervention channel's dimensionality, redrawn once per seed and frozen, used in Ablation 10 (Section 14.4 below).

### 14.4 World-model status

This is designated a **minimal structured patient world model**: it satisfies Requirement 9.11 because it implements intervention-conditioned state transitions (the GRU state-transition module) and produces multi-step rollouts across the three landmark timepoints (baseline → week 20 → week 96) with a rollout-horizon evaluation (Section 14 secondary endpoints: week-20 CD4, week-96 CD4, full trajectory).

The paper must not claim general-purpose patient simulation, multimodal clinical understanding, or physiological completeness (Section 15).

---

## 15. Uncertainty method

- Primary method: **deep ensemble**, 5 independently initialized replicas.
- Ensemble mean → point prediction; ensemble variance → epistemic uncertainty.
- Treatment-effect intervals: 1,000 patient-level bootstrap replicates.
- Calibration fitted on validation data only.
- For the 96-week binary outcome: IPCW Brier score, calibration slope, calibration intercept, reliability curve.
- Conformal treatment-effect intervals are explicitly **not** a P0 requirement — ordinary split-conformal methods do not automatically yield valid individual counterfactual-effect intervals, and implementing a causally valid conformal method is out of scope for the critical path.

---

## 16. Witness aggregation and definitions

**Primary aggregation policy:** all three witnesses required for release. Two-of-three is reported only as an exploratory sensitivity comparison, never as the primary gate.

**Patient-level retention condition:**

```text
patient_supported =
    propensity_in_common_support
    AND ensemble_uncertainty <= validation_90th_percentile
    AND protocol_valid
    AND no_hard_clinical_or_temporal_violation
```

**Cohort-level release condition:**

```text
release =
    overlap_valid
    AND balance_valid
    AND uncertainty_valid
    AND temporal_witness == pass
    AND causal_witness == pass
    AND clinical_witness == pass
```

### Temporal witness — passes when

- all baseline features precede treatment;
- outcomes occur after time zero;
- state transitions preserve chronological ordering;
- no future information enters any prediction;
- held-out longitudinal (CD4/CD8) error is no worse than 5% above the factual GRU baseline (Section 14.2).

### Causal witness — pass / fail / uncertain

**Pass** when all of the following hold:

- positivity requirements hold (Section 12);
- balance requirements hold (Section 12);
- weighted effective sample size is sufficient (Section 12);
- estimates are stable under the prespecified propensity and clipping sensitivities (Section 11);
- both negative controls (Section 11.3) return **pass** under their frozen thresholds.

**Fail** when any hard requirement above is violated, or either negative control returns **fail**.

**Uncertain** when no hard requirement is violated but either negative control returns **uncertain** (i.e., the check is inconclusive rather than clearly failing).

Under the primary all-witnesses-required policy, both **fail** and **uncertain** block release.

### Clinical witness — passes when

- eligibility is satisfied (Section 4);
- treatment assignment is permitted;
- dose/treatment identity is valid;
- event timing is possible;
- endpoint construction follows the protocol (Section 6);
- zero hard protocol violations remain.

---

## 17. Primary ablations (frozen list, exactly these ten)

1. No treatment input.
2. Treatment as an ordinary feature.
3. Explicit intervention channel.
4. Intervention channel without protocol representation.
5. Complete model without overlap gate.
6. Complete model without uncertainty gate.
7. Complete model without clinical constraints.
8. Each witness removed independently.
9. Predictive accuracy alone (no causal adjustment).
10. Random capacity-matched intervention embeddings (placebo channel, Section 14.3).

Mechanism/pathway encoders remain out of scope; if one is added later, shuffled-label controls become mandatory (per Requirement 12.7–12.8).

---

## 18. Primary evaluation metrics

**Factual prediction:** time-dependent concordance; 96-week AUROC; 96-week AUPRC; IPCW Brier score; calibration slope and intercept; CD4/CD8 mean absolute error.

**Causal recovery:** estimated hazard ratio; absolute log-hazard-ratio error

$$
E_{\log HR} = \left|\log(\hat{HR}) - \log(0.50)\right|
$$

effect-direction agreement; confidence-interval overlap; 96-week risk-difference error against the internal randomized reference (Section 8, item 2 — not the published reference, per the reference-effect policy).

**Selective prediction:** retained coverage; selective risk; risk-coverage curve; area under the risk-coverage curve; incorrect-release rate; unnecessary-abstention rate; error at matched coverage.

**Consistency:** temporal violations; protocol violations; impossible-transition rate; witness-detected failures.

---

## 19. Statistical plan

- 5 random seeds per neural configuration.
- 1,000 patient-level bootstrap replicates.
- Paired bootstrap comparisons on identical test patients.
- Holm correction for the three primary model comparisons (baseline GRU vs. intervention-aware model; intervention-aware model vs. IPCW-AIPW; complete gated model vs. ungated model).
- 95% confidence intervals throughout.
- Subgroup effects always reported with sample size and support diagnostics.
- No predictive–causal correlation claim unless enough independent configurations exist to support it (per Requirement 14.6–14.7).
- No arbitrary clinical-equivalence margin (confirmed in Section 8); report discrepancy and its CI.

---

## 20. Primary robustness analysis

**Primary subgroup analysis (frozen):** prior antiretroviral-treatment status — naive vs. experienced.

**Sensitivity analyses:**

- logistic vs. boosted propensity models;
- alternative weight clipping (Section 11);
- alternative common-support thresholds (Section 12);
- increased laboratory missingness (stress test);
- removal of week-20 measurements;
- treatment/comparator reversal;
- shifted endpoint horizon;
- stronger observational-selection severity (modified $r_i$ coefficients, documented separately from the frozen primary mechanism).

**Explicitly out of scope:** calendar-time (temporal) shift and cross-site validation. ACTG 175 does not support either credibly — this resolves Requirement 13.2's "chosen based on dataset support" by ruling out the two options that don't apply, leaving subgroup analysis as the only viable primary robustness check for this dataset.

---

## 21. Success criterion

The model is successful only if, on held-out (sealed) test data:

1. it reduces treatment-effect error relative to the treatment-as-feature GRU baseline;
2. its factual performance is no worse than the prespecified 5% tolerance (Section 16, temporal witness);
3. the complete release gate reduces incorrect releases at matched coverage relative to no gating;
4. every released estimate passes all three witnesses;
5. the result is directionally stable across the 5 seeds and the primary sensitivity analyses (Section 20).

**A valid negative result**, to be reported as such if observed:

> Better factual trajectory prediction did not produce better randomized treatment-effect recovery, while overlap-aware abstention prevented unsupported causal claims.

---

## 22. Permitted and prohibited claims

### Permitted

- explicit intervention representations improve, or fail to improve, controlled treatment-effect recovery;
- predictive accuracy and causal recovery are empirically distinct;
- overlap-aware abstention reduces unsupported releases;
- independent witnesses expose different failure modes;
- the benchmark evaluates causal recovery under known, controlled confounding.

### Prohibited

- claiming the cohort is independent real-world evidence;
- claiming individual counterfactual outcomes are ground truth;
- claiming the system replaces randomized trials;
- claiming ACTG 175 demonstrates modern clinical-deployment readiness;
- claiming the model represents complete patient physiology;
- claiming clinical plausibility establishes causal identification.

---

## 23. Final scope freeze

### In scope

one dataset; one binary treatment comparison; one primary composite endpoint; longitudinal CD4/CD8 state prediction; controlled confounding via documented observationalization; causal baselines; intervention-aware model; uncertainty and overlap abstention; three witnesses; the ten frozen ablations; the frozen sensitivity analyses.

### Out of scope

multimodal notes or imaging; multiple trials; external hospitals/cross-site validation; natural-language protocol parsing; pathway or mechanism encoders (unless added later, in which case shuffled-label controls are mandatory); autonomous agents; individualized treatment recommendations; synthetic replacement trial arms; production or regulatory deployment.

---

## 24. Primary trial source

Hammer SM, et al. *A trial comparing nucleoside monotherapy with combination therapy in HIV-infected adults with CD4 cell counts from 200 to 500 per cubic millimeter.* New England Journal of Medicine, 1996.

- [PubMed PMID 8813038](https://pubmed.ncbi.nlm.nih.gov/8813038/)
- DOI: `10.1056/NEJM199610103351501`

---

## 25. Neural-training implementation freeze (PyTorch build)

Section 14.3 freezes the intervention-aware architecture's *structure*
(encoders, GRU transition, branch heads, loss terms, lambda grid) and
Section 19 freezes "5 random seeds per neural configuration" as a count.
Neither section fixes the concrete hyperparameters an actual training run
requires. This section freezes every one of those, before any PyTorch
training code is written, per this document's own "freeze before
implementing" convention (Section 9's frozen order of operations; Section
14.3's frozen lambda grid). These are genuine frozen decisions (an
amendment after this point requires a Section 26 log entry, exactly like
every other value in this document) — not defaults an implementer may
silently vary per run.

**Five replication seeds** (Section 19's "5 random seeds", now with concrete
values): `[101, 202, 303, 404, 505]`. Used, in order, for every seeded
neural-configuration run (baseline GRU, intervention-aware model, and their
5-member deep ensembles) across the 5-seed replication (Section 4 of the
implementation plan). Each seed's own ensemble-replica sub-seeds are still
derived via `infra.seeding.derive_seed(seed, f"replica_{i}")`
(`uncertainty.ensemble.DeepEnsemble`'s existing convention) — these five
values are the top-level seeds that derivation starts from, never a
substitute for it.

**Hidden width:** `hidden_size = 16` for both the sequence-GRU baseline and
the intervention-aware model's `GRUTransition` (same width for both, so the
two are capacity-comparable — the "treatment-as-feature vs. explicit-channel"
ablation contrast, Section 17 items 2-3, isolates the *representational*
difference, not a capacity difference). Chosen as the upper end of
`sequence_baseline.py`'s already-documented "4-16" range: ACTG175's
development split (~370-380 patients, 3 landmark timesteps) is small enough
that a wider hidden state would risk overfitting the discrete-hazard/CD4-CD8
heads, and 16 keeps total parameter count small relative to patient count
while giving the intervention-aware model's three concatenated encoder
channels (`STATE_INPUT_DIM=4 + INTERVENTION_INPUT_DIM=3 + PROTOCOL_INPUT_DIM=2
= 9`-wide input per landmark) enough hidden capacity to combine them
non-trivially.

**Encoder channel widths:** unchanged from the existing pure-Python
prototype's documented defaults (`models.intervention_aware.py`):
`STATE_INPUT_DIM=4`, `INTERVENTION_INPUT_DIM=3`, `PROTOCOL_INPUT_DIM=2`. The
placebo/null channel (Section 14.3, Ablation 10) matches
`INTERVENTION_INPUT_DIM`'s width exactly, per that section's "fixed random
vector matching the intervention channel's dimensionality."

**Discrete-time survival intervals:** 8-week-wide bins, `[0,8), [8,16), ...,
[168,176)`, plus one final open bin `[176, ∞)` for the small right tail
beyond 176 weeks — 23 bins total. Chosen because the real ACTG175 v3.2
distribution's observed follow-up spans 14–1231 days (≈2–176 weeks; verified
at ingestion, `data/raw/actg175_v3_2/outcomes.csv`/`censoring.csv`), so 8-week
bins give reasonable per-bin event/censoring counts on a ~380-patient
development split without so many bins that most are empty. Every patient's
discrete hazard target is the bin index containing their
`endpoint_result.event_time` (elapsed weeks since assignment), reusing
`causal.baselines._elapsed_weeks`'s existing elapsed-time convention. This is
a genuinely new frozen choice (`decisions.md` previously only fixed the
*landmark* timepoints 0/20/96 for state prediction, never the *hazard*
discretization) — the two are independent: state prediction still happens
only at the three landmarks; the survival loss operates on this finer
23-bin discretization of the same follow-up window.

**Optimizer:** Adam, `betas=(0.9, 0.999)`, `weight_decay=1e-4`. Chosen as the
standard, low-maintenance default for a small recurrent model with this
few parameters — no learning-rate-schedule/warmup is used (the training run
is short enough, per patience below, that a schedule adds a tunable choice
this dataset's size cannot reliably justify).

**Learning rate:** `1e-3`, fixed (no schedule).

**Batch size:** full-batch (the entire development-retained population per
gradient step) — `csv` sizes here are ~370-380 patients, well within a
single A100's memory for this model's width, so mini-batching would only add
gradient noise without a real memory or speed benefit.

**Gradient clipping:** global-norm clip at `1.0` (`torch.nn.utils.clip_grad_norm_`),
applied every step — a defensive default for a GRU trained on a small
dataset where occasional large steps are more likely to reflect the
train/val split composition than a real learning-rate problem.

**Early-stopping patience:** `20` epochs without validation-loss improvement
(validation = the observationalized validation split, never development or
test), `max_epochs = 500` as a hard ceiling regardless of patience.
Checkpoint selection uses the validation-loss-minimizing epoch's weights,
consistent with `models.training`'s existing "early stopping/checkpoint
selection must never see the historical randomized-trial effect" scope note
— this rule is unchanged, only now enforced over real gradient-descent
training rather than a numerical-gradient smoke test.

**Device / determinism convention:** every training script accepts a
`--device` flag (`cpu` default, `cuda` when available) and must produce
identical loss curves and final weights on both, up to floating-point
accumulation-order differences — enforced by seeding
`torch.manual_seed`/`torch.cuda.manual_seed_all` from the same top-level
seed used for every other seeded sub-stream in this pipeline
(`infra.seeding.derive_seed`), and by setting
`torch.use_deterministic_algorithms(True)` wherever a deterministic CUDA
kernel exists for the ops this model uses. A run's device is recorded in
its manifest (`infra.manifest.ManifestWriter`) precisely because "ran on
CPU" vs. "ran on GPU" is itself provenance-relevant, not because the two are
expected to disagree.

**Checkpoint format:** `torch.save` of a plain `dict` (`{"model_state_dict":
..., "optimizer_state_dict": ..., "epoch": ..., "seed": ..., "val_loss":
...}`), never a pickled model object — matches the existing
`models.training.save_checkpoint`'s "plain dict, not a pickled class
instance" convention, so a checkpoint never depends on this codebase's
exact class definitions still existing to be unpickled later.

---

## 26. Amendment log

Any change to a decision in this document after design.md/tasks.md are generated must be logged here with date, section, prior value, new value, and reason.

| Date | Section | Prior value | New value | Reason |
|---|---|---|---|---|
| 2026-09-07 | 2 (Dataset) | `checksum_sha256` placeholder (all-zeros, 64 hex digits) in `configs/datasets/actg175_v3_2.yaml` | `checksum_sha256: "df3867e608060bbd965c299cb6f071fabd53c82e7e9a1a5121078fba6ffb83e3"`, computed by `compute_actg175_checksum()` over the real six-table `data/raw/actg175_v3_2/` directory | Real ACTG175 v3.2 data obtained (see `docs/data_access.md` Section 5 for the access-channel/DUA record) and ingested; the placeholder checksum documented in Section 2/`decisions.md` as "not yet computed" is replaced with the frozen real value now that the raw loader (task 5.1) has real data to check against. No other Section 2 value changed. |
| 2026-09-07 | Section numbering (23-26) | "Final scope freeze" = 23, "Primary trial source" = 24, "Amendment log" = 25 | "Final scope freeze" = 23, "Primary trial source" = 24, "Neural-training implementation freeze" = 25 (new), "Amendment log" = 26 | Adding the new Section 25 (neural-training implementation freeze, below) before the amendment log required renumbering the amendment log from 25 to 26; every in-document cross-reference to "Section 25" (the amendment-log convention) and every code comment referencing `decisions.md` Section 25 was updated to Section 26 in the same change. |
| 2026-09-07 | 25 (new: Neural-training implementation freeze) | not previously specified (Section 14.3 froze architecture structure only; Section 19 froze seed *count*, not seed *values*) | Freezes: 5 replication seeds `[101, 202, 303, 404, 505]`; `hidden_size=16` for both GRU baseline and intervention-aware model; encoder channel widths unchanged from the existing prototype (4/3/2); 23 discrete-time survival bins (8-week-wide, plus one open final bin); Adam optimizer (`betas=(0.9,0.999)`, `weight_decay=1e-4`); learning rate `1e-3`, fixed; full-batch training; global-norm gradient clipping at `1.0`; early-stopping patience `20` epochs, `max_epochs=500`; CPU/GPU determinism convention (seeded, `torch.use_deterministic_algorithms(True)`); checkpoint format (plain dict via `torch.save`, never a pickled model object). | Per the plan's own "freeze remaining modeling choices before implementing" first step, and this document's established convention of resolving every open implementation choice before code is written against it (Section 9's frozen order of operations; Section 14.3's frozen lambda grid) -- adding a real PyTorch training implementation requires concrete hyperparameter values this document did not previously fix. |
| 2026-09-07 | 9 (Controlled observationalization) / 20 (Primary robustness analysis, `stronger_selection_severity` variant) | not previously specified: Section 20 names a "stronger observational-selection severity" sensitivity analysis but fixes no coefficients for it; Section 9 fixes only the primary mechanism (`r_i` coefficients `-0.8 / -0.3 / 0.3 / 0.6 / 0.6 / 0.4`, selection-logit intercept `1.2`) | The stronger-severity mechanism is the primary Section 9 mechanism with every risk-score coefficient scaled by exactly `1.5` and the selection-logit intercept left **unchanged** at `1.2`. In Section 9's form: $r^{\text{stronger}}_i = -1.2\,z(\mathrm{CD4}_i) - 0.45\,z(\mathrm{Karnofsky}_i) + 0.45\,z(\mathrm{Age}_i) + 0.9\,\mathrm{Symptomatic}_i + 0.9\,\mathrm{PriorOI}_i + 0.6\,\mathrm{PriorART}_i$ and $P(S_i = 1 \mid A_i, X_i) = \sigma\bigl(1.2 + (2A_i - 1)\,r^{\text{stronger}}_i\bigr)$. Coefficient by coefficient: CD4 `-1.2`, Karnofsky `-0.45`, Age `0.45`, Symptomatic `0.9`, PriorOI `0.9`, PriorART `0.6`, selection-logit intercept `1.2`. Every other element of Section 9 is unchanged for this variant: the same six covariates, the same development-fitted z-scoring parameters, the same frozen order of operations, the same per-split HMAC-derived seeds, and the same Bernoulli draw. | Requirement 8.7 of `.kiro/specs/neural-experiment-pipeline/requirements.md` requires the stronger-selection coefficients to be read from this document before the first sensitivity run executes, and Requirement 18.2 forbids depending on a value this document does not fix. Rationale for the chosen severity: (a) a uniform `1.5` multiplier on the risk score amplifies the treatment-dependent term `(2A_i - 1) r_i`, which is the confounding channel the variant is meant to stress, without changing which covariates confound or their relative contributions, so a change in the estimate is attributable to severity alone; (b) the intercept is deliberately held at `1.2` so overall retention stays comparable to the primary cohort and the variant does not silently confound "stronger confounding" with "smaller sample"; (c) selection stays strictly probabilistic -- at `1.5x` the logit's extreme reach is roughly `+/- 6`, so selection probabilities remain strictly inside `(0, 1)` and population positivity holds, while empirical overlap tightens, which is exactly the stress the variant is for; a larger multiplier (`2x` or more) pushes extreme-risk patients to near-deterministic selection, where the variant would test near-violated positivity rather than stronger confounding, and a smaller one (`1.25x`) is within the noise of the primary mechanism. This is a **new variant-only mechanism, not a recalibration** of the primary mechanism: Section 9's one-time pilot recalibration allowance is untouched and unused, `DEFAULT_COEFFICIENTS` and the primary `COEF_*` constants in `src/cohorts/selection.py` are unchanged, and the primary estimate continues to use them. |
| 2026-09-07 | 19 (Statistical plan) / 20 (Primary robustness analysis) / 25 (Neural-training implementation freeze, five replication seeds) | not previously specified: Section 19 fixes the seed *count* ("5 random seeds") and Section 25 the seed *values* `[101, 202, 303, 404, 505]`, and Section 20 fixes the eight sensitivity variants, but no section fixes what cross-seed agreement counts as *directionally stable* -- the aggregation rule the five-seed replication is judged by | The frozen cross-seed stability criterion is: effect-direction agreement in **at least 4 of the 5** frozen replication seeds (`101, 202, 303, 404, 505`), **together with** no required sensitivity variant contradicting the aggregate direction. Recorded verbatim as the human-readable criterion the aggregator reports alongside its determination: "Cross-seed stability holds if and only if the effect direction agrees in at least 4 of the 5 frozen replication seeds (101, 202, 303, 404, 505) and no required sensitivity variant contradicts the aggregate direction; otherwise the replication result is reported as directionally unstable with every per-seed value retained." Definitions, so the criterion is mechanically applicable: (a) a seed *agrees* when its effect direction equals the aggregate direction, computed with the existing `evaluation.metrics.effect_direction_agreement`; (b) the *required* sensitivity variants are exactly the eight frozen Section 20 variants (`evaluation.robustness.SENSITIVITY_VARIANTS`); (c) a required variant *contradicts* when it executed successfully and its effect direction is the opposite non-null direction of the aggregate -- a variant that failed to execute is recorded unexecuted and forces overall sensitivity stability to `undetermined` (neural-experiment-pipeline Requirement 8.10), so an unexecuted variant can never be silently counted as agreement; (d) failing the criterion yields the determination `directionally_unstable`, and every per-seed value is retained in the report in that branch exactly as in the stable branch, with no best-seed or worst-seed selection anywhere. Frozen in code as `StabilityCriterion(min_agreeing_seeds=4, total_seeds=5, require_no_contradicting_sensitivity_variant=True, criterion_description=<the sentence above>)` and the module-level constant `CROSS_SEED_STABILITY_CRITERION` in `scripts/run_replication.py`. | Requirements 9.6/9.7 of `.kiro/specs/neural-experiment-pipeline/requirements.md` require the stability criterion to be recorded in this document *before the first replication run executes* and reported alongside the determination, and Requirement 18.2 forbids depending on a value this document does not fix; recording it here is what makes the criterion prespecified rather than chosen after seeing five seeds' worth of estimates. Rationale for 4-of-5: (a) it tolerates exactly one dissenting seed, which is the realistic amount of sign instability at this sample size (N ~ 380 development-retained patients, ~127 per evaluated split) without letting a bare majority count as agreement; (b) unanimity (5-of-5) was rejected as brittle -- with a `null` direction category available, a single near-zero estimate flipping to `null` would discard an otherwise consistent result, which would bias reporting toward discarding rather than toward honesty; (c) a bare majority (3-of-5) was rejected as indistinguishable from noise, since a 3-2 split carries essentially no information about a consistent direction. This criterion is explicitly **a prespecified consistency filter, not an inferential test**: under a null of independent per-seed sign flips, at least 4 of 5 seeds landing on the majority side has probability 12/32 = 0.375, so agreement alone establishes nothing about significance. All inferential claims continue to come from Section 19's Holm-corrected patient-level bootstrap intervals on the sealed test evaluation; this criterion only decides whether the five-seed development-stage result is reported as directionally stable or directionally unstable. The conjunction with the sensitivity condition is deliberate: seed-to-seed agreement measures robustness to initialization and split randomness only, so without the sensitivity clause a result could be perfectly seed-stable and still reverse under a frozen analytic variation, and be reported as "stable". |
| 2026-09-07 | Run identity (`experiment_id`): `design.md` Section 2.2, implemented as `infra.config.ResolvedConfig.canonical_dict()` + `infra.registry.compute_experiment_id`; no numbered section of this document fixes the config *schema*, which is why this entry amends the identity rule rather than a numbered value | `ExperimentConfig` has exactly five fields — `experiment_name`, `seed`, `exploratory`, `witness_policy`, `tags` — so `canonical_dict()` (a `model_dump(mode="json")` of the whole `ResolvedConfig`) contains exactly those five under `experiment`, and `experiment_id = sha256(canonical_json(...))[:16]` is computed over that five-field payload. A second, parallel identity scheme also existed: `scripts/run_causal_analysis.py::_compute_content_experiment_id(dataset_checksum, trial_spec_version, master_seed, nc_bootstrap)`. | `ExperimentConfig` gains three fields, each defaulted to the primary/frozen setting: `variant: VariantOverrides = VariantOverrides()` (a frozen, `extra="forbid"` sub-model whose own eight fields all default to the primary settings — `propensity_model="logistic"`, `weight_clipping="primary_1_99"`, `support_thresholds="primary_0.10_0.90"`, `missingness_stress="none"`, `landmark_masking="none"`, `comparator_orientation="primary"`, `endpoint_horizon_weeks=96.0`, `selection_severity="primary"`); `model_variant: ExperimentVariantSpec or None = None`; and `negative_control_n_bootstrap: int = 1000` (the one `_compute_content_experiment_id` input not already inside `ResolvedConfig`, absorbed here so that second identity scheme can be deleted and `compute_experiment_id` becomes the single identity function). Because `canonical_dict()` serializes defaults, **every existing config, textually unchanged, now produces a `canonical_dict()` containing `variant: {all-primary values}`, `model_variant: null`, and `negative_control_n_bootstrap: 1000`, and therefore a different `experiment_id`.** This is a one-time, deliberate, whole-space **re-identification of every previously computed `experiment_id`**: no prior identifier survives the change, and none is expected to. Migration policy, frozen here: prior result directories under `results/<experiment_id>/` and prior `results/registry.jsonl` entries are **left in place, unmigrated, and un-renamed** — they are not recomputed, back-filled, deleted, or aliased to new identifiers. Any identifier minted **before this amendment date belongs to the pre-neural config schema** and must be read as such; identifiers minted on or after it belong to the post-amendment schema. The two populations are not one namespace: an old and a new directory for what is substantively the same run carry different names, and that difference is a schema-version signal, not a discrepancy to reconcile. (As of this entry no `results/<experiment_id>/` directory and no `results/registry.jsonl` exist in the working tree, so no migration is owed in practice; the policy is recorded because it governs any such artifact wherever one exists or is later recovered from a prior run.) | Requirement 9.9 of `.kiro/specs/neural-experiment-pipeline/requirements.md` requires that a rerun from an unchanged `ResolvedConfig` reproduce its content-hashed identifier, and Requirement 6.6 that two runs differing in any model- or variant-affecting field get different identifiers; the sensitivity variants and model variants that spec adds are exactly such fields, so they must be inside the hash. Per `design.md` Section 2.6 tension 3 of that spec, three options were considered and two rejected. (a) *Bypass the config* — pass the variant knobs as function arguments so the hash never sees them: **rejected**, because the knobs then do not enter `canonical_dict()`, so two materially different sensitivity runs would share one `experiment_id` and `ExperimentRegistry.register` would either raise `DuplicateExperimentError` on the second or overwrite the first; a silent collision between two different analyses is precisely the failure `compute_experiment_id` exists to prevent, and it is a worse failure than any amount of renaming. (b) *Relax `extra="forbid"`* so the knobs can be smuggled in as untyped extras: **rejected**, because `extra="forbid"` is the mechanism that makes a config typo loud, and trading a loud typo for a quiet one to preserve hash continuity buys stability in the identifiers at the cost of correctness in the configs. (c) **Chosen: extend `ExperimentConfig` with explicit, defaulted, validated fields and let the hash change.** Hash-inclusion is the honest outcome because the config space genuinely grew: a run pinned to the primary settings out of eight newly *expressible* alternatives is a different, more specific claim than a run made when those alternatives could not be expressed at all, and the identifier should say so. Excluding the new fields from the hash — or special-casing "all-default hashes as if absent" — would preserve the old identifiers by making the identifier stop being a function of the full resolved configuration, which is the one property it has. Recording the break here, with its date, is what makes it an auditable schema version rather than an unexplained discontinuity in `results/`. This amendment changes **no** frozen numeric value, threshold, formula, or named algorithm in this document: the primary settings are unchanged and are exactly what each new default encodes, so a post-amendment primary run computes the same numbers as a pre-amendment one under a different name. |
| 2026-09-30 | 12 (Overlap and support rules), third support condition; 13 (Causal graph and adjustment set), conformance of the implemented propensity model; 11.3 (Negative controls), scope reaffirmed | Section 12 reads "every **adjusted** baseline covariate has absolute standardized mean difference (SMD) < 0.10". Implemented in `causal/overlap.py::_smd` as the **raw, unweighted** pooled-SD SMD over `timeline.static_covariates`. `causal/overlap.py::_raw_numeric_covariate` returns `None` for absent or non-numeric values, and `smd_by_covariate` then omits that covariate with no entry. The propensity model (`causal.propensity.COVARIATE_FIELDS`) uses 6 covariates (`baseline_cd4`, `karnofsky_score`, `age`, `symptomatic_status`, `prior_opportunistic_infection`, binary `prior_art`), not the 13-covariate Section 13 set. | **(1) Weighted basis over the full adjustment set.** The Section 12 SMD condition is evaluated on **stabilized inverse-probability-weighted** covariate means and variances: numerator = marginal treatment prevalence (`P(A=1)/e` for arm 1, `P(A=0)/(1-e)` for arm 0), weights clipped with the Section 11 `"primary"` 1st/99th-percentile preset via the existing `causal.weighting.clip_weights`. It is evaluated over the **full Section 13 adjustment set**, not restricted to `COVARIATE_FIELDS`. The 0.10 threshold, the pooled-SD form, the four-condition structure of `support_valid`, the `[0.10, 0.90]` / `[0.05, 0.95]` bands, and the `extreme_weight_count` positivity proxy are **unchanged**. Raw SMDs remain reported as a descriptive diagnostic, labelled raw, and are never gated on. **(2) Propensity-model conformance to Section 13.** The propensity model's covariates are brought into conformance with the frozen Section 13 adjustment set. This is a conformance fix to a frozen decision, not a change to Section 13. Encodings for the categorical and non-numeric fields (`sex`, `race_ethnicity`, `intravenous_drug_use_history`, `prior_antiretroviral_exposure_and_duration`, and `weight`) are specified in the implementing task (7.5), are fit on development folds only, and the existing leakage audit must still pass. **(3) Section 11.3 kept exactly.** Baseline CD8 is excluded only from the right-hand side of the negative-control-outcome check. If that check uses any propensity-based quantity, it uses a separate propensity fit that excludes baseline CD8, so it cannot pass tautologically. In every other model, including the primary propensity model, baseline CD8 is an ordinary adjustment covariate. **(4) No silent omission.** Every Section 13 covariate appears in the balance report with either a weighted SMD value or an explicit, named exclusion reason. The current `_raw_numeric_covariate` → `None` → no-entry behavior is not allowed. A covariate with an exclusion reason and no weighted SMD cannot be counted as balanced. | The implemented quantity contradicts Section 12's own word "adjusted". Section 9's mechanism deliberately induces raw imbalance on six named covariates, so gating on raw balance makes the support condition fail exactly when the benchmark works as designed. On the canonical path, raw SMDs up to 0.3744 fall to at most 0.0427 under stabilized IPW on the modeled covariates (evidence table B of `.kiro/specs/neural-experiment-pipeline/proposed-decisions-section-26-amendment-smd-basis.md`; measured values in `results/diagnostics/overlap_smd_raw_vs_weighted.json`, regenerated by `tests/diagnostics/test_overlap_diagnosis.py`). The one residual weighted offender, `baseline_cd8` (-0.1323), is unmoved because the propensity model omits it, and it omits it because `COVARIATE_FIELDS` does not conform to Section 13, not because Section 11.3 forbids it (see "Correction: Section 11.3 does not forbid adjusting for baseline CD8" in that evidence file). Restricting the gate to `COVARIATE_FIELDS`, as the earlier draft proposed, would have hidden that non-conformance by excluding the covariates the model fails to adjust for. Requiring an entry or a named exclusion per covariate stops an unparseable covariate from looking the same as a perfectly balanced one. Scope: Section 12's body text is not edited (this entry fixes how its existing word "adjusted" is computed); Section 13 is **conformed to, not changed** — its 13-covariate list is unchanged and the code is brought into line with it; Section 11.3 is **kept exactly** — its text, the negative-control classification rules, and its thresholds are unchanged. No threshold, band, or condition in this document changes. Approved in direction by the spec owner on 2026-09-30; applied here by task 7.4 of `.kiro/specs/neural-experiment-pipeline/tasks.md`; code changes are tasks 7.5 and 7.6. |
| 2026-10-10 | Estimator implementation (no numbered section fixed it): the optimizers behind Section 11's primary logistic propensity model, Section 14.1's Cox PH baseline and the other Section 11 conventional estimators (`causal.baselines` Cox and outcome-logistic fits), Section 11.3's negative-control-outcome regression, and the calibration fit used by Section 19's metrics | Not frozen. Each fit was full-batch gradient descent from a zero start with a fixed budget and no stopping tolerance or convergence check: `LogisticPropensityModel` 500 steps at learning rate 0.1; `causal.baselines._fit_cox` and `_fit_logistic` 300 steps at 0.05; `causal.negative_controls._fit_linear_regression` 300 steps at 0.1; `evaluation.metrics._fit_logistic_1d` 500 steps at 0.1. | **Frozen solver rule** (`src/causal/solvers.py`): logistic fits by Newton/IRLS on mean binary cross-entropy; the linear fit by an exact least-squares (single Newton) solve; Cox fits by Newton-Raphson on the unchanged weighted Breslow partial log-likelihood (same tie handling, weights and `eta` clip of 30). All start at zero. The step is the minimum-norm Newton step (symmetric pseudo-inverse, relative eigenvalue cutoff 1e-10), halved up to 30 times if the objective does not improve. A fit **converges** when the accepted step's max absolute component is below **1e-8**, within at most **100** iterations. A fit that hits the cap, meets a singular Hessian, fails the line search or goes non-finite is **not converged**: its coefficients are never used. The primary estimate and the internal reference then raise `SolverConvergenceError`; a bootstrap replicate is dropped and counted in the replicate tally, as a NaN replicate already was; a calibration fit on perfectly separated predictions reports its slope and intercept as undefined (NaN). Every propensity fold records a `ConvergenceDiagnostic`. Objectives, encodings, feature order, adjustment sets, cross-fitting, clips and all thresholds are unchanged; only the optimizer changes. | The fixed budgets left fits far from their optima, and no frozen decision chose that. On the canonical development cohort (seed 20240101, n=383), every propensity fold stopped at gradient norm 6.7e-3 to 1.0e-2 with coefficients up to 1.09 from the optimum. The Cox fits were shrunk 40-50% toward the null: `unadjusted_cox` log-HR -0.377 shipped vs -0.641 converged (HR 0.686 vs 0.527; the Section 7 published reference is 0.50), `adjusted_cox` -0.499 vs -0.953, `ps_weighted_marginal_cox` -0.586 vs -1.007. `outcome_regression` was roughly halved (-0.083 vs -0.159). The primary IPCW-AIPW estimate was nearly unaffected (-0.158 vs -0.159), as double robustness predicts. The negative-control-outcome regression (-0.1144 vs -0.1150, `uncertain` both ways) and the calibration fit (slope 0.573 vs 0.583) were close to converged. Newton converges on this cohort in 4-6 iterations to gradient norm below 1e-16 (Hessian condition number 106-208, no separation); plain gradient descent would need about 21k-35k steps. Evidence: `.agents/tasks/propensity-convergence-investigation/findings.md` and `out/*.json`. **Sealed reference:** `cohorts.sealed_reference` computes the internal randomized reference with the same `unadjusted_cox`, so the reference changes with this fix. No sealed artifact had been written to disk and no locked evaluation had been consumed (`results/locked_evaluation/` does not exist), so this recomputes an in-memory value before it was ever sealed or read. It does not reopen a sealed result. **Not changed:** the prior-ART duration encoding from the Section 13 conformance amendment. Under converged fits one control patient remains outside the [0.10, 0.90] primary band (propensity 0.933; Karnofsky 4.4 SD below the mean, long prior-ART exposure), so `no_unresolved_positivity_warning` fails. This is reported as a finding (Requirement 19), not engineered away. **Section 11.3 conformance fix found while re-measuring:** the negative-control-outcome regression looked up its right-hand-side covariates by raw Section 13 name, so `weight`, `sex`, `race_ethnicity`, `intravenous_drug_use_history` and prior-ART exposure and duration resolved to nothing and entered as all-zero columns (unadjusted). They are now encoded through the shared Section 13 mapping (`causal.adjustment_set`), with `baseline_cd8` still excluded from its own right-hand side exactly as Section 11.3 requires. The adjusted SMD moves from -0.1150 (`uncertain`) to -0.0906 (`pass`, CI about [-0.26, 0.10]), stable across all five frozen seeds at both 200 and 1000 replicates and on both seed paths. The 0.10 threshold and the classification rules are unchanged. Implemented by neural-experiment-pipeline tasks 7.7-7.9. |
