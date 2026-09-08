# Responsible use

**Governs:** Requirement 15.7 ("Responsible-use documentation SHALL cover
confounding, privacy, historical bias, subgroup performance, unsupported
individual recommendations, misuse as an RCT replacement, uncertainty,
clinical oversight, and auditability"), Requirement 15.4 (model card
content, implemented in `src/reporting/model_card.py`), `decisions.md`
Section 22 ("Permitted and prohibited claims") and Section 23 ("Final scope
freeze").

**Status:** This document, together with the model card
(`src/reporting/model_card.py`) and `docs/limitations.md`, is the
responsible-use documentation required before any paper artifact leaves
protected storage. If a figure, table, or manuscript sentence contradicts
this document, this document governs and the artifact is the bug.

---

## 1. Confounding

The pseudo-observational cohort used to evaluate causal recovery is not
confounded by nature -- it is deliberately confounded by a fixed,
documented mechanism (`decisions.md` Section 9, "Controlled
observationalization"): patients are retained or dropped according to a
frozen logistic selection rule that depends on treatment arm and a baseline
risk score, applied independently to development, validation, and test
splits with distinct, deterministic per-split seeds.

This has two consequences for anyone reading or reusing this project's
results:

- The confounding is *known and by construction*, not estimated from the
  data. Any claim about "recovering the treatment effect under
  confounding" refers specifically to recovering it under this documented
  mechanism -- it is not evidence that the pipeline would perform
  comparably under an arbitrary, unknown real-world confounding structure.
- The mechanism's coefficients, seeds, and per-split retained-patient lists
  are frozen artifacts (`decisions.md` Section 9, "Frozen artifacts") and
  must not be re-tuned after seeing validation or test results. A pipeline
  that appears to "recover" the treatment effect only because the
  observationalization mechanism was adjusted after the fact would not be
  demonstrating anything about confounding robustness.

The internal randomized reference (`decisions.md` Section 8, item 2) is
sealed before observationalization is applied, precisely so that the
comparison between the model's cohort-level estimate and the true
randomized effect is not contaminated by knowledge of how the confounding
was generated.

**Limit:** a single documented observationalization mechanism, calibrated
once on development data (`decisions.md` Section 9, "One-time pilot
recalibration"), demonstrates recovery under *that* confounding structure.
It does not demonstrate recovery under confounding mechanisms this project
did not test, including unmeasured confounding, which is out of scope for
the primary analysis (`decisions.md` Section 14.9 references an
unmeasured-confounding sensitivity analysis only "when defensible").

## 2. Privacy

ACTG175 is a historical, de-identified clinical-trial dataset, but it
contains quasi-identifying and sensitive baseline covariates (age, sex,
race/ethnicity, hemophilia status, IV drug-use history, homosexual
activity) that this project treats as protected clinical data regardless
of the absence of direct identifiers. See `docs/data_access.md` for the
full policy: approved storage locations, access rules, the data-use
agreement, and the list of permitted vs. protected derived artifacts.

In short, for anyone consuming this project's outputs rather than its raw
data:

- Nothing that has not passed the automated export scan
  (`scripts/scan_exports.py`) and, where applicable, the minimum-cell-size
  check (`src/reporting/tables.py::apply_min_cell_size`,
  `src/reporting/min_cell_size.py`) may leave protected storage.
- Patient-level predictions and potential-outcome estimates are never
  reported individually -- only in aggregate or minimum-cell-size-enforced
  subgroup form (`docs/data_access.md` Section 6).
- Secrets and credentials follow `docs/secrets.md` and are never present in
  any artifact this document governs.

## 3. Bias

This benchmark is built from one historical randomized trial (ACTG175,
1990s-era HIV antiretroviral therapy), not a contemporary or diverse
real-world population. Two forms of bias follow directly from that choice:

- **Historical bias.** ACTG175's enrolled population, standard of care, and
  treatment regimens reflect HIV care as practiced at the time of the
  trial. Findings should not be read as generalizing to current clinical
  populations, current standards of care, or other diseases.
- **Subgroup limitations.** The frozen primary robustness analysis
  (`decisions.md` Section 20) reports results by prior antiretroviral-
  treatment status (naive vs. experienced) with sample size, treatment
  counts, endpoint counts, overlap, and uncertainty attached to every
  subgroup result (Requirement 13.3). Small or unsupported subgroups are
  suppressed or marked uncertain rather than reported
  (`src/evaluation/robustness.py::SubgroupRobustness`, Requirement 13.4).
  Cross-site and calendar-time-shift robustness are explicitly out of
  scope for this dataset (`docs/limitations.md`) -- their absence is a
  reported limitation, not evidence of robustness across sites or time.

## 4. Unsupported individual recommendations

This is a cohort-level causal-recovery benchmark, not an individual-patient
decision-support tool. Every potential-outcome prediction the
intervention-aware model produces for a given patient
(`src/models/aggregation.py::PatientLevelEstimate`) is explicitly labelled
a *model-dependent estimate*, never an observed individual treatment effect
(`decisions.md` Section 7, "Patient-level predicted potential-outcome
differences are model estimates and must never be reported or described
as observed individual treatment effects"; Requirement 9.7). No component
of this system recommends a treatment for a specific patient, and none of
its outputs should be used to do so. This is consistent with the
out-of-scope list in `decisions.md` Section 23: "individualized treatment
recommendations" and "regulatory decision support" are both explicitly
excluded from this project.

## 5. Misuse as an RCT replacement

`decisions.md` Section 22 ("Prohibited") freezes this claim as prohibited,
verbatim:

> claiming the system replaces randomized trials

This system does the opposite of replacing a randomized trial: its central
comparison is between a model-estimated, observational-cohort treatment
effect and a sealed, independently-computed *randomized* reference effect
from the same underlying trial (`decisions.md` Sections 7-8). A favorable
result demonstrates that a given causal-recovery method can approximate a
known randomized effect under *known, controlled confounding*
(`decisions.md` Section 22, "Permitted": "the benchmark evaluates causal
recovery under known, controlled confounding") -- it does not demonstrate
that observational modeling can substitute for running a trial in a
setting where the true effect is unknown. Two related prohibited claims
from the same section reinforce this boundary and must also never appear
in any artifact generated from this project:

> claiming the cohort is independent real-world evidence
>
> claiming ACTG 175 demonstrates modern clinical-deployment readiness

## 6. Uncertainty

The primary uncertainty method is a 5-replica deep ensemble
(`src/uncertainty/ensemble.py::DeepEnsemble`), whose variance is reported as
*epistemic* uncertainty over the model's own predictions, calibrated on
validation data only (`src/uncertainty/calibration.py::Calibrator`,
`decisions.md` Section 15). The patient-support gate
(`src/uncertainty/support_gate.py::PatientSupportGate`) combines that
uncertainty with propensity-based common support, protocol validity, and
absence of hard clinical/temporal violations to decide whether a given
patient's prediction is supported, always attaching machine-readable
reasons (Requirement 10.9-10.10).

What this ensemble uncertainty **can** tell you: how consistent the
model's own point predictions are across independently initialized
replicas trained on the same data, and, via the support gate, whether a
given patient falls inside the region where the model and the causal
design are jointly well-supported.

What it **cannot** tell you: it is not a measure of confidence in an
*individual counterfactual outcome* being correct (`decisions.md` Section
22, "Prohibited": "claiming individual counterfactual outcomes are ground
truth"), and it is not a causally valid interval for an individual
treatment effect unless a causal conformal method and its assumptions are
explicitly implemented, which this project does not do (Requirement 10.6).
Cohort-level treatment-effect intervals instead use patient-level bootstrap
resampling (`src/uncertainty/bootstrap.py`, 1,000 replicates,
`decisions.md` Section 15), a separate mechanism from the ensemble's
epistemic-uncertainty estimate.

## 7. Clinical oversight

This system has no autonomous clinical role. `decisions.md` Section 23
("Final scope freeze," out-of-scope list) explicitly excludes autonomous
clinical or protocol-design agents, individualized treatment
recommendations, regulatory decision support, prospective clinical
validation, and production or regulatory deployment. The clinical-
constraint witness (`src/witnesses/clinical.py::ClinicalWitness`) checks an
explicit, auditable set of protocol and eligibility rules -- it is
deliberately named a *clinical-constraint* witness rather than a claim of
comprehensive clinical plausibility, per Requirement 11.7, because no
qualified clinician review of its rule set has been performed. Any use of
this system's outputs in an actual care setting, by a clinician or
otherwise, is outside its intended use and is explicitly a prohibited
deployment context (see `src/reporting/model_card.py`'s `intended_use`
field).

## 8. Auditability

Every released cohort-level estimate is accompanied by a structured audit
report (`src/witnesses/audit_report.py::build_audit_report`,
`AuditReport.to_dict()`, Requirement 11.10) containing the protocol
version, the (supported) target population and treatment contrast, the
point estimate and interval, overlap diagnostics, the pass/fail/uncertain
outcome of each of the three witnesses (temporal, causal, clinical),
support-restriction (abstention) reasons, protocol violations, and
provenance (experiment ID, seed, commit, timestamp). Nothing in this audit
report is computed by the report itself -- it is a pure bundling of
already-computed, already-frozen artifacts from earlier pipeline stages,
so that any released estimate can be traced back to the exact
configuration, data version, and gating decisions that produced it without
re-running the pipeline.
