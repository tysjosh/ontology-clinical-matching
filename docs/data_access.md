# Data Access Policy — ACTG175

**Governs:** Requirement 2 (`requirements.md`, "Clinical-data privacy and security"), Acceptance
Criterion 2.1 ("Before data access, the project SHALL document the approved storage location,
access rules, data-use agreement, and permitted derived artifacts").

**Status:** Must be read and satisfied before any raw ACTG175 file is copied onto a development
machine or into this repository's working tree. This document is the policy that
`scripts/scan_exports.py` (task 2.1) and the `min_cell_size` checks in `src/reporting/tables.py`
and `src/reporting/figures.py` (task 2.2) implement mechanically. If those implementations ever
diverge from this document, this document governs and the code is the bug.

---

## 1. Dataset identity

- Dataset: `LongCART::ACTG175`, version `3.2` (frozen in `decisions.md` Section 2).
- Data integrity: a SHA-256 checksum is computed at ingestion and frozen in the dataset contract
  (`configs/datasets/actg175_v3_2.yaml`, task 5). The pipeline refuses to run on a mismatched
  checksum (Requirement 5.3-5.4).
- Primary trial source / provenance: Hammer SM, et al., *NEJM* 1996, PMID
  [8813038](https://pubmed.ncbi.nlm.nih.gov/8813038/), DOI `10.1056/NEJM199610103351501`
  (`decisions.md` Section 24).
- ACTG175 is a historical, de-identified clinical-trial dataset. It contains no free text, no
  imaging, and no direct patient identifiers (name, MRN, address, exact date of birth) in its
  packaged form. It does contain quasi-identifying and sensitive baseline covariates (age, sex,
  race/ethnicity, hemophilia status, IV drug-use history, homosexual activity) that must be
  treated as protected clinical data under this policy regardless of the absence of direct
  identifiers.

## 2. Approved storage location

| Content | Location | Committed to git? |
|---|---|---|
| Raw ACTG175 tables (as distributed) | `data/raw/` | No — gitignored (`.gitignore`: `data/raw/`) |
| Derived patient-level records, timelines, `ClinicalEvent`s, split membership, checkpoints trained on real patient data, patient-level predictions | `results/<experiment_id>/` | No — gitignored (`.gitignore`: `results/*`, only `.gitkeep` is tracked) |
| Dataset contract, target-trial spec, experiment configs (no patient rows, only schema/metadata/frozen parameters) | `configs/` | Yes |
| Aggregate, privacy-scanned, minimum-cell-size-enforced tables/figures/reports intended for the paper | `results/<experiment_id>/report/` (written by `src/reporting/`) | Only after passing the export scan (task 2.1) and min-cell-size checks (task 2.2); copies destined for the paper source live outside this repo's git history unless explicitly cleared |

**Protected storage**, for the purposes of this policy and of every later access-control check in
the codebase, means exactly the two gitignored locations above: `data/raw/` and
`results/<experiment_id>/` (all subpaths). Nothing that has not passed the privacy scanner
(`PrivacyScanner.scan`, task 1.6) and, where applicable, the minimum-cell-size check (task 2.2)
may leave protected storage.

Both locations are local-filesystem-only. Neither may be placed in, or synced through, cloud
storage, personal drives (Dropbox/Google Drive/iCloud/OneDrive), email, chat tools, shared
network drives without equivalent access controls, or any managed-notebook environment that
persists data outside this repository's approved machine.

## 3. Unauthorized filesystem paths (definition for `scan_exports.py` / `PrivacyScanner`)

An **unauthorized filesystem path**, for the purposes of the export scanner (task 2.1) and the
manifest privacy scanner (task 1.6), is any path that is not one of:

- `data/raw/**` (raw data, read-only after ingestion, never written to by pipeline stages);
- `results/<experiment_id>/**` for a valid, registered `experiment_id` (all pipeline-written
  outputs, including manifests, checkpoints, predictions, and reports);
- `configs/**`, `src/**`, `scripts/**`, `tests/**`, `docs/**` (code and schema/metadata only — these
  must never contain patient rows, checkpoints, or predictions).

Concretely, any of the following is a violation the scanner must flag:

- a home-directory path outside the repository root (e.g. `~/Downloads`, `~/Desktop`, `/tmp`);
- any absolute path referencing a cloud-sync folder;
- any path under `configs/`, `src/`, `scripts/`, `tests/`, or `docs/` that contains a per-patient
  record, a patient identifier, or an unaggregated prediction;
- any manifest, log, checkpoint, or report field containing an absolute path outside the
  repository root (this also catches accidental leakage of a developer's local machine layout).

## 4. Access rules — who and how

- Only project contributors listed as approved analysts for this study may read `data/raw/` or
  patient-level content under `results/`. Access is granted per person, not per machine.
- Access requires the contributor to have agreed to the data-use terms in Section 5 before the
  raw files are copied onto their machine.
- Data may be accessed only on a machine the contributor controls and that is not shared with
  unapproved users; no remote/shared compute access without an equivalent per-user access
  control list.
- No raw or patient-level derived data is ever transmitted over email, chat, or ticketing
  systems. If a contributor needs to share a derived artifact with another approved contributor,
  it is shared by copying within protected storage (Section 2) on a shared approved machine, not
  by exporting it elsewhere.
- Any credential or token needed to obtain the dataset (e.g. an access-portal API key) is loaded
  through an approved secret mechanism (environment variable or secret store) and is never
  written to a config file, log, or the git history — this is implemented in task 2.1.
- Before any artifact is copied out of protected storage, it must pass the automated export scan
  (`scripts/scan_exports.py`, task 2.1) and, if it is a public table or figure, the
  minimum-cell-size check (task 2.2).
- A privacy review is completed before any artifact is released beyond the approved contributor
  group (Requirement 2.9) — at minimum, confirmation that the export scan and min-cell-size check
  both passed for that artifact.

## 5. Data-use agreement

ACTG175 is distributed as a historical, de-identified clinical-trial dataset (via the packaged
`LongCART::ACTG175` distribution referenced in Section 1). Whatever license or data-use agreement
accompanies the specific `LongCART::ACTG175` v3.2 distribution actually obtained governs this
project's use of the data, and its terms take precedence over anything in this document if they
are stricter.

Before any raw file is copied into `data/raw/`, the contributor obtaining access must:

1. record the specific access channel used (e.g. archive/repository name, accession or request
   ID, and date obtained) in this section; and
2. confirm the accompanying license/data-use terms permit the uses described in Section 6 below
   (research use, derived-model training, and publication of aggregate results).

**Access channel / DUA record for this project:** Obtained 2026-09-07 from the official CRAN
source distributions `LongCART_3.2.tar.gz` (the frozen project source) and
`speff2trial_1.0.5.tar.gz` (the upstream source named by `LongCART`'s ACTG175 documentation), via
`https://cran.r-project.org/src/contrib/`. No account, accession request, credential, or
restricted-data agreement was required. The packages declare GPL-compatible terms
(`LongCART`: GPL >= 2; `speff2trial`: GPL-2), which permit research use, derived-model training,
and publication of aggregate results. This project's stricter non-redistribution and protected-
storage rules continue to govern the extracted patient-level data.

**Ingestion record:** The protected source archives are stored under
`data/raw/source_packages/`. Their SHA-256 hashes are
`c155d40a5da5f680beeb039eed52cece5fdb301f0ad91bd049ca89b238a15f16`
(`LongCART_3.2.tar.gz`) and
`ecb3579670c360786da55346f783091f6b71bf6a49a920ab1224d254cd55e4a1`
(`speff2trial_1.0.5.tar.gz`). The exact upstream ACTG175 text table has SHA-256
`7b52e6b3701a1f24090ef0388f1c148d5286450f97276d9c8fbe611ce0c76779`; the
exact `LongCART` R data object has SHA-256
`92261b027a14a000130c3f0e7a0626192310e6ae27b4f0d1de13ead3d68f9e95`.
It was deterministically reshaped into the six loader-compatible tables under
`data/raw/actg175_v3_2/`; their loader checksum is frozen in the dataset contract.

The source has CD8 at baseline and week 20 but does not provide week-96 CD8, so every week-96
CD8 value is explicitly missing. It provides the composite event indicator/time but not the
component event subtype or a specific censoring reason; compatibility fields therefore use
`composite_event_unspecified` and `administrative_or_last_observation`, respectively. These
labels record source limitations and must not be interpreted as more granular observations.

Regardless of the specific license terms, this project additionally commits to:

- using the data only for the research described in `requirements.md` / `decisions.md`;
- never attempting re-identification of any patient;
- never redistributing the raw or patient-level derived data to anyone outside the approved
  contributor group;
- following the derived-artifact rules in Section 6 for anything that leaves protected storage.

## 6. Permitted derived artifacts

**May leave protected storage** (subject to passing the export scan and, where applicable, the
minimum-cell-size check):

- aggregate statistics: cohort counts, cohort-flow-table counts, treatment/endpoint prevalence,
  missingness rates, overlap/balance diagnostics, effective sample sizes;
- cohort-level treatment-effect estimates and their confidence intervals;
- model configuration, hyperparameters, and the frozen formulas/coefficients from `decisions.md`;
- witness pass/fail/uncertain outcomes and rule-level evidence that does not include patient
  identifiers;
- tables and figures built from the above, with the minimum-cell-size policy enforced on every
  reported cell (Requirement 2.5, task 2.2);
- environment/version metadata (`pip freeze` snapshot, source commit, config hash);
- the audit report (Requirement 11.10), provided it contains only the above categories.

**Must remain in protected storage and never leave it:**

- raw ACTG175 records and any field taken directly from them;
- `ClinicalEvent`-level records and patient timelines;
- direct or quasi-identifier values (age, sex, race/ethnicity, hemophilia status, IV drug-use
  history, homosexual activity, or any combination of covariates small enough to be
  re-identifying) at the individual level;
- patient-level predictions, potential-outcome estimates, and uncertainty scores (Requirement
  2.6) — these are reported only in aggregate or subgroup form with the minimum-cell-size policy
  applied;
- model checkpoints trained on real patient data (checkpoints are registered in the manifest and
  stored under `results/<experiment_id>/`, not exported);
- the sealed internal randomized reference artifact, until the locked external-evaluation stage
  (`decisions.md` Section 8), and even then only its resulting aggregate estimate — never
  patient-level data — is eligible for export;
- any raw free text or debugging log containing the above (Requirement 2.3).

## 7. Relationship to later enforcement tasks

This document is descriptive policy; the following tasks make it mechanically enforced and must
not contradict it:

- **Task 2.1** (`scripts/scan_exports.py`, secret handling): implements the automated identifier
  and secret scan referenced in Sections 3 and 4, and the approved-secret-mechanism requirement
  in Section 4.
- **Task 2.2** (`min_cell_size` in `src/reporting/tables.py` / `figures.py`): implements the
  minimum-cell-size suppression/aggregation referenced in Section 6.
- **Task 1.6** (`PrivacyScanner`, already implemented): blocks manifest writes that violate
  Section 3's unauthorized-path definition or expose the identifiers listed in Section 6.
