# Synthetic ACTG175-shaped fixture

**This directory contains fabricated, synthetic data. It is not real ACTG175
patient data and must never be treated as such.**

Real ACTG175 raw tables are not available in this development environment
(`docs/data_access.md` Section 5: the access-channel/DUA record for this
project is "not yet recorded"). `tests/unit/test_actg175_loader.py` and any
other test exercising `src/data/loaders/actg175.py` therefore load this
fixture instead of `data/raw/` (which is gitignored and, per
`docs/data_access.md`, only ever holds real, protected patient data).

The six CSV files here mirror the *shape* of the real ACTG175 raw tables
described in `decisions.md` (demographics, treatment arm, baseline
covariates, longitudinal CD4/CD8, censoring, outcomes) with six fabricated
patients (`SYN-001` through `SYN-006`). Every value was invented for testing
purposes only:

- `demographics.csv`
- `treatment_arm.csv` (arms 0-3 represented, matching the full raw trial
  population before the arm-0/arm-1 analysis restriction that happens
  downstream in cohort construction, not in this loader)
- `baseline_covariates.csv`
- `longitudinal_cd4_cd8.csv` (baseline/week-20/week-96 landmark rows per
  patient, with one intentionally missing `cd8` cell to exercise the
  nullable-field path)
- `censoring.csv`
- `outcomes.csv`

Do not add real patient data to this directory.
