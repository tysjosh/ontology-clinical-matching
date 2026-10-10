# Investigation Plan: propensity under-convergence and the canonical support failure

Read-only investigation. It produces findings and a recommendation; it does not change code, configs, `decisions.md`, or any spec task status. A fix is a separate, user-approved step after `findings.md` is reviewed.

## Question

After task 7.5 widened `LogisticPropensityModel` to the 14-column Section 13 design, the canonical development cohort fails support: `extreme_weight_count` went 0 -> 1, ESS dropped to 184.74 / 144.40, `support_valid = False`. The fit is pure-Python batch gradient descent (zero init, `LEARNING_RATE = 0.1`, `N_ITERATIONS = 500`, no tolerance) and is measurably unconverged (gradient norm 6.7e-3 to 1.0e-2 per fold, loss still falling). Is the support failure an under-fitting artifact or a real overlap signal, and which other fixed-budget fits in the repo share the defect?

## Conventions

- `R=/Users/olukotunjosh/Downloads/clinical`, `PY="env PYTHONPATH=src /Users/olukotunjosh/Downloads/SADL/.venv/bin/python"` (torch 2.4.1, numpy 1.26.4). Run from `R`.
- `T=/Users/olukotunjosh/Downloads/clinical/.agents/tasks/propensity-convergence-investigation`. Every script lives in `$T/scripts/`; every output lives in `$T/out/`. Nothing is written anywhere else.
- Every command is capped: `perl -e 'alarm shift; exec @ARGV' 900 <cmd>`. Earlier runs died at a 60-minute agent limit, so each item is sized to finish in well under 30 minutes.
- Canonical path: master seed `20240101`; `derive_seed(master_seed, "causal_analysis_pipeline")` for the cohort and `derive_seed(master_seed, "propensity")` for the model, exactly as `scripts/run_causal_analysis.py` (~lines 311, 327); 5-fold cross-fitting; Development_Split only. Baseline: n=383, arms 201/182.
- Sealed: never call `require_test_split`, `unseal`, or `read_sealed_artifact`; never read the Test_Split or sealed reference.
- No package installs. numpy may be used in `$T/scripts/` for reference solves only.
- Git: read-only (`git show`, `git diff`). Never commit, stash, reset, checkout, or clean. The working tree carries uncommitted task 7.4/7.5/10 work that must stay byte-for-byte.
- Patient identifiers appear in outputs only as a pseudonymous index or a SHA-256 prefix.
- Every item appends what it ran (command, wall time, result) to the Verification log at the end of this file and to its `features/INV-00n.json` `findings`.

## Dependency overview

| items | depend on | independent of |
|---|---|---|
| 1 (harness) | — | — |
| 2-4 (INV-001: propensity convergence and support) | 1 | 5-6 |
| 5-6 (INV-002: other fixed-budget fits) | 1 | 2-4 |
| 7 (INV-003: synthesis) | 2-6 | — |

INV-001 and INV-002 can run in parallel once item 1 is done.

---

## INV-000: shared harness

- [x] 1. Build the canonical-cohort harness.
      - `$T/scripts/harness.py`: load data, build the canonical dev-retained cohort, and expose the 5 cross-fitting folds with the shipped standardization and 14-column encoding (reuse `causal.propensity` / `causal.adjustment_set` functions; do not reimplement encodings).
      - Assert n=383 and arms 201/182; print them. Record wall time.
      - Done when: `$PY $T/scripts/harness.py` prints the counts and exits 0 in under 60 s.

## INV-001: propensity convergence and its effect on support

- [x] 2. Convergence sweep.
      - Refit every fold at 500 (shipped), 1000, 2000, 5000, 20000 gradient-descent iterations, plus a reference Newton/IRLS solve to gradient norm < 1e-8.
      - Per fold and setting: iterations, final gradient norm, final loss, max |w|, IRLS iterations, Hessian condition number, and any quasi-separation sign (coefficients growing without bound). Watch `prior_opportunistic_infection` (non-zero for 7/383).
      - Report iterations plain gradient descent needs to reach gradient norm 1e-6.
      - Output: `$T/out/convergence_sweep.json` and a compact table.
- [x] 3. Support under each fit.
      - Feed each setting's cross-fitted scores to the existing `causal/overlap.py` functions: propensity range, per-arm ESS, `extreme_weight_count`, patients outside [0.10, 0.90] and [0.05, 0.95], the four support booleans, `support_valid`. State that the SMD condition is still raw (task 7.6 not done) and so does not depend on the scores.
      - For each out-of-band patient: pseudonymous id, arm, encoded covariates, and the coefficient x covariate terms driving the logit, under the shipped and converged fits.
      - Output: `$T/out/support_by_fit.json`.
- [x] 4. Six-covariate comparison.
      - Reconstruct the pre-7.5 model with `git show HEAD:src/causal/propensity.py > $T/scripts/propensity_head.py` and import it from there. Repeat items 2-3 for 500 iterations vs converged.
      - Answers: were the earlier results (ESS 187.23 / 161.00, `extreme_weight_count` 0) also an under-fitting artifact?
      - Output: `$T/out/six_covariate_comparison.json`.

## INV-002: other fixed-budget gradient-descent fits

- [x] 5. Inventory.
      - Grep `src/` for fixed-iteration optimizer loops (`N_ITERATIONS`, `LEARNING_RATE`, `n_iter`, `for _ in range(` near a gradient). Verify the reported candidates: `causal.negative_controls._fit_cd8_regression_smd` (300 / 0.1), `causal.baselines` Cox and logistic fits (300 / 0.05), `models.non_neural_baseline.CoxBaseline`, `evaluation.metrics` calibration (500 / 0.1).
      - Output: `$T/out/fit_inventory.md` (file, function, budget, learning rate, objective, what reported number it feeds).
- [x] 6. Convergence of each fit on canonical inputs.
      - Mirror each loop exactly in `$T/scripts/` to log the final gradient norm and loss; compute a converged reference on the same objective.
      - Report the change in the reported quantity: treatment coefficient / hazard ratio, IPCW-AIPW estimate, the CD8 negative-control SMD and its Section 11.3 classification (point estimate; at most 50 bootstrap replicates if a CI is needed), calibration slope/intercept. If calibration has no canonical dev input, use a labelled synthetic input.
      - Verdict per fit: converged / materially unconverged (changes a reported number or classification) / unconverged but immaterial.
      - Output: `$T/out/other_fits.json`.

## INV-003: synthesis

- [x] 7. Write `$T/findings.md`.
      - Verdict on the support failure: under-fitting artifact, or genuine overlap signal that the converged fit confirms or worsens; whether converging helps or hurts ESS.
      - Recommended optimizer and stopping rule (Newton/IRLS vs gradient descent with tolerance; ridge only with separation evidence), with a draft `decisions.md` Section 26 amendment row (not applied).
      - Per-fit verdicts from item 6 and which need follow-up.
      - Impact on task 7.6 of `.kiro/specs/neural-experiment-pipeline/tasks.md`.
      - Everything not measured, stated plainly.

## Stop conditions (ask the user, do not work around)

- The harness cannot reproduce n=383 / arms 201/182.
- Any step would need the Test_Split, the sealed reference, a package install, or an edit outside `$T`.
- IRLS fails to converge for reasons other than separation.

## Verification log

(append entries: item, command, wall time, result)

- Item 1 (2026-09-30): `PYTHONPATH=src:$T/scripts $PY $T/scripts/harness.py` -> n=383, arms {1: 201, 0: 182}, 14 columns, wall 0.48 s. PASS. Also extracted `git show HEAD:src/causal/propensity.py` to `$T/scripts/propensity_head.py` for item 4 (6 columns, LR 0.1, 500 iterations).
- Items 2-3 (2026-09-30): `$T/scripts/sweep.py` (numpy mirror of `_fit`; max |diff| vs shipped 500-iteration scores 1.1e-16), wall 4.6 s. Outputs `out/convergence_sweep.json`, `out/support_by_fit.json`. 500 iterations: grad norm 6.7e-3 to 1.0e-2, loss gap 2.8e-3 to 7.4e-3, coefficients up to 1.09 from the IRLS optimum. IRLS converges in every fold (grad norm < 1e-16, cond(H) 106-208, no separation). GD needs 21,069-35,086 iterations to reach grad norm 1e-6. Support: extreme_weight_count = 1 under every setting (same the out-of-band patient, arm 0, p 0.9305 -> 0.9327); ESS falls from 184.74/144.40 to 179.55/138.47 when converged; inside the [0.05, 0.95] sensitivity band throughout; support_valid False throughout.
- Item 4 (2026-09-30): `$T/scripts/six_cov.py`, mirror error 2.2e-16. HEAD 6-covariate model was also unconverged (grad norm 2.6e-3 to 6.3e-3, coefficients up to 1.08 off). Converged: ESS 185.28/157.75 (vs 187.23/161.00), extreme_weight_count 0 both ways. Patient the out-of-band patient sits at 0.876 (500) / 0.869 (IRLS) under 6 covariates. Output `out/six_covariate_comparison.json`.
- Item 5 (2026-09-30): grep of `src/` for fixed-iteration loops; five fits found. Output `out/fit_inventory.md`.
- Item 6 (2026-09-30): `$T/scripts/other_fits.py`, real functions with budgets patched to 1x/10x/100x, wall about 125 s. Cox log-HRs shrunk ~40-50% at the shipped 300 iterations (unadjusted -0.377 vs -0.641); outcome regression -0.083 vs -0.159; IPCW-AIPW -0.158 vs -0.158; CD8 NCO -0.1144 vs OLS -0.1150, `uncertain` both ways (50 bootstrap); calibration slope 0.573 vs 0.583. `cohorts.sealed_reference` uses the same `unadjusted_cox` (code check only). Output `out/other_fits.json`.
- Item 7 (2026-09-30): `findings.md` written.
