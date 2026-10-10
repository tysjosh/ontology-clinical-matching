# Findings: propensity under-convergence and the canonical support failure

Read-only investigation on the canonical development cohort (master seed 20240101, canonical `derive_seed` path, 5-fold cross-fitting, n=383, arms 201/182). No repo file, config, spec or task status was changed. Test_Split and sealed reference were not opened. Evidence: `out/*.json`, scripts in `scripts/`, run log in `plan.md`.

## 1. The support failure is real, not an under-fitting artifact

- The shipped 500-iteration propensity fit is unconverged in every fold (gradient norm 6.7e-3 to 1.0e-2; coefficients up to 1.09 from the optimum; e.g. fold 0 hemophilia 0.22 vs 1.09). Newton/IRLS converges cleanly: condition number 106-208, no separation, `prior_opportunistic_infection` finite at about -1.3.
- One patient (arm 0) is above the 0.90 primary bound under every budget and under the converged fit (0.9305 -> 0.9327). It is inside the [0.05, 0.95] sensitivity band throughout.
- Driver: Karnofsky score 4.4 SD below the fold mean (+1.56 on the logit), plus a long prior-ART duration (z = +2.94, +0.58 to +0.78) and homosexual activity. Under the old 6-covariate model the same patient was at 0.876, so **the task 7.5 widening, specifically the new prior-ART duration column, pushed this patient out of band.** Converging does not.
- Converging makes overlap slightly worse: ESS 184.74/144.40 -> 179.55/138.47 (14 columns) and 187.23/161.00 -> 185.28/157.75 (6 columns). All well above the 100 floor.
- So on the converged 14-column model, `no_unresolved_positivity_warning` still fails (extreme_weight_count = 1), and `support_valid` stays False independently of the raw-SMD question.

## 2. The bigger problem: the Cox fits are badly unconverged

| estimate | shipped (300 iter) | 3,000 | 30,000 | verdict |
|---|---|---|---|---|
| `unadjusted_cox` log-HR | -0.377 (HR 0.686) | -0.641 | -0.641 (HR 0.527) | materially unconverged |
| `adjusted_cox` log-HR | -0.499 | -0.947 | -0.953 | materially unconverged |
| `ps_weighted_marginal_cox` log-HR | -0.586 | -0.957 | -0.957 | materially unconverged |
| `CoxBaseline.treatment_arm` | -0.499 | -0.947 | -0.953 | materially unconverged |
| `outcome_regression` RD | -0.083 | -0.152 | -0.159 | materially unconverged |
| `ipcw_aipw` RD (primary) | -0.158 | -0.156 | -0.158 | unconverged but immaterial (doubly robust) |
| `ipcw_ipw` RD | -0.166 | -0.166 | -0.166 | no fit involved |

- Every Cox log-HR is shrunk toward the null by roughly 40-50%. Outcome regression is shrunk by about half. The primary IPCW-AIPW estimate is nearly unaffected.
- `cohorts.sealed_reference.unadjusted_cox_reference_estimator` calls the same `unadjusted_cox`, so **the sealed internal-reference log-HR was very likely computed with the same unconverged fit.** I confirmed this from the code only; I did not open the sealed reference, so its stored value is unmeasured. Every log-HR error, direction-agreement and CI-overlap metric against that reference inherits the problem.

## 3. Fits that are fine

- Negative-control-outcome CD8 regression: -0.1144 shipped vs -0.1150 exact OLS; classification `uncertain` either way (50-replicate check, CI about [-0.28, 0.06]). Converged enough.
- Calibration fit: slope 0.573 vs 0.583 on a representative real input, unchanged on synthetic. Immaterial.

## 4. Recommendation

1. Replace every fixed-budget fit above with Newton/IRLS (logistic, linear) or Newton-Raphson (Cox partial likelihood), pure Python, deterministic, stopping on a frozen tolerance (e.g. max |step| < 1e-8) with a hard cap (e.g. 100 iterations) and a recorded per-fit convergence diagnostic. Plain GD would need about 21k-35k iterations for the propensity model alone. No ridge is needed for the propensity model on this cohort.
2. Freeze that rule with one `decisions.md` Section 26 row. Draft (not applied):

   | Date | Section | Prior value | New value | Reason |
   |---|---|---|---|---|
   | _on approval_ | 11 (propensity), 14.1 (Cox), estimator implementations in `causal/baselines.py`, `causal/negative_controls.py`, `evaluation/metrics.py` | Unfrozen fixed-budget gradient descent (500 or 300 iterations, LR 0.1 or 0.05, no tolerance) | Newton/IRLS (logistic, linear) and Newton-Raphson (Cox), zero init, stop when max abs step < 1e-8, hard cap 100 iterations; a fit that hits the cap or goes non-finite is flagged not-converged and blocks the estimate | The shipped budgets leave fits unconverged: propensity gradient norm ~1e-2, Cox log-HR shrunk ~40-50% (e.g. unadjusted -0.377 vs -0.641). Evidence: `.agents/tasks/propensity-convergence-investigation/findings.md`. |
3. Decide how to handle the sealed internal reference. If it was computed with the unconverged Cox fit, a corrected reference must be recomputed under the existing locked-evaluation protocol. That is a protocol decision for you, not something to do quietly.
4. Treat the positivity failure as a genuine finding from the 7.5 encoding: one control patient with extreme Karnofsky and prior-ART duration. Options are to report it as is, or to revisit how prior-ART duration is encoded (e.g. a bounded or log transform). Either needs your decision; no threshold should move.

## 5. Impact on task 7.6

7.6 should wait until the propensity fit is converged; otherwise weighted SMDs are computed from shrunk scores. On the converged fit, 7.6 will still face extreme_weight_count = 1, so `support_valid` stays False unless the item 4 decision changes the encoding.

## Not measured

- The stored sealed-reference value (sealed by design).
- Gradient norms for the Cox and outcome-logistic fits; convergence was judged by budget stability (300 vs 3,000 vs 30,000).
- Negative-control outcome at the frozen 200/1000 bootstrap counts (50 used).
- The calibration fit on real model predictions (none exist on the dev split yet).

## 6. Outcome (2026-10-10): fix applied as neural-experiment-pipeline tasks 7.7-7.9

- Converged Newton-type solvers (`src/causal/solvers.py`) replace every fixed-budget fit. The rule is frozen in `decisions.md` Section 26 (2026-10-10 row).
- Canonical dev cohort, before -> after: `unadjusted_cox` -0.377 -> -0.641 (HR 0.527 vs the published 0.50); `adjusted_cox` -0.499 -> -0.953; `ps_weighted_marginal_cox` -0.586 -> -1.007; `outcome_regression` -0.083 -> -0.159; `ipcw_aipw` -0.158 -> -0.159.
- New defect found while re-measuring: the negative-control-outcome regression dropped five Section 13 covariates (raw-name lookup). Fixed via the shared mapping; classification `uncertain` (-0.1150) -> `pass` (-0.0906), stable over five seeds x {200, 1000} and both seed paths.
- Still blocking `support_valid`: one arm-0 patient at propensity 0.933 (largest IPW weight, 7.1 before clipping) and the raw-SMD gate pending task 7.6. The prior-ART encoding was deliberately not changed after seeing this.
- Sealed reference: recomputed in memory by the pipeline (no artifact on disk, no locked evaluation consumed).

## 7. Task 7.6 outcome (2026-10-10)

- The SMD gate now reads stabilized-IPW SMDs over all 14 encoded Section 13 columns. Canonical cohort: raw |SMD| up to 0.374, weighted at most 0.048; `baseline_cd8` -0.136 -> -0.002. The SMD condition passes.
- `support_valid` is False on positivity alone (one arm-0 patient at e = 0.933).
- Reported, not adopted: trimming that patient at the primary band would make `support_valid` True, and nobody is outside the [0.05, 0.95] sensitivity band. Choosing either after seeing the result would be post-hoc, so the primary gate stays as frozen.
