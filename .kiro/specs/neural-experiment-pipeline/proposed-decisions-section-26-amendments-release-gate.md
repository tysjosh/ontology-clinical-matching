# APPLIED — decisions.md Section 26, 2026-10-12

**Status:** approved by the spec owner and applied on **2026-10-12** as four
rows in `.kiro/specs/causal-witness-pwm/decisions.md` Section 26 (Amendment
log): A (Section 16, `uncertainty_valid`), B (Section 12, common-support
trimming), C (Section 20, increased-missingness rate), and D (Sections 14.2
and 16, the GRU next-state head and the temporal-witness comparison). The
amendment rows are the frozen text. This file keeps the options that were
considered and the evidence.

All four were decided on development data, after development-stage results
had been seen. The Test_Split and the sealed reference remain unopened, and
no locked evaluation has been consumed.

Numbers are from the canonical development cohort (master seed 20240101,
n = 383) unless marked as replication seed 101.

---

## A. `uncertainty_valid`: chosen A1 (procedural validity)

Section 16 names `uncertainty_valid` but no section defined it, and every
runner passed `False`.

- **A1 (chosen):** `uncertainty_valid` holds when the 5-replica ensemble
  trained to completion, the validation-only calibration gave a finite slope
  and intercept, and the supported population is not empty.
- A2 (rejected): A1 plus a calibration-quality bound. It needs a new
  threshold with no frozen basis.
- A3 (rejected): leave it undefined, so nothing can ever be released.

Seed 101 (CPU) after the change: valid. All 5 replicas trained, calibration
slope 0.639 and intercept -0.296, and 124 of 129 validation patients
supported.

## B. The patient at propensity 0.933: chosen B2 (trim)

- **B2 (chosen):** remove patients outside [0.10, 0.90] from the estimation
  population before the support check, identically everywhere, and report
  the count trimmed per arm.
- B1 (rejected): report the untrimmed abstention only.

Canonical cohort after the change: 1 patient trimmed (arm 0),
`support_valid` True, and IPCW-AIPW -0.1527 (untrimmed -0.1593). One side
effect is reported, not adjusted: the negative-control-outcome
classification moves from `pass` to `uncertain` (adjusted difference -0.103
against the 0.10 threshold, CI containing zero). That makes the causal
witness `uncertain` on this run.

## C. Increased-missingness stress test: chosen C1, rate 0.40

- **C1 (chosen):** after the split, delete baseline CD4 and CD8 completely
  at random with probability 0.40, then refit, trim, and re-estimate. The
  rate is anchored to the trial: week-96 CD4 is missing for 38% of the
  development cohort.
- C2 (rejected): remove the variant from Section 20.

Canonical cohort: -0.1414.

## D. The GRU state head: chosen D1 (next-state prediction)

The torch GRU predicted each landmark from the hidden state after reading
that landmark's CD4/CD8. The prototype and Section 14.2 specify next-state
prediction.

- **D1 (chosen):** predict landmark k from the hidden state before reading
  it, and score the temporal witness on weeks 20 and 96 only.
- D2 and D3 (rejected): rescoring without retraining, and reporting as is.

Seed 101 (CPU) after the change: MAE 0.656 (intervention-aware) against
0.599 (GRU). The intervention-aware model is still more than 5% worse, so
the temporal witness still fails. That is now a genuine result about the
models, and it is reported, not tuned.

**Resolved 2026-10-16.** The GRU's survival-hazard head read the final
hidden state, which had already consumed the week-20 and week-96 labs. The
Section 26 amendment of 2026-10-16 moves it to the time-zero hidden state.
The same amendment also makes both neural models read the full Section 13
adjustment set: they had been reading only 4 of its 14 columns, so 3 of the
6 selection confounders were invisible to them. It also freezes a stopping
rule: these are the last model-design changes before the locked
evaluation.

---

## Sensitivity variants on the canonical cohort after all four

| Variant | Estimate |
|---|---|
| Primary (IPCW-AIPW, trimmed) | -0.1527 |
| Boosted propensity | -0.1161 |
| Weight clipping (narrow / wide) | -0.1562 / -0.1612 |
| Support thresholds (0.05–0.95 / 0.10–0.90) | -0.1593 / -0.1527 |
| Increased missingness | -0.1414 |
| Week-20 removed | -0.1527 |
| Comparator reversal | -0.1527 |
| Shifted horizon (48 weeks) | -0.0609 |
| Stronger selection | -0.1413 |

All eight executed, every direction agrees with the primary, and
`overall_stability` is `stable`.
