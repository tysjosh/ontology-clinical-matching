"""Item 6: do the other fixed-budget GD fits converge on canonical dev inputs?

Method: rerun the real functions with their module-level iteration budget
patched (shipped, 10x, 100x), so the code under test is exactly the shipped
loop. The CD8 linear regression also gets an exact OLS reference.
Propensity scores fed to the baselines are the shipped 500-iteration ones.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

import causal.baselines as B
import causal.negative_controls as NC
import evaluation.metrics as M
from causal.propensity import LogisticPropensityModel
from harness import canonical_dev_patients
from infra.seeding import derive_seed
from models.non_neural_baseline import CoxBaseline

OUT = Path(__file__).resolve().parents[1] / "out"
patients, spec = canonical_dev_patients()
scores = LogisticPropensityModel(seed=derive_seed(20240101, "propensity")).fit_cross_fitted(patients, 5).propensity_scores
result = {}

# --- causal.baselines (_fit_cox, _fit_logistic) and CoxBaseline ------------
shipped_b = B._N_ITERATIONS
rows = {}
for budget in (shipped_b, shipped_b * 10, shipped_b * 100):
    B._N_ITERATIONS = budget
    t = time.perf_counter()
    est = {e.name: e.estimate for e in B.run_all_baselines(patients, scores)}
    cb = CoxBaseline().fit(patients).fitted.coefficients["treatment_arm"]
    est["CoxBaseline.treatment_arm"] = cb
    rows[budget] = {"estimates": est, "wall_s": time.perf_counter() - t}
    print(f"baselines budget={budget}: " + ", ".join(f"{k}={v:+.4f}" for k, v in est.items()) + f"  ({rows[budget]['wall_s']:.1f}s)")
B._N_ITERATIONS = shipped_b
result["baselines"] = {"learning_rate": B._LEARNING_RATE, "by_budget": rows}

# --- negative-control outcome CD8 linear regression -------------------------
covs = [f for f in spec.adjustment_set if f != NC.BASELINE_CD8_FIELD]
shipped_nc = NC._LR_N_ITERATIONS
nc_rows = {}
for budget in (shipped_nc, shipped_nc * 10, shipped_nc * 100):
    NC._LR_N_ITERATIONS = budget
    nc_rows[budget] = NC._fit_cd8_regression_smd(patients, covs)
NC._LR_N_ITERATIONS = shipped_nc

shipped_linear = NC._fit_linear_regression
def ols(features, labels):
    X = np.column_stack([np.ones(len(features)), np.array(features, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, np.array(labels, dtype=float), rcond=None)
    return float(beta[0]), [float(b) for b in beta[1:]]

NC._fit_linear_regression = ols
ols_point = NC._fit_cd8_regression_smd(patients, covs)
seed = derive_seed(20240101, "negative_controls")
nc_ols = NC.run_negative_control_outcome(patients, spec.adjustment_set, seed=seed, n_bootstrap=50)
NC._fit_linear_regression = shipped_linear
nc_ship = NC.run_negative_control_outcome(patients, spec.adjustment_set, seed=seed, n_bootstrap=50)
result["negative_control_outcome"] = {
    "regression_covariates": covs, "learning_rate": NC._LR_LEARNING_RATE,
    "point_by_budget": nc_rows, "ols_point": ols_point,
    "n_bootstrap_used": 50,
    "shipped": {"point": nc_ship.point_estimate, "ci": [nc_ship.ci_low, nc_ship.ci_high], "class": nc_ship.classification},
    "ols": {"point": nc_ols.point_estimate, "ci": [nc_ols.ci_low, nc_ols.ci_high], "class": nc_ols.classification},
}
print("CD8 NCO point by budget:", {k: round(v, 4) for k, v in nc_rows.items()}, "OLS exact:", round(ols_point, 4))
print("CD8 NCO (50 boot) shipped:", result["negative_control_outcome"]["shipped"])
print("CD8 NCO (50 boot) OLS    :", result["negative_control_outcome"]["ols"])

# --- evaluation.metrics calibration fit -------------------------------------
shipped_m = M._N_ITERATIONS
ids = sorted(scores)
real_p = [scores[i] for i in ids]
real_y = [bool(next(p for p in patients if p.patient_id == i).treatment_arm) for i in ids]
rng = random.Random(0)
syn_p = [rng.uniform(0.02, 0.98) for _ in range(400)]
syn_y = [rng.random() < 1 / (1 + np.exp(-(0.3 + 0.6 * np.log(p / (1 - p))))) for p in syn_p]
cal = {}
for label, (pp, yy) in {"propensity_vs_treatment (real dev input, representative only)": (real_p, real_y),
                        "synthetic (true intercept 0.3, slope 0.6)": (syn_p, syn_y)}.items():
    cal[label] = {}
    for budget in (shipped_m, shipped_m * 10, shipped_m * 100):
        M._N_ITERATIONS = budget
        cal[label][budget] = M.calibration_slope_intercept(pp, yy)
    print(f"calibration {label}:", {k: (round(a, 4), round(b, 4)) for k, (a, b) in cal[label].items()})
M._N_ITERATIONS = shipped_m
result["calibration"] = {"learning_rate": M._LEARNING_RATE, "by_input": cal}

(OUT / "other_fits.json").write_text(json.dumps(result, indent=2, default=str))
