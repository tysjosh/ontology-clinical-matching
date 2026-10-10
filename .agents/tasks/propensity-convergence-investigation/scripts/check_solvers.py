"""Validate causal.solvers on the canonical cohort against numpy references and old GD at long budgets."""
import time
import numpy as np
import causal.baselines as B
from causal.solvers import fit_logistic_newton, fit_linear_exact, fit_cox_newton
from harness import canonical_dev_patients, design, folds
from sweep import irls

patients, spec = canonical_dev_patients()
# Logistic propensity, every fold, vs numpy IRLS
for fold, train, hold in folds(patients):
    xt, yt, _, _ = design(train, hold)
    t = time.perf_counter(); b, w, d = fit_logistic_newton(xt, yt); dt = time.perf_counter() - t
    X = np.column_stack([np.ones(len(xt)), np.array(xt)])
    ref = irls(X, np.array(yt))[0]
    print(f"logistic fold {fold}: {d.reason} iters={d.iterations} |g|={d.final_gradient_norm:.1e} "
          f"max diff vs numpy={np.max(np.abs(np.array([b, *w]) - ref)):.1e} ({dt:.2f}s)")
# Linear vs lstsq
rng = np.random.default_rng(0); X = rng.normal(size=(300, 5)); X[:, 4] = 0.0
y = X @ np.array([1, -2, 0.5, 3, 0]) + 0.3 + rng.normal(size=300)
b, w, d = fit_linear_exact(X.tolist(), y.tolist())
ref = np.linalg.lstsq(np.column_stack([np.ones(300), X[:, :4]]), y, rcond=None)[0]
print("linear:", d.reason, d.iterations, "max diff", np.max(np.abs(np.array([b, *w[:4]]) - ref)), "zero col coef", w[4])
# Cox: unadjusted and adjusted on canonical cohort vs old GD at 30000 iterations
for name in ("unadjusted_cox", "adjusted_cox"):
    rows = []
    if name == "unadjusted_cox":
        feats = lambda p: [float(p.treatment_arm)]
    else:
        st = B._fit_standardization(patients); feats = lambda p: [float(p.treatment_arm), *B._feature_vector(p, st)]
    rows = sorted((B._CoxRow(tuple(feats(p)), p.endpoint_result.event_time, p.endpoint_result.event_indicator, 1.0) for p in patients), key=lambda r: r.event_time)
    t = time.perf_counter()
    beta, d, info = fit_cox_newton([r.features for r in rows], [r.event_time for r in rows], [r.event_indicator for r in rows], [r.weight for r in rows], B._ETA_CLIP)
    dt = time.perf_counter() - t
    B._N_ITERATIONS = 30000; gd_beta, _ = B._fit_cox(rows); B._N_ITERATIONS = 300
    print(f"{name}: newton beta0={beta[0]:+.5f} vs GD30000 {gd_beta[0]:+.5f} | {d.reason} iters={d.iterations} |g|={d.final_gradient_norm:.1e} ({dt:.2f}s)")
