"""Items 2-3: GD budget sweep vs Newton/IRLS reference, and support under each fit."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from causal.overlap import OverlapDiagnostics
from causal.propensity import COVARIATE_FIELDS, LogisticPropensityModel
from harness import canonical_dev_patients, design, folds, pseudonym

OUT = Path(__file__).resolve().parents[1] / "out"
BUDGETS = [500, 1000, 2000, 5000, 20000]
LR = 0.1


def loss_grad(beta, X, y):
    z = X @ beta
    p = 1.0 / (1.0 + np.exp(-z))
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    return loss, X.T @ (p - y) / len(y), p


def gd_path(X, y, checkpoints, tol_target=1e-6, hard_cap=2_000_000):
    """Mirror of LogisticPropensityModel._fit (zero init, full-batch, LR 0.1)."""
    beta = np.zeros(X.shape[1])
    snaps, it, hit = {}, 0, None
    last = max(checkpoints)
    while it < hard_cap:
        loss, g, _ = loss_grad(beta, X, y)
        gn = float(np.linalg.norm(g))
        if hit is None and gn <= tol_target:
            hit = it
        if it in checkpoints:
            snaps[it] = (beta.copy(), loss, gn)
        if it >= last and hit is not None:
            break
        beta = beta - LR * g
        it += 1
    return snaps, hit


def irls(X, y, tol=1e-10, cap=100):
    beta = np.zeros(X.shape[1])
    for k in range(1, cap + 1):
        loss, g, p = loss_grad(beta, X, y)
        H = (X * (p * (1 - p))[:, None]).T @ X / len(y)
        step = np.linalg.solve(H, g)
        beta = beta - step
        if np.linalg.norm(step) < tol:
            break
    loss, g, p = loss_grad(beta, X, y)
    H = (X * (p * (1 - p))[:, None]).T @ X / len(y)
    return beta, loss, float(np.linalg.norm(g)), k, float(np.linalg.cond(H))


def main():
    t0 = time.perf_counter()
    patients, spec = canonical_dev_patients()
    by_id = {p.patient_id: p for p in patients}
    names = ["bias", *COVARIATE_FIELDS]
    settings = [*map(str, BUDGETS), "irls"]
    scores = {s: {} for s in settings}
    per_fold, coefs = [], {s: [] for s in settings}
    gd_to_1e6 = []

    for fold, train, hold in folds(patients):
        xt, yt, xh, hid = design(train, hold)
        X = np.column_stack([np.ones(len(xt)), np.array(xt)])
        Xh = np.column_stack([np.ones(len(xh)), np.array(xh)])
        y = np.array(yt)
        snaps, hit = gd_path(X, y, set(BUDGETS))
        gd_to_1e6.append(hit)
        b_ref, l_ref, g_ref, k_ref, cond = irls(X, y)
        rows = {}
        for s in settings:
            if s == "irls":
                beta, loss, gn, iters = b_ref, l_ref, g_ref, k_ref
            else:
                beta, loss, gn = snaps[int(s)]
                iters = int(s)
            rows[s] = {
                "iterations": iters,
                "grad_norm": gn,
                "loss": loss,
                "loss_gap_vs_irls": loss - l_ref,
                "max_abs_w": float(np.max(np.abs(beta[1:]))),
                "max_coef_dist_vs_irls": float(np.max(np.abs(beta - b_ref))),
            }
            coefs[s].append(dict(zip(names, map(float, beta))))
            ph = 1.0 / (1.0 + np.exp(-(Xh @ beta)))
            scores[s].update(dict(zip(hid, map(float, ph))))
        per_fold.append({"fold": fold, "n_train": len(train), "irls_condition_number": cond,
                         "gd_iterations_to_grad_1e-6": hit, "settings": rows})

    # Sanity: numpy mirror at 500 matches the shipped pure-Python model.
    shipped = LogisticPropensityModel(seed=0).fit_cross_fitted(patients, 5).propensity_scores
    mirror_err = max(abs(shipped[i] - scores["500"][i]) for i in shipped)

    support = {}
    for s in settings:
        rep = OverlapDiagnostics.compute(patients, scores[s], spec.adjustment_set)
        vals = list(scores[s].values())
        out_primary = sorted((i for i, v in scores[s].items() if not 0.10 <= v <= 0.90))
        out_sens = sorted((i for i, v in scores[s].items() if not 0.05 <= v <= 0.95))
        support[s] = {
            "propensity_min": min(vals), "propensity_max": max(vals),
            "ess_by_arm": {str(k): v for k, v in rep.ess_by_arm.items()},
            "extreme_weight_count": rep.extreme_weight_count,
            "outside_primary_band": [pseudonym(i) for i in out_primary],
            "outside_sensitivity_band": [pseudonym(i) for i in out_sens],
            "arm_counts_sufficient": rep.arm_counts_sufficient,
            "ess_sufficient": rep.ess_sufficient,
            "smd_sufficient": rep.smd_sufficient,
            "no_unresolved_positivity_warning": rep.no_unresolved_positivity_warning,
            "support_valid": rep.support_valid,
        }

    # Attribution for every patient out of band under any setting.
    flagged = sorted({i for s in settings for i, v in scores[s].items() if not 0.10 <= v <= 0.90})
    attribution = []
    for pid in flagged:
        p = by_id[pid]
        fold = next(f for f, _, h in folds(patients) if any(q.patient_id == pid for q in h))
        _, train, hold = folds(patients)[fold]
        _, _, xh, hid = design(train, hold)
        x = dict(zip(COVARIATE_FIELDS, xh[hid.index(pid)]))
        entry = {"pseudonym": pseudonym(pid), "arm": int(p.treatment_arm), "fold": fold,
                 "encoded": x, "propensity": {s: scores[s][pid] for s in settings}, "top_terms": {}}
        for s in ("500", "irls"):
            c = coefs[s][fold]
            terms = {k: c[k] * x[k] for k in COVARIATE_FIELDS}
            entry["top_terms"][s] = dict(sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:5])
            entry["top_terms"][s]["bias"] = c["bias"]
        attribution.append(entry)

    result = {"per_fold": per_fold, "coefficients": coefs, "support": support,
              "out_of_band_patients": attribution,
              "numpy_mirror_max_abs_err_vs_shipped_500": mirror_err,
              "note": "SMD condition is still raw (task 7.6 not done); it does not depend on the scores.",
              "wall_seconds": time.perf_counter() - t0}
    (OUT / "convergence_sweep.json").write_text(json.dumps(
        {k: result[k] for k in ("per_fold", "coefficients", "numpy_mirror_max_abs_err_vs_shipped_500")}, indent=2))
    (OUT / "support_by_fit.json").write_text(json.dumps(
        {k: result[k] for k in ("support", "out_of_band_patients", "note")}, indent=2))

    print(f"mirror err vs shipped 500: {mirror_err:.2e}   wall {result['wall_seconds']:.1f}s")
    print("\nfold | setting | grad_norm | loss_gap | max|w| | max dist to IRLS")
    for f in per_fold:
        for s, r in f["settings"].items():
            print(f"{f['fold']} | {s:>5} | {r['grad_norm']:.2e} | {r['loss_gap_vs_irls']:.2e} | {r['max_abs_w']:.3f} | {r['max_coef_dist_vs_irls']:.3f}")
        print(f"  cond(H)={f['irls_condition_number']:.1f}  GD iters to |g|<=1e-6: {f['gd_iterations_to_grad_1e-6']}")
    print("\nsetting | p range | ESS 1/0 | extreme | out_primary | out_sens | positivity | support_valid")
    for s, r in support.items():
        print(f"{s:>5} | [{r['propensity_min']:.4f},{r['propensity_max']:.4f}] | {r['ess_by_arm']['1']:.2f}/{r['ess_by_arm']['0']:.2f} | {r['extreme_weight_count']} | {r['outside_primary_band']} | {r['outside_sensitivity_band']} | {r['no_unresolved_positivity_warning']} | {r['support_valid']}")
    print("\nIRLS coefficients fold 0:", {k: round(v, 3) for k, v in coefs['irls'][0].items()})
    print("GD500 coefficients fold 0:", {k: round(v, 3) for k, v in coefs['500'][0].items()})
    for a in attribution:
        print(f"\npatient {a['pseudonym']} arm={a['arm']} fold={a['fold']}")
        print("  propensity:", {k: round(v, 4) for k, v in a['propensity'].items()})
        print("  encoded:", {k: round(v, 3) for k, v in a['encoded'].items()})
        for s, t in a["top_terms"].items():
            print(f"  logit terms {s}:", {k: round(v, 3) for k, v in t.items()})


if __name__ == "__main__":
    sys.exit(main())
