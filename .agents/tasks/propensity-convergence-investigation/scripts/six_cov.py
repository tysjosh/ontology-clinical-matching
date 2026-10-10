"""Item 4: pre-7.5 six-covariate model (git HEAD), 500-iteration GD vs converged IRLS."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import propensity_head as head
from causal.overlap import OverlapDiagnostics
from harness import canonical_dev_patients, folds, pseudonym
from sweep import gd_path, irls

OUT = Path(__file__).resolve().parents[1] / "out"
patients, spec = canonical_dev_patients()
shipped = head.LogisticPropensityModel(seed=0).fit_cross_fitted(patients, 5).propensity_scores
scores = {"500": {}, "irls": {}}
rows = []
for fold, train, hold in folds(patients):
    params = head._fit_standardization(train)
    X = np.column_stack([np.ones(len(train)), np.array([head._feature_vector(p, params) for p in train])])
    Xh = np.column_stack([np.ones(len(hold)), np.array([head._feature_vector(p, params) for p in hold])])
    y = np.array([float(p.treatment_arm) for p in train])
    snaps, hit = gd_path(X, y, {500})
    b5, l5, g5 = snaps[500]
    br, lr, gr, kr, cond = irls(X, y)
    rows.append({"fold": fold, "gd500_grad_norm": g5, "loss_gap": l5 - lr,
                 "max_coef_dist": float(np.max(np.abs(b5 - br))), "irls_iters": kr,
                 "cond": cond, "gd_iters_to_1e-6": hit})
    for s, b in (("500", b5), ("irls", br)):
        ph = 1 / (1 + np.exp(-(Xh @ b)))
        scores[s].update({p.patient_id: float(v) for p, v in zip(hold, ph)})
mirror = max(abs(shipped[i] - scores["500"][i]) for i in shipped)
support = {}
for s in scores:
    rep = OverlapDiagnostics.compute(patients, scores[s], spec.adjustment_set)
    v = list(scores[s].values())
    support[s] = {"range": [min(v), max(v)], "ess_by_arm": {str(k): x for k, x in rep.ess_by_arm.items()},
                  "extreme_weight_count": rep.extreme_weight_count,
                  "outside_primary": [pseudonym(i) for i, x in scores[s].items() if not 0.10 <= x <= 0.90],
                  "support_valid": rep.support_valid}
# The 14-column model's out-of-band patient, selected by condition (no id in source).
import sweep_flagged
flag_id = sweep_flagged.flagged_patient_id(patients)
out = {"per_fold": rows, "support": support, "mirror_err": mirror,
       "item3_patient_propensity": {s: scores[s][flag_id] for s in scores}}
(OUT / "six_covariate_comparison.json").write_text(json.dumps(out, indent=2))
print(f"mirror err vs HEAD shipped: {mirror:.2e}")
for r in rows:
    print(f"fold {r['fold']}: |g|500={r['gd500_grad_norm']:.2e} loss_gap={r['loss_gap']:.2e} "
          f"max_dist={r['max_coef_dist']:.3f} cond={r['cond']:.1f} GD->1e-6={r['gd_iters_to_1e-6']}")
for s, r in support.items():
    print(f"{s}: range [{r['range'][0]:.4f},{r['range'][1]:.4f}] ESS {r['ess_by_arm']['1']:.2f}/{r['ess_by_arm']['0']:.2f} "
          f"extreme={r['extreme_weight_count']} out={r['outside_primary']} support_valid={r['support_valid']}")
print("out-of-band patient under 6-cov:", {k: round(v, 4) for k, v in out["item3_patient_propensity"].items()})
