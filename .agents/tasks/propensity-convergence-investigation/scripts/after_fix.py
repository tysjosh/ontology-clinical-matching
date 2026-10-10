"""Task 7.9 measurement: canonical dev cohort with the converged solvers (repo code as wired)."""
import json
import time
from pathlib import Path

from causal.baselines import run_all_baselines
from causal.negative_controls import run_negative_control_exposure, run_negative_control_outcome
from causal.overlap import OverlapDiagnostics
from causal.propensity import LogisticPropensityModel
from harness import MASTER_SEED, canonical_dev_patients, pseudonym
from infra.seeding import derive_seed
from models.non_neural_baseline import CoxBaseline

OUT = Path(__file__).resolve().parents[1] / "out"
t0 = time.perf_counter()
patients, spec = canonical_dev_patients()
cf = LogisticPropensityModel(seed=derive_seed(MASTER_SEED, "propensity")).fit_cross_fitted(patients, 5)
scores = cf.propensity_scores
est = {e.name: {"estimate": e.estimate, "se": e.se} for e in run_all_baselines(patients, scores)}
est["CoxBaseline.treatment_arm"] = {"estimate": CoxBaseline().fit(patients).fitted.coefficients["treatment_arm"], "se": None}
rep = OverlapDiagnostics.compute(patients, scores, spec.adjustment_set)
nc_seed = derive_seed(MASTER_SEED, "negative_controls")
exp_r = run_negative_control_exposure(patients, scores, seed=nc_seed, n_bootstrap=200)
out_r = run_negative_control_outcome(patients, spec.adjustment_set, seed=nc_seed, n_bootstrap=200)
vals = list(scores.values())
result = {
    "propensity_convergence": {str(k): d.as_dict() for k, d in cf.convergence.items()},
    "propensity_range": [min(vals), max(vals)],
    "estimates": est,
    "overlap": {
        "ess_by_arm": {str(k): v for k, v in rep.ess_by_arm.items()},
        "extreme_weight_count": rep.extreme_weight_count,
        "outside_primary": [pseudonym(i) for i, v in scores.items() if not 0.10 <= v <= 0.90],
        "arm_counts_sufficient": rep.arm_counts_sufficient,
        "ess_sufficient": rep.ess_sufficient,
        "smd_sufficient (raw; 7.6 not done)": rep.smd_sufficient,
        "no_unresolved_positivity_warning": rep.no_unresolved_positivity_warning,
        "support_valid": rep.support_valid,
    },
    "negative_control_exposure": {"class": exp_r.classification, "point": exp_r.point_estimate, "ci": [exp_r.ci_low, exp_r.ci_high], "n_bootstrap": 200},
    "negative_control_outcome": {"class": out_r.classification, "point": out_r.point_estimate, "ci": [out_r.ci_low, out_r.ci_high], "n_bootstrap": 200,
                                 "n_bootstrap_successful": out_r.evidence["n_bootstrap_successful"]},
    "wall_seconds": time.perf_counter() - t0,
}
(OUT / "after_fix.json").write_text(json.dumps(result, indent=2, default=str))
print(json.dumps({k: v for k, v in result.items() if k != "propensity_convergence"}, indent=1, default=str))
print("propensity folds:", {k: (d["iterations"], d["reason"], f"{d['final_gradient_norm']:.1e}", d["rank"]) for k, d in result["propensity_convergence"].items()})
