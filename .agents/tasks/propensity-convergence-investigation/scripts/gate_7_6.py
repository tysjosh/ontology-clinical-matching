"""Task 7.6: weighted-SMD gate on the canonical dev cohort, converged propensities."""
from causal.overlap import OverlapDiagnostics
from causal.propensity import LogisticPropensityModel
from harness import MASTER_SEED, canonical_dev_patients
from infra.seeding import derive_seed

patients, spec = canonical_dev_patients()
scores = LogisticPropensityModel(seed=derive_seed(MASTER_SEED, "propensity")).fit_cross_fitted(patients, 5).propensity_scores
r = OverlapDiagnostics.compute(patients, scores, spec.adjustment_set)
print(f"{'column':32s} {'raw':>8s} {'weighted':>9s}")
for c in r.smd_by_covariate:
    flag = "  <-- fails" if abs(r.smd_by_covariate[c]) >= 0.10 else ""
    print(f"{c:32s} {r.raw_smd_by_covariate[c]:+8.4f} {r.smd_by_covariate[c]:+9.4f}{flag}")
print("exclusions:", r.smd_exclusions)
print("basis:", r.smd_basis)
print({k: getattr(r, k) for k in ("arm_counts_sufficient", "ess_sufficient", "smd_sufficient", "no_unresolved_positivity_warning", "support_valid")})
print("ESS", {k: round(v, 2) for k, v in r.ess_by_arm.items()}, "extreme", r.extreme_weight_count)
