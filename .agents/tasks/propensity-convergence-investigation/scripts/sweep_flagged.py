"""The patient the converged 14-column propensity model puts outside [0.10, 0.90]."""
from causal.propensity import LogisticPropensityModel


def flagged_patient_id(patients):
    scores = LogisticPropensityModel(seed=0).fit_cross_fitted(patients, 5).propensity_scores
    (patient_id,) = [i for i, v in scores.items() if not 0.10 <= v <= 0.90]
    return patient_id
