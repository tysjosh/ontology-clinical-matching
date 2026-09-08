"""Non-neural predictive baseline: `CoxBaseline` (Requirement 8.1;
`decisions.md` Section 14.1 "Endpoint-specific non-neural baseline").

`decisions.md` Section 14.1 resolves Requirement 8.1's "select a non-neural
baseline to match the endpoint type" for this project's primary endpoint
(time-to-composite-event) to **Cox proportional-hazards regression** --
not logistic regression, survival forest, or gradient boosting.

This module does **not** reimplement Cox partial-likelihood fitting. The
exact same machinery already exists in `causal.baselines` (`_fit_cox`,
`_CoxRow`) and is used there by `adjusted_cox` (Cox PH with
`treatment_arm` plus the full frozen covariate list,
`causal.propensity.COVARIATE_FIELDS`). `CoxBaseline` below is a thin
`models`-module-facing wrapper around that existing machinery: it fits the
identical model (same covariate composition, same optimizer) and exposes
it under the naming/interface convention this module's sibling
(`models.sequence_baseline.SequenceGRU`) already establishes -- a `.fit()`
method returning a fitted object that can produce per-patient predictions
-- rather than introducing any new causal-estimation logic.

Where this differs from `causal.baselines.adjusted_cox`
----------------------------------------------------------

`adjusted_cox` returns a single `CausalEstimate` (the fitted treatment
log-hazard-ratio, for causal-baseline comparison). `CoxBaseline` is used
for the *predictive* baseline comparison (Requirement 8, "predictive
accuracy ... separated from causal validity"), so it additionally exposes:

- every fitted coefficient (not just the treatment one), keyed by
  covariate name;
- a per-patient risk score (`exp(beta . x)`, the Cox model's relative
  hazard/linear predictor) suitable for factual predictive metrics such as
  time-dependent concordance (Requirement 8.4/`evaluation.metrics`).

Per Requirement 8.5, this risk score is a **factual predictive quantity**
(how well the model ranks/predicts observed outcomes) -- it is not, and
must not be described as, a causal estimate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from causal.baselines import _CoxRow, _fit_cox
from causal.propensity import COVARIATE_FIELDS, _feature_vector, _fit_standardization
from cohorts.cohort_construction import PatientRecord

#: Covariate names in the exact order `CoxBaseline` fits them in, matching
#: `causal.baselines.adjusted_cox`'s feature composition
#: (`treatment_arm` followed by `causal.propensity.COVARIATE_FIELDS`).
COEFFICIENT_NAMES: tuple[str, ...] = ("treatment_arm", *COVARIATE_FIELDS)


@dataclass(frozen=True)
class FittedCoxBaseline:
    """A fitted `CoxBaseline`: coefficients plus per-patient risk scores.

    Attributes:
        coefficients: `{covariate_name: beta}`, one entry per
            `COEFFICIENT_NAMES` (`"treatment_arm"` plus the frozen
            adjustment set).
        risk_scores: `{patient_id: exp(beta . x)}` -- the fitted model's
            relative hazard (linear predictor exponentiated) for each
            training patient, a factual predictive quantity (Requirement
            8.5: not a causal estimate).
    """

    coefficients: dict[str, float]
    risk_scores: dict[str, float]

    def predict_risk(self, patient: PatientRecord, standardization) -> float:
        """The relative-hazard risk score `exp(beta . x)` for one (possibly
        held-out) patient, using this fit's standardization parameters.

        Exposed for out-of-sample prediction (e.g. a held-out evaluation
        split); `risk_scores` already covers every patient the model was
        fit on directly.
        """
        beta = [self.coefficients[name] for name in COEFFICIENT_NAMES]
        features = [float(patient.treatment_arm), *_feature_vector(patient, standardization)]
        eta = sum(b * x for b, x in zip(beta, features, strict=True))
        return math.exp(eta)


class CoxBaseline:
    """Non-neural predictive baseline: Cox PH on the composite endpoint.

    Thin `models`-module wrapper around `causal.baselines`'s existing Cox
    partial-likelihood machinery (see module docstring) -- fits the
    identical model as `causal.baselines.adjusted_cox` (`treatment_arm`
    plus the frozen adjustment set `causal.propensity.COVARIATE_FIELDS`),
    but exposes all fitted coefficients and per-patient risk scores rather
    than only the treatment coefficient, for use as a factual predictive
    baseline (Requirement 8.1) alongside `models.sequence_baseline`.
    """

    def __init__(self) -> None:
        self._fitted: FittedCoxBaseline | None = None
        self._standardization = None

    def fit(self, patients: Sequence[PatientRecord]) -> "CoxBaseline":
        """Fit Cox PH on `patients`' composite endpoint.

        Reuses `causal.propensity._fit_standardization`/`_feature_vector`
        for covariate extraction (fit once, on the full `patients` input --
        same as `causal.baselines.adjusted_cox`) and
        `causal.baselines._fit_cox` for the actual optimization.
        """
        standardization = _fit_standardization(patients)

        def features(patient: PatientRecord) -> list[float]:
            return [float(patient.treatment_arm), *_feature_vector(patient, standardization)]

        rows = [
            _CoxRow(
                features=tuple(features(patient)),
                event_time=patient.endpoint_result.event_time,
                event_indicator=patient.endpoint_result.event_indicator,
                weight=1.0,
            )
            for patient in patients
        ]
        beta, _inverse_information = _fit_cox(rows)

        coefficients = dict(zip(COEFFICIENT_NAMES, beta, strict=True))
        risk_scores = {
            patient.patient_id: math.exp(
                sum(b * x for b, x in zip(beta, features(patient), strict=True))
            )
            for patient in patients
        }

        self._standardization = standardization
        self._fitted = FittedCoxBaseline(coefficients=coefficients, risk_scores=risk_scores)
        return self

    @property
    def fitted(self) -> FittedCoxBaseline:
        """The `FittedCoxBaseline` result. Raises if `.fit()` was not called."""
        if self._fitted is None:
            raise RuntimeError("CoxBaseline.fit() must be called before accessing `fitted`")
        return self._fitted

    def predict_risk(self, patient: PatientRecord) -> float:
        """The fitted model's relative-hazard risk score for one patient.

        Uses the standardization parameters fit during `.fit()`, so this
        also works for patients not in the original training set (e.g. a
        held-out evaluation split), consistent with `predict_risk`'s
        contract on `FittedCoxBaseline`.
        """
        return self.fitted.predict_risk(patient, self._standardization)


def fit_cox_baseline(patients: Sequence[PatientRecord]) -> FittedCoxBaseline:
    """Function-style convenience wrapper: `CoxBaseline().fit(patients).fitted`."""
    return CoxBaseline().fit(patients).fitted
