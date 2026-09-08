"""The controlled observationalization mechanism (Requirement 7.1;
Correctness Property 2; `design.md` Section 3.4, Stage 7; `decisions.md`
Section 9).

`decisions.md` Section 9 freezes an exact, treatment- and covariate-
dependent selection mechanism that converts the randomized cohort into a
reproducible pseudo-observational cohort. This module implements exactly
that formula and nothing else -- every coefficient is a module-level
constant, copied verbatim from `decisions.md` Section 9, so an accidental
edit is guaranteed to be caught by `tests/unit/test_selection.py`'s
coefficient-equality test.

One variant mechanism sits alongside the primary one: the
stronger-observational-selection coefficients frozen by the `decisions.md`
Section 26 amendment dated 2026-09-07 (every risk-score coefficient scaled
by 1.5, selection-logit intercept unchanged), used only by the
`stronger_selection_severity` sensitivity variant and reached exclusively
through `observationalize(..., selection_severity="stronger")` /
`coefficients_for_severity`. It is a separate frozen bundle, never a
mutation of the primary constants -- see `STRONGER_SELECTION_COEFFICIENTS`
below.

**Baseline risk score** (from development-fitted standardized baseline
covariates):

    r_i = -0.8 * z(CD4_i) - 0.3 * z(Karnofsky_i) + 0.3 * z(Age_i)
          + 0.6 * Symptomatic_i + 0.6 * PriorOI_i + 0.4 * PriorART_i

**Selection probability**:

    P(S_i = 1 | A_i, X_i) = sigmoid(1.2 + (2*A_i - 1) * r_i)

where `S_i = 1` means the patient is retained in the pseudo-observational
cohort, `A_i` is the treatment arm (`0` or `1`), and
`sigmoid(x) = 1 / (1 + exp(-x))`. Note the risk score itself has no
intercept term -- the `+1.2` intercept belongs only to the selection-
probability logit, never to `r_i`.

Where the six covariates come from
------------------------------------

`decisions.md` Section 9's covariate list is CD4, Karnofsky, Age,
Symptomatic, PriorOI, PriorART. Three of these (CD4, Karnofsky, Age) are
the exact three continuous fields `cohorts.preprocessing.FittedPreprocessing`
standardizes by default
(`cohorts.preprocessing.DEFAULT_CONTINUOUS_FIELDS = ("baseline_cd4",
"karnofsky_score", "age")`) -- `compute_risk_score` below reproduces that
same z-score arithmetic (`(raw - mean) / stdev`, or `0.0` when `stdev` is
`0.0` or the field is unobserved) using `fitted_preprocessing.fitted_params`
directly, on the *raw* (untransformed) covariate values still present on
`PatientRecord.timeline.static_covariates`. This means `observationalize`
below is called on the raw `PatientSet` splits `stratified_patient_split`
produces -- it does not require a caller to have already called
`FittedPreprocessing.transform()` on the split; the standardization implied
by `decisions.md` Section 9 step 5 ("calculate the risk score using those
development-fitted transformations") happens *inside* `compute_risk_score`,
reading the same frozen `fitted_params` `FittedPreprocessing.transform()`
would have used, rather than reimplementing `.fit()`.

The remaining three covariates -- Symptomatic, PriorOI, PriorART -- are
raw, non-standardized boolean/indicator covariates that
`FittedPreprocessing` never touches (it only standardizes the three
continuous fields above). They are read directly from
`PatientRecord.timeline.static_covariates`:

- `Symptomatic_i`: the `"symptomatic_status"` concept_id (a `bool` per
  `data.loaders.actg175.BaselineCovariateRecord.symptomatic_status`),
  coerced to `1.0`/`0.0`.
- `PriorOI_i`: the `"prior_opportunistic_infection"` concept_id (a `bool`
  per `data.loaders.actg175.BaselineCovariateRecord.prior_opportunistic_infection`),
  coerced to `1.0`/`0.0`.
- `PriorART_i`: derived, not read directly -- there is no boolean
  "prior ART yes/no" concept_id in the raw schema, only a continuous
  exposure-duration field. Mirroring
  `cohorts.splitting._prior_art_status`'s derivation (checked, in order:
  `"prior_art_exposure_days"`, then `"prior_antiretroviral_exposure_duration"`
  -- the raw and normalized concept_id spellings for the same underlying
  covariate), `PriorART_i = 1.0` if the first present, numeric exposure
  value is `> 0`, else `0.0`. A patient missing both concept_ids gets
  `PriorART_i = 0.0` (treated as "no prior ART" rather than raising --
  see `_prior_art_indicator` below), consistent with `splitting.py`'s
  "unknown" patients not being permitted to crash the pipeline, though
  here there is no third bucket to fall back to since `PriorART_i` is a
  numeric coefficient input, not a stratification key.

Deterministic per-patient selection draws
-------------------------------------------

`observationalize` draws each patient's `S_i ~ Bernoulli(P(S_i=1|A_i,X_i))`
using a single `random.Random(seed)` instance, consuming one uniform draw
per patient in `patient_id`-sorted order (never relying on `dict`/`set`
iteration order) -- so the same `(patients, fitted_preprocessing, seed)`
always reproduces the same retained set, and a different `seed` reliably
produces a different one. The `seed` passed in is expected to already be
the split-specific seed derived via `infra.seeding.derive_seed(master_seed,
split_name)` (`decisions.md` Section 9 step 7) -- see `pipeline.py`, which
performs that derivation once per split so no random-number stream is ever
reused across dev/val/test.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from cohorts.cohort_construction import PatientRecord
from cohorts.preprocessing import DevelopmentSet, FittedPreprocessing
from cohorts.splitting import PatientSet

# ---------------------------------------------------------------------------
# Frozen coefficients (decisions.md Section 9) -- copied verbatim. Every
# coefficient the formula uses is named here as its own constant so an
# accidental edit is caught by tests/unit/test_selection.py's coefficient
# equality test, rather than silently drifting.
# ---------------------------------------------------------------------------

#: Coefficient on z(CD4_i) in the baseline risk score.
COEF_CD4: float = -0.8

#: Coefficient on z(Karnofsky_i) in the baseline risk score.
COEF_KARNOFSKY: float = -0.3

#: Coefficient on z(Age_i) in the baseline risk score.
COEF_AGE: float = 0.3

#: Coefficient on Symptomatic_i in the baseline risk score.
COEF_SYMPTOMATIC: float = 0.6

#: Coefficient on PriorOI_i in the baseline risk score.
COEF_PRIOR_OI: float = 0.6

#: Coefficient on PriorART_i in the baseline risk score.
COEF_PRIOR_ART: float = 0.4

#: The intercept in the selection-probability logit,
#: `sigmoid(1.2 + (2*A_i - 1) * r_i)`. The risk score r_i itself has no
#: intercept term -- decisions.md Section 9 states no intercept for r_i;
#: this +1.2 belongs only to the selection logit.
SELECTION_LOGIT_INTERCEPT: float = 1.2


@dataclass(frozen=True)
class SelectionCoefficients:
    """A bundle of all seven coefficients the selection mechanism uses.

    Exists so the frozen module-level constants above (which remain the
    single source of truth for the *default*, `decisions.md`-Section-9
    formula) can be threaded through `compute_risk_score`/
    `selection_probability` as one value, rather than as seven separate
    hardcoded module globals -- this is what makes one-time pilot
    recalibration (`decisions.md` Section 9, "One-time pilot
    recalibration") possible without ever mutating the frozen constants
    themselves: recalibration produces a *new* `SelectionCoefficients`
    instance, it never reassigns `COEF_CD4` etc.

    Attributes:
        coef_cd4: coefficient on z(CD4_i) in the risk score.
        coef_karnofsky: coefficient on z(Karnofsky_i) in the risk score.
        coef_age: coefficient on z(Age_i) in the risk score.
        coef_symptomatic: coefficient on Symptomatic_i in the risk score.
        coef_prior_oi: coefficient on PriorOI_i in the risk score.
        coef_prior_art: coefficient on PriorART_i in the risk score.
        selection_logit_intercept: the intercept in the selection-
            probability logit.
    """

    coef_cd4: float
    coef_karnofsky: float
    coef_age: float
    coef_symptomatic: float
    coef_prior_oi: float
    coef_prior_art: float
    selection_logit_intercept: float


#: The frozen, `decisions.md`-Section-9 default coefficients, built from the
#: module-level constants above. Every ordinary (non-recalibration) caller
#: of `compute_risk_score`/`selection_probability` gets exactly this bundle
#: by default -- recalibration never mutates this instance or the
#: constants it is built from; it only ever produces and returns a
#: *different* `SelectionCoefficients` instance for the caller to pass
#: explicitly.
DEFAULT_COEFFICIENTS = SelectionCoefficients(
    coef_cd4=COEF_CD4,
    coef_karnofsky=COEF_KARNOFSKY,
    coef_age=COEF_AGE,
    coef_symptomatic=COEF_SYMPTOMATIC,
    coef_prior_oi=COEF_PRIOR_OI,
    coef_prior_art=COEF_PRIOR_ART,
    selection_logit_intercept=SELECTION_LOGIT_INTERCEPT,
)

# ---------------------------------------------------------------------------
# Frozen stronger-severity coefficients (`decisions.md` Section 26
# amendment dated 2026-09-07, amending Sections 9/20 for the
# `stronger_selection_severity` sensitivity variant) -- copied verbatim,
# same convention as the primary coefficients above: one named constant per
# coefficient so an accidental edit is caught by
# `tests/frozen/test_decisions_match.py`.
#
# The amendment freezes the stronger mechanism as the primary Section 9
# mechanism with every risk-score coefficient scaled by exactly 1.5 and the
# selection-logit intercept left unchanged at 1.2:
#
#     r_stronger_i = -1.2*z(CD4_i) - 0.45*z(Karnofsky_i) + 0.45*z(Age_i)
#                    + 0.9*Symptomatic_i + 0.9*PriorOI_i + 0.6*PriorART_i
#
#     P(S_i = 1 | A_i, X_i) = sigmoid(1.2 + (2*A_i - 1) * r_stronger_i)
#
# The literal values are written out here rather than computed as
# `1.5 * COEF_CD4` etc. so this module states the frozen numbers exactly as
# the amendment records them, and so a future edit to a primary coefficient
# cannot silently move a stronger-severity value with it.
#
# This is a variant-only mechanism, NOT a pilot recalibration: it never
# touches the primary constants above, `DEFAULT_COEFFICIENTS`, or
# `PilotRecalibrator`'s one-time allowance, and the primary estimate
# continues to use the primary coefficients.
# ---------------------------------------------------------------------------

#: Stronger-severity coefficient on z(CD4_i).
STRONGER_COEF_CD4: float = -1.2

#: Stronger-severity coefficient on z(Karnofsky_i).
STRONGER_COEF_KARNOFSKY: float = -0.45

#: Stronger-severity coefficient on z(Age_i).
STRONGER_COEF_AGE: float = 0.45

#: Stronger-severity coefficient on Symptomatic_i.
STRONGER_COEF_SYMPTOMATIC: float = 0.9

#: Stronger-severity coefficient on PriorOI_i.
STRONGER_COEF_PRIOR_OI: float = 0.9

#: Stronger-severity coefficient on PriorART_i.
STRONGER_COEF_PRIOR_ART: float = 0.6

#: The stronger-severity selection-logit intercept -- deliberately
#: identical to `SELECTION_LOGIT_INTERCEPT`, so overall retention stays
#: comparable to the primary cohort and the variant does not confound
#: stronger confounding with a smaller sample.
STRONGER_SELECTION_LOGIT_INTERCEPT: float = 1.2

#: The frozen stronger-observational-selection coefficient bundle, built
#: from the `STRONGER_*` constants above. Reached through
#: `coefficients_for_severity("stronger")` or
#: `observationalize(..., selection_severity="stronger")`; never the
#: default for any caller.
STRONGER_SELECTION_COEFFICIENTS = SelectionCoefficients(
    coef_cd4=STRONGER_COEF_CD4,
    coef_karnofsky=STRONGER_COEF_KARNOFSKY,
    coef_age=STRONGER_COEF_AGE,
    coef_symptomatic=STRONGER_COEF_SYMPTOMATIC,
    coef_prior_oi=STRONGER_COEF_PRIOR_OI,
    coef_prior_art=STRONGER_COEF_PRIOR_ART,
    selection_logit_intercept=STRONGER_SELECTION_LOGIT_INTERCEPT,
)

#: The two selection severities the pipeline recognizes. `"primary"` is the
#: frozen `decisions.md` Section 9 mechanism used for every primary result;
#: `"stronger"` is the Section 26 amendment's variant mechanism, used only
#: by the `stronger_selection_severity` sensitivity variant.
SelectionSeverity = Literal["primary", "stronger"]

#: Severity name -> frozen coefficient bundle. The single lookup any caller
#: uses to turn a `selection_severity` string into coefficients, so no
#: caller re-states the coefficient values.
COEFFICIENTS_BY_SEVERITY: dict[str, SelectionCoefficients] = {
    "primary": DEFAULT_COEFFICIENTS,
    "stronger": STRONGER_SELECTION_COEFFICIENTS,
}


def coefficients_for_severity(selection_severity: SelectionSeverity) -> SelectionCoefficients:
    """Resolve a `selection_severity` name to its frozen coefficient bundle.

    Args:
        selection_severity: `"primary"` (the frozen `decisions.md` Section 9
            mechanism) or `"stronger"` (the Section 26 amendment's
            stronger-observational-selection mechanism).

    Returns:
        The frozen `SelectionCoefficients` for that severity.

    Raises:
        ValueError: if `selection_severity` is not a recognized severity
            name -- an unrecognized severity is never silently treated as
            `"primary"`.
    """
    try:
        return COEFFICIENTS_BY_SEVERITY[selection_severity]
    except KeyError:
        permitted = ", ".join(repr(name) for name in sorted(COEFFICIENTS_BY_SEVERITY))
        raise ValueError(
            f"unknown selection_severity {selection_severity!r}; permitted: {permitted}"
        ) from None


# ---------------------------------------------------------------------------
# Covariate field/concept_id names (see module docstring).
# ---------------------------------------------------------------------------

#: Continuous covariate fields whose z-scores enter the risk score, using
#: the exact field names `cohorts.preprocessing.FittedPreprocessing`
#: standardizes by default.
_CD4_FIELD = "baseline_cd4"
_KARNOFSKY_FIELD = "karnofsky_score"
_AGE_FIELD = "age"

#: Raw boolean-indicator concept_ids, read directly from
#: `static_covariates` (never standardized by `FittedPreprocessing`).
_SYMPTOMATIC_CONCEPT_ID = "symptomatic_status"
_PRIOR_OI_CONCEPT_ID = "prior_opportunistic_infection"

#: `concept_id`s checked, in order, when deriving `PriorART_i` -- the raw
#: and normalized spellings of prior-antiretroviral-exposure duration
#: (mirrors `cohorts.splitting._PRIOR_ART_CONCEPT_IDS`).
_PRIOR_ART_CONCEPT_IDS: tuple[str, ...] = (
    "prior_art_exposure_days",
    "prior_antiretroviral_exposure_duration",
)


def _standardized_value(
    patient: PatientRecord, field_name: str, fitted_preprocessing: FittedPreprocessing
) -> float:
    """Reproduce `FittedPreprocessing`'s z-score arithmetic for one field,
    reading its frozen, development-fit `mean`/`stdev` from
    `fitted_preprocessing.fitted_params` (never recomputing fitting here).

    Mirrors `cohorts.preprocessing.FittedPreprocessing._transform_one`'s
    per-field logic exactly: a field with no development-observed values at
    fit time, or a missing/non-numeric value for this patient, standardizes
    to `0.0`; a field whose development-fitted `stdev` is `0.0` also
    standardizes to `0.0` rather than dividing by zero.
    """
    params = fitted_preprocessing.fitted_params
    if field_name not in params.means:
        return 0.0

    raw_value = patient.timeline.static_covariates.get(field_name)
    if raw_value is None:
        return 0.0
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0

    mean = params.means[field_name]
    stdev = params.stdevs.get(field_name, 0.0)
    return (value - mean) / stdev if stdev > 0 else 0.0


def _boolean_indicator(patient: PatientRecord, concept_id: str) -> float:
    """Read a raw boolean-indicator covariate, coerced to `1.0`/`0.0`.

    A missing (`None`) value is treated as `0.0` (absent = not present) --
    boolean clinical indicators in `BaselineCovariateRecord` are non-
    nullable at the source, so absence here means the concept simply was
    not recorded on this timeline, not a genuine "unknown" third state.
    """
    raw_value = patient.timeline.static_covariates.get(concept_id)
    if raw_value is None:
        return 0.0
    return 1.0 if bool(raw_value) else 0.0


def _prior_art_indicator(patient: PatientRecord) -> float:
    """Derive `PriorART_i` from prior-antiretroviral exposure duration.

    See module docstring: checks `_PRIOR_ART_CONCEPT_IDS` in order, returns
    `1.0` if the first present, numeric value is `> 0`, else `0.0`
    (including when neither concept_id is present -- see module docstring
    for why "no prior ART" rather than a raise is the right default here).
    """
    static_covariates = patient.timeline.static_covariates
    for concept_id in _PRIOR_ART_CONCEPT_IDS:
        if concept_id not in static_covariates:
            continue
        raw_value = static_covariates[concept_id]
        if raw_value is None:
            continue
        try:
            exposure = float(raw_value)
        except (TypeError, ValueError):
            continue
        return 1.0 if exposure > 0 else 0.0
    return 0.0


def compute_risk_score(
    patient: PatientRecord,
    fitted_preprocessing: FittedPreprocessing,
    coefficients: SelectionCoefficients = DEFAULT_COEFFICIENTS,
) -> float:
    """Compute the baseline risk score r_i for one patient.

        r_i = coef_cd4 * z(CD4_i) + coef_karnofsky * z(Karnofsky_i)
              + coef_age * z(Age_i) + coef_symptomatic * Symptomatic_i
              + coef_prior_oi * PriorOI_i + coef_prior_art * PriorART_i

    z(CD4_i), z(Karnofsky_i), z(Age_i) are computed from
    `fitted_preprocessing`'s frozen, development-fit mean/stdev (never
    refit here -- see `_standardized_value`); Symptomatic_i, PriorOI_i,
    PriorART_i are raw indicator covariates read directly from
    `patient.timeline.static_covariates` (see module docstring).

    Args:
        patient: the patient to score. May be a patient from any split
            (development, validation, or test) -- this function itself
            does not distinguish between splits; only the *parameters*
            inside `fitted_preprocessing` (always development-fit) do.
        fitted_preprocessing: an already-`.fit()`-ted
            `cohorts.preprocessing.FittedPreprocessing` instance (fit on
            the development split only, per Requirement 6.5/6.6).
        coefficients: the coefficient bundle to use. Defaults to
            `DEFAULT_COEFFICIENTS` (the frozen `decisions.md` Section 9
            values) -- this default preserves every existing caller's
            behavior unchanged. Only `PilotRecalibrator.recalibrate()`'s
            caller should ever pass a different bundle here, and only
            after a documented, development-only pilot recalibration
            (`decisions.md` Section 9, "One-time pilot recalibration").

    Returns:
        The patient's baseline risk score r_i.

    Raises:
        cohorts.preprocessing.PreprocessingNotFittedError: if
            `fitted_preprocessing` has not been `.fit()` yet (propagated
            from `fitted_preprocessing.fitted_params`).
    """
    z_cd4 = _standardized_value(patient, _CD4_FIELD, fitted_preprocessing)
    z_karnofsky = _standardized_value(patient, _KARNOFSKY_FIELD, fitted_preprocessing)
    z_age = _standardized_value(patient, _AGE_FIELD, fitted_preprocessing)

    symptomatic = _boolean_indicator(patient, _SYMPTOMATIC_CONCEPT_ID)
    prior_oi = _boolean_indicator(patient, _PRIOR_OI_CONCEPT_ID)
    prior_art = _prior_art_indicator(patient)

    return (
        coefficients.coef_cd4 * z_cd4
        + coefficients.coef_karnofsky * z_karnofsky
        + coefficients.coef_age * z_age
        + coefficients.coef_symptomatic * symptomatic
        + coefficients.coef_prior_oi * prior_oi
        + coefficients.coef_prior_art * prior_art
    )


def selection_probability(
    risk_score: float,
    treatment_arm: int,
    coefficients: SelectionCoefficients = DEFAULT_COEFFICIENTS,
) -> float:
    """The selection probability P(S_i=1 | A_i, X_i).

        P(S_i = 1 | A_i, X_i) = sigmoid(coefficients.selection_logit_intercept
                                         + (2*A_i - 1) * r_i)

    Args:
        risk_score: the patient's baseline risk score r_i
            (`compute_risk_score`'s output).
        treatment_arm: the patient's treatment arm, `0` or `1`.
        coefficients: the coefficient bundle to use (only
            `selection_logit_intercept` is read here). Defaults to
            `DEFAULT_COEFFICIENTS`, preserving existing behavior unchanged
            -- see `compute_risk_score`'s `coefficients` docstring.

    Returns:
        The selection probability, in `(0, 1)`.

    Raises:
        ValueError: if `treatment_arm` is not `0` or `1`.
    """
    if treatment_arm not in (0, 1):
        raise ValueError(f"treatment_arm must be 0 or 1, got {treatment_arm!r}")

    logit = coefficients.selection_logit_intercept + (2 * treatment_arm - 1) * risk_score
    return 1.0 / (1.0 + math.exp(-logit))


def _resolve_coefficients(
    coefficients: SelectionCoefficients, selection_severity: SelectionSeverity
) -> SelectionCoefficients:
    """Reconcile an explicit `coefficients` bundle with a
    `selection_severity` name into the one bundle that actually runs.

    `"primary"` keeps whatever `coefficients` the caller passed (the frozen
    `DEFAULT_COEFFICIENTS` by default, or a documented pilot-recalibration
    bundle). Any other severity resolves its own frozen bundle -- and
    rejects a simultaneously-passed non-default `coefficients`, since the
    two would name conflicting coefficients and quietly picking one would
    obscure which frozen mechanism produced the cohort.
    """
    if selection_severity == "primary":
        return coefficients

    severity_coefficients = coefficients_for_severity(selection_severity)
    if coefficients != DEFAULT_COEFFICIENTS:
        raise ValueError(
            "cannot pass an explicit coefficients bundle together with "
            f"selection_severity={selection_severity!r}: the two specify "
            "conflicting coefficients. Pass selection_severity alone to use "
            "the frozen severity coefficients, or coefficients alone with "
            "selection_severity='primary'."
        )
    return severity_coefficients


@dataclass(frozen=True)
class ObservationalizedSplit:
    """The result of observationalizing one split (`design.md` Stage 7).

    Attributes:
        retained: the `PatientSet` of retained patients (S_i = 1), sorted
            by `patient_id` -- this is the split's pseudo-observational
            cohort membership manifest.
        risk_scores: `{patient_id: r_i}` for every patient considered
            (retained or not), for provenance/auditing (`decisions.md`
            Section 9's frozen-artifacts list).
        selection_probabilities: `{patient_id: P(S_i=1|A_i,X_i)}` for every
            patient considered.
        seed: the split-specific seed used to draw selection indicators.
        selection_severity: which frozen mechanism produced this split --
            `"primary"` (`decisions.md` Section 9) or `"stronger"` (the
            Section 26 amendment's coefficients). Recorded so a
            provenance/audit consumer can tell the two apart without
            re-deriving them from the coefficients.
    """

    retained: PatientSet
    risk_scores: dict[str, float]
    selection_probabilities: dict[str, float]
    seed: int
    selection_severity: SelectionSeverity = "primary"

    @property
    def retained_patient_ids(self) -> tuple[str, ...]:
        """The retained cohort's `patient_id`s -- the per-split manifest
        (`decisions.md` Section 9's frozen artifact "resulting retained
        patient ID list per split")."""
        return self.retained.patient_ids


def observationalize(
    patients: PatientSet,
    fitted_preprocessing: FittedPreprocessing,
    seed: int,
    coefficients: SelectionCoefficients = DEFAULT_COEFFICIENTS,
    selection_severity: SelectionSeverity = "primary",
) -> ObservationalizedSplit:
    """Apply the frozen selection mechanism to one split, independently.

    For every patient in `patients` (processed in `patient_id`-sorted
    order, never `dict`/`set` iteration order):

    1. compute the risk score r_i (`compute_risk_score`);
    2. compute the selection probability P(S_i=1|A_i,X_i)
       (`selection_probability`);
    3. draw S_i ~ Bernoulli(P(S_i=1|A_i,X_i)) using one shared
       `random.Random(seed)` instance (one uniform draw consumed per
       patient, in order) -- deterministic given `seed`.

    Patients with S_i = 1 are retained.

    Args:
        patients: the split's `PatientSet` (development, validation, or
            test -- this function applies the identical mechanism to
            whichever split it is given; only `seed` should differ across
            splits, per `decisions.md` Section 9 step 7).
        fitted_preprocessing: the development-fit
            `cohorts.preprocessing.FittedPreprocessing` used to standardize
            CD4/Karnofsky/Age for the risk score (never refit here, even
            when `patients` is the development split itself).
        seed: the split-specific seed. Callers must derive this via
            `infra.seeding.derive_seed(master_seed, split_name)`
            (`decisions.md` Section 9 step 7) -- see `pipeline.py` for the
            frozen order that performs this derivation once per split, so
            no random-number stream is reused across dev/val/test.
        coefficients: the coefficient bundle to use. Defaults to
            `DEFAULT_COEFFICIENTS`, preserving existing behavior unchanged
            -- see `compute_risk_score`'s `coefficients` docstring for when
            a non-default bundle (produced by a documented pilot
            recalibration) is appropriate.
        selection_severity: which frozen mechanism to apply -- `"primary"`
            (the default, `decisions.md` Section 9) or `"stronger"` (the
            Section 26 amendment's stronger-observational-selection
            coefficients, used only by the `stronger_selection_severity`
            sensitivity variant). When this is not `"primary"`, the
            coefficients are resolved from
            `coefficients_for_severity(selection_severity)`.

    Returns:
        An `ObservationalizedSplit` bundling the retained `PatientSet`,
        every considered patient's risk score and selection probability,
        the seed used, and the severity applied.

    Raises:
        ValueError: if `selection_severity` is unrecognized, or if a
            non-default `coefficients` bundle is passed together with a
            non-primary `selection_severity` -- the two would specify
            conflicting coefficients, and silently letting one win would
            hide which frozen mechanism actually ran.
    """
    resolved_coefficients = _resolve_coefficients(coefficients, selection_severity)

    sorted_patients = sorted(patients.patients, key=lambda record: record.patient_id)

    rng = random.Random(seed)

    risk_scores: dict[str, float] = {}
    selection_probabilities: dict[str, float] = {}
    retained_patients: list[PatientRecord] = []

    for patient in sorted_patients:
        risk_score = compute_risk_score(patient, fitted_preprocessing, resolved_coefficients)
        probability = selection_probability(
            risk_score, patient.treatment_arm, resolved_coefficients
        )

        risk_scores[patient.patient_id] = risk_score
        selection_probabilities[patient.patient_id] = probability

        draw = rng.random()
        if draw < probability:
            retained_patients.append(patient)

    return ObservationalizedSplit(
        retained=PatientSet(tuple(retained_patients)),
        risk_scores=risk_scores,
        selection_probabilities=selection_probabilities,
        seed=seed,
        selection_severity=selection_severity,
    )


# ---------------------------------------------------------------------------
# One-time pilot recalibration (`decisions.md` Section 9, "One-time pilot
# recalibration").
# ---------------------------------------------------------------------------
#
# `decisions.md` Section 9 permits the selection mechanism's coefficients to
# be recalibrated *exactly once*, and only during a documented pilot run on
# development data -- never triggered by, or applied via, validation or test
# results. The recalibration decision (old coefficients, new coefficients,
# reason) must be recorded in a changelog before the mechanism is frozen for
# real; after that freeze, no further recalibration is permitted.
#
# `PilotRecalibrator` implements this as an explicit, stateful mechanism
# layered on top of the frozen `DEFAULT_COEFFICIENTS` default -- it never
# mutates the module-level constants or `DEFAULT_COEFFICIENTS` itself.
# `.recalibrate()` returns a new `SelectionCoefficients` instance for the
# caller to thread through `compute_risk_score`/`selection_probability`/
# `observationalize`'s `coefficients` parameter in subsequent calls; ordinary
# callers that never pass `coefficients` are entirely unaffected.
#
# Enforcement of "development data only":
#     `.recalibrate()` requires its `development` argument to be a
#     `cohorts.preprocessing.DevelopmentSet` -- the exact same
#     "development-typed wrapper" pattern `FittedPreprocessing.fit()` uses
#     (see `preprocessing.py`'s module docstring). A bare `PatientSet` --
#     even a validation or test split, which is structurally identical to
#     the development split -- is rejected outright with
#     `RecalibrationRejectedError`, checked *before* the freeze check, so a
#     validation/test-typed call is never mistaken for (or silently
#     tolerated as) a legitimate but-frozen recalibration attempt.
#
# Enforcement of "no recalibration after freeze, even across new instances":
#     A plain instance-level freeze flag would let a caller route around a
#     frozen `PilotRecalibrator` simply by constructing a new one. To close
#     that gap without building a full persistent-freeze-file mechanism
#     (disproportionate for this stage -- task 16's real go/no-go checklist
#     supersedes this with real irreversible gating later, per this task's
#     brief), the freeze flag is tracked at the *class* level, keyed by the
#     resolved `changelog_path`: `.freeze()` records that path as frozen in
#     `PilotRecalibrator._frozen_changelog_paths`, a process-global
#     `ClassVar`. Any `PilotRecalibrator` instance -- old or newly
#     constructed -- pointed at the same `changelog_path` then refuses to
#     recalibrate. This is intentionally process-global, not
#     filesystem-persistent: a freeze does NOT survive a process restart.
#     That is a deliberate, documented limitation of this stage's
#     mechanism, not an oversight.


class RecalibrationRejectedError(TypeError):
    """Raised when `.recalibrate()` is attempted with anything other than a
    `cohorts.preprocessing.DevelopmentSet`.

    This is the enforcement mechanism for `decisions.md` Section 9's "never
    on validation or test": a bare `PatientSet` -- even a validation or test
    split, which is structurally identical to the development split -- is
    rejected outright, checked before any freeze-state check. Mirrors
    `cohorts.preprocessing.DevelopmentSetRequiredError`.
    """

    def __init__(self, received: object) -> None:
        self.received = received
        super().__init__(
            "PilotRecalibrator.recalibrate() requires a DevelopmentSet, got "
            f"{type(received).__name__!r}. Wrap the development split in "
            "DevelopmentSet(patients=...) to recalibrate on it; "
            "recalibration may never be attempted on, or triggered by, "
            "validation or test PatientSets, even though they are the same "
            "PatientSet type as development."
        )


class RecalibrationFreezeError(Exception):
    """Raised when `.recalibrate()` is attempted after the recalibration
    mechanism has been frozen (via `.freeze()`) for the recalibrator's
    `changelog_path`.

    `decisions.md` Section 9 permits recalibration exactly once; once
    `.freeze()` has been called for a given `changelog_path`, every
    subsequent `.recalibrate()` call against that same path raises this
    error -- including calls made on a brand-new `PilotRecalibrator`
    instance pointed at the same path (see module-level comment above
    `RecalibrationRejectedError` for how this is enforced across
    instances).
    """

    def __init__(self, changelog_path: Path) -> None:
        self.changelog_path = changelog_path
        super().__init__(
            "Pilot recalibration has already been frozen for changelog "
            f"{changelog_path}; decisions.md Section 9 permits exactly one "
            "recalibration, and no further recalibration is permitted "
            "after the freeze. Construct a new changelog_path only for a "
            "genuinely new, separately-documented experiment -- never to "
            "route around this freeze for the same mechanism."
        )


class PilotRecalibrator:
    """Implements `decisions.md` Section 9's one-time pilot recalibration.

    Recalibration is development-only (`.recalibrate()` rejects anything
    that isn't a `DevelopmentSet` -- see `RecalibrationRejectedError`) and
    permitted exactly once per `changelog_path` (subsequent calls after
    `.freeze()` raise `RecalibrationFreezeError` -- see that class's
    docstring for how the freeze is enforced even across new
    `PilotRecalibrator` instances pointed at the same `changelog_path`).

    Freeze-state persistence is process-global, not filesystem-persistent:
    a freeze does not survive a process restart. This is a deliberate,
    documented limitation appropriate to this stage of the pipeline --
    task 16's real go/no-go checklist supersedes this mechanism with real
    irreversible gating later. Every successful recalibration and every
    `.freeze()` call is still durably recorded in the JSON-Lines
    `changelog_path` file on disk (see `.recalibrate()`), so the *history*
    of what was recalibrated, and when the mechanism was frozen, survives
    process restarts even though the in-memory refusal-to-recalibrate-again
    does not.
    """

    #: Process-global set of `changelog_path`s (as resolved, absolute
    #: `Path`s) that have been frozen via `.freeze()`. Deliberately a
    #: *class* attribute (shared by every instance, including ones
    #: constructed after a freeze) rather than an instance attribute -- see
    #: the module-level comment above `RecalibrationRejectedError` for why:
    #: a caller must not be able to route around a frozen mechanism simply
    #: by constructing a fresh `PilotRecalibrator`.
    _frozen_changelog_paths: set[Path] = set()

    def _is_frozen(self, changelog_path: Path) -> bool:
        return changelog_path.resolve() in PilotRecalibrator._frozen_changelog_paths

    def freeze(self, changelog_path: Path) -> None:
        """Freeze pilot recalibration for `changelog_path`.

        After this call, every `.recalibrate()` call against
        `changelog_path` -- on this instance or any other
        `PilotRecalibrator` instance in this process -- raises
        `RecalibrationFreezeError`. Also appends a `"freeze"` marker record
        to `changelog_path` for on-disk auditability.
        """
        resolved = changelog_path.resolve()
        PilotRecalibrator._frozen_changelog_paths.add(resolved)

        changelog_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_record = {
            "event": "freeze",
            "changelog_path": str(changelog_path),
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        with changelog_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(freeze_record, sort_keys=True, ensure_ascii=True))
            handle.write("\n")

    def recalibrate(
        self,
        development: DevelopmentSet,
        new_coefficients: SelectionCoefficients,
        reason: str,
        changelog_path: Path,
        old_coefficients: SelectionCoefficients = DEFAULT_COEFFICIENTS,
    ) -> SelectionCoefficients:
        """Perform (and durably log) a one-time pilot recalibration.

        Args:
            development: a `DevelopmentSet` wrapping the development
                split's `PatientSet`. Anything else -- including a bare
                `PatientSet` such as an unwrapped validation or test split
                -- raises `RecalibrationRejectedError`, checked before any
                freeze-state check.
            new_coefficients: the recalibrated `SelectionCoefficients` to
                record and return.
            reason: a human-readable, documented reason for the
                recalibration (`decisions.md` Section 9 requires the
                "recalibration decision ... and reason" to be recorded).
            changelog_path: the JSON-Lines file to append the changelog
                entry to (and whose freeze state is checked/keyed).
            old_coefficients: the coefficients being replaced, recorded in
                the changelog entry for provenance. Defaults to
                `DEFAULT_COEFFICIENTS` (the frozen `decisions.md` Section 9
                values), since that is the coefficients bundle every
                caller starts from before its first (and only permitted)
                recalibration.

        Returns:
            `new_coefficients`, unchanged, for the caller to pass as the
            `coefficients` argument to `compute_risk_score`/
            `selection_probability`/`observationalize` in subsequent calls.

        Raises:
            RecalibrationRejectedError: if `development` is not a
                `DevelopmentSet`.
            RecalibrationFreezeError: if `changelog_path` has already been
                frozen (via `.freeze()`, on this instance or any other
                `PilotRecalibrator` instance in this process).
        """
        if not isinstance(development, DevelopmentSet):
            raise RecalibrationRejectedError(development)

        if self._is_frozen(changelog_path):
            raise RecalibrationFreezeError(changelog_path)

        entry = {
            "event": "recalibration",
            "old_coefficients": asdict(old_coefficients),
            "new_coefficients": asdict(new_coefficients),
            "reason": reason,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        changelog_path.parent.mkdir(parents=True, exist_ok=True)
        with changelog_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=True))
            handle.write("\n")

        return new_coefficients
