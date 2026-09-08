"""Development-only preprocessing fitting (Requirements 6.5, 6.6; Correctness
Property 2; `design.md` Section 3.4, Stage 6; `decisions.md` Section 9 step 4,
Section 10).

`decisions.md` Section 9's frozen order of operations requires that "all
preprocessing transformations (standardization/z-scoring of covariates used
in the risk score, imputation if any) [are] fit on the development split
only" (step 4), then "applied ... to every split" (step 5) unchanged.
`design.md` Section 3.4 names the resulting type `FittedPreprocessing` and
states its `.fit()` "only accepts `SplitCohorts.dev`."

Why a `DevelopmentSet` wrapper, not a bare `PatientSet`
--------------------------------------------------------

`src/cohorts/splitting.py`'s `SplitCohorts.dev` field is typed as a plain
`PatientSet` -- the exact same type as `.val` and `.test`. If
`FittedPreprocessing.fit()` accepted a bare `PatientSet`, nothing at the
type level would stop a caller from passing `split_cohorts.val` or
`split_cohorts.test` by mistake; the enforcement would live only in a
runtime `is this actually the dev split?` check (which `PatientSet` alone
cannot answer -- it carries no "which split am I" tag).

This module therefore introduces `DevelopmentSet`, a thin frozen wrapper
around a `PatientSet`, exactly analogous to `data.imputation.DevelopmentFold`
(see that module's docstring: "an explicit, visible marker that `records` is
development/training data"). `FittedPreprocessing.fit()` requires a
`DevelopmentSet` instance and raises `DevelopmentSetRequiredError` for
anything else -- including a bare `PatientSet` such as `split_cohorts.val`
or `split_cohorts.test`, which are structurally identical to
`split_cohorts.dev` and therefore cannot be rejected by type alone unless
wrapped. A caller must explicitly write
`DevelopmentSet(split_cohorts.dev)` to fit -- passing `split_cohorts.val`
unwrapped is a visible, conspicuous mismatch at the call site, not a silent
bug. This is the *same* "fit-only-on-development, enforced via a visible
wrapper" pattern `data.imputation.DevelopmentFold` already establishes for
imputation; this module applies it to preprocessing (standardization/
z-scoring of continuous baseline covariates) instead.

Chosen preprocessing operation: standardization (z-scoring)
-------------------------------------------------------------

`decisions.md` Section 9's baseline risk score consumes standardized
(z-scored) baseline covariates: $z(\\mathrm{CD4}_i)$, $z(\\mathrm{Karnofsky}_i)$,
$z(\\mathrm{Age}_i)$. `FittedPreprocessing` implements exactly this
operation -- per-field mean/stdev computed from development patients'
`timeline.static_covariates` only, for a configurable list of continuous
covariate fields (defaulting to those three risk-score inputs) -- and
applies the frozen parameters unchanged to any split at transform time.
This is a well-scoped, concrete choice consistent with "preprocessing" in
the causal/ML pipeline sense, and reuses the exact covariate fields the
observationalization mechanism (task 6.4) will need.

Why plain per-patient dict access, not a numeric-imputation-style tabular
projection
---------------------------------------------------------------------------

Unlike `data.imputation.FoldSafeImputer` (which intentionally decouples
itself from `PatientTimeline`'s shape by operating on a plain
`list[dict[str, float | None]]`), this module operates directly on
`cohorts.splitting.PatientSet`/`PatientRecord` because `.transform()` must
*return* a `PatientSet` of updated `PatientRecord`s (per `design.md`
Section 3.4: `.transform(self, split: PatientSet) -> PatientSet`) -- there
is no intermediate tabular projection step to decouple from here, since the
input and output shapes are the same domain type.

`PatientRecord` and `PatientTimeline` are both frozen dataclasses (see
`cohorts.cohort_construction` and `data.timelines`), so `.transform()`
produces modified copies via `dataclasses.replace()` rather than mutating
the originals -- the input `PatientSet`'s patients and their timelines are
never altered in place.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from cohorts.cohort_construction import PatientRecord
from cohorts.splitting import PatientSet

#: The default continuous covariate fields standardized by
#: `FittedPreprocessing` -- exactly the three baseline covariates
#: `decisions.md` Section 9's risk score z-scores ($z(\mathrm{CD4})$,
#: $z(\mathrm{Karnofsky})$, $z(\mathrm{Age})$), read from
#: `PatientRecord.timeline.static_covariates` under the concept_ids used
#: throughout this codebase (see `data.loaders.actg175.BaselineCovariateRecord`,
#: `data.loaders.actg175.DemographicsRecord`).
DEFAULT_CONTINUOUS_FIELDS: tuple[str, ...] = ("baseline_cd4", "karnofsky_score", "age")


class PreprocessingNotFittedError(Exception):
    """Raised when `.transform()` is called before `.fit()`."""

    def __init__(self) -> None:
        super().__init__(
            "FittedPreprocessing.transform() was called before .fit(). Call "
            ".fit(DevelopmentSet(...)) first."
        )


class DevelopmentSetRequiredError(TypeError):
    """Raised when `.fit()` is called with anything other than a `DevelopmentSet`.

    This is the enforcement mechanism for Requirements 6.5/6.6 (Correctness
    Property 2): preprocessing parameters must be computed from development
    data only. A bare `PatientSet` -- even `SplitCohorts.val` or
    `SplitCohorts.test`, which are structurally identical to
    `SplitCohorts.dev` -- is rejected outright. The caller must explicitly
    construct a `DevelopmentSet` to fit on it, which makes any accidental
    fit-on-validation/test call visible at the call site rather than
    silently wrong. Mirrors `data.imputation.DevelopmentFoldRequiredError`.
    """

    def __init__(self, received: object) -> None:
        self.received = received
        super().__init__(
            "FittedPreprocessing.fit() requires a DevelopmentSet, got "
            f"{type(received).__name__!r}. Wrap the development split in "
            "DevelopmentSet(patients=split_cohorts.dev) to fit on it; "
            "validation/test PatientSets must never be passed to .fit(), "
            "even though they are the same PatientSet type as dev."
        )


@dataclass(frozen=True)
class DevelopmentSet:
    """An explicit, visible marker that `patients` is the development split.

    Analogous to `data.imputation.DevelopmentFold`: `FittedPreprocessing.fit()`
    only accepts a `DevelopmentSet`, so a caller cannot pass validation or
    test `PatientSet`s to `.fit()` without explicitly (and visibly)
    constructing a `DevelopmentSet` around them first -- which is itself a
    conspicuous mislabeling, not an accident.

    Attributes:
        patients: the development split's `PatientSet` (typically
            `SplitCohorts.dev`).
    """

    patients: PatientSet


@dataclass(frozen=True)
class FittedPreprocessingParams:
    """The frozen, per-field standardization parameters produced by `.fit()`.

    Attributes:
        means: development-only per-field arithmetic mean of observed
            (non-`None`, numeric) `static_covariates` values.
        stdevs: development-only per-field sample standard deviation of
            observed values (`0.0` if fewer than two observed values, or if
            all observed values are identical).
    """

    means: Mapping[str, float]
    stdevs: Mapping[str, float]


def _numeric_value(patient: PatientRecord, field_name: str) -> float | None:
    raw_value = patient.timeline.static_covariates.get(field_name)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


@dataclass
class FittedPreprocessing:
    """Fits standardization (z-scoring) parameters on development data only.

    `.fit(DevelopmentSet(dev))` computes per-field mean/stdev from the
    development split's patients' `timeline.static_covariates` only
    (enforced by requiring a `DevelopmentSet` argument -- see
    `DevelopmentSetRequiredError`). `.transform(split)` then applies those
    *frozen* parameters, unchanged, to any split's patients (development,
    validation, or test) -- the fitted parameters are never recomputed from
    the split being transformed.

    Attributes:
        fields: the continuous covariate field names this preprocessor
            standardizes. Fields present in a patient's
            `static_covariates` but not listed here are passed through
            untouched by `.transform()`. Defaults to
            `DEFAULT_CONTINUOUS_FIELDS`.
    """

    fields: Sequence[str] = field(default_factory=lambda: DEFAULT_CONTINUOUS_FIELDS)
    _params: FittedPreprocessingParams | None = field(default=None, init=False, repr=False)

    def fit(self, development: DevelopmentSet) -> FittedPreprocessing:
        """Compute per-field standardization parameters from development data only.

        Args:
            development: a `DevelopmentSet` wrapping the development
                split's `PatientSet`. Passing anything else (including a
                bare `PatientSet`, e.g. an unwrapped `SplitCohorts.val` or
                `.test`) raises `DevelopmentSetRequiredError` -- this is
                the enforced "fit only on development data" contract.

        Returns:
            `self`, for chaining
            (`preprocessing = FittedPreprocessing().fit(DevelopmentSet(dev))`).

        Raises:
            DevelopmentSetRequiredError: if `development` is not a
                `DevelopmentSet`.
        """
        if not isinstance(development, DevelopmentSet):
            raise DevelopmentSetRequiredError(development)

        means: dict[str, float] = {}
        stdevs: dict[str, float] = {}
        for field_name in self.fields:
            observed_values = [
                value
                for patient in development.patients
                if (value := _numeric_value(patient, field_name)) is not None
            ]
            if not observed_values:
                continue
            means[field_name] = statistics.fmean(observed_values)
            stdevs[field_name] = (
                statistics.stdev(observed_values) if len(observed_values) >= 2 else 0.0
            )

        self._params = FittedPreprocessingParams(means=means, stdevs=stdevs)
        return self

    @property
    def fitted_params(self) -> FittedPreprocessingParams:
        """The frozen parameters computed by `.fit()`.

        Raises:
            PreprocessingNotFittedError: if `.fit()` has not been called yet.
        """
        if self._params is None:
            raise PreprocessingNotFittedError()
        return self._params

    def transform(self, split: PatientSet) -> PatientSet:
        """Apply the frozen, development-fitted parameters to any split's patients.

        The mean/stdev used here are exactly the ones produced by `.fit()`
        on the development split -- this method never recomputes them from
        `split`, regardless of which split (dev, val, or test) `split`
        belongs to.

        A field with no development-observed values at fit time (so no
        entry in `fitted_params.means`) is left untouched for every
        patient, rather than guessed. A field whose development-fitted
        `stdev` is `0.0` (fewer than two observed development values, or
        all identical) is standardized to `0.0` for every present numeric
        value, rather than dividing by zero. A patient whose value for a
        given field is missing (`None`) or non-numeric keeps that field
        untouched.

        Args:
            split: the `PatientSet` to transform (may be development,
                validation, or test -- `.transform()` itself does not
                distinguish between splits, only `.fit()` does).

        Returns:
            A new `PatientSet` with new `PatientRecord`s (via
            `dataclasses.replace()`) whose `timeline.static_covariates`
            have `self.fields` standardized; `split` and its patients and
            timelines are left unmodified.

        Raises:
            PreprocessingNotFittedError: if `.fit()` has not been called
                yet.
        """
        params = self.fitted_params

        transformed_patients = tuple(
            self._transform_one(patient, params) for patient in split.patients
        )
        return PatientSet(transformed_patients)

    def _transform_one(
        self, patient: PatientRecord, params: FittedPreprocessingParams
    ) -> PatientRecord:
        updated_covariates = dict(patient.timeline.static_covariates)
        for field_name in self.fields:
            if field_name not in params.means:
                # No development-observed values for this field at fit
                # time; nothing to standardize against, leave untouched.
                continue
            value = _numeric_value(patient, field_name)
            if value is None:
                # Missing or non-numeric for this patient; leave untouched.
                continue
            mean = params.means[field_name]
            stdev = params.stdevs.get(field_name, 0.0)
            updated_covariates[field_name] = (value - mean) / stdev if stdev > 0 else 0.0

        updated_timeline = replace(patient.timeline, static_covariates=updated_covariates)
        return replace(patient, timeline=updated_timeline)
