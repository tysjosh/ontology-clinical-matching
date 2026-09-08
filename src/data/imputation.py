"""Missingness indicators and fold-safe imputation (Requirement 5.12-5.13).

`decisions.md` Section 10 and `design.md` Section 3.4 both require that any
preprocessing fit on the cohort — imputation included — is fit on the
development/training split only and then applied unchanged to validation
and test. Task 6.3's `FittedPreprocessing` (per `design.md` Section 3.4)
will later enforce this same "fit() only accepts a development-typed
argument" contract for standardization/z-scoring; that class does not exist
yet (it is a later task). This module introduces its own minimal,
self-contained typed wrapper — `DevelopmentFold` — so the "fit only on
development data" contract is enforceable *now*, for imputation
specifically, without waiting on task 6.3. Task 6.3 may later reuse
`DevelopmentFold`, or introduce its own equivalent marker type for the
broader `PatientSet`/`SplitCohorts` typing scheme described in
`design.md` Section 3.4 — whichever convention task 6.3 adopts, this
module's contract (fit() rejects anything that is not explicitly
tagged as development data) is satisfied either way.

Why plain tabular input, not `PatientTimeline`
-----------------------------------------------

`design.md` Section 3.3 already gives `data.timelines.PatientTimeline` a
`missingness_mask: dict[str, MissingnessState]`, populated directly from
each `ClinicalEvent.missingness` at timeline-construction time. Re-deriving
missingness from timelines is a thin, direct read of that existing field
(`missingness_indicators_from_timelines` below does exactly this). But
`FoldSafeImputer` needs *numeric values* to compute per-field means, and
`PatientTimeline.static_covariates` is a loosely-typed `dict[str, Any]`
without a fixed field set — mixing timeline traversal with numeric
imputation logic would couple this module tightly to the timeline's
internal shape. Instead, `FoldSafeImputer` operates on a simpler,
self-describing tabular shape, `list[dict[str, float | None]]` (one dict
per patient, `None` meaning missing) — documented directly in `fit`'s and
`transform`'s docstrings. A caller holding `PatientTimeline`s can trivially
project `static_covariates` (or any other bucket) into this shape before
calling `FoldSafeImputer`; this module does not do that projection itself,
to keep it decoupled from `data.timelines`.

Dependency choice
-----------------

`pyproject.toml` does not currently depend on `numpy`, `pandas`, or
`scikit-learn`. Per the task instructions, this module therefore uses a
plain-Python implementation (the standard library's `statistics` module)
rather than adding a new dependency for mean/median/stdev computation or a
full `IterativeImputer`.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from data.event_model import MissingnessState
from data.timelines import PatientTimeline

Record = dict[str, float | None]

#: The two supported point-estimate strategies for `FoldSafeImputer`.
ImputationStrategy = Literal["mean", "median"]


class ImputerNotFittedError(Exception):
    """Raised when `.transform()` is called before `.fit()`."""

    def __init__(self) -> None:
        super().__init__(
            "FoldSafeImputer.transform() was called before .fit(). Call "
            ".fit(development_fold) first."
        )


class DevelopmentFoldRequiredError(TypeError):
    """Raised when `.fit()` is called with anything other than a `DevelopmentFold`.

    This is the enforcement mechanism for Requirement 5.12/6.5: imputation
    parameters must be computed from development/training data only. A
    bare, unlabeled `list[dict[...]]` (e.g. a validation or test fold that
    the caller forgot, or chose not, to relabel) is rejected outright — the
    caller must explicitly construct a `DevelopmentFold` to fit on it,
    which makes any accidental fit-on-validation/test call visible at the
    call site rather than silently wrong.
    """

    def __init__(self, received: object) -> None:
        self.received = received
        super().__init__(
            "FoldSafeImputer.fit() requires a DevelopmentFold, got "
            f"{type(received).__name__!r}. Wrap the development-split "
            "records in DevelopmentFold(records=...) to fit on them; "
            "validation/test data must never be passed to .fit()."
        )


@dataclass(frozen=True)
class DevelopmentFold:
    """An explicit, visible marker that `records` is development/training data.

    This is the "unambiguous development data" wrapper described in the
    task: `FoldSafeImputer.fit()` only accepts a `DevelopmentFold`, so a
    caller cannot pass validation or test records to `.fit()` without
    explicitly (and visibly) constructing a `DevelopmentFold` around them
    first — which is itself a conspicuous mislabeling, not an accident.

    Attributes:
        records: development-split patient records, one `dict` per patient,
            each mapping field name to a numeric value or `None` (missing).
    """

    records: Sequence[Record]


@dataclass(frozen=True)
class FittedImputationParams:
    """The frozen, per-field imputation parameters produced by `.fit()`.

    Attributes:
        means: development-only per-field arithmetic mean of observed
            (non-`None`) values.
        medians: development-only per-field median of observed values.
        stdevs: development-only per-field sample standard deviation of
            observed values (0.0 if fewer than two observed values), used
            only to draw stochastic multiple-imputation values around the
            point estimate.
        strategy: which of `means`/`medians` `.transform()`'s point-estimate
            fill uses.
    """

    means: Mapping[str, float]
    medians: Mapping[str, float]
    stdevs: Mapping[str, float]
    strategy: ImputationStrategy


@dataclass
class FoldSafeImputer:
    """Fits per-field imputation parameters on development data only.

    `.fit(development)` computes per-field mean/median/stdev from
    development-tagged records only (enforced by requiring a
    `DevelopmentFold` argument — see `DevelopmentFoldRequiredError`).
    `.transform(records)` then applies those *frozen* parameters, unchanged,
    to any fold's records (development, validation, or test) — the fitted
    parameters are never recomputed from the fold being transformed.

    Imputation strategy: simple mean or median point-estimate imputation
    (`strategy`, default `"mean"`). When `n_imputations > 1` is requested at
    transform time, this becomes a simple multiple-imputation scheme: each
    of the `n_imputations` draws fills a missing value with the frozen point
    estimate plus Gaussian noise drawn from `Normal(0, stdev)`, using the
    frozen development-fitted `stdev` for that field. This is documented
    explicitly as a simple stochastic-draws-around-the-mean/median
    approach, not a full MICE/chained-equations implementation (not
    required by the task).

    Attributes:
        fields: the field names this imputer fits/imputes. Fields present
            in a record but not listed here are passed through untouched
            by `.transform()`.
        strategy: `"mean"` (default) or `"median"` point-estimate fill.
    """

    fields: Sequence[str]
    strategy: ImputationStrategy = "mean"
    _params: FittedImputationParams | None = field(default=None, init=False, repr=False)

    def fit(self, development: DevelopmentFold) -> FoldSafeImputer:
        """Compute per-field imputation parameters from development data only.

        Args:
            development: a `DevelopmentFold` wrapping development-split
                records. Passing a bare `list`/`Sequence` (anything that is
                not a `DevelopmentFold`) raises `DevelopmentFoldRequiredError`
                — this is the enforced "fit only on development data"
                contract.

        Returns:
            `self`, for chaining (`imputer = FoldSafeImputer(fields).fit(dev)`).

        Raises:
            DevelopmentFoldRequiredError: if `development` is not a
                `DevelopmentFold`.
        """
        if not isinstance(development, DevelopmentFold):
            raise DevelopmentFoldRequiredError(development)

        means: dict[str, float] = {}
        medians: dict[str, float] = {}
        stdevs: dict[str, float] = {}
        for field_name in self.fields:
            observed_values = [
                record[field_name]
                for record in development.records
                if record.get(field_name) is not None
            ]
            if not observed_values:
                continue
            means[field_name] = statistics.fmean(observed_values)
            medians[field_name] = statistics.median(observed_values)
            stdevs[field_name] = (
                statistics.stdev(observed_values) if len(observed_values) >= 2 else 0.0
            )

        self._params = FittedImputationParams(
            means=means, medians=medians, stdevs=stdevs, strategy=self.strategy
        )
        return self

    @property
    def fitted_params(self) -> FittedImputationParams:
        """The frozen parameters computed by `.fit()`.

        Raises:
            ImputerNotFittedError: if `.fit()` has not been called yet.
        """
        if self._params is None:
            raise ImputerNotFittedError()
        return self._params

    def transform(
        self,
        records: Sequence[Record],
        n_imputations: int = 1,
        seed: int | None = None,
    ) -> list[list[Record]] | list[Record]:
        """Apply the frozen, development-fitted parameters to any fold's records.

        The parameters used here are exactly the ones produced by `.fit()`
        on development data — this method never recomputes them from
        `records`, regardless of which fold `records` belongs to (dev, val,
        or test).

        Args:
            records: the records to impute (may be development,
                validation, or test — `.transform()` itself does not
                distinguish between folds, only `.fit()` does).
            n_imputations: number of multiple-imputation draws to produce.
                `1` (the default) returns a single imputed dataset filled
                with the frozen point estimate (mean or median, per
                `strategy`). `> 1` returns that many stochastically varied
                datasets (see class docstring), for simple multiple
                imputation.
            seed: optional seed for the stochastic draws used when
                `n_imputations > 1`. Ignored when `n_imputations == 1`
                (point-estimate imputation is deterministic).

        Returns:
            If `n_imputations == 1`: a single `list[Record]`, one dict per
            input record, with `None` values in `self.fields` replaced by
            the fitted point estimate. If `n_imputations > 1`: a
            `list[list[Record]]` of that many imputed datasets.

        Raises:
            ImputerNotFittedError: if `.fit()` has not been called yet.
        """
        params = self.fitted_params
        point_estimates = params.means if params.strategy == "mean" else params.medians

        if n_imputations == 1:
            return [
                self._impute_one(record, point_estimates, params.stdevs, rng=None)
                for record in records
            ]

        rng = random.Random(seed)
        return [
            [
                self._impute_one(record, point_estimates, params.stdevs, rng=rng)
                for record in records
            ]
            for _ in range(n_imputations)
        ]

    def _impute_one(
        self,
        record: Record,
        point_estimates: Mapping[str, float],
        stdevs: Mapping[str, float],
        rng: random.Random | None,
    ) -> Record:
        imputed = dict(record)
        for field_name in self.fields:
            if imputed.get(field_name) is not None:
                continue
            if field_name not in point_estimates:
                # No development-observed values for this field at fit time;
                # nothing to impute from, leave as missing rather than guess.
                continue
            point_estimate = point_estimates[field_name]
            if rng is None:
                imputed[field_name] = point_estimate
            else:
                stdev = stdevs.get(field_name, 0.0)
                if stdev > 0:
                    imputed[field_name] = point_estimate + rng.gauss(0.0, stdev)
                else:
                    imputed[field_name] = point_estimate
        return imputed


def missingness_indicators_from_records(
    records: Sequence[Record], fields: Sequence[str]
) -> list[dict[str, int]]:
    """Build a companion missingness-indicator dict per record, for plain tabular input.

    Args:
        records: one `dict` per patient, mapping field name to a numeric
            value or `None` (missing).
        fields: the field names to produce indicators for.

    Returns:
        One `dict[str, int]` per input record (same order), mapping each
        field in `fields` to `1` if that record's value is `None`
        (missing) or `0` if observed.
    """
    return [
        {field_name: 1 if record.get(field_name) is None else 0 for field_name in fields}
        for record in records
    ]


def missingness_indicators_from_timelines(
    timelines: Mapping[str, PatientTimeline] | Sequence[PatientTimeline],
    fields: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Build a companion missingness-indicator dict per patient, from `PatientTimeline`s.

    A thin, direct read of `PatientTimeline.missingness_mask` (populated at
    timeline-construction time in `data.timelines.build_patient_timeline`
    from each contributing `ClinicalEvent.missingness`): a field is
    considered missing for a patient if that patient's
    `missingness_mask` has no entry for it, or if the recorded
    `MissingnessState` is `MISSING` or `NOT_APPLICABLE`. A field explicitly
    recorded as `OBSERVED` is `0`.

    Note on keying: per `data.timelines`'s documented keying rule, static
    (baseline) covariates are keyed by `concept_id` alone in
    `missingness_mask`, while every other bucket is keyed by
    `f"{concept_id}@{observed_at.isoformat()}"`. This helper looks fields up
    by bare `concept_id` (matching the static-covariate keying), since
    imputation of non-time-varying baseline fields is this module's
    intended use (`decisions.md` Section 13's adjustment set is all
    baseline covariates). A field name that does not match any static-key
    entry is treated as missing, since no baseline value for it was
    observed.

    Args:
        timelines: the cohort's timelines, either keyed by `patient_id` or
            as a plain list (each timeline's own `patient_id` attribute is
            used regardless of input shape).
        fields: the (baseline) concept_ids to produce indicators for.

    Returns:
        `{patient_id: {field_name: 1 or 0}}`.
    """
    timeline_list = list(timelines.values()) if isinstance(timelines, Mapping) else list(timelines)

    result: dict[str, dict[str, int]] = {}
    for timeline in timeline_list:
        indicators: dict[str, int] = {}
        for field_name in fields:
            state = timeline.missingness_mask.get(field_name)
            is_missing = state is None or state in (
                MissingnessState.MISSING,
                MissingnessState.NOT_APPLICABLE,
            )
            indicators[field_name] = 1 if is_missing else 0
        result[timeline.patient_id] = indicators
    return result
