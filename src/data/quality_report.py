"""The data-quality report (Requirement 5.14, `design.md` Section 3.3).

`design.md` Section 3.3 describes this module as producing a
`DataQualityReport` with "cohort size, missingness, treatment prevalence,
endpoint prevalence, follow-up distribution, event frequency, subgroup
counts, censoring, treatment switching." This module computes every one of
those fields directly from a cohort of `data.timelines.PatientTimeline`s
(reusing `data.imputation.missingness_indicators_from_timelines` for the
missingness computation rather than re-deriving it), plus two small,
optional, structured inputs a caller may supply when available:

- `endpoint_results`: `{patient_id: EndpointResult}` from
  `protocols.endpoints.CompositeEndpoint.compute` (Requirement 3.12). When
  supplied, `event_indicator`/`event_type`/`censoring_indicator` are read
  directly from it rather than re-derived from the timeline's own
  `endpoint_events`/`censoring_events` buckets. This matters because a raw
  `endpoint_events` bucket only ever contains concept-level clinical events
  actually recorded in the source data (e.g. an `aids_defining_event` or
  `death` record) — it cannot represent the `cd4_decline` component, which
  is a *derived* comparison against `baseline_cd4` computed by
  `CompositeEndpoint`, not a raw recorded event. Supplying `endpoint_results`
  is therefore the only way `event_frequency` can ever report a
  `cd4_decline` count.
- `subgroup_fields`: which fields to break `subgroup_counts` down by.
  `"treatment_arm"` is always available (derived from each patient's
  `treatment_events`, see `_treatment_arm_for` below); any other name is
  looked up directly in `PatientTimeline.static_covariates` (e.g.
  `"prior_art_status"`, if that concept is present in the cohort).

When `endpoint_results` is omitted, `event_indicator`/`censoring_indicator`
fall back to a direct read of the timeline's own buckets (a patient has an
event if `endpoint_events` is non-empty; the patient is censored if
`censoring_events` is non-empty and `endpoint_events` is empty) — a
reasonable, self-contained default that keeps `build_data_quality_report`
usable with nothing but timelines, at the cost of never reporting a
`cd4_decline` count in `event_frequency` (per the paragraph above).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from data.event_model import ClinicalEvent
from data.imputation import missingness_indicators_from_timelines
from data.timelines import PatientTimeline
from protocols.endpoints import EndpointResult

#: Recognized raw `endpoint_events` concept_ids that directly name one of
#: the three composite-endpoint components (`decisions.md` Section 6). Used
#: only by the fallback (no-`EndpointResult`-supplied) derivation in
#: `_resolve_event_status` to populate `event_type` when a raw event's
#: concept_id happens to spell out a component name explicitly (e.g. a
#: `"death"` or `"aids_defining_event"` record). This can never recognize
#: `"cd4_decline"` from a raw event alone, since that component is derived
#: (a comparison against baseline_cd4), not a recorded concept — see module
#: docstring.
_RECOGNIZED_RAW_EVENT_TYPES = frozenset({"aids_defining_event", "death"})

#: Default subgroup breakdown: treatment arm only. Callers with additional
#: available static covariates (e.g. `"prior_art_status"`) may pass a wider
#: tuple to `build_data_quality_report`.
DEFAULT_SUBGROUP_FIELDS: tuple[str, ...] = ("treatment_arm",)

#: Sentinel bucket label used when a subgroup field is absent for a patient
#: (no treatment event recorded, or the static covariate concept is
#: missing) — a patient is never silently dropped from `subgroup_counts`.
_UNKNOWN_SUBGROUP_VALUE = "unknown"


@dataclass(frozen=True)
class DataQualityReport:
    """The cohort-level data-quality report (`decisions.md`/Requirement 5.14).

    Attributes:
        cohort_size: number of patients in the cohort.
        missingness: per-(baseline)-field missingness rate, in `[0, 1]`,
            keyed by the static-covariate concept_id (e.g.
            `{"karnofsky_score": 0.1}`).
        treatment_prevalence: per-arm proportion of the cohort, keyed by the
            treatment concept_id observed in each patient's earliest
            `treatment_events` entry (e.g.
            `{"zidovudine": 0.5, "zidovudine+didanosine": 0.5}`), plus
            `"unknown"` for any patient with no recorded treatment event.
            Proportions sum to `1.0` (up to floating-point rounding) over a
            non-empty cohort.
        endpoint_prevalence: proportion of the cohort with
            `event_indicator=True`, in `[0, 1]`.
        follow_up: summary statistics (in days) of per-patient follow-up
            time, with keys `"mean"`, `"median"`, `"min"`, `"max"`. Follow-up
            time is the elapsed time from the patient's earliest recorded
            treatment event (or, absent one, their earliest event overall)
            to their latest recorded event across every timeline bucket.
        event_frequency: counts of each endpoint component
            (`"cd4_decline"`, `"aids_defining_event"`, `"death"`) among
            patients with `event_indicator=True`. Only components actually
            observed in the cohort appear as keys (an absent key means zero,
            not "not computed").
        subgroup_counts: per-subgroup-field breakdown, keyed by the
            subgroup field name (e.g. `"treatment_arm"`), each mapping that
            field's observed values to patient counts (e.g.
            `{"treatment_arm": {"zidovudine": 3, "zidovudine+didanosine": 3}}`).
        censoring: `{"censored_count": int, "censoring_rate": float}` — the
            number and proportion of the cohort with
            `censoring_indicator=True`.
        treatment_switching: `{"count": int, "rate": float}` — the number
            and proportion of the cohort whose `treatment_events` contain
            more than one distinct treatment concept_id (i.e. a recorded
            switch). `0`/`0.0` for a cohort with no tracked switching, which
            is an expected, valid value (`design.md` Section 3.3: "may be
            0/absent if not tracked in the synthetic input").
    """

    cohort_size: int
    missingness: dict[str, float]
    treatment_prevalence: dict[str, float]
    endpoint_prevalence: float
    follow_up: dict[str, float]
    event_frequency: dict[str, int]
    subgroup_counts: dict[str, dict[str, int]]
    censoring: dict[str, float]
    treatment_switching: dict[str, float]


def _timeline_list(
    timelines: Mapping[str, PatientTimeline] | Sequence[PatientTimeline],
) -> list[PatientTimeline]:
    return list(timelines.values()) if isinstance(timelines, Mapping) else list(timelines)


def _treatment_arm_for(timeline: PatientTimeline) -> str:
    """The concept_id of a patient's earliest recorded treatment event, or `"unknown"`."""
    if not timeline.treatment_events:
        return _UNKNOWN_SUBGROUP_VALUE
    earliest = min(timeline.treatment_events, key=lambda event: event.observed_at)
    return earliest.concept_id


def _all_events(timeline: PatientTimeline) -> list[ClinicalEvent]:
    return [
        *timeline.timestamped_observations,
        *timeline.treatment_events,
        *timeline.endpoint_events,
        *timeline.censoring_events,
    ]


def _follow_up_days(timeline: PatientTimeline) -> float:
    """Elapsed days from the patient's earliest treatment/first event to their latest event.

    Returns `0.0` for a patient with fewer than two events to span (e.g. no
    events at all, or a single event) — there is no observable follow-up
    duration in that case.
    """
    events = _all_events(timeline)
    if len(events) < 2:
        return 0.0

    start_time = (
        min(event.observed_at for event in timeline.treatment_events)
        if timeline.treatment_events
        else min(event.observed_at for event in events)
    )
    end_time = max(event.observed_at for event in events)
    return (end_time - start_time).total_seconds() / 86400.0


def _resolve_event_status(
    patient_id: str,
    timeline: PatientTimeline,
    endpoint_results: Mapping[str, EndpointResult] | None,
) -> tuple[bool, bool, str | None]:
    """Return `(event_indicator, censoring_indicator, event_type)` for one patient."""
    if endpoint_results is not None and patient_id in endpoint_results:
        result = endpoint_results[patient_id]
        return result.event_indicator, result.censoring_indicator, result.event_type

    event_indicator = len(timeline.endpoint_events) > 0
    censoring_indicator = not event_indicator and len(timeline.censoring_events) > 0

    event_type: str | None = None
    if event_indicator:
        # Fixed component order (cd4_decline, aids_defining_event, death),
        # matching CompositeEndpoint's tie-break order; only concept_ids
        # that directly name a recognized component can be identified this
        # way (see _RECOGNIZED_RAW_EVENT_TYPES docstring).
        recognized = [
            event.concept_id
            for event in timeline.endpoint_events
            if event.concept_id in _RECOGNIZED_RAW_EVENT_TYPES
        ]
        if recognized:
            event_type = min(
                recognized,
                key=lambda concept_id: ("aids_defining_event", "death").index(concept_id),
            )

    return event_indicator, censoring_indicator, event_type


def _subgroup_value(
    field_name: str, timeline: PatientTimeline
) -> str:
    if field_name == "treatment_arm":
        return _treatment_arm_for(timeline)
    value = timeline.static_covariates.get(field_name)
    return _UNKNOWN_SUBGROUP_VALUE if value is None else str(value)


def build_data_quality_report(
    timelines: Mapping[str, PatientTimeline] | Sequence[PatientTimeline],
    endpoint_results: Mapping[str, EndpointResult] | None = None,
    subgroup_fields: Sequence[str] = DEFAULT_SUBGROUP_FIELDS,
) -> DataQualityReport:
    """Compute the cohort-level `DataQualityReport` from `PatientTimeline`s.

    Args:
        timelines: the cohort's timelines, either keyed by `patient_id` or
            as a plain list (each timeline's own `patient_id` attribute is
            used regardless of input shape). Must be non-empty.
        endpoint_results: optional `{patient_id: EndpointResult}` from
            `protocols.endpoints.CompositeEndpoint.compute`. When a
            patient's `patient_id` is present, its `event_indicator`,
            `censoring_indicator`, and `event_type` are read from the
            supplied result rather than derived from the timeline's raw
            `endpoint_events`/`censoring_events` buckets (see module
            docstring for why this is the only way `event_frequency` can
            report a `"cd4_decline"` count). Patients absent from this
            mapping fall back to the direct-timeline derivation.
        subgroup_fields: which fields to report `subgroup_counts` for.
            `"treatment_arm"` is derived from each patient's
            `treatment_events`; any other name is looked up in
            `PatientTimeline.static_covariates`. Defaults to
            `("treatment_arm",)`.

    Returns:
        The computed `DataQualityReport`.

    Raises:
        ValueError: if `timelines` is empty.
    """
    timeline_list = _timeline_list(timelines)
    if not timeline_list:
        raise ValueError("build_data_quality_report requires a non-empty cohort of timelines.")

    cohort_size = len(timeline_list)

    # --- Missingness: reuse the existing helper rather than re-deriving it. ---
    baseline_fields = sorted(
        {concept_id for timeline in timeline_list for concept_id in timeline.static_covariates}
    )
    indicators_by_patient = missingness_indicators_from_timelines(timeline_list, baseline_fields)
    missingness = {
        field_name: statistics.fmean(
            indicators_by_patient[timeline.patient_id][field_name] for timeline in timeline_list
        )
        for field_name in baseline_fields
    }

    # --- Treatment prevalence. ---
    arm_counts = Counter(_treatment_arm_for(timeline) for timeline in timeline_list)
    treatment_prevalence = {arm: count / cohort_size for arm, count in arm_counts.items()}

    # --- Endpoint status per patient (event_indicator/censoring/event_type). ---
    statuses = [
        _resolve_event_status(timeline.patient_id, timeline, endpoint_results)
        for timeline in timeline_list
    ]

    endpoint_prevalence = sum(1 for indicator, _, _ in statuses if indicator) / cohort_size

    event_frequency: dict[str, int] = dict(
        Counter(
            event_type
            for indicator, _, event_type in statuses
            if indicator and event_type is not None
        )
    )

    censored_count = sum(1 for _, censoring_indicator, _ in statuses if censoring_indicator)
    censoring = {
        "censored_count": censored_count,
        "censoring_rate": censored_count / cohort_size,
    }

    # --- Follow-up summary statistics. ---
    follow_up_days = [_follow_up_days(timeline) for timeline in timeline_list]
    follow_up = {
        "mean": statistics.fmean(follow_up_days),
        "median": statistics.median(follow_up_days),
        "min": min(follow_up_days),
        "max": max(follow_up_days),
    }

    # --- Treatment switching: more than one distinct treatment concept_id. ---
    switch_count = sum(
        1
        for timeline in timeline_list
        if len({event.concept_id for event in timeline.treatment_events}) > 1
    )
    treatment_switching = {
        "count": switch_count,
        "rate": switch_count / cohort_size,
    }

    # --- Subgroup counts. ---
    subgroup_counts: dict[str, dict[str, int]] = {}
    for field_name in subgroup_fields:
        subgroup_counts[field_name] = dict(
            Counter(_subgroup_value(field_name, timeline) for timeline in timeline_list)
        )

    return DataQualityReport(
        cohort_size=cohort_size,
        missingness=missingness,
        treatment_prevalence=treatment_prevalence,
        endpoint_prevalence=endpoint_prevalence,
        follow_up=follow_up,
        event_frequency=event_frequency,
        subgroup_counts=subgroup_counts,
        censoring=censoring,
        treatment_switching=treatment_switching,
    )
