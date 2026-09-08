"""Cohort construction: the complete eligible arm-0/arm-1 randomized cohort.

Requirements 6.1, 7.1, 7.2. `decisions.md` Section 4 (target population),
Section 3-4 ("Arms 2 and 3 ... are excluded before any modeling, with no
exceptions"). `design.md` Section 3.4 names this module's output type,
`RandomizedCohort`, as the Stage-3 pipeline output that task 6.1's
`SealedReference` will be "constructible only from" — later modules
(`sealed_reference.py`, `splitting.py`, causal baselines) import and build
against `RandomizedCohort` and `PatientRecord` defined here. Their field
names are chosen to stay stable for those consumers; see the module-level
docstrings on each dataclass for exactly what is guaranteed.

Input shape
-----------

`data.loaders.actg175.RawTables` carries raw per-table rows (demographics,
treatment_arm, baseline_covariates, longitudinal_cd4_cd8, censoring,
outcomes) but no timelines; `data.timelines.build_patient_timeline`/
`build_patient_timelines` is what turns `ClinicalEvent`s into per-patient
`data.timelines.PatientTimeline`s. This module therefore takes that richer,
canonical timeline type as input — one per patient — plus a separate
`patient_id -> treatment_arm` mapping (since `data.timelines.PatientTimeline`
itself carries no arm field; arm assignment is a raw-table concept, not a
timeline bucket). This mirrors the pattern `design.md` Section 3.4 explicitly
anticipates ("a collection of per-patient `data.timelines.PatientTimeline`s
plus raw treatment-arm assignments, since `RawTables` alone doesn't carry
timelines").

Two small adapters bridge existing types without modifying them:

- `_adapt_to_eligibility_timeline`: `data.timelines.PatientTimeline.static_covariates`
  (a flat `concept_id -> value` mapping) into
  `protocols.eligibility_dsl.PatientTimeline.baseline_covariates` (the same
  flat shape, by construction) so `ACTG175_ELIGIBILITY.evaluate(...)` can run
  unchanged.
- `_adapt_to_endpoint_record`: derives a `protocols.endpoints.PatientEndpointRecord`
  from a `data.timelines.PatientTimeline`'s buckets (baseline CD4, the
  longitudinal CD4 lab series, AIDS-defining-event/death endpoint events, and
  time zero/last-observed timestamps) so `CompositeEndpoint.compute(...)` can
  run unchanged.

Why the DSL constant, not a spec-derived expression
-----------------------------------------------------

`build_randomized_cohort` takes `spec: TargetTrialSpec` (as the design
anticipates), but `TargetTrialSpec.population.eligibility_criteria` is a
list of *names*, not an executable expression tree — the one executable
tree ACTG175 needs is already frozen as
`protocols.eligibility_dsl.ACTG175_ELIGIBILITY` (`decisions.md` Section 4;
`design.md` Section 3.2's "no speculative operators" note). `spec` is kept
as an explicit parameter for provenance/consistency with the rest of the
pipeline's function signatures and for `spec.follow_up`-based checks a
caller may add later, even though the eligibility expression itself is not
re-derived from it.

Determinism
------------

`build_randomized_cohort` is a pure function of its inputs: patients are
processed in `patient_id`-sorted order (never relying on `dict`/`set`
iteration order), so repeated calls with the same input always produce the
same `RandomizedCohort.patients` tuple, membership, and counts — mirroring
the reproducible-sort convention already established by
`data.event_model.ClinicalEvent`'s total ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from data.timelines import PatientTimeline as DataPatientTimeline
from protocols.eligibility_dsl import ACTG175_ELIGIBILITY, EligibilityResult
from protocols.eligibility_dsl import PatientTimeline as EligibilityPatientTimeline
from protocols.endpoints import CompositeEndpoint, EndpointResult, PatientEndpointRecord
from protocols.schema import TargetTrialSpec

#: The two treatment arms retained by the randomized cohort (`decisions.md`
#: Section 3-4): arm 0 (zidovudine alone, comparator) and arm 1
#: (zidovudine+didanosine, intervention). Arms 2/3 are excluded "with no
#: exceptions" and never appear in a `RandomizedCohort`.
RETAINED_ARMS: frozenset[int] = frozenset({0, 1})

#: The `concept_id` under which a patient's baseline CD4 value is expected
#: to be recorded in `PatientTimeline.static_covariates` (matches
#: `protocols.eligibility_dsl.ACTG175_ELIGIBILITY`'s `Threshold("baseline_cd4", ...)`
#: field name and `data.loaders.actg175.BaselineCovariateRecord.baseline_cd4`).
BASELINE_CD4_CONCEPT_ID = "baseline_cd4"

#: The `concept_id` under which longitudinal (post-baseline) CD4 lab
#: measurements are expected to be recorded among
#: `PatientTimeline.timestamped_observations` (matches
#: `data.loaders.actg175.LongitudinalCD4CD8Record.cd4`, normalized to this
#: concept name; see `tests/unit/test_timelines.py`'s `LAB_CD4_WEEK20`
#: fixture for the established convention).
CD4_LAB_CONCEPT_ID = "cd4_count"

#: `concept_id`s recognized among `PatientTimeline.endpoint_events` as the
#: two non-CD4 composite-endpoint components (`decisions.md` Section 6).
AIDS_DEFINING_EVENT_CONCEPT_ID = "aids_defining_event"
DEATH_CONCEPT_ID = "death"


@dataclass(frozen=True)
class PatientRecord:
    """One eligible, arm-0/1-restricted patient in a `RandomizedCohort`.

    Field names are kept stable and purpose-built for downstream consumers
    (`design.md` Section 3.4: task 6.1's `SealedReference`, task 6.2's
    `splitting.py`, and the `causal`/`models` baselines) — do not rename
    without checking those call sites.

    Attributes:
        patient_id: the patient's stable identifier.
        treatment_arm: `0` (comparator, zidovudine alone) or `1`
            (intervention, zidovudine+didanosine). Never `2` or `3` — a
            `RandomizedCohort` never contains a non-retained arm.
        timeline: the patient's full `data.timelines.PatientTimeline`,
            carried through unchanged so downstream causal/endpoint
            analysis (adjustment-set covariates, longitudinal CD4/CD8,
            treatment/censoring events) has everything it needs without
            re-deriving it from raw tables.
        eligibility_result: the `protocols.eligibility_dsl.EligibilityResult`
            that admitted this patient (always `eligible=True` for a
            retained `PatientRecord` — kept for audit/provenance so a
            downstream report can show *why* the patient was retained, not
            just that they were).
        endpoint_result: the patient's composite time-to-event outcome,
            computed once here via `protocols.endpoints.CompositeEndpoint.compute`
            so later stages (causal baselines, sealed reference) never need
            to recompute it.
    """

    patient_id: str
    treatment_arm: int
    timeline: DataPatientTimeline
    eligibility_result: EligibilityResult
    endpoint_result: EndpointResult


@dataclass(frozen=True)
class RandomizedCohort:
    """The complete eligible arm-0/arm-1 randomized cohort (`design.md` Stage 3).

    Constructed only by `build_randomized_cohort` below. `patients` is
    always sorted by `patient_id` (ascending), so two `RandomizedCohort`s
    built from the same input are equal (`==`) and iterate identically —
    this is what makes repeated construction reproducible (Requirement
    7.2) and is the property task 6.1's `SealedReference` and task 6.2's
    `stratified_patient_split` depend on.
    """

    patients: tuple[PatientRecord, ...]

    def __len__(self) -> int:
        return len(self.patients)

    def __iter__(self):
        return iter(self.patients)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        """`patient_id`s in the same sorted order as `patients`."""
        return tuple(patient.patient_id for patient in self.patients)

    def by_arm(self, treatment_arm: int) -> tuple[PatientRecord, ...]:
        """Retained patients assigned to `treatment_arm` (`0` or `1`), sorted by `patient_id`."""
        return tuple(
            patient for patient in self.patients if patient.treatment_arm == treatment_arm
        )


def _adapt_to_eligibility_timeline(timeline: DataPatientTimeline) -> EligibilityPatientTimeline:
    """Bridge `data.timelines.PatientTimeline.static_covariates` into
    `protocols.eligibility_dsl.PatientTimeline.baseline_covariates`.

    Both are flat `concept_id -> value` mappings by construction, so this
    adapter is a direct copy — no renaming or reshaping is needed as long
    as the canonical timeline's baseline concept_ids (e.g. `"hiv_positive"`,
    `"baseline_cd4"`, `"age"`) match the field names
    `protocols.eligibility_dsl.ACTG175_ELIGIBILITY` looks up (they do, by
    the shared naming convention already used in
    `tests/unit/test_consistency_checker.py` and `tests/unit/test_timelines.py`).

    Neither `data.timelines.PatientTimeline` nor `protocols.eligibility_dsl.PatientTimeline`
    is modified by this function or module — it exists solely to translate
    between the two independently-owned types.
    """
    return EligibilityPatientTimeline(
        baseline_covariates=dict(timeline.static_covariates),
    )


def _earliest_treatment_time(timeline: DataPatientTimeline) -> datetime | None:
    """The patient's time-zero (earliest recorded treatment-start event), or `None`."""
    if not timeline.treatment_events:
        return None
    return min(event.observed_at for event in timeline.treatment_events)


def _last_observed_time(timeline: DataPatientTimeline) -> datetime | None:
    """The latest recorded observation across every timeline bucket, or `None`.

    Mirrors the "administrative end of follow-up, or last recorded
    observation" definition (`decisions.md` Section 5) as the latest
    timestamp actually present in the patient's record, across
    observations, treatment, endpoint, and censoring events alike.
    """
    all_times = [
        event.observed_at
        for bucket in (
            timeline.timestamped_observations,
            timeline.treatment_events,
            timeline.endpoint_events,
            timeline.censoring_events,
        )
        for event in bucket
    ]
    if not all_times:
        return None
    return max(all_times)


def _adapt_to_endpoint_record(timeline: DataPatientTimeline) -> PatientEndpointRecord | None:
    """Build a `protocols.endpoints.PatientEndpointRecord` from a canonical timeline.

    Derives every `PatientEndpointRecord` field from `timeline`'s existing
    buckets:

    - `baseline_cd4`: `timeline.static_covariates[BASELINE_CD4_CONCEPT_ID]`.
    - `assignment_time`: the earliest recorded treatment-start event
      (`decisions.md` Section 5: time zero is randomized treatment
      assignment).
    - `last_observed_time`: the latest recorded timestamp across every
      bucket (see `_last_observed_time`).
    - `cd4_observations`: every `timestamped_observations` entry whose
      `concept_id == CD4_LAB_CONCEPT_ID` and whose `value` is not missing,
      as `(observed_at, value)` pairs.
    - `aids_defining_event_time` / `death_time`: the earliest
      `endpoint_events` entry (if any) matching
      `AIDS_DEFINING_EVENT_CONCEPT_ID` / `DEATH_CONCEPT_ID` respectively.

    Returns `None` — rather than raising or guessing — when the timeline
    lacks the minimum information the composite endpoint needs (no baseline
    CD4 value, no recorded treatment-start time, or no recorded observation
    to serve as `last_observed_time`). This is exactly `decisions.md`
    Section 4's fifth eligibility condition, "have sufficient information to
    construct the primary composite endpoint" — a patient failing it is
    excluded by `build_randomized_cohort`, never fabricated a value.
    """
    baseline_cd4 = timeline.static_covariates.get(BASELINE_CD4_CONCEPT_ID)
    assignment_time = _earliest_treatment_time(timeline)
    last_observed_time = _last_observed_time(timeline)

    if baseline_cd4 is None or assignment_time is None or last_observed_time is None:
        return None

    cd4_observations = [
        (event.observed_at, event.value)
        for event in timeline.timestamped_observations
        if event.concept_id == CD4_LAB_CONCEPT_ID and event.value is not None
    ]

    aids_defining_event_times = [
        event.observed_at
        for event in timeline.endpoint_events
        if event.concept_id == AIDS_DEFINING_EVENT_CONCEPT_ID
    ]
    death_times = [
        event.observed_at
        for event in timeline.endpoint_events
        if event.concept_id == DEATH_CONCEPT_ID
    ]

    aids_defining_event_time = (
        min(aids_defining_event_times) if aids_defining_event_times else None
    )
    death_time = min(death_times) if death_times else None

    return PatientEndpointRecord(
        baseline_cd4=float(baseline_cd4),
        assignment_time=assignment_time,
        last_observed_time=last_observed_time,
        cd4_observations=cd4_observations,
        aids_defining_event_time=aids_defining_event_time,
        death_time=death_time,
    )


def build_randomized_cohort(
    timelines: Sequence[DataPatientTimeline] | Mapping[str, DataPatientTimeline],
    treatment_arms: Mapping[str, int],
    spec: TargetTrialSpec,
    precomputed_endpoint_results: Mapping[str, EndpointResult] | None = None,
) -> RandomizedCohort:
    """Construct the complete eligible arm-0/arm-1 randomized cohort.

    For each patient (processed in `patient_id`-sorted order, never in
    `dict`/`set` iteration order, so the result is reproducible):

    1. look up `treatment_arms[patient_id]`; exclude the patient if it is
       absent or not in `RETAINED_ARMS` (`decisions.md` Section 3-4: arms
       2/3 excluded "with no exceptions");
    2. adapt the timeline's baseline covariates and evaluate
       `protocols.eligibility_dsl.ACTG175_ELIGIBILITY`; exclude the patient
       if ineligible;
    3. adapt the timeline into a `PatientEndpointRecord`; exclude the
       patient if there is insufficient information to do so (see
       `_adapt_to_endpoint_record`);
    4. compute the composite endpoint via `CompositeEndpoint.compute` and
       retain the patient, with their `eligibility_result` and
       `endpoint_result` attached.

    Args:
        timelines: the candidate patients' `data.timelines.PatientTimeline`s,
            either as a plain sequence or as a `{patient_id: PatientTimeline}`
            mapping. Every `PatientTimeline.patient_id` must be unique.
        treatment_arms: `{patient_id: arm}` raw treatment-arm assignments
            (arm `0`-`3`), e.g. read from
            `data.loaders.actg175.RawTables.treatment_arm`. A patient absent
            from this mapping is excluded (treated as an invalid/unknown
            assignment, `decisions.md` Section 4 item 3: "valid baseline
            treatment ... records").
        spec: the target-trial specification. Not currently used to derive
            the eligibility expression (see module docstring for why), but
            accepted so this function's signature matches the rest of the
            pipeline's `(..., spec: TargetTrialSpec)` convention and is
            available for future spec-driven checks.

        precomputed_endpoint_results: optional `{patient_id: EndpointResult}`
            mapping. When supplied, a patient's composite-endpoint result is
            read directly from this mapping (a patient absent from it is
            treated exactly like `_adapt_to_endpoint_record` returning
            `None` -- excluded as having insufficient information for the
            primary composite endpoint, `decisions.md` Section 4 item 5) --
            never recomputed via `_adapt_to_endpoint_record`/
            `CompositeEndpoint.compute`. This is the real-data path: the
            packaged ACTG175 v3.2 raw `outcomes.csv` table supplies only a
            single composite event indicator/time (no per-component
            CD4-decline/AIDS-defining-event/death breakdown -- see
            `docs/data_access.md` Section 5 and `decisions.md` Section 26's
            2026-09-07 amendment), so a real-data caller (see
            `data.loaders.actg175_adapter.raw_tables_to_pipeline_inputs`)
            builds this mapping directly from that composite outcome rather
            than fabricating per-component times the source data does not
            provide. `None` (the default) preserves the original
            recompute-from-timeline-buckets behavior unchanged, exercised by
            every synthetic-fixture-based test in this suite.

    Returns:
        A `RandomizedCohort` containing exactly the eligible, arm-0/1-
        restricted patients, each with their composite-endpoint result
        attached, sorted by `patient_id`.
    """
    del spec  # Not currently consulted; see module docstring.

    timeline_list = list(timelines.values()) if isinstance(timelines, Mapping) else list(timelines)
    sorted_timelines = sorted(timeline_list, key=lambda timeline: timeline.patient_id)

    retained: list[PatientRecord] = []
    for timeline in sorted_timelines:
        arm = treatment_arms.get(timeline.patient_id)
        if arm not in RETAINED_ARMS:
            continue

        eligibility_timeline = _adapt_to_eligibility_timeline(timeline)
        eligibility_result = ACTG175_ELIGIBILITY.evaluate(eligibility_timeline)
        if not eligibility_result.eligible:
            continue

        if precomputed_endpoint_results is not None:
            endpoint_result = precomputed_endpoint_results.get(timeline.patient_id)
            if endpoint_result is None:
                continue
        else:
            endpoint_record = _adapt_to_endpoint_record(timeline)
            if endpoint_record is None:
                continue
            endpoint_result = CompositeEndpoint.compute(endpoint_record)

        retained.append(
            PatientRecord(
                patient_id=timeline.patient_id,
                treatment_arm=arm,
                timeline=timeline,
                eligibility_result=eligibility_result,
                endpoint_result=endpoint_result,
            )
        )

    return RandomizedCohort(patients=tuple(retained))
