"""Adapter: real ACTG175 `RawTables` -> `ClinicalEvent`/`PatientTimeline`
(task: real-data pipeline wiring).

`src.data.loaders.actg175.load_actg175` returns `RawTables` -- six typed,
schema-validated tuples of per-table Pydantic records. Every other pipeline
stage (`cohorts.cohort_construction.build_randomized_cohort`,
`cohorts.pipeline.run_observationalization_pipeline`, and beyond) consumes
`data.timelines.PatientTimeline`s plus a `{patient_id: treatment_arm}`
mapping, built from `data.event_model.ClinicalEvent`s -- never `RawTables`
directly. Before this module, nothing in the production codebase performed
that conversion for real data; every existing pipeline entry point
(`scripts/run_stage.py`'s `full_pipeline` stage,
`tests/integration/test_run_ablations.py`, `tests/unit/test_pipeline.py`)
only ever exercised a hand-built *synthetic* `ClinicalEvent` fixture. This
module is the real, production adapter.

Treatment-arm handling (arm is authoritative)
-----------------------------------------------

`decisions.md` Section 3-4 retains only arms 0 (zidovudine alone) and 1
(zidovudine+didanosine); arms 2/3 (zalcitabine-containing regimens) are
"excluded before any modeling, with no exceptions." This module enforces
that exclusion at the earliest possible point: `TreatmentArmRecord.arm`
(an `int`, `0`-`3`) is the single authoritative field this module reads to
decide inclusion and which frozen treatment concept_id
(`TREATMENT_CONCEPT_ID_BY_ARM`) to emit -- never `TreatmentArmRecord.treatment_label`
(a free-text string from the source distribution). A patient whose `arm`
is `2` or `3` is dropped before any `ClinicalEvent` is built for them at
all -- they are never normalized into a treatment-modality event, matching
`cohorts.cohort_construction.RETAINED_ARMS`'s own arm-0/1-only contract
one layer earlier than that module enforces it.

`treatment_label` is retained only for a documented consistency check
(`assert_arm_label_consistent`) and, when requested, as inert provenance --
it is never used to decide inclusion or to choose the emitted treatment
concept_id. `assert_arm_label_consistent` raises
`ArmLabelInconsistentError` (not silently ignored) if a retained (arm 0/1)
row's `treatment_label` does not match the frozen expected label for that
arm, since a mismatch there would indicate the source distribution's `arm`
and `treatment_label` columns have drifted out of sync -- a data-integrity
concern worth surfacing loudly, even though `arm` alone still governs
inclusion and concept_id selection.

Composite-endpoint handling (real data has no per-component breakdown)
--------------------------------------------------------------------------

See `docs/data_access.md` Section 5 and `decisions.md` Section 26's
2026-09-07 amendment: the real ACTG175 v3.2 raw `outcomes.csv` table
supplies only a single composite `event_indicator`/`event_time_days` pair
per patient, with `event_type` always `None` or the literal string
`"composite_event_unspecified"` -- there is no CD4-decline/AIDS-defining-
event/death breakdown anywhere in this data release.
`protocols.endpoints.CompositeEndpoint.compute` cannot be run meaningfully
against this shape (it requires per-component candidate times to pick the
earliest one); recomputing it from fabricated component times would
silently encode information the source data does not contain. This module
therefore builds `protocols.endpoints.EndpointResult` directly from each
patient's `OutcomeRecord`/`CensoringRecord` pair, with `event_type` always
`None` (honestly "unspecified composite" rather than guessing a component),
and passes the resulting `{patient_id: EndpointResult}` mapping to
`cohorts.cohort_construction.build_randomized_cohort`'s
`precomputed_endpoint_results` parameter -- never through
`CompositeEndpoint.compute`. This is a real-data-only code path: synthetic-
fixture-based tests continue to exercise the original per-component
recomputation path unchanged (`precomputed_endpoint_results=None`, the
default).

`OutcomeRecord.event_indicator` and `CensoringRecord.censored` are exact
logical complements in the real ACTG175 v3.2 distribution (verified at
ingestion: every patient's `event_indicator == (not censored)` and
`event_time_days == censoring_time_days`), so this module asserts that
invariant explicitly (`_build_endpoint_result`) rather than silently
trusting one table over the other.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from data.event_model import ClinicalEvent, MissingnessState
from data.loaders.actg175 import (
    BaselineCovariateRecord,
    CensoringRecord,
    DemographicsRecord,
    LongitudinalCD4CD8Record,
    OutcomeRecord,
    RawTables,
    TreatmentArmRecord,
)
from data.timelines import PatientTimeline, build_patient_timeline
from protocols.endpoints import EndpointResult

#: The two retained treatment arms and their frozen treatment concept_ids
#: (`decisions.md` Section 3; `configs/target_trials/main.yaml`'s
#: `intervention.concept_id`/`comparator.concept_id`). `arm` is the sole
#: authoritative field used to select from this mapping -- see module
#: docstring.
TREATMENT_CONCEPT_ID_BY_ARM: dict[int, str] = {
    0: "zidovudine",
    1: "zidovudine+didanosine",
}

#: Arms excluded before any event construction (`decisions.md` Section 3-4:
#: "excluded before any modeling, with no exceptions").
_EXCLUDED_ARMS: frozenset[int] = frozenset({2, 3})

#: Expected raw `treatment_label` values for the two retained arms, used
#: only by `assert_arm_label_consistent`'s documented consistency check --
#: never for inclusion/concept_id decisions (see module docstring).
_EXPECTED_LABEL_BY_ARM: dict[int, str] = {
    0: "zidovudine",
    1: "zidovudine+didanosine",
}

#: A fixed reference calendar date used as this adapter's time-zero anchor
#: (randomization / treatment assignment, `decisions.md` Section 5). The
#: real ACTG175 v3.2 distribution's tables record only *elapsed* time
#: (`age` in years, `*_days` durations from randomization), never absolute
#: calendar dates -- so this module anchors every patient's randomization
#: at the same fixed, arbitrary UTC instant and expresses every other
#: timestamp as an offset from it via `timedelta`. This preserves every
#: relative-time computation the rest of the pipeline performs (elapsed
#: days/weeks since assignment, ordering of events) exactly, since those
#: computations only ever depend on *differences* between timestamps, never
#: on absolute calendar dates. Not a `decisions.md`-frozen value -- an
#: arbitrary, documented anchor.
ASSIGNMENT_TIME_ANCHOR: datetime = datetime(2000, 1, 1, 0, 0, 0)

#: Longitudinal CD4/CD8 landmark weeks present in `longitudinal_cd4_cd8.csv`
#: (`decisions.md` Section 5: baseline, week 20, week 96).
_LANDMARK_WEEK_OFFSETS: dict[int, timedelta] = {
    0: timedelta(weeks=0),
    20: timedelta(weeks=20),
    96: timedelta(weeks=96),
}


class ArmLabelInconsistentError(Exception):
    """Raised by `assert_arm_label_consistent` when a retained (arm 0/1)
    patient's raw `treatment_label` does not match the frozen expected
    label for their `arm`.

    `arm` alone governs inclusion and concept_id selection (see module
    docstring) -- this error exists only to surface a source-distribution
    data-integrity concern loudly, never to change which patients are
    retained or which concept_id they are assigned.
    """

    def __init__(self, patient_id: str, arm: int, treatment_label: str, expected: str) -> None:
        self.patient_id = patient_id
        self.arm = arm
        self.treatment_label = treatment_label
        self.expected = expected
        super().__init__(
            f"patient_id={patient_id!r}: arm={arm!r} has treatment_label="
            f"{treatment_label!r}, expected {expected!r}. `arm` remains "
            "authoritative for inclusion/concept_id selection regardless of "
            "this mismatch -- see actg175_adapter module docstring -- but a "
            "label/arm mismatch in the source distribution should be "
            "investigated before trusting this ingestion."
        )


def assert_arm_label_consistent(record: TreatmentArmRecord) -> None:
    """Raise `ArmLabelInconsistentError` if a retained arm's raw
    `treatment_label` does not match the frozen expected label.

    Only checked for arms 0/1 (the retained arms); arms 2/3 have no frozen
    expected label here since they are excluded before this check would
    matter for modeling purposes. Never called on the inclusion/concept_id
    decision path itself -- `arm` alone governs that (see module
    docstring).
    """
    expected = _EXPECTED_LABEL_BY_ARM.get(record.arm)
    if expected is not None and record.treatment_label != expected:
        raise ArmLabelInconsistentError(
            record.patient_id, record.arm, record.treatment_label, expected
        )


def _build_endpoint_result(outcome: OutcomeRecord, censoring: CensoringRecord) -> EndpointResult:
    """Build one patient's `EndpointResult` directly from their real
    composite `OutcomeRecord`/`CensoringRecord` pair (see module docstring
    -- never via `CompositeEndpoint.compute`, since the real data provides
    no per-component breakdown).

    Asserts the two tables' logical-complement invariant
    (`event_indicator == (not censored)`, `event_time_days ==
    censoring_time_days`) rather than silently trusting one table over the
    other; both hold across the entire real ACTG175 v3.2 distribution as
    ingested (verified at ingestion time).
    """
    if outcome.event_indicator == censoring.censored:
        raise ValueError(
            f"patient_id={outcome.patient_id!r}: outcomes.event_indicator="
            f"{outcome.event_indicator!r} is not the logical complement of "
            f"censoring.censored={censoring.censored!r}."
        )
    if outcome.event_time_days != censoring.censoring_time_days:
        raise ValueError(
            f"patient_id={outcome.patient_id!r}: outcomes.event_time_days="
            f"{outcome.event_time_days!r} does not match "
            f"censoring.censoring_time_days={censoring.censoring_time_days!r}."
        )

    event_time = ASSIGNMENT_TIME_ANCHOR + timedelta(days=outcome.event_time_days)
    return EndpointResult(
        event_time=event_time,
        event_indicator=outcome.event_indicator,
        censoring_indicator=censoring.censored,
        # The real data release never breaks the composite down into its
        # three components (see module docstring) -- always None, honestly
        # "unspecified composite" rather than a guessed component.
        event_type=None,
    )


def _demographics_events(record: DemographicsRecord) -> list[ClinicalEvent]:
    """Baseline demographic covariates -> `modality="demographic"` events
    (`decisions.md` Section 13 adjustment set)."""
    fields: list[tuple[str, float | str | bool, str | None]] = [
        ("age", record.age, "years"),
        ("sex", record.sex, None),
        ("race_ethnicity", record.race_ethnicity, None),
        ("weight_kg", record.weight_kg, "kg"),
        ("hemophilia", record.hemophilia, None),
        ("homosexual_activity", record.homosexual_activity, None),
        ("iv_drug_use_history", record.iv_drug_use_history, None),
    ]
    return [
        ClinicalEvent(
            patient_id=record.patient_id,
            encounter_id=None,
            concept_id=concept_id,
            value=value,
            unit=unit,
            modality="demographic",
            observed_at=ASSIGNMENT_TIME_ANCHOR,
            valid_from=None,
            valid_to=None,
            source="actg175",
            missingness=MissingnessState.OBSERVED,
            provenance="loader:actg175",
        )
        for concept_id, value, unit in fields
    ]


def _baseline_covariate_events(record: BaselineCovariateRecord) -> list[ClinicalEvent]:
    """Baseline clinical covariates -> `modality="baseline"` events
    (`decisions.md` Section 13 adjustment set)."""
    fields: list[tuple[str, float | bool, str | None]] = [
        ("karnofsky_score", record.karnofsky_score, "score"),
        ("prior_opportunistic_infection", record.prior_opportunistic_infection, None),
        ("symptomatic_status", record.symptomatic_status, None),
        ("baseline_cd4", record.baseline_cd4, "cells/mm3"),
        ("baseline_cd8", record.baseline_cd8, "cells/mm3"),
        ("prior_art_exposure_days", record.prior_art_exposure_days, "days"),
        # hiv_positive is not a distinct column in the real distribution
        # (ACTG175's own entry criteria required documented HIV-1 infection
        # for every enrolled patient, per `protocols.eligibility_dsl`'s
        # module docstring); every real, packaged row is HIV-positive by
        # construction, so this adapter records it as an always-True
        # baseline fact rather than leaving `ACTG175_ELIGIBILITY`'s
        # `Threshold("hiv_positive", "==", True)` unsatisfiable for real
        # patients.
        ("hiv_positive", True, None),
    ]
    return [
        ClinicalEvent(
            patient_id=record.patient_id,
            encounter_id=None,
            concept_id=concept_id,
            value=value,
            unit=unit,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME_ANCHOR,
            valid_from=None,
            valid_to=None,
            source="actg175",
            missingness=MissingnessState.OBSERVED,
            provenance="loader:actg175",
        )
        for concept_id, value, unit in fields
    ]


def _treatment_event(patient_id: str, arm: int) -> ClinicalEvent:
    """The single treatment-assignment event for a retained (arm 0/1)
    patient, using the frozen concept_id `TREATMENT_CONCEPT_ID_BY_ARM[arm]`
    -- `arm` is the sole input governing which concept_id is used; see
    module docstring."""
    concept_id = TREATMENT_CONCEPT_ID_BY_ARM[arm]
    return ClinicalEvent(
        patient_id=patient_id,
        encounter_id=None,
        concept_id=concept_id,
        value=1.0,
        unit=None,
        modality="treatment",
        observed_at=ASSIGNMENT_TIME_ANCHOR,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )


def _longitudinal_events(record: LongitudinalCD4CD8Record) -> list[ClinicalEvent]:
    """Longitudinal CD4/CD8 landmark measurements -> `modality="lab"` events.

    A `None` (missing) `cd4`/`cd8` value is still emitted as an event, with
    `missingness=MissingnessState.MISSING` -- never silently dropped, per
    `data.timelines`'s missingness-mask convention (Requirement 5.12-5.13).
    """
    if record.week not in _LANDMARK_WEEK_OFFSETS:
        raise ValueError(
            f"patient_id={record.patient_id!r}: unrecognized longitudinal "
            f"week {record.week!r}; expected one of "
            f"{sorted(_LANDMARK_WEEK_OFFSETS)!r} (decisions.md Section 5)."
        )
    observed_at = ASSIGNMENT_TIME_ANCHOR + _LANDMARK_WEEK_OFFSETS[record.week]

    events: list[ClinicalEvent] = []
    for concept_id, value in (("cd4_count", record.cd4), ("cd8_count", record.cd8)):
        events.append(
            ClinicalEvent(
                patient_id=record.patient_id,
                encounter_id=None,
                concept_id=concept_id,
                value=value,
                unit="cells/mm3",
                modality="lab",
                observed_at=observed_at,
                valid_from=None,
                valid_to=None,
                source="actg175",
                missingness=(
                    MissingnessState.MISSING if value is None else MissingnessState.OBSERVED
                ),
                provenance="loader:actg175",
            )
        )
    return events


def _censoring_event(record: CensoringRecord) -> ClinicalEvent:
    """The administrative end-of-follow-up/censoring event.

    Emitted for every patient regardless of `record.censored` -- this event
    marks the last-recorded-observation/administrative-censoring timestamp
    (`decisions.md` Section 5), which `cohorts.cohort_construction`'s
    `_last_observed_time` reads across every timeline bucket including this
    one, independent of whether the patient went on to have the composite
    endpoint event.
    """
    observed_at = ASSIGNMENT_TIME_ANCHOR + timedelta(days=record.censoring_time_days)
    return ClinicalEvent(
        patient_id=record.patient_id,
        encounter_id=None,
        concept_id="administrative_censoring",
        value=1.0,
        unit=None,
        modality="censoring",
        observed_at=observed_at,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )


def raw_tables_to_pipeline_inputs(
    tables: RawTables,
) -> tuple[list[PatientTimeline], dict[str, int], dict[str, EndpointResult]]:
    """Convert real ACTG175 `RawTables` into `(timelines, treatment_arms,
    endpoint_results)`, ready for
    `cohorts.cohort_construction.build_randomized_cohort`/
    `cohorts.pipeline.run_observationalization_pipeline`.

    Arms 2/3 are excluded before any `ClinicalEvent` is built for that
    patient (see module docstring) -- they never appear in the returned
    `timelines`/`treatment_arms`/`endpoint_results`. Every retained (arm
    0/1) patient's raw `treatment_label` is checked via
    `assert_arm_label_consistent` (raises `ArmLabelInconsistentError` on a
    mismatch); `arm` itself is what actually selects the emitted treatment
    concept_id.

    Returns:
        `timelines`: one `data.timelines.PatientTimeline` per retained
            patient, built from that patient's demographic, baseline-
            covariate, treatment, longitudinal CD4/CD8, and censoring
            events.
        `treatment_arms`: `{patient_id: arm}` (`0` or `1`) for every
            retained patient -- the same shape
            `cohorts.cohort_construction.build_randomized_cohort` expects.
        `endpoint_results`: `{patient_id: EndpointResult}` built directly
            from each retained patient's composite `OutcomeRecord`/
            `CensoringRecord` pair (never via `CompositeEndpoint.compute`
            -- see module docstring). Pass this as
            `build_randomized_cohort`'s `precomputed_endpoint_results`
            argument.

    Raises:
        ArmLabelInconsistentError: if a retained patient's raw
            `treatment_label` does not match the frozen expected label for
            their `arm`.
        ValueError: if a patient's `OutcomeRecord`/`CensoringRecord` pair
            fails the logical-complement invariant check, or if a
            longitudinal record's `week` is not one of the recognized
            landmark weeks.
    """
    treatment_arm_by_id: dict[str, int] = {}
    for record in tables.treatment_arm:
        if record.arm in _EXCLUDED_ARMS:
            continue
        assert_arm_label_consistent(record)
        treatment_arm_by_id[record.patient_id] = record.arm

    retained_ids = set(treatment_arm_by_id)

    demographics_by_id = {
        record.patient_id: record
        for record in tables.demographics
        if record.patient_id in retained_ids
    }
    baseline_covariates_by_id = {
        record.patient_id: record
        for record in tables.baseline_covariates
        if record.patient_id in retained_ids
    }
    censoring_by_id = {
        record.patient_id: record
        for record in tables.censoring
        if record.patient_id in retained_ids
    }
    outcomes_by_id = {
        record.patient_id: record for record in tables.outcomes if record.patient_id in retained_ids
    }
    longitudinal_by_id: dict[str, list[LongitudinalCD4CD8Record]] = {}
    for record in tables.longitudinal_cd4_cd8:
        if record.patient_id not in retained_ids:
            continue
        longitudinal_by_id.setdefault(record.patient_id, []).append(record)

    timelines: list[PatientTimeline] = []
    endpoint_results: dict[str, EndpointResult] = {}

    for patient_id in sorted(retained_ids):
        arm = treatment_arm_by_id[patient_id]
        events: list[ClinicalEvent] = []

        if patient_id in demographics_by_id:
            events.extend(_demographics_events(demographics_by_id[patient_id]))
        if patient_id in baseline_covariates_by_id:
            events.extend(_baseline_covariate_events(baseline_covariates_by_id[patient_id]))

        events.append(_treatment_event(patient_id, arm))

        for longitudinal_record in longitudinal_by_id.get(patient_id, []):
            events.extend(_longitudinal_events(longitudinal_record))

        if patient_id in censoring_by_id:
            events.append(_censoring_event(censoring_by_id[patient_id]))

        if not events:
            # No data at all for this patient in any table besides
            # treatment_arm -- skip rather than building a degenerate
            # single-event timeline. Not expected in the real distribution
            # (see referential-integrity checks performed at ingestion),
            # but defensively handled rather than assumed.
            continue

        timelines.append(build_patient_timeline(events))

        if patient_id in outcomes_by_id and patient_id in censoring_by_id:
            endpoint_results[patient_id] = _build_endpoint_result(
                outcomes_by_id[patient_id], censoring_by_id[patient_id]
            )

    return timelines, dict(treatment_arm_by_id.items()), endpoint_results
