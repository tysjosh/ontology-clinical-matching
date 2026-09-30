"""Patient timeline construction (Requirement 5.9, `design.md` Section 3.3).

`design.md` Section 3.3 describes this module as producing, from a list of
`ClinicalEvent`s, "static covariates, timestamped observations, treatment
events, endpoint events, censoring events, and a missingness mask." This is
the richer, canonical patient-timeline type for the data pipeline —
distinct from (and not to be unified with) the small, self-contained
`src.protocols.eligibility_dsl.PatientTimeline` used only by the eligibility
DSL. The two types share a name but live in different modules and serve
different consumers; later tasks (leakage audit, protocol-consistency
reconciliation) may bridge them, but this module's job is only to build the
canonical timeline from `ClinicalEvent`s.

Categorization rule
--------------------

Each `ClinicalEvent` is routed into exactly one timeline bucket based on its
`modality` field:

| `modality`               | Bucket                     |
|---------------------------|-----------------------------|
| `"demographic"`, `"baseline"` | `static_covariates`     |
| `"lab"`                   | `timestamped_observations` |
| `"treatment"`              | `treatment_events`         |
| `"endpoint"`, `"outcome"`  | `endpoint_events`           |
| `"censoring"`              | `censoring_events`         |

This mirrors `decisions.md` Section 5's distinction between baseline
(non-time-varying) covariates and longitudinal (time-varying) CD4/CD8
observations: a baseline measurement (e.g. `baseline_cd4`) is expected to
carry modality `"baseline"` (or `"demographic"` for non-lab baseline
fields), while the *same underlying concept* measured longitudinally at
week 20/week 96 is expected to carry modality `"lab"`. The distinction lives
in the event's `modality`, not its `concept_id` — `normalize_concept()`
(`src/data/normalization.py`) may map `baseline_cd4` and `cd4` to the same
normalized `concept_id` (`"cd4_count"`), but only the latter is a
timestamped observation for timeline purposes.

A `modality` value outside this table raises `UnrecognizedModalityError`
rather than silently dropping the event (matching the "record retained, not
dropped" convention already used by `UnresolvedConceptFlag` in
`src/data/normalization.py`).

Missingness mask
-----------------

`missingness_mask` records the `MissingnessState` of every event that
contributed to the timeline, keyed so that timepoints are disambiguated:

- static covariates are keyed by `concept_id` alone (a baseline value is
  not time-varying, so one key per concept is sufficient and matches how
  `static_covariates` itself is keyed);
- every other bucket (timestamped observations, treatment, endpoint,
  censoring events) is keyed by `f"{concept_id}@{observed_at.isoformat()}"`,
  since the same concept can recur at multiple timepoints (e.g. `cd4_count`
  at week 20 and week 96) and each occurrence's missingness must be
  distinguishable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from data.event_model import ClinicalEvent, MissingnessState

#: Modalities routed to `static_covariates` (baseline, non-time-varying values).
_STATIC_MODALITIES = frozenset({"demographic", "baseline"})
#: Modalities routed to `timestamped_observations` (longitudinal, time-varying values).
_OBSERVATION_MODALITIES = frozenset({"lab"})
#: Modalities routed to `treatment_events`.
_TREATMENT_MODALITIES = frozenset({"treatment"})
#: Modalities routed to `endpoint_events`.
_ENDPOINT_MODALITIES = frozenset({"endpoint", "outcome"})
#: Modalities routed to `censoring_events`.
_CENSORING_MODALITIES = frozenset({"censoring"})

_STATIC = "static"
_OBSERVATION = "observation"
_TREATMENT = "treatment"
_ENDPOINT = "endpoint"
_CENSORING = "censoring"


class UnrecognizedModalityError(Exception):
    """Raised when an event's `modality` does not match any known timeline bucket.

    Raised (not silently dropped), matching the convention already used by
    `UnresolvedConceptFlag` in `src/data/normalization.py`: an event whose
    modality has no defined categorization must not disappear from the
    timeline unnoticed.
    """

    def __init__(self, modality: str):
        self.modality = modality
        known = sorted(
            _STATIC_MODALITIES
            | _OBSERVATION_MODALITIES
            | _TREATMENT_MODALITIES
            | _ENDPOINT_MODALITIES
            | _CENSORING_MODALITIES
        )
        super().__init__(
            f"Unrecognized event modality '{modality}': no timeline bucket is "
            f"defined for it. Expected one of: {known}."
        )


class MultiplePatientsError(Exception):
    """Raised when `build_patient_timeline` receives events for more than one patient.

    `build_patient_timeline` builds a single patient's timeline; a mixed
    list spanning multiple `patient_id`s must use `build_patient_timelines`
    instead, rather than being silently merged into one timeline.
    """

    def __init__(self, patient_ids: set[str]):
        self.patient_ids = patient_ids
        super().__init__(
            "build_patient_timeline received events for multiple patients "
            f"{sorted(patient_ids)!r}; use build_patient_timelines(events) to "
            "group a mixed-patient event list instead."
        )


@dataclass(frozen=True)
class PatientTimeline:
    """The canonical per-patient timeline (`design.md` Section 3.3).

    Attributes:
        patient_id: the single patient this timeline belongs to.
        static_covariates: baseline, non-time-varying values keyed by
            `concept_id` (e.g. `{"age": 34, "cd4_count": 350}`).
        timestamped_observations: time-varying observations (e.g.
            longitudinal CD4/CD8), sorted per `ClinicalEvent`'s total
            ordering.
        treatment_events: treatment-modality events, sorted.
        endpoint_events: endpoint/outcome-modality events, sorted.
        censoring_events: censoring-modality events, sorted.
        missingness_mask: `MissingnessState` per contributing field/timepoint
            (see module docstring for the exact keying rule).
    """

    patient_id: str
    static_covariates: dict[str, Any]
    timestamped_observations: list[ClinicalEvent]
    treatment_events: list[ClinicalEvent]
    endpoint_events: list[ClinicalEvent]
    censoring_events: list[ClinicalEvent]
    missingness_mask: dict[str, MissingnessState]


def _categorize_modality(modality: str) -> str:
    if modality in _STATIC_MODALITIES:
        return _STATIC
    if modality in _OBSERVATION_MODALITIES:
        return _OBSERVATION
    if modality in _TREATMENT_MODALITIES:
        return _TREATMENT
    if modality in _ENDPOINT_MODALITIES:
        return _ENDPOINT
    if modality in _CENSORING_MODALITIES:
        return _CENSORING
    raise UnrecognizedModalityError(modality)


def _missingness_key(event: ClinicalEvent, category: str) -> str:
    if category == _STATIC:
        return event.concept_id
    return f"{event.concept_id}@{event.observed_at.isoformat()}"


def build_patient_timeline(events: Sequence[ClinicalEvent]) -> PatientTimeline:
    """Build a single patient's `PatientTimeline` from their `ClinicalEvent`s.

    Every event is routed into exactly one of `static_covariates`,
    `timestamped_observations`, `treatment_events`, `endpoint_events`, or
    `censoring_events` based on `event.modality` (see module docstring for
    the categorization table), and its `missingness` is recorded in
    `missingness_mask`. Non-static buckets are ordered using
    `ClinicalEvent`'s total ordering (`sorted(events)`), so the resulting
    lists are reproducible regardless of input order.

    Args:
        events: the events belonging to exactly one patient. Must be
            non-empty.

    Returns:
        The constructed `PatientTimeline`.

    Raises:
        ValueError: if `events` is empty.
        MultiplePatientsError: if `events` contains more than one distinct
            `patient_id`.
        UnrecognizedModalityError: if any event's `modality` does not match
            a known timeline bucket.
    """
    if not events:
        raise ValueError("build_patient_timeline requires at least one event.")

    patient_ids = {event.patient_id for event in events}
    if len(patient_ids) > 1:
        raise MultiplePatientsError(patient_ids)
    patient_id = next(iter(patient_ids))

    static_covariates: dict[str, Any] = {}
    timestamped_observations: list[ClinicalEvent] = []
    treatment_events: list[ClinicalEvent] = []
    endpoint_events: list[ClinicalEvent] = []
    censoring_events: list[ClinicalEvent] = []
    missingness_mask: dict[str, MissingnessState] = {}

    for event in sorted(events):
        category = _categorize_modality(event.modality)
        missingness_mask[_missingness_key(event, category)] = event.missingness

        if category == _STATIC:
            static_covariates[event.concept_id] = event.value
        elif category == _OBSERVATION:
            timestamped_observations.append(event)
        elif category == _TREATMENT:
            treatment_events.append(event)
        elif category == _ENDPOINT:
            endpoint_events.append(event)
        else:  # category == _CENSORING
            censoring_events.append(event)

    return PatientTimeline(
        patient_id=patient_id,
        static_covariates=static_covariates,
        timestamped_observations=timestamped_observations,
        treatment_events=treatment_events,
        endpoint_events=endpoint_events,
        censoring_events=censoring_events,
        missingness_mask=missingness_mask,
    )


def build_patient_timelines(events: Sequence[ClinicalEvent]) -> dict[str, PatientTimeline]:
    """Group a mixed-patient event list and build one `PatientTimeline` per patient.

    A convenience wrapper around `build_patient_timeline` for callers (e.g.
    the Stage-2 "build ClinicalEvents" pipeline stage in `design.md` Section
    2.1) that hold events for many patients at once.

    Args:
        events: events for zero or more patients, in any order.

    Returns:
        A mapping from `patient_id` to that patient's `PatientTimeline`.
        Empty if `events` is empty.
    """
    grouped: dict[str, list[ClinicalEvent]] = {}
    for event in events:
        grouped.setdefault(event.patient_id, []).append(event)

    return {
        patient_id: build_patient_timeline(patient_events)
        for patient_id, patient_events in grouped.items()
    }
