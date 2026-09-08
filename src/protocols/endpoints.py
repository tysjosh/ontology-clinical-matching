"""The ACTG composite time-to-event endpoint (Requirement 3.12, `decisions.md` Section 6).

`CompositeEndpoint` implements time-to-first-of the three components that make
up the primary ACTG175 endpoint:

- a >=50% decline in CD4 count from baseline;
- an AIDS-defining event;
- death.

This is "exactly the one endpoint type required" (`design.md` Section 3.2) —
no other endpoint family is implemented here, and no speculative component is
added beyond the three `decisions.md` Section 6 names.

Input shape
-----------
`src/protocols/schema.py::OutcomeSpec` models the composite endpoint
*abstractly* (its `components` field lists the three component names as
strings) — it does not carry per-patient longitudinal data. The current
`src/protocols/eligibility_dsl.py::PatientTimeline` is a minimal shape built
for eligibility/time-zero checks and does not model a patient's longitudinal
CD4 series or dated event records either, and `src/data/timelines.py` (the
eventual richer, authoritative timeline builder) does not exist yet.

Rather than depend on either the not-yet-built `src/data/timelines.py` or
grow `PatientTimeline` to carry data it does not otherwise need,
`PatientEndpointRecord` below is a small, self-contained input type owned by
this module. When `src/data/timelines.py` lands, its output should be
adapted (or a call-site adapter written) to construct a
`PatientEndpointRecord` — `CompositeEndpoint.compute`'s logic does not need
to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# The three named components, in the exact order `decisions.md` Section 6
# lists them. `EventType` is restricted to these three plus `None`
# (censored) — no speculative fourth component is introduced.
EventType = Literal["cd4_decline", "aids_defining_event", "death"]

# The frozen decline threshold from `decisions.md` Section 6: ">=50% decline
# in CD4 count from baseline". Expressed as a module-level constant (not a
# free parameter) so it cannot silently drift, mirroring how `decisions.md`
# Section 9's risk-score coefficients are frozen as module-level constants
# elsewhere in this codebase.
CD4_DECLINE_THRESHOLD = 0.5


@dataclass(frozen=True)
class PatientEndpointRecord:
    """Minimal, self-contained per-patient input to `CompositeEndpoint.compute`.

    Attributes:
        baseline_cd4: the patient's baseline (time-zero) CD4 count. Must be
            strictly positive — a nonpositive baseline makes "percent
            decline from baseline" undefined.
        last_observed_time: the timestamp of administrative end of
            follow-up or the patient's last recorded observation, whichever
            defines the censoring time (`decisions.md` Section 5). Used as
            the event time when no component occurs during follow-up.
        assignment_time: the patient's treatment-assignment ("time zero")
            timestamp. Carried for context/provenance and potential future
            validation (e.g. rejecting components dated before assignment);
            not otherwise used by the current `compute()` logic.
        cd4_observations: the patient's longitudinal CD4 series as
            `(observed_at, cd4_value)` pairs. Need not be pre-sorted;
            `CompositeEndpoint.compute` sorts by time internally. May be
            empty if no post-baseline CD4 measurements are available.
        aids_defining_event_time: the timestamp of the patient's first
            AIDS-defining event, or `None` if none occurred (or none is
            recorded).
        death_time: the timestamp of the patient's death, or `None` if the
            patient did not die (or death is not recorded) during
            follow-up.
    """

    baseline_cd4: float
    last_observed_time: datetime
    assignment_time: datetime
    cd4_observations: list[tuple[datetime, float]] = field(default_factory=list)
    aids_defining_event_time: datetime | None = None
    death_time: datetime | None = None


@dataclass(frozen=True)
class EndpointResult:
    """The outcome of evaluating the composite endpoint for one patient.

    Attributes:
        event_time: the time of the earliest-occurring component, if any
            component occurred at or before `last_observed_time`; otherwise
            `last_observed_time` (administrative censoring).
        event_indicator: `True` if any component occurred at or before
            `last_observed_time`; `False` if the patient is censored.
        censoring_indicator: the logical complement of `event_indicator`,
            carried as its own field (rather than left implicit) so callers
            do not need to remember to negate `event_indicator`.
        event_type: which component defined `event_time`
            (`"cd4_decline"`, `"aids_defining_event"`, or `"death"`), or
            `None` when the patient is censored.
    """

    event_time: datetime
    event_indicator: bool
    censoring_indicator: bool
    event_type: EventType | None


class CompositeEndpoint:
    """Computes the ACTG composite time-to-event endpoint for one patient."""

    @staticmethod
    def compute(record: PatientEndpointRecord) -> EndpointResult:
        """Compute time-to-first-of CD4 decline, AIDS-defining event, or death.

        Each of the three components is reduced to a single candidate
        `(event_type, time)` pair (or no candidate, if that component never
        occurs):

        - `cd4_decline`: the earliest `cd4_observations` entry, by time,
          whose value represents a decline of `CD4_DECLINE_THRESHOLD` or
          more from `baseline_cd4` (i.e. `(baseline - value) / baseline >=
          0.5`; boundary-inclusive, per "≥50%").
        - `aids_defining_event`: `record.aids_defining_event_time`, if set.
        - `death`: `record.death_time`, if set.

        Only candidates occurring at or before `last_observed_time` are
        eligible to define the event; a component recorded after
        `last_observed_time` is outside the observed follow-up window and is
        ignored (the patient is administratively censored at
        `last_observed_time` if no other component qualifies). Among
        qualifying candidates, the earliest wins; a tie between two
        components is broken by the fixed order (`cd4_decline`,
        `aids_defining_event`, `death`) declared in `decisions.md` Section 6
        so ties are resolved deterministically rather than arbitrarily.

        Args:
            record: the patient's longitudinal CD4 series and event
                timestamps.

        Returns:
            An `EndpointResult` with the earliest qualifying component (event)
            or, if none qualifies, administrative censoring at
            `record.last_observed_time`.
        """
        candidates: list[tuple[datetime, EventType]] = []

        cd4_decline_time = CompositeEndpoint._first_cd4_decline_time(record)
        if cd4_decline_time is not None:
            candidates.append((cd4_decline_time, "cd4_decline"))
        if record.aids_defining_event_time is not None:
            candidates.append((record.aids_defining_event_time, "aids_defining_event"))
        if record.death_time is not None:
            candidates.append((record.death_time, "death"))

        eligible = [
            (time, event_type)
            for time, event_type in candidates
            if time <= record.last_observed_time
        ]

        if not eligible:
            return EndpointResult(
                event_time=record.last_observed_time,
                event_indicator=False,
                censoring_indicator=True,
                event_type=None,
            )

        # Earliest time wins; a tie is broken by fixed component order
        # (cd4_decline, aids_defining_event, death) via the stable sort's
        # original insertion order above.
        earliest_time, earliest_type = min(eligible, key=lambda pair: pair[0])
        return EndpointResult(
            event_time=earliest_time,
            event_indicator=True,
            censoring_indicator=False,
            event_type=earliest_type,
        )

    @staticmethod
    def _first_cd4_decline_time(record: PatientEndpointRecord) -> datetime | None:
        """Earliest observation time showing a >=50% decline from baseline, or `None`."""
        if record.baseline_cd4 <= 0:
            raise ValueError(
                f"baseline_cd4 must be strictly positive to compute percent decline; "
                f"got {record.baseline_cd4!r}."
            )
        for observed_at, cd4_value in sorted(record.cd4_observations, key=lambda pair: pair[0]):
            decline_fraction = (record.baseline_cd4 - cd4_value) / record.baseline_cd4
            if decline_fraction >= CD4_DECLINE_THRESHOLD:
                return observed_at
        return None
