"""The `Intervention` object (Requirement 9.1; `design.md` "Data Models").

`design.md`'s "Data Models" section specifies `Intervention` as a frozen
dataclass with an exact field set. This module implements exactly that
dataclass, following the same convention established by
`src/data/event_model.py::ClinicalEvent` -- a frozen dataclass with a
module docstring tying the fields back to the requirement and decisions
that fixed them, and per-field documentation where the field's meaning or
default is not self-evident from its name alone.

Requirement 9.1 states that "an intervention SHALL be represented with only
the fields available and relevant to the selected trial, which may include
concept, dose, unit, route, frequency, initiation, discontinuation, and
adherence." For ACTG175, every one of those fields is available and
relevant except `mechanism_ids`: `decisions.md` Section 23 ("Out of scope")
places "pathway or mechanism encoders" out of scope unless external
mechanism knowledge is later added, in which case Requirement 12.7-12.8
would require shuffled-label controls. `mechanism_ids` is therefore kept on
the dataclass (so the field exists for that possible future extension) but
defaults to an empty tuple `()`, since no mechanism/pathway encoding is
used or required by the frozen ACTG175 trial spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Intervention:
    """A single patient intervention (treatment exposure), `design.md`
    "Data Models".

    Fields mirror `design.md`'s exact specification:
    `concept_id, dose, unit, route, frequency, start_time, stop_time,
    adherence, mechanism_ids`.

    Optional fields (`dose`, `unit`, `route`, `frequency`, `stop_time`,
    `adherence`) are `None` when not available or not relevant to the
    selected trial (Requirement 9.1) -- for example `stop_time` is `None`
    for an intervention that has not been discontinued as of the relevant
    observation window.
    """

    #: Normalized concept identifier for the intervention (e.g. the drug or
    #: regimen concept), using the same concept-normalization convention as
    #: `src/data/normalization.py`.
    concept_id: str
    #: Dose amount, or `None` if not available/relevant to this trial.
    dose: float | None
    #: Unit for `dose`, or `None` if not available/relevant to this trial.
    unit: str | None
    #: Route of administration, or `None` if not available/relevant.
    route: str | None
    #: Dosing frequency, or `None` if not available/relevant.
    frequency: str | None
    #: Time the intervention was initiated.
    start_time: datetime
    #: Time the intervention was discontinued, or `None` if it was not
    #: discontinued (or discontinuation is not observed) as of the relevant
    #: observation window.
    stop_time: datetime | None
    #: Adherence measure (e.g. proportion of prescribed doses taken), or
    #: `None` if not available/relevant to this trial.
    adherence: float | None
    #: Mechanism/pathway identifiers associated with this intervention.
    #: Defaults to an empty tuple: ACTG175's frozen trial spec does not
    #: require mechanism/pathway encoding (`decisions.md` Section 23,
    #: "Out of scope" -- mechanism/pathway encoders), so no intervention
    #: constructed for this study populates this field.
    mechanism_ids: tuple[str, ...] = ()
