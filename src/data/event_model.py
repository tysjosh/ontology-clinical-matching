"""The `ClinicalEvent` model and reproducible event ordering (Requirement 5.5-5.6).

`design.md` Section 3.3 / "Data Models" specifies `ClinicalEvent` as a frozen
dataclass with an exact field set, and requires "a total ordering ... on
`(patient_id, observed_at, concept_id)` for reproducible sorting". This
module implements exactly that: the dataclass itself, a `MissingnessState`
enum for its `missingness` field, and a total ordering.

No existing missingness-state convention was found elsewhere in the
codebase to reuse (`src/data/loaders/actg175.py` represents missingness only
implicitly, via nullable `float | None` fields on raw records — it does not
define a named missingness-state type). `MissingnessState` below is
introduced here as the first such convention; later modules (e.g.
`src/data/normalization.py`, `src/data/timelines.py`) should reuse it rather
than defining their own.

Ordering / reproducibility:

Python's `sorted()` is a stable sort, so sorting by a fully-specified,
deterministic key never depends on input order, dict/set iteration order,
or process/run identity — repeated sorts of the same input (or of any
shuffling of the same underlying elements) always produce the same output
order. `ClinicalEvent` provides this via `sort_key()` plus `__lt__`/`__eq__`
(wired up with `functools.total_ordering` for the remaining comparisons),
so both `event.sort_key()` and plain `sorted(events)` are supported and
agree with each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import total_ordering


class MissingnessState(str, Enum):
    """The missingness state of a single `ClinicalEvent`'s value.

    Kept intentionally small: only the states a raw clinical event actually
    needs to represent at ingestion time. Downstream missingness handling
    (Requirement 5.12-5.13, missingness indicators / fold-safe imputation)
    consumes this value rather than redefining it.
    """

    #: The value was recorded and is present.
    OBSERVED = "observed"
    #: The value is missing with no further distinction available.
    MISSING = "missing"
    #: The value is missing because it was not applicable/collected by
    #: protocol at this timepoint (e.g. an assessment not scheduled for this
    #: visit), distinct from a value that should have been observed but
    #: was not.
    NOT_APPLICABLE = "not_applicable"


@total_ordering
@dataclass(frozen=True)
class ClinicalEvent:
    """A single normalized clinical event (`design.md` Section 3.3 / Data Models).

    Every field mirrors the exact set specified in `design.md`:
    `patient_id, encounter_id, concept_id, value, unit, modality, observed_at,
    valid_from, valid_to, source, missingness, provenance`.

    Total ordering is defined on `(patient_id, observed_at, concept_id)`
    only (Requirement 5.6) — the remaining fields do not participate in
    comparison, so two events differing only in e.g. `value` or `provenance`
    but sharing the same `(patient_id, observed_at, concept_id)` key compare
    as equal for ordering purposes. Because `sorted()` is stable, such ties
    retain their relative input order deterministically; they are never
    reordered based on unrelated fields.
    """

    patient_id: str
    encounter_id: str | None
    concept_id: str
    value: float | str | None
    unit: str | None
    modality: str
    observed_at: datetime
    valid_from: datetime | None
    valid_to: datetime | None
    source: str
    missingness: MissingnessState
    provenance: str

    def sort_key(self) -> tuple[str, datetime, str]:
        """The `(patient_id, observed_at, concept_id)` total-ordering key.

        Used by `__lt__`/`__eq__` below, and safe to call directly (e.g. as
        `sorted(events, key=ClinicalEvent.sort_key)`) for callers that want
        an explicit key rather than relying on the dataclass's own rich
        comparisons.
        """
        return (self.patient_id, self.observed_at, self.concept_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClinicalEvent):
            return NotImplemented
        return self.sort_key() == other.sort_key()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ClinicalEvent):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    def __hash__(self) -> int:
        return hash(self.sort_key())
