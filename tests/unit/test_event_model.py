"""Unit tests for `src/data/event_model.py`.

**Validates: Requirements 5.5, 5.6**
"""

import random
from datetime import datetime, timedelta

from data.event_model import ClinicalEvent, MissingnessState


def _make_event(patient_id: str, observed_at: datetime, concept_id: str) -> ClinicalEvent:
    return ClinicalEvent(
        patient_id=patient_id,
        encounter_id=None,
        concept_id=concept_id,
        value=1.0,
        unit="cells/mm3",
        modality="lab",
        observed_at=observed_at,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )


def _underlying_events() -> list[ClinicalEvent]:
    """A fixed set of events including deliberate ties on
    `(patient_id, observed_at, concept_id)` and shared `patient_id`/`observed_at`
    pairs with differing `concept_id`.
    """
    t0 = datetime(2020, 1, 1)
    t1 = t0 + timedelta(days=20 * 7)  # week 20
    t2 = t0 + timedelta(days=96 * 7)  # week 96

    return [
        _make_event("A", t1, "cd4"),
        _make_event("A", t1, "cd8"),  # same patient/time, different concept
        _make_event("A", t0, "cd4"),  # earlier observed_at, same patient
        _make_event("B", t0, "cd4"),  # different (later-sorting) patient
        _make_event("A", t2, "cd4"),
        _make_event("A", t1, "cd4"),  # exact duplicate key of the first entry
        _make_event("B", t0, "cd8"),
    ]


def test_sort_orders_by_patient_then_observed_at_then_concept_id():
    """Sorting orders events by `(patient_id, observed_at, concept_id)`:
    patient "A" events sort before patient "B" events, and within a patient,
    earlier `observed_at` sorts before later `observed_at`.
    """
    events = _underlying_events()

    sorted_events = sorted(events)
    keys = [e.sort_key() for e in sorted_events]

    assert keys == sorted(keys)

    # All "A" events precede all "B" events.
    patient_ids = [e.patient_id for e in sorted_events]
    last_a_index = max(i for i, pid in enumerate(patient_ids) if pid == "A")
    first_b_index = min(i for i, pid in enumerate(patient_ids) if pid == "B")
    assert last_a_index < first_b_index

    # Within patient "A", observed_at is non-decreasing.
    a_times = [e.observed_at for e in sorted_events if e.patient_id == "A"]
    assert a_times == sorted(a_times)

    # Within a tied (patient_id, observed_at), concept_id is ascending.
    a_week20 = [
        e.concept_id
        for e in sorted_events
        if e.patient_id == "A" and e.observed_at == datetime(2020, 1, 1) + timedelta(days=20 * 7)
    ]
    assert a_week20 == sorted(a_week20)


def test_sorting_is_stable_and_reproducible_across_repeated_runs():
    """Sorting the same underlying set of events (including ties on the
    ordering key) always produces the same output order, across repeated
    sorts of the same list and across many random shufflings of it.
    """
    base_events = _underlying_events()

    reference_order = [e.sort_key() for e in sorted(base_events)]

    # Sorting the exact same input list repeatedly is identical every time.
    for _ in range(5):
        result = [e.sort_key() for e in sorted(base_events)]
        assert result == reference_order

    # Sorting many random shuffles of the same underlying elements always
    # produces the same key order (ties broken deterministically by the
    # total ordering, not by input/shuffle order).
    rng = random.Random(1234)
    for _ in range(20):
        shuffled = list(base_events)
        rng.shuffle(shuffled)
        result = [e.sort_key() for e in sorted(shuffled)]
        assert result == reference_order


def test_tied_events_preserve_relative_input_order_via_stable_sort():
    """Two events sharing the exact same `(patient_id, observed_at,
    concept_id)` key compare as equal for ordering, and Python's stable
    `sorted()` preserves their relative input order rather than reordering
    them arbitrarily.
    """
    t0 = datetime(2020, 1, 1)
    tied_first = _make_event("A", t0, "cd4")
    tied_second = _make_event("A", t0, "cd4")
    other = _make_event("B", t0, "cd4")

    assert tied_first == tied_second
    assert not (tied_first < tied_second)
    assert not (tied_second < tied_first)

    result = sorted([other, tied_first, tied_second])
    assert result[0] is tied_first
    assert result[1] is tied_second
    assert result[2] is other

    # Reversing the tie's input order reverses their relative output order,
    # confirming stability rather than an incidental identical result.
    result_reversed_input = sorted([other, tied_second, tied_first])
    assert result_reversed_input[0] is tied_second
    assert result_reversed_input[1] is tied_first
