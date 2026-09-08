"""Unit tests for `src/protocols/endpoints.py::CompositeEndpoint`.

Hand-checked patient examples covering each of the three components winning,
censoring when none occur, the exact-50%-decline boundary, a sub-threshold
decline, and a tie between two components.

**Validates: Requirements 3.12**
"""

from datetime import datetime

from protocols.endpoints import CompositeEndpoint, PatientEndpointRecord

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
LAST_OBSERVED = datetime(2021, 1, 15, 9, 0, 0)


def test_death_before_cd4_decline_or_aids_event_wins():
    """Patient dies before any CD4 decline or AIDS event -> event_type = death."""
    death_time = datetime(2020, 6, 1)
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[(datetime(2020, 5, 1), 380)],  # no decline
        aids_defining_event_time=None,
        death_time=death_time,
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_time == death_time
    assert result.event_type == "death"
    assert result.event_indicator is True
    assert result.censoring_indicator is False


def test_cd4_decline_first_wins():
    """CD4 drops >=50% from baseline first, before any AIDS event or death."""
    decline_time = datetime(2020, 4, 1)
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[
            (datetime(2020, 2, 1), 350),  # 12.5% decline, not an event
            (decline_time, 200),  # exactly 50% decline
            (datetime(2020, 6, 1), 100),  # later, larger decline (should not win)
        ],
        aids_defining_event_time=datetime(2020, 8, 1),
        death_time=datetime(2020, 9, 1),
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_time == decline_time
    assert result.event_type == "cd4_decline"
    assert result.event_indicator is True
    assert result.censoring_indicator is False


def test_aids_defining_event_first_wins():
    """AIDS-defining event occurs before CD4 decline or death."""
    aids_time = datetime(2020, 3, 1)
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[(datetime(2020, 5, 1), 210)],  # decline after AIDS event
        aids_defining_event_time=aids_time,
        death_time=datetime(2020, 12, 1),
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_time == aids_time
    assert result.event_type == "aids_defining_event"
    assert result.event_indicator is True
    assert result.censoring_indicator is False


def test_no_component_occurs_before_censoring():
    """None of the three components occur before last_observed_time -> censored."""
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[(datetime(2020, 6, 1), 380)],  # only 5% decline
        aids_defining_event_time=None,
        death_time=None,
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_time == LAST_OBSERVED
    assert result.event_type is None
    assert result.event_indicator is False
    assert result.censoring_indicator is True


def test_exactly_50_percent_decline_counts_as_event():
    """A CD4 decline of exactly 50% (boundary) counts as an event per '>=50%'."""
    decline_time = datetime(2020, 4, 1)
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[(decline_time, 200)],  # (400-200)/400 == 0.5
        aids_defining_event_time=None,
        death_time=None,
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_type == "cd4_decline"
    assert result.event_time == decline_time
    assert result.event_indicator is True


def test_49_percent_decline_does_not_count():
    """A CD4 decline of 49% does not count as an event."""
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[(datetime(2020, 4, 1), 204)],  # (400-204)/400 == 0.49
        aids_defining_event_time=None,
        death_time=None,
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_type is None
    assert result.event_indicator is False
    assert result.censoring_indicator is True
    assert result.event_time == LAST_OBSERVED


def test_tie_between_two_components_earliest_wins():
    """A tie/near-tie between two components: the earliest one wins."""
    earlier = datetime(2020, 4, 1)
    later = datetime(2020, 4, 2)
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[(earlier, 150)],  # 62.5% decline, at `earlier`
        aids_defining_event_time=later,
        death_time=None,
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_time == earlier
    assert result.event_type == "cd4_decline"
    assert result.event_indicator is True


def test_exact_simultaneous_tie_breaks_by_fixed_component_order():
    """When two components occur at the exact same instant, the fixed order
    (cd4_decline, aids_defining_event, death) breaks the tie deterministically."""
    same_time = datetime(2020, 4, 1)
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[(same_time, 150)],  # decline at same_time
        aids_defining_event_time=None,
        death_time=same_time,  # death at the exact same instant
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_time == same_time
    assert result.event_type == "cd4_decline"


def test_component_after_last_observed_time_is_ignored():
    """A component recorded after last_observed_time is outside follow-up and
    does not define the event; the patient is censored instead."""
    record = PatientEndpointRecord(
        baseline_cd4=400,
        assignment_time=ASSIGNMENT_TIME,
        last_observed_time=LAST_OBSERVED,
        cd4_observations=[],
        aids_defining_event_time=None,
        death_time=datetime(2022, 1, 1),  # after LAST_OBSERVED
    )
    result = CompositeEndpoint.compute(record)
    assert result.event_indicator is False
    assert result.censoring_indicator is True
    assert result.event_time == LAST_OBSERVED
    assert result.event_type is None
