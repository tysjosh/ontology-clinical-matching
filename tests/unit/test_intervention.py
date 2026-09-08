"""Unit tests for `src/models/intervention.py`.

**Validates: Requirements 9.1**
"""

import dataclasses
from datetime import datetime

import pytest

from models.intervention import Intervention


def _make_intervention(**overrides) -> Intervention:
    defaults = dict(
        concept_id="zdv_ddi",
        dose=200.0,
        unit="mg",
        route="oral",
        frequency="bid",
        start_time=datetime(2020, 1, 1),
        stop_time=datetime(2020, 6, 1),
        adherence=0.95,
        mechanism_ids=("pathway_1",),
    )
    defaults.update(overrides)
    return Intervention(**defaults)


def test_construct_with_all_fields():
    """A valid `Intervention` can be constructed with all fields populated."""
    intervention = _make_intervention()

    assert intervention.concept_id == "zdv_ddi"
    assert intervention.dose == 200.0
    assert intervention.unit == "mg"
    assert intervention.route == "oral"
    assert intervention.frequency == "bid"
    assert intervention.start_time == datetime(2020, 1, 1)
    assert intervention.stop_time == datetime(2020, 6, 1)
    assert intervention.adherence == 0.95
    assert intervention.mechanism_ids == ("pathway_1",)


def test_is_immutable():
    """`Intervention` is frozen: attempting to set an attribute raises."""
    intervention = _make_intervention()

    with pytest.raises(dataclasses.FrozenInstanceError):
        intervention.dose = 100.0


def test_mechanism_ids_defaults_to_empty_tuple():
    """`mechanism_ids` defaults to an empty tuple when omitted, since
    ACTG175 does not require mechanism/pathway encoding."""
    intervention = Intervention(
        concept_id="zdv",
        dose=None,
        unit=None,
        route=None,
        frequency=None,
        start_time=datetime(2020, 1, 1),
        stop_time=None,
        adherence=None,
    )

    assert intervention.mechanism_ids == ()


def test_optional_fields_accept_none():
    """`dose`, `unit`, `route`, `frequency`, `stop_time`, and `adherence`
    can each be `None` when not available or not relevant to the trial."""
    intervention = Intervention(
        concept_id="zdv",
        dose=None,
        unit=None,
        route=None,
        frequency=None,
        start_time=datetime(2020, 1, 1),
        stop_time=None,
        adherence=None,
    )

    assert intervention.dose is None
    assert intervention.unit is None
    assert intervention.route is None
    assert intervention.frequency is None
    assert intervention.stop_time is None
    assert intervention.adherence is None
