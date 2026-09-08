"""Unit tests for `src/data/timelines.py`.

Constructs a small set of hand-verified `ClinicalEvent` records covering
every timeline category (static covariate, timestamped lab observation,
treatment event, endpoint event, censoring event, and a `MISSING` value),
then asserts `build_patient_timeline()`'s output matches those hand-checked
source records field by field.

**Validates: Requirement 5.9**
"""

from datetime import datetime

import pytest

from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import (
    MultiplePatientsError,
    UnrecognizedModalityError,
    build_patient_timeline,
    build_patient_timelines,
)

PATIENT_ID = "P001"
ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = datetime(2020, 6, 4, 9, 0, 0)
WEEK_96 = datetime(2021, 12, 2, 9, 0, 0)
END_OF_FOLLOWUP = datetime(2022, 1, 1, 0, 0, 0)


def _event(**overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=PATIENT_ID,
        encounter_id=None,
        concept_id="cd4_count",
        value=350.0,
        unit="cells/mm3",
        modality="lab",
        observed_at=WEEK_20,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


# Hand-verified source records, one per timeline category.
BASELINE_AGE = _event(
    concept_id="age",
    value=34.0,
    unit="years",
    modality="baseline",
    observed_at=ASSIGNMENT_TIME,
)
DEMOGRAPHIC_SEX = _event(
    concept_id="sex",
    value="M",
    unit=None,
    modality="demographic",
    observed_at=ASSIGNMENT_TIME,
)
LAB_CD4_WEEK20 = _event(
    concept_id="cd4_count",
    value=310.0,
    unit="cells/mm3",
    modality="lab",
    observed_at=WEEK_20,
)
LAB_CD4_WEEK96_MISSING = _event(
    concept_id="cd4_count",
    value=None,
    unit="cells/mm3",
    modality="lab",
    observed_at=WEEK_96,
    missingness=MissingnessState.MISSING,
)
TREATMENT_ASSIGNMENT = _event(
    concept_id="zidovudine+didanosine",
    value=1.0,
    unit=None,
    modality="treatment",
    observed_at=ASSIGNMENT_TIME,
)
ENDPOINT_EVENT = _event(
    concept_id="aids_defining_event",
    value=1.0,
    unit=None,
    modality="endpoint",
    observed_at=WEEK_96,
)
CENSORING_EVENT = _event(
    concept_id="administrative_censoring",
    value=1.0,
    unit=None,
    modality="censoring",
    observed_at=END_OF_FOLLOWUP,
)

ALL_EVENTS = [
    BASELINE_AGE,
    DEMOGRAPHIC_SEX,
    LAB_CD4_WEEK20,
    LAB_CD4_WEEK96_MISSING,
    TREATMENT_ASSIGNMENT,
    ENDPOINT_EVENT,
    CENSORING_EVENT,
]


def test_timeline_categorizes_each_event_into_the_correct_bucket():
    timeline = build_patient_timeline(ALL_EVENTS)

    assert timeline.patient_id == PATIENT_ID

    # Static covariates: keyed by concept_id, values match the hand-verified
    # source records exactly.
    assert timeline.static_covariates == {"age": 34.0, "sex": "M"}

    # Timestamped observations: both cd4_count lab events, sorted by the
    # ClinicalEvent total ordering (observed_at ascending here).
    assert timeline.timestamped_observations == [LAB_CD4_WEEK20, LAB_CD4_WEEK96_MISSING]

    # Treatment / endpoint / censoring events match their hand-verified
    # source records exactly.
    assert timeline.treatment_events == [TREATMENT_ASSIGNMENT]
    assert timeline.endpoint_events == [ENDPOINT_EVENT]
    assert timeline.censoring_events == [CENSORING_EVENT]


def test_missingness_mask_reflects_each_events_missingness_state():
    timeline = build_patient_timeline(ALL_EVENTS)

    # Static covariates keyed by concept_id alone.
    assert timeline.missingness_mask["age"] == MissingnessState.OBSERVED
    assert timeline.missingness_mask["sex"] == MissingnessState.OBSERVED

    # Time-varying entries keyed by concept_id@observed_at, disambiguating
    # the two cd4_count occurrences (week 20 observed, week 96 missing).
    assert (
        timeline.missingness_mask[f"cd4_count@{WEEK_20.isoformat()}"]
        == MissingnessState.OBSERVED
    )
    assert (
        timeline.missingness_mask[f"cd4_count@{WEEK_96.isoformat()}"]
        == MissingnessState.MISSING
    )

    assert (
        timeline.missingness_mask[f"zidovudine+didanosine@{ASSIGNMENT_TIME.isoformat()}"]
        == MissingnessState.OBSERVED
    )
    assert (
        timeline.missingness_mask[f"aids_defining_event@{WEEK_96.isoformat()}"]
        == MissingnessState.OBSERVED
    )
    assert (
        timeline.missingness_mask[f"administrative_censoring@{END_OF_FOLLOWUP.isoformat()}"]
        == MissingnessState.OBSERVED
    )


def test_missing_value_is_retained_not_dropped():
    """A MISSING event still appears in its bucket (value None), matching the
    "record retained, not dropped" convention (design.md Error Handling table).
    """
    timeline = build_patient_timeline(ALL_EVENTS)

    missing_events = [
        event
        for event in timeline.timestamped_observations
        if event.missingness == MissingnessState.MISSING
    ]
    assert missing_events == [LAB_CD4_WEEK96_MISSING]
    assert missing_events[0].value is None


def test_build_patient_timeline_requires_at_least_one_event():
    with pytest.raises(ValueError):
        build_patient_timeline([])


def test_build_patient_timeline_rejects_mixed_patient_events():
    other_patient_event = _event(patient_id="P002")
    with pytest.raises(MultiplePatientsError):
        build_patient_timeline([BASELINE_AGE, other_patient_event])


def test_build_patient_timeline_rejects_unrecognized_modality():
    bad_event = _event(modality="imaging")
    with pytest.raises(UnrecognizedModalityError):
        build_patient_timeline([bad_event])


def test_build_patient_timelines_groups_events_by_patient_id():
    other_patient_baseline = _event(patient_id="P002", concept_id="age", value=50.0, modality="baseline")
    other_patient_lab = _event(patient_id="P002", concept_id="cd4_count", value=400.0, modality="lab")

    timelines = build_patient_timelines(ALL_EVENTS + [other_patient_baseline, other_patient_lab])

    assert set(timelines.keys()) == {PATIENT_ID, "P002"}
    assert timelines[PATIENT_ID] == build_patient_timeline(ALL_EVENTS)
    assert timelines["P002"].static_covariates == {"age": 50.0}
    assert timelines["P002"].timestamped_observations == [other_patient_lab]


def test_build_patient_timelines_returns_empty_dict_for_empty_input():
    assert build_patient_timelines([]) == {}
