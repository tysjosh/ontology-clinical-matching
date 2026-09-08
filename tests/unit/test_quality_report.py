"""Unit tests for `src/data/quality_report.py`.

Builds a small synthetic cohort of `PatientTimeline`s (varying treatment
arms, some missing baseline fields, a mix of event types, censored
patients, one recorded treatment switch), calls
`build_data_quality_report()`, and asserts every required field
(Requirement 5.14) is present, non-`None`, and reflects plausible/correct
values given the synthetic input.

**Validates: Requirement 5.14**
"""

from datetime import datetime

import pytest

from data.event_model import ClinicalEvent, MissingnessState
from data.quality_report import build_data_quality_report
from data.timelines import build_patient_timeline
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = datetime(2020, 6, 4, 9, 0, 0)
WEEK_96 = datetime(2021, 12, 2, 9, 0, 0)
END_OF_FOLLOWUP = datetime(2022, 1, 1, 0, 0, 0)


def _event(**overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id="P001",
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


def _build_cohort():
    """Six synthetic patients:

    - P001: arm zidovudine, karnofsky observed, censored (no event).
    - P002: arm zidovudine+didanosine, karnofsky MISSING, death event.
    - P003: arm zidovudine, karnofsky observed, aids_defining_event.
    - P004: arm zidovudine+didanosine, karnofsky MISSING, censored.
    - P005: arm zidovudine, karnofsky observed, recorded treatment switch
      (starts zidovudine, switches to zidovudine+didanosine), death event.
    - P006: arm zidovudine+didanosine, karnofsky observed, cd4_decline event
      (via supplied EndpointResult, since raw endpoint_events cannot encode
      a derived cd4_decline component).
    """
    timelines = {}

    # P001: censored, arm zidovudine, karnofsky observed.
    p1_events = [
        _event(patient_id="P001", concept_id="age", value=30.0, modality="baseline"),
        _event(
            patient_id="P001",
            concept_id="karnofsky_score",
            value=90.0,
            modality="baseline",
        ),
        _event(
            patient_id="P001",
            concept_id="zidovudine",
            value=1.0,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id="P001",
            concept_id="administrative_censoring",
            value=1.0,
            modality="censoring",
            observed_at=END_OF_FOLLOWUP,
        ),
    ]
    timelines["P001"] = build_patient_timeline(p1_events)

    # P002: death event, arm zidovudine+didanosine, karnofsky MISSING.
    p2_events = [
        _event(patient_id="P002", concept_id="age", value=40.0, modality="baseline"),
        _event(
            patient_id="P002",
            concept_id="karnofsky_score",
            value=None,
            modality="baseline",
            missingness=MissingnessState.MISSING,
        ),
        _event(
            patient_id="P002",
            concept_id="zidovudine+didanosine",
            value=1.0,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id="P002",
            concept_id="death",
            value=1.0,
            modality="endpoint",
            observed_at=WEEK_96,
        ),
    ]
    timelines["P002"] = build_patient_timeline(p2_events)

    # P003: aids_defining_event, arm zidovudine, karnofsky observed.
    p3_events = [
        _event(patient_id="P003", concept_id="age", value=50.0, modality="baseline"),
        _event(
            patient_id="P003",
            concept_id="karnofsky_score",
            value=80.0,
            modality="baseline",
        ),
        _event(
            patient_id="P003",
            concept_id="zidovudine",
            value=1.0,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id="P003",
            concept_id="aids_defining_event",
            value=1.0,
            modality="endpoint",
            observed_at=WEEK_20,
        ),
    ]
    timelines["P003"] = build_patient_timeline(p3_events)

    # P004: censored, arm zidovudine+didanosine, karnofsky MISSING.
    p4_events = [
        _event(patient_id="P004", concept_id="age", value=35.0, modality="baseline"),
        _event(
            patient_id="P004",
            concept_id="karnofsky_score",
            value=None,
            modality="baseline",
            missingness=MissingnessState.MISSING,
        ),
        _event(
            patient_id="P004",
            concept_id="zidovudine+didanosine",
            value=1.0,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id="P004",
            concept_id="administrative_censoring",
            value=1.0,
            modality="censoring",
            observed_at=END_OF_FOLLOWUP,
        ),
    ]
    timelines["P004"] = build_patient_timeline(p4_events)

    # P005: recorded treatment switch, death event, arm zidovudine (initial).
    p5_events = [
        _event(patient_id="P005", concept_id="age", value=45.0, modality="baseline"),
        _event(
            patient_id="P005",
            concept_id="karnofsky_score",
            value=70.0,
            modality="baseline",
        ),
        _event(
            patient_id="P005",
            concept_id="zidovudine",
            value=1.0,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id="P005",
            concept_id="zidovudine+didanosine",
            value=1.0,
            modality="treatment",
            observed_at=WEEK_20,
        ),
        _event(
            patient_id="P005",
            concept_id="death",
            value=1.0,
            modality="endpoint",
            observed_at=WEEK_96,
        ),
    ]
    timelines["P005"] = build_patient_timeline(p5_events)

    # P006: cd4_decline via supplied EndpointResult, arm zidovudine+didanosine.
    p6_events = [
        _event(patient_id="P006", concept_id="age", value=28.0, modality="baseline"),
        _event(
            patient_id="P006",
            concept_id="karnofsky_score",
            value=95.0,
            modality="baseline",
        ),
        _event(
            patient_id="P006",
            concept_id="zidovudine+didanosine",
            value=1.0,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
    ]
    timelines["P006"] = build_patient_timeline(p6_events)

    endpoint_results = {
        "P006": EndpointResult(
            event_time=WEEK_20,
            event_indicator=True,
            censoring_indicator=False,
            event_type="cd4_decline",
        ),
    }

    return timelines, endpoint_results


def test_all_required_fields_are_populated_and_plausible():
    timelines, endpoint_results = _build_cohort()

    report = build_data_quality_report(
        timelines,
        endpoint_results=endpoint_results,
        subgroup_fields=("treatment_arm",),
    )

    # cohort_size matches the number of patients.
    assert report.cohort_size == 6

    # missingness: karnofsky_score missing for P002/P004 -> 2/6.
    assert report.missingness is not None
    assert report.missingness["karnofsky_score"] == 2 / 6
    assert report.missingness["age"] == 0.0

    # treatment_prevalence: P001/P003/P005 initially zidovudine (3), but
    # P005 switches -> arm is still keyed by *earliest* treatment event, so
    # zidovudine=3 (P001,P003,P005), zidovudine+didanosine=3 (P002,P004,P006).
    assert report.treatment_prevalence is not None
    assert sum(report.treatment_prevalence.values()) == pytest.approx(1.0)
    assert report.treatment_prevalence["zidovudine"] == pytest.approx(3 / 6)
    assert report.treatment_prevalence["zidovudine+didanosine"] == pytest.approx(3 / 6)

    # endpoint_prevalence: P002 (death), P003 (aids), P005 (death), P006 (cd4_decline)
    # have events; P001, P004 are censored -> 4/6.
    assert report.endpoint_prevalence == pytest.approx(4 / 6)

    # follow_up: summary stats present with the expected keys, all finite.
    assert set(report.follow_up.keys()) == {"mean", "median", "min", "max"}
    for value in report.follow_up.values():
        assert value is not None
        assert value >= 0.0
    assert report.follow_up["min"] <= report.follow_up["mean"] <= report.follow_up["max"]

    # event_frequency: one death recorded via raw endpoint_events (P002),
    # one death via raw endpoint_events (P005) -> death=2; one aids event
    # (P003) -> aids_defining_event=1; one cd4_decline via supplied
    # EndpointResult (P006) -> cd4_decline=1.
    assert report.event_frequency == {
        "death": 2,
        "aids_defining_event": 1,
        "cd4_decline": 1,
    }

    # subgroup_counts: treatment_arm breakdown sums to cohort_size.
    assert "treatment_arm" in report.subgroup_counts
    assert sum(report.subgroup_counts["treatment_arm"].values()) == 6
    assert report.subgroup_counts["treatment_arm"]["zidovudine"] == 3
    assert report.subgroup_counts["treatment_arm"]["zidovudine+didanosine"] == 3

    # censoring: P001 and P004 are censored -> count=2, rate=2/6.
    assert report.censoring == {"censored_count": 2, "censoring_rate": pytest.approx(2 / 6)}

    # treatment_switching: only P005 has more than one distinct treatment
    # concept_id recorded -> count=1, rate=1/6.
    assert report.treatment_switching["count"] == 1
    assert report.treatment_switching["rate"] == pytest.approx(1 / 6)


def test_build_data_quality_report_rejects_empty_cohort():
    with pytest.raises(ValueError):
        build_data_quality_report([])


def test_treatment_switching_is_zero_when_no_switch_is_recorded():
    """`treatment_switching` is a valid `0`/`0.0` when no switch is tracked
    in the synthetic input (`design.md` Section 3.3)."""
    timelines, endpoint_results = _build_cohort()
    del timelines["P005"]  # remove the only patient with a recorded switch
    del endpoint_results["P006"]  # keep endpoint_results consistent, though unused here

    report = build_data_quality_report(timelines)

    assert report.treatment_switching == {"count": 0, "rate": 0.0}
