"""Unit tests for `src/data/loaders/actg175_adapter.py`.

Uses the same clearly-documented synthetic fixture as
`tests/unit/test_actg175_loader.py`
(`tests/fixtures/data/actg175_synthetic/`) -- never real patient data.
That fixture has six patients spanning all four raw arms (SYN-001/003 arm 1,
SYN-002/004 arm 0, SYN-005 arm 2, SYN-006 arm 3), so it exercises the
arm-2/3 exclusion path directly.

**Validates:** the real-data adapter introduced for real-data pipeline
wiring (task 18 follow-up); no single formal requirement ID names this
adapter, since it did not exist when `requirements.md`/`tasks.md` were
written -- it implements Requirement 1.2/1.3's "the pipeline SHALL run
against real ACTG175 data" for the raw-to-timeline conversion step
specifically.
"""

from pathlib import Path

import pytest

from data.event_model import MissingnessState
from data.loaders.actg175 import TreatmentArmRecord, compute_actg175_checksum, load_actg175
from data.loaders.actg175_adapter import (
    TREATMENT_CONCEPT_ID_BY_ARM,
    ArmLabelInconsistentError,
    assert_arm_label_consistent,
    raw_tables_to_pipeline_inputs,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "data" / "actg175_synthetic"


@pytest.fixture
def tables():
    checksum = compute_actg175_checksum(FIXTURE_DIR)
    return load_actg175(FIXTURE_DIR, expected_checksum=checksum)


def test_arms_2_and_3_are_excluded_before_any_event_is_built(tables):
    timelines, treatment_arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)

    patient_ids = {timeline.patient_id for timeline in timelines}

    assert patient_ids == {"SYN-001", "SYN-002", "SYN-003", "SYN-004"}
    assert set(treatment_arms) == patient_ids
    assert set(endpoint_results) == patient_ids
    assert "SYN-005" not in patient_ids  # arm 2
    assert "SYN-006" not in patient_ids  # arm 3


def test_arm_field_is_authoritative_for_treatment_concept_id(tables):
    _, treatment_arms, _ = raw_tables_to_pipeline_inputs(tables)

    assert treatment_arms["SYN-001"] == 1
    assert treatment_arms["SYN-002"] == 0
    assert treatment_arms["SYN-003"] == 1
    assert treatment_arms["SYN-004"] == 0


def test_treatment_event_concept_id_matches_frozen_arm_mapping(tables):
    timelines, treatment_arms, _ = raw_tables_to_pipeline_inputs(tables)

    by_id = {timeline.patient_id: timeline for timeline in timelines}
    for patient_id, arm in treatment_arms.items():
        treatment_events = by_id[patient_id].treatment_events
        assert len(treatment_events) == 1
        assert treatment_events[0].concept_id == TREATMENT_CONCEPT_ID_BY_ARM[arm]


def test_endpoint_results_use_composite_outcome_directly_with_no_event_type(tables):
    _, _, endpoint_results = raw_tables_to_pipeline_inputs(tables)

    # SYN-002 has event_indicator=True (per outcomes.csv) even though the
    # raw event_type column says "cd4_decline" -- the adapter must not read
    # that per-component column at all, since real ACTG175 data never
    # populates it meaningfully (see module docstring).
    assert endpoint_results["SYN-002"].event_indicator is True
    assert endpoint_results["SYN-002"].event_type is None

    assert endpoint_results["SYN-001"].event_indicator is False
    assert endpoint_results["SYN-001"].censoring_indicator is True
    assert endpoint_results["SYN-001"].event_type is None

    # Every retained patient's event_type is None -- never fabricated.
    assert all(result.event_type is None for result in endpoint_results.values())


def test_missing_longitudinal_value_is_recorded_as_missing_not_dropped(tables):
    timelines, _, _ = raw_tables_to_pipeline_inputs(tables)

    syn_002 = next(t for t in timelines if t.patient_id == "SYN-002")
    week_96_cd4_events = [
        event
        for event in syn_002.timestamped_observations
        if event.concept_id == "cd4_count" and event.missingness == MissingnessState.MISSING
    ]
    assert len(week_96_cd4_events) == 1
    assert week_96_cd4_events[0].value is None


def test_baseline_covariates_are_present_and_hiv_positive_defaults_true(tables):
    timelines, _, _ = raw_tables_to_pipeline_inputs(tables)

    syn_001 = next(t for t in timelines if t.patient_id == "SYN-001")
    assert syn_001.static_covariates["hiv_positive"] is True
    assert syn_001.static_covariates["baseline_cd4"] == 350.0
    assert syn_001.static_covariates["age"] == 34.0


def test_assert_arm_label_consistent_passes_for_matching_labels():
    record = TreatmentArmRecord(
        patient_id="P1", arm=1, treatment_label="zidovudine+didanosine"
    )
    assert_arm_label_consistent(record)  # does not raise


def test_assert_arm_label_consistent_raises_on_mismatched_label():
    record = TreatmentArmRecord(patient_id="P1", arm=1, treatment_label="wrong_label")

    with pytest.raises(ArmLabelInconsistentError, match="wrong_label"):
        assert_arm_label_consistent(record)


def test_assert_arm_label_consistent_is_a_noop_for_excluded_arms():
    # Arms 2/3 have no frozen expected label -- never raises, since arm
    # alone (checked elsewhere) governs exclusion for those rows.
    record = TreatmentArmRecord(patient_id="P1", arm=2, treatment_label="zalcitabine")
    assert_arm_label_consistent(record)  # does not raise


def test_pipeline_inputs_feed_build_randomized_cohort_without_error(tables):
    """End-to-end smoke check: the adapter's output is accepted by
    `cohorts.cohort_construction.build_randomized_cohort` with
    `precomputed_endpoint_results` and produces a non-empty cohort."""
    import yaml

    from cohorts.cohort_construction import build_randomized_cohort
    from protocols.schema import TargetTrialSpec

    timelines, treatment_arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)

    trial_path = Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"
    with trial_path.open("r", encoding="utf-8") as handle:
        raw_spec = yaml.safe_load(handle)
    spec = TargetTrialSpec.model_validate(raw_spec)

    cohort = build_randomized_cohort(
        timelines, treatment_arms, spec, precomputed_endpoint_results=endpoint_results
    )

    assert len(cohort) > 0
    for patient in cohort.patients:
        assert patient.endpoint_result.event_type is None
