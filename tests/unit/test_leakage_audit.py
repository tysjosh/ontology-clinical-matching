"""Unit tests for `src/data/leakage_audit.py`.

Covers the four `LeakageAuditor.audit` checks (post-treatment baseline
variables, outcome-derived features, indirect treatment/outcome encodings,
duplicate patients across splits) plus the shared
`assert_no_self_adjustment` CD8 double-use safeguard, using synthetic
leakage fixtures:

1. week-20 CD4 value present in a patient's static_covariates (post-treatment
   baseline variable).
2. a patient_id present in both "dev" and "test" split membership lists
   (duplicate across splits).
3. `assert_no_self_adjustment("baseline_cd8", ["baseline_cd8", "age", ...])`
   raising `SelfAdjustmentError` (CD8 used as both outcome and covariate in
   the same regression).

A clean-cohort test confirms no violations are raised for a well-formed
synthetic cohort.

**Validates: Requirements 5.10, 5.11**
"""

from datetime import datetime

import pytest

from data.event_model import ClinicalEvent, MissingnessState
from data.leakage_audit import (
    LeakageAuditor,
    LeakageViolation,
    SelfAdjustmentError,
    assert_no_self_adjustment,
)
from data.timelines import build_patient_timeline

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = datetime(2020, 6, 4, 9, 0, 0)


def _event(**overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id="P001",
        encounter_id=None,
        concept_id="age",
        value=34.0,
        unit="years",
        modality="baseline",
        observed_at=ASSIGNMENT_TIME,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


ADJUSTMENT_SET = ["age", "baseline_cd4", "baseline_cd8", "karnofsky"]


def _clean_timeline(patient_id: str = "P001"):
    """A well-formed timeline: baseline covariates all precede treatment,
    no outcome-concept overlap, no denylisted concept_ids."""
    events = [
        _event(patient_id=patient_id, concept_id="age", value=34.0, modality="baseline"),
        _event(
            patient_id=patient_id,
            concept_id="baseline_cd4",
            value=350.0,
            modality="baseline",
        ),
        _event(
            patient_id=patient_id,
            concept_id="baseline_cd8",
            value=800.0,
            modality="baseline",
        ),
        _event(
            patient_id=patient_id,
            concept_id="zidovudine+didanosine",
            value=1.0,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id=patient_id,
            concept_id="cd4_count",
            value=310.0,
            modality="lab",
            observed_at=WEEK_20,
        ),
        _event(
            patient_id=patient_id,
            concept_id="aids_defining_event",
            value=0.0,
            modality="endpoint",
            observed_at=WEEK_20,
        ),
    ]
    return build_patient_timeline(events)


class TestPostTreatmentBaselineVariable:
    def test_week_20_cd4_leaked_into_baseline_triggers_violation(self):
        """Fixture 1: week-20 CD4 value present in static_covariates, with an
        originating event timestamp after treatment assignment."""
        events = [
            _event(patient_id="P001", concept_id="age", value=34.0, modality="baseline"),
            _event(
                patient_id="P001",
                concept_id="zidovudine+didanosine",
                value=1.0,
                modality="treatment",
                observed_at=ASSIGNMENT_TIME,
            ),
            # Leaked: a week-20 CD4 value materialized as a *baseline*
            # (static) covariate rather than a timestamped observation.
            _event(
                patient_id="P001",
                concept_id="week_20_cd4",
                value=200.0,
                modality="baseline",
                observed_at=WEEK_20,
            ),
        ]
        timeline = build_patient_timeline(events)
        assert "week_20_cd4" in timeline.static_covariates

        static_covariate_events = {
            "P001": {
                "age": ASSIGNMENT_TIME,
                "week_20_cd4": WEEK_20,
            }
        }

        violations = LeakageAuditor().audit(
            {"P001": timeline},
            adjustment_set=["age", "week_20_cd4"],
            static_covariate_events=static_covariate_events,
        )

        matching = [v for v in violations if v.rule == "post_treatment_baseline_variable"]
        assert len(matching) == 1
        assert matching[0].patient_id == "P001"
        assert matching[0].evidence["concept_id"] == "week_20_cd4"

    def test_baseline_variable_at_or_before_treatment_is_not_flagged(self):
        timeline = _clean_timeline()
        static_covariate_events = {
            "P001": {
                "age": ASSIGNMENT_TIME,
                "baseline_cd4": ASSIGNMENT_TIME,
                "baseline_cd8": ASSIGNMENT_TIME,
            }
        }
        violations = LeakageAuditor().audit(
            {"P001": timeline},
            adjustment_set=ADJUSTMENT_SET,
            static_covariate_events=static_covariate_events,
        )
        assert [v for v in violations if v.rule == "post_treatment_baseline_variable"] == []

    def test_no_static_covariate_events_supplied_skips_check(self):
        timeline = _clean_timeline()
        violations = LeakageAuditor().audit(
            {"P001": timeline}, adjustment_set=ADJUSTMENT_SET, static_covariate_events=None
        )
        assert [v for v in violations if v.rule == "post_treatment_baseline_variable"] == []


class TestOutcomeDerivedFeature:
    def test_static_covariate_matching_endpoint_concept_id_triggers_violation(self):
        events = [
            _event(patient_id="P001", concept_id="age", value=34.0, modality="baseline"),
            _event(
                patient_id="P001",
                concept_id="aids_defining_event",
                value=0.0,
                modality="baseline",  # incorrectly also materialized as baseline
            ),
            _event(
                patient_id="P001",
                concept_id="aids_defining_event",
                value=1.0,
                modality="endpoint",
                observed_at=WEEK_20,
            ),
        ]
        timeline = build_patient_timeline(events)

        violations = LeakageAuditor().audit(
            {"P001": timeline}, adjustment_set=["age", "aids_defining_event"]
        )

        matching = [v for v in violations if v.rule == "outcome_derived_feature"]
        assert len(matching) == 1
        assert matching[0].evidence["concept_id"] == "aids_defining_event"

    def test_covariate_not_in_adjustment_set_is_not_flagged(self):
        events = [
            _event(patient_id="P001", concept_id="age", value=34.0, modality="baseline"),
            _event(
                patient_id="P001",
                concept_id="aids_defining_event",
                value=0.0,
                modality="baseline",
            ),
            _event(
                patient_id="P001",
                concept_id="aids_defining_event",
                value=1.0,
                modality="endpoint",
                observed_at=WEEK_20,
            ),
        ]
        timeline = build_patient_timeline(events)
        # adjustment_set deliberately excludes "aids_defining_event".
        violations = LeakageAuditor().audit({"P001": timeline}, adjustment_set=["age"])
        assert [v for v in violations if v.rule == "outcome_derived_feature"] == []


class TestIndirectEncoding:
    def test_denylisted_concept_id_triggers_violation(self):
        events = [
            _event(patient_id="P001", concept_id="age", value=34.0, modality="baseline"),
            _event(
                patient_id="P001",
                concept_id="treatment_switch_indicator",
                value=1.0,
                modality="baseline",
            ),
        ]
        timeline = build_patient_timeline(events)

        violations = LeakageAuditor().audit(
            {"P001": timeline}, adjustment_set=["age", "treatment_switch_indicator"]
        )

        matching = [
            v for v in violations if v.rule == "indirect_treatment_or_outcome_encoding"
        ]
        assert len(matching) == 1
        assert matching[0].evidence["concept_id"] == "treatment_switch_indicator"

    def test_clean_concept_id_is_not_flagged(self):
        timeline = _clean_timeline()
        violations = LeakageAuditor().audit({"P001": timeline}, adjustment_set=ADJUSTMENT_SET)
        assert [
            v for v in violations if v.rule == "indirect_treatment_or_outcome_encoding"
        ] == []


class TestDuplicatePatientAcrossSplits:
    def test_patient_in_dev_and_test_triggers_violation(self):
        """Fixture 2: a patient_id present in both "dev" and "test" split lists."""
        timeline = _clean_timeline()
        split_membership = {
            "P001": ["dev", "test"],
        }

        violations = LeakageAuditor().audit(
            {"P001": timeline},
            adjustment_set=ADJUSTMENT_SET,
            split_membership=split_membership,
        )

        matching = [v for v in violations if v.rule == "duplicate_patient_across_splits"]
        assert len(matching) == 1
        assert matching[0].patient_id == "P001"
        assert matching[0].evidence["splits"] == ["dev", "test"]

    def test_patient_in_single_split_is_not_flagged(self):
        timeline = _clean_timeline()
        split_membership = {"P001": ["dev"]}

        violations = LeakageAuditor().audit(
            {"P001": timeline},
            adjustment_set=ADJUSTMENT_SET,
            split_membership=split_membership,
        )
        assert [v for v in violations if v.rule == "duplicate_patient_across_splits"] == []

    def test_no_split_membership_supplied_skips_check(self):
        timeline = _clean_timeline()
        violations = LeakageAuditor().audit(
            {"P001": timeline}, adjustment_set=ADJUSTMENT_SET, split_membership=None
        )
        assert [v for v in violations if v.rule == "duplicate_patient_across_splits"] == []


class TestSelfAdjustmentSafeguard:
    def test_cd8_as_outcome_and_covariate_raises(self):
        """Fixture 3: CD8 used as both outcome and covariate in the same
        regression must raise SelfAdjustmentError."""
        with pytest.raises(SelfAdjustmentError):
            assert_no_self_adjustment("baseline_cd8", ["baseline_cd8", "age", "karnofsky"])

    def test_outcome_not_in_adjustment_set_does_not_raise(self):
        # Should not raise.
        assert_no_self_adjustment("baseline_cd8", ["age", "karnofsky"])

    def test_error_message_names_the_outcome_field(self):
        with pytest.raises(SelfAdjustmentError, match="baseline_cd8"):
            assert_no_self_adjustment("baseline_cd8", ["baseline_cd8"])


def test_clean_cohort_produces_no_violations():
    """A well-formed synthetic cohort of two patients triggers zero violations
    across every check when supplied consistent, non-leaky inputs."""
    timeline_a = _clean_timeline("P001")
    timeline_b = _clean_timeline("P002")

    covariate_timestamps = {
        "age": ASSIGNMENT_TIME,
        "baseline_cd4": ASSIGNMENT_TIME,
        "baseline_cd8": ASSIGNMENT_TIME,
    }
    static_covariate_events = {
        "P001": covariate_timestamps,
        "P002": covariate_timestamps,
    }
    split_membership = {"P001": ["dev"], "P002": ["test"]}

    violations = LeakageAuditor().audit(
        {"P001": timeline_a, "P002": timeline_b},
        adjustment_set=ADJUSTMENT_SET,
        split_membership=split_membership,
        static_covariate_events=static_covariate_events,
    )

    assert violations == []


def test_audit_accepts_list_input_as_well_as_dict():
    """`timelines` may be a plain list rather than a dict keyed by patient_id."""
    timeline_a = _clean_timeline("P001")
    timeline_b = _clean_timeline("P002")

    violations = LeakageAuditor().audit([timeline_a, timeline_b], adjustment_set=ADJUSTMENT_SET)
    assert violations == []


def test_leakage_violation_is_frozen_dataclass():
    violation = LeakageViolation(
        patient_id="P001", rule="post_treatment_baseline_variable", evidence={}
    )
    with pytest.raises(AttributeError):
        violation.patient_id = "P002"  # type: ignore[misc]
