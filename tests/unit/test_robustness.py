"""Unit tests for `src/evaluation/robustness.py`.

Builds a synthetic `RandomizedCohort` with one LARGE `prior_art_status`
subgroup ("naive", 30 patients) and one SMALL subgroup ("experienced", 3
patients, below `min_cell_size`), and confirms `SubgroupRobustness.run()`
suppresses only the small subgroup.

**Validates: Requirements 13.1, 13.2, 13.3, 13.4**
"""

from datetime import datetime

from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from evaluation.robustness import (
    DEFAULT_MIN_CELL_SIZE,
    SENSITIVITY_VARIANTS,
    SensitivityRobustness,
    SubgroupRobustness,
)
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


def _event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="prior_art_exposure_days",
        value=0.0,
        unit="days",
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


def _make_patient_record(
    patient_id: str,
    *,
    treatment_arm: int,
    prior_art_exposure_days: float,
    event_indicator: bool,
) -> PatientRecord:
    events = [
        _event(patient_id, concept_id="prior_art_exposure_days", value=prior_art_exposure_days),
    ]
    timeline = build_patient_timeline(events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME,
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="death" if event_indicator else None,
        ),
    )


def _build_cohort() -> RandomizedCohort:
    patients: list[PatientRecord] = []

    # LARGE subgroup: 30 "naive" patients (prior_art_exposure_days <= 0),
    # split across both arms with a mix of events/censoring.
    for i in range(30):
        patients.append(
            _make_patient_record(
                f"naive_{i:03d}",
                treatment_arm=i % 2,
                prior_art_exposure_days=0.0,
                event_indicator=(i % 3 == 0),
            )
        )

    # SMALL subgroup: 3 "experienced" patients (prior_art_exposure_days > 0),
    # below DEFAULT_MIN_CELL_SIZE.
    for i in range(3):
        patients.append(
            _make_patient_record(
                f"experienced_{i:03d}",
                treatment_arm=i % 2,
                prior_art_exposure_days=100.0,
                event_indicator=(i == 0),
            )
        )

    patients.sort(key=lambda record: record.patient_id)
    return RandomizedCohort(patients=tuple(patients))


class TestSubgroupSuppression:
    def test_small_subgroup_suppressed_large_subgroup_not(self):
        cohort = _build_cohort()

        results = SubgroupRobustness.run(cohort, subgroup_field="prior_art_status")

        assert set(results) == {"naive", "experienced"}

        naive = results["naive"]
        experienced = results["experienced"]

        assert naive.sample_size == 30
        assert naive.suppressed is False
        assert naive.suppression_reason is None

        assert experienced.sample_size == 3
        assert experienced.suppressed is True
        assert experienced.suppression_reason is not None
        assert "3" in experienced.suppression_reason

    def test_treatment_and_endpoint_counts_reported_per_subgroup(self):
        cohort = _build_cohort()

        results = SubgroupRobustness.run(cohort, subgroup_field="prior_art_status")
        naive = results["naive"]

        assert sum(naive.treatment_counts.values()) == naive.sample_size
        assert sum(naive.endpoint_counts.values()) == naive.sample_size
        assert set(naive.endpoint_counts) == {"event", "censored"}

    def test_overlap_none_when_propensities_not_supplied(self):
        cohort = _build_cohort()

        results = SubgroupRobustness.run(cohort, subgroup_field="prior_art_status")

        for result in results.values():
            assert result.overlap is None
            assert result.uncertainty is None

    def test_custom_min_cell_size_changes_suppression_boundary(self):
        cohort = _build_cohort()

        # With a lower min_cell_size, the 3-patient subgroup is no longer
        # below threshold.
        results = SubgroupRobustness.run(
            cohort, subgroup_field="prior_art_status", min_cell_size=2
        )
        assert results["experienced"].suppressed is False

        # Sanity: the default threshold is higher than 3, confirming the
        # first test's suppression is meaningful, not incidental.
        assert DEFAULT_MIN_CELL_SIZE > 3

    def test_unsupported_subgroup_field_raises(self):
        cohort = _build_cohort()

        try:
            SubgroupRobustness.run(cohort, subgroup_field="site")
        except NotImplementedError:
            pass
        else:
            raise AssertionError("Expected NotImplementedError for unsupported subgroup_field")


# ---------------------------------------------------------------------------
# Sensitivity-analysis variants (decisions.md Section 20, "Sensitivity
# analyses"; Requirements 13.5, 13.6, 13.7; task 13.1).
# ---------------------------------------------------------------------------

EXPECTED_SENSITIVITY_NAMES = [
    "boosted_propensity_model",
    "alternative_weight_clipping",
    "alternative_support_thresholds",
    "increased_missingness_stress",
    "week_20_removed",
    "treatment_comparator_reversal",
    "shifted_endpoint_horizon",
    "stronger_selection_severity",
]


class TestSensitivityRobustness:
    def test_exactly_eight_variants(self):
        """`decisions.md` Section 20 freezes exactly eight sensitivity variants."""
        assert len(SENSITIVITY_VARIANTS) == 8
        assert len(SensitivityRobustness.run()) == 8

    def test_variant_names_are_unique(self):
        assert len({variant.name for variant in SENSITIVITY_VARIANTS}) == 8

    def test_variant_names_are_stable(self):
        """Names must match the frozen set exactly -- an accidental rename of
        any entry must fail this test rather than pass silently.
        """
        assert [variant.name for variant in SENSITIVITY_VARIANTS] == EXPECTED_SENSITIVITY_NAMES

    def test_every_variant_has_a_nonempty_description_and_config_diff(self):
        for variant in SENSITIVITY_VARIANTS:
            assert isinstance(variant.description, str)
            assert variant.description.strip() != ""
            assert isinstance(variant.config_diff, dict)
            assert len(variant.config_diff) > 0

    def test_every_variant_config_diff_is_distinct(self):
        """Each variant produces a distinct, correctly-labeled config."""
        diffs = [variant.config_diff for variant in SENSITIVITY_VARIANTS]
        for i, diff_a in enumerate(diffs):
            for j, diff_b in enumerate(diffs):
                if i != j:
                    assert diff_a != diff_b, (
                        f"config_diff for {SENSITIVITY_VARIANTS[i].name!r} matches "
                        f"{SENSITIVITY_VARIANTS[j].name!r}"
                    )

    def test_run_returns_the_module_level_constant(self):
        assert SensitivityRobustness.run() == SENSITIVITY_VARIANTS
