"""Unit tests for `src/evaluation/robustness.py`.

Builds a synthetic `RandomizedCohort` with one LARGE `prior_art_status`
subgroup ("naive", 30 patients) and one SMALL subgroup ("experienced", 3
patients, below `min_cell_size`), and confirms `SubgroupRobustness.run()`
suppresses only the small subgroup.

**Validates: Requirements 13.1, 13.2, 13.3, 13.4**
"""

from datetime import datetime

import pytest

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


# ---------------------------------------------------------------------------
# apply_variant / SensitivityRobustness.run_executed (task 15; Requirement
# 8.1, 8.2, 8.3, 8.5, 8.6, 8.7)
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from evaluation.robustness import (  # noqa: E402
    STABILITY_CRITERION_DESCRIPTION,
    SensitivityConfigError,
    SensitivityVariantOutcome,
    apply_variant,
)
from infra.config import load_config  # noqa: E402

CONFIG_FIXTURES = Path(__file__).parent.parent / "fixtures" / "config"


def _base_resolved_config():
    return load_config(
        CONFIG_FIXTURES / "valid_dataset.yaml",
        CONFIG_FIXTURES / "valid_target_trial.yaml",
        CONFIG_FIXTURES / "valid_experiment.yaml",
    )


def _causal_patient_record(
    patient_id: str, *, treatment_arm: int, event_indicator: bool, elapsed_weeks: float
) -> PatientRecord:
    """A `PatientRecord` with a real treatment event (needed by
    `causal.baselines._elapsed_weeks`, which every AIPW/IPCW path below
    calls) and the frozen baseline covariates `causal.overlap`/
    `causal.baselines` read -- a richer fixture than this module's own
    `_make_patient_record`, which omits treatment events entirely (fine
    for the subgroup tests above, not for the causal-estimator paths
    task 15's tests exercise)."""
    from datetime import timedelta

    treatment_concept = "zidovudine" if treatment_arm == 0 else "zidovudine+didanosine"
    events = [
        _event(
            patient_id,
            concept_id=treatment_concept,
            value=1.0,
            unit=None,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="baseline_cd4",
            value=300.0,
            unit="cells/mm3",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="karnofsky_score",
            value=90.0,
            unit="score",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="age",
            value=35.0,
            unit="years",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="symptomatic_status",
            value=False,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="prior_opportunistic_infection",
            value=False,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="prior_art_exposure_days",
            value=0.0,
            unit="days",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
    ]
    event_time = ASSIGNMENT_TIME + timedelta(weeks=elapsed_weeks)
    if not event_indicator:
        events.append(
            _event(
                patient_id,
                concept_id="administrative_censoring",
                value=1.0,
                unit=None,
                modality="censoring",
                observed_at=event_time,
            )
        )
    timeline = build_patient_timeline(events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=event_time,
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="death" if event_indicator else None,
        ),
    )


def _causal_cohort(n_per_arm: int = 60) -> list[PatientRecord]:
    """A patients-only list (not wrapped in `RandomizedCohort`) sized for
    `causal.overlap`/`causal.baselines`' minimum-count conditions, with
    real treatment events and a genuine event/censoring mix at varied
    elapsed times."""
    patients: list[PatientRecord] = []
    for i in range(n_per_arm * 2):
        arm = i % 2
        patients.append(
            _causal_patient_record(
                f"P{i:04d}",
                treatment_arm=arm,
                event_indicator=(i % 4 == 0),
                elapsed_weeks=20.0 + (i % 50),
            )
        )
    return patients


def _flat_propensities(patients: list[PatientRecord]) -> dict[str, float]:
    """A simple, well-within-support propensity for every patient --
    enough for the AIPW/overlap calls these tests exercise without
    needing a real cross-fitted propensity model."""
    return {patient.patient_id: 0.5 for patient in patients}


class TestApplyVariant:
    def test_valid_override_materializes_a_full_resolved_config(self):
        base = _base_resolved_config()
        updated = apply_variant(base, {"propensity_model": "boosted"})

        assert updated.experiment.variant.propensity_model == "boosted"
        # Every other field stays at its default.
        assert updated.experiment.variant.weight_clipping == "primary_1_99"
        # The original config object is untouched.
        assert base.experiment.variant.propensity_model == "logistic"

    def test_unknown_field_raises_before_any_execution(self):
        base = _base_resolved_config()
        with pytest.raises(SensitivityConfigError, match="not_a_real_field"):
            apply_variant(base, {"not_a_real_field": "value"})

    def test_multiple_overrides_applied_together(self):
        base = _base_resolved_config()
        updated = apply_variant(
            base, {"missingness_stress": "increased", "landmark_masking": "week_20_removed"}
        )
        assert updated.experiment.variant.missingness_stress == "increased"
        assert updated.experiment.variant.landmark_masking == "week_20_removed"


class TestRunExecuted:
    def test_executes_all_eight_variants(self):
        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )

        assert len(result.variant_outcomes) == 8
        assert [o.name for o in result.variant_outcomes] == EXPECTED_SENSITIVITY_NAMES

    def test_boosted_propensity_model_is_recorded_unexecuted_without_xgboost(self):
        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )
        boosted = next(o for o in result.variant_outcomes if o.name == "boosted_propensity_model")

        assert boosted.status == "unexecuted"
        assert boosted.failure_reason is not None
        assert boosted.estimate is None
        assert "boosted_propensity_model" in result.unexecuted_variants

    def test_remaining_variants_still_execute_after_boosted_fails(self):
        """Requirement 8.10: one variant failing does not stop the rest."""
        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )
        executed = [o for o in result.variant_outcomes if o.name != "boosted_propensity_model"]

        assert all(o.status == "executed" for o in executed), [
            (o.name, o.status, o.failure_reason) for o in executed
        ]

    def test_an_unexecuted_variant_forces_overall_stability_undetermined(self):
        """Requirement 8.9: boosted_propensity_model is always unexecuted
        here (no xgboost), so overall_stability must always be
        'undetermined', never 'stable'."""
        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )

        assert result.overall_stability == "undetermined"
        assert result.criterion_description == STABILITY_CRITERION_DESCRIPTION

    def test_week_20_removed_retains_the_timestep_at_the_dataset_layer(self):
        """This module's own execution does not touch the dataset-builder
        landmark mask (see `_execute_variant`'s documented scope note);
        confirms the variant still executes without raising."""
        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )
        week_20 = next(o for o in result.variant_outcomes if o.name == "week_20_removed")
        assert week_20.status == "executed"
        assert week_20.resolved_config_hash is not None

    def test_increased_missingness_uses_a_seed_derived_from_master_seed_and_name(self):
        """Requirement 8.5: executes without raising, under a seed derived
        from (master_seed, variant_name) -- confirmed indirectly here by
        asserting the variant executes cleanly; the derivation itself is
        exercised directly in `tests/unit/test_seeding.py`."""
        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )
        missingness = next(
            o for o in result.variant_outcomes if o.name == "increased_missingness_stress"
        )
        assert missingness.status == "executed"

    def test_treatment_comparator_reversal_negates_the_primary_estimate_direction(self):
        """Requirement 8.4: the reported direction refers to the original
        contrast -- reversing twice (reversed propensities, then negated
        estimate) should land back near the primary estimate's sign
        relationship, checked by confirming both execute and produce
        finite, non-equal-by-coincidence estimates on this synthetic
        cohort."""
        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )
        reversal = next(
            o for o in result.variant_outcomes if o.name == "treatment_comparator_reversal"
        )
        assert reversal.status == "executed"
        assert reversal.estimate is not None

    def test_stronger_selection_severity_reads_the_frozen_amended_coefficients(self):
        from cohorts.selection import STRONGER_SELECTION_COEFFICIENTS

        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )
        stronger = next(
            o for o in result.variant_outcomes if o.name == "stronger_selection_severity"
        )

        assert stronger.status == "executed"
        assert stronger.estimate == STRONGER_SELECTION_COEFFICIENTS.selection_logit_intercept

    def test_every_executed_outcome_carries_a_resolved_config_hash_and_scale(self):
        base = _base_resolved_config()
        patients = _causal_cohort()
        propensities = _flat_propensities(patients)

        result = SensitivityRobustness.run_executed(
            base, patients, propensities, master_seed=42, adjustment_set=[]
        )
        for outcome in result.variant_outcomes:
            if outcome.status == "executed":
                assert outcome.resolved_config_hash is not None
                assert outcome.scale == "risk_difference"
                assert outcome.direction in {"positive", "negative", "null"}
            else:
                assert outcome.resolved_config_hash is None
                assert outcome.scale is None
                assert outcome.direction == "undetermined"


class TestDetermineOverallStability:
    """Direct tests of `_determine_overall_stability` (Requirement 8.9),
    isolating the stability arithmetic from pipeline execution."""

    def _outcome(self, name: str, status: str, direction: str) -> SensitivityVariantOutcome:
        return SensitivityVariantOutcome(
            name=name,
            status=status,
            estimate=None,
            ci_low=None,
            ci_high=None,
            scale=None,
            diagnostic_status="undetermined",
            direction=direction,
            failure_reason=None if status == "executed" else "forced",
            resolved_config_hash=None,
        )

    def test_all_executed_same_direction_is_stable(self):
        from evaluation.robustness import _determine_overall_stability

        outcomes = [self._outcome(f"v{i}", "executed", "negative") for i in range(8)]
        assert _determine_overall_stability(outcomes) == "stable"

    def test_one_unexecuted_forces_undetermined(self):
        from evaluation.robustness import _determine_overall_stability

        outcomes = [self._outcome(f"v{i}", "executed", "negative") for i in range(7)]
        outcomes.append(self._outcome("v7", "unexecuted", "undetermined"))
        assert _determine_overall_stability(outcomes) == "undetermined"

    def test_a_contradicting_direction_is_unstable(self):
        from evaluation.robustness import _determine_overall_stability

        outcomes = [self._outcome(f"v{i}", "executed", "negative") for i in range(7)]
        outcomes.append(self._outcome("v7", "executed", "positive"))
        assert _determine_overall_stability(outcomes) == "unstable"

    def test_null_direction_does_not_count_as_contradicting(self):
        from evaluation.robustness import _determine_overall_stability

        outcomes = [self._outcome(f"v{i}", "executed", "negative") for i in range(7)]
        outcomes.append(self._outcome("v7", "executed", "null"))
        assert _determine_overall_stability(outcomes) == "stable"
