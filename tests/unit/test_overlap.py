"""Unit tests for `src/causal/overlap.py`.

Builds two synthetic cohorts (same hand-built-event style as
`test_propensity.py`):

- a "well-overlapping" cohort: >=100 patients per arm, propensities
  clustered near 0.5 (well within the primary `[0.10, 0.90]` band), and
  covariates with near-identical means/variances between arms (small SMDs)
  -- asserts ALL FOUR frozen conditions pass and `support_valid=True`.
- a "poorly-overlapping" cohort: few patients per arm, propensities pushed
  to extremes, and covariates with very different means between arms
  (large SMD) -- asserts at least one condition fails and
  `support_valid=False`.

**Validates: Requirements 7.6, 7.7**
"""

from datetime import datetime

from causal.overlap import (
    MAX_ABS_SMD,
    MIN_ARM_COUNT,
    MIN_EFFECTIVE_SAMPLE_SIZE,
    OverlapDiagnostics,
)
from cohorts.cohort_construction import PatientRecord
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
ADJUSTMENT_SET = ["baseline_cd4", "karnofsky_score", "age"]


def _event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="baseline_cd4",
        value=350.0,
        unit="cells/mm3",
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
    baseline_cd4: float,
    karnofsky_score: float,
    age: float,
) -> PatientRecord:
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=baseline_cd4),
        _event(patient_id, concept_id="karnofsky_score", value=karnofsky_score, unit="score"),
        _event(patient_id, concept_id="age", value=age, unit="years"),
    ]
    timeline = build_patient_timeline(events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=timeline,
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME,
            event_indicator=False,
            censoring_indicator=True,
            event_type=None,
        ),
    )


def _well_overlapping_cohort() -> tuple[list[PatientRecord], dict[str, float]]:
    """120 patients per arm, propensities near 0.5, near-identical covariates."""
    patients: list[PatientRecord] = []
    propensities: dict[str, float] = {}

    n_per_arm = 120
    for arm in (0, 1):
        for i in range(n_per_arm):
            pid = f"A{arm}_P{i:04d}"
            # Covariates drawn from the same small range regardless of arm,
            # so means/variances are near-identical between arms (small SMD).
            baseline_cd4 = 300.0 + (i % 20)
            karnofsky_score = 80.0 + (i % 5)
            age = 35.0 + (i % 15)
            patients.append(
                _make_patient_record(
                    pid,
                    treatment_arm=arm,
                    baseline_cd4=baseline_cd4,
                    karnofsky_score=karnofsky_score,
                    age=age,
                )
            )
            # Propensities clustered tightly around 0.5, alternating
            # slightly above/below so there's variation without leaving
            # the primary common-support band.
            propensities[pid] = 0.5 + (0.01 if i % 2 == 0 else -0.01)

    return patients, propensities


def _poorly_overlapping_cohort() -> tuple[list[PatientRecord], dict[str, float]]:
    """Few patients per arm, extreme propensities, and large covariate imbalance."""
    patients: list[PatientRecord] = []
    propensities: dict[str, float] = {}

    n_per_arm = 20  # well below MIN_ARM_COUNT
    for arm in (0, 1):
        for i in range(n_per_arm):
            pid = f"B{arm}_P{i:04d}"
            # Arm 1 covariates are far from arm 0's -- large SMD.
            if arm == 1:
                baseline_cd4 = 600.0 + i
                karnofsky_score = 95.0 + i
                age = 60.0 + i
            else:
                baseline_cd4 = 150.0 + i
                karnofsky_score = 60.0 + i
                age = 25.0 + i
            patients.append(
                _make_patient_record(
                    pid,
                    treatment_arm=arm,
                    baseline_cd4=baseline_cd4,
                    karnofsky_score=karnofsky_score,
                    age=age,
                )
            )
            # Propensities pushed to extremes, outside the primary band.
            propensities[pid] = 0.97 if arm == 1 else 0.02

    return patients, propensities


class TestWellOverlappingCohort:
    def test_all_four_conditions_pass_and_support_valid(self):
        patients, propensities = _well_overlapping_cohort()
        report = OverlapDiagnostics.compute(patients, propensities, ADJUSTMENT_SET)

        assert report.arm_counts_sufficient is True
        assert report.ess_sufficient is True
        assert report.smd_sufficient is True
        assert report.no_unresolved_positivity_warning is True
        assert report.support_valid is True

        assert report.arm_counts[0] >= MIN_ARM_COUNT
        assert report.arm_counts[1] >= MIN_ARM_COUNT
        assert report.ess_by_arm[0] >= MIN_EFFECTIVE_SAMPLE_SIZE
        assert report.ess_by_arm[1] >= MIN_EFFECTIVE_SAMPLE_SIZE
        for smd in report.smd_by_covariate.values():
            assert abs(smd) < MAX_ABS_SMD
        assert report.extreme_weight_count == 0
        assert report.common_support.primary_fraction == 1.0

    def test_arm_propensity_summary_reflects_clustering_near_half(self):
        patients, propensities = _well_overlapping_cohort()
        report = OverlapDiagnostics.compute(patients, propensities, ADJUSTMENT_SET)

        for arm_summary in report.arm_propensity_summary.values():
            assert 0.45 <= arm_summary.mean <= 0.55
            assert 0.10 <= arm_summary.minimum
            assert arm_summary.maximum <= 0.90


class TestPoorlyOverlappingCohort:
    def test_at_least_one_condition_fails_and_support_invalid(self):
        patients, propensities = _poorly_overlapping_cohort()
        report = OverlapDiagnostics.compute(patients, propensities, ADJUSTMENT_SET)

        assert report.support_valid is False
        assert report.arm_counts_sufficient is False  # only 20 per arm
        assert report.no_unresolved_positivity_warning is False  # all extreme
        assert report.extreme_weight_count == len(patients)

    def test_large_covariate_imbalance_yields_high_smd(self):
        patients, propensities = _poorly_overlapping_cohort()
        report = OverlapDiagnostics.compute(patients, propensities, ADJUSTMENT_SET)

        assert report.smd_sufficient is False
        assert any(abs(smd) >= MAX_ABS_SMD for smd in report.smd_by_covariate.values())


class TestErrorHandling:
    def test_missing_propensity_raises_value_error(self):
        patients, propensities = _well_overlapping_cohort()
        incomplete = dict(propensities)
        del incomplete[patients[0].patient_id]

        try:
            OverlapDiagnostics.compute(patients, incomplete, ADJUSTMENT_SET)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for missing propensity score")
