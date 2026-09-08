"""Unit tests for `src/cohorts/selection.py`.

Builds hand-controllable synthetic patients (same hand-built-event style as
`test_splitting.py`/`test_preprocessing.py`) and confirms:

- every module-level coefficient constant equals its frozen `decisions.md`
  Section 9 value exactly;
- `compute_risk_score` on a hand-constructed patient with known z-scores
  and indicator values matches a hand-computed expected risk score;
- `selection_probability` matches `1/(1+exp(-(1.2 + (2*arm-1)*r)))` for a
  few hand-checked `(arm, r)` pairs;
- `observationalize` with a fixed seed is reproducible, and different
  seeds can produce different retained sets.

**Validates: Requirements 7.1 (decisions.md Section 9); Correctness Property 2**
"""

import json
import math
from datetime import datetime

import pytest

from cohorts.cohort_construction import PatientRecord
from cohorts.preprocessing import DevelopmentSet, FittedPreprocessing
from cohorts.selection import (
    COEF_AGE,
    COEF_CD4,
    COEF_KARNOFSKY,
    COEF_PRIOR_ART,
    COEF_PRIOR_OI,
    COEF_SYMPTOMATIC,
    COEFFICIENTS_BY_SEVERITY,
    DEFAULT_COEFFICIENTS,
    SELECTION_LOGIT_INTERCEPT,
    STRONGER_SELECTION_COEFFICIENTS,
    ObservationalizedSplit,
    PilotRecalibrator,
    RecalibrationFreezeError,
    RecalibrationRejectedError,
    SelectionCoefficients,
    coefficients_for_severity,
    compute_risk_score,
    observationalize,
    selection_probability,
)
from cohorts.splitting import PatientSet
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


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
    symptomatic_status: bool,
    prior_opportunistic_infection: bool,
    prior_art_exposure_days: float,
) -> PatientRecord:
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=baseline_cd4),
        _event(patient_id, concept_id="karnofsky_score", value=karnofsky_score, unit="score"),
        _event(patient_id, concept_id="age", value=age, unit="years"),
        _event(patient_id, concept_id="symptomatic_status", value=symptomatic_status, unit=None),
        _event(
            patient_id,
            concept_id="prior_opportunistic_infection",
            value=prior_opportunistic_infection,
            unit=None,
        ),
        _event(
            patient_id,
            concept_id="prior_art_exposure_days",
            value=prior_art_exposure_days,
            unit="days",
        ),
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


# ---------------------------------------------------------------------------
# Coefficient equality with decisions.md Section 9's frozen formula
# ---------------------------------------------------------------------------


def test_frozen_coefficients_match_decisions_md_section_9():
    assert COEF_CD4 == -0.8
    assert COEF_KARNOFSKY == -0.3
    assert COEF_AGE == 0.3
    assert COEF_SYMPTOMATIC == 0.6
    assert COEF_PRIOR_OI == 0.6
    assert COEF_PRIOR_ART == 0.4
    assert SELECTION_LOGIT_INTERCEPT == 1.2


# ---------------------------------------------------------------------------
# compute_risk_score
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_preprocessing() -> FittedPreprocessing:
    """Fit development parameters on a small, hand-known development set so
    the resulting z-scores for the test patient below are easy to
    hand-compute.

    Development baseline_cd4 values: [200, 400] -> mean=300, stdev=141.4213562...
    Development karnofsky_score values: [80, 100] -> mean=90, stdev=14.142135...
    Development age values: [30, 50] -> mean=40, stdev=14.142135...
    """
    dev_patients = tuple(
        _make_patient_record(
            f"DEV{i:03d}",
            treatment_arm=0,
            baseline_cd4=cd4,
            karnofsky_score=karnofsky,
            age=age,
            symptomatic_status=False,
            prior_opportunistic_infection=False,
            prior_art_exposure_days=0.0,
        )
        for i, (cd4, karnofsky, age) in enumerate([(200.0, 80.0, 30.0), (400.0, 100.0, 50.0)])
    )
    dev_set = DevelopmentSet(PatientSet(dev_patients))
    preprocessing = FittedPreprocessing()
    preprocessing.fit(dev_set)
    return preprocessing


def test_compute_risk_score_matches_hand_computed_value(
    fitted_preprocessing: FittedPreprocessing,
):
    # A patient at baseline_cd4=300 (exactly the dev mean -> z=0), karnofsky=100
    # (z = (100-90)/14.142135... = +0.70710678...), age=30 (z = (30-40)/14.142135...
    # = -0.70710678...), symptomatic=True, prior_oi=True, prior_art_exposure_days=10
    # (> 0 -> PriorART=1.0).
    patient = _make_patient_record(
        "P_TEST",
        treatment_arm=1,
        baseline_cd4=300.0,
        karnofsky_score=100.0,
        age=30.0,
        symptomatic_status=True,
        prior_opportunistic_infection=True,
        prior_art_exposure_days=10.0,
    )

    params = fitted_preprocessing.fitted_params
    z_cd4 = (300.0 - params.means["baseline_cd4"]) / params.stdevs["baseline_cd4"]
    z_karnofsky = (100.0 - params.means["karnofsky_score"]) / params.stdevs["karnofsky_score"]
    z_age = (30.0 - params.means["age"]) / params.stdevs["age"]

    expected = (
        COEF_CD4 * z_cd4
        + COEF_KARNOFSKY * z_karnofsky
        + COEF_AGE * z_age
        + COEF_SYMPTOMATIC * 1.0
        + COEF_PRIOR_OI * 1.0
        + COEF_PRIOR_ART * 1.0
    )

    actual = compute_risk_score(patient, fitted_preprocessing)
    assert actual == pytest.approx(expected)
    # z_cd4 should be exactly 0.0 (patient value equals dev mean).
    assert z_cd4 == pytest.approx(0.0)


def test_compute_risk_score_with_all_indicators_false_and_no_prior_art(
    fitted_preprocessing: FittedPreprocessing,
):
    patient = _make_patient_record(
        "P_ZERO",
        treatment_arm=0,
        baseline_cd4=300.0,  # z = 0
        karnofsky_score=90.0,  # z = 0
        age=40.0,  # z = 0
        symptomatic_status=False,
        prior_opportunistic_infection=False,
        prior_art_exposure_days=0.0,
    )

    risk_score = compute_risk_score(patient, fitted_preprocessing)
    assert risk_score == pytest.approx(0.0)


def test_prior_art_derived_from_exposure_days_greater_than_zero(
    fitted_preprocessing: FittedPreprocessing,
):
    patient_with_prior_art = _make_patient_record(
        "P_ART",
        treatment_arm=0,
        baseline_cd4=300.0,
        karnofsky_score=90.0,
        age=40.0,
        symptomatic_status=False,
        prior_opportunistic_infection=False,
        prior_art_exposure_days=45.0,
    )
    patient_without_prior_art = _make_patient_record(
        "P_NO_ART",
        treatment_arm=0,
        baseline_cd4=300.0,
        karnofsky_score=90.0,
        age=40.0,
        symptomatic_status=False,
        prior_opportunistic_infection=False,
        prior_art_exposure_days=0.0,
    )

    risk_with_art = compute_risk_score(patient_with_prior_art, fitted_preprocessing)
    risk_without_art = compute_risk_score(patient_without_prior_art, fitted_preprocessing)

    assert risk_with_art - risk_without_art == pytest.approx(COEF_PRIOR_ART)


# ---------------------------------------------------------------------------
# selection_probability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "treatment_arm,risk_score",
    [
        (1, 0.0),
        (0, 0.0),
        (1, 0.5),
        (0, 0.5),
        (1, -1.2),
        (0, 2.3),
    ],
)
def test_selection_probability_matches_hand_computed_sigmoid(treatment_arm, risk_score):
    expected = 1.0 / (1.0 + math.exp(-(1.2 + (2 * treatment_arm - 1) * risk_score)))
    actual = selection_probability(risk_score, treatment_arm)
    assert actual == pytest.approx(expected)


def test_selection_probability_rejects_invalid_treatment_arm():
    with pytest.raises(ValueError):
        selection_probability(0.5, 2)


def test_selection_probability_arm_1_increases_with_positive_risk_score():
    """For arm 1, sigmoid(1.2 + r) is increasing in r."""
    low = selection_probability(risk_score=-1.0, treatment_arm=1)
    high = selection_probability(risk_score=1.0, treatment_arm=1)
    assert high > low


def test_selection_probability_arm_0_decreases_with_positive_risk_score():
    """For arm 0, sigmoid(1.2 - r) is decreasing in r."""
    low_r = selection_probability(risk_score=-1.0, treatment_arm=0)
    high_r = selection_probability(risk_score=1.0, treatment_arm=0)
    assert high_r < low_r


# ---------------------------------------------------------------------------
# observationalize
# ---------------------------------------------------------------------------


def _large_patient_set(n: int = 60) -> PatientSet:
    patients = tuple(
        _make_patient_record(
            f"P{i:04d}",
            treatment_arm=i % 2,
            baseline_cd4=200.0 + 5.0 * i,
            karnofsky_score=70.0 + (i % 5),
            age=20.0 + (i % 40),
            symptomatic_status=(i % 3 == 0),
            prior_opportunistic_infection=(i % 4 == 0),
            prior_art_exposure_days=float(i % 10),
        )
        for i in range(n)
    )
    return PatientSet(patients)


@pytest.fixture
def large_fitted_preprocessing() -> FittedPreprocessing:
    dev_set = DevelopmentSet(_large_patient_set(60))
    preprocessing = FittedPreprocessing()
    preprocessing.fit(dev_set)
    return preprocessing


def test_observationalize_returns_expected_type(large_fitted_preprocessing: FittedPreprocessing):
    patients = _large_patient_set(60)
    result = observationalize(patients, large_fitted_preprocessing, seed=123)

    assert isinstance(result, ObservationalizedSplit)
    assert result.seed == 123
    assert set(result.risk_scores) == set(patients.patient_ids)
    assert set(result.selection_probabilities) == set(patients.patient_ids)
    # Retained is a subset of the input.
    assert set(result.retained_patient_ids) <= set(patients.patient_ids)


def test_observationalize_same_seed_is_reproducible(
    large_fitted_preprocessing: FittedPreprocessing,
):
    patients = _large_patient_set(60)

    result_1 = observationalize(patients, large_fitted_preprocessing, seed=42)
    result_2 = observationalize(patients, large_fitted_preprocessing, seed=42)

    assert result_1.retained_patient_ids == result_2.retained_patient_ids
    assert result_1.risk_scores == result_2.risk_scores
    assert result_1.selection_probabilities == result_2.selection_probabilities


def test_observationalize_different_seeds_can_produce_different_retained_sets(
    large_fitted_preprocessing: FittedPreprocessing,
):
    patients = _large_patient_set(60)

    result_1 = observationalize(patients, large_fitted_preprocessing, seed=1)
    result_2 = observationalize(patients, large_fitted_preprocessing, seed=999999)

    assert result_1.retained_patient_ids != result_2.retained_patient_ids


# ---------------------------------------------------------------------------
# Stronger-severity selection (decisions.md Section 26 amendment,
# 2026-09-07, amending Sections 9/20)
# ---------------------------------------------------------------------------


def test_coefficients_for_severity_resolves_both_frozen_bundles():
    assert coefficients_for_severity("primary") is DEFAULT_COEFFICIENTS
    assert coefficients_for_severity("stronger") is STRONGER_SELECTION_COEFFICIENTS
    assert COEFFICIENTS_BY_SEVERITY["stronger"] is STRONGER_SELECTION_COEFFICIENTS


def test_coefficients_for_severity_rejects_unknown_severity():
    """An unrecognized severity is never silently treated as primary."""
    with pytest.raises(ValueError, match="unknown selection_severity"):
        coefficients_for_severity("strongest")  # type: ignore[arg-type]


def test_observationalize_default_severity_is_primary(
    large_fitted_preprocessing: FittedPreprocessing,
):
    patients = _large_patient_set(60)
    result = observationalize(patients, large_fitted_preprocessing, seed=7)

    assert result.selection_severity == "primary"
    # Naming the primary severity explicitly must be indistinguishable from
    # omitting it -- the frozen Section 9 mechanism is unchanged.
    explicit = observationalize(
        patients, large_fitted_preprocessing, seed=7, selection_severity="primary"
    )
    assert explicit.risk_scores == result.risk_scores
    assert explicit.selection_probabilities == result.selection_probabilities
    assert explicit.retained_patient_ids == result.retained_patient_ids


def test_observationalize_stronger_severity_applies_amended_coefficients(
    large_fitted_preprocessing: FittedPreprocessing,
):
    """The stronger severity must actually use the amended coefficients:
    every risk score is the primary score scaled by 1.5, and selection
    probabilities stay strictly inside (0, 1) so positivity holds.
    """
    patients = _large_patient_set(60)

    primary = observationalize(patients, large_fitted_preprocessing, seed=7)
    stronger = observationalize(
        patients, large_fitted_preprocessing, seed=7, selection_severity="stronger"
    )

    assert stronger.selection_severity == "stronger"
    assert stronger.seed == 7

    for patient_id, primary_score in primary.risk_scores.items():
        assert stronger.risk_scores[patient_id] == pytest.approx(1.5 * primary_score)

    for probability in stronger.selection_probabilities.values():
        assert 0.0 < probability < 1.0

    # At least one patient's probability must actually move, otherwise the
    # severity path is a no-op.
    assert stronger.selection_probabilities != primary.selection_probabilities


def test_observationalize_stronger_severity_is_reproducible(
    large_fitted_preprocessing: FittedPreprocessing,
):
    patients = _large_patient_set(60)

    result_1 = observationalize(
        patients, large_fitted_preprocessing, seed=42, selection_severity="stronger"
    )
    result_2 = observationalize(
        patients, large_fitted_preprocessing, seed=42, selection_severity="stronger"
    )

    assert result_1.retained_patient_ids == result_2.retained_patient_ids
    assert result_1.selection_probabilities == result_2.selection_probabilities


def test_observationalize_rejects_conflicting_coefficients_and_severity(
    large_fitted_preprocessing: FittedPreprocessing,
):
    """Passing an explicit bundle and a non-primary severity together would
    name conflicting coefficients; the ambiguity is refused rather than
    resolved silently."""
    patients = _large_patient_set(60)

    with pytest.raises(ValueError, match="conflicting coefficients"):
        observationalize(
            patients,
            large_fitted_preprocessing,
            seed=7,
            coefficients=_recalibrated_coefficients(),
            selection_severity="stronger",
        )


def test_stronger_severity_never_mutates_the_frozen_primary_coefficients(
    large_fitted_preprocessing: FittedPreprocessing,
):
    """The stronger mechanism is a variant, not a recalibration: the frozen
    Section 9 constants and `DEFAULT_COEFFICIENTS` are untouched by running
    it."""
    patients = _large_patient_set(60)

    observationalize(
        patients, large_fitted_preprocessing, seed=7, selection_severity="stronger"
    )

    assert DEFAULT_COEFFICIENTS == SelectionCoefficients(
        coef_cd4=COEF_CD4,
        coef_karnofsky=COEF_KARNOFSKY,
        coef_age=COEF_AGE,
        coef_symptomatic=COEF_SYMPTOMATIC,
        coef_prior_oi=COEF_PRIOR_OI,
        coef_prior_art=COEF_PRIOR_ART,
        selection_logit_intercept=SELECTION_LOGIT_INTERCEPT,
    )
    assert STRONGER_SELECTION_COEFFICIENTS != DEFAULT_COEFFICIENTS


# ---------------------------------------------------------------------------
# One-time pilot recalibration (decisions.md Section 9)
# ---------------------------------------------------------------------------


def _recalibrated_coefficients() -> SelectionCoefficients:
    return SelectionCoefficients(
        coef_cd4=-0.5,
        coef_karnofsky=-0.2,
        coef_age=0.2,
        coef_symptomatic=0.5,
        coef_prior_oi=0.5,
        coef_prior_art=0.3,
        selection_logit_intercept=1.0,
    )


def test_recalibrate_on_development_set_appends_changelog_and_returns_new_coefficients(
    tmp_path,
):
    dev_set = DevelopmentSet(_large_patient_set(10))
    changelog_path = tmp_path / "recalibration_changelog.jsonl"
    new_coefficients = _recalibrated_coefficients()

    recalibrator = PilotRecalibrator()
    result = recalibrator.recalibrate(
        development=dev_set,
        new_coefficients=new_coefficients,
        reason="Development arm-0 sample below minimum count (Section 12).",
        changelog_path=changelog_path,
    )

    assert result == new_coefficients

    assert changelog_path.exists()
    lines = changelog_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "recalibration"
    assert entry["reason"] == "Development arm-0 sample below minimum count (Section 12)."
    assert entry["old_coefficients"]["coef_cd4"] == DEFAULT_COEFFICIENTS.coef_cd4
    assert entry["new_coefficients"]["coef_cd4"] == new_coefficients.coef_cd4
    assert "recorded_at" in entry

    # The returned coefficients are usable in compute_risk_score/
    # selection_probability without error.
    patient = _large_patient_set(10).patients[0]
    dummy_preprocessing = FittedPreprocessing().fit(dev_set)
    risk_score = compute_risk_score(patient, dummy_preprocessing, result)
    probability = selection_probability(risk_score, patient.treatment_arm, result)
    assert 0.0 < probability < 1.0


def test_freeze_then_recalibrate_raises_freeze_error(tmp_path):
    dev_set = DevelopmentSet(_large_patient_set(10))
    changelog_path = tmp_path / "recalibration_changelog.jsonl"

    recalibrator = PilotRecalibrator()
    recalibrator.recalibrate(
        development=dev_set,
        new_coefficients=_recalibrated_coefficients(),
        reason="First and only recalibration.",
        changelog_path=changelog_path,
    )
    recalibrator.freeze(changelog_path)

    with pytest.raises(RecalibrationFreezeError):
        recalibrator.recalibrate(
            development=dev_set,
            new_coefficients=_recalibrated_coefficients(),
            reason="Second attempt should be rejected.",
            changelog_path=changelog_path,
        )


def test_freeze_persists_across_new_recalibrator_instances(tmp_path):
    """A caller cannot route around a frozen mechanism by constructing a
    fresh `PilotRecalibrator` pointed at the same changelog_path."""
    dev_set = DevelopmentSet(_large_patient_set(10))
    changelog_path = tmp_path / "recalibration_changelog.jsonl"

    first_recalibrator = PilotRecalibrator()
    first_recalibrator.freeze(changelog_path)

    second_recalibrator = PilotRecalibrator()
    with pytest.raises(RecalibrationFreezeError):
        second_recalibrator.recalibrate(
            development=dev_set,
            new_coefficients=_recalibrated_coefficients(),
            reason="Should be rejected even from a brand-new instance.",
            changelog_path=changelog_path,
        )


def test_recalibrate_on_bare_patient_set_raises_rejected_error_outright(tmp_path):
    """A bare PatientSet (not wrapped in DevelopmentSet) -- e.g. a
    validation/test split -- is rejected outright, even before any freeze
    check."""
    bare_patient_set = _large_patient_set(10)
    changelog_path = tmp_path / "recalibration_changelog.jsonl"

    recalibrator = PilotRecalibrator()
    with pytest.raises(RecalibrationRejectedError):
        recalibrator.recalibrate(
            development=bare_patient_set,
            new_coefficients=_recalibrated_coefficients(),
            reason="Should be rejected: not a DevelopmentSet.",
            changelog_path=changelog_path,
        )

    # No changelog entry should have been written for a rejected attempt.
    assert not changelog_path.exists()


def test_recalibrate_rejects_bare_patient_set_even_when_already_frozen(tmp_path):
    """Rejection on non-DevelopmentSet input takes priority over (and holds
    independently of) freeze state."""
    bare_patient_set = _large_patient_set(10)
    changelog_path = tmp_path / "recalibration_changelog.jsonl"

    recalibrator = PilotRecalibrator()
    recalibrator.freeze(changelog_path)

    with pytest.raises(RecalibrationRejectedError):
        recalibrator.recalibrate(
            development=bare_patient_set,
            new_coefficients=_recalibrated_coefficients(),
            reason="Should still be rejected as non-DevelopmentSet.",
            changelog_path=changelog_path,
        )


def test_compute_risk_score_default_behavior_unchanged(
    fitted_preprocessing: FittedPreprocessing,
):
    """compute_risk_score's default (no coefficients argument) behavior is
    unchanged from before pilot recalibration support was added."""
    patient = _make_patient_record(
        "P_DEFAULT",
        treatment_arm=1,
        baseline_cd4=300.0,
        karnofsky_score=100.0,
        age=30.0,
        symptomatic_status=True,
        prior_opportunistic_infection=True,
        prior_art_exposure_days=10.0,
    )

    with_explicit_default = compute_risk_score(patient, fitted_preprocessing, DEFAULT_COEFFICIENTS)
    with_implicit_default = compute_risk_score(patient, fitted_preprocessing)

    assert with_implicit_default == pytest.approx(with_explicit_default)


def test_selection_probability_default_behavior_unchanged():
    """selection_probability's default (no coefficients argument) behavior
    is unchanged from before pilot recalibration support was added."""
    expected = 1.0 / (1.0 + math.exp(-(1.2 + (2 * 1 - 1) * 0.5)))
    actual = selection_probability(0.5, 1)
    assert actual == pytest.approx(expected)
