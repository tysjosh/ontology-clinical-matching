"""Unit tests for the Section 13 adjustment-set mapping and the propensity
model's 14-column encoding (neural-experiment-pipeline task 7.5).

Covers:

- the explicit Section-13-name -> concept-id(s) mapping in
  `causal.adjustment_set`, checked against `configs/target_trials/main.yaml`
  and against the keys the real-data adapter actually writes;
- each encoding in `causal.propensity` (continuous z-score, boolean,
  `sex` / `race_ethnicity` categorical with reference level, unseen-category
  and missing-value rules, prior-ART exposure *and* duration);
- that every fitted parameter comes from the fitting patients only
  (development / training folds), including inside cross-fitting;
- the leakage audit over the concept ids the model reads, and that no
  post-baseline measurement enters the feature vector;
- that the negative-control-outcome regression's covariate list is
  unchanged: `spec.adjustment_set` minus `baseline_cd8`, never read from
  `COVARIATE_FIELDS` and never from a propensity score (Section 11.3).

**Validates: Requirements 1.1, 1.4, 1.7, 11.3, 18.1, 18.2, 18.5**
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

import causal.negative_controls as negative_controls
from causal.adjustment_set import (
    ADJUSTMENT_COVARIATES_BY_NAME,
    ADJUSTMENT_SET_NAMES,
    SECTION_13_ADJUSTMENT_SET,
    resolve_raw_value,
)
from causal.negative_controls import BASELINE_CD8_FIELD, run_negative_control_outcome
from causal.propensity import (
    COVARIATE_FIELDS,
    LogisticPropensityModel,
    _feature_vector,
    _fit_standardization,
)
from cohorts.cohort_construction import PatientRecord
from data.event_model import ClinicalEvent, MissingnessState
from data.leakage_audit import LeakageAuditor
from data.loaders.actg175 import compute_actg175_checksum, load_actg175
from data.loaders.actg175_adapter import ASSIGNMENT_TIME_ANCHOR, raw_tables_to_pipeline_inputs
from data.timelines import build_patient_timeline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult
from protocols.schema import TargetTrialSpec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRIAL_CONFIG = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "data" / "actg175_synthetic"

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)

EXPECTED_COLUMNS = (
    "age",
    "weight",
    "sex_female",
    "race_ethnicity_non_white",
    "hemophilia",
    "homosexual_activity",
    "intravenous_drug_use_history",
    "karnofsky_score",
    "prior_opportunistic_infection",
    "symptomatic_status",
    "baseline_cd4",
    "baseline_cd8",
    "prior_art",
    "prior_art_duration",
)

#: Default stored values, keyed by the loader's concept ids.
_DEFAULTS: dict[str, Any] = {
    "age": 35.0,
    "weight_kg": 70.0,
    "sex": "M",
    "race_ethnicity": "white",
    "hemophilia": False,
    "homosexual_activity": False,
    "iv_drug_use_history": False,
    "karnofsky_score": 90.0,
    "prior_opportunistic_infection": False,
    "symptomatic_status": False,
    "baseline_cd4": 350.0,
    "baseline_cd8": 900.0,
    "prior_art_exposure_days": 0.0,
}


def _event(patient_id: str, concept_id: str, value: Any, **overrides: Any) -> ClinicalEvent:
    fields: dict[str, Any] = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id=concept_id,
        value=value,
        unit=None,
        modality="baseline",
        observed_at=ASSIGNMENT_TIME,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    fields.update(overrides)
    return ClinicalEvent(**fields)


def _patient(
    patient_id: str,
    *,
    treatment_arm: int = 0,
    omit: tuple[str, ...] = (),
    extra_events: tuple[ClinicalEvent, ...] = (),
    **values: Any,
) -> PatientRecord:
    stored = {**_DEFAULTS, **values}
    events = [
        _event(patient_id, concept_id, value)
        for concept_id, value in stored.items()
        if concept_id not in omit
    ]
    events.extend(extra_events)
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=build_patient_timeline(events),
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME + timedelta(weeks=50),
            event_indicator=False,
            censoring_indicator=True,
            event_type=None,
        ),
    )


def _column(row: list[float], name: str) -> float:
    return row[COVARIATE_FIELDS.index(name)]


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


def test_mapping_names_are_exactly_the_frozen_adjustment_set_in_order():
    spec = TargetTrialSpec.model_validate(yaml.safe_load(TRIAL_CONFIG.read_text("utf-8")))
    assert ADJUSTMENT_SET_NAMES == tuple(spec.adjustment_set)
    assert len(SECTION_13_ADJUSTMENT_SET) == 13


def test_covariate_fields_are_the_fourteen_encoded_columns():
    assert COVARIATE_FIELDS == EXPECTED_COLUMNS
    assert len(COVARIATE_FIELDS) == 14


def test_renamed_covariates_map_to_the_loader_keys_not_their_section_13_names():
    by_name = ADJUSTMENT_COVARIATES_BY_NAME
    assert by_name["weight"].concept_ids[0] == "weight_kg"
    assert by_name["intravenous_drug_use_history"].concept_ids == ("iv_drug_use_history",)
    assert by_name["prior_antiretroviral_exposure_and_duration"].concept_ids == (
        "prior_art_exposure_days",
        "prior_antiretroviral_exposure_duration",
    )
    # A value stored only under the Section 13 name is not picked up.
    assert (
        resolve_raw_value(
            {"intravenous_drug_use_history": True}, by_name["intravenous_drug_use_history"]
        )
        is None
    )


def test_every_covariate_resolves_on_adapter_built_timelines():
    tables = load_actg175(FIXTURE_DIR, expected_checksum=compute_actg175_checksum(FIXTURE_DIR))
    timelines, _, _ = raw_tables_to_pipeline_inputs(tables)
    assert timelines
    for timeline in timelines:
        for covariate in SECTION_13_ADJUSTMENT_SET:
            assert (
                resolve_raw_value(timeline.static_covariates, covariate) is not None
            ), f"{covariate.name} did not resolve for {timeline.patient_id}"


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------


def test_sex_encoding_reference_male_and_missing_or_unknown_is_zero():
    fit = [_patient("A", sex="M"), _patient("B", sex="F")]
    params = _fit_standardization(fit)
    assert params.category_sets["sex"] == frozenset({"M", "F"})

    assert _column(_feature_vector(_patient("X", sex="F"), params), "sex_female") == 1.0
    assert _column(_feature_vector(_patient("X", sex="M"), params), "sex_female") == 0.0
    assert _column(_feature_vector(_patient("X", omit=("sex",)), params), "sex_female") == 0.0
    # Outside the closed {"M", "F"} set: treated as missing.
    assert _column(_feature_vector(_patient("X", sex="U"), params), "sex_female") == 0.0


def test_sex_level_not_seen_in_fitting_patients_encodes_as_reference():
    params = _fit_standardization([_patient("A", sex="M"), _patient("B", sex="M")])
    assert _column(_feature_vector(_patient("X", sex="F"), params), "sex_female") == 0.0


def test_race_ethnicity_reference_white_unseen_and_missing_are_zero():
    fit = [
        _patient("A", race_ethnicity="white"),
        _patient("B", race_ethnicity="non-white"),
    ]
    params = _fit_standardization(fit)
    assert params.category_sets["race_ethnicity"] == frozenset({"white", "non-white"})

    def encoded(patient: PatientRecord) -> float:
        return _column(_feature_vector(patient, params), "race_ethnicity_non_white")

    assert encoded(_patient("X", race_ethnicity="non-white")) == 1.0
    assert encoded(_patient("X", race_ethnicity="white")) == 0.0
    assert encoded(_patient("X", race_ethnicity="unseen-level")) == 0.0
    assert encoded(_patient("X", omit=("race_ethnicity",))) == 0.0


def test_boolean_covariates_read_through_the_mapping():
    params = _fit_standardization([_patient("A"), _patient("B")])
    row = _feature_vector(
        _patient(
            "X",
            hemophilia=True,
            homosexual_activity=True,
            iv_drug_use_history=True,
            prior_opportunistic_infection=True,
            symptomatic_status=True,
        ),
        params,
    )
    for name in (
        "hemophilia",
        "homosexual_activity",
        "intravenous_drug_use_history",
        "prior_opportunistic_infection",
        "symptomatic_status",
    ):
        assert _column(row, name) == 1.0

    missing = _feature_vector(
        _patient("Y", omit=("hemophilia", "iv_drug_use_history", "symptomatic_status")), params
    )
    for name in ("hemophilia", "intravenous_drug_use_history", "symptomatic_status"):
        assert _column(missing, name) == 0.0


def test_weight_is_z_scored_from_weight_kg_and_missing_is_excluded_from_the_fit():
    fit = [
        _patient("A", weight_kg=60.0),
        _patient("B", weight_kg=80.0),
        _patient("C", omit=("weight_kg",)),
    ]
    params = _fit_standardization(fit)
    assert params.means["weight"] == pytest.approx(70.0)
    assert params.stdevs["weight"] == pytest.approx(10.0)

    assert _column(_feature_vector(_patient("X", weight_kg=90.0), params), "weight") == (
        pytest.approx(2.0)
    )
    assert _column(_feature_vector(_patient("X", omit=("weight_kg",)), params), "weight") == 0.0


def test_baseline_cd8_is_an_ordinary_z_scored_column():
    params = _fit_standardization(
        [_patient("A", baseline_cd8=500.0), _patient("B", baseline_cd8=1500.0)]
    )
    row = _feature_vector(_patient("X", baseline_cd8=1500.0), params)
    assert _column(row, "baseline_cd8") == pytest.approx(1.0)


def test_prior_art_encodes_exposure_and_duration():
    fit = [
        _patient("A", prior_art_exposure_days=0.0),
        _patient("B", prior_art_exposure_days=400.0),
    ]
    params = _fit_standardization(fit)
    assert params.means["prior_art_duration"] == pytest.approx(200.0)
    assert params.stdevs["prior_art_duration"] == pytest.approx(200.0)

    naive = _feature_vector(_patient("N", prior_art_exposure_days=0.0), params)
    short = _feature_vector(_patient("S", prior_art_exposure_days=100.0), params)
    long_ = _feature_vector(_patient("L", prior_art_exposure_days=600.0), params)

    assert (_column(naive, "prior_art"), _column(naive, "prior_art_duration")) == (
        0.0,
        pytest.approx(-1.0),
    )
    # Both experienced patients share the indicator; duration separates them.
    assert _column(short, "prior_art") == _column(long_, "prior_art") == 1.0
    assert _column(short, "prior_art_duration") == pytest.approx(-0.5)
    assert _column(long_, "prior_art_duration") == pytest.approx(2.0)

    missing = _feature_vector(_patient("M", omit=("prior_art_exposure_days",)), params)
    assert (_column(missing, "prior_art"), _column(missing, "prior_art_duration")) == (0.0, 0.0)


def test_prior_art_falls_back_to_the_normalized_concept_id():
    fit = [
        _patient(
            "A", omit=("prior_art_exposure_days",), prior_antiretroviral_exposure_duration=0.0
        ),
        _patient(
            "B", omit=("prior_art_exposure_days",), prior_antiretroviral_exposure_duration=100.0
        ),
    ]
    params = _fit_standardization(fit)
    row = _feature_vector(fit[1], params)
    assert _column(row, "prior_art") == 1.0
    assert _column(row, "prior_art_duration") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Development-fold-only fitting
# ---------------------------------------------------------------------------


def test_fitted_parameters_ignore_patients_outside_the_fitting_set():
    development = [_patient("A", age=30.0), _patient("B", age=40.0, race_ethnicity="white")]
    held_out = _patient("Z", age=1000.0, race_ethnicity="non-white")

    params = _fit_standardization(development)
    assert params == _fit_standardization(list(development))
    assert params.means["age"] == pytest.approx(35.0)
    assert "non-white" not in params.category_sets["race_ethnicity"]
    # The held-out category was never fit, so it encodes as the reference.
    assert _column(_feature_vector(held_out, params), "race_ethnicity_non_white") == 0.0


def test_cross_fitting_fits_each_fold_on_its_training_patients_only():
    patients = [
        _patient(
            f"P{i:02d}",
            treatment_arm=i % 2,
            age=25.0 + i,
            weight_kg=55.0 + 2 * i,
            sex="F" if i % 3 == 0 else "M",
            race_ethnicity="non-white" if i % 4 == 0 else "white",
            prior_art_exposure_days=float(30 * (i % 5)),
        )
        for i in range(20)
    ]
    by_id = {patient.patient_id: patient for patient in patients}
    model = LogisticPropensityModel(seed=0)
    fitted_params = []
    original_fit = model._fit

    def recording_fit(train_patients):
        fitted = original_fit(train_patients)
        fitted_params.append((tuple(p.patient_id for p in train_patients), fitted.standardization))
        return fitted

    model._fit = recording_fit  # type: ignore[method-assign]
    result = model.fit_cross_fitted(patients, n_folds=5)

    assert len(fitted_params) == 5
    for fold, (train_ids, params) in enumerate(fitted_params):
        assert train_ids == result.training_patient_ids_by_fold[fold]
        assert not set(train_ids) & set(result.prediction_ids_for(fold))
        assert params == _fit_standardization([by_id[pid] for pid in train_ids])


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_leakage_audit_passes_over_the_concept_ids_the_model_reads():
    tables = load_actg175(FIXTURE_DIR, expected_checksum=compute_actg175_checksum(FIXTURE_DIR))
    timelines, _, _ = raw_tables_to_pipeline_inputs(tables)
    read_concept_ids = sorted(
        {
            concept_id
            for covariate in SECTION_13_ADJUSTMENT_SET
            for concept_id in covariate.concept_ids
        }
    )
    static_covariate_events = {
        timeline.patient_id: {
            concept_id: ASSIGNMENT_TIME_ANCHOR
            for concept_id in read_concept_ids
            if concept_id in timeline.static_covariates
        }
        for timeline in timelines
    }

    violations = LeakageAuditor().audit(
        timelines, read_concept_ids, static_covariate_events=static_covariate_events
    )
    assert violations == []


def test_no_section_13_excluded_field_is_mapped():
    """Section 13 excludes week-20/96 CD4/CD8, treatment discontinuation,
    future measurements, event-derived and post-time-zero censoring fields.
    None of the concept ids that name them is read."""
    read = {c for covariate in SECTION_13_ADJUSTMENT_SET for c in covariate.concept_ids}
    excluded = {
        "cd4_count",
        "cd8_count",
        "administrative_censoring",
        "zidovudine",
        "zidovudine+didanosine",
    }
    assert read.isdisjoint(excluded)
    for concept_id in read:
        lowered = concept_id.lower()
        for marker in ("week", "outcome", "endpoint", "event", "censor", "discontinu"):
            assert marker not in lowered, concept_id


def test_post_baseline_measurements_do_not_change_the_feature_vector():
    later_labs = tuple(
        _event(
            "X",
            concept_id,
            value,
            modality="lab",
            unit="cells/mm3",
            observed_at=ASSIGNMENT_TIME + timedelta(weeks=week),
        )
        for week in (20, 96)
        for concept_id, value in (("cd4_count", 50.0), ("cd8_count", 3000.0))
    )
    params = _fit_standardization([_patient("A"), _patient("B", baseline_cd4=500.0)])
    without = _feature_vector(_patient("X"), params)
    with_labs = _feature_vector(_patient("X", extra_events=later_labs), params)
    assert with_labs == without


# ---------------------------------------------------------------------------
# Negative-control outcome (Section 11.3) is unaffected
# ---------------------------------------------------------------------------


def test_negative_control_outcome_covariates_are_adjustment_set_minus_cd8():
    spec = TargetTrialSpec.model_validate(yaml.safe_load(TRIAL_CONFIG.read_text("utf-8")))
    patients = [
        _patient(f"P{i:02d}", treatment_arm=i % 2, baseline_cd8=700.0 + 25 * i, age=30.0 + i)
        for i in range(12)
    ]
    result = run_negative_control_outcome(patients, spec.adjustment_set, seed=7, n_bootstrap=3)

    expected = [name for name in spec.adjustment_set if name != BASELINE_CD8_FIELD]
    assert result.evidence["regression_covariates"] == expected
    assert len(expected) == 12
    assert BASELINE_CD8_FIELD not in result.evidence["regression_covariates"]


def test_negative_control_outcome_reads_no_propensity_model_state():
    """The regression reads no propensity model or score: the function takes
    none, and the module never constructs a propensity model. Since task 7.8
    it imports `causal.propensity`'s *encoding* helpers, so the Section 13
    covariates are encoded the same way everywhere; that reads covariates,
    not propensity scores."""
    import ast

    tree = ast.parse(inspect.getsource(negative_controls))
    imported_from_propensity = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "causal.propensity"
        for alias in node.names
    }
    encoding_helpers = {"COVARIATE_FIELDS", "_feature_vector", "_fit_standardization"}
    assert imported_from_propensity <= encoding_helpers
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not called & {"LogisticPropensityModel", "BoostedPropensityModel", "fit_cross_fitted"}
    parameters = inspect.signature(run_negative_control_outcome).parameters
    assert list(parameters) == ["patients", "adjustment_set", "seed", "n_bootstrap"]


def test_negative_control_outcome_regression_uses_the_shared_section_13_encoding():
    """Task 7.8: the CD8 regression resolves Section 13 names through the shared
    mapping, so stored-key and categorical covariates are real columns, not
    silent zeros, and `baseline_cd8` stays off its own right-hand side."""
    from causal.negative_controls import _regression_covariate_rows

    patients = [
        _patient(
            f"P{i:02d}",
            treatment_arm=i % 2,
            age=25.0 + i,
            weight_kg=55.0 + 2 * i,
            sex="F" if i % 3 == 0 else "M",
            race_ethnicity="non-white" if i % 4 == 0 else "white",
            prior_art_exposure_days=float(30 * (i % 5)),
        )
        for i in range(20)
    ]
    covariates = [name for name in ADJUSTMENT_SET_NAMES if name != BASELINE_CD8_FIELD]
    rows = _regression_covariate_rows(patients, covariates)
    expected = [
        column
        for column in COVARIATE_FIELDS
        if column not in ADJUSTMENT_COVARIATES_BY_NAME[BASELINE_CD8_FIELD].columns
    ]
    assert len(rows[0]) == len(expected) == len(COVARIATE_FIELDS) - 1
    for name in ("weight", "sex_female", "race_ethnicity_non_white", "prior_art_duration"):
        column = expected.index(name)
        assert any(row[column] != 0.0 for row in rows), name
