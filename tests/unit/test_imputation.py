"""Unit tests for `src/data/imputation.py`.

Covers the fold-safe imputer's development-only fitting contract
(Requirement 5.12, 6.5) and the missingness-indicator helpers
(Requirement 5.13).

**Validates: Requirements 5.12, 5.13**
"""

from datetime import datetime

import pytest

from data.event_model import ClinicalEvent, MissingnessState
from data.imputation import (
    DevelopmentFold,
    DevelopmentFoldRequiredError,
    FoldSafeImputer,
    ImputerNotFittedError,
    missingness_indicators_from_records,
    missingness_indicators_from_timelines,
)
from data.timelines import build_patient_timeline

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


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


class TestFoldSafeImputerDevelopmentOnlyFit:
    """`FoldSafeImputer.fit()` must compute parameters from development data only."""

    # Development fold: cd4 values 100, 200, missing -> mean of observed = 150
    DEV_RECORDS = [
        {"cd4": 100.0, "age": 30.0},
        {"cd4": 200.0, "age": 40.0},
        {"cd4": None, "age": 50.0},
    ]
    # Validation fold with deliberately different underlying values: mean would be 900
    # if (incorrectly) used to fit.
    VAL_RECORDS = [
        {"cd4": 800.0, "age": 60.0},
        {"cd4": 1000.0, "age": 70.0},
    ]

    def test_fit_requires_development_fold_wrapper(self):
        """A bare list (unlabeled data) must be rejected by .fit()."""
        imputer = FoldSafeImputer(fields=["cd4", "age"])
        with pytest.raises(DevelopmentFoldRequiredError):
            imputer.fit(self.DEV_RECORDS)  # not wrapped in DevelopmentFold

    def test_fitted_means_come_from_development_data_only(self):
        """Fitted means must match development-only computation, not validation."""
        imputer = FoldSafeImputer(fields=["cd4", "age"], strategy="mean")
        imputer.fit(DevelopmentFold(records=self.DEV_RECORDS))

        params = imputer.fitted_params
        # mean of observed cd4 values in DEV_RECORDS: (100 + 200) / 2 = 150
        assert params.means["cd4"] == pytest.approx(150.0)
        # mean of age in DEV_RECORDS: (30 + 40 + 50) / 3 = 40
        assert params.means["age"] == pytest.approx(40.0)

        # Prove validation data was not used: fitting on validation alone would
        # produce a very different mean.
        val_only_imputer = FoldSafeImputer(fields=["cd4", "age"], strategy="mean")
        val_only_imputer.fit(DevelopmentFold(records=self.VAL_RECORDS))
        val_params = val_only_imputer.fitted_params
        assert val_params.means["cd4"] == pytest.approx(900.0)
        assert params.means["cd4"] != val_params.means["cd4"]

    def test_fitted_medians_come_from_development_data_only(self):
        imputer = FoldSafeImputer(fields=["cd4"], strategy="median")
        imputer.fit(DevelopmentFold(records=self.DEV_RECORDS))
        assert imputer.fitted_params.medians["cd4"] == pytest.approx(150.0)

    def test_transform_applies_frozen_dev_params_to_validation_fold(self):
        """.transform() on validation data must use dev-fitted params unchanged,
        not recompute parameters from the validation data it's given."""
        imputer = FoldSafeImputer(fields=["cd4"], strategy="mean")
        imputer.fit(DevelopmentFold(records=self.DEV_RECORDS))

        val_with_missing = [{"cd4": None}, {"cd4": 500.0}]
        imputed = imputer.transform(val_with_missing)

        # The missing value must be filled with the dev-fitted mean (150.0),
        # not the validation fold's own values.
        assert imputed[0]["cd4"] == pytest.approx(150.0)
        assert imputed[1]["cd4"] == pytest.approx(500.0)  # observed value untouched

    def test_transform_before_fit_raises(self):
        imputer = FoldSafeImputer(fields=["cd4"])
        with pytest.raises(ImputerNotFittedError):
            imputer.transform([{"cd4": None}])

    def test_multiple_imputation_draws_are_deterministic_given_seed(self):
        imputer = FoldSafeImputer(fields=["cd4"], strategy="mean")
        imputer.fit(DevelopmentFold(records=self.DEV_RECORDS))

        draws_a = imputer.transform([{"cd4": None}], n_imputations=3, seed=42)
        draws_b = imputer.transform([{"cd4": None}], n_imputations=3, seed=42)
        assert draws_a == draws_b
        assert len(draws_a) == 3

    def test_field_with_no_observed_development_values_is_left_missing(self):
        dev = [{"never_observed": None}, {"never_observed": None}]
        imputer = FoldSafeImputer(fields=["never_observed"])
        imputer.fit(DevelopmentFold(records=dev))

        imputed = imputer.transform([{"never_observed": None}])
        assert imputed[0]["never_observed"] is None


class TestMissingnessIndicators:
    """Missingness-indicator helpers against small synthetic sets."""

    def test_indicators_from_plain_records(self):
        records = [
            {"cd4": 100.0, "karnofsky": None},
            {"cd4": None, "karnofsky": 90.0},
        ]
        indicators = missingness_indicators_from_records(records, fields=["cd4", "karnofsky"])

        assert indicators == [
            {"cd4": 0, "karnofsky": 1},
            {"cd4": 1, "karnofsky": 0},
        ]

    def test_indicators_from_timelines_observed_and_missing(self):
        observed_event = _event(
            patient_id="P001", concept_id="age", value=34.0, missingness=MissingnessState.OBSERVED
        )
        missing_event = _event(
            patient_id="P001",
            concept_id="karnofsky",
            value=None,
            missingness=MissingnessState.MISSING,
        )
        timeline = build_patient_timeline([observed_event, missing_event])

        indicators = missingness_indicators_from_timelines(
            {"P001": timeline}, fields=["age", "karnofsky"]
        )

        assert indicators == {"P001": {"age": 0, "karnofsky": 1}}

    def test_indicators_from_timelines_field_absent_entirely_is_missing(self):
        observed_event = _event(patient_id="P001", concept_id="age", value=34.0)
        timeline = build_patient_timeline([observed_event])

        indicators = missingness_indicators_from_timelines(
            [timeline], fields=["age", "baseline_cd4"]
        )

        assert indicators == {"P001": {"age": 0, "baseline_cd4": 1}}

    def test_indicators_from_timelines_not_applicable_counts_as_missing(self):
        na_event = _event(
            patient_id="P001",
            concept_id="prior_art_duration",
            value=None,
            missingness=MissingnessState.NOT_APPLICABLE,
        )
        timeline = build_patient_timeline([na_event])

        indicators = missingness_indicators_from_timelines(
            [timeline], fields=["prior_art_duration"]
        )

        assert indicators == {"P001": {"prior_art_duration": 1}}
