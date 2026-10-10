"""The Section 12 SMD condition gates on weighted balance over the full
adjustment set (neural-experiment-pipeline task 7.6; `decisions.md` Section 26,
2026-09-30 amendment)."""

from __future__ import annotations

import math

import pytest

from causal.adjustment_set import ADJUSTMENT_SET_NAMES, ENCODED_COLUMNS
from causal.overlap import MAX_ABS_SMD, SMD_BASIS, OverlapDiagnostics
from tests.unit.test_adjustment_set_encoding import _patient
from tests.unit.test_overlap import _make_patient_record

ADJUSTMENT_SET = ["baseline_cd4", "karnofsky_score", "age"]


def _confounded_cohort():
    """Age drives treatment; the supplied propensities are the true ones.

    Ten patients per age from 20 to 79; at each age the number treated is
    round(10 * e(age)) with e(age) = sigmoid((age - 50) / 15), and every
    patient's propensity is that age's empirical treated fraction, so
    inverse-probability weighting balances age exactly before clipping.
    """
    patients, propensities = [], {}
    for age in range(20, 80):
        n_treated = round(10 / (1 + math.exp(-(age - 50) / 15)))
        for k in range(10):
            pid = f"A{age}_{k}"
            arm = 1 if k < n_treated else 0
            patients.append(
                _make_patient_record(
                    pid, treatment_arm=arm, baseline_cd4=300.0, karnofsky_score=90.0, age=age
                )
            )
            propensities[pid] = min(max(n_treated / 10, 0.05), 0.95)
    return patients, propensities


def test_raw_imbalance_from_designed_confounding_does_not_block_the_gate():
    patients, propensities = _confounded_cohort()
    report = OverlapDiagnostics.compute(patients, propensities, ADJUSTMENT_SET)
    assert report.smd_basis == SMD_BASIS
    assert abs(report.raw_smd_by_covariate["age"]) >= MAX_ABS_SMD  # descriptive
    assert abs(report.smd_by_covariate["age"]) < MAX_ABS_SMD  # gated
    assert report.smd_exclusions == {}
    assert report.smd_sufficient is True


def test_constant_propensity_gives_unit_weights_so_weighted_equals_raw():
    patients, _ = _confounded_cohort()
    share = sum(p.treatment_arm for p in patients) / len(patients)
    report = OverlapDiagnostics.compute(
        patients, {p.patient_id: share for p in patients}, ADJUSTMENT_SET
    )
    for column, raw in report.raw_smd_by_covariate.items():
        assert report.smd_by_covariate[column] == pytest.approx(raw, abs=1e-12)
    assert report.smd_sufficient is False  # unadjusted age imbalance remains


def test_a_covariate_unobserved_in_one_arm_is_excluded_and_blocks_the_gate():
    patients, propensities = _confounded_cohort()
    report = OverlapDiagnostics.compute(patients, propensities, [*ADJUSTMENT_SET, "not_recorded"])
    assert set(report.smd_exclusions) == {"not_recorded"}
    assert "not counted as balanced" in report.smd_exclusions["not_recorded"]
    assert "not_recorded" not in report.smd_by_covariate
    assert report.smd_sufficient is False
    assert report.support_valid is False


def test_full_section_13_set_resolves_stored_keys_and_categories():
    patients = [
        _patient(
            f"P{i:03d}",
            treatment_arm=i % 2,
            age=25.0 + (i % 17),
            weight_kg=55.0 + (i % 13),
            sex="F" if i % 3 == 0 else "M",
            race_ethnicity="non-white" if i % 4 == 0 else "white",
            prior_art_exposure_days=float(30 * (i % 5)),
        )
        for i in range(240)
    ]
    propensities = {p.patient_id: 0.5 for p in patients}
    report = OverlapDiagnostics.compute(patients, propensities, list(ADJUSTMENT_SET_NAMES))
    assert report.smd_exclusions == {}
    assert list(report.smd_by_covariate) == list(ENCODED_COLUMNS)
    assert set(report.raw_smd_by_covariate) == set(ENCODED_COLUMNS)


def test_constant_covariates_have_exactly_zero_weighted_smd():
    patients, propensities = _confounded_cohort()
    report = OverlapDiagnostics.compute(patients, propensities, ADJUSTMENT_SET)
    assert report.smd_by_covariate["baseline_cd4"] == 0.0
    assert report.smd_by_covariate["karnofsky_score"] == 0.0
