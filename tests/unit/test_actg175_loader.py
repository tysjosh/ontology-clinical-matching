"""Unit tests for `src/data/loaders/actg175.py`.

**Validates: Requirements 5.3, 5.4**

Real ACTG175 raw tables are not available in this environment
(`docs/data_access.md`). These tests load the synthetic, clearly-documented
fixture at `tests/fixtures/data/actg175_synthetic/` instead of real patient
data.
"""

from pathlib import Path

import pytest

from data.loaders.actg175 import (
    Actg175DataNotFoundError,
    ChecksumMismatchError,
    MalformedRecordError,
    RawTables,
    compute_actg175_checksum,
    load_actg175,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "data" / "actg175_synthetic"


def test_checksum_verification_passes_when_expected_checksum_matches():
    """Loading with the fixture's actual computed checksum succeeds and
    returns schema-validated `RawTables`.
    """
    actual_checksum = compute_actg175_checksum(FIXTURE_DIR)

    tables = load_actg175(FIXTURE_DIR, expected_checksum=actual_checksum)

    assert isinstance(tables, RawTables)
    assert tables.checksum_sha256 == actual_checksum
    assert len(tables.demographics) == 6
    assert len(tables.treatment_arm) == 6
    assert len(tables.baseline_covariates) == 6
    assert len(tables.longitudinal_cd4_cd8) == 18
    assert len(tables.censoring) == 6
    assert len(tables.outcomes) == 6


def test_checksum_verification_fails_with_deliberately_wrong_expected_checksum():
    """A deliberately wrong expected checksum raises `ChecksumMismatchError`
    and does not return any parsed data.
    """
    wrong_checksum = "0" * 64

    with pytest.raises(ChecksumMismatchError, match="Checksum mismatch"):
        load_actg175(FIXTURE_DIR, expected_checksum=wrong_checksum)


def test_load_actg175_raises_on_missing_directory(tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(Actg175DataNotFoundError):
        load_actg175(missing_dir, expected_checksum="0" * 64)


def test_load_actg175_raises_on_missing_table_file(tmp_path):
    """A directory missing one of the six expected table files raises
    `Actg175DataNotFoundError` rather than silently loading a partial set.
    """
    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    for name in (
        "demographics.csv",
        "treatment_arm.csv",
        "baseline_covariates.csv",
        "longitudinal_cd4_cd8.csv",
        "censoring.csv",
        # outcomes.csv intentionally omitted
    ):
        (incomplete_dir / name).write_text("patient_id\n")

    with pytest.raises(Actg175DataNotFoundError):
        load_actg175(incomplete_dir, expected_checksum="0" * 64)


def test_compute_checksum_is_deterministic_across_repeated_calls():
    first = compute_actg175_checksum(FIXTURE_DIR)
    second = compute_actg175_checksum(FIXTURE_DIR)

    assert first == second
    assert len(first) == 64


def test_schema_validation_succeeds_for_well_formed_synthetic_rows():
    """Every row in the synthetic fixture is well-formed; loading succeeds
    and yields typed records with the expected field values for one patient.
    """
    actual_checksum = compute_actg175_checksum(FIXTURE_DIR)

    tables = load_actg175(FIXTURE_DIR, expected_checksum=actual_checksum)

    demographics_by_id = {record.patient_id: record for record in tables.demographics}
    patient = demographics_by_id["SYN-001"]
    assert patient.age == 34
    assert patient.sex == "M"
    assert patient.hemophilia is False

    treatment_by_id = {record.patient_id: record for record in tables.treatment_arm}
    assert treatment_by_id["SYN-001"].arm == 1
    assert treatment_by_id["SYN-002"].arm == 0

    # A blank CD8 cell for SYN-002 at week 96 is preserved as None, not
    # coerced to 0.0 or silently dropped.
    week_96_syn002 = next(
        record
        for record in tables.longitudinal_cd4_cd8
        if record.patient_id == "SYN-002" and record.week == 96
    )
    assert week_96_syn002.cd4 is None
    assert week_96_syn002.cd8 == 910.0


def test_schema_validation_fails_clearly_for_missing_required_field(tmp_path):
    """A row missing a required field raises `MalformedRecordError` naming the
    table and the offending line, rather than silently coercing/dropping it.
    """
    bad_dir = tmp_path / "bad_missing_field"
    bad_dir.mkdir()
    for name in (
        "treatment_arm.csv",
        "baseline_covariates.csv",
        "longitudinal_cd4_cd8.csv",
        "censoring.csv",
        "outcomes.csv",
    ):
        (bad_dir / name).write_text((FIXTURE_DIR / name).read_text())

    # demographics.csv with the required `age` column entirely removed.
    (bad_dir / "demographics.csv").write_text(
        "patient_id,sex,race_ethnicity,weight_kg,hemophilia,homosexual_activity,"
        "iv_drug_use_history\n"
        "SYN-001,M,white,72.5,False,True,False\n"
    )

    checksum = compute_actg175_checksum(bad_dir)

    with pytest.raises(MalformedRecordError, match="demographics"):
        load_actg175(bad_dir, expected_checksum=checksum)


def test_schema_validation_fails_clearly_for_wrong_type(tmp_path):
    """A row with a non-numeric value in a numeric field raises
    `MalformedRecordError`.
    """
    bad_dir = tmp_path / "bad_wrong_type"
    bad_dir.mkdir()
    for name in (
        "demographics.csv",
        "treatment_arm.csv",
        "longitudinal_cd4_cd8.csv",
        "censoring.csv",
        "outcomes.csv",
    ):
        (bad_dir / name).write_text((FIXTURE_DIR / name).read_text())

    # baseline_covariates.csv with a non-numeric `baseline_cd4` value.
    (bad_dir / "baseline_covariates.csv").write_text(
        "patient_id,karnofsky_score,prior_opportunistic_infection,symptomatic_status,"
        "baseline_cd4,baseline_cd8,prior_art_exposure_days\n"
        "SYN-001,90,False,False,not_a_number,820.0,120.0\n"
    )

    checksum = compute_actg175_checksum(bad_dir)

    with pytest.raises(MalformedRecordError, match="baseline_covariates"):
        load_actg175(bad_dir, expected_checksum=checksum)


def test_schema_validation_rejects_unknown_column(tmp_path):
    """`extra="forbid"`: an unrecognized column is rejected rather than
    silently ignored.
    """
    bad_dir = tmp_path / "bad_extra_column"
    bad_dir.mkdir()
    for name in (
        "demographics.csv",
        "treatment_arm.csv",
        "baseline_covariates.csv",
        "longitudinal_cd4_cd8.csv",
        "outcomes.csv",
    ):
        (bad_dir / name).write_text((FIXTURE_DIR / name).read_text())

    (bad_dir / "censoring.csv").write_text(
        "patient_id,censored,censoring_time_days,censoring_reason,unexpected_extra_column\n"
        "SYN-001,True,672.0,administrative_end_of_follow_up,surprise\n"
    )

    checksum = compute_actg175_checksum(bad_dir)

    with pytest.raises(MalformedRecordError, match="censoring"):
        load_actg175(bad_dir, expected_checksum=checksum)


def test_load_actg175_uses_frozen_contract_checksum_by_default():
    """When `expected_checksum` is omitted, the loader falls back to the
    checksum recorded in the frozen dataset contract
    (`configs/datasets/actg175_v3_2.yaml`). Since that value is currently a
    documented placeholder (real data has not been ingested), loading the
    synthetic fixture without an explicit checksum must fail with
    `ChecksumMismatchError`, not silently succeed.
    """
    with pytest.raises(ChecksumMismatchError):
        load_actg175(FIXTURE_DIR)
