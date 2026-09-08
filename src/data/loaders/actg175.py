"""Raw ACTG175 table loader (Requirement 5.3-5.4).

`src/data/contract.py::DatasetContract` owns the frozen dataset identity and
the frozen SHA-256 checksum (`decisions.md` Section 2: "SHA-256 checksum
computed and frozen at ingestion; pipeline refuses to run on a mismatched
checksum"). This module is the concrete loader `design.md` Section 3.3
describes as `load_actg175(path) -> RawTables`: it reads the packaged
`LongCART::ACTG175` v3.2 raw tables, verifies their checksum against the
frozen contract value (or an explicitly supplied one), and parses every row
into a typed, schema-validated Pydantic record.

Real ACTG175 raw files are not available in this environment
(`docs/data_access.md` Section 5: "not yet recorded"). This loader is
implemented against a small **synthetic fixture** that mirrors the real
ACTG175 table structure — see `tests/fixtures/data/actg175_synthetic/`,
which is explicitly documented there as synthetic, not real patient data.
The loader itself makes no assumption that the data is synthetic: once real
raw tables are obtained and placed under `data/raw/` (per
`docs/data_access.md` Section 2), this same function loads them unchanged.

Raw-table directory layout expected at `path`:

```text
<path>/
├── demographics.csv          -> DemographicsRecord
├── treatment_arm.csv         -> TreatmentArmRecord
├── baseline_covariates.csv   -> BaselineCovariateRecord
├── longitudinal_cd4_cd8.csv  -> LongitudinalCD4CD8Record
├── censoring.csv             -> CensoringRecord
└── outcomes.csv              -> OutcomeRecord
```

Every record model uses `extra="forbid"` and is frozen, matching the
strictness convention already used by `src/data/contract.py` and
`src/protocols/schema.py`: a malformed row (missing required field, wrong
type, unrecognized column) raises a clear error rather than being silently
coerced or dropped (Requirement 5.4).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from data.contract import DatasetContract, assert_contract_present

# Every model in this file shares the same strictness contract: no unknown
# keys, no in-place mutation (mirrors `src/data/contract.py` and
# `src/protocols/schema.py`).
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

#: Location of the frozen dataset contract that supplies the default
#: expected checksum when the caller does not pass one explicitly.
DEFAULT_CONTRACT_PATH = (
    Path(__file__).parent.parent.parent.parent / "configs" / "datasets" / "actg175_v3_2.yaml"
)

#: The six raw tables this loader parses, and their per-table CSV filenames.
_TABLE_FILENAMES: dict[str, str] = {
    "demographics": "demographics.csv",
    "treatment_arm": "treatment_arm.csv",
    "baseline_covariates": "baseline_covariates.csv",
    "longitudinal_cd4_cd8": "longitudinal_cd4_cd8.csv",
    "censoring": "censoring.csv",
    "outcomes": "outcomes.csv",
}


class Actg175DataNotFoundError(Exception):
    """Raised when the raw-table directory, or one of its expected table files, is missing."""


class ChecksumMismatchError(Exception):
    """Raised when the computed SHA-256 checksum of the raw tables does not match the
    expected checksum (`decisions.md` Section 2: "pipeline refuses to run on a
    mismatched checksum").
    """


class MalformedRecordError(Exception):
    """Raised when a raw-table row fails schema validation.

    Carries the table name, source file, and (1-indexed, header-inclusive)
    line number of the offending row, plus the underlying
    `pydantic.ValidationError`, so a malformed row is diagnosable without
    re-running the loader under a debugger.
    """


class DemographicsRecord(BaseModel):
    """One patient's demographic baseline record (`decisions.md` Section 13
    adjustment set: age, weight, sex, race/ethnicity, hemophilia, homosexual
    activity, IV drug-use history).
    """

    model_config = _MODEL_CONFIG

    patient_id: str
    age: float
    sex: Literal["M", "F"]
    race_ethnicity: str
    weight_kg: float
    hemophilia: bool
    homosexual_activity: bool
    iv_drug_use_history: bool


class TreatmentArmRecord(BaseModel):
    """One patient's randomized treatment-arm assignment.

    Raw arm values span the full trial (0-3); restricting the analysis
    cohort to arms 0/1 (`decisions.md` Section 3-4) is a downstream cohort-
    construction step, not a loader-level filter — the raw loader must not
    silently drop arms 2/3 rows.
    """

    model_config = _MODEL_CONFIG

    patient_id: str
    arm: int = Field(ge=0, le=3)
    treatment_label: str


class BaselineCovariateRecord(BaseModel):
    """One patient's baseline clinical covariates (`decisions.md` Section 13
    adjustment set: Karnofsky score, prior opportunistic infection,
    symptomatic status, baseline CD4, baseline CD8, prior antiretroviral
    exposure).
    """

    model_config = _MODEL_CONFIG

    patient_id: str
    karnofsky_score: float
    prior_opportunistic_infection: bool
    symptomatic_status: bool
    baseline_cd4: float
    baseline_cd8: float
    prior_art_exposure_days: float


class LongitudinalCD4CD8Record(BaseModel):
    """One patient/timepoint longitudinal CD4/CD8 measurement
    (`decisions.md` Section 5 landmark timepoints: baseline, week 20, week 96).

    `cd4`/`cd8` are nullable because not every landmark measurement is
    available for every patient (missingness is expected and handled
    downstream by `src/data/normalization.py` / the missingness mask, not
    silently dropped here).
    """

    model_config = _MODEL_CONFIG

    patient_id: str
    week: int
    cd4: float | None
    cd8: float | None


class CensoringRecord(BaseModel):
    """One patient's censoring status (`decisions.md` Section 5: administrative
    end of follow-up, or last recorded observation).
    """

    model_config = _MODEL_CONFIG

    patient_id: str
    censored: bool
    censoring_time_days: float
    censoring_reason: str | None


class OutcomeRecord(BaseModel):
    """One patient's composite-endpoint outcome (`decisions.md` Section 6)."""

    model_config = _MODEL_CONFIG

    patient_id: str
    event_indicator: bool
    event_time_days: float
    event_type: str | None


class RawTables(BaseModel):
    """The complete set of typed, schema-validated raw ACTG175 tables
    returned by `load_actg175` (`design.md` Section 3.3).

    Each table is a tuple of validated records; `checksum_sha256` records
    the verified checksum of the raw tables this instance was built from, so
    downstream provenance/manifest code can record exactly what was loaded
    without recomputing the hash.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    demographics: tuple[DemographicsRecord, ...]
    treatment_arm: tuple[TreatmentArmRecord, ...]
    baseline_covariates: tuple[BaselineCovariateRecord, ...]
    longitudinal_cd4_cd8: tuple[LongitudinalCD4CD8Record, ...]
    censoring: tuple[CensoringRecord, ...]
    outcomes: tuple[OutcomeRecord, ...]
    checksum_sha256: str


def compute_actg175_checksum(path: Path | str) -> str:
    """Compute the SHA-256 checksum of the raw ACTG175 table directory at `path`.

    The checksum is computed deterministically over the six expected table
    files, in a fixed (sorted-by-table-name) order, hashing each file's name
    and exact byte content. This makes the checksum sensitive to any change
    in any table's content, independent of filesystem directory-listing
    order.

    Raises:
        Actg175DataNotFoundError: if `path` is not a directory, or a table
            file listed in `_TABLE_FILENAMES` is missing from it.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise Actg175DataNotFoundError(
            f"Raw ACTG175 data path not found or not a directory: {directory}"
        )

    hasher = hashlib.sha256()
    for table_name in sorted(_TABLE_FILENAMES):
        table_path = directory / _TABLE_FILENAMES[table_name]
        if not table_path.exists():
            raise Actg175DataNotFoundError(
                f"Expected raw ACTG175 table file not found: {table_path}"
            )
        hasher.update(table_name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(table_path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _clean_row(row: dict[str, str | None]) -> dict[str, str | None]:
    """Convert empty-string CSV cells to `None` before schema validation.

    `csv.DictReader` returns `""` for empty cells; without this conversion,
    an optional nullable field (e.g. `cd4`) with a blank cell would fail
    `float | None` validation instead of being treated as missing.
    """
    return {key: (value if value != "" else None) for key, value in row.items()}


def _parse_table(
    csv_path: Path,
    model_cls: type[BaseModel],
    table_name: str,
) -> tuple[BaseModel, ...]:
    records: list[BaseModel] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):  # header occupies line 1
            try:
                records.append(model_cls.model_validate(_clean_row(row)))
            except ValidationError as exc:
                raise MalformedRecordError(
                    f"Malformed row in '{table_name}' table "
                    f"({csv_path.name}, line {line_number}): {exc}"
                ) from exc
    return tuple(records)


def load_actg175(path: Path | str, expected_checksum: str | None = None) -> RawTables:
    """Load, checksum-verify, and schema-validate the raw ACTG175 tables at `path`.

    Args:
        path: directory containing the six raw ACTG175 table CSVs (see
            module docstring for the expected layout).
        expected_checksum: the checksum to verify against. If omitted, the
            expected checksum is read from the frozen dataset contract at
            `DEFAULT_CONTRACT_PATH` (`configs/datasets/actg175_v3_2.yaml`).

    Returns:
        A `RawTables` instance with every row validated against its
        per-table Pydantic schema.

    Raises:
        Actg175DataNotFoundError: if `path` or an expected table file within
            it is missing.
        ChecksumMismatchError: if the computed checksum does not match the
            expected checksum.
        MalformedRecordError: if any row in any table fails schema
            validation (Requirement 5.4 — malformed rows are rejected, never
            silently coerced or dropped).
    """
    directory = Path(path)

    actual_checksum = compute_actg175_checksum(directory)

    if expected_checksum is None:
        contract: DatasetContract = assert_contract_present(DEFAULT_CONTRACT_PATH)
        expected_checksum = contract.checksum_sha256

    if actual_checksum != expected_checksum:
        raise ChecksumMismatchError(
            f"Checksum mismatch for raw ACTG175 data at {directory}: "
            f"expected {expected_checksum}, computed {actual_checksum}. "
            "The pipeline refuses to run on a mismatched checksum "
            "(decisions.md Section 2)."
        )

    demographics = _parse_table(
        directory / _TABLE_FILENAMES["demographics"], DemographicsRecord, "demographics"
    )
    treatment_arm = _parse_table(
        directory / _TABLE_FILENAMES["treatment_arm"], TreatmentArmRecord, "treatment_arm"
    )
    baseline_covariates = _parse_table(
        directory / _TABLE_FILENAMES["baseline_covariates"],
        BaselineCovariateRecord,
        "baseline_covariates",
    )
    longitudinal_cd4_cd8 = _parse_table(
        directory / _TABLE_FILENAMES["longitudinal_cd4_cd8"],
        LongitudinalCD4CD8Record,
        "longitudinal_cd4_cd8",
    )
    censoring = _parse_table(
        directory / _TABLE_FILENAMES["censoring"], CensoringRecord, "censoring"
    )
    outcomes = _parse_table(directory / _TABLE_FILENAMES["outcomes"], OutcomeRecord, "outcomes")

    return RawTables(
        demographics=demographics,
        treatment_arm=treatment_arm,
        baseline_covariates=baseline_covariates,
        longitudinal_cd4_cd8=longitudinal_cd4_cd8,
        censoring=censoring,
        outcomes=outcomes,
        checksum_sha256=actual_checksum,
    )
