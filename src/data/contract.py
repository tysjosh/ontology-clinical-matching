"""Authoritative clinical dataset contract (Requirement 5.1-5.2).

`src/infra/config.py::DatasetContractConfig` is a lightweight, config-loading-time
schema used only to fail fast on unknown top-level keys before a run starts
(`design.md` Section 3.1). This module owns the authoritative, richly-typed
`DatasetContract` that the rest of the data pipeline imports and validates
against: it supersedes `DatasetContractConfig` for every downstream consumer,
mirroring the frozen dataset contract in `decisions.md` Section 2
(`design.md` Section 3.3). `decisions.md` is the single source of truth for
every concrete value here; this module must not re-derive or re-choose any
of them.

Every model below uses `extra="forbid"`, so an incomplete or malformed
contract fails loudly with a field-level `pydantic.ValidationError` rather
than silently accepting an unrecognized key or dropping a missing one. Every
model is also frozen (immutable) — a loaded `DatasetContract` cannot be
mutated in place; encoding a different dataset version requires constructing
a new contract object, never an in-place edit of a locked value.

`assert_contract_present()` is the function later pipeline stages (the raw
ACTG175 loader, task 5.1, and beyond) call to refuse running without the
dataset contract (Requirement 5.2, `decisions.md` Section 2's "pipeline
refuses to run on a mismatched checksum" is enforced downstream in the
loader once real data is available; this module enforces the weaker but
prior condition that the contract file itself must exist and validate).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

# Every model in this file shares the same strictness contract: no unknown
# keys, no in-place mutation. Defined once and reused so the contract can't
# drift between models.
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class MissingContractError(Exception):
    """Raised when the dataset contract file is absent or fails to validate.

    The pipeline refuses to run without a valid dataset contract
    (Requirement 5.2). This single exception type covers both "the contract
    file does not exist" and "the contract file exists but is malformed or
    incomplete" — both cases mean the pipeline has no trustworthy dataset
    contract to proceed with.
    """


class PopulationInfo(BaseModel):
    """Packaged population size and description (`decisions.md` Section 2)."""

    model_config = _MODEL_CONFIG

    size: int
    description: str


class DatasetContract(BaseModel):
    """The authoritative, frozen clinical dataset contract (Requirement 5.1).

    Mirrors `decisions.md` Section 2 exactly:

    - `name`, `version`: dataset identity (`LongCART::ACTG175`, version `3.2`).
    - `population`: packaged population size/description before treatment-arm
      restriction.
    - `treatment`, `comparator`: the frozen treatment comparison
      (`decisions.md` Section 3).
    - `endpoint`: the frozen primary endpoint type (`decisions.md` Section 6).
    - `study_period`: the trial's follow-up window and landmark timepoints
      (`decisions.md` Section 5).
    - `available_modalities`: the modalities packaged with this dataset
      version (`decisions.md` Section 2).
    - `reference_effect_source`: citation for the published historical
      reference effect (`decisions.md` Section 7).
    - `checksum_sha256`: the frozen data-integrity checksum. The pipeline
      refuses to run on a mismatched checksum (`decisions.md` Section 2);
      that mismatch check is performed by the raw-data loader once real data
      is available (task 5.1), not by this schema.
    """

    model_config = _MODEL_CONFIG

    name: str
    version: str
    population: PopulationInfo
    treatment: str
    comparator: str
    endpoint: str
    study_period: str
    available_modalities: list[str]
    reference_effect_source: str
    checksum_sha256: str


def _load_yaml_mapping(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise MissingContractError(
            f"Dataset contract at {path} is not valid YAML: {exc}"
        ) from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise MissingContractError(
            f"Dataset contract at {path} must be a mapping at the top level, "
            f"got {type(data).__name__}"
        )
    return data


def assert_contract_present(contract_path: Path | str) -> DatasetContract:
    """Load and validate the dataset contract at `contract_path`.

    This is the function later pipeline stages call to refuse running
    without the dataset contract (Requirement 5.2). It raises
    `MissingContractError` if the file does not exist, is not valid YAML, is
    not a mapping at the top level, or fails `DatasetContract` schema
    validation. Otherwise it returns the validated `DatasetContract`.

    Args:
        contract_path: path to the dataset-contract YAML file (e.g.
            `configs/datasets/actg175_v3_2.yaml`).

    Returns:
        The validated `DatasetContract`.

    Raises:
        MissingContractError: if the contract is absent, malformed, or
            incomplete.
    """
    path = Path(contract_path)
    if not path.exists():
        raise MissingContractError(
            f"Dataset contract not found at {path}. The pipeline refuses to "
            "run without a versioned dataset contract (Requirement 5.2)."
        )

    raw = _load_yaml_mapping(path)

    try:
        return DatasetContract.model_validate(raw)
    except ValidationError as exc:
        raise MissingContractError(
            f"Dataset contract at {path} failed schema validation:\n{exc}"
        ) from exc
