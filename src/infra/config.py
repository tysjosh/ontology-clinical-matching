"""Configuration loading and validation.

Every run is driven by three composed YAML sources — the dataset contract, the
target-trial specification, and the experiment config (`design.md` Section 2.2).
`load_config` loads each independently, validates it against a Pydantic schema,
and composes the three into a single `ResolvedConfig`. Unknown keys in any of the
three sources are never silently ignored: they raise a `ConfigError`.

This strictness matters beyond hygiene: `experiment_id` is defined as a hash of
the fully-resolved config (`design.md` Section 2.2), so a typo'd or stray key that
was silently dropped would make two logically-different configs hash identically,
or would make a config's resolved form depend on which keys happened to be
recognized by whichever version of the code loaded it.

The schemas below are intentionally scoped to what config *loading* needs to
validate at this stage (Requirement 1.1). They are not the authoritative,
fully-detailed domain schemas — `src/data/contract.py` will own the authoritative
`DatasetContract` and `src/protocols/schema.py` will own the authoritative
`TargetTrialSpec`, each with full field-level semantic validation (Requirements
3.1-3.2, 5.1-5.2). Those modules are expected to supersede the corresponding
schemas here once implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or fails schema validation."""


class PopulationInfoConfig(BaseModel):
    """Packaged population size/description, matching
    `src.data.contract.PopulationInfo`'s shape exactly (this schema and that
    authoritative one must not diverge in field names)."""

    model_config = ConfigDict(extra="forbid")

    size: int
    description: str


class DatasetContractConfig(BaseModel):
    """Dataset-contract fields required at config-load time.

    Mirrors the frozen dataset contract in `decisions.md` Section 2 and
    Requirement 5.1 (dataset version, population, treatment, comparator,
    endpoint, study period, available modalities, reference-effect source,
    data-integrity checksum) -- and, field-for-field, the authoritative
    `src.data.contract.DatasetContract` this config-loading-time schema is
    superseded by (see module docstring). Both `configs/datasets/actg175_v3_2.yaml`
    (the real, frozen contract) and `tests/fixtures/config/valid_dataset.yaml`
    (the lightweight test fixture) validate against this same shape.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    population: PopulationInfoConfig
    treatment: str
    comparator: str
    endpoint: str
    study_period: str
    available_modalities: list[str]
    reference_effect_source: str
    checksum_sha256: str


class TargetTrialConfig(BaseModel):
    """Target-trial-spec fields required at config-load time.

    Mirrors the required top-level fields from Requirement 3.1 and `design.md`
    Section 3.2 (`TargetTrialSpec`): version, population, intervention,
    comparator, time_zero, follow_up, outcome, censoring, estimand, analysis,
    plus the causal-specification fields from Requirement 3.4-3.6.
    `version` matches `protocols.schema.TargetTrialSpec.version` (Requirement
    3.3's protocol-amendment versioning) -- optional here (defaulting to
    `"unversioned"`) so pre-existing lightweight fixtures that predate
    versioning need not all be rewritten, but present and required on every
    real, authoritative target-trial config such as
    `configs/target_trials/main.yaml`.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = "unversioned"
    population: dict[str, Any]
    intervention: dict[str, Any]
    comparator: dict[str, Any]
    time_zero: dict[str, Any]
    follow_up: dict[str, Any]
    outcome: dict[str, Any]
    censoring: dict[str, Any]
    estimand: dict[str, Any]
    analysis: dict[str, Any]
    causal_diagram: dict[str, Any]
    adjustment_set: list[str]
    assumptions: dict[str, Any]


class ExperimentConfig(BaseModel):
    """Experiment-level configuration: identity, seed, and run-policy fields."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    seed: int
    exploratory: bool = False
    witness_policy: str = "ALL_REQUIRED"
    tags: list[str] = []

    @model_validator(mode="after")
    def _witness_policy_is_valid_and_gated(self) -> ExperimentConfig:
        allowed = {"ALL_REQUIRED", "TWO_OF_THREE", "RISK_WEIGHTED"}
        if self.witness_policy not in allowed:
            raise ValueError(
                f"witness_policy must be one of {sorted(allowed)}, got {self.witness_policy!r}"
            )
        # decisions.md Section 16 / Requirement 11.9: ALL_REQUIRED is mandatory
        # for any run that will produce a paper-reported (non-exploratory) number.
        if self.witness_policy != "ALL_REQUIRED" and not self.exploratory:
            raise ValueError(
                "witness_policy other than 'ALL_REQUIRED' is only permitted on "
                "configs with exploratory=true"
            )
        return self


class ResolvedConfig(BaseModel):
    """The fully composed, schema-validated configuration for one run."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetContractConfig
    target_trial: TargetTrialConfig
    experiment: ExperimentConfig

    def canonical_dict(self) -> dict[str, Any]:
        """Deterministic dict representation, suitable for canonical-JSON hashing.

        Used by `src/infra/registry.py` to compute
        `experiment_id = sha256(canonical_json(resolved_config))[:16]`.
        """
        return self.model_dump(mode="json")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Top-level content of {path} must be a mapping, got {type(data).__name__}"
        )
    return data


def load_config(
    dataset_path: str | Path,
    target_trial_path: str | Path,
    experiment_path: str | Path,
) -> ResolvedConfig:
    """Load and merge the dataset contract, target-trial spec, and experiment config.

    Each source is parsed as YAML and validated independently against its own
    Pydantic schema before being composed into a `ResolvedConfig`. Any key not
    recognized by the corresponding schema raises a `ConfigError` — unknown keys
    are never dropped silently.

    Raises:
        ConfigError: if any file is missing, is not valid YAML, is not a mapping
            at the top level, or fails schema validation (including unknown keys).
    """
    dataset_raw = _load_yaml(Path(dataset_path))
    target_trial_raw = _load_yaml(Path(target_trial_path))
    experiment_raw = _load_yaml(Path(experiment_path))

    try:
        dataset = DatasetContractConfig.model_validate(dataset_raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid dataset contract config ({dataset_path}):\n{exc}") from exc

    try:
        target_trial = TargetTrialConfig.model_validate(target_trial_raw)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid target-trial spec config ({target_trial_path}):\n{exc}"
        ) from exc

    try:
        experiment = ExperimentConfig.model_validate(experiment_raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid experiment config ({experiment_path}):\n{exc}") from exc

    try:
        return ResolvedConfig(dataset=dataset, target_trial=target_trial, experiment=experiment)
    except ValidationError as exc:  # pragma: no cover - defensive; inputs are already validated
        raise ConfigError(f"Failed to compose resolved config: {exc}") from exc
