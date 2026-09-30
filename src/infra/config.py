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
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator


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


class VariantOverrides(BaseModel):
    """The analytic knobs the eight frozen sensitivity variants turn.

    Every field defaults to the primary/frozen setting of
    `.kiro/specs/causal-witness-pwm/decisions.md`, so a config that names
    none of them describes the primary analysis exactly. The alternatives
    are closed `Literal` sets rather than free strings: a sensitivity
    variant that misspells its own knob must fail loudly at config-load
    time, not silently execute the primary analysis under a variant's name.

    `extra="forbid"` and `frozen=True` are both load-bearing. `forbid`
    keeps a typo'd override from being absorbed as an untyped extra;
    `frozen` keeps a knob from being mutated after the config was hashed
    into an `experiment_id`, which would make the identifier stop
    describing the run.

    Because these fields have defaults and `ResolvedConfig.canonical_dict()`
    serializes defaults, every config -- textually unchanged -- now hashes
    with `variant: {all-primary values}` present. That one-time
    re-identification of every prior `experiment_id` is deliberate and is
    recorded as a `decisions.md` Section 26 amendment (neural-experiment-pipeline
    Requirement 9.9, 18.1; `design.md` Section 2.6 tension 3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    propensity_model: Literal["logistic", "boosted"] = "logistic"
    weight_clipping: Literal["primary_1_99", "narrow_0p5_99p5", "wide_5_95"] = "primary_1_99"
    support_thresholds: Literal["primary_0.10_0.90", "strict_0.20_0.80"] = "primary_0.10_0.90"
    missingness_stress: Literal["none", "increased"] = "none"
    landmark_masking: Literal["none", "week_20_removed"] = "none"
    comparator_orientation: Literal["primary", "reversed"] = "primary"
    endpoint_horizon_weeks: float = 96.0
    selection_severity: Literal["primary", "stronger"] = "primary"


class ExperimentVariantSpec(BaseModel):
    """Config-side mirror of `models.variants.ExperimentVariant`.

    Direction of the mirror: **this model is the source, the dataclass is
    the mirror.** `ExperimentVariantSpec` exists here so a model variant can
    be *named in a config file* and therefore enter
    `ResolvedConfig.canonical_dict()` and the `experiment_id` hash
    (neural-experiment-pipeline Requirement 6.6, 9.9). The frozen
    `ExperimentVariant` dataclass in `src/models/variants.py` does not exist
    yet -- it arrives with the model factory -- and it is required to carry
    these field names, these types, and these allowed value sets;
    `design.md` Section 4.2 fixes them for both.
    Whoever writes `ExperimentVariant` should mirror this declaration, not
    re-choose it; a divergence between the two is a defect in the dataclass.

    Scope note: field-level validation (allowed value sets, non-empty name,
    witness subset) lives here. The two *structurally impossible*
    combinations of `design.md` Section 3.5 -- `sequence_gru` with
    `explicit_channel`, and `cox` with an active protocol channel -- are
    cross-field constraints enforced by `ExperimentVariant.__post_init__`,
    which is the single construction point the model factory goes through.

    `enabled_witnesses` is a sorted, de-duplicated tuple rather than the
    dataclass's `frozenset`, because it has to serialize deterministically:
    `model_dump(mode="json")` of a `frozenset` has no defined element
    order, which would make one config hash to different `experiment_id`
    values across processes.
    """

    # `protected_namespaces=()` only silences pydantic's warning about the
    # `model_`-prefixed field names; `model_family` is the field name
    # `design.md` Section 4.2 fixes and is not renamed to accommodate a
    # library convention.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    name: str
    model_family: Literal["cox", "sequence_gru", "intervention_aware"]
    treatment_representation: Literal[
        "absent", "ordinary_feature", "explicit_channel", "placebo_channel"
    ]
    protocol_channel: Literal["active", "zeroed"]
    overlap_gate: bool
    uncertainty_gate: bool
    clinical_constraints: bool
    enabled_witnesses: tuple[Literal["temporal", "causal", "clinical"], ...]
    causal_adjustment: bool
    balance_penalty_enabled: bool

    @field_validator("name")
    @classmethod
    def _name_is_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must be a non-empty string")
        return value

    @field_validator("enabled_witnesses")
    @classmethod
    def _witnesses_are_sorted_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize to sorted-unique so the serialized form is canonical.

        A config listing `["causal", "temporal"]` and one listing
        `["temporal", "causal", "temporal"]` describe the same variant and
        must hash to the same `experiment_id`.
        """
        return tuple(sorted(set(value)))


class ExperimentConfig(BaseModel):
    """Experiment-level configuration: identity, seed, and run-policy fields."""

    # See `ExperimentVariantSpec` for why `protected_namespaces` is cleared:
    # `model_variant` is the field name `design.md` Section 2.6 fixes.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    experiment_name: str
    seed: int
    exploratory: bool = False
    witness_policy: str = "ALL_REQUIRED"
    tags: list[str] = []
    #: Analytic knobs the frozen sensitivity variants turn; all-primary by
    #: default (`design.md` Section 2.6 tension 3).
    variant: VariantOverrides = VariantOverrides()
    #: The model variant this run builds, when one is named. `None` means
    #: no neural model variant was selected -- which is itself a hashed
    #: fact about the run, not an absence.
    model_variant: ExperimentVariantSpec | None = None
    #: Bootstrap replicate count for both negative controls. This was the
    #: only input covered by the second, now-deleted identity scheme in
    #: `scripts/run_causal_analysis.py` that did not already live inside
    #: `ResolvedConfig`; absorbed here so
    #: `infra.registry.compute_experiment_id` is the single identity
    #: function (`design.md` Section 2.6 tension 6).
    negative_control_n_bootstrap: int = 1000

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

        Defaults are serialized, not omitted. That is what makes two runs
        differing only in a `variant` override, a `model_variant`, or
        `negative_control_n_bootstrap` get different `experiment_id`
        values; it is also why adding those fields re-identified every
        previously computed `experiment_id` (a one-time, documented break,
        recorded as a `decisions.md` Section 26 amendment).
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
