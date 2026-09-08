"""Authoritative target-trial protocol specification (Requirement 3).

`src/infra/config.py::TargetTrialConfig` is a lightweight, config-loading-time
schema (dict-valued fields) used only to fail fast on unknown top-level keys
before a run starts (`design.md` Section 3.1). This module owns the
authoritative, richly-typed `TargetTrialSpec` that the rest of the pipeline
imports and validates against: it supersedes `TargetTrialConfig` for every
downstream consumer (`decisions.md` is the frozen source of every concrete
value; this module must not re-derive or re-choose any of them).

Every model below uses `extra="forbid"`, so an incomplete or malformed
specification fails loudly with a field-level `pydantic.ValidationError`
(Requirement 3.2) rather than silently accepting an unrecognized key or
dropping a missing one. Every model is also frozen (immutable) — a locked
`TargetTrialSpec` cannot be mutated in place; a protocol amendment requires
constructing a new spec object with an incremented `version` string, per
Requirement 3.3 and `decisions.md` Section 26's amendment-log convention.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Every model in this file shares the same strictness contract: no unknown
# keys, no in-place mutation. Defined once and reused so the contract can't
# drift between models.
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class PopulationSpec(BaseModel):
    """Target population and eligibility (Requirement 3.1, `decisions.md` Section 4)."""

    model_config = _MODEL_CONFIG

    eligibility_criteria: list[str]
    eligibility_expression: str
    excluded_arms: list[str] = Field(default_factory=list)
    source_population: str
    post_hoc_outcome_exclusion_permitted: bool = False


class InterventionSpec(BaseModel):
    """A treatment-arm specification. Reused for both `intervention` and `comparator`
    (Requirement 3.1, `decisions.md` Section 3).
    """

    model_config = _MODEL_CONFIG

    concept_id: str
    description: str
    assignment_time: str
    strategy: str
    treatment_switching_handling: str | None = None


class TimeZeroSpec(BaseModel):
    """Time-zero definition (Requirement 3.1, 3.10, `decisions.md` Section 5)."""

    model_config = _MODEL_CONFIG

    definition: str
    description: str | None = None


class FollowUpSpec(BaseModel):
    """Follow-up window and landmark timepoints (`decisions.md` Section 5)."""

    model_config = _MODEL_CONFIG

    primary: str
    secondary_horizon_weeks: int
    landmark_weeks: list[int] = Field(default_factory=list)
    censoring_note: str | None = None


class OutcomeSpec(BaseModel):
    """Endpoint/outcome definition (Requirement 3.1, 3.12, `decisions.md` Section 6)."""

    model_config = _MODEL_CONFIG

    type: str
    primary_definition: str
    components: list[str] = Field(default_factory=list)
    secondary_endpoints: list[str] = Field(default_factory=list)


class CensoringSpec(BaseModel):
    """Censoring rule (Requirement 3.1, `decisions.md` Section 5)."""

    model_config = _MODEL_CONFIG

    definition: str
    mechanism: str | None = None


class EstimandSpec(BaseModel):
    """Causal contrast / estimand (Requirement 3.1, `decisions.md` Section 7-8)."""

    model_config = _MODEL_CONFIG

    primary: str
    secondary: str | None = None
    external_reference_hr: float | None = None
    external_reference_source: str | None = None
    internal_reference_note: str | None = None


class AnalysisSpec(BaseModel):
    """Analysis method (Requirement 3.1, `decisions.md` Section 11)."""

    model_config = _MODEL_CONFIG

    method: str
    propensity_model: str
    weight_clipping: str
    baseline_estimators: list[str] = Field(default_factory=list)


class CausalDiagram(BaseModel):
    """Structured causal model / causal diagram (Requirement 3.4, `decisions.md` Section 13).

    `nodes` identifies treatment, outcome, baseline confounders, time-varying
    confounders, mediators, colliders, instruments, and censoring variables
    (Requirement 3.5) via the node labels used in `decisions.md` Section 13's
    diagram (X, A, L, Y, S, C). `edges` is a list of `"FROM->TO"` strings.
    """

    model_config = _MODEL_CONFIG

    nodes: list[str]
    edges: list[str] = Field(default_factory=list)
    description: str | None = None


class AssumptionsSpec(BaseModel):
    """The six named causal assumptions required by Requirement 3.6.

    Each field is a free-text statement of how the assumption is operationalized
    or defended for this trial (`decisions.md` Sections 9, 11.3, 12, 13).
    """

    model_config = _MODEL_CONFIG

    consistency: str
    conditional_exchangeability: str
    positivity: str
    no_interference: str
    missing_data: str
    censoring: str


class TargetTrialSpec(BaseModel):
    """The authoritative target-trial protocol specification (Requirement 3.1).

    Requires population, intervention, comparator, time_zero, follow_up,
    outcome, censoring, estimand, analysis, causal_diagram, adjustment_set,
    and assumptions. Frozen (immutable) once constructed: an amendment after
    analysis lock (Requirement 3.3) must be recorded in `decisions.md` Section
    25's amendment log and expressed as a new `TargetTrialSpec` instance with
    an incremented `version`, never as an in-place mutation of this object.
    """

    model_config = _MODEL_CONFIG

    version: str
    population: PopulationSpec
    intervention: InterventionSpec
    comparator: InterventionSpec
    time_zero: TimeZeroSpec
    follow_up: FollowUpSpec
    outcome: OutcomeSpec
    censoring: CensoringSpec
    estimand: EstimandSpec
    analysis: AnalysisSpec
    causal_diagram: CausalDiagram
    adjustment_set: list[str]
    assumptions: AssumptionsSpec
