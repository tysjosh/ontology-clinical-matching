"""Experiment_Variant: one validated description of a model-building arm
(Requirement 6; `design.md` Section 3.5, Section 4.2).

Why this module exists
-------------------------------------------------------------------------

There is no single construction point today for "the model this ablation
or sensitivity arm needs" -- `scripts/run_ablations.py` currently hardcodes
one `CoxBaseline` for every arm. `ExperimentVariant` is the frozen,
validated description of an arm; `build_model` (task 12.1) is the single
factory that turns one into a constructed model. This module is the sole
home for both, and is deliberately torch-free at module scope so that
`evaluation.ablation_matrix` and `evaluation.robustness` -- both of which
must import and run with torch absent (`design.md` Section 2.4) -- can
depend on `ExperimentVariant` without pulling in torch themselves.

Mirror relationship with `infra.config.ExperimentVariantSpec`
-------------------------------------------------------------------------

`infra.config.ExperimentVariantSpec` already exists (task 5) as the
config-side pydantic model carrying this exact field set, so that a model
variant can be *named in a config file* and enter
`ResolvedConfig.canonical_dict()` / the `experiment_id` hash before this
dataclass existed. That model's docstring states plainly: "this model is
the source, the dataclass is the mirror" -- `ExperimentVariant` below is
required to carry the same field names, the same types, and the same
allowed value sets `design.md` Section 4.2 fixes for both. A divergence
between the two is a defect in this dataclass, not in that one.

One difference is deliberate, not a divergence: `ExperimentVariantSpec`
represents `enabled_witnesses` as a sorted tuple (so it serializes to a
canonical JSON form for hashing), while `ExperimentVariant` below uses a
`frozenset`, exactly as `design.md` Section 3.5's code block specifies --
construction-time witness-subset membership checks don't care about
order, and a `frozenset` says so directly.

Validation scope (Requirement 6.2)
-------------------------------------------------------------------------

`__post_init__` enforces two kinds of constraint:

1. **Per-field allowed-set membership.** Every `Literal`-typed field is
   checked against its declared allowed set (the `bool` fields are
   checked by Python's own type system at the boundary this module
   controls: `__post_init__` still verifies they are actually `bool`,
   since a dataclass constructor accepts anything positionally/by
   keyword and type hints are not enforced at runtime).
2. **The two structurally impossible combinations** named in `design.md`
   Section 3.5:
   - `model_family="sequence_gru"` with
     `treatment_representation="explicit_channel"` -- the sequence-GRU
     baseline has no intervention channel at all
     (`models.torch_sequence_gru.TorchSequenceGRU` carries treatment
     inside the state/sequence input, never as a separate channel), so
     "explicit channel" is not a representation that model family can
     express.
   - `model_family="cox"` with `protocol_channel="active"` -- `CoxBaseline`
     (`models.non_neural_baseline.CoxBaseline`) has no protocol encoder or
     any other mechanism to condition on a protocol channel; it is a
     covariate-plus-treatment proportional-hazards fit and nothing else.

Both are raised as one `VariantValidationError` naming every offending
field at once (the same "name every offender together" convention
`cohorts.sealed_reference.LockedEvaluationToken.__post_init__` already
uses for its own field-presence checks), rather than raising on the first
problem found and leaving the rest for a second attempt.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal, Protocol, get_args

#: `design.md` Section 4.2.
ModelFamily = Literal["cox", "sequence_gru", "intervention_aware"]
TreatmentRepresentation = Literal[
    "absent", "ordinary_feature", "explicit_channel", "placebo_channel"
]
ProtocolChannel = Literal["active", "zeroed"]
Witness = Literal["temporal", "causal", "clinical"]

_MODEL_FAMILY_VALUES: frozenset[str] = frozenset(get_args(ModelFamily))
_TREATMENT_REPRESENTATION_VALUES: frozenset[str] = frozenset(get_args(TreatmentRepresentation))
_PROTOCOL_CHANNEL_VALUES: frozenset[str] = frozenset(get_args(ProtocolChannel))
_WITNESS_VALUES: frozenset[str] = frozenset(get_args(Witness))

#: The boolean-typed fields, checked for actual `bool`-ness in
#: `__post_init__` since dataclasses do not enforce type hints at
#: runtime.
_BOOL_FIELDS: tuple[str, ...] = (
    "overlap_gate",
    "uncertainty_gate",
    "clinical_constraints",
    "causal_adjustment",
    "balance_penalty_enabled",
)


class VariantValidationError(ValueError):
    """Raised when an `ExperimentVariant` field is out-of-set, or when the
    variant describes one of the two structurally impossible
    (`model_family`, `treatment_representation`/`protocol_channel`)
    combinations (Requirement 6.2).

    Every offending field or combination is named in the message at once
    -- never only the first one found -- so a caller fixing the variant
    does not have to re-run construction once per problem.
    """


@dataclass(frozen=True)
class ExperimentVariant:
    """One validated, frozen description of a model-building arm.

    Field set, types, and allowed values are fixed by `design.md` Section
    4.2 and mirrored exactly by `infra.config.ExperimentVariantSpec` (see
    module docstring). `build_model` (task 12.1) is the only intended
    consumer that turns an instance of this class into a constructed
    model; `model_affecting_fields` (task 12.1) projects the subset of
    fields that drive the collision detector and the retrain-vs-reuse
    decision.

    Attributes:
        name: non-empty, stable identifier for this variant (e.g. an
            ablation or sensitivity-variant name, or `"complete"`).
        model_family: which architecture this variant builds --
            `"cox"`, `"sequence_gru"`, or `"intervention_aware"`.
        treatment_representation: how treatment enters the model --
            `"absent"` (zeroed/no treatment input), `"ordinary_feature"`
            (treatment inside the state/sequence channel, no separate
            intervention channel), `"explicit_channel"` (the
            intervention-aware model's dedicated channel), or
            `"placebo_channel"` (the frozen random placebo vector in
            place of the learned intervention channel).
        protocol_channel: whether the protocol channel carries its real
            features (`"active"`) or an all-zero vector of the same width
            (`"zeroed"`).
        overlap_gate: whether the overlap/support gate is active for this
            variant.
        uncertainty_gate: whether the uncertainty gate is active for this
            variant.
        clinical_constraints: whether clinical-constraint checking is
            active for this variant.
        enabled_witnesses: the subset of `{"temporal", "causal",
            "clinical"}` active for this variant. May be empty (every
            witness removed) but every element must be one of the three
            named witnesses.
        causal_adjustment: whether causal adjustment (as opposed to a
            purely factual/predictive objective) is active for this
            variant.
        balance_penalty_enabled: whether the representation-balance (MMD)
            penalty term is included in the training objective for this
            variant.

    Raises:
        VariantValidationError: if any field holds a value outside its
            declared allowed set, if `name` is empty or not a string, if
            any `bool`-typed field does not hold an actual `bool`, or if
            the variant describes either structurally impossible
            combination.
    """

    name: str
    model_family: ModelFamily
    treatment_representation: TreatmentRepresentation
    protocol_channel: ProtocolChannel
    overlap_gate: bool
    uncertainty_gate: bool
    clinical_constraints: bool
    enabled_witnesses: frozenset[Witness]
    causal_adjustment: bool
    balance_penalty_enabled: bool

    def __post_init__(self) -> None:
        problems: list[str] = []

        if not isinstance(self.name, str) or not self.name.strip():
            problems.append(f"name must be a non-empty string, got {self.name!r}")

        if self.model_family not in _MODEL_FAMILY_VALUES:
            problems.append(
                f"model_family={self.model_family!r} is not one of "
                f"{sorted(_MODEL_FAMILY_VALUES)}"
            )

        if self.treatment_representation not in _TREATMENT_REPRESENTATION_VALUES:
            problems.append(
                f"treatment_representation={self.treatment_representation!r} is not "
                f"one of {sorted(_TREATMENT_REPRESENTATION_VALUES)}"
            )

        if self.protocol_channel not in _PROTOCOL_CHANNEL_VALUES:
            problems.append(
                f"protocol_channel={self.protocol_channel!r} is not one of "
                f"{sorted(_PROTOCOL_CHANNEL_VALUES)}"
            )

        for field_name in _BOOL_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                problems.append(
                    f"{field_name} must be a bool, got {type(value).__name__} ({value!r})"
                )

        if not isinstance(self.enabled_witnesses, frozenset):
            problems.append(
                "enabled_witnesses must be a frozenset, got "
                f"{type(self.enabled_witnesses).__name__}"
            )
        else:
            unknown_witnesses = self.enabled_witnesses - _WITNESS_VALUES
            if unknown_witnesses:
                problems.append(
                    f"enabled_witnesses contains unknown witness(es) "
                    f"{sorted(unknown_witnesses)}; allowed values are "
                    f"{sorted(_WITNESS_VALUES)}"
                )

        # Structurally impossible combinations (design.md Section 3.5).
        if (
            self.model_family == "sequence_gru"
            and self.treatment_representation == "explicit_channel"
        ):
            problems.append(
                "model_family='sequence_gru' cannot be combined with "
                "treatment_representation='explicit_channel': the sequence-GRU "
                "baseline has no separate intervention channel; treatment is "
                "carried inside its state/sequence input only"
            )

        if self.model_family == "cox" and self.protocol_channel == "active":
            problems.append(
                "model_family='cox' cannot be combined with "
                "protocol_channel='active': the Cox baseline has no protocol "
                "encoder or any mechanism to condition on a protocol channel"
            )

        if problems:
            joined = "; ".join(problems)
            raise VariantValidationError(
                f"invalid ExperimentVariant (name={self.name!r}): {joined}"
            )


#: The complete model: every gate, every witness, the full intervention
#: channel, and the active protocol channel (`design.md` Section 4.2).
#: Every ablation and sensitivity variant is expressed as a delta from
#: this constant.
COMPLETE: ExperimentVariant = ExperimentVariant(
    name="complete",
    model_family="intervention_aware",
    treatment_representation="explicit_channel",
    protocol_channel="active",
    overlap_gate=True,
    uncertainty_gate=True,
    clinical_constraints=True,
    enabled_witnesses=frozenset({"temporal", "causal", "clinical"}),
    causal_adjustment=True,
    balance_penalty_enabled=True,
)


# ---------------------------------------------------------------------------
# model_affecting_fields (task 12.1; Requirement 6.4, 7.10)
# ---------------------------------------------------------------------------

#: The subset of `ExperimentVariant` fields that change what model gets
#: *built* -- its architecture or the content of its inputs -- as opposed
#: to fields that only change post-hoc gating/witness behavior applied to
#: a model's existing predictions.
#:
#: This is the explicit projection `design.md` Section 3.5 names:
#: "the collision detector and the retrain-vs-reuse decision" both read
#: it. A variant pair differing only outside this set (e.g.
#: `overlap_gate` or `enabled_witnesses`) describes the *same* trained
#: model evaluated under different gates/witnesses, so Requirement
#: 7.6-7.8's reuse runs are correct to skip retraining for them. A pair
#: differing *inside* this set must be retrained (Requirement 7.10).
MODEL_AFFECTING_FIELD_NAMES: tuple[str, ...] = (
    "model_family",
    "treatment_representation",
    "protocol_channel",
    "balance_penalty_enabled",
    "causal_adjustment",
)


def model_affecting_fields(variant: ExperimentVariant) -> dict[str, object]:
    """Project `variant` onto the fields that determine what gets *built*.

    Args:
        variant: the variant to project.

    Returns:
        A dict with exactly the keys in `MODEL_AFFECTING_FIELD_NAMES`,
        mapped to `variant`'s values for those fields. Two variants with
        equal `model_affecting_fields(...)` results describe the same
        model under different gates/witnesses; two variants with
        differing results describe models that must differ in
        architecture, input content, or both (Requirement 6.4).
    """
    return {name: getattr(variant, name) for name in MODEL_AFFECTING_FIELD_NAMES}


# ---------------------------------------------------------------------------
# build_model (task 12.1; Requirement 6.3, 6.5)
# ---------------------------------------------------------------------------


class MissingDependencyError(RuntimeError):
    """Raised when `build_model` needs torch and torch is not installed.

    Named identically to, but a distinct class from,
    `causal.propensity.MissingDependencyError` -- that one names `xgboost`
    for `BoostedPropensityModel`; this one names the `neural` extra. Both
    follow the same convention: raised at the call site that actually
    needs the dependency, not at module import, so a torch-free caller can
    import `models.variants` freely and only pays for the missing
    dependency if it tries to build a neural model (`design.md` Section
    2.4; Requirement 5.1-5.2 of the parent spec's torch-isolation design).
    """

    def __init__(self) -> None:
        super().__init__(
            "build_model() requires torch for model_family="
            "'sequence_gru'/'intervention_aware', which is not installed. "
            "Install the project's 'neural' extra (torch==2.4.1) to build "
            "neural model variants; model_family='cox' does not require it."
        )


def build_model(variant: ExperimentVariant, dimensions: object, seed: int) -> object:
    """The single construction entry point for every model-building arm.

    Dispatches on `variant.model_family`:

    - `"cox"` -> `models.non_neural_baseline.CoxBaseline()`, unfitted
      (`.fit(patients)` is the caller's next step; this factory only
      constructs).
    - `"sequence_gru"` -> `models.torch_sequence_gru.TorchSequenceGRU`,
      constructed against `dimensions`.
    - `"intervention_aware"` -> `models.torch_intervention_aware.TorchInterventionAwareModel`,
      with `intervention_mode` / `protocol_mode` derived from
      `variant.treatment_representation` / `variant.protocol_channel`, and
      `placebo_vector` supplied (drawn from `seed`) exactly when
      `treatment_representation == "placebo_channel"`.

    Every width the constructed model reads comes from `dimensions`
    (Requirement 6.3's "no hard-coded value") -- this function never
    passes a literal width to either torch constructor.

    Torch is imported inside this function's body, not at module scope
    (`design.md` Section 2.4), so `from models.variants import
    ExperimentVariant` and `model_affecting_fields` succeed without torch
    installed, and only a `"sequence_gru"`/`"intervention_aware"` call
    into this function requires it.

    Parameter initialization is seed-determined: `seed_everything(seed)`
    (`models.torch_training`) runs before construction for the two torch
    model families, so two calls with the same `variant` and `seed`
    produce bitwise-identical initial parameters (Requirement 6.5).
    `"cox"` has no randomly initialized parameters, so `seed` is accepted
    but unused for that family -- `CoxBaseline`'s fit is a deterministic
    optimization from a fixed starting point, not a randomly initialized
    model.

    Args:
        variant: the validated `ExperimentVariant` describing the model to
            build.
        dimensions: a `models.dataset.DatasetDimensions` instance (for the
            two torch families) or any value, ignored, for `"cox"`.
        seed: the seed determining parameter initialization for the torch
            families.

    Returns:
        For `"cox"`: an unfitted `CoxBaseline`. For `"sequence_gru"`: a
        constructed `TorchSequenceGRU`. For `"intervention_aware"`: a
        constructed `TorchInterventionAwareModel`.

    Raises:
        MissingDependencyError: if `variant.model_family` is
            `"sequence_gru"` or `"intervention_aware"` and torch is not
            installed.
        ValueError: propagated from the underlying torch constructor if
            `dimensions` does not describe a `DatasetDimensions` the
            constructor accepts (e.g. a width mismatch).
    """
    if variant.model_family == "cox":
        from models.non_neural_baseline import CoxBaseline

        return CoxBaseline()

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError() from exc

    from models.torch_training import seed_everything

    if variant.model_family == "sequence_gru":
        from models.torch_sequence_gru import TorchSequenceGRU

        seed_everything(seed)
        return TorchSequenceGRU(dimensions)

    if variant.model_family == "intervention_aware":
        from infra.seeding import derive_seed
        from models.intervention_aware import INTERVENTION_INPUT_DIM
        from models.torch_intervention_aware import (
            InterventionMode,
            ProtocolMode,
            TorchInterventionAwareModel,
        )

        if variant.treatment_representation == "ordinary_feature":
            raise VariantValidationError(
                "treatment_representation='ordinary_feature' has no meaning for "
                "model_family='intervention_aware': that model's intervention "
                "channel is only ever LEARNED, ZEROED, or PLACEBO (design.md "
                "Section 3.3). 'ordinary_feature' describes "
                "model_family='sequence_gru', which carries treatment inside "
                "its state/sequence input instead of a separate channel."
            )
        intervention_mode = {
            "explicit_channel": InterventionMode.LEARNED,
            "absent": InterventionMode.ZEROED,
            "placebo_channel": InterventionMode.PLACEBO,
        }[variant.treatment_representation]
        protocol_mode = (
            ProtocolMode.ACTIVE if variant.protocol_channel == "active" else ProtocolMode.ZEROED
        )

        placebo_vector: tuple[float, ...] | None = None
        if intervention_mode is InterventionMode.PLACEBO:
            import random

            placebo_seed = derive_seed(seed, "placebo_intervention_channel")
            rng = random.Random(placebo_seed)
            placebo_vector = tuple(
                rng.uniform(-0.5, 0.5) for _ in range(INTERVENTION_INPUT_DIM)
            )

        seed_everything(seed)
        return TorchInterventionAwareModel(
            dimensions,
            intervention_mode=intervention_mode,
            protocol_mode=protocol_mode,
            placebo_vector=placebo_vector,
        )

    # Unreachable: ExperimentVariant.__post_init__ already restricts
    # model_family to the three literals handled above.
    raise AssertionError(  # pragma: no cover - defensive
        f"unhandled model_family {variant.model_family!r}; this is a bug in "
        "build_model, not a validation failure (ExperimentVariant already "
        "restricts model_family to 'cox'/'sequence_gru'/'intervention_aware')"
    )


# ---------------------------------------------------------------------------
# variant_for_ablation / variant_for_sensitivity (task 13; Requirement 7.1,
# 6.1, tension 2 of design.md Section 2.6)
# ---------------------------------------------------------------------------


def _delta(**overrides: object) -> ExperimentVariant:
    """`COMPLETE` with `overrides` applied -- every ablation variant below
    is expressed this way, matching `design.md` Section 3.6's "variant
    delta from the complete model" table."""
    kwargs = {f.name: getattr(COMPLETE, f.name) for f in dataclasses.fields(ExperimentVariant)}
    kwargs.update(overrides)
    return ExperimentVariant(**kwargs)


class AblationRunLike(Protocol):
    """The two `AblationRun` fields `variant_for_ablation` needs.

    A structural (`Protocol`) type rather than an import of
    `evaluation.ablation_matrix.AblationRun`: `evaluation.ablation_matrix`
    imports `ExperimentVariant` from this module (tension 2), so this
    module importing `AblationRun` back from there would be a cycle.
    `evaluation.ablation_matrix.AblationRun` satisfies this protocol
    structurally -- no explicit registration needed.
    """

    source_entry: str
    sub_run_key: str | None


#: Keyed by `(source_entry, sub_run_key)`. `sub_run_key` is `None` for the
#: nine single-run ablation entries and one of `{"temporal", "causal",
#: "clinical"}` for the three `witness_removed` sub-runs -- the only entry
#: whose variant depends on more than its own name (`design.md` Section
#: 3.6's execution-mode table).
#:
#: `explicit_intervention_channel` maps to `COMPLETE` itself: per that
#: table, this ablation *is* the complete model (Requirement 7.14 fixes it
#: as the width-3-intervention-channel arm the other ablations are
#: contrasted against), so its own retrained run is a from-scratch
#: reinitialization of the same architecture, not a distinct variant.
_ABLATION_VARIANTS: dict[tuple[str, str | None], ExperimentVariant] = {
    ("no_treatment_input", None): _delta(
        name="no_treatment_input", treatment_representation="absent"
    ),
    ("treatment_as_ordinary_feature", None): _delta(
        name="treatment_as_ordinary_feature",
        model_family="sequence_gru",
        treatment_representation="ordinary_feature",
        # TorchSequenceGRU has no protocol encoder at all (design.md
        # Section 3.2); "zeroed" is the honest value for a model that
        # never conditions on a protocol channel, not merely an ablated
        # one.
        protocol_channel="zeroed",
    ),
    ("explicit_intervention_channel", None): _delta(name="explicit_intervention_channel"),
    ("intervention_channel_without_protocol", None): _delta(
        name="intervention_channel_without_protocol", protocol_channel="zeroed"
    ),
    ("no_overlap_gate", None): _delta(name="no_overlap_gate", overlap_gate=False),
    ("no_uncertainty_gate", None): _delta(name="no_uncertainty_gate", uncertainty_gate=False),
    ("no_clinical_constraints", None): _delta(
        name="no_clinical_constraints", clinical_constraints=False
    ),
    ("witness_removed", "temporal"): _delta(
        name="witness_removed::temporal",
        enabled_witnesses=frozenset({"causal", "clinical"}),
    ),
    ("witness_removed", "causal"): _delta(
        name="witness_removed::causal",
        enabled_witnesses=frozenset({"temporal", "clinical"}),
    ),
    ("witness_removed", "clinical"): _delta(
        name="witness_removed::clinical",
        enabled_witnesses=frozenset({"temporal", "causal"}),
    ),
    ("predictive_accuracy_only", None): _delta(
        name="predictive_accuracy_only",
        balance_penalty_enabled=False,
        causal_adjustment=False,
    ),
    ("placebo_intervention_channel", None): _delta(
        name="placebo_intervention_channel", treatment_representation="placebo_channel"
    ),
}

#: The ten frozen ablation entry names this mapping covers, independent of
#: `evaluation.ablation_matrix.ABLATIONS` (which this module must not
#: import -- see `AblationRunLike`). A frozen test in
#: `tests/unit/test_ablation_matrix.py` or `tests/unit/test_variants.py`
#: asserts this set equals `{a.name for a in ABLATIONS}`.
ABLATION_ENTRY_NAMES: frozenset[str] = frozenset(key[0] for key in _ABLATION_VARIANTS)


def variant_for_ablation(run: AblationRunLike) -> ExperimentVariant:
    """The `ExperimentVariant` for one expanded ablation run.

    A total, explicit mapping keyed by `(run.source_entry,
    run.sub_run_key)` -- never a parse of `config_diff`, which stays the
    human-readable record (`design.md` Section 2.6 tension 2).

    Args:
        run: anything exposing `source_entry: str` and
            `sub_run_key: str | None`, in practice an
            `evaluation.ablation_matrix.AblationRun`.

    Returns:
        The variant that run builds.

    Raises:
        KeyError: if `(run.source_entry, run.sub_run_key)` is not one of
            the twelve frozen keys -- an ablation entry or sub-run key
            this mapping does not (yet) cover.
    """
    key = (run.source_entry, run.sub_run_key)
    try:
        return _ABLATION_VARIANTS[key]
    except KeyError:
        raise KeyError(
            f"no ExperimentVariant mapped for ablation (source_entry={run.source_entry!r}, "
            f"sub_run_key={run.sub_run_key!r}); known keys are "
            f"{sorted(_ABLATION_VARIANTS.keys())!r}"
        ) from None


#: Every one of `decisions.md` Section 20's eight frozen sensitivity
#: variants maps to `COMPLETE`: each is a *data/analysis*-level config
#: transform (propensity model choice, weight clipping, support
#: thresholds, missingness stress, landmark masking, comparator
#: orientation, endpoint horizon, selection severity) evaluated through
#: the complete model and the complete pipeline (`design.md` Section 3.7
#: -- "all eight through the same pipeline stages rather than bespoke
#: code paths"), never a change to model architecture, treatment
#: representation, or protocol conditioning. None of the eight appears in
#: `MODEL_AFFECTING_FIELD_NAMES`'s domain, so mapping every name to the
#: same `ExperimentVariant` is not a loss of information -- the actual
#: per-variant distinction lives in each `SensitivityVariantConfig.config_diff`
#: (`evaluation.robustness.SENSITIVITY_VARIANTS`), not in the model built.
_SENSITIVITY_VARIANT_NAMES: tuple[str, ...] = (
    "boosted_propensity_model",
    "alternative_weight_clipping",
    "alternative_support_thresholds",
    "increased_missingness_stress",
    "week_20_removed",
    "treatment_comparator_reversal",
    "shifted_endpoint_horizon",
    "stronger_selection_severity",
)

_SENSITIVITY_VARIANTS: dict[str, ExperimentVariant] = dict.fromkeys(
    _SENSITIVITY_VARIANT_NAMES, COMPLETE
)


def variant_for_sensitivity(name: str) -> ExperimentVariant:
    """The `ExperimentVariant` for one of the eight frozen sensitivity
    variants (`decisions.md` Section 20).

    Every sensitivity variant maps to `COMPLETE` -- see the module-level
    `_SENSITIVITY_VARIANTS` comment for why that is correct rather than a
    placeholder.

    Args:
        name: one of the eight frozen sensitivity-variant names
            (`evaluation.robustness.SENSITIVITY_VARIANTS`' `name` fields).

    Returns:
        The variant that sensitivity analysis builds (always `COMPLETE`).

    Raises:
        KeyError: if `name` is not one of the eight frozen names.
    """
    try:
        return _SENSITIVITY_VARIANTS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not one of the eight frozen sensitivity-variant names: "
            f"{sorted(_SENSITIVITY_VARIANTS)!r}"
        ) from None
