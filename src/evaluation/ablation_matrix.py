"""The ten frozen primary ablations (`decisions.md` Section 17; Requirement 12).

`decisions.md` Section 17 freezes the primary ablation list to exactly ten
named entries, in a fixed order, and forbids adding, removing, or reordering
them without a documented amendment (Section 25). This module is the single
source of truth for that list: `ABLATIONS` below is consumed by
`scripts/run_ablations.py` (task 12.1) to drive every ablation run through
the same pipeline stages, splits, seeds, and evaluation procedures
(Requirement 12.9).

Each entry's `config_diff` is a small, illustrative config-override
dictionary expressing the *intent* of the ablation relative to the base
experiment config. It intentionally reuses field names already present in
`src/infra/registry.RegistryEntry` (`intervention_representation`,
`protocol_constraints`, `overlap_gate`, `uncertainty_gate`) where an ablation
maps directly onto one of those fields, so the diff vocabulary stays
consistent with the rest of the codebase. These diffs are structural
placeholders only -- they do not yet wire into `infra.config.ResolvedConfig`
or `ExperimentConfig`'s schema (that wiring is scoped to task 12.1's runner,
not this module).

Ablation 8 ("Each witness removed independently", Requirement 12.6) is a
single frozen list entry here -- `len(ABLATIONS) == 10` must match the ten
enumerated items in `decisions.md` Section 17, not the three per-witness
runs it expands to in practice. Its `config_diff` records the three
witnesses it is applied to independently (`applies_to`); the per-witness
expansion into three separate runs is `scripts/run_ablations.py`'s
responsibility (task 12.1), not this module's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from models.variants import ExperimentVariant, variant_for_ablation


@dataclass(frozen=True)
class AblationConfig:
    """One frozen primary ablation.

    Attributes:
        name: short, stable, unique identifier for this ablation. Used as a
            directory/result key (e.g. by `scripts/run_ablations.py` under
            `results/<experiment_id>/ablations/<name>/`), so renaming it is a
            breaking change -- covered by the "stable names" unit test.
        description: the human-readable text for this ablation, verbatim
            from `decisions.md` Section 17.
        config_diff: a config-override dict expressing what changes
            relative to the base experiment config. Structural/illustrative
            only -- see module docstring.
    """

    name: str
    description: str
    config_diff: dict[str, Any] = field(default_factory=dict)


ABLATIONS: list[AblationConfig] = [
    AblationConfig(
        name="no_treatment_input",
        description="No treatment input.",
        config_diff={"intervention_representation": "none"},
    ),
    AblationConfig(
        name="treatment_as_ordinary_feature",
        description="Treatment as an ordinary feature.",
        config_diff={"intervention_representation": "ordinary_feature"},
    ),
    AblationConfig(
        name="explicit_intervention_channel",
        description="Explicit intervention channel.",
        config_diff={"intervention_representation": "explicit_channel"},
    ),
    AblationConfig(
        name="intervention_channel_without_protocol",
        description="Intervention channel without protocol representation.",
        config_diff={
            "intervention_representation": "explicit_channel",
            "protocol_constraints": "none",
        },
    ),
    AblationConfig(
        name="no_overlap_gate",
        description="Complete model without overlap gate.",
        config_diff={"overlap_gate": False},
    ),
    AblationConfig(
        name="no_uncertainty_gate",
        description="Complete model without uncertainty gate.",
        config_diff={"uncertainty_gate": False},
    ),
    AblationConfig(
        name="no_clinical_constraints",
        description="Complete model without clinical constraints.",
        config_diff={"clinical_constraints": False},
    ),
    AblationConfig(
        name="witness_removed",
        description="Each witness removed independently.",
        config_diff={
            "witness_removed": "each_independently",
            "applies_to": ["temporal", "causal", "clinical"],
        },
    ),
    AblationConfig(
        name="predictive_accuracy_only",
        description="Predictive accuracy alone (no causal adjustment).",
        config_diff={"causal_adjustment": False},
    ),
    AblationConfig(
        name="placebo_intervention_channel",
        description=(
            "Random capacity-matched intervention embeddings "
            "(placebo channel, Section 14.3)."
        ),
        config_diff={"intervention_representation": "placebo_channel"},
    ),
]


# ---------------------------------------------------------------------------
# AblationRun / expand_ablations (task 13; design.md Section 2.6 tension 1;
# Requirement 7.1, 7.8)
# ---------------------------------------------------------------------------

#: `design.md` Section 2.6 tension 1's execution-mode table. Six of the ten
#: entries retrain from initialization (the model-affecting arms); the
#: other four reuse the complete model's saved predictions and only bypass
#: a gate, constraint, or witness (Requirement 7.6, 7.7, 7.8). This is a
#: separate frozen mapping, not read off `ExperimentVariant`, because
#: execution mode is a *runner* decision about whether to retrain, not a
#: property `models.variants.model_affecting_fields` happens to encode for
#: every entry equally (it is consistent with that projection for all ten
#: entries here, but the two are deliberately kept as independently
#: statable facts -- see the frozen test asserting they agree).
_EXECUTION_MODE_BY_ENTRY: dict[str, Literal["retrain", "reuse_predictions"]] = {
    "no_treatment_input": "retrain",
    "treatment_as_ordinary_feature": "retrain",
    "explicit_intervention_channel": "retrain",
    "intervention_channel_without_protocol": "retrain",
    "no_overlap_gate": "reuse_predictions",
    "no_uncertainty_gate": "reuse_predictions",
    "no_clinical_constraints": "reuse_predictions",
    "witness_removed": "reuse_predictions",
    "predictive_accuracy_only": "retrain",
    "placebo_intervention_channel": "retrain",
}


@dataclass(frozen=True)
class AblationRun:
    """One of the twelve runs the ten frozen `ABLATIONS` entries expand to.

    Attributes:
        run_id: unique id for this run. Equal to `source_entry` for the
            nine single-run entries; `f"{source_entry}::{sub_run_key}"`
            for the three `witness_removed` sub-runs (`design.md` Section
            2.6 tension 1).
        source_entry: the frozen `AblationConfig.name` this run expands
            from. Exactly one of `{a.name for a in ABLATIONS}`.
        sub_run_key: `None` for every entry except `witness_removed`,
            where it is one of `{"temporal", "causal", "clinical"}` -- the
            witness this particular run removes.
        variant: the `ExperimentVariant` this run builds, from
            `models.variants.variant_for_ablation`.
        execution_mode: `"retrain"` for the six model-affecting arms,
            `"reuse_predictions"` for the four gate/constraint/witness
            arms (Requirement 7.6, 7.7, 7.8).
    """

    run_id: str
    source_entry: str
    sub_run_key: str | None
    variant: ExperimentVariant
    execution_mode: Literal["retrain", "reuse_predictions"]


#: Separator for a `witness_removed` sub-run id, matching
#: `infra.readiness.SUB_RUN_SEPARATOR` exactly (that module's fallback
#: expansion must agree with this one, and a frozen test asserts it).
_SUB_RUN_SEPARATOR = "::"


def expand_ablations(ablations: list[AblationConfig] = ABLATIONS) -> tuple[AblationRun, ...]:
    """Expand the ten frozen `ABLATIONS` entries into the twelve runs
    `design.md` Section 2.6 tension 1 and Requirement 7.1/7.8 require.

    Every entry expands to exactly one run, except `witness_removed`,
    which expands to one run per witness named in its
    `config_diff["applies_to"]` (three witnesses -> three runs). `ABLATIONS`
    itself is never mutated or reordered by this function.

    Args:
        ablations: the frozen entries to expand. Defaults to `ABLATIONS`;
            overridable only for tests that want to exercise the expansion
            rule against a synthetic list.

    Returns:
        A tuple of `AblationRun`, in the same order as `ablations`, with
        `witness_removed`'s three sub-runs in the order its
        `config_diff["applies_to"]` lists them.

    Raises:
        KeyError: if an entry's name is not covered by
            `models.variants.variant_for_ablation`'s frozen mapping (an
            `ABLATIONS` entry added without a corresponding variant).
        ValueError: if `witness_removed`'s `config_diff` carries no
            non-empty `"applies_to"` list.
    """
    runs: list[AblationRun] = []
    for entry in ablations:
        if entry.name != "witness_removed":
            run = AblationRun(
                run_id=entry.name,
                source_entry=entry.name,
                sub_run_key=None,
                variant=variant_for_ablation(
                    _SimpleAblationRunKey(source_entry=entry.name, sub_run_key=None)
                ),
                execution_mode=_EXECUTION_MODE_BY_ENTRY[entry.name],
            )
            runs.append(run)
            continue

        applies_to = entry.config_diff.get("applies_to")
        if not applies_to:
            raise ValueError(
                f"ablation entry {entry.name!r} carries no non-empty "
                "config_diff['applies_to'], so its per-witness expansion "
                "(design.md Section 2.6 tension 1) cannot be derived."
            )
        for witness in applies_to:
            run = AblationRun(
                run_id=f"{entry.name}{_SUB_RUN_SEPARATOR}{witness}",
                source_entry=entry.name,
                sub_run_key=witness,
                variant=variant_for_ablation(
                    _SimpleAblationRunKey(source_entry=entry.name, sub_run_key=witness)
                ),
                execution_mode=_EXECUTION_MODE_BY_ENTRY[entry.name],
            )
            runs.append(run)

    return tuple(runs)


@dataclass(frozen=True)
class _SimpleAblationRunKey:
    """Minimal `models.variants.AblationRunLike` satisfier.

    `variant_for_ablation` only reads `source_entry`/`sub_run_key`, and
    `expand_ablations` builds the real `AblationRun` *from* the variant it
    returns -- so a tiny local key type avoids a chicken-and-egg
    construction (`AblationRun(variant=variant_for_ablation(the_very_AblationRun_being_built))`).
    """

    source_entry: str
    sub_run_key: str | None


# ---------------------------------------------------------------------------
# detect_variant_collisions (task 14.3; Requirement 6.4, 7.13, 18.3;
# design.md Section 3.6; Property 11)
# ---------------------------------------------------------------------------

#: Absolute per-patient prediction-agreement tolerance below which two
#: runs' predictions are considered a collision (`decisions.md`/Requirement
#: 7.13: "agree ... within an absolute difference of 1e-9").
COLLISION_PREDICTION_TOLERANCE: float = 1e-9

#: The six retrain-mode run ids `detect_variant_collisions`'s
#: distinct-metric-values check is defined over (Requirement 7.13's "the
#: six retrained runs yield fewer than six distinct primary-metric
#: values"). Resolved from `expand_ablations()` rather than hardcoded, so
#: it tracks the real retrain-mode subset if that set ever changes.
def _retrain_run_ids() -> frozenset[str]:
    return frozenset(run.run_id for run in expand_ablations() if run.execution_mode == "retrain")


class VariantCollisionError(Exception):
    """Raised when `detect_variant_collisions` finds a collision.

    Names every colliding run pair and/or the fewer-than-six-distinct-
    values condition, so a caller's error message does not have to be
    reconstructed from a separately-inspected `CollisionReport`.
    """


@dataclass(frozen=True)
class PredictionCollision:
    """One pair of model-affecting-different runs whose predictions
    nonetheless agree within tolerance for every compared patient.

    Attributes:
        run_id_a / run_id_b: the colliding runs' ids.
        seed: the seed at which the collision was detected.
        max_absolute_difference: the largest per-patient absolute
            prediction difference found between the two runs -- below
            `COLLISION_PREDICTION_TOLERANCE` for every patient, which is
            exactly what makes this a collision.
    """

    run_id_a: str
    run_id_b: str
    seed: int
    max_absolute_difference: float


@dataclass(frozen=True)
class CollisionReport:
    """`detect_variant_collisions`'s output.

    Attributes:
        has_collision: `True` if any `prediction_collisions` entry exists
            or `insufficient_distinct_metrics` is `True`.
        prediction_collisions: every detected `PredictionCollision`.
        insufficient_distinct_metrics: `True` if the six retrain-mode
            runs' primary-metric values number fewer than six distinct
            values, at the seed(s) checked.
        distinct_metric_count: how many distinct retrain-mode primary-
            metric values were actually found.
        checked_seeds: which seeds were checked for prediction collisions.
    """

    has_collision: bool
    prediction_collisions: tuple[PredictionCollision, ...]
    insufficient_distinct_metrics: bool
    distinct_metric_count: int
    checked_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "has_collision": self.has_collision,
            "prediction_collisions": [
                {
                    "run_id_a": c.run_id_a,
                    "run_id_b": c.run_id_b,
                    "seed": c.seed,
                    "max_absolute_difference": c.max_absolute_difference,
                }
                for c in self.prediction_collisions
            ],
            "insufficient_distinct_metrics": self.insufficient_distinct_metrics,
            "distinct_metric_count": self.distinct_metric_count,
            "checked_seeds": list(self.checked_seeds),
        }


def detect_variant_collisions(
    runs: Sequence[AblationRun],
    predictions_by_run: dict[str, dict[int, dict[str, float]]],
    metrics_by_run: dict[str, dict[int, float]],
) -> CollisionReport:
    """Detect genuine model-swapping failures across a set of executed runs
    (Requirement 7.13; Property 11).

    Two independent checks, matching `design.md` Section 3.6 exactly:

    1. **Prediction collision.** At a fixed seed, any two runs in `runs`
       whose `models.variants.model_affecting_fields` differ, but whose
       per-patient predictions agree within
       `COLLISION_PREDICTION_TOLERANCE` for *every* patient both runs
       have a prediction for at that seed, is a collision -- it means two
       supposedly-different models produced indistinguishable output,
       which is evidence the variant difference never actually reached
       the model.
    2. **Insufficient distinct metrics.** Across the six retrain-mode
       runs (`execution_mode == "retrain"`), if the primary-metric values
       recorded in `metrics_by_run` (one value per run, at whichever
       single seed is common to all of them -- the first seed every
       retrain-mode run in `metrics_by_run` has an entry for) number
       fewer than six *distinct* values, that is also reported as a
       collision.

    Args:
        runs: the runs to check for prediction collisions (typically all
            twelve from `expand_ablations()`, or any subset).
        predictions_by_run: `{run_id: {seed: {patient_id: prediction}}}`.
            A run/seed absent from this mapping is simply not compared at
            that seed -- no error.
        metrics_by_run: `{run_id: {seed: primary_metric_value}}`, used
            for the distinct-metric-count check.

    Returns:
        A `CollisionReport`. Never raises -- `VariantCollisionError` is
        for callers that want to turn a positive detection into an
        exception (e.g. a top-level runner), not for this function
        itself, which only detects and reports.
    """
    from models.variants import model_affecting_fields

    checked_seeds: set[int] = set()
    for run in runs:
        checked_seeds.update(predictions_by_run.get(run.run_id, {}).keys())
    checked_seeds_sorted = tuple(sorted(checked_seeds))

    collisions: list[PredictionCollision] = []
    for seed in checked_seeds_sorted:
        runs_at_seed = [run for run in runs if seed in predictions_by_run.get(run.run_id, {})]
        for i, run_a in enumerate(runs_at_seed):
            for run_b in runs_at_seed[i + 1 :]:
                if model_affecting_fields(run_a.variant) == model_affecting_fields(
                    run_b.variant
                ):
                    continue  # not model-affecting-different; not a candidate collision
                preds_a = predictions_by_run[run_a.run_id][seed]
                preds_b = predictions_by_run[run_b.run_id][seed]
                shared_patient_ids = set(preds_a) & set(preds_b)
                if not shared_patient_ids:
                    continue
                max_diff = max(
                    abs(preds_a[pid] - preds_b[pid]) for pid in shared_patient_ids
                )
                if max_diff < COLLISION_PREDICTION_TOLERANCE:
                    collisions.append(
                        PredictionCollision(
                            run_id_a=run_a.run_id,
                            run_id_b=run_b.run_id,
                            seed=seed,
                            max_absolute_difference=max_diff,
                        )
                    )

    retrain_ids = _retrain_run_ids()
    common_seed_for_retrain: int | None = None
    for seed in checked_seeds_sorted:
        if all(seed in metrics_by_run.get(run_id, {}) for run_id in retrain_ids):
            common_seed_for_retrain = seed
            break
    if common_seed_for_retrain is None:
        # No seed has a metric recorded for every retrain-mode run; fall
        # back to every (run, seed) metric value present, pooled, so the
        # distinct-value check still has something to evaluate against.
        pooled_values = {
            value
            for run_id in retrain_ids
            for value in metrics_by_run.get(run_id, {}).values()
        }
        distinct_count = len(pooled_values)
    else:
        values_at_seed = {
            metrics_by_run[run_id][common_seed_for_retrain] for run_id in retrain_ids
        }
        distinct_count = len(values_at_seed)

    insufficient_distinct_metrics = distinct_count < len(retrain_ids)

    return CollisionReport(
        has_collision=bool(collisions) or insufficient_distinct_metrics,
        prediction_collisions=tuple(collisions),
        insufficient_distinct_metrics=insufficient_distinct_metrics,
        distinct_metric_count=distinct_count,
        checked_seeds=checked_seeds_sorted,
    )
