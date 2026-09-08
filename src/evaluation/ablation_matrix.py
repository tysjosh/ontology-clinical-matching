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

from dataclasses import dataclass, field
from typing import Any


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
