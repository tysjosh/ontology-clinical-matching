"""Experiment identity and the append-only experiment registry.

Every run's identity is a deterministic hash of its fully-resolved config
(`design.md` Section 2.2):

    experiment_id = sha256(canonical_json(resolved_config)).hexdigest()[:16]

Because the seed is part of `ResolvedConfig` (see `src/infra/config.py`,
`ExperimentConfig.seed`), two runs with identical configs but different seeds
always get different experiment IDs, so re-running with a new seed never
collides with, or overwrites, a prior run's results directory
(Requirement 1.6).

`ExperimentRegistry.register()` is the single write path to the append-only
`results/registry.jsonl` store described in `design.md` Section 3.1. It
raises `DuplicateExperimentError` if the computed `experiment_id` already has
a non-empty results directory on disk — this is a filesystem check, not a
check against prior registry entries, so re-registering the same config after
its results directory was removed (e.g. during development) is not blocked,
while accidentally clobbering a completed run's outputs is (`design.md`
"Error Handling": "Duplicate experiment ID with existing results" ->
`DuplicateExperimentError`, run aborted before any file write).

Requirement 1.7 requires the registry to record: experiment ID, dataset
version, trial specification version, causal-model version, model, seed,
split, intervention representation, protocol constraints, overlap gate,
uncertainty gate, and witness policy.

Variant-sourced fields (task 12.2; `design.md` Section 2.6 tension 6)
-------------------------------------------------------------------------

`trial_specification_version` now comes from `resolved_config.target_trial.version`
unconditionally — it was always derivable from the config, independent of
any model variant, and did not need to wait for `ExperimentVariant` to
exist.

`model`, `split`, `intervention_representation`, `protocol_constraints`,
`overlap_gate`, and `uncertainty_gate` were reserved for whichever module
would eventually describe "the model this run builds" — that module is
`models.variants.ExperimentVariant` (task 12). `register()` now accepts an
optional `variant: ExperimentVariant | None` and, when one is supplied,
populates these six fields from it:

- `model` <- `variant.model_family`
- `intervention_representation` <- `variant.treatment_representation`
- `protocol_constraints` <- `variant.protocol_channel`
- `overlap_gate` <- `str(variant.overlap_gate)`
- `uncertainty_gate` <- `str(variant.uncertainty_gate)`

`split` is deliberately **not** sourced from `variant` — no field of
`ExperimentVariant` describes which data split a run evaluates (that is a
property of *when* a registration happens relative to the pipeline, e.g.
"this registration is for the development-split training run," not a
property of the model-building arm). It stays `None` until whichever
runner calls `register()` is in a position to say which split it is
registering a run against; passing a `split` string explicitly through
`register()`'s `split` parameter is how that will be wired up, not through
`variant`.

Without a `variant` (the pre-task-12 call shape, kept working so existing
callers do not break), all six remain `None`, exactly as before — this is
not a breaking change for `scripts/run_causal_analysis.py` or
`scripts/run_stage.py`, neither of which constructs an `ExperimentVariant`
today.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infra.config import ResolvedConfig

# `models.variants` is torch-free at module scope (torch is deferred inside
# `build_model`'s body), so importing `ExperimentVariant` here does not pull
# torch into this module's import graph — this module stays torch-free as
# required by `tests/unit/test_torch_isolation.py`.
from models.variants import ExperimentVariant

#: Default location of the append-only registry store, relative to the
#: repository/results root (`design.md` Section 3.1).
DEFAULT_REGISTRY_PATH = Path("results/registry.jsonl")

#: Default root under which per-experiment results directories live
#: (`design.md` Section 2.3: `results/<experiment_id>/`).
DEFAULT_RESULTS_ROOT = Path("results")

#: Length, in hex characters, of the truncated experiment-id hash.
_EXPERIMENT_ID_LENGTH = 16


class DuplicateExperimentError(Exception):
    """Raised when `experiment_id` already has a non-empty results directory.

    Signals that `register()` refused to proceed because
    `results_root/<experiment_id>/` already exists and contains at least one
    entry, which would indicate a prior run's outputs could be silently
    overwritten (Requirement 1.6, Requirement 1.8).
    """


def canonical_json(resolved_config: ResolvedConfig) -> str:
    """Deterministic JSON serialization of a `ResolvedConfig`.

    Uses `resolved_config.canonical_dict()` (itself a deterministic dict via
    Pydantic's `model_dump(mode="json")`) and serializes it with sorted keys
    and no whitespace ambiguity (fixed, minimal separators), so the same
    resolved config always serializes to the exact same byte string,
    regardless of dict-insertion order or platform.
    """
    canonical_dict = resolved_config.canonical_dict()
    return json.dumps(
        canonical_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def compute_experiment_id(resolved_config: ResolvedConfig) -> str:
    """Compute the deterministic experiment ID for a resolved config.

    `experiment_id = sha256(canonical_json(resolved_config)).hexdigest()[:16]`
    (`design.md` Section 2.2). Deterministic: the same resolved config (seed
    included, since it lives on `resolved_config.experiment.seed`) always
    produces the same ID, on any machine, in any process.
    """
    digest = hashlib.sha256(canonical_json(resolved_config).encode("utf-8")).hexdigest()
    return digest[:_EXPERIMENT_ID_LENGTH]


@dataclass(frozen=True)
class RegistryEntry:
    """One append-only record in `results/registry.jsonl`.

    Fields map to Requirement 1.7's required registry contents. Fields not
    yet modeled elsewhere in the codebase are `None` — see the module
    docstring for which fields those are and why.
    """

    experiment_id: str
    dataset_version: str
    trial_specification_version: str | None
    model: str | None
    seed: int
    split: str | None
    intervention_representation: str | None
    protocol_constraints: str | None
    overlap_gate: str | None
    uncertainty_gate: str | None
    witness_policy: str
    git_commit: str | None
    dirty: bool
    registered_at: str

    def to_json_line(self) -> str:
        """Serialize this entry as one JSON-Lines record (sorted keys)."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=True)


def _build_entry(
    resolved_config: ResolvedConfig,
    experiment_id: str,
    git_commit: str | None,
    dirty: bool,
    split: str | None = None,
    variant: ExperimentVariant | None = None,
) -> RegistryEntry:
    if variant is None:
        model = None
        intervention_representation = None
        protocol_constraints = None
        overlap_gate = None
        uncertainty_gate = None
    else:
        model = variant.model_family
        intervention_representation = variant.treatment_representation
        protocol_constraints = variant.protocol_channel
        overlap_gate = str(variant.overlap_gate)
        uncertainty_gate = str(variant.uncertainty_gate)

    return RegistryEntry(
        experiment_id=experiment_id,
        dataset_version=resolved_config.dataset.version,
        trial_specification_version=resolved_config.target_trial.version,
        model=model,
        seed=resolved_config.experiment.seed,
        split=split,
        intervention_representation=intervention_representation,
        protocol_constraints=protocol_constraints,
        overlap_gate=overlap_gate,
        uncertainty_gate=uncertainty_gate,
        witness_policy=resolved_config.experiment.witness_policy,
        git_commit=git_commit,
        dirty=dirty,
        registered_at=datetime.now(UTC).isoformat(),
    )


class ExperimentRegistry:
    """Append-only registry of experiment registrations.

    Two on-disk locations are involved:

    - `registry_path`: the append-only JSONL log every registration is
      written to (`design.md` Section 3.1). Never truncated or rewritten by
      `register()` — only appended to.
    - `results_root`: the directory under which per-experiment results live
      (`results/<experiment_id>/`, `design.md` Section 2.3). `register()`
      checks this location for a pre-existing, non-empty directory matching
      the computed `experiment_id` and raises `DuplicateExperimentError` if
      found, but does not itself create or write into that directory —
      creating the results directory is the responsibility of whichever
      stage runs after registration.
    """

    def __init__(
        self,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        results_root: str | Path = DEFAULT_RESULTS_ROOT,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.results_root = Path(results_root)

    def _results_dir_is_nonempty(self, experiment_id: str) -> bool:
        results_dir = self.results_root / experiment_id
        return results_dir.exists() and any(results_dir.iterdir())

    def register(
        self,
        resolved_config: ResolvedConfig,
        git_commit: str | None,
        dirty: bool,
        results_root: str | Path | None = None,
        split: str | None = None,
        variant: ExperimentVariant | None = None,
    ) -> str:
        """Register an experiment and append its record to the registry.

        Args:
            resolved_config: the fully-resolved, validated config for this
                run (`src/infra/config.py`'s `ResolvedConfig`).
            git_commit: the source commit the run was executed from, or
                `None` if unavailable.
            dirty: whether the working tree had uncommitted changes at run
                time (Requirement 1.5).
            results_root: overrides the results root configured on this
                registry instance for this call only, if provided.
            split: the data split this registration describes (e.g.
                `"dev"`, `"val"`), or `None` if the caller is not
                registering a run against a specific split. Not derived
                from `variant` — see module docstring.
            variant: the `models.variants.ExperimentVariant` this run
                builds, or `None`. When supplied, `model`,
                `intervention_representation`, `protocol_constraints`,
                `overlap_gate`, and `uncertainty_gate` are populated from
                it (Requirement 6.6); when omitted, those fields remain
                `None` exactly as before task 12.

        Returns:
            The computed `experiment_id`.

        Raises:
            DuplicateExperimentError: if `results_root/<experiment_id>/`
                already exists and is non-empty.
        """
        active_results_root = Path(results_root) if results_root is not None else self.results_root
        experiment_id = compute_experiment_id(resolved_config)

        results_dir = active_results_root / experiment_id
        if results_dir.exists() and any(results_dir.iterdir()):
            raise DuplicateExperimentError(
                f"experiment_id {experiment_id!r} already has a non-empty results "
                f"directory at {results_dir}; refusing to register a duplicate "
                "experiment (this run would otherwise risk overwriting prior "
                "results)."
            )

        entry = _build_entry(
            resolved_config,
            experiment_id,
            git_commit,
            dirty,
            split=split,
            variant=variant,
        )

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json_line())
            handle.write("\n")

        return experiment_id

    def read_all(self) -> list[dict[str, Any]]:
        """Read every record currently in the registry file, in append order.

        Returns an empty list if the registry file does not exist yet.
        Provided primarily for tests and audit tooling; `register()` itself
        never reads the registry file.
        """
        if not self.registry_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.registry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
