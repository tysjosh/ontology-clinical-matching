"""Concept/unit normalization (Requirement 5.7-5.8).

`design.md` Section 3.3 describes this module as a "deterministic,
versioned concept/unit mapping table (a static YAML lookup table checked
into `configs/`)"; unresolved concepts are flagged, never silently dropped
(Requirement 5.8).

The mapping is loaded from `configs/normalization/concept_unit_mapping.yaml`
(see `DEFAULT_MAPPING_PATH`), which carries its own `version` field. Because
the mapping is a static, versioned file rather than something computed at
runtime, looking up the same `raw_code` always returns the same normalized
concept/unit within a given mapping version — determinism follows directly
from the lookup being a pure read of frozen, checked-in data, with no
network access, randomness, or environment-dependent state involved.

`normalize_concept(raw_code)` raises `UnresolvedConceptFlag(raw_code)` for
any code not present in the mapping. This is deliberately an exception, not
a sentinel return value (e.g. `None`), so a caller cannot accidentally treat
an unresolved concept as "no unit" or otherwise let it disappear from the
pipeline unnoticed (`design.md` Error Handling table: "record retained, not
dropped").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

# Shares the strictness convention used throughout `src/data`: no unknown
# keys, immutable once loaded.
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

#: Default location of the versioned static concept/unit mapping table.
DEFAULT_MAPPING_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "normalization"
    / "concept_unit_mapping.yaml"
)


class MalformedMappingError(Exception):
    """Raised when the concept/unit mapping YAML is absent, not valid YAML,
    not a mapping at the top level, or fails schema validation.
    """


class UnresolvedConceptFlag(Exception):
    """Raised (not silently dropped) when `raw_code` has no entry in the
    concept/unit mapping (Requirement 5.8).

    Carries the unmapped raw code as `raw_code` so callers can surface it
    (e.g. in the data-quality report or an audit log) rather than losing
    track of which code was unresolved.
    """

    def __init__(self, raw_code: str):
        self.raw_code = raw_code
        super().__init__(f"Unresolved concept: no mapping entry for raw code '{raw_code}'")


class ConceptUnitEntry(BaseModel):
    """One mapping-table entry: a raw code's normalized concept and unit."""

    model_config = _MODEL_CONFIG

    concept_id: str
    unit: str | None = None


class ConceptUnitMapping(BaseModel):
    """The complete versioned static concept/unit mapping table.

    `version` is the mapping table's own version string (distinct from the
    dataset contract's `version`); `mappings` is the raw-code -> normalized
    concept/unit lookup table itself.
    """

    model_config = _MODEL_CONFIG

    version: str
    mappings: dict[str, ConceptUnitEntry]


@dataclass(frozen=True)
class NormalizedConcept:
    """The result of normalizing one raw code: its concept ID and unit
    (unit is `None` for concepts that have no associated unit, e.g.
    treatment-arm concepts).
    """

    concept_id: str
    unit: str | None


def _load_yaml_mapping(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise MalformedMappingError(
            f"Concept/unit mapping at {path} is not valid YAML: {exc}"
        ) from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise MalformedMappingError(
            f"Concept/unit mapping at {path} must be a mapping at the top level, "
            f"got {type(data).__name__}"
        )
    return data


def load_concept_unit_mapping(
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
) -> ConceptUnitMapping:
    """Load and schema-validate the static concept/unit mapping table.

    Args:
        mapping_path: path to the mapping YAML file. Defaults to the frozen
            `configs/normalization/concept_unit_mapping.yaml`.

    Returns:
        The validated `ConceptUnitMapping`, including its `version` field.

    Raises:
        MalformedMappingError: if the file is absent, not valid YAML, not a
            mapping at the top level, or fails schema validation.
    """
    path = Path(mapping_path)
    if not path.exists():
        raise MalformedMappingError(f"Concept/unit mapping not found at {path}.")

    raw = _load_yaml_mapping(path)

    try:
        return ConceptUnitMapping.model_validate(raw)
    except ValidationError as exc:
        raise MalformedMappingError(
            f"Concept/unit mapping at {path} failed schema validation:\n{exc}"
        ) from exc


def normalize_concept(
    raw_code: str,
    mapping: ConceptUnitMapping | None = None,
) -> NormalizedConcept:
    """Normalize `raw_code` into a `NormalizedConcept(concept_id, unit)`.

    The lookup is a deterministic read of a static, versioned mapping table:
    the same `raw_code` against the same mapping version always yields the
    same `NormalizedConcept`, across repeated calls and across processes,
    since no randomness or external state is involved.

    Args:
        raw_code: the raw source field name / code to normalize (e.g.
            `"baseline_cd4"`, `"arm_1"`).
        mapping: an already-loaded `ConceptUnitMapping`. If omitted, the
            default mapping (`DEFAULT_MAPPING_PATH`) is loaded fresh for
            this call.

    Returns:
        The `NormalizedConcept` for `raw_code`.

    Raises:
        UnresolvedConceptFlag: if `raw_code` has no entry in the mapping.
            Raised, not swallowed: the caller must handle it explicitly
            rather than receiving `None` or an empty result.
    """
    if mapping is None:
        mapping = load_concept_unit_mapping()

    entry = mapping.mappings.get(raw_code)
    if entry is None:
        raise UnresolvedConceptFlag(raw_code)

    return NormalizedConcept(concept_id=entry.concept_id, unit=entry.unit)
