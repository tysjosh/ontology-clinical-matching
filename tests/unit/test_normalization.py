"""Unit tests for `src/data/normalization.py`.

**Validates: Requirements 5.7, 5.8**
"""

from pathlib import Path

import pytest
import yaml

from data.normalization import (
    ConceptUnitMapping,
    MalformedMappingError,
    NormalizedConcept,
    UnresolvedConceptFlag,
    load_concept_unit_mapping,
    normalize_concept,
)

FROZEN_MAPPING_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "normalization"
    / "concept_unit_mapping.yaml"
)


def _load_frozen_mapping_dict() -> dict:
    with FROZEN_MAPPING_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_frozen_mapping_loads_and_exposes_version():
    """The mapping's `version` field is accessible/loaded correctly."""
    mapping = load_concept_unit_mapping(FROZEN_MAPPING_PATH)

    assert isinstance(mapping, ConceptUnitMapping)
    assert mapping.version == "1.0.0"
    assert "baseline_cd4" in mapping.mappings


def test_known_raw_code_maps_to_expected_concept_and_unit():
    mapping = load_concept_unit_mapping(FROZEN_MAPPING_PATH)

    result = normalize_concept("baseline_cd4", mapping=mapping)

    assert isinstance(result, NormalizedConcept)
    assert result.concept_id == "cd4_count"
    assert result.unit == "cells/mm3"


def test_known_raw_code_maps_deterministically_across_repeated_calls():
    """The same raw_code always maps to the same normalized concept/unit
    across repeated calls (Requirement 5.7: deterministic normalization).
    """
    mapping = load_concept_unit_mapping(FROZEN_MAPPING_PATH)

    results = [normalize_concept("baseline_cd8", mapping=mapping) for _ in range(5)]

    assert all(r == results[0] for r in results)
    assert results[0].concept_id == "cd8_count"
    assert results[0].unit == "cells/mm3"


def test_known_raw_code_maps_deterministically_across_independent_loads():
    """Determinism also holds when the mapping is reloaded from disk each
    time, simulating separate process invocations rather than a single
    cached in-memory mapping.
    """
    first = normalize_concept("age", mapping=load_concept_unit_mapping(FROZEN_MAPPING_PATH))
    second = normalize_concept("age", mapping=load_concept_unit_mapping(FROZEN_MAPPING_PATH))

    assert first == second == NormalizedConcept(concept_id="age", unit="years")


def test_treatment_arm_codes_map_to_expected_concept_ids():
    """Raw arm codes map to the treatment concept_ids frozen in
    `configs/target_trials/main.yaml` (`decisions.md` Section 3).
    """
    mapping = load_concept_unit_mapping(FROZEN_MAPPING_PATH)

    arm_0 = normalize_concept("arm_0", mapping=mapping)
    arm_1 = normalize_concept("arm_1", mapping=mapping)

    assert arm_0.concept_id == "zidovudine"
    assert arm_1.concept_id == "zidovudine+didanosine"


def test_unmapped_raw_code_raises_unresolved_concept_flag():
    """An unmapped/unknown raw code raises `UnresolvedConceptFlag`, not
    silently returning `None` or an empty result (Requirement 5.8).
    """
    mapping = load_concept_unit_mapping(FROZEN_MAPPING_PATH)

    with pytest.raises(UnresolvedConceptFlag) as exc_info:
        normalize_concept("totally_unknown_raw_code", mapping=mapping)

    assert exc_info.value.raw_code == "totally_unknown_raw_code"


def test_excluded_treatment_arms_are_unresolved_not_silently_mapped():
    """ACTG175 arms 2/3 are excluded before any modeling with no exceptions
    (`decisions.md` Section 3); they must not have a normalized treatment
    concept and must raise rather than resolve to something plausible-looking.
    """
    mapping = load_concept_unit_mapping(FROZEN_MAPPING_PATH)

    with pytest.raises(UnresolvedConceptFlag):
        normalize_concept("arm_2", mapping=mapping)
    with pytest.raises(UnresolvedConceptFlag):
        normalize_concept("arm_3", mapping=mapping)


def test_normalize_concept_loads_default_mapping_when_none_given(monkeypatch):
    """When no `mapping` is passed, `normalize_concept` loads the default
    frozen mapping itself.
    """
    result = normalize_concept("weight_kg")

    assert result == NormalizedConcept(concept_id="weight", unit="kg")


def test_load_concept_unit_mapping_raises_on_nonexistent_path(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(MalformedMappingError, match="not found"):
        load_concept_unit_mapping(missing_path)


def test_load_concept_unit_mapping_raises_on_malformed_yaml(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed\n")

    with pytest.raises(MalformedMappingError):
        load_concept_unit_mapping(bad_yaml)


def test_load_concept_unit_mapping_raises_on_non_mapping_yaml(tmp_path):
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- one\n- two\n")

    with pytest.raises(MalformedMappingError, match="mapping"):
        load_concept_unit_mapping(list_yaml)


def test_load_concept_unit_mapping_raises_on_missing_version_field(tmp_path):
    raw = _load_frozen_mapping_dict()
    del raw["version"]
    incomplete_path = tmp_path / "incomplete.yaml"
    incomplete_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(MalformedMappingError):
        load_concept_unit_mapping(incomplete_path)


def test_mapping_rejects_unknown_top_level_key(tmp_path):
    """`extra="forbid"`: an unrecognized top-level key is rejected rather
    than silently ignored.
    """
    raw = _load_frozen_mapping_dict()
    raw_with_extra = {**raw, "unexpected_extra_field": "nope"}
    extra_path = tmp_path / "extra.yaml"
    extra_path.write_text(yaml.safe_dump(raw_with_extra))

    with pytest.raises(MalformedMappingError):
        load_concept_unit_mapping(extra_path)


def test_entry_missing_concept_id_fails_validation(tmp_path):
    raw = {
        "version": "1.0.0",
        "mappings": {
            "bad_entry": {"unit": "years"},
        },
    }
    bad_path = tmp_path / "bad_entry.yaml"
    bad_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(MalformedMappingError):
        load_concept_unit_mapping(bad_path)
