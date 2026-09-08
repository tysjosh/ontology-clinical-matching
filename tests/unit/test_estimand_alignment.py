"""Unit tests for `src/protocols/estimand_alignment.py`.

**Validates: Requirements 4.1, 4.2, 4.4, 4.5, 4.6, 4.7**
"""

from pathlib import Path

import pytest
import yaml

from protocols.estimand_alignment import (
    REQUIRED_ALIGNMENT_DIMENSIONS,
    AlignmentClassification,
    build_estimand_alignment_report,
)
from protocols.schema import TargetTrialSpec

MAIN_SPEC_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "target_trials" / "main.yaml"
)


def _load_frozen_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


def test_report_contains_an_entry_for_every_required_dimension():
    """Requirement 4.1: every required dimension is present, exactly once."""
    spec = _load_frozen_spec()

    report = build_estimand_alignment_report(spec)

    dimensions = [entry.dimension for entry in report.entries]
    assert dimensions == list(REQUIRED_ALIGNMENT_DIMENSIONS)
    assert len(dimensions) == len(set(dimensions))


def test_every_entry_has_a_valid_classification_and_populated_fields():
    """Requirement 4.2: every entry is classified as one of the four allowed
    outcomes, and every entry documents both definitions plus notes.
    """
    spec = _load_frozen_spec()

    report = build_estimand_alignment_report(spec)

    for entry in report.entries:
        assert entry.classification in set(AlignmentClassification)
        assert entry.observational_definition
        assert entry.randomized_definition
        assert entry.notes


def test_dimensions_aligned_by_construction_are_classified_aligned():
    """Since ACTG175's observational emulation is built from the same
    trial's randomized data (decisions.md Sections 4, 8-9), dimensions other
    than target_population share an identical definition and should be
    classified `aligned` rather than left ambiguous.
    """
    spec = _load_frozen_spec()

    report = build_estimand_alignment_report(spec)

    for dimension in REQUIRED_ALIGNMENT_DIMENSIONS:
        if dimension == "target_population":
            continue
        entry = report.entry_for(dimension)
        assert entry.classification == AlignmentClassification.ALIGNED
        assert entry.observational_definition == entry.randomized_definition


def test_target_population_reflects_observationalization_selection():
    """The one dimension where the pseudo-observational cohort is a biased
    subset (decisions.md Section 9) must not be silently marked `aligned`.
    """
    spec = _load_frozen_spec()

    report = build_estimand_alignment_report(spec)

    entry = report.entry_for("target_population")
    assert entry.classification == AlignmentClassification.APPROXIMATELY_ALIGNED
    assert entry.observational_definition != entry.randomized_definition


def test_non_independence_disclosure_is_present_and_non_empty():
    """Requirement 4.5-4.6: any lack of independence between the historical
    reference and the observational cohort must be disclosed, not omitted.
    """
    spec = _load_frozen_spec()

    report = build_estimand_alignment_report(spec)

    assert report.independence_disclosure
    assert isinstance(report.independence_disclosure, str)
    assert len(report.independence_disclosure.strip()) > 0


def test_non_independence_disclosure_mentions_shared_provenance_and_corroboration():
    """The frozen `decisions.md` Section 8 determination: the published
    reference is historical corroboration, not independent validation, due
    to shared trial provenance -- this framing must be present verbatim
    enough to be unambiguous.
    """
    spec = _load_frozen_spec()

    report = build_estimand_alignment_report(spec)
    disclosure = report.independence_disclosure.lower()

    assert "historical corroboration" in disclosure
    assert "not independent validation" in disclosure
    assert "same trial" in disclosure or "shared trial provenance" in disclosure


@pytest.mark.parametrize("dimension", REQUIRED_ALIGNMENT_DIMENSIONS)
def test_entry_for_returns_the_matching_entry(dimension):
    """`entry_for` is a convenience accessor; confirm it returns the correct
    entry for every dimension rather than only being exercised implicitly.
    """
    spec = _load_frozen_spec()

    report = build_estimand_alignment_report(spec)

    entry = report.entry_for(dimension)
    assert entry.dimension == dimension


def test_entry_for_unknown_dimension_raises_key_error():
    spec = _load_frozen_spec()
    report = build_estimand_alignment_report(spec)

    with pytest.raises(KeyError):
        report.entry_for("not_a_real_dimension")
