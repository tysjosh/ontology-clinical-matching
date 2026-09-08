"""Unit tests for the frozen ablation matrix (`decisions.md` Section 17;
Requirement 12).
"""

from __future__ import annotations

from src.evaluation.ablation_matrix import ABLATIONS

EXPECTED_NAMES = [
    "no_treatment_input",
    "treatment_as_ordinary_feature",
    "explicit_intervention_channel",
    "intervention_channel_without_protocol",
    "no_overlap_gate",
    "no_uncertainty_gate",
    "no_clinical_constraints",
    "witness_removed",
    "predictive_accuracy_only",
    "placebo_intervention_channel",
]


def test_exactly_ten_ablations():
    """`decisions.md` Section 17 freezes exactly ten primary ablations."""
    assert len(ABLATIONS) == 10


def test_ablation_names_are_unique():
    assert len({a.name for a in ABLATIONS}) == 10


def test_ablation_names_are_stable():
    """Names must match the frozen set exactly -- an accidental rename of any
    entry (e.g. by a future refactor) must fail this test rather than pass
    silently.
    """
    assert [a.name for a in ABLATIONS] == EXPECTED_NAMES


def test_every_ablation_has_a_nonempty_description_and_config_diff():
    for ablation in ABLATIONS:
        assert isinstance(ablation.description, str)
        assert ablation.description.strip() != ""
        assert isinstance(ablation.config_diff, dict)
        assert len(ablation.config_diff) > 0
