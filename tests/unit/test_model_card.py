"""Unit tests for `src/reporting/model_card.py`.

**Validates: Requirements 15.4, 15.6**
"""

from reporting.model_card import (
    PERMITTED_CLAIMS,
    PROHIBITED_USES,
    build_model_card,
    to_markdown,
)

_REQUIRED_KEYS = {
    "intended_use",
    "prohibited_uses",
    "permitted_claims",
    "training_population",
    "endpoint",
    "uncertainty_behavior",
    "known_failure_modes",
    "subgroup_findings",
}


def test_build_model_card_runs_without_error_and_has_required_sections():
    card = build_model_card()

    assert _REQUIRED_KEYS.issubset(card.keys())
    for key in _REQUIRED_KEYS:
        assert card[key], f"section {key!r} must not be empty"


def test_all_six_prohibited_claims_appear_verbatim():
    card = build_model_card()
    prohibited_uses = card["prohibited_uses"]

    assert len(PROHIBITED_USES) == 6
    for claim in PROHIBITED_USES:
        assert claim in prohibited_uses


def test_all_five_permitted_claims_appear_verbatim():
    card = build_model_card()
    permitted_claims = card["permitted_claims"]

    assert len(PERMITTED_CLAIMS) == 5
    for claim in PERMITTED_CLAIMS:
        assert claim in permitted_claims


def test_prohibited_and_permitted_claims_appear_verbatim_in_markdown_rendering():
    markdown = to_markdown()

    for claim in PROHIBITED_USES:
        assert claim in markdown
    for claim in PERMITTED_CLAIMS:
        assert claim in markdown


def test_subgroup_findings_is_documented_as_a_placeholder_not_a_claim():
    card = build_model_card()

    assert "TEMPLATE PLACEHOLDER" in card["subgroup_findings"]
