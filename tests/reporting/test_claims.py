"""Claims lint test (task 15.3).

**Validates: Requirements 15.5**

This is a *lint* test: it scans already-generated reporting artifacts (as
plain text) for prohibited phrasing patterns derived from `decisions.md`
Section 22 ("Permitted and prohibited claims" -> "Prohibited"), and fails
if any is found. Its job is to catch an accidental prohibited claim being
asserted as fact in generated prose (a table caption, a figure label, a
model-card sentence) before it would ever appear in a real paper artifact.

Scoping decision (read this before adding a new assertion below)
-------------------------------------------------------------------
`decisions.md` Section 22's prohibited-claims list, and its verbatim copy
in `src/reporting/model_card.py::PROHIBITED_USES`, necessarily *contain*
the prohibited phrasing themselves -- they are the documentation of what
must never be claimed, so they legitimately use words like "replaces
randomized trials" as part of *describing* the prohibition, not *making*
it. Scanning those listing sections with this lint test would be a false
positive: the list doing its job would look identical, to a naive
substring scan, to a claim violating its own rule.

The same is true of `docs/responsible_use.md`, which explicitly *quotes*
several of the frozen prohibited claims (Section 5, "Misuse as an RCT
replacement") in order to document that they are forbidden -- e.g. it
quotes "claiming the system replaces randomized trials" verbatim, inside a
blockquote, specifically to say "this must never appear as an assertion."

Given that, this test file scopes the *strict* "must be completely clean"
assertion to artifacts that should never legitimately contain this
phrasing at all, regardless of context:

- `model_card.py`'s non-listing sections (`intended_use`,
  `training_population`, `endpoint`, `uncertainty_behavior`,
  `known_failure_modes`, `subgroup_findings`) -- excluding
  `prohibited_uses`/`permitted_claims`, which are the listing sections
  described above.
- `docs/limitations.md`, which documents accepted scope limitations
  without quoting any of the six frozen prohibited claims.
- `tables.py`'s `build_main_results_table()` output and `figures.py`'s
  structured figure data -- these are numeric/data-structure artifacts
  that should never legitimately contain this phrasing under any framing.

For `docs/responsible_use.md`, which intentionally quotes prohibited
claims to document the prohibition (see above), this test only confirms
the scanner can run against it without erroring -- it deliberately does
NOT assert the file is fully clean, since a "clean" result would in fact
indicate the quoted-prohibition documentation had been removed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from reporting.figures import build_architecture_diagram
from reporting.model_card import build_model_card, to_markdown
from reporting.tables import REQUIRED_COLUMNS, build_main_results_table

_REPO_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Prohibited phrasing patterns (decisions.md Section 22, "Prohibited").
# ---------------------------------------------------------------------------

#: Simple, self-contained regex patterns. Each one independently detects a
#: prohibited claim without needing surrounding context. Names are used as
#: the match labels returned by `scan_text_for_prohibited_claims()`.
_SIMPLE_PATTERNS: dict[str, re.Pattern[str]] = {
    # "claiming the system replaces randomized trials"
    "replaces_randomized_trials": re.compile(
        r"replaces?\s+randomized\s+trials?", re.IGNORECASE
    ),
    # "claiming the cohort is independent real-world evidence"
    "independent_real_world_evidence": re.compile(
        r"independent\s+real[\s-]world\s+evidence", re.IGNORECASE
    ),
    # Broader catch for the task's own example phrasing, even without the
    # word "independent" immediately adjacent.
    "real_world_validation": re.compile(
        r"real[\s-]world\s+validation", re.IGNORECASE
    ),
    # "claiming ACTG 175 demonstrates modern clinical-deployment readiness"
    "clinical_deployment_readiness": re.compile(
        r"clinical[\s-]deployment\s+readiness", re.IGNORECASE
    ),
    # "claiming the model represents complete patient physiology"
    "complete_patient_physiology": re.compile(
        r"complete\s+patient\s+physiology", re.IGNORECASE
    ),
    # "claiming clinical plausibility establishes causal identification"
    "clinical_plausibility_establishes": re.compile(
        r"clinical\s+plausibility\s+establishes", re.IGNORECASE
    ),
}

#: "claiming individual counterfactual outcomes are ground truth" is caught
#: via proximity rather than one fixed phrase, since the prohibited claim
#: can be phrased several ways ("counterfactual outcomes are ground
#: truth", "treated as ground truth for the counterfactual", etc.). A
#: standalone `"ground truth"` pattern would over-match ordinary ML usage
#: (e.g. "compared against ground-truth labels" in a factual-prediction
#: context), so this only flags "ground truth" appearing near
#: "counterfactual".
_GROUND_TRUTH_PATTERN = re.compile(r"ground[\s-]truth", re.IGNORECASE)
_COUNTERFACTUAL_PATTERN = re.compile(r"counterfactual", re.IGNORECASE)
_PROXIMITY_WINDOW_CHARS = 80


def _counterfactual_ground_truth_proximity(text: str) -> bool:
    """True if "ground truth" appears within `_PROXIMITY_WINDOW_CHARS`
    characters of "counterfactual" anywhere in `text`."""
    for match in _GROUND_TRUTH_PATTERN.finditer(text):
        start = max(0, match.start() - _PROXIMITY_WINDOW_CHARS)
        end = min(len(text), match.end() + _PROXIMITY_WINDOW_CHARS)
        if _COUNTERFACTUAL_PATTERN.search(text[start:end]):
            return True
    return False


def scan_text_for_prohibited_claims(text: str) -> list[str]:
    """Scan `text` for prohibited-claim phrasing patterns.

    Args:
        text: plain text extracted from a generated reporting artifact.

    Returns:
        A list of match labels (pattern names) that were found. Empty
        means the text is clean. Order follows `_SIMPLE_PATTERNS`'
        definition order, with the counterfactual/ground-truth proximity
        check appended last if it matched.
    """
    matched = [name for name, pattern in _SIMPLE_PATTERNS.items() if pattern.search(text)]
    if _counterfactual_ground_truth_proximity(text):
        matched.append("counterfactual_ground_truth_proximity")
    return matched


def _flatten_to_text(value: Any) -> str:
    """Flatten an arbitrary JSON-like structure (str/dict/list/other) into
    a single string suitable for `scan_text_for_prohibited_claims()`.

    Table rows and figure data are dicts (sometimes nested) mixing numeric
    and string values -- this walks them and concatenates every string
    leaf, so the scanner can be pointed at a whole artifact rather than
    requiring the caller to pick out string fields by hand.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_to_text(v) for v in value.values())
    if isinstance(value, list | tuple):
        return " ".join(_flatten_to_text(v) for v in value)
    return str(value)


# ---------------------------------------------------------------------------
# Test case 1: model card's non-listing sections must be clean.
# ---------------------------------------------------------------------------

#: Sections that legitimately name the prohibited/permitted claims as
#: *descriptions of the prohibition* -- excluded from the strict scan, per
#: the module docstring's scoping decision.
_MODEL_CARD_LISTING_SECTIONS = {"prohibited_uses", "permitted_claims"}


def test_model_card_non_listing_sections_are_clean():
    card = build_model_card()
    non_listing_sections = {
        key: value for key, value in card.items() if key not in _MODEL_CARD_LISTING_SECTIONS
    }
    assert non_listing_sections, "expected at least one non-listing section to scan"

    for section_name, section_text in non_listing_sections.items():
        matches = scan_text_for_prohibited_claims(_flatten_to_text(section_text))
        assert matches == [], (
            f"model card section {section_name!r} unexpectedly matched "
            f"prohibited-claim pattern(s): {matches!r}"
        )


def test_model_card_markdown_prohibited_uses_heading_is_not_itself_flagged_as_a_new_violation():
    """Sanity check on the scanner's own design: the *listing* sections do
    contain this phrasing (by design -- they document the prohibition), so
    the full markdown rendering is expected to match. This confirms the
    scanner behaves as documented rather than silently missing the
    listing text (which would mean the scanner is not actually working).
    """
    markdown = to_markdown()
    matches = scan_text_for_prohibited_claims(markdown)
    assert matches, (
        "expected the full model card markdown (which includes the "
        "verbatim prohibited-uses listing) to match at least one pattern; "
        "an empty result here would suggest the scanner is not scanning "
        "the listing text at all, which should be verified separately, "
        "not silently assumed clean"
    )


# ---------------------------------------------------------------------------
# Test case 2: a synthetic bad artifact is correctly flagged.
# ---------------------------------------------------------------------------


def test_synthetic_prohibited_claim_is_flagged():
    bad_text = "This model replaces randomized trials for clinical decision-making."
    matches = scan_text_for_prohibited_claims(bad_text)
    assert "replaces_randomized_trials" in matches


def test_synthetic_independent_real_world_evidence_claim_is_flagged():
    bad_text = "The cohort constitutes independent real-world evidence of efficacy."
    matches = scan_text_for_prohibited_claims(bad_text)
    assert "independent_real_world_evidence" in matches


def test_synthetic_counterfactual_ground_truth_claim_is_flagged():
    bad_text = "The predicted counterfactual outcome is treated as ground truth here."
    matches = scan_text_for_prohibited_claims(bad_text)
    assert "counterfactual_ground_truth_proximity" in matches


def test_clean_synthetic_text_is_not_flagged():
    clean_text = "The model reports a hazard ratio with a 95% confidence interval."
    assert scan_text_for_prohibited_claims(clean_text) == []


# ---------------------------------------------------------------------------
# Test case 3: policy documents. See module docstring for scoping.
# ---------------------------------------------------------------------------


def test_limitations_doc_is_clean():
    """`docs/limitations.md` documents accepted scope limitations without
    quoting any of the six frozen prohibited claims, so unlike
    `docs/responsible_use.md` it is asserted fully clean.
    """
    text = (_REPO_ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    matches = scan_text_for_prohibited_claims(text)
    assert matches == [], f"docs/limitations.md unexpectedly matched: {matches!r}"


def test_responsible_use_doc_scan_runs_without_error():
    """`docs/responsible_use.md` deliberately quotes several prohibited
    claims verbatim (Section 5, "Misuse as an RCT replacement") in order
    to document that they must never be asserted -- so this test does not
    assert the file is clean (see module docstring). It only confirms the
    scanner runs against real prose content without erroring, and
    documents why a "dirty" result here is expected and acceptable.
    """
    text = (_REPO_ROOT / "docs" / "responsible_use.md").read_text(encoding="utf-8")
    matches = scan_text_for_prohibited_claims(text)
    assert isinstance(matches, list)
    # Expected: this file quotes prohibited claims by design, so matches
    # here are an accepted, documented false positive for this file only.


# ---------------------------------------------------------------------------
# Test case 4: tables/figures data artifacts must be clean.
# ---------------------------------------------------------------------------


def test_main_results_table_output_is_clean():
    row: dict[str, Any] = {column: 0.0 for column in REQUIRED_COLUMNS}
    row["model_name"] = "intervention_aware_model"
    row["config_label"] = "primary_gated_configuration"

    table = build_main_results_table([row])

    matches = scan_text_for_prohibited_claims(_flatten_to_text(table))
    assert matches == [], f"main results table unexpectedly matched: {matches!r}"


def test_architecture_diagram_mermaid_text_is_clean():
    diagram = build_architecture_diagram()

    matches = scan_text_for_prohibited_claims(_flatten_to_text(diagram))
    assert matches == [], f"architecture diagram unexpectedly matched: {matches!r}"
