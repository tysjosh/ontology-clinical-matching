"""Model card (Requirement 15.4; `decisions.md` Section 22).

Requirement 15.4 freezes the model card's required content (copied
verbatim, not re-derived):

    A model card SHALL state intended use, prohibited uses, training
    population, endpoint, uncertainty behavior, known failure modes, and
    subgroup findings.

Design choices
--------------

- **Fixed template, not parameterized per-run.** Unlike
  `src/reporting/tables.py` or `src/witnesses/audit_report.py`, this module
  takes no per-run arguments. `decisions.md` Section 22's permitted and
  prohibited claims, the training population (Section 2), the endpoint
  (Section 6), and the uncertainty method (Section 15) are all frozen and
  identical across every experiment run in this project's critical path.
  There is nothing here for a caller to supply.
- **A plain `dict`, not a dataclass.** A model card is meant to be read by
  humans (and machine-scanned by the claims-lint test, task 15.3) as a
  structured document, not manipulated as a typed object elsewhere in the
  pipeline. `build_model_card()` therefore returns a plain
  `dict[str, object]` with a fixed set of string keys, which is trivially
  serializable to JSON/YAML and trivially convertible to Markdown via
  `to_markdown()` below. There is no need for a dataclass wrapper because
  nothing downstream needs type-checked field access -- the claims-lint
  test and any report writer just need the values as strings/lists.
- **`prohibited_uses` and `permitted_claims` are copied verbatim.** The six
  prohibited claims and five permitted claims are quoted character-for-
  character from `decisions.md` Section 22 ("Permitted and prohibited
  claims"). Do not paraphrase, reorder, or otherwise edit these lists --
  `tests/unit/test_model_card.py` and the claims-lint test (task 15.3)
  depend on verbatim matches.
- **`subgroup_findings` is a template placeholder, not a claim.** The
  actual subgroup results depend on a real experiment run
  (`evaluation.robustness.SubgroupRobustness.run()`, task 13) and are not
  available at model-card-template-definition time. This field documents
  that fact explicitly rather than fabricating or guessing a finding.
"""

from __future__ import annotations

__all__ = [
    "PERMITTED_CLAIMS",
    "PROHIBITED_USES",
    "build_model_card",
    "to_markdown",
]


#: The five permitted claims, quoted verbatim from `decisions.md` Section 22
#: ("Permitted"). Do not edit -- see module docstring.
PERMITTED_CLAIMS: tuple[str, ...] = (
    "explicit intervention representations improve, or fail to improve, "
    "controlled treatment-effect recovery",
    "predictive accuracy and causal recovery are empirically distinct",
    "overlap-aware abstention reduces unsupported releases",
    "independent witnesses expose different failure modes",
    "the benchmark evaluates causal recovery under known, controlled "
    "confounding",
)

#: The six prohibited claims, quoted verbatim from `decisions.md` Section 22
#: ("Prohibited"). Do not edit -- see module docstring.
PROHIBITED_USES: tuple[str, ...] = (
    "claiming the cohort is independent real-world evidence",
    "claiming individual counterfactual outcomes are ground truth",
    "claiming the system replaces randomized trials",
    "claiming ACTG 175 demonstrates modern clinical-deployment readiness",
    "claiming the model represents complete patient physiology",
    "claiming clinical plausibility establishes causal identification",
)


def build_model_card() -> dict[str, object]:
    """Build the fixed model card (Requirement 15.4).

    Returns a plain `dict` with the following keys, all frozen and
    identical across every run of this project's critical path (see module
    docstring for why this is not parameterized):

    - `intended_use`: what this system is for and, explicitly, what it is
      not for.
    - `prohibited_uses`: the six frozen prohibited claims (`decisions.md`
      Section 22), verbatim, as a list.
    - `permitted_claims`: the five frozen permitted claims (`decisions.md`
      Section 22), verbatim, as a list.
    - `training_population`: the ACTG175 cohort this model is trained on.
    - `endpoint`: the primary composite time-to-event endpoint.
    - `uncertainty_behavior`: a summary of the ensemble/support-gate
      uncertainty mechanism.
    - `known_failure_modes`: documented limitations of the approach.
    - `subgroup_findings`: a placeholder note -- see module docstring.

    Returns:
        A `dict[str, object]` with the fixed model-card content above.
    """
    return {
        "intended_use": (
            "Research prototype for studying whether representing treatment "
            "as an explicit intervention improves controlled treatment-effect "
            "recovery on a single historical clinical trial dataset "
            "(ACTG175), and whether overlap- and uncertainty-aware abstention "
            "reduces unsupported causal releases. This is a controlled causal "
            "benchmark, not a clinical tool: it is NOT for clinical "
            "decision-making, NOT for individualized treatment "
            "recommendations, and NOT for deployment in any care setting."
        ),
        "prohibited_uses": list(PROHIBITED_USES),
        "permitted_claims": list(PERMITTED_CLAIMS),
        "training_population": (
            "LongCART::ACTG175 version 3.2, restricted to the eligible "
            "arm-0 (zidovudine alone) and arm-1 (zidovudine + didanosine) "
            "cohort under intention-to-treat assignment (`decisions.md` "
            "Sections 2-4). Source trial: Hammer SM, et al., "
            "\"A trial comparing nucleoside monotherapy with combination "
            "therapy in HIV-infected adults with CD4 cell counts from 200 "
            "to 500 per cubic millimeter,\" New England Journal of "
            "Medicine, 1996 (PMID 8813038, DOI "
            "10.1056/NEJM199610103351501; `decisions.md` Section 24). "
            "Arms 2 and 3 (zalcitabine-containing regimens) are excluded."
        ),
        "endpoint": (
            "Primary: time to the ACTG composite outcome, defined as the "
            "first occurrence of a >=50% decline in CD4 count from "
            "baseline, an AIDS-defining event, or death (`decisions.md` "
            "Section 6). Secondary endpoints (96-week composite-event "
            "risk, week-20 CD4, week-96 CD4, longitudinal CD4 trajectory) "
            "are also modeled but do not carry the primary "
            "causal-validation claim."
        ),
        "uncertainty_behavior": (
            "Predictive uncertainty is estimated with a 5-replica deep "
            "ensemble (mean -> point prediction, variance -> epistemic "
            "uncertainty), calibrated on validation data only "
            "(`decisions.md` Section 15). A patient is marked 'supported' "
            "only if the patient is within common propensity support, "
            "ensemble uncertainty is at or below the validation 90th "
            "percentile, the protocol is valid for that patient, and no "
            "hard clinical or temporal violation is present "
            "(`decisions.md` Section 16). Unsupported patients are not "
            "silently discarded; their counts and reasons are reported. "
            "Cohort-level estimates are additionally gated by overlap, "
            "balance, and all three witnesses (temporal, causal, "
            "clinical) before release."
        ),
        "known_failure_modes": (
            "Insufficient overlap in extreme-propensity subgroups, where "
            "the causal witness fails or returns uncertain even when "
            "factual prediction is accurate (`decisions.md` Section 16); "
            "small-subgroup estimates are suppressed or marked uncertain "
            "rather than reported (`decisions.md` Section 20, "
            "Requirement 13.3-13.4); the packaged cohort and the "
            "published external reference share the same trial, so the "
            "external reference is historical corroboration only, never "
            "independent validation (`decisions.md` Section 8); single "
            "historical-trial non-independence limits generalization "
            "claims beyond ACTG175; cross-site and calendar-time-shift "
            "robustness are out of scope for this dataset "
            "(`docs/limitations.md`)."
        ),
        "subgroup_findings": (
            "TEMPLATE PLACEHOLDER -- not a claim. This field is populated "
            "per-run from `evaluation.robustness.SubgroupRobustness.run()` "
            "results (the frozen primary subgroup analysis: prior "
            "antiretroviral-treatment status, naive vs. experienced; "
            "`decisions.md` Section 20). No subgroup finding is stated "
            "here because it depends on a completed experiment run, which "
            "does not exist at model-card-template-definition time. A "
            "generated model card for an actual run must replace this "
            "placeholder with that run's `SubgroupResult`s (or an explicit "
            "note that they were suppressed per the minimum-cell-size "
            "policy), not with an assumed or illustrative result."
        ),
    }


def to_markdown(card: dict[str, object] | None = None) -> str:
    """Render a model card `dict` (as returned by `build_model_card()`) as
    Markdown text.

    Args:
        card: a model card dict, e.g. from `build_model_card()`. Defaults
            to a freshly built one if omitted.

    Returns:
        A Markdown-formatted string with one heading per section and the
        `prohibited_uses`/`permitted_claims` lists rendered as bullet
        points.
    """
    if card is None:
        card = build_model_card()

    def _bullets(items: object) -> str:
        return "\n".join(f"- {item}" for item in items)  # type: ignore[union-attr]

    return (
        "# Model Card: Causal Witness Patient World Model\n\n"
        "## Intended use\n\n"
        f"{card['intended_use']}\n\n"
        "## Prohibited uses\n\n"
        f"{_bullets(card['prohibited_uses'])}\n\n"
        "## Permitted claims\n\n"
        f"{_bullets(card['permitted_claims'])}\n\n"
        "## Training population\n\n"
        f"{card['training_population']}\n\n"
        "## Endpoint\n\n"
        f"{card['endpoint']}\n\n"
        "## Uncertainty behavior\n\n"
        f"{card['uncertainty_behavior']}\n\n"
        "## Known failure modes\n\n"
        f"{card['known_failure_modes']}\n\n"
        "## Subgroup findings\n\n"
        f"{card['subgroup_findings']}\n"
    )
