"""Estimand-alignment reporting (Requirement 4, `decisions.md` Sections 4, 7-9).

`build_estimand_alignment_report(spec)` compares the observational-emulation
definition of each estimand-relevant dimension against the randomized/RCT
definition, for the ten dimensions Requirement 4.1 names: eligibility,
treatment versions, comparator, time zero, follow-up, endpoint, effect
scale, adherence strategy, censoring strategy, and target population. Every
dimension is classified as `"aligned"`, `"approximately_aligned"`,
`"correctable"`, or `"non_alignable"` (Requirement 4.2).

Why most dimensions classify as "aligned"
------------------------------------------
ACTG175's pseudo-observational cohort is not an independently collected
real-world cohort with its own eligibility rule, treatment definitions,
time-zero convention, endpoint, or censoring rule. It is built from the
*same* `TargetTrialSpec` as the randomized emulation: `decisions.md`
Sections 4 and 8-9 construct the complete eligible arm-0/arm-1 randomized
cohort first, using one eligibility rule, one treatment/comparator pair, one
time-zero definition, one follow-up window, one endpoint, and one censoring
rule — and only *afterwards* apply the frozen, covariate-and-treatment
dependent selection mechanism (Section 9) that decides which of those
already-eligible, already-time-zero-anchored patients are retained in the
pseudo-observational cohort. Because both cohorts are built from one
`TargetTrialSpec`, most of these dimensions are aligned *by construction*,
not by coincidence — and that fact is worth stating explicitly in the
report (via each entry's `notes`) rather than leaving it implicit, since a
reader comparing an observational-vs-randomized alignment report would
otherwise reasonably expect to see genuine cross-cohort differences.

The one dimension that is not simply "aligned by construction" is
`target_population`: the observationalization mechanism (Section 9) applies
a covariate- and treatment-dependent Bernoulli selection to the eligible
cohort, so the *realized* pseudo-observational population is a biased
subset of the randomized target population, not an identical set of
patients, even though both share the same eligibility rule. This is
classified `"approximately_aligned"` rather than `"aligned"` (Requirement
4.1-4.2), and flags the Requirement 4.3 transport/standardization question
as a downstream consideration when material differences are found.

Independence disclosure (Requirement 4.5-4.6, `decisions.md` Section 8)
-------------------------------------------------------------------------
Separately from the ten observational-vs-randomized dimensions above, the
frozen determination in `decisions.md` Section 8 concerns the relationship
between the *internal* randomized reference and the *published external*
reference (Hammer et al., NEJM 1996): because the packaged dataset
originates from the same trial as the published reference, the published
reference is not an independent external cohort and must be described as
historical corroboration, never as independent validation. That
determination does not vary by run or by `TargetTrialSpec` instance — it is
a fixed fact about the dataset's provenance — so it is encoded here as a
frozen module-level constant, `FROZEN_INDEPENDENCE_DISCLOSURE`, and surfaced
on every report via `EstimandAlignmentReport.independence_disclosure`. It
must never be omitted from a generated report (Requirement 4.6: "any lack
of independence SHALL be disclosed").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from protocols.schema import TargetTrialSpec


class AlignmentClassification(str, Enum):
    """The four alignment outcomes named in Requirement 4.2.

    A `str` subclass so classifications compare equal to, and serialize as,
    their plain string value (e.g. in JSON audit artifacts) without a
    separate `.value` lookup at every call site.
    """

    ALIGNED = "aligned"
    APPROXIMATELY_ALIGNED = "approximately_aligned"
    CORRECTABLE = "correctable"
    NON_ALIGNABLE = "non_alignable"


# The exact ten dimensions Requirement 4.1 lists, in that order. Exported so
# callers/tests can assert report completeness without duplicating the list.
REQUIRED_ALIGNMENT_DIMENSIONS: tuple[str, ...] = (
    "eligibility",
    "treatment_versions",
    "comparator",
    "time_zero",
    "follow_up",
    "endpoint",
    "effect_scale",
    "adherence_strategy",
    "censoring_strategy",
    "target_population",
)


# The frozen non-independence determination from `decisions.md` Section 8.
# This text must not be re-derived per run: the shared-trial-provenance fact
# it states is fixed for the ACTG175 dataset, not a computed diagnostic.
FROZEN_INDEPENDENCE_DISCLOSURE = (
    "The published external reference (Hammer SM, et al., NEJM 1996; PMID "
    "8813038; external hazard ratio 0.50 for zidovudine+didanosine vs. "
    "zidovudine alone) is historical corroboration only, not independent "
    "validation. The packaged ACTG175 dataset used to build this study's "
    "pseudo-observational cohort originates from the same trial as the "
    "published reference, so the two are not independent: the "
    "observational cohort and the published historical reference share "
    "trial provenance. This lack of independence is disclosed permanently "
    "(decisions.md Section 8) and must be reflected in how the historical "
    "comparison is interpreted and reported -- as descriptive historical "
    "corroboration, never as independent-cohort validation of the model's "
    "causal-effect recovery (Requirement 4.4-4.6)."
)


@dataclass(frozen=True)
class AlignmentEntry:
    """One dimension's observational-vs-randomized alignment comparison.

    Attributes:
        dimension: one of `REQUIRED_ALIGNMENT_DIMENSIONS`.
        observational_definition: how this dimension is defined/realized in
            the pseudo-observational emulation cohort.
        randomized_definition: how this dimension is defined/realized in
            the randomized (pre-observationalization) target-trial
            emulation.
        classification: one of `AlignmentClassification`.
        notes: free-text explanation of why this classification applies,
            always populated (never left implicit, per this module's
            docstring).
    """

    dimension: str
    observational_definition: str
    randomized_definition: str
    classification: AlignmentClassification
    notes: str


@dataclass(frozen=True)
class EstimandAlignmentReport:
    """The complete estimand-alignment report for one `TargetTrialSpec`.

    Attributes:
        entries: exactly one `AlignmentEntry` per
            `REQUIRED_ALIGNMENT_DIMENSIONS`, in that order.
        independence_disclosure: the frozen non-independence determination
            (`FROZEN_INDEPENDENCE_DISCLOSURE`) — always present and
            non-empty (Requirement 4.5-4.6).
    """

    entries: tuple[AlignmentEntry, ...]
    independence_disclosure: str

    def entry_for(self, dimension: str) -> AlignmentEntry:
        """Return the entry for `dimension`, raising `KeyError` if absent."""
        for entry in self.entries:
            if entry.dimension == dimension:
                return entry
        raise KeyError(f"No alignment entry for dimension {dimension!r}")


def _adherence_summary(spec: TargetTrialSpec) -> str:
    """Combine intervention/comparator strategy + switching-handling text.

    Both arms share the same ITT strategy and the same treatment-switching
    handling in `configs/target_trials/main.yaml`; this helper renders that
    shared text once rather than duplicating it per arm.
    """
    return (
        f"strategy={spec.intervention.strategy}; "
        f"treatment_switching_handling="
        f"{spec.intervention.treatment_switching_handling}"
    )


def _eligibility_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = spec.population.eligibility_expression
    return AlignmentEntry(
        dimension="eligibility",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "The pseudo-observational cohort is built by applying the "
            "documented observationalization selection mechanism "
            "(decisions.md Section 9) to patients already found eligible "
            "under this same eligibility expression; eligibility itself is "
            "not redefined for the observational emulation. Aligned by "
            "construction, stated explicitly rather than left implicit."
        ),
    )


def _treatment_versions_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = f"{spec.intervention.concept_id} ({spec.intervention.description})"
    return AlignmentEntry(
        dimension="treatment_versions",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "The observational emulation uses the identical treatment-arm "
            "definition as the randomized emulation; observationalization "
            "changes which patients are retained, never what the "
            "treatment version is. Aligned by construction."
        ),
    )


def _comparator_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = f"{spec.comparator.concept_id} ({spec.comparator.description})"
    return AlignmentEntry(
        dimension="comparator",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "The observational emulation uses the identical comparator-arm "
            "definition as the randomized emulation, for the same reason "
            "as treatment_versions. Aligned by construction."
        ),
    )


def _time_zero_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = spec.time_zero.definition
    return AlignmentEntry(
        dimension="time_zero",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "Both the observational and randomized emulations anchor time "
            "zero at randomized treatment assignment; the observationalization "
            "mechanism resamples which already-randomized patients are "
            "retained but does not shift time zero for any retained patient. "
            "Aligned by construction."
        ),
    )


def _follow_up_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = (
        f"primary={spec.follow_up.primary}; "
        f"secondary_horizon_weeks={spec.follow_up.secondary_horizon_weeks}; "
        f"landmark_weeks={spec.follow_up.landmark_weeks}"
    )
    return AlignmentEntry(
        dimension="follow_up",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "Follow-up window and landmark timepoints are identical for "
            "retained patients in both the observational and randomized "
            "emulations; observationalization affects retention, not the "
            "follow-up definition. Aligned by construction."
        ),
    )


def _endpoint_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = spec.outcome.primary_definition
    return AlignmentEntry(
        dimension="endpoint",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "The composite time-to-event endpoint is computed identically "
            "in both emulations from the same `CompositeEndpoint` logic "
            "(decisions.md Section 6). Aligned by construction."
        ),
    )


def _effect_scale_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = spec.estimand.primary
    return AlignmentEntry(
        dimension="effect_scale",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "Both emulations target the same marginal-hazard-ratio-under-ITT "
            "effect scale (and, secondarily, the same 96-week marginal risk "
            "difference); the observational emulation does not redefine the "
            "estimand's scale. Aligned by construction."
        ),
    )


def _adherence_strategy_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = _adherence_summary(spec)
    return AlignmentEntry(
        dimension="adherence_strategy",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "Both emulations use intention-to-treat with treatment "
            "switching recorded and described descriptively but ignored "
            "in the primary analysis, for both arms. Aligned by "
            "construction."
        ),
    )


def _censoring_strategy_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    definition = spec.censoring.definition
    return AlignmentEntry(
        dimension="censoring_strategy",
        observational_definition=definition,
        randomized_definition=definition,
        classification=AlignmentClassification.ALIGNED,
        notes=(
            "Administrative censoring (end of follow-up or last recorded "
            "observation) applies identically to retained patients in both "
            "emulations. Aligned by construction."
        ),
    )


def _target_population_entry(spec: TargetTrialSpec) -> AlignmentEntry:
    randomized_definition = (
        f"complete eligible arm-0/arm-1 cohort: {spec.population.source_population}"
    )
    observational_definition = (
        "covariate- and treatment-dependent Bernoulli-selected subset of the "
        "same eligible arm-0/arm-1 cohort, per the frozen observationalization "
        "selection mechanism (decisions.md Section 9); same eligibility rule, "
        "different realized composition per split."
    )
    return AlignmentEntry(
        dimension="target_population",
        observational_definition=observational_definition,
        randomized_definition=randomized_definition,
        classification=AlignmentClassification.APPROXIMATELY_ALIGNED,
        notes=(
            "This is the one dimension not aligned purely by construction: "
            "the frozen selection mechanism (decisions.md Section 9) "
            "systematically over/under-retains patients based on treatment "
            "and baseline covariates, so the realized pseudo-observational "
            "population is a biased subset of the randomized target "
            "population rather than an identical set of patients, even "
            "though both share the identical eligibility rule. Classified "
            "approximately_aligned rather than aligned or non-alignable: "
            "the selection mechanism and its coefficients are fully known "
            "and documented, so residual compositional differences are "
            "well-characterized rather than unknown or uncorrectable "
            "(Requirement 4.3's transport/standardization question would "
            "apply here if material differences are found empirically)."
        ),
    )


_ENTRY_BUILDERS = (
    _eligibility_entry,
    _treatment_versions_entry,
    _comparator_entry,
    _time_zero_entry,
    _follow_up_entry,
    _endpoint_entry,
    _effect_scale_entry,
    _adherence_strategy_entry,
    _censoring_strategy_entry,
    _target_population_entry,
)


def build_estimand_alignment_report(spec: TargetTrialSpec) -> EstimandAlignmentReport:
    """Build the estimand-alignment report for `spec` (Requirement 4.1-4.2).

    Produces exactly one `AlignmentEntry` per `REQUIRED_ALIGNMENT_DIMENSIONS`
    (in that order), comparing the observational-emulation definition of
    each dimension (as encoded in `spec`, since ACTG175's observational
    emulation is built from the same trial's randomized data via the frozen
    Section 9 selection mechanism) against the randomized/RCT definition.
    Also attaches the frozen, non-alignable-independence disclosure
    (`FROZEN_INDEPENDENCE_DISCLOSURE`) required by Requirement 4.5-4.6.

    Args:
        spec: the target-trial specification driving both the randomized
            emulation and, via Section 9's selection mechanism, the
            pseudo-observational emulation.

    Returns:
        An `EstimandAlignmentReport` with ten entries and a populated
        `independence_disclosure`.
    """
    entries = tuple(builder(spec) for builder in _ENTRY_BUILDERS)
    return EstimandAlignmentReport(
        entries=entries,
        independence_disclosure=FROZEN_INDEPENDENCE_DISCLOSURE,
    )
