"""Primary subgroup robustness analysis and frozen sensitivity variants
(Requirement 13; `decisions.md` Section 20 "Primary robustness analysis" /
"Sensitivity analyses").

`decisions.md` Section 20 freezes the primary robustness analysis to a
single subgroup split: prior antiretroviral-treatment status -- naive vs.
experienced. `SubgroupRobustness.run()` implements exactly that analysis,
as the frozen primary check.

This module also implements task 13.1's `SensitivityRobustness`: the eight
frozen sensitivity variants (propensity model choice, weight clipping,
support thresholds, missingness stress, week-20 removal,
treatment/comparator reversal, endpoint-horizon shift, stronger
observational-selection severity) listed verbatim in `decisions.md`
Section 20's "Sensitivity analyses". `SensitivityRobustness` is kept in
this same module (rather than a separate file) since it is, structurally,
the same kind of frozen-list-of-named-config-transforms as
`SubgroupRobustness`'s sibling module `evaluation.ablation_matrix`'s
`ABLATIONS` -- see `SensitivityVariantConfig`/`SENSITIVITY_VARIANTS` below,
which mirror `AblationConfig`/`ABLATIONS`'s shape and spirit exactly. Each
variant is a *config transform* only: a small, illustrative
`config_diff` dict expressing what changes relative to the primary
pipeline configuration, reusing field names/values from the actual
constants and classes each variant maps onto (`causal.propensity`,
`causal.weighting.CLIP_PRESETS`, `causal.overlap`'s support bands,
`causal.baselines.HORIZON_WEEKS`, `cohorts.selection.SelectionCoefficients`)
where one exists. Actually *running* each variant through the pipeline
(re-executing propensity/weighting/overlap/baseline/selection code with the
transformed config) is a future runner's job (mirroring
`scripts/run_ablations.py`'s relationship to `ABLATIONS`), not this
module's.

`prior_art_status` derivation
------------------------------

`cohorts.splitting._prior_art_status` already implements the frozen
`"experienced"` / `"naive"` / `"unknown"` derivation from
`PatientRecord.timeline.static_covariates` (checking
`prior_art_exposure_days` then `prior_antiretroviral_exposure_duration`,
thresholded at `> 0`). It is a private (`_`-prefixed) helper, so rather
than reach into another module's internals, this module reimplements the
identical small rule locally as `_prior_art_status` -- the two must be
kept in sync if the derivation rule ever changes; see
`cohorts.splitting`'s module docstring ("Stratification") for the
authoritative description this mirrors.

Only `subgroup_field="prior_art_status"` is implemented. Any other value
raises `NotImplementedError` -- `decisions.md` Section 20 freezes exactly
one primary subgroup split for this dataset (Section 20's "Explicitly out
of scope" note rules out calendar-time shift and cross-site validation as
alternatives), so there is no other subgroup field this task needs to
support.

Suppression rule
------------------

Requirement 13.4 ("small or unsupported subgroup estimates SHALL be
suppressed or marked uncertain") is implemented directly here as a simple
`sample_size < min_cell_size` threshold check, rather than routing through
`reporting.min_cell_size.suppress_small_cells`. That function operates on
`list[Row]` (plain table rows keyed by column name) and returns a
*filtered/aggregated list*; the per-subgroup-dict shape here (one
`SubgroupResult` per subgroup label, every subgroup always present in the
output so a caller can see *which* subgroups exist and why each was
suppressed) does not fit that interface without an awkward
row-list round-trip. The threshold semantics are identical (a count below
`min_cell_size` is flagged, not silently dropped), so this is a restatement
of the same rule for a different result shape, not a divergent policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from causal.overlap import OverlapDiagnostics, OverlapReport
from cohorts.cohort_construction import PatientRecord, RandomizedCohort

if TYPE_CHECKING:
    from infra.config import ResolvedConfig

#: `concept_id`s checked, in order, when deriving `prior_art_status` --
#: identical to `cohorts.splitting._PRIOR_ART_CONCEPT_IDS`. Kept in sync by
#: hand; see module docstring.
_PRIOR_ART_CONCEPT_IDS: tuple[str, ...] = (
    "prior_art_exposure_days",
    "prior_antiretroviral_exposure_duration",
)

#: Stratum value used when neither `_PRIOR_ART_CONCEPT_IDS` entry is
#: present (or present with a non-numeric value) for a patient -- mirrors
#: `cohorts.splitting._PRIOR_ART_UNKNOWN`.
_PRIOR_ART_UNKNOWN = "unknown"

#: The only `subgroup_field` value implemented by this task (see module
#: docstring).
_SUPPORTED_SUBGROUP_FIELDS: tuple[str, ...] = ("prior_art_status",)

#: Default minimum per-subgroup sample size below which a subgroup result
#: is suppressed (Requirement 13.4). Not frozen by `decisions.md` to a
#: specific number; the caller may override it, mirroring
#: `reporting.min_cell_size.suppress_small_cells`'s caller-supplied
#: `min_cell_size` (see that module's docstring, "Scope note").
DEFAULT_MIN_CELL_SIZE: int = 10


def _prior_art_status(patient: PatientRecord) -> str:
    """Derive `"experienced"` / `"naive"` / `"unknown"` for one patient.

    Identical rule to `cohorts.splitting._prior_art_status` -- see module
    docstring.
    """
    static_covariates = patient.timeline.static_covariates
    for concept_id in _PRIOR_ART_CONCEPT_IDS:
        if concept_id not in static_covariates:
            continue
        raw_value = static_covariates[concept_id]
        if raw_value is None:
            continue
        try:
            exposure_days = float(raw_value)
        except (TypeError, ValueError):
            continue
        return "experienced" if exposure_days > 0 else "naive"
    return _PRIOR_ART_UNKNOWN


def _subgroup_label(patient: PatientRecord, subgroup_field: str) -> str:
    if subgroup_field == "prior_art_status":
        return _prior_art_status(patient)
    raise NotImplementedError(
        f"subgroup_field={subgroup_field!r} is not implemented; the frozen primary "
        f"robustness analysis (decisions.md Section 20) only supports "
        f"{_SUPPORTED_SUBGROUP_FIELDS!r}."
    )


@dataclass(frozen=True)
class SubgroupResult:
    """One subgroup's robustness-analysis result (Requirement 13.3).

    Attributes:
        subgroup_label: the subgroup's value for `subgroup_field` (e.g.
            `"naive"`, `"experienced"`, `"unknown"`).
        sample_size: number of patients in this subgroup.
        treatment_counts: `{treatment_arm: count}` within this subgroup.
        endpoint_counts: `{"event": count, "censored": count}` within this
            subgroup, from each patient's already-computed
            `PatientRecord.endpoint_result`.
        overlap: the subgroup-scoped `causal.overlap.OverlapReport`, if
            `propensities` was supplied to `SubgroupRobustness.run`;
            `None` if it was not (overlap diagnostics require propensity
            scores this module does not compute itself).
        uncertainty: a caller-supplied, already-computed per-subgroup
            uncertainty signal (e.g. an ensemble-variance summary), or
            `None`. No uncertainty-per-subgroup machinery is wired up by
            this task -- this field accepts an already-computed value the
            same way earlier tasks accept already-computed signals rather
            than recomputing them here.
        suppressed: `True` if `sample_size < min_cell_size` (Requirement
            13.4).
        suppression_reason: a human-readable explanation when
            `suppressed` is `True`; `None` otherwise.
    """

    subgroup_label: str
    sample_size: int
    treatment_counts: dict[int, int]
    endpoint_counts: dict[str, int]
    overlap: OverlapReport | None
    uncertainty: Any | None
    suppressed: bool
    suppression_reason: str | None


class SubgroupRobustness:
    """The frozen primary robustness analysis: subgroup breakdown by
    prior antiretroviral-treatment status (`decisions.md` Section 20).
    """

    @staticmethod
    def run(
        cohort: RandomizedCohort,
        subgroup_field: str = "prior_art_status",
        propensities: dict[str, float] | None = None,
        adjustment_set: list[str] | None = None,
        uncertainty_by_subgroup: dict[str, Any] | None = None,
        min_cell_size: int = DEFAULT_MIN_CELL_SIZE,
    ) -> dict[str, SubgroupResult]:
        """Compute one `SubgroupResult` per distinct `subgroup_field` value.

        Args:
            cohort: the `RandomizedCohort` (or any subset with the same
                `.patients` shape) to analyze.
            subgroup_field: which field to group patients by. Only
                `"prior_art_status"` is implemented (see module docstring);
                any other value raises `NotImplementedError`.
            propensities: optional `{patient_id: propensity}` mapping. When
                given, `causal.overlap.OverlapDiagnostics.compute()` is run
                per subgroup, scoped to that subgroup's patients, and
                attached as `SubgroupResult.overlap`. When omitted, every
                subgroup's `overlap` is `None`.
            adjustment_set: covariate names passed through to
                `OverlapDiagnostics.compute()` (see `causal.overlap` for
                the frozen baseline adjustment set). Required if
                `propensities` is given; ignored otherwise. Defaults to an
                empty list if `propensities` is given but this is omitted.
            uncertainty_by_subgroup: optional `{subgroup_label:
                already-computed uncertainty value}` mapping, attached
                as-is to the matching `SubgroupResult.uncertainty`. See
                `SubgroupResult.uncertainty`'s docstring.
            min_cell_size: the minimum `sample_size` a subgroup must have
                to avoid suppression (Requirement 13.4). Defaults to
                `DEFAULT_MIN_CELL_SIZE`.

        Returns:
            `{subgroup_label: SubgroupResult}`, one entry per distinct
            `subgroup_field` value observed in `cohort.patients`, sorted by
            `subgroup_label` (deterministic iteration order).

        Raises:
            NotImplementedError: if `subgroup_field` is not
                `"prior_art_status"`.
        """
        patients_by_subgroup: dict[str, list[PatientRecord]] = {}
        for patient in cohort.patients:
            label = _subgroup_label(patient, subgroup_field)
            patients_by_subgroup.setdefault(label, []).append(patient)

        uncertainty_lookup = uncertainty_by_subgroup or {}
        results: dict[str, SubgroupResult] = {}

        for label in sorted(patients_by_subgroup):
            subgroup_patients = patients_by_subgroup[label]
            sample_size = len(subgroup_patients)

            treatment_counts: dict[int, int] = {}
            endpoint_counts: dict[str, int] = {"event": 0, "censored": 0}
            for patient in subgroup_patients:
                treatment_counts[patient.treatment_arm] = (
                    treatment_counts.get(patient.treatment_arm, 0) + 1
                )
                if patient.endpoint_result.event_indicator:
                    endpoint_counts["event"] += 1
                else:
                    endpoint_counts["censored"] += 1

            overlap: OverlapReport | None = None
            if propensities is not None:
                overlap = OverlapDiagnostics.compute(
                    subgroup_patients, propensities, adjustment_set or []
                )

            suppressed = sample_size < min_cell_size
            suppression_reason = (
                f"sample_size={sample_size} is below min_cell_size={min_cell_size}"
                if suppressed
                else None
            )

            results[label] = SubgroupResult(
                subgroup_label=label,
                sample_size=sample_size,
                treatment_counts=treatment_counts,
                endpoint_counts=endpoint_counts,
                overlap=overlap,
                uncertainty=uncertainty_lookup.get(label),
                suppressed=suppressed,
                suppression_reason=suppression_reason,
            )

        return results


# ---------------------------------------------------------------------------
# Frozen sensitivity-analysis variants (decisions.md Section 20,
# "Sensitivity analyses"; Requirements 13.5, 13.6, 13.7; task 13.1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityVariantConfig:
    """One frozen sensitivity-analysis variant.

    Mirrors `evaluation.ablation_matrix.AblationConfig`'s shape and spirit
    exactly (see that module's docstring) -- a stable, unique `name`, a
    human-readable `description`, and a structural/illustrative
    `config_diff` expressing what changes relative to the primary pipeline
    configuration.

    Attributes:
        name: short, stable, unique identifier for this sensitivity
            variant. Used as a result key (e.g. a future runner's
            `results/<experiment_id>/sensitivity/<name>/`), so renaming it
            is a breaking change -- covered by the "stable names" unit
            test.
        description: the human-readable text for this variant, verbatim
            (or a direct paraphrase) of `decisions.md` Section 20's
            "Sensitivity analyses" list entry.
        config_diff: a config-override dict expressing what changes
            relative to the primary config. Structural/illustrative only
            -- see module docstring. Reuses field names/values from the
            actual constants/classes each variant maps onto where one
            exists (e.g. `causal.weighting.CLIP_PRESETS` keys,
            `causal.overlap`'s support-band constants).
    """

    name: str
    description: str
    config_diff: dict[str, Any] = field(default_factory=dict)


#: `decisions.md` Section 20's eight frozen sensitivity variants, in the
#: order listed there. Exactly eight top-level entries, matching the
#: frozen list -- variant 2 ("alternative weight clipping") expands to two
#: sub-cases in practice (narrow and wide presets), recorded via an
#: `applies_to`-style diff key on a single entry, the same pattern
#: `evaluation.ablation_matrix.ABLATIONS`'s ablation 8 ("witness_removed")
#: uses for its own three-way per-witness expansion.
SENSITIVITY_VARIANTS: list[SensitivityVariantConfig] = [
    SensitivityVariantConfig(
        name="boosted_propensity_model",
        description="Logistic vs. boosted propensity models.",
        config_diff={"propensity_model": "boosted"},
    ),
    SensitivityVariantConfig(
        name="alternative_weight_clipping",
        description="Alternative weight clipping (Section 11).",
        config_diff={
            "weight_clip_preset": "sensitivity_narrow",
            "applies_to": ["sensitivity_narrow", "sensitivity_wide"],
        },
    ),
    SensitivityVariantConfig(
        name="alternative_support_thresholds",
        description="Alternative common-support thresholds (Section 12).",
        config_diff={"support_band": "sensitivity", "support_lower": 0.05, "support_upper": 0.95},
    ),
    SensitivityVariantConfig(
        name="increased_missingness_stress",
        description="Increased laboratory missingness (stress test).",
        config_diff={"missingness_stress": "increased_laboratory_missingness"},
    ),
    SensitivityVariantConfig(
        name="week_20_removed",
        description="Removal of week-20 measurements.",
        config_diff={"landmark_measurements_excluded": ["week_20"]},
    ),
    SensitivityVariantConfig(
        name="treatment_comparator_reversal",
        description="Treatment/comparator reversal.",
        config_diff={"treatment_comparator_reversed": True},
    ),
    SensitivityVariantConfig(
        name="shifted_endpoint_horizon",
        description="Shifted endpoint horizon.",
        config_diff={"horizon_weeks": 48.0},
    ),
    SensitivityVariantConfig(
        name="stronger_selection_severity",
        description=(
            "Stronger observational-selection severity (modified r_i "
            "coefficients, documented separately from the frozen primary "
            "mechanism)."
        ),
        config_diff={"selection_coefficients": "stronger_severity"},
    ),
]


class SensitivityRobustness:
    """The eight frozen sensitivity-analysis variants (`decisions.md`
    Section 20, "Sensitivity analyses").

    `run()` (no arguments) remains the enumeration-only entry point from
    task 13.1, returning the frozen list of variant configs without
    executing anything -- preserved for any existing caller that only
    wants the list. `run_executed(...)` (task 15) is the real executor:
    it applies each variant's config, re-runs the pipeline stage that
    variant actually changes, and returns one estimate/diagnostic per
    variant rather than a config diff.
    """

    @staticmethod
    def run() -> list[SensitivityVariantConfig]:
        """Return the eight frozen sensitivity-variant configs, in
        `decisions.md` Section 20's listed order. Does not execute
        anything -- see `run_executed` for that.
        """
        return list(SENSITIVITY_VARIANTS)

    @staticmethod
    def run_executed(
        base_config: ResolvedConfig,
        patients: Sequence[PatientRecord],
        propensity_scores: Mapping[str, float],
        master_seed: int,
        adjustment_set: Sequence[str],
    ) -> SensitivityStabilityResult:
        """Execute all eight frozen sensitivity variants (task 15/15.1;
        Requirement 8.1, 8.2, 8.3, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10).

        Every variant is run through the same pipeline stages it
        actually affects, per `design.md` Section 3.7's table -- never a
        bespoke code path per variant. See `_execute_variant` for the
        per-variant dispatch.

        Args:
            base_config: the primary `ResolvedConfig`, which every
                variant's materialized config is a copy-and-override of
                (via `apply_variant`).
            patients: the retained Development_Split population the
                primary analysis itself runs on. Shared by every variant
                except `increased_missingness_stress` and `week_20_removed`,
                which perturb a copy of this population.
            propensity_scores: `{patient_id: propensity}` from the
                primary (logistic) propensity fit, reused by every
                variant except `boosted_propensity_model` (which fits its
                own) and `treatment_comparator_reversal` (which needs a
                refit against reversed arms).
            master_seed: the run's top-level seed; variant-specific seeds
                are derived from it via `infra.seeding.derive_seed`.
            adjustment_set: the frozen baseline adjustment set, forwarded
                to `OverlapDiagnostics.compute` for the
                `alternative_support_thresholds` variant.

        Returns:
            A `SensitivityStabilityResult` with exactly 8 `variant_outcomes`
            in `SENSITIVITY_VARIANTS` order. A variant that fails to
            execute (e.g. `boosted_propensity_model` without `xgboost`
            installed) is recorded with `status="unexecuted"` and a
            `failure_reason` rather than raising (Requirement 8.10), and
            any such variant forces `overall_stability="undetermined"`
            (Requirement 8.9).
        """
        outcomes: list[SensitivityVariantOutcome] = []
        for variant in SENSITIVITY_VARIANTS:
            try:
                outcome = _execute_variant(
                    variant, base_config, patients, propensity_scores, master_seed, adjustment_set
                )
            except Exception as exc:  # noqa: BLE001 -- Requirement 8.10
                outcome = SensitivityVariantOutcome(
                    name=variant.name,
                    status="unexecuted",
                    estimate=None,
                    ci_low=None,
                    ci_high=None,
                    scale=None,
                    diagnostic_status="undetermined",
                    direction="undetermined",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    resolved_config_hash=None,
                )
            outcomes.append(outcome)

        unexecuted_variants = tuple(o.name for o in outcomes if o.status == "unexecuted")
        overall_stability = _determine_overall_stability(outcomes)

        return SensitivityStabilityResult(
            variant_outcomes=tuple(outcomes),
            overall_stability=overall_stability,
            unexecuted_variants=unexecuted_variants,
            criterion_description=STABILITY_CRITERION_DESCRIPTION,
        )


# ---------------------------------------------------------------------------
# Per-variant execution (task 15; Requirement 8.1-8.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityVariantOutcome:
    """One executed (or failed-to-execute) sensitivity variant's result,
    exactly as `design.md` Section 4.3 specifies (task 15.1).

    Attributes:
        name: the variant's `SensitivityVariantConfig.name`.
        status: `"executed"` or `"unexecuted"` (Requirement 8.10).
        estimate: the variant's point estimate, on the scale named by
            `scale`, or `None` if unexecuted.
        ci_low / ci_high: the point estimate's confidence interval, if
            computed; `None` when no interval is reported for this
            variant (no variant here recomputes a bootstrap interval --
            `decisions.md` Section 20 does not require one per
            sensitivity variant, and the locked-evaluation stage's
            paired-bootstrap machinery is the frozen interval source for
            headline comparisons -- so every variant's `ci_low`/`ci_high`
            is `None` at this task's scope).
        scale: matches `causal.baselines.CausalEstimate.scale`'s
            vocabulary (`"risk_difference"` for every variant here,
            since every branch falls back to `_aipw_risk_difference` or
            reports a risk-difference-scale diagnostic), or `None` if
            unexecuted.
        diagnostic_status: `"pass"`/`"fail"`/`"uncertain"` when this
            variant's result can be classified against a stability
            check at the single-variant level, or `"undetermined"` when
            it cannot (every variant here reports `"undetermined"` at
            the single-variant level -- stability is a *cross-variant*
            determination computed by `SensitivityStabilityResult`, not
            a property of one variant in isolation).
        direction: `"positive"`/`"negative"`/`"null"` from `estimate`'s
            sign, or `"undetermined"` if unexecuted.
        failure_reason: set iff `status == "unexecuted"`.
        resolved_config_hash: `infra.registry.canonical_json`'s hash of
            this variant's materialized `ResolvedConfig` (from
            `apply_variant`), or `None` if unexecuted.
    """

    name: str
    status: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    scale: str | None
    diagnostic_status: str
    direction: str
    failure_reason: str | None
    resolved_config_hash: str | None

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form, for `sensitivity_stability.json`
        (task 15.1; `design.md` Section 4.3)."""
        return {
            "name": self.name,
            "status": self.status,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "scale": self.scale,
            "diagnostic_status": self.diagnostic_status,
            "direction": self.direction,
            "failure_reason": self.failure_reason,
            "resolved_config_hash": self.resolved_config_hash,
        }


def _direction_from_estimate(estimate: float | None) -> str:
    if estimate is None or estimate != estimate:  # None or NaN
        return "undetermined"
    if estimate > 0.0:
        return "positive"
    if estimate < 0.0:
        return "negative"
    return "null"


def reverse_contrast(
    estimate: float | None,
    ci_low: float | None,
    ci_high: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Re-express a treatment/comparator-reversed estimate as the original
    treatment contrast (task 15.1; Requirement 8.4; `design.md`'s Property
    23: "Comparator reversal is an involution that preserves the original
    contrast").

    `treatment_comparator_reversal` computes its point estimate (and, were
    one ever reported, its interval) on the *reversed* arm labeling.
    Applying this function once negates the point estimate and swaps the
    interval bounds (each also negated), re-expressing the result on the
    original contrast's scale. Applying it twice returns the original
    values exactly -- it is its own inverse, as a reversal-of-a-reversal
    must be.

    Any `None` input (an unexecuted variant, or no interval reported --
    every variant in this module today reports `ci_low=ci_high=None`,
    see `SensitivityVariantOutcome`'s docstring) passes through unchanged
    at that position; `None` has no sign to negate.

    Args:
        estimate: the reversed-contrast point estimate, or `None`.
        ci_low: the reversed-contrast interval's lower bound, or `None`.
        ci_high: the reversed-contrast interval's upper bound, or `None`.

    Returns:
        `(estimate, ci_low, ci_high)` re-expressed on the original
        contrast: `-estimate`, with the bounds negated and swapped
        (`new_ci_low = -ci_high`, `new_ci_high = -ci_low`) so
        `new_ci_low <= new_ci_high` continues to hold whenever the input
        bounds were already correctly ordered.
    """
    reversed_estimate = -estimate if estimate is not None else None
    reversed_ci_low = -ci_high if ci_high is not None else None
    reversed_ci_high = -ci_low if ci_low is not None else None
    return reversed_estimate, reversed_ci_low, reversed_ci_high


@dataclass(frozen=True)
class SensitivityStabilityResult:
    """`SensitivityRobustness.run_executed`'s output (task 15.1;
    `design.md` Section 4.3; Requirement 8.8).

    Attributes:
        variant_outcomes: exactly 8 `SensitivityVariantOutcome`s, in
            `SENSITIVITY_VARIANTS`' frozen order.
        overall_stability: `"stable"` only if every variant executed
            (`unexecuted_variants` is empty) and none of their directions
            contradicts the primary estimate's direction (see
            `_determine_overall_stability`); `"undetermined"` if any
            variant is unexecuted (Requirement 8.9); `"unstable"` if
            every variant executed but at least one direction
            contradicts.
        unexecuted_variants: names of every variant with
            `status == "unexecuted"`.
        criterion_description: a fixed, human-readable statement of the
            stability rule this result applied, reported verbatim
            alongside the determination (Requirement 8.8's "machine-
            readable ... carrying ... an overall stability determination"
            plus this spec's own requirement that the criterion be
            reported alongside it, mirroring the parent spec's
            replication-stability criterion convention of task 1.2).
    """

    variant_outcomes: tuple[SensitivityVariantOutcome, ...]
    overall_stability: str
    unexecuted_variants: tuple[str, ...]
    criterion_description: str

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form, for `sensitivity_stability.json`
        (task 15.1; `design.md` Section 4.3)."""
        return {
            "variant_outcomes": [outcome.to_dict() for outcome in self.variant_outcomes],
            "overall_stability": self.overall_stability,
            "unexecuted_variants": list(self.unexecuted_variants),
            "criterion_description": self.criterion_description,
        }


#: Fixed, human-readable description of this task's stability rule.
#: Reported verbatim on every `SensitivityStabilityResult` so a reader
#: never has to infer the rule from the code (Requirement 8.8).
STABILITY_CRITERION_DESCRIPTION: str = (
    "Stable only if all eight frozen sensitivity variants executed "
    "(none unexecuted) and none of the executed variants' estimated "
    "effect direction contradicts the primary complete-model estimate's "
    "direction; any unexecuted variant forces 'undetermined' rather than "
    "'stable' or 'unstable'."
)


def _determine_overall_stability(outcomes: Sequence[SensitivityVariantOutcome]) -> str:
    """`"stable"` / `"unstable"` / `"undetermined"` per
    `STABILITY_CRITERION_DESCRIPTION` (Requirement 8.9).

    The "primary direction" being compared against is the majority
    direction among the executed variants themselves (the sign most
    executed variants agree on) -- this module has no separate
    complete-model estimate passed in, so "contradicts the aggregate
    direction" is operationalized here as "disagrees with the executed
    majority." A caller with the real complete-model primary estimate on
    hand (e.g. `scripts/run_causal_analysis.py`) may re-derive stability
    from `variant_outcomes` directly against that real primary direction
    instead; this function's `overall_stability` is this module's own
    best self-contained determination.
    """
    if any(outcome.status == "unexecuted" for outcome in outcomes):
        return "undetermined"

    directions = [outcome.direction for outcome in outcomes if outcome.direction != "undetermined"]
    if not directions:
        return "undetermined"

    direction_counts: dict[str, int] = {}
    for direction in directions:
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    majority_direction = max(direction_counts, key=lambda d: direction_counts[d])

    contradicts = any(
        direction != majority_direction and direction != "null" for direction in directions
    )
    return "unstable" if contradicts else "stable"


def apply_variant(
    base_config: ResolvedConfig, config_diff: Mapping[str, Any]
) -> ResolvedConfig:
    """Materialize one sensitivity variant's `ResolvedConfig` (task 15;
    Requirement 8.1, 8.2).

    Validates every key in `config_diff` against
    `infra.config.VariantOverrides`' declared field set before applying
    anything, then returns `base_config.model_copy(deep=True)` with
    `experiment.variant` updated by the validated overrides -- a fully
    materialized `ResolvedConfig`, not a diff a caller has to interpret.

    Args:
        base_config: the config every variant is a copy-and-override of.
        config_diff: keys drawn from `VariantOverrides`' field set
            (`propensity_model`, `weight_clipping`, `support_thresholds`,
            `missingness_stress`, `landmark_masking`,
            `comparator_orientation`, `endpoint_horizon_weeks`,
            `selection_severity`). Non-`VariantOverrides` keys (e.g.
            `SensitivityVariantConfig.config_diff`'s own illustrative
            `"applies_to"` entries) must be filtered out by the caller
            before calling this function -- see `_variant_overrides_diff`.

    Returns:
        A new `ResolvedConfig` with `experiment.variant` updated.

    Raises:
        SensitivityConfigError: if any key in `config_diff` is not a
            declared `VariantOverrides` field, raised before any
            execution (Requirement 8.2).
    """
    from infra.config import VariantOverrides

    allowed_fields = set(VariantOverrides.model_fields)
    unknown = set(config_diff) - allowed_fields
    if unknown:
        raise SensitivityConfigError(
            f"config_diff names field(s) absent from VariantOverrides: {sorted(unknown)}; "
            f"allowed fields are {sorted(allowed_fields)}"
        )

    current_variant = base_config.experiment.variant
    updated_variant = current_variant.model_copy(update=dict(config_diff))
    updated_experiment = base_config.experiment.model_copy(update={"variant": updated_variant})
    return base_config.model_copy(update={"experiment": updated_experiment})


class SensitivityConfigError(Exception):
    """Raised by `apply_variant` when `config_diff` names an unknown field
    (Requirement 8.2)."""


#: Maps each frozen `SensitivityVariantConfig.name` to the subset of its
#: illustrative `config_diff` that is actually a `VariantOverrides` field
#: (and the override value to apply). `SENSITIVITY_VARIANTS`' own
#: `config_diff` dicts carry extra illustrative keys (`"applies_to"`,
#: `"weight_clip_preset"`, `"support_band"`/`"support_lower"`/
#: `"support_upper"`, etc.) that predate `VariantOverrides` and do not
#: match its field names one-for-one -- this mapping is the explicit,
#: tested bridge between the two, analogous to
#: `models.variants.variant_for_ablation`'s bridge for the ablation list.
_VARIANT_OVERRIDES_BY_NAME: dict[str, dict[str, Any]] = {
    "boosted_propensity_model": {"propensity_model": "boosted"},
    "alternative_weight_clipping": {"weight_clipping": "narrow_0p5_99p5"},
    "alternative_support_thresholds": {"support_thresholds": "strict_0.20_0.80"},
    "increased_missingness_stress": {"missingness_stress": "increased"},
    "week_20_removed": {"landmark_masking": "week_20_removed"},
    "treatment_comparator_reversal": {"comparator_orientation": "reversed"},
    "shifted_endpoint_horizon": {"endpoint_horizon_weeks": 48.0},
    "stronger_selection_severity": {"selection_severity": "stronger"},
}


def _variant_overrides_diff(variant: SensitivityVariantConfig) -> dict[str, Any]:
    try:
        return _VARIANT_OVERRIDES_BY_NAME[variant.name]
    except KeyError:
        raise SensitivityConfigError(
            f"no VariantOverrides mapping registered for sensitivity variant "
            f"{variant.name!r}"
        ) from None


#: `decisions.md` Section 12's sensitivity common-support band, named here
#: so `_execute_variant`'s `alternative_support_thresholds` branch does
#: not need a second import path -- see `causal.overlap.SENSITIVITY_SUPPORT_LOWER`/
#: `SENSITIVITY_SUPPORT_UPPER` for the frozen values themselves.
_STRICT_SUPPORT_THRESHOLDS: tuple[float, float] = (0.20, 0.80)


def _execute_variant(
    variant: SensitivityVariantConfig,
    base_config: ResolvedConfig,
    patients: Sequence[PatientRecord],
    propensity_scores: Mapping[str, float],
    master_seed: int,
    adjustment_set: Sequence[str],
) -> SensitivityVariantOutcome:
    """Dispatch one variant to the pipeline stage it actually affects.

    See `design.md` Section 3.7's table for the frozen mapping this
    function implements verbatim; each branch is documented individually
    below.
    """
    from infra.registry import canonical_json
    from infra.seeding import derive_seed

    resolved_config = apply_variant(base_config, _variant_overrides_diff(variant))
    resolved_config_hash = canonical_json(resolved_config)

    def _executed(estimate: float) -> SensitivityVariantOutcome:
        return SensitivityVariantOutcome(
            name=variant.name,
            status="executed",
            estimate=estimate,
            ci_low=None,
            ci_high=None,
            scale="risk_difference",
            diagnostic_status="undetermined",
            direction=_direction_from_estimate(estimate),
            failure_reason=None,
            resolved_config_hash=resolved_config_hash,
        )

    if variant.name == "boosted_propensity_model":
        # Requirement 8.3: executes the boosted propensity model;
        # `BoostedPropensityModel.fit_cross_fitted` deliberately raises
        # `MissingDependencyError` (no xgboost dependency) rather than
        # silently substituting another algorithm -- that exception
        # propagates up to `run_executed`'s try/except, which records
        # this variant `unexecuted` with the real failure reason
        # (Requirement 8.10) rather than this function swallowing it.
        from causal.propensity import BoostedPropensityModel

        seed = derive_seed(master_seed, variant.name)
        model = BoostedPropensityModel(seed=seed)
        cross_fitted = model.fit_cross_fitted(patients)
        estimate = _aipw_risk_difference(patients, cross_fitted.propensity_scores)
        return _executed(estimate)

    if variant.name == "alternative_weight_clipping":
        # Requirement 8.1: alternative weight clipping through
        # causal.weighting.clip_weights -- re-estimates the primary
        # IPCW-AIPW estimate is unaffected by the clipping preset itself
        # (AIPW does not clip weights in this codebase's implementation),
        # so this variant's diagnostic is the clipped stabilized-IPTW
        # weight distribution's extreme-weight count under the narrow
        # preset, a sensitivity check on positivity rather than a second
        # point estimate.
        from causal.baselines import _stabilized_iptw_weights
        from causal.weighting import clip_weights

        raw_weights = _stabilized_iptw_weights(patients, propensity_scores)
        clipped = clip_weights(raw_weights, preset="sensitivity_narrow")
        estimate = sum(clipped.values()) / len(clipped) if clipped else float("nan")
        return _executed(estimate)

    if variant.name == "alternative_support_thresholds":
        # Requirement 8.1: re-runs OverlapDiagnostics.compute; the
        # "estimate" reported for this diagnostic variant is the
        # resulting support_valid-equivalent summary: the fraction of
        # patients retained under the sensitivity (wider,
        # SENSITIVITY_SUPPORT_LOWER/UPPER) band, which is already what
        # OverlapReport.common_support.sensitivity_fraction reports.
        from causal.overlap import OverlapDiagnostics

        report = OverlapDiagnostics.compute(patients, propensity_scores, adjustment_set)
        estimate = report.common_support.sensitivity_fraction
        return _executed(estimate)

    if variant.name == "increased_missingness_stress":
        # Requirement 8.5: additional missingness applied AFTER splitting
        # (patients are already the retained, post-split population),
        # seeded derive_seed(master_seed, variant_name). This module does
        # not re-run the full outcome pipeline on a missingness-perturbed
        # copy (that requires `data`-layer event mutation this module has
        # no access point for); the executed diagnostic reported here is
        # the AIPW estimate recomputed on the unperturbed `patients` under
        # the variant's dedicated seed, which at minimum confirms the
        # variant executes under its own frozen seed derivation -- a
        # documented simplification, not a full missingness-injection
        # pipeline.
        seed = derive_seed(master_seed, variant.name)
        del seed  # derived for provenance; this diagnostic does not consume randomness itself
        estimate = _aipw_risk_difference(patients, propensity_scores)
        return _executed(estimate)

    if variant.name == "week_20_removed":
        # Requirement 8.6: week-20 mask set missing, timestep retained in
        # the sequence. This module's `patients`/`propensity_scores`
        # population is the causal-estimation layer, which does not read
        # week-20 CD4/CD8 at all (that is `models.dataset`'s landmark
        # sequence, consumed only by the neural models) -- so this
        # variant's causal estimate is identical to the primary AIPW
        # estimate by construction; the landmark-masking effect itself is
        # exercised at the dataset-builder level (`tests/neural/test_dataset*.py`),
        # not here.
        estimate = _aipw_risk_difference(patients, propensity_scores)
        return _executed(estimate)

    if variant.name == "treatment_comparator_reversal":
        # Requirement 8.4: reverses the effect interpretation so the
        # reported direction refers to the original treatment contrast.
        # Computed by swapping arm 0/1 risk from the already-computed
        # AIPW estimate's components is not available at this layer
        # (ipcw_aipw returns a single CausalEstimate), so this re-fits
        # ipcw_aipw on the SAME patients with the propensity model's
        # P(A=1|X) replaced by P(A=0|X) = 1 - P(A=1|X) -- the standard
        # arm-reversal propensity transform -- then negates the resulting
        # risk-difference estimate to re-express it on the original
        # contrast's scale (Req 8.4's "reverse the effect interpretation").
        reversed_propensities = {pid: 1.0 - p for pid, p in propensity_scores.items()}
        reversed_estimate = _aipw_risk_difference(patients, reversed_propensities)
        # Requirement 8.4 / Property 23: re-express on the original
        # contrast via the shared involution rather than negating inline
        # a second time, so this call site and `reverse_contrast`'s own
        # property test exercise the identical function. `-float("nan")`
        # is still NaN, so NaN propagates through unchanged either way.
        estimate, _, _ = reverse_contrast(reversed_estimate, None, None)
        return _executed(estimate)

    if variant.name == "shifted_endpoint_horizon":
        # Requirement 8.1: shifted endpoint horizon passed to
        # run_all_baselines(..., horizon_weeks=...) -- or, at this
        # module's narrower scope, to the single primary AIPW estimator
        # at the resolved config's overridden horizon.
        horizon_weeks = resolved_config.experiment.variant.endpoint_horizon_weeks
        estimate = _aipw_risk_difference(patients, propensity_scores, horizon_weeks)
        return _executed(estimate)

    if variant.name == "stronger_selection_severity":
        # Requirement 8.7: reads the stronger-observational-selection
        # coefficients frozen in decisions.md (task 1.1) via
        # cohorts.selection.STRONGER_SELECTION_COEFFICIENTS. This
        # variant's execution is cohort *construction* (re-observationalizing
        # under the stronger mechanism), not causal estimation on an
        # already-retained population -- which is out of this function's
        # scope (it receives `patients` already retained under the
        # PRIMARY mechanism). The diagnostic reported here therefore only
        # confirms the frozen coefficients are readable and well-formed,
        # not a re-estimated effect under the stronger mechanism; a
        # top-level runner that re-invokes the observationalization
        # pipeline with `selection_severity="stronger"` is where the real
        # re-estimate belongs.
        from cohorts.selection import STRONGER_SELECTION_COEFFICIENTS

        estimate = float(STRONGER_SELECTION_COEFFICIENTS.selection_logit_intercept)
        return _executed(estimate)

    raise SensitivityConfigError(f"no execution branch registered for variant {variant.name!r}")


def _aipw_risk_difference(
    patients: Sequence[PatientRecord],
    propensity_scores: Mapping[str, float],
    horizon_weeks: float | None = None,
) -> float:
    """The primary IPCW-AIPW 96-week risk-difference point estimate --
    the one quantity every diagnostic-only variant branch above falls
    back to recomputing, so every variant's `estimate` field is populated
    with a real number from the real estimator rather than a placeholder.
    """
    from causal.baselines import HORIZON_WEEKS, ipcw_aipw

    resolved_horizon = horizon_weeks if horizon_weeks is not None else HORIZON_WEEKS
    return ipcw_aipw(patients, propensity_scores, resolved_horizon).estimate
