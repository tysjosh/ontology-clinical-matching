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
        stronger_selection_cohort: StrongerSelectionCohort | None = None,
        support_population: SupportPopulation | None = None,
        propensity_n_folds: int = 5,
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
                except `stronger_selection_severity`, which uses
                `stronger_selection_cohort`.
            propensity_scores: `{patient_id: propensity}` from the
                primary (logistic) propensity fit, reused by every
                variant except `boosted_propensity_model` (which fits its
                own) and `treatment_comparator_reversal` (which needs a
                refit against reversed arms).
            master_seed: the run's top-level seed; variant-specific seeds
                are derived from it via `infra.seeding.derive_seed`.
            adjustment_set: the frozen baseline adjustment set.
            stronger_selection_cohort: the development-retained cohort
                re-observationalized under the Section 26 stronger
                mechanism, with its own cross-fitted propensities (see
                `build_stronger_selection_cohort`). `patients` was
                retained under the primary mechanism, so it cannot stand
                in; without this argument `stronger_selection_severity`
                is recorded unexecuted.
            support_population: the same development population before
                common-support trimming, with its primary propensities.
                The support-threshold and missingness variants start from
                it. `None` means `patients` was not trimmed (they are
                then used as is).
            propensity_n_folds: cross-fitting folds for the missingness
                variant's propensity refit; production callers pass the
                frozen 5.

        Stability is judged against the primary IPCW-AIPW estimate's
        direction on `patients`/`propensity_scores` (the estimate the
        primary analysis reports), per `STABILITY_CRITERION_DESCRIPTION`.

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
                    variant,
                    base_config,
                    patients,
                    propensity_scores,
                    master_seed,
                    adjustment_set,
                    stronger_selection_cohort,
                    support_population,
                    propensity_n_folds,
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
        try:
            primary_direction = _direction_from_estimate(
                _aipw_risk_difference(patients, propensity_scores)
            )
        except Exception:  # noqa: BLE001 -- no primary estimate means no direction
            primary_direction = "undetermined"
        overall_stability = _determine_overall_stability(outcomes, primary_direction)

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
        component_estimates: `(label, estimate)` pairs for a variant that
            Section 11/12 defines with more than one setting -- the two
            sensitivity clipping presets, or the two common-support bands.
            `estimate` is the first component; every component's
            direction counts toward stability, so a contradiction in any
            one of them is never hidden behind the reported headline
            value. Empty for single-setting variants.
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
    component_estimates: tuple[tuple[str, float], ...] = ()

    def directions(self) -> tuple[str, ...]:
        """Every effect direction this outcome contributes to stability:
        its own, plus one per component estimate."""
        return (self.direction,) + tuple(
            _direction_from_estimate(value) for _, value in self.component_estimates
        )

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
            "component_estimates": {label: value for label, value in self.component_estimates},
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


def _determine_overall_stability(
    outcomes: Sequence[SensitivityVariantOutcome], primary_direction: str | None = None
) -> str:
    """`"stable"` / `"unstable"` / `"undetermined"` per
    `STABILITY_CRITERION_DESCRIPTION` (Requirement 8.9).

    Any unexecuted variant forces `"undetermined"`.

    `primary_direction` is the primary complete-model estimate's
    direction -- what the criterion text compares against.
    `run_executed` always supplies it, from the primary IPCW-AIPW estimate
    on the same patients and propensities. A variant contradicts when any
    of its directions (`SensitivityVariantOutcome.directions`, which
    includes every component estimate) is the opposite non-null direction
    of the primary one. A `"null"` primary direction has no opposite, so
    nothing can contradict it.

    When `primary_direction` is `None` (a direct caller with no primary
    estimate), the majority direction among the executed variants stands
    in for it.
    """
    if any(outcome.status == "unexecuted" for outcome in outcomes):
        return "undetermined"

    directions = [
        direction
        for outcome in outcomes
        for direction in outcome.directions()
        if direction != "undetermined"
    ]
    if not directions:
        return "undetermined"

    if primary_direction is None or primary_direction == "undetermined":
        direction_counts: dict[str, int] = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        reference_direction = max(direction_counts, key=lambda d: direction_counts[d])
    else:
        reference_direction = primary_direction

    if primary_direction == "null":
        return "stable"

    contradicts = any(
        direction != reference_direction and direction != "null" for direction in directions
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


@dataclass(frozen=True)
class StrongerSelectionCohort:
    """The development-retained cohort under the Section 26 stronger
    selection mechanism, for the `stronger_selection_severity` variant.

    Attributes:
        patients: the development split's retained patients when the
            identical pipeline (same cohort, split, preprocessing, and
            seeds) is observationalized with `selection_severity="stronger"`,
            after the same common-support trim the primary analysis uses.
        propensity_scores: cross-fitted primary (logistic) propensities
            refit on exactly these patients.
    """

    patients: tuple[PatientRecord, ...]
    propensity_scores: Mapping[str, float]


def build_stronger_selection_cohort(
    raw_timelines: Any,
    treatment_arms: Mapping[str, int],
    spec: Any,
    pipeline_seed: int,
    propensity_seed: int,
    n_folds: int,
    precomputed_endpoint_results: Any = None,
) -> StrongerSelectionCohort:
    """Re-run the observationalization pipeline under the stronger
    mechanism and refit the primary propensity model on the result.

    `pipeline_seed` and `propensity_seed` must be the primary analysis's
    own seeds, so the stronger cohort shares its split and differs only
    in which patients selection retains (the Section 26 amendment: "the
    same per-split HMAC-derived seeds"). The sealed reference is not
    needed here, so the pipeline's default placeholder estimator is used.
    """
    from causal.overlap import trim_to_common_support
    from causal.propensity import LogisticPropensityModel
    from cohorts.pipeline import run_observationalization_pipeline

    result = run_observationalization_pipeline(
        raw_timelines,
        treatment_arms,
        spec,
        pipeline_seed,
        precomputed_endpoint_results=precomputed_endpoint_results,
        selection_severity="stronger",
    )
    retained_ids = set(result.dev.retained_patient_ids)
    dev_patients = tuple(p for p in result.split_cohorts.dev if p.patient_id in retained_ids)
    cross_fitted = LogisticPropensityModel(seed=propensity_seed).fit_cross_fitted(
        dev_patients, n_folds=n_folds
    )
    # The same common-support trim the primary analysis applies (Section 26
    # amendment of 2026-10-12, item B).
    trimmed = trim_to_common_support(dev_patients, cross_fitted.propensity_scores)
    return StrongerSelectionCohort(
        patients=trimmed.retained, propensity_scores=cross_fitted.propensity_scores
    )


@dataclass(frozen=True)
class SupportPopulation:
    """The development-retained population *before* common-support
    trimming, with the primary propensities that defined the trim.

    `run_executed`'s `patients` is the trimmed estimation population. Two
    variants need to start from before the trim: the support-threshold
    variant (it re-trims at a different band) and the missingness stress
    test (it perturbs covariates, refits the propensity model, and re-trims
    under the same rule).
    """

    patients: tuple[PatientRecord, ...]
    propensity_scores: Mapping[str, float]


#: The frozen increased-laboratory-missingness stress test (`decisions.md`
#: Section 26 amendment of 2026-10-12, item C): after the split, each
#: development-retained patient's observed baseline CD4 and baseline CD8
#: are deleted independently with this probability, missing completely at
#: random. 0.40 is anchored to the trial itself: week-96 CD4 is missing for
#: 38% of the development cohort (139 of 362, seed-101 replication split),
#: rounded to the nearest tenth.
INCREASED_MISSINGNESS_RATE: float = 0.40

#: The baseline laboratory fields the stress test deletes, as stored
#: `static_covariates` keys (the Section 13 `baseline_cd4`/`baseline_cd8`
#: covariates; `causal.adjustment_set` maps them to these same keys).
INCREASED_MISSINGNESS_FIELDS: tuple[str, ...] = ("baseline_cd4", "baseline_cd8")


def apply_lab_missingness(
    patients: Sequence[PatientRecord], rate: float, seed: int
) -> list[PatientRecord]:
    """Delete each observed `INCREASED_MISSINGNESS_FIELDS` value with
    probability `rate`, completely at random.

    One `random.Random(seed)` stream, one uniform draw per (patient, field),
    patients in `patient_id` order and fields in tuple order -- drawn even
    when the value is already missing, so the stream does not depend on
    the data. A deleted value becomes `None`, which every downstream
    encoding treats as missing under its existing rule (`causal.propensity`
    module docstring). Returns new records in input order; the input
    records are not modified.
    """
    import dataclasses
    import random

    rng = random.Random(seed)
    replaced: dict[str, PatientRecord] = {}
    for patient in sorted(patients, key=lambda p: p.patient_id):
        static = dict(patient.timeline.static_covariates)
        for field_name in INCREASED_MISSINGNESS_FIELDS:
            draw = rng.random()
            if static.get(field_name) is not None and draw < rate:
                static[field_name] = None
        timeline = dataclasses.replace(patient.timeline, static_covariates=static)
        replaced[patient.patient_id] = dataclasses.replace(patient, timeline=timeline)
    return [replaced[p.patient_id] for p in patients]


#: `decisions.md` Section 11's two sensitivity clipping presets, in the
#: order reported (`causal.weighting.CLIP_PRESETS` names). The first is the
#: variant's headline `estimate`.
_SENSITIVITY_CLIP_PRESETS: tuple[str, ...] = ("sensitivity_narrow", "sensitivity_wide")


def _support_band_estimates(
    patients: Sequence[PatientRecord], propensity_scores: Mapping[str, float]
) -> tuple[tuple[str, float], ...]:
    """IPCW-AIPW re-estimated on the patients inside each frozen
    `decisions.md` Section 12 common-support band: the sensitivity band
    `[0.05, 0.95]` first (the headline), then the primary band
    `[0.10, 0.90]`. Bands come from `causal.overlap`'s constants.

    `patients` should be the population *before* trimming, so the two
    bands are two alternative trims of the same cohort; the primary-band
    component then equals the primary (trimmed) estimate."""
    from causal.overlap import (
        PRIMARY_SUPPORT_LOWER,
        PRIMARY_SUPPORT_UPPER,
        SENSITIVITY_SUPPORT_LOWER,
        SENSITIVITY_SUPPORT_UPPER,
    )

    bands = (
        ("sensitivity_0.05_0.95", SENSITIVITY_SUPPORT_LOWER, SENSITIVITY_SUPPORT_UPPER),
        ("primary_0.10_0.90", PRIMARY_SUPPORT_LOWER, PRIMARY_SUPPORT_UPPER),
    )
    estimates: list[tuple[str, float]] = []
    for label, lower, upper in bands:
        inside = [p for p in patients if lower <= propensity_scores[p.patient_id] <= upper]
        estimates.append((label, _aipw_risk_difference(inside, propensity_scores)))
    return tuple(estimates)


class SensitivityVariantNotExecutable(Exception):
    """A variant that cannot be run honestly with what this call was given
    (a missing frozen value or a missing input). Recorded `unexecuted` by
    `run_executed`, which forces `overall_stability="undetermined"`."""


def _execute_variant(
    variant: SensitivityVariantConfig,
    base_config: ResolvedConfig,
    patients: Sequence[PatientRecord],
    propensity_scores: Mapping[str, float],
    master_seed: int,
    adjustment_set: Sequence[str],
    stronger_selection_cohort: StrongerSelectionCohort | None = None,
    support_population: SupportPopulation | None = None,
    propensity_n_folds: int = 5,
) -> SensitivityVariantOutcome:
    """Dispatch one variant to the pipeline stage it actually affects.

    Every executed variant's `estimate` is the primary IPCW-AIPW 96-week
    risk difference recomputed under that variant's change, so every
    direction compared for stability is a treatment-effect direction.
    Each branch is documented individually below.
    """
    from infra.registry import canonical_json
    from infra.seeding import derive_seed

    resolved_config = apply_variant(base_config, _variant_overrides_diff(variant))
    resolved_config_hash = canonical_json(resolved_config)

    def _executed(
        estimate: float, components: tuple[tuple[str, float], ...] = ()
    ) -> SensitivityVariantOutcome:
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
            component_estimates=components,
        )

    if variant.name == "boosted_propensity_model":
        # Requirement 8.3: refits the propensity with the frozen Section 11
        # gradient-boosting model and re-estimates. Without xgboost,
        # `fit_cross_fitted` raises `MissingDependencyError`, which
        # `run_executed` records as unexecuted (Requirement 8.10).
        from causal.propensity import BoostedPropensityModel

        seed = derive_seed(master_seed, variant.name)
        model = BoostedPropensityModel(seed=seed)
        cross_fitted = model.fit_cross_fitted(patients)
        estimate = _aipw_risk_difference(patients, cross_fitted.propensity_scores)
        return _executed(estimate)

    if variant.name == "alternative_weight_clipping":
        # Requirement 8.1 / Section 11's sensitivity clipping: IPCW-AIPW
        # with every inverse-propensity weight clamped to each sensitivity
        # preset's percentile bounds. Only the weights change; the outcome
        # model and censoring weights are the primary ones.
        components = tuple(
            (preset, _aipw_risk_difference(patients, propensity_scores, clip_preset=preset))
            for preset in _SENSITIVITY_CLIP_PRESETS
        )
        return _executed(components[0][1], components)

    untrimmed = support_population or SupportPopulation(
        patients=tuple(patients), propensity_scores=propensity_scores
    )

    if variant.name == "alternative_support_thresholds":
        # Requirement 8.1 / Section 12: the cohort trimmed at each frozen
        # common-support band instead, then IPCW-AIPW.
        components = _support_band_estimates(untrimmed.patients, untrimmed.propensity_scores)
        return _executed(components[0][1], components)

    if variant.name == "increased_missingness_stress":
        # Requirement 8.5 / Section 26 amendment of 2026-10-12, item C: after
        # the split, baseline CD4 and CD8 deleted completely at random at
        # `INCREASED_MISSINGNESS_RATE`, seeded by
        # derive_seed(master_seed, variant_name). The perturbed cohort then
        # goes through the primary path again: refit the primary propensity
        # model, apply the same common-support trim, and re-estimate.
        from causal.overlap import trim_to_common_support
        from causal.propensity import LogisticPropensityModel

        perturbed = apply_lab_missingness(
            untrimmed.patients,
            INCREASED_MISSINGNESS_RATE,
            seed=derive_seed(master_seed, variant.name),
        )
        refit = LogisticPropensityModel(
            seed=derive_seed(master_seed, "propensity")
        ).fit_cross_fitted(perturbed, n_folds=propensity_n_folds)
        trimmed = trim_to_common_support(perturbed, refit.propensity_scores)
        estimate = _aipw_risk_difference(list(trimmed.retained), refit.propensity_scores)
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
        # Requirement 8.4: swap which arm is "treatment". Both the arm
        # labels and the propensities are reversed (A' = 1 - A,
        # e' = 1 - e), IPCW-AIPW is re-estimated -- refitting the outcome
        # model on the relabeled arms -- and the result is re-expressed on
        # the original contrast. (Before this change only the
        # propensities were reversed, so arm-1 patients were weighted by
        # 1/(1-e), which is not a reversal of anything.)
        import dataclasses

        reversed_patients = [
            dataclasses.replace(p, treatment_arm=1 - p.treatment_arm) for p in patients
        ]
        reversed_propensities = {pid: 1.0 - p for pid, p in propensity_scores.items()}
        reversed_estimate = _aipw_risk_difference(reversed_patients, reversed_propensities)
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
        # Requirement 8.7: IPCW-AIPW on the cohort retained under the
        # Section 26 stronger mechanism, with propensities refit on that
        # cohort. `patients` was retained under the primary mechanism, so
        # the caller must supply the re-observationalized cohort
        # (`build_stronger_selection_cohort`). (Before this change the
        # variant reported the frozen selection intercept, +1.2, as its
        # "estimate" -- a coefficient, not an effect.)
        if stronger_selection_cohort is None:
            raise SensitivityVariantNotExecutable(
                "no stronger-selection cohort was supplied; pass "
                "stronger_selection_cohort=build_stronger_selection_cohort(...)"
            )
        estimate = _aipw_risk_difference(
            stronger_selection_cohort.patients, stronger_selection_cohort.propensity_scores
        )
        return _executed(estimate)

    raise SensitivityConfigError(f"no execution branch registered for variant {variant.name!r}")


def _aipw_risk_difference(
    patients: Sequence[PatientRecord],
    propensity_scores: Mapping[str, float],
    horizon_weeks: float | None = None,
    clip_preset: str | None = None,
) -> float:
    """The primary IPCW-AIPW 96-week risk-difference point estimate --
    the one quantity every diagnostic-only variant branch above falls
    back to recomputing, so every variant's `estimate` field is populated
    with a real number from the real estimator rather than a placeholder.
    """
    from causal.baselines import HORIZON_WEEKS, ipcw_aipw

    resolved_horizon = horizon_weeks if horizon_weeks is not None else HORIZON_WEEKS
    return ipcw_aipw(
        patients, propensity_scores, resolved_horizon, inverse_weight_clip_preset=clip_preset
    ).estimate
