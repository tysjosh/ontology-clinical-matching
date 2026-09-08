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

from dataclasses import dataclass, field
from typing import Any

from causal.overlap import OverlapDiagnostics, OverlapReport
from cohorts.cohort_construction import PatientRecord, RandomizedCohort

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

    Each variant is a config transform reusing the existing pipeline (see
    module docstring) -- this class does not itself re-run propensity
    estimation, weighting, overlap diagnostics, or baseline estimation; it
    only enumerates the frozen list of variants and their config diffs, the
    same way `evaluation.ablation_matrix.ABLATIONS` enumerates the ten
    frozen ablations without itself running them.
    """

    @staticmethod
    def run() -> list[SensitivityVariantConfig]:
        """Return the eight frozen sensitivity-variant configs, in
        `decisions.md` Section 20's listed order.
        """
        return list(SENSITIVITY_VARIANTS)
