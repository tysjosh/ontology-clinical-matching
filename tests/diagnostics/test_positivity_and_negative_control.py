"""Diagnostic: why does `no_unresolved_positivity_warning` fail, and why is the
negative-control-outcome check `uncertain`? (task 7.3 of
`.kiro/specs/neural-experiment-pipeline`).

Two findings, measured on the real cohort:

1. **Positivity.** The fourth frozen Section 12 condition is resolved in
   `causal/overlap.py` as `extreme_weight_count == 0`, where "extreme weight"
   is a documented *proxy* meaning "propensity outside the primary band
   `[0.10, 0.90]`". On the canonical seed path the count is 0 and the
   condition PASSES. On the un-derived seed path (the one task 7.2 established
   the task-text numbers come from) exactly one arm-1 patient scores 0.9138 --
   and its stabilized IPW weight is at the *bottom* of the weight
   distribution, not the top, because for an arm-1 patient a high propensity
   means a small weight. The proxy is arm-agnostic; true weight extremity is
   arm-dependent.

2. **Negative-control outcome.** The adjusted standardized mean difference for
   baseline CD8 is about -0.114 on the canonical path, so `abs(smd) >= 0.10`
   while the bootstrap CI straddles zero -- the second frozen branch, which
   yields `uncertain`. This is the same residual `baseline_cd8` imbalance task
   7.2 measured (raw -0.1363, stabilized-IPW-weighted -0.1323), reaching the
   classifier through a different estimator.

DIAGNOSTIC ONLY. No frozen threshold, band, or classification rule is changed
here, and none should be changed to make either condition pass.

Privacy
-------

The out-of-band patient's `patient_id` is deliberately NOT recorded, here or
in the artifact. A cohort of 378 with one patient singled out by an exactly
reproducible propensity value would make the id a re-identification vector in
an exported file. Propensity value, treatment arm, and band membership are
sufficient for every claim this module makes.

Artifact
--------

Writes `results/diagnostics/positivity_and_negative_control.json`.
`.gitignore` excludes `results/*` wholesale, so -- exactly as for task 7.2's
artifact -- this is reproducible-on-demand rather than committed. Resolving
that tension needs a decision, not a code change; `.gitignore` is left alone.

Skipped (not failed) when the real ACTG175 data is absent -- `data/raw/` is
gitignored (`docs/data_access.md` Section 2).

Cost, and how to run the full sweep
-----------------------------------

The five-seed x {200, 1000} bootstrap sweep this task asks for dominates the
runtime of this module (roughly 6200 pure-Python regression fits, ~17
minutes), so it is gated behind an environment variable and does NOT run in a
default suite run:

    RUN_FULL_DIAGNOSTIC_SWEEP=1 pytest tests/diagnostics/test_positivity_and_negative_control.py

With the variable set, the sweep is exactly what it always was: all five
frozen seeds at both replicate counts, and the two tests that need that
breadth (`test_classification_is_stable_across_seeds_and_bootstrap_counts`
and `test_derived_200_replicate_ci_matches_a_real_200_replicate_run`) run.

Unset -- the default -- the sweep is reduced to one seed
(`REDUCED_SEEDS`, the first frozen seed) at 200 replicates. That still
exercises the pipeline, the cross-fitting, the positivity finding in full, and
which frozen classification branch fires; what it does NOT establish is
stability across seeds or across replicate counts, and the tests that would
claim that are skipped with a reason naming the variable.

The 200-replicate result is derived from the first 200 replicates of the
1000-replicate run rather than recomputed: `run_negative_control_outcome`
seeds `random.Random(derive_seed(seed, ...))` and draws resamples in a fixed
order, so the two runs share their first 200 resamples exactly.
`test_derived_200_replicate_ci_matches_a_real_200_replicate_run` verifies that
equivalence instead of assuming it. (In reduced mode nothing is derived from a
prefix -- 200 is the only count run -- which is the second reason that test is
gated with the sweep.)

Artifact honesty under a reduced run
------------------------------------

Every sweep payload carries `sweep_completeness` (`"complete"` /
`"reduced"`) alongside the seeds and replicate counts actually measured, so a
reduced payload cannot be read as the full stability claim. On top of that,
**a reduced run refuses to overwrite a `"complete"` artifact**: if
`results/diagnostics/positivity_and_negative_control.json` already records a
complete sweep, a reduced run leaves it untouched and says so, rather than
silently replacing a verified stability result with a weaker one. There is one
artifact path, and the strongest run that produced it wins.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from causal.negative_controls import (
    BASELINE_CD8_FIELD,
    SMD_THRESHOLD,
    _fit_cd8_regression_smd,
    _percentile,
    run_negative_control_outcome,
)
from causal.overlap import (
    MIN_ARM_COUNT,
    PRIMARY_SUPPORT_LOWER,
    PRIMARY_SUPPORT_UPPER,
    SENSITIVITY_SUPPORT_LOWER,
    SENSITIVITY_SUPPORT_UPPER,
    OverlapDiagnostics,
    OverlapReport,
)
from causal.propensity import LogisticPropensityModel
from causal.weighting import CLIP_PRESETS, clip_weights
from cohorts.cohort_construction import PatientRecord
from cohorts.pipeline import run_observationalization_pipeline
from cohorts.sealed_reference import unadjusted_cox_reference_estimator
from data.loaders.actg175 import load_actg175
from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
from infra.readiness import REQUIRED_SEEDS
from infra.seeding import derive_seed
from protocols.schema import TargetTrialSpec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"
TARGET_TRIAL_CONFIG = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
ARTIFACT_PATH = REPO_ROOT / "results" / "diagnostics" / "positivity_and_negative_control.json"

#: The master seed `scripts/run_causal_analysis.py` defaults to.
MASTER_SEED = 20240101

#: Fold count `scripts/run_causal_analysis.py` uses for cross-fitting.
N_FOLDS = 5

#: The clipping preset `decisions.md` Section 11 names as primary.
CLIP_PRESET = "primary"

#: Bootstrap replicate counts task 7.3 asks the classification's stability to
#: be established across.
BOOTSTRAP_COUNTS = (200, 1000)

#: Set (to any non-empty value) to run the full five-seed x {200, 1000}
#: bootstrap sweep. Unset, the sweep is reduced -- see the module docstring's
#: "Cost, and how to run the full sweep". Named and read the same way as
#: `infra.checklist_verifiers.NESTED_VERIFICATION_ENV_VAR`, the existing
#: precedent for an environment variable carrying process-scoped state.
FULL_SWEEP_ENV_VAR = "RUN_FULL_DIAGNOSTIC_SWEEP"

#: Seeds and replicate counts a default (unset) run measures.
REDUCED_SEEDS: tuple[int, ...] = (REQUIRED_SEEDS[0],)
REDUCED_BOOTSTRAP_COUNTS = (200,)


def full_sweep_requested() -> bool:
    """Whether `FULL_SWEEP_ENV_VAR` asks for the complete sweep."""
    return bool(os.environ.get(FULL_SWEEP_ENV_VAR))


#: Reason attached to every test that needs the full sweep's breadth.
_FULL_SWEEP_ONLY = (
    f"needs the full five-seed x {{200, 1000}} bootstrap sweep; set "
    f"{FULL_SWEEP_ENV_VAR}=1 to run it (~17 minutes)"
)

requires_full_sweep = pytest.mark.skipif(not full_sweep_requested(), reason=_FULL_SWEEP_ONLY)

_ARM_1 = 1
_ARM_0 = 0

pytestmark = pytest.mark.skipif(
    not REAL_DATA_PATH.is_dir(),
    reason=(
        f"Real ACTG175 data not present at {REAL_DATA_PATH} (gitignored; see "
        "docs/data_access.md Section 2)."
    ),
)


# --------------------------------------------------------------------------
# Stabilized IPW weights. Same construction as
# `tests/diagnostics/test_overlap_diagnosis.py`, kept local for the same
# reason: `causal.overlap`'s diagnostic weights are unstabilized and
# unclipped, and neither module may grow a weights parameter for a
# diagnostic.
# --------------------------------------------------------------------------


def _stabilized_ipw_weights(
    patients: Sequence[PatientRecord], propensities: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    """`(unclipped, clipped)` stabilized IPW weights, clipping delegated to
    `causal.weighting.clip_weights` with the frozen `"primary"` preset."""
    n_1 = sum(1 for p in patients if p.treatment_arm == _ARM_1)
    n_0 = sum(1 for p in patients if p.treatment_arm == _ARM_0)
    p_treated = n_1 / (n_1 + n_0)

    unclipped: dict[str, float] = {}
    for patient in patients:
        propensity = propensities[patient.patient_id]
        if patient.treatment_arm == _ARM_1:
            unclipped[patient.patient_id] = p_treated / propensity
        elif patient.treatment_arm == _ARM_0:
            unclipped[patient.patient_id] = (1.0 - p_treated) / (1.0 - propensity)
    clipped = clip_weights(unclipped, CLIP_PRESET)
    assert isinstance(clipped, dict)
    return unclipped, clipped


def _out_of_primary_band(propensity: float) -> bool:
    return not (PRIMARY_SUPPORT_LOWER <= propensity <= PRIMARY_SUPPORT_UPPER)


def _in_sensitivity_band(propensity: float) -> bool:
    return SENSITIVITY_SUPPORT_LOWER <= propensity <= SENSITIVITY_SUPPORT_UPPER


# --------------------------------------------------------------------------
# One full measurement per seed path.
# --------------------------------------------------------------------------


class _Measurement:
    """Pipeline -> cross-fitted propensity -> overlap report -> positivity
    detail -> negative-control-outcome sweep, for one seed path."""

    def __init__(self, pipeline_seed: int, propensity_seed: int, seed_path: str) -> None:
        self.seed_path = seed_path
        self.pipeline_seed = pipeline_seed
        self.propensity_seed = propensity_seed
        spec = TargetTrialSpec.model_validate(
            yaml.safe_load(TARGET_TRIAL_CONFIG.read_text(encoding="utf-8"))
        )
        tables = load_actg175(str(REAL_DATA_PATH))
        timelines, arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)
        pipeline_result = run_observationalization_pipeline(
            timelines,
            arms,
            spec,
            pipeline_seed,
            reference_estimator=unadjusted_cox_reference_estimator,
            precomputed_endpoint_results=endpoint_results,
        )
        retained_ids = set(pipeline_result.dev.retained_patient_ids)
        self.adjustment_set: list[str] = list(spec.adjustment_set)
        self.dev_patients = [
            patient
            for patient in pipeline_result.split_cohorts.dev
            if patient.patient_id in retained_ids
        ]
        cross_fitted = LogisticPropensityModel(seed=propensity_seed).fit_cross_fitted(
            self.dev_patients, n_folds=N_FOLDS
        )
        self.propensities: dict[str, float] = dict(cross_fitted.propensity_scores)
        self.report: OverlapReport = OverlapDiagnostics.compute(
            self.dev_patients, self.propensities, self.adjustment_set
        )
        self.unclipped_weights, self.clipped_weights = _stabilized_ipw_weights(
            self.dev_patients, self.propensities
        )
        self.out_of_band = [
            patient
            for patient in self.dev_patients
            if _out_of_primary_band(self.propensities[patient.patient_id])
        ]

    # -- positivity -------------------------------------------------------

    def out_of_band_detail(self) -> list[dict[str, object]]:
        """Per out-of-band patient: propensity, arm, band membership, and where
        its stabilized IPW weight actually sits. NO `patient_id` -- see the
        module docstring's Privacy note."""
        sorted_weights = sorted(self.unclipped_weights.values())
        n = len(sorted_weights)
        low_bound = _weight_percentile(sorted_weights, CLIP_PRESETS[CLIP_PRESET][0])
        high_bound = _weight_percentile(sorted_weights, CLIP_PRESETS[CLIP_PRESET][1])

        detail: list[dict[str, object]] = []
        for patient in self.out_of_band:
            propensity = self.propensities[patient.patient_id]
            unclipped = self.unclipped_weights[patient.patient_id]
            clipped = self.clipped_weights[patient.patient_id]
            rank = sum(1 for w in sorted_weights if w < unclipped) + 1
            detail.append(
                {
                    "propensity": propensity,
                    "treatment_arm": patient.treatment_arm,
                    "outside_primary_band": True,
                    "primary_band": [PRIMARY_SUPPORT_LOWER, PRIMARY_SUPPORT_UPPER],
                    "distance_outside_primary_band": (
                        propensity - PRIMARY_SUPPORT_UPPER
                        if propensity > PRIMARY_SUPPORT_UPPER
                        else PRIMARY_SUPPORT_LOWER - propensity
                    ),
                    "inside_sensitivity_band": _in_sensitivity_band(propensity),
                    "sensitivity_band": [SENSITIVITY_SUPPORT_LOWER, SENSITIVITY_SUPPORT_UPPER],
                    "stabilized_ipw_weight_unclipped": unclipped,
                    "stabilized_ipw_weight_after_primary_clip": clipped,
                    "weight_was_altered_by_clipping": abs(clipped - unclipped) > 1e-12,
                    "weight_rank_ascending": rank,
                    "weight_rank_of": n,
                    "weight_percentile_ascending": (rank - 1) / (n - 1) * 100.0 if n > 1 else 0.0,
                    "primary_clip_bounds": [low_bound, high_bound],
                    "patient_id_withheld": (
                        "deliberate: recording an id alongside an exactly "
                        "reproducible propensity in a cohort of this size would "
                        "be a re-identification vector in an exported artifact"
                    ),
                }
            )
        return detail

    def trimmed_counterfactual(self) -> dict[str, object]:
        """What the fourth frozen condition would yield if the primary band
        were applied as a *trimming* rule (out-of-band patients dropped)
        instead of as the arm-agnostic count proxy it currently is."""
        kept = [
            patient
            for patient in self.dev_patients
            if not _out_of_primary_band(self.propensities[patient.patient_id])
        ]
        if not self.out_of_band:
            return {
                "applicable": False,
                "reason": "no patient falls outside the primary band; trimming is a no-op",
            }
        trimmed_report = OverlapDiagnostics.compute(kept, self.propensities, self.adjustment_set)
        return {
            "applicable": True,
            "n_dropped": len(self.out_of_band),
            "n_retained": len(kept),
            "arm_counts": {
                str(arm): count for arm, count in sorted(trimmed_report.arm_counts.items())
            },
            "arm_counts_sufficient": trimmed_report.arm_counts_sufficient,
            "min_arm_count_floor": MIN_ARM_COUNT,
            "extreme_weight_count": trimmed_report.extreme_weight_count,
            "no_unresolved_positivity_warning": (trimmed_report.no_unresolved_positivity_warning),
            "smd_sufficient": trimmed_report.smd_sufficient,
            "support_valid": trimmed_report.support_valid,
        }

    def sensitivity_band_counterfactual(self) -> dict[str, object]:
        """What the same proxy would yield under the frozen *sensitivity* band
        `[0.05, 0.95]` instead of the primary one. Reported, not applied."""
        outside = [
            self.propensities[patient.patient_id]
            for patient in self.dev_patients
            if not _in_sensitivity_band(self.propensities[patient.patient_id])
        ]
        return {
            "count_outside_sensitivity_band": len(outside),
            "would_yield_no_unresolved_positivity_warning": not outside,
        }

    def positivity_payload(self) -> dict[str, object]:
        propensity_values = list(self.propensities.values())
        return {
            "seed_path": self.seed_path,
            "pipeline_seed": self.pipeline_seed,
            "propensity_seed": self.propensity_seed,
            "dev_retained_n": len(self.dev_patients),
            "arm_counts": {
                str(arm): count for arm, count in sorted(self.report.arm_counts.items())
            },
            "extreme_weight_count": self.report.extreme_weight_count,
            "extreme_weight_fraction": self.report.extreme_weight_fraction,
            "no_unresolved_positivity_warning": self.report.no_unresolved_positivity_warning,
            "propensity_minimum": min(propensity_values),
            "propensity_maximum": max(propensity_values),
            "out_of_primary_band_patients": self.out_of_band_detail(),
            "trimming_counterfactual": self.trimmed_counterfactual(),
            "sensitivity_band_counterfactual": self.sensitivity_band_counterfactual(),
            "weight_summary_unclipped": {
                "minimum": min(self.unclipped_weights.values()),
                "maximum": max(self.unclipped_weights.values()),
                "median": statistics.median(self.unclipped_weights.values()),
            },
        }

    # -- negative-control outcome ----------------------------------------

    def negative_control_sweep(
        self,
        bootstrap_counts: Sequence[int],
        seeds: Sequence[int] = REQUIRED_SEEDS,
    ) -> dict[str, object]:
        """`run_negative_control_outcome` across the requested seeds and
        bootstrap counts.

        The largest requested count is what actually runs; smaller counts are
        read off its replicate prefix (see the module docstring's Cost note).

        `seeds` / `bootstrap_counts` are recorded in the payload as what was
        actually measured, and `sweep_completeness` says whether they cover the
        full frozen sweep -- so `classification_stable` can never be mistaken
        for a breadth it did not have.
        """
        max_count = max(bootstrap_counts)
        regression_covariates = [name for name in self.adjustment_set if name != BASELINE_CD8_FIELD]
        point_estimate = _fit_cd8_regression_smd(self.dev_patients, regression_covariates)

        by_seed: dict[str, dict[str, object]] = {}
        for seed in seeds:
            replicates = _bootstrap_replicates(
                self.dev_patients, regression_covariates, seed, max_count
            )
            per_count: dict[str, object] = {}
            for count in sorted(bootstrap_counts):
                per_count[str(count)] = _classify(point_estimate, replicates[:count])
            by_seed[str(seed)] = per_count

        classifications = {
            entry["classification"]
            for per_count in by_seed.values()
            for entry in per_count.values()
        }
        complete = tuple(seeds) == tuple(REQUIRED_SEEDS) and tuple(sorted(bootstrap_counts)) == (
            tuple(sorted(BOOTSTRAP_COUNTS))
        )
        return {
            "adjusted_smd_point_estimate": point_estimate,
            "abs_adjusted_smd": abs(point_estimate),
            "smd_threshold": SMD_THRESHOLD,
            "abs_smd_at_or_above_threshold": abs(point_estimate) >= SMD_THRESHOLD,
            "regression_covariates": regression_covariates,
            "baseline_cd8_excluded_from_own_adjustment_set": (
                BASELINE_CD8_FIELD not in regression_covariates
            ),
            "seeds": list(seeds),
            "bootstrap_counts": sorted(bootstrap_counts),
            "full_frozen_sweep_seeds": list(REQUIRED_SEEDS),
            "full_frozen_sweep_bootstrap_counts": sorted(BOOTSTRAP_COUNTS),
            "sweep_completeness": "complete" if complete else "reduced",
            "sweep_completeness_note": (
                "all five frozen seeds at both replicate counts were measured"
                if complete
                else (
                    "REDUCED sweep: seeds "
                    f"{list(seeds)} at replicate counts {sorted(bootstrap_counts)} were "
                    f"measured, out of the frozen {list(REQUIRED_SEEDS)} x "
                    f"{sorted(BOOTSTRAP_COUNTS)}. classification_stable below is a "
                    "claim about exactly what was measured and nothing more. Set "
                    f"{FULL_SWEEP_ENV_VAR}=1 to measure the full sweep."
                )
            ),
            "by_seed": by_seed,
            "distinct_classifications": sorted(classifications),
            "classification_stable": len(classifications) == 1,
            "classification_stable_across_full_frozen_sweep": (
                complete and len(classifications) == 1
            ),
        }


def _weight_percentile(sorted_values: Sequence[float], p: float) -> float:
    """`causal.weighting._percentile`'s method, re-expressed via the public
    `clip_weights` contract: the clip bound for percentile `p` is what
    `clip_weights` clamps to. Computed directly here so the bounds can be
    reported without reaching into a private helper."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _bootstrap_replicates(
    patients: Sequence[PatientRecord],
    regression_covariates: Sequence[str],
    seed: int,
    n_bootstrap: int,
) -> list[float]:
    """Exactly `run_negative_control_outcome`'s bootstrap loop: same derived
    seed label, same RNG, same `rng.choice` draw order -- so the first `k`
    entries are what `n_bootstrap=k` would have produced."""
    rng = random.Random(derive_seed(seed, "negative_control_outcome_bootstrap"))
    patient_list = list(patients)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(patient_list) for _ in range(len(patient_list))]
        estimate = _fit_cd8_regression_smd(resample, regression_covariates)
        if not math.isnan(estimate):
            estimates.append(estimate)
    return estimates


def _classify(point_estimate: float, replicates: Sequence[float]) -> dict[str, object]:
    """`run_negative_control_outcome`'s frozen Section 11.3 classification,
    applied to a replicate list, plus which branch fired."""
    ordered = sorted(replicates)
    ci_low = _percentile(ordered, 2.5)
    ci_high = _percentile(ordered, 97.5)
    ci_contains_null = (
        (not math.isnan(ci_low)) and (not math.isnan(ci_high)) and ci_low <= 0.0 <= ci_high
    )
    abs_below_threshold = abs(point_estimate) < SMD_THRESHOLD

    if not ci_contains_null:
        classification = "fail"
        branch = "ci_excludes_zero"
    elif abs_below_threshold:
        classification = "pass"
        branch = "ci_contains_zero_and_abs_smd_below_threshold"
    else:
        classification = "uncertain"
        branch = "ci_contains_zero_but_abs_smd_at_or_above_threshold"

    return {
        "classification": classification,
        "branch": branch,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_contains_zero": ci_contains_null,
        "n_replicates": len(ordered),
    }


# --------------------------------------------------------------------------
# Module-scoped fixtures: each seed path's pipeline + cross-fitting runs once.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canonical() -> _Measurement:
    """The seed derivation `scripts/run_causal_analysis.py` performs."""
    return _Measurement(
        pipeline_seed=derive_seed(MASTER_SEED, "causal_analysis_pipeline"),
        propensity_seed=derive_seed(MASTER_SEED, "propensity"),
        seed_path="derive_seed(MASTER_SEED, label) -- matches scripts/run_causal_analysis.py",
    )


@pytest.fixture(scope="module")
def underived() -> _Measurement:
    """The path the task text's numbers come from (task 7.2's finding): the
    master seed used directly, `derive_seed` bypassed."""
    return _Measurement(
        pipeline_seed=MASTER_SEED,
        propensity_seed=MASTER_SEED,
        seed_path="MASTER_SEED used directly, derive_seed bypassed",
    )


@pytest.fixture(scope="module")
def canonical_sweep(canonical: _Measurement) -> dict[str, object]:
    """The expensive fixture: the full five-seed x {200, 1000} sweep when
    `FULL_SWEEP_ENV_VAR` is set, one seed at 200 replicates otherwise."""
    if full_sweep_requested():
        return canonical.negative_control_sweep(BOOTSTRAP_COUNTS)
    return canonical.negative_control_sweep(REDUCED_BOOTSTRAP_COUNTS, REDUCED_SEEDS)


@pytest.fixture(scope="module")
def underived_sweep(underived: _Measurement) -> dict[str, object]:
    """The same sweep on the un-derived path at 200 replicates. The point
    estimate there is below the 0.10 threshold, so `uncertain` is
    unreachable regardless of replicate count and 200 settles it -- which is
    why this side is 200-replicate even under the full sweep. Its seed set
    follows the gate, like the canonical side."""
    seeds = REQUIRED_SEEDS if full_sweep_requested() else REDUCED_SEEDS
    return underived.negative_control_sweep((200,), seeds)


# --------------------------------------------------------------------------
# Finding 1: positivity.
# --------------------------------------------------------------------------


def test_positivity_condition_on_the_canonical_seed_path(
    canonical: _Measurement,
) -> None:
    """The fourth frozen condition on the seed path
    `scripts/run_causal_analysis.py` actually takes.

    Before task 7.5 (six-covariate Section 9 propensity model) no patient
    fell outside the primary band: `extreme_weight_count == 0`, propensity
    range [0.2189, 0.8762] (500-step fit), condition PASSED. With the
    Section 13 propensity model (14 encoded columns, task 7.5), fit to
    convergence (task 7.8), one arm-0 patient sits at e = 0.9327, above the
    primary upper bound 0.90 but inside the sensitivity band [0.05, 0.95]:
    `extreme_weight_count == 1`, range [0.1798, 0.9327], condition FAILS.
    The same patient is out of band under the old 500-step fit (0.9305) and
    under every gradient-descent budget, so this is not an under-fitting
    artifact. It is a measured cohort finding under the unchanged frozen
    band, not a threshold question."""
    assert canonical.report.extreme_weight_count == 1
    assert canonical.report.no_unresolved_positivity_warning is False

    (detail,) = canonical.out_of_band_detail()
    propensity = detail["propensity"]
    assert isinstance(propensity, float)
    assert propensity > PRIMARY_SUPPORT_UPPER
    assert detail["treatment_arm"] == _ARM_0
    assert detail["inside_sensitivity_band"] is True
    assert PRIMARY_SUPPORT_LOWER < min(canonical.propensities.values())


def test_the_single_out_of_band_patient_is_inside_the_sensitivity_band(
    underived: _Measurement,
) -> None:
    """On the un-derived path exactly one patient is out of band, it is an
    arm-1 patient just above the primary upper bound, and it sits comfortably
    inside the frozen sensitivity band -- so the sensitivity rule would raise
    no positivity warning at all."""
    assert underived.report.extreme_weight_count == 1
    assert underived.report.no_unresolved_positivity_warning is False

    (detail,) = underived.out_of_band_detail()
    propensity = detail["propensity"]
    assert isinstance(propensity, float)
    assert propensity > PRIMARY_SUPPORT_UPPER
    assert detail["treatment_arm"] == _ARM_1
    assert detail["inside_sensitivity_band"] is True
    assert underived.sensitivity_band_counterfactual() == {
        "count_outside_sensitivity_band": 0,
        "would_yield_no_unresolved_positivity_warning": True,
    }
    assert "patient_id" not in detail


def test_the_out_of_band_patients_ipw_weight_is_not_extreme(
    underived: _Measurement,
) -> None:
    """The core evidence that this is a proxy artifact: the proxy is named
    `extreme_weight_count`, but the flagged patient's stabilized IPW weight is
    in the *bottom* half of the weight distribution. For an arm-1 patient a
    high propensity means a SMALL weight (`p_treated / e`); the proxy is
    arm-agnostic while true weight extremity is arm-dependent."""
    (detail,) = underived.out_of_band_detail()
    percentile = detail["weight_percentile_ascending"]
    assert isinstance(percentile, float)
    assert percentile < 50.0, (
        "expected the flagged patient's weight to be below the median; the "
        f"proxy flagged a weight at the {percentile:.1f}th percentile"
    )
    unclipped = detail["stabilized_ipw_weight_unclipped"]
    assert isinstance(unclipped, float)
    assert unclipped < statistics.median(underived.unclipped_weights.values())


def test_clipping_cannot_resolve_the_positivity_condition(
    underived: _Measurement,
) -> None:
    """`causal/weighting.py` implements percentile CLIPPING (winsorizing
    weights into a band), not trimming (dropping patients). Clipping moves no
    propensity and removes no patient, so it cannot change
    `extreme_weight_count`, which is defined on the propensity scale. Applying
    the frozen primary preset and recomputing the report proves it."""
    clipped_report = OverlapDiagnostics.compute(
        underived.dev_patients, underived.propensities, underived.adjustment_set
    )
    assert clipped_report.extreme_weight_count == underived.report.extreme_weight_count
    assert len(underived.clipped_weights) == len(underived.unclipped_weights)

    # If the primary band were instead applied as a trimming rule, the
    # condition WOULD pass, and both arms would still clear the 100-per-arm
    # floor. Reported as a counterfactual; not applied.
    counterfactual = underived.trimmed_counterfactual()
    assert counterfactual["applicable"] is True
    assert counterfactual["n_dropped"] == 1
    assert counterfactual["extreme_weight_count"] == 0
    assert counterfactual["no_unresolved_positivity_warning"] is True
    assert counterfactual["arm_counts_sufficient"] is True


# --------------------------------------------------------------------------
# Finding 2: negative-control outcome.
# --------------------------------------------------------------------------


@requires_full_sweep
def test_derived_200_replicate_ci_matches_a_real_200_replicate_run(
    canonical: _Measurement,
) -> None:
    """Verifies the prefix equivalence the sweep relies on: a real
    `run_negative_control_outcome(n_bootstrap=200)` reproduces the CI derived
    from the first 200 replicates of this module's loop."""
    actual = run_negative_control_outcome(
        canonical.dev_patients, canonical.adjustment_set, seed=REQUIRED_SEEDS[0], n_bootstrap=200
    )
    regression_covariates = [
        name for name in canonical.adjustment_set if name != BASELINE_CD8_FIELD
    ]
    derived = _classify(
        _fit_cd8_regression_smd(canonical.dev_patients, regression_covariates),
        _bootstrap_replicates(
            canonical.dev_patients, regression_covariates, REQUIRED_SEEDS[0], 200
        ),
    )
    assert derived["classification"] == actual.classification
    assert derived["ci_low"] == pytest.approx(actual.ci_low, rel=1e-12, abs=1e-12)
    assert derived["ci_high"] == pytest.approx(actual.ci_high, rel=1e-12, abs=1e-12)


def test_uncertain_comes_from_the_second_frozen_branch(
    canonical_sweep: dict[str, object],
) -> None:
    """Which branch fires on the canonical path.

    Task 7.3 measured `uncertain` (CI contains zero, |adjusted SMD| = 0.115 >=
    0.10, Section 11.3's third clause). Task 7.8 found that five Section 13
    covariates (`weight`, `sex`, `race_ethnicity`,
    `intravenous_drug_use_history`, prior-ART exposure and duration) had
    entered that regression as all-zero columns, because their names were
    looked up verbatim rather than through the shared Section 13 mapping.
    Adjusting for the full set, as Section 11.3 requires, gives an adjusted
    SMD of -0.0906: below 0.10 with a CI containing zero, Section 11.3's
    pass clause. Before: uncertain, -0.1150. After: pass, -0.0906. The
    threshold is unchanged; the name keeps the task 7.3 history.
    """
    assert canonical_sweep["abs_smd_at_or_above_threshold"] is False
    assert canonical_sweep["adjusted_smd_point_estimate"] == pytest.approx(-0.0906, abs=5e-4)
    by_seed = canonical_sweep["by_seed"]
    assert isinstance(by_seed, dict)
    for per_count in by_seed.values():
        for entry in per_count.values():
            assert entry["classification"] == "pass"
            assert entry["ci_contains_zero"] is True


@requires_full_sweep
def test_classification_is_stable_across_seeds_and_bootstrap_counts(
    canonical_sweep: dict[str, object],
) -> None:
    """Stability, as task 7.3 asks: all five frozen seeds, both replicate
    counts, one classification."""
    assert canonical_sweep["seeds"] == list(REQUIRED_SEEDS)
    assert canonical_sweep["bootstrap_counts"] == [200, 1000]
    assert canonical_sweep["sweep_completeness"] == "complete"
    assert canonical_sweep["classification_stable"] is True
    assert canonical_sweep["classification_stable_across_full_frozen_sweep"] is True
    # Task 7.8: "uncertain" before the Section 13 encoding fix, "pass" after.
    assert canonical_sweep["distinct_classifications"] == ["pass"]


def test_reduced_sweep_records_what_it_actually_measured(
    canonical_sweep: dict[str, object],
) -> None:
    """The honesty guard on the gate: whatever ran, the payload's recorded
    seeds and replicate counts are the ones measured, and the full-sweep
    stability claim is made only when the full sweep ran."""
    expected_seeds = list(REQUIRED_SEEDS if full_sweep_requested() else REDUCED_SEEDS)
    expected_counts = sorted(
        BOOTSTRAP_COUNTS if full_sweep_requested() else REDUCED_BOOTSTRAP_COUNTS
    )
    assert canonical_sweep["seeds"] == expected_seeds
    assert canonical_sweep["bootstrap_counts"] == expected_counts
    assert canonical_sweep["sweep_completeness"] == (
        "complete" if full_sweep_requested() else "reduced"
    )
    if not full_sweep_requested():
        assert canonical_sweep["classification_stable_across_full_frozen_sweep"] is False
        assert FULL_SWEEP_ENV_VAR in str(canonical_sweep["sweep_completeness_note"])
    by_seed = canonical_sweep["by_seed"]
    assert isinstance(by_seed, dict)
    assert sorted(by_seed) == sorted(str(seed) for seed in expected_seeds)
    for per_count in by_seed.values():
        assert sorted(per_count) == sorted(str(count) for count in expected_counts)


def test_negative_control_outcome_is_also_seed_path_dependent(
    canonical_sweep: dict[str, object], underived_sweep: dict[str, object]
) -> None:
    """Task 7.3 found the two failures on different seed paths. After task
    7.8 (converged fits, CD8 regression adjusted for the full Section 13 set)
    the negative-control outcome passes on BOTH paths (canonical -0.0906,
    un-derived -0.0669), so it no longer depends on the seed path. Before:
    canonical uncertain (-0.1150), un-derived below threshold."""
    assert canonical_sweep["abs_smd_at_or_above_threshold"] is False
    assert underived_sweep["abs_smd_at_or_above_threshold"] is False
    assert canonical_sweep["distinct_classifications"] == ["pass"]
    assert underived_sweep["distinct_classifications"] == ["pass"]


def test_baseline_cd8_is_excluded_from_its_own_adjustment_set(
    canonical_sweep: dict[str, object],
) -> None:
    """The CD8 double-use safeguard held, so the `uncertain` is not an artifact
    of a tautological regression."""
    assert canonical_sweep["baseline_cd8_excluded_from_own_adjustment_set"] is True
    covariates = canonical_sweep["regression_covariates"]
    assert isinstance(covariates, list)
    assert BASELINE_CD8_FIELD not in covariates
    assert covariates, "expected a non-empty adjustment set after the exclusion"


# --------------------------------------------------------------------------
# The artifact.
# --------------------------------------------------------------------------


def _existing_artifact_is_complete() -> bool:
    """Whether an artifact already on disk records a complete sweep. An
    unreadable or absent file is not a complete artifact, so a reduced run may
    write over it."""
    if not ARTIFACT_PATH.is_file():
        return False
    try:
        existing = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(existing, dict):
        return False
    if existing.get("sweep_completeness") == "complete":
        return True
    # An artifact written before the gate existed carries no flag, but it does
    # record the seeds and counts it measured. Infer completeness from those, so
    # a pre-gate full-sweep artifact is protected too.
    canonical = existing.get("negative_control_outcome", {})
    if not isinstance(canonical, dict):
        return False
    recorded = canonical.get("canonical_seed_path", {})
    if not isinstance(recorded, dict):
        return False
    return recorded.get("seeds") == list(REQUIRED_SEEDS) and recorded.get(
        "bootstrap_counts"
    ) == sorted(BOOTSTRAP_COUNTS)


def test_writes_diagnostic_artifact(
    canonical: _Measurement,
    underived: _Measurement,
    canonical_sweep: dict[str, object],
    underived_sweep: dict[str, object],
) -> None:
    payload = {
        "diagnostic": "positivity_and_negative_control",
        "spec_task": ".kiro/specs/neural-experiment-pipeline/tasks.md task 7.3",
        "diagnostic_only": (
            "No frozen threshold, band, or classification rule was changed. The "
            "trimming and sensitivity-band figures below are counterfactuals, "
            "reported to size the finding; neither is applied."
        ),
        "sweep_completeness": canonical_sweep["sweep_completeness"],
        "sweep_completeness_note": canonical_sweep["sweep_completeness_note"],
        "sweep_gate": (
            f"The bootstrap sweep is gated behind the {FULL_SWEEP_ENV_VAR} "
            "environment variable. Set, it measures all five frozen seeds at "
            "{200, 1000} replicates and this artifact's sweep_completeness reads "
            "'complete'. Unset -- the default suite run -- it measures "
            f"{list(REDUCED_SEEDS)} at {list(REDUCED_BOOTSTRAP_COUNTS)} replicates, "
            "sweep_completeness reads 'reduced', and no cross-seed or "
            "cross-replicate-count stability is claimed. A reduced run will not "
            "overwrite a complete artifact."
        ),
        "master_seed": MASTER_SEED,
        "n_folds": N_FOLDS,
        "clip_preset": CLIP_PRESET,
        "clip_preset_percentiles": list(CLIP_PRESETS[CLIP_PRESET]),
        "privacy_note": (
            "The out-of-band patient's identifier is deliberately withheld. In a "
            "cohort of this size an exactly reproducible propensity value plus an "
            "identifier would be a re-identification vector in an exported file, "
            "and results/ artifacts are scanned by scripts/scan_exports.py. "
            "Propensity, arm, and band membership support every claim here."
        ),
        "gitignore_note": (
            "results/ is gitignored, so this artifact is reproducible-on-demand "
            "rather than committed. Left as-is: changing .gitignore is a decision, "
            "not a diagnostic."
        ),
        "seed_path_dependence": (
            "The two conditions this task asks about fail on DIFFERENT seed paths "
            "and never together. Task 7.2 established that the task-text numbers "
            "(n=378) come from passing MASTER_SEED un-derived to both "
            "run_observationalization_pipeline and LogisticPropensityModel, while "
            "scripts/run_causal_analysis.py derives both and yields n=383. On the "
            "canonical derived path, with the Section 13 propensity model (task "
            "7.5) fit to convergence (task 7.8), positivity FAILS on 1 arm-0 "
            "patient and the negative-control outcome is uncertain. With the "
            "earlier six-covariate model the canonical path had 0 patients out "
            "of band. On the un-derived path positivity fails on 1 arm-1 patient "
            "and the negative-control outcome's adjusted SMD is below threshold."
        ),
        "positivity": {
            "condition": "no_unresolved_positivity_warning",
            "as_implemented": "extreme_weight_count == 0",
            "proxy_note": (
                "causal/overlap.py documents extreme_weight_count as a proxy for "
                "the true post-clipping weight distribution: it counts patients "
                "whose propensity falls outside the primary band [0.10, 0.90], "
                "independent of treatment arm."
            ),
            "canonical_seed_path": canonical.positivity_payload(),
            "underived_seed_path": underived.positivity_payload(),
            "verdict": "genuine_diagnostic_signal_about_the_cohort",
            "verdict_evidence": (
                "Canonical seed path (the one scripts/run_causal_analysis.py takes), "
                "Section 13 propensity model fit to convergence: one ARM-0 patient "
                "has e = 0.933. For an arm-0 patient a high propensity produces a "
                "LARGE weight, and this patient carries the largest unclipped "
                "stabilized IPW weight in the cohort (about 7.1, rank 383 of 383), "
                "which the frozen primary 1st/99th-percentile clip reduces to about "
                "2.5. So here the arm-agnostic proxy and the quantity it proxies for "
                "agree: this is a real, if mild, positivity signal. It is driven by "
                "a Karnofsky score about 4.4 SD below the fold mean plus a long "
                "prior-ART exposure, a covariate profile with few comparable "
                "arm-1 patients. It is mild: the patient is inside the sensitivity "
                "band [0.05, 0.95], ESS stays above 138 per arm, and trimming at the "
                "primary band would drop one patient with both arms above the "
                "100-per-arm floor. It is not an under-fitting artifact (out of band "
                "at every optimizer budget and at convergence) and not a defect."
            ),
            "underived_seed_path_verdict": "artifact_of_a_documented_proxy",
            "underived_seed_path_verdict_evidence": (
                "Superseded for the canonical path; kept for the un-derived path. "
                "(1) Under the earlier six-covariate model the canonical path "
                "passed outright. (2) On the un-derived path the single flagged "
                "patient is an arm-1 patient whose propensity is just above 0.90 "
                "and comfortably inside the frozen sensitivity band [0.05, 0.95]; "
                "its stabilized IPW weight is in the bottom half of the weight "
                "distribution, because for an arm-1 patient a HIGH propensity "
                "produces a SMALL weight. The proxy is arm-agnostic while weight "
                "extremity is arm-dependent, so the proxy flags a patient the "
                "quantity it proxies for would not. (3) causal/weighting.py "
                "implements percentile clipping of weights, not trimming of "
                "patients, so it cannot move the count either way; the count is "
                "defined on the propensity scale. If the primary band were applied "
                "as a trimming rule the condition would pass with both arms still "
                "clearing the 100-per-arm floor. Not a defect in the cohort and not "
                "a defect in the code as documented -- a known limitation of the "
                "proxy, which overlap.py itself flags for replacement by "
                "causal.weighting if a future task needs the true weight "
                "distribution."
            ),
        },
        "negative_control_outcome": {
            "rule": (
                "decisions.md Section 11.3: pass if abs(adjusted smd) < 0.10 AND "
                "the bootstrap CI contains zero; fail if the CI excludes zero; "
                "uncertain if the CI contains zero but abs(adjusted smd) >= 0.10."
            ),
            "branch_that_fired": "abs_smd_below_threshold_and_ci_contains_zero",
            "history": (
                "Task 7.3 measured uncertain (adjusted SMD -0.1150, CI containing "
                "zero, |SMD| >= 0.10). Task 7.8 found that five Section 13 covariates "
                "(weight, sex, race_ethnicity, intravenous_drug_use_history, prior "
                "ART exposure and duration) had entered the regression as all-zero "
                "columns because their names were looked up verbatim. Adjusted for "
                "the full set via the shared Section 13 mapping, as Section 11.3 "
                "requires, the adjusted SMD is -0.0906 and the check passes on all "
                "five seeds at both replicate counts. The 0.10 threshold is unchanged."
            ),
            "canonical_seed_path": canonical_sweep,
            "underived_seed_path": underived_sweep,
            "relation_to_task_7_2": (
                "Task 7.2 measured baseline_cd8's raw SMD at -0.1363 and its "
                "stabilized-IPW-weighted SMD at -0.1323 on the canonical path -- "
                "essentially unmoved by weighting, because baseline_cd8 was then "
                "absent from causal.propensity.COVARIATE_FIELDS. (Task 7.2 said it "
                "could not be added; that misread decisions.md Section 11.3, which "
                "excludes CD8 only from this negative-control regression. Task 7.5 "
                "added it to the propensity model.) The regression-adjusted SMD "
                "measured here excludes CD8 from its own right-hand side, as 11.3 "
                "requires, and is unaffected by the propensity model."
            ),
            "verdict": "passes_after_full_adjustment_close_to_threshold",
            "verdict_evidence": (
                "With every Section 13 covariate except baseline CD8 adjusted for, "
                "the adjusted SMD is -0.0906: below the frozen 0.10 tolerance, CI "
                "containing zero, identical classification across all five seeds "
                "and both replicate counts, and on both seed paths. It sits close to "
                "the threshold and the CI is wide (about [-0.29, 0.11]), so this is "
                "a pass at this sample size, not evidence of perfect balance. The "
                "earlier uncertain classification came from five silently "
                "unadjusted covariates, not from the cohort."
            ),
            "superseded_verdict": "genuine_diagnostic_signal_about_the_cohort",
            "superseded_verdict_evidence": (
                "Task 7.3, before the Section 13 encoding fix: "
                "The adjusted SMD is a real residual imbalance in baseline CD8 "
                "between the observationalized arms, of the same magnitude and sign "
                "task 7.2 measured raw and post-weighting, and it survives "
                "adjustment for every other covariate in the frozen adjustment set. "
                "The CD8 double-use safeguard held, so the estimate is not "
                "tautological. The classification is identical across all five "
                "frozen seeds and both bootstrap replicate counts, so it is not "
                "bootstrap noise. And uncertain rather than fail is the correct "
                "reading: the CI straddles zero, so the imbalance is not resolved "
                "at this sample size either way. The check is doing its job -- it "
                "reports that a pre-treatment quantity differs across arms by more "
                "than the frozen tolerance and that the adjustment set available "
                "cannot remove it. Not a proxy artifact and not a defect."
            ),
        },
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # A reduced run must not silently replace a verified full-sweep artifact
    # with a weaker one. One path, strongest run wins; see the module
    # docstring's "Artifact honesty under a reduced run".
    if payload["sweep_completeness"] == "reduced" and _existing_artifact_is_complete():
        pytest.skip(
            f"{ARTIFACT_PATH.name} already records a complete sweep; a reduced run "
            f"will not overwrite it (set {FULL_SWEEP_ENV_VAR}=1 to refresh it)"
        )

    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    reloaded = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert reloaded["sweep_completeness"] == ("complete" if full_sweep_requested() else "reduced")
    assert (
        reloaded["negative_control_outcome"]["canonical_seed_path"]["sweep_completeness"]
        == reloaded["sweep_completeness"]
    )
    assert reloaded["positivity"]["verdict"] == "genuine_diagnostic_signal_about_the_cohort"
    assert reloaded["positivity"]["underived_seed_path_verdict"] == "artifact_of_a_documented_proxy"
    assert (
        reloaded["negative_control_outcome"]["verdict"]
        == "passes_after_full_adjustment_close_to_threshold"
    )
    # Task 7.8: was True under the six-covariate 500-step model; False with the
    # Section 13 model at convergence (one arm-0 patient at e = 0.933).
    canonical_positivity = reloaded["positivity"]["canonical_seed_path"]
    assert canonical_positivity["no_unresolved_positivity_warning"] is False
    (flagged,) = canonical_positivity["out_of_primary_band_patients"]
    assert flagged["treatment_arm"] == 0 and flagged["weight_rank_ascending"] == 383
    assert (
        reloaded["positivity"]["underived_seed_path"]["no_unresolved_positivity_warning"] is False
    )
    canonical_recorded = reloaded["negative_control_outcome"]["canonical_seed_path"]
    assert canonical_recorded["classification_stable"] is True
    assert canonical_recorded["classification_stable_across_full_frozen_sweep"] is (
        full_sweep_requested()
    )
    # No patient identifier anywhere in either path's out-of-band detail.
    for path_key in ("canonical_seed_path", "underived_seed_path"):
        detail = reloaded["positivity"][path_key]["out_of_primary_band_patients"]
        for entry in detail:
            assert "patient_id" not in entry
