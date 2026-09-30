"""Frozen-value regression tests (Requirement: Correctness Property 8;
`design.md` Section "Correctness Properties", Property 8: "Frozen-value
consistency").

This file is a mechanical backstop, not a behavioral test suite: it does
not exercise any pipeline logic. It only imports the actual module-level
constants (and, where a frozen value is a default parameter rather than a
named constant, reads that default via `inspect.signature`) already
defined throughout `src/`, and asserts each one still equals the value
frozen in `decisions.md`.

If any assertion here ever fails, it means either:

- a module-level constant was edited (accidentally or intentionally) so it
  no longer matches `decisions.md`, or
- this test's understanding of `decisions.md` is stale.

Either way, `decisions.md` is the single source of truth (see that
document's header: "Status: Frozen. Changes after this point require a
documented amendment...") -- a failure here must be resolved by a human,
never by silently updating the assertion to match whatever the code
currently does.

Tests are grouped one-per-section (or one-parametrized-test-per-section),
each with a comment citing the `decisions.md` section it protects.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

#: `decisions.md` itself, read by the amendment-text assertions below (the
#: only tests here that compare code against the document's *prose* rather
#: than against a value transcribed from it).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DECISIONS_PATH = REPO_ROOT / ".kiro" / "specs" / "causal-witness-pwm" / "decisions.md"

#: `scripts/` is not an installed package (it holds CLI entry points), so it
#: is added to `sys.path` directly -- the established convention in
#: `tests/unit/test_scan_exports.py` and `tests/unit/test_check_go_no_go.py`.
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _amendment_log_text() -> str:
    """Return the text of `decisions.md` Section 26 (the amendment log)."""
    text = DECISIONS_PATH.read_text(encoding="utf-8")
    heading = "## 26. Amendment log"
    assert heading in text, f"{DECISIONS_PATH} has no {heading!r} section"
    return text.split(heading, 1)[1]


# ---------------------------------------------------------------------------
# decisions.md Section 9 -- risk-score coefficients and selection-logit
# intercept.
# ---------------------------------------------------------------------------


def test_section_9_risk_score_coefficients_and_intercept():
    """`decisions.md` Section 9:

        r_i = -0.8*z(CD4) - 0.3*z(Karnofsky) + 0.3*z(Age)
              + 0.6*Symptomatic + 0.6*PriorOI + 0.4*PriorART

        P(S_i=1|A_i,X_i) = sigmoid(1.2 + (2*A_i - 1)*r_i)
    """
    from cohorts import selection

    assert selection.COEF_CD4 == -0.8
    assert selection.COEF_KARNOFSKY == -0.3
    assert selection.COEF_AGE == 0.3
    assert selection.COEF_SYMPTOMATIC == 0.6
    assert selection.COEF_PRIOR_OI == 0.6
    assert selection.COEF_PRIOR_ART == 0.4
    assert selection.SELECTION_LOGIT_INTERCEPT == 1.2

    # DEFAULT_COEFFICIENTS must be built from exactly these module-level
    # constants, not an independently-drifted copy.
    assert selection.DEFAULT_COEFFICIENTS == selection.SelectionCoefficients(
        coef_cd4=-0.8,
        coef_karnofsky=-0.3,
        coef_age=0.3,
        coef_symptomatic=0.6,
        coef_prior_oi=0.6,
        coef_prior_art=0.4,
        selection_logit_intercept=1.2,
    )


# ---------------------------------------------------------------------------
# decisions.md Section 26 amendment (2026-09-07, amending Sections 9/20) --
# the stronger-observational-selection coefficients used by the
# `stronger_selection_severity` sensitivity variant.
# ---------------------------------------------------------------------------


def test_section_26_stronger_selection_coefficients_and_intercept():
    """`decisions.md` Section 26 amendment dated 2026-09-07:

        r_stronger_i = -1.2*z(CD4) - 0.45*z(Karnofsky) + 0.45*z(Age)
                       + 0.9*Symptomatic + 0.9*PriorOI + 0.6*PriorART

        P(S_i=1|A_i,X_i) = sigmoid(1.2 + (2*A_i - 1)*r_stronger_i)

    i.e. every primary risk-score coefficient scaled by exactly 1.5, with
    the selection-logit intercept deliberately unchanged at 1.2.
    """
    from cohorts import selection

    assert selection.STRONGER_COEF_CD4 == -1.2
    assert selection.STRONGER_COEF_KARNOFSKY == -0.45
    assert selection.STRONGER_COEF_AGE == 0.45
    assert selection.STRONGER_COEF_SYMPTOMATIC == 0.9
    assert selection.STRONGER_COEF_PRIOR_OI == 0.9
    assert selection.STRONGER_COEF_PRIOR_ART == 0.6
    assert selection.STRONGER_SELECTION_LOGIT_INTERCEPT == 1.2

    # The bundle must be built from exactly those constants.
    assert selection.STRONGER_SELECTION_COEFFICIENTS == selection.SelectionCoefficients(
        coef_cd4=-1.2,
        coef_karnofsky=-0.45,
        coef_age=0.45,
        coef_symptomatic=0.9,
        coef_prior_oi=0.9,
        coef_prior_art=0.6,
        selection_logit_intercept=1.2,
    )


def test_section_26_stronger_severity_is_primary_scaled_by_one_point_five():
    """The amendment defines the stronger mechanism *relationally*: the
    primary Section 9 risk-score coefficients scaled by exactly 1.5, with
    the intercept left unchanged. Asserting the relation as well as the
    literals catches an edit that changes only one of the two.

    The relation is checked with a tolerance because the frozen literals are
    the exact decimal values the amendment records (e.g. `-1.2`), while
    `1.5 * -0.8` in binary floating point is `-1.2000000000000002`. The
    literals are the frozen values; the relation is why they are those
    values.
    """
    from cohorts import selection

    primary = selection.DEFAULT_COEFFICIENTS
    stronger = selection.STRONGER_SELECTION_COEFFICIENTS

    for field_name in (
        "coef_cd4",
        "coef_karnofsky",
        "coef_age",
        "coef_symptomatic",
        "coef_prior_oi",
        "coef_prior_art",
    ):
        assert getattr(stronger, field_name) == pytest.approx(
            1.5 * getattr(primary, field_name)
        ), field_name

    assert stronger.selection_logit_intercept == primary.selection_logit_intercept


def test_section_26_stronger_severity_reachable_through_severity_path():
    """The amendment's coefficients must be reachable by severity name, and
    the primary path must remain the default -- a caller that names no
    severity still gets the frozen Section 9 mechanism.
    """
    from cohorts import selection

    assert selection.coefficients_for_severity("primary") is selection.DEFAULT_COEFFICIENTS
    assert (
        selection.coefficients_for_severity("stronger")
        is selection.STRONGER_SELECTION_COEFFICIENTS
    )
    assert set(selection.COEFFICIENTS_BY_SEVERITY) == {"primary", "stronger"}

    severity_default = inspect.signature(selection.observationalize).parameters[
        "selection_severity"
    ].default
    assert severity_default == "primary"


# ---------------------------------------------------------------------------
# decisions.md Section 10 -- data splitting: split fractions and
# stratification fields.
# ---------------------------------------------------------------------------


def test_section_10_split_fractions_default():
    """`decisions.md` Section 10: development/validation/test = 60%/20%/20%."""
    from cohorts.splitting import stratified_patient_split

    default_fractions = inspect.signature(stratified_patient_split).parameters[
        "fractions"
    ].default
    assert default_fractions == (0.6, 0.2, 0.2)


def test_section_10_stratification_fields():
    """`decisions.md` Section 10: stratified on treatment arm,
    primary-event indicator, prior-antiretroviral-treatment stratum."""
    from cohorts.splitting import DEFAULT_STRATA

    assert DEFAULT_STRATA == (
        "treatment_arm",
        "primary_event_indicator",
        "prior_art_status",
    )


# ---------------------------------------------------------------------------
# decisions.md Section 11 -- causal baselines: propensity models and
# weight-clipping presets.
# ---------------------------------------------------------------------------


def test_section_11_boosted_propensity_model_hyperparameters():
    """`decisions.md` Section 11: sensitivity propensity model is XGBoost
    with `max_depth=3`, `n_estimators=200`, `learning_rate=0.05`."""
    from causal.propensity import BoostedPropensityModel

    assert BoostedPropensityModel.MAX_DEPTH == 3
    assert BoostedPropensityModel.N_ESTIMATORS == 200
    assert BoostedPropensityModel.LEARNING_RATE == 0.05


def test_section_11_weight_clipping_presets():
    """`decisions.md` Section 11: primary clipping 1st/99th percentile;
    sensitivity clipping 0.5th/99.5th and 5th/95th percentile."""
    from causal.weighting import CLIP_PRESETS

    assert CLIP_PRESETS == {
        "primary": (1.0, 99.0),
        "sensitivity_narrow": (0.5, 99.5),
        "sensitivity_wide": (5.0, 95.0),
    }


def test_section_11_3_negative_control_bands():
    """`decisions.md` Section 11.3: sham-exposure null-tolerance band
    (risk-difference-scale adaptation of [0.90, 1.10]) and the baseline-CD8
    negative-control-outcome SMD threshold."""
    from causal.negative_controls import NULL_TOLERANCE_BAND, SMD_THRESHOLD

    assert NULL_TOLERANCE_BAND == (-0.10, 0.10)
    assert SMD_THRESHOLD == 0.10


# ---------------------------------------------------------------------------
# decisions.md Section 12 -- overlap and support rules.
# ---------------------------------------------------------------------------


def test_section_12_overlap_and_support_thresholds():
    """`decisions.md` Section 12: primary common-support band
    [0.10, 0.90]; sensitivity band [0.05, 0.95]; >=100 patients/ESS per
    arm; SMD < 0.10 for every adjusted covariate."""
    from causal import overlap

    assert overlap.PRIMARY_SUPPORT_LOWER == 0.10
    assert overlap.PRIMARY_SUPPORT_UPPER == 0.90
    assert overlap.SENSITIVITY_SUPPORT_LOWER == 0.05
    assert overlap.SENSITIVITY_SUPPORT_UPPER == 0.95
    assert overlap.MIN_ARM_COUNT == 100
    assert overlap.MIN_EFFECTIVE_SAMPLE_SIZE == 100.0
    assert overlap.MAX_ABS_SMD == 0.10


# ---------------------------------------------------------------------------
# decisions.md Section 14.3 -- training objective: frozen lambda grid.
# ---------------------------------------------------------------------------


def test_section_14_3_lambda_grid():
    """`decisions.md` Section 14.3: lambda_state, lambda_balance each in
    {0.1, 0.3, 1.0}, giving 9 frozen combinations."""
    from models.losses import load_lambda_grid

    grid = load_lambda_grid()

    expected_values = {0.1, 0.3, 1.0}
    expected_combinations = {
        (state, balance) for state in expected_values for balance in expected_values
    }

    assert set(grid) == expected_combinations
    assert len(grid) == 9


# ---------------------------------------------------------------------------
# decisions.md Section 15 -- uncertainty method: ensemble size and
# bootstrap replicate count.
# ---------------------------------------------------------------------------


def test_section_15_deep_ensemble_size_default():
    """`decisions.md` Section 15: primary uncertainty method is a deep
    ensemble of 5 independently initialized replicas."""
    from uncertainty.ensemble import DeepEnsemble

    default_n_replicas = inspect.signature(DeepEnsemble.__init__).parameters[
        "n_replicas"
    ].default
    assert default_n_replicas == 5


def test_section_15_bootstrap_replicate_count_default():
    """`decisions.md` Section 15/19: 1,000 patient-level bootstrap
    replicates, used by both `uncertainty.bootstrap.patient_level_bootstrap`
    and `evaluation.statistics.paired_bootstrap_test`."""
    from evaluation.statistics import paired_bootstrap_test
    from uncertainty.bootstrap import patient_level_bootstrap

    default_bootstrap_replicates = inspect.signature(
        patient_level_bootstrap
    ).parameters["n_replicates"].default
    default_paired_replicates = inspect.signature(paired_bootstrap_test).parameters[
        "n_replicates"
    ].default

    assert default_bootstrap_replicates == 1000
    assert default_paired_replicates == 1000


# ---------------------------------------------------------------------------
# decisions.md Section 17 -- primary ablations (exactly ten, frozen).
# ---------------------------------------------------------------------------


def test_section_17_ablations_count_and_uniqueness():
    """`decisions.md` Section 17: exactly ten frozen ablations."""
    from evaluation.ablation_matrix import ABLATIONS

    names = [ablation.name for ablation in ABLATIONS]
    assert len(ABLATIONS) == 10
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# decisions.md Section 18 -- primary evaluation metrics: published
# reference hazard ratio used in E_logHR.
# ---------------------------------------------------------------------------


def test_section_18_published_reference_hazard_ratio():
    """`decisions.md` Section 7/8/18: published external reference
    HR = 0.50 (Hammer et al., NEJM 1996), used in
    `E_logHR = |log(HR_hat) - log(0.50)|`."""
    from evaluation import metrics

    assert metrics._PUBLISHED_REFERENCE_HAZARD_RATIO == 0.50


# ---------------------------------------------------------------------------
# decisions.md Section 19 -- statistical plan: primary comparisons and the
# small-sample-guard minimum configuration count.
# ---------------------------------------------------------------------------


def test_section_19_primary_comparisons():
    """`decisions.md` Section 19: Holm correction for exactly the three
    frozen primary model comparisons."""
    from evaluation.statistics import PRIMARY_COMPARISONS

    assert set(PRIMARY_COMPARISONS) == {
        "baseline_vs_intervention_aware",
        "intervention_aware_vs_ipcw_aipw",
        "gated_vs_ungated",
    }
    assert len(PRIMARY_COMPARISONS) == 3


def test_section_19_default_min_configurations():
    """`decisions.md` Section 19/Requirement 14.6-14.7: no
    predictive-causal correlation claim unless enough independent
    configurations exist. Frozen default minimum is 5."""
    from evaluation.statistics import DEFAULT_MIN_CONFIGURATIONS

    assert DEFAULT_MIN_CONFIGURATIONS == 5


# ---------------------------------------------------------------------------
# decisions.md Section 20 -- primary robustness analysis: eight frozen
# sensitivity variants.
# ---------------------------------------------------------------------------


def test_section_20_sensitivity_variants_count_and_uniqueness():
    """`decisions.md` Section 20: eight frozen sensitivity-analysis
    variants."""
    from evaluation.robustness import SENSITIVITY_VARIANTS

    names = [variant.name for variant in SENSITIVITY_VARIANTS]
    assert len(SENSITIVITY_VARIANTS) == 8
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# decisions.md Section 26 amendment (2026-09-07, amending Sections 19/20/25)
# -- the cross-seed stability criterion the Replication_Aggregator applies.
# ---------------------------------------------------------------------------


def test_section_26_cross_seed_stability_criterion_values():
    """`decisions.md` Section 26 amendment dated 2026-09-07: effect-direction
    agreement in at least 4 of the 5 frozen replication seeds, together with
    no required sensitivity variant contradicting the aggregate direction.
    """
    import dataclasses

    from run_replication import CROSS_SEED_STABILITY_CRITERION

    assert CROSS_SEED_STABILITY_CRITERION.min_agreeing_seeds == 4
    assert CROSS_SEED_STABILITY_CRITERION.total_seeds == 5
    assert CROSS_SEED_STABILITY_CRITERION.require_no_contradicting_sensitivity_variant is True

    # Frozen means frozen: the criterion is a `decisions.md` value, not a
    # runtime knob a caller may retune mid-run.
    with pytest.raises(dataclasses.FrozenInstanceError):
        CROSS_SEED_STABILITY_CRITERION.min_agreeing_seeds = 3  # type: ignore[misc]


def test_section_26_cross_seed_stability_criterion_matches_amendment_text():
    """The criterion's human-readable description must appear *verbatim* in
    the Section 26 amendment entry.

    Requirement 9.7 has the aggregator report the applied criterion
    alongside its determination, and it reports this string as-is. Comparing
    it against the document's own prose (rather than only against
    transcribed numbers) is what keeps the reported criterion and the frozen
    criterion from drifting apart: editing either one alone fails here.
    """
    from run_replication import (
        CROSS_SEED_STABILITY_CRITERION,
        CROSS_SEED_STABILITY_CRITERION_DESCRIPTION,
    )

    assert (
        CROSS_SEED_STABILITY_CRITERION.criterion_description
        == CROSS_SEED_STABILITY_CRITERION_DESCRIPTION
    )

    amendment_log = _amendment_log_text()
    assert CROSS_SEED_STABILITY_CRITERION_DESCRIPTION in amendment_log, (
        "the frozen cross-seed stability criterion description does not appear "
        "verbatim in decisions.md Section 26; resolve this by amending "
        "decisions.md, never by editing the constant to match the code"
    )


# ---------------------------------------------------------------------------
# decisions.md Section 25 -- the frozen GRU hidden width, shared by both
# torch models. Lives here rather than in `tests/neural/` because it is a
# frozen-value consistency assertion spanning two modules, which is exactly
# this file's purpose; the torch dependency is confined to the test body by
# `importorskip` so the file stays importable (and the rest of it runnable)
# without torch installed.
# ---------------------------------------------------------------------------


def test_section_25_hidden_width_is_one_shared_value():
    """`decisions.md` Section 25: "Hidden width: `hidden_size = 16` for both
    the sequence-GRU baseline and the intervention-aware model's
    `GRUTransition`".

    The word "both" is the substance of this test. The frozen decision is
    not "each model uses 16"; it is that the two models share *one* width,
    because the treatment-as-feature vs. explicit-channel ablation contrast
    (`decisions.md` Section 17 items 2-3) isolates the representational
    difference only if capacity is held equal. Two independent literals
    could drift to 16 and 32 with every other test still passing and
    silently invalidate that contrast, so this asserts both that the value
    is 16 *and* that the two exported names are the same value -- sourced
    from the one canonical definition in `models.dataset`.
    """
    pytest.importorskip("torch")

    from models.dataset import FROZEN_HIDDEN_SIZE
    from models.torch_intervention_aware import HIDDEN_SIZE
    from models.torch_sequence_gru import FROZEN_HIDDEN_SIZE as SEQUENCE_GRU_HIDDEN_SIZE

    assert FROZEN_HIDDEN_SIZE == 16
    assert SEQUENCE_GRU_HIDDEN_SIZE == FROZEN_HIDDEN_SIZE, (
        "models.torch_sequence_gru.FROZEN_HIDDEN_SIZE has diverged from the "
        "canonical models.dataset.FROZEN_HIDDEN_SIZE; the sequence-GRU "
        "baseline must re-export the shared width, not redefine it"
    )
    assert HIDDEN_SIZE == FROZEN_HIDDEN_SIZE, (
        "models.torch_intervention_aware.HIDDEN_SIZE has diverged from the "
        "canonical models.dataset.FROZEN_HIDDEN_SIZE; the two models must "
        "share one hidden width or the Section 17 ablation contrast measures "
        "a capacity difference instead of a representational one"
    )

    # The two models' constructor defaults are the same width, which is the
    # property the ablation contrast actually depends on -- the constants
    # agreeing would not help if a default were hard-coded past them.
    from models.torch_intervention_aware import TorchInterventionAwareModel
    from models.torch_sequence_gru import TorchSequenceGRU

    sequence_default = (
        inspect.signature(TorchSequenceGRU.__init__).parameters["hidden_size"].default
    )
    intervention_default = (
        inspect.signature(TorchInterventionAwareModel.__init__)
        .parameters["hidden_size"]
        .default
    )
    assert sequence_default == intervention_default == 16
