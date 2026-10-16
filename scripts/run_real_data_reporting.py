#!/usr/bin/env python
"""Translate `scripts/run_causal_analysis.py`'s real-data output into
`scripts/run_paper.py`'s documented `--experiment` JSON schema, then call
that script's reproduction stage -- closing the real-data-to-reporting gap
`run_paper.py`'s own docstring identifies ("regenerates ... from
already-saved predictions ... does not re-run the causal/model pipeline").

Why this is a separate script, not a change to `run_paper.py` itself
------------------------------------------------------------------------

`run_paper.py` is deliberately scoped to only ever read an
already-prepared JSON file (Requirement 15.5's "regenerate ... from saved
predictions"); it never runs the pipeline itself, and its locked-evaluation
refusal logic (Requirement 15.6) depends on that scoping staying narrow --
see that script's module docstring. Rather than blur that boundary by
teaching `run_paper.py` to also run the pipeline, this script sits in
front of it: it runs `run_causal_analysis.run_causal_analysis()` (or reads
an already-produced result if `--skip-analysis` is given, pointing
`--causal-analysis-summary` at a previously-written manifest), builds the
`--experiment` JSON `run_paper.py` expects, writes it to disk, and then
invokes `run_paper.py.run_reproduction()` directly (in-process, not a
subprocess) with that file.

What this script can and cannot honestly report
----------------------------------------------------

`reporting.tables.REQUIRED_COLUMNS` spans four frozen metric groups
(`decisions.md` Section 18): factual prediction, causal recovery,
selective prediction, and consistency. This codebase's real-data pipeline
(`run_causal_analysis.py`) only computes the **causal recovery** group in
full (six causal baselines, `evaluation.metrics.estimated_hazard_ratio`/
`absolute_log_hr_error`/`effect_direction_agreement`), plus rank-based
**factual-prediction** metrics from the predictive Cox baseline's relative-
hazard scores (concordance, AUROC, AUPRC -- all rank-based, so they need
only a risk *ordering*, not a calibrated probability). It does NOT compute:

- `ipcw_brier_score`, `calibration_slope`, `calibration_intercept`: these
  require the Cox baseline's risk scores to be interpreted as calibrated
  probabilities of a 96-week event, which requires a baseline-hazard
  (Breslow) estimator this codebase never implements (`CoxBaseline` only
  exposes the linear-predictor-exponentiated relative hazard, not an
  absolute probability).
- `cd4_cd8_mae`, and the entire **selective-prediction** group
  (`retained_coverage`, `selective_risk`, `aurcc`,
  `incorrect_release_rate`, `unnecessary_abstention_rate`,
  `error_at_matched_coverage`): these require the sequence-GRU/
  intervention-aware neural models and the ensemble-uncertainty machinery
  (`uncertainty.ensemble.DeepEnsemble`), neither of which this script (or
  `run_causal_analysis.py`) trains.
- `risk_difference_error` (the 96-week risk-difference error against the
  **internal** randomized reference, `decisions.md` Section 18): this is
  intentionally never computed here, at any point, on purpose --
  `risk_difference_error_vs_internal_reference` requires the sealed
  reference's real value, and `decisions.md` Section 8 forbids opening it
  before the locked external-evaluation stage. This script never holds a
  `LockedEvaluationToken` and never calls `SealedReference.unseal()` or
  `read_sealed_artifact()`.

Every column this script cannot honestly back is reported as `float("nan")`
in the results-table row it builds -- never a fabricated placeholder
number -- with the specific reason recorded in a companion
`unsupported_metrics` dict included in the written experiment JSON (outside
`results_table_rows`, purely for provenance; `reporting.tables.build_main_results_table`
does not read it). `nan` survives Python's `json.dumps` (non-standard JSON
but valid Python `json` output, matching how `float("nan")` already
appears elsewhere in this codebase's outputs, e.g.
`evaluation.metrics.concordance_index`'s documented no-comparable-pairs
return value) and is a truthful "not computed," not a disguised zero or
omitted key that `build_main_results_table`'s completeness check would
otherwise reject.

Usage::

    python scripts/run_real_data_reporting.py \
        --real-data-path data/raw/actg175_v3_2 \
        --trial-config configs/target_trials/main.yaml \
        --results-root results \
        --master-seed 20240101 \
        --output-dir results/real_data_reporting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from run_causal_analysis import (
    DEFAULT_REAL_DATA_PATH,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TRIAL_CONFIG,
    NEGATIVE_CONTROL_N_BOOTSTRAP,
    run_causal_analysis,
)
from run_paper import run_reproduction

from causal.baselines import adjusted_cox
from evaluation.metrics import (
    absolute_log_hr_error,
    auprc,
    auroc,
    concordance_index,
    effect_direction_agreement,
    estimated_hazard_ratio,
)
from models.non_neural_baseline import CoxBaseline
from protocols.schema import TargetTrialSpec
from reporting.tables import build_blocked_outcome_report, build_executed_results_report

# ---------------------------------------------------------------------------
# Task 19.2 (Requirement 16.5, 16.6): the controlled semi-synthetic
# benchmark's reporting section. See module-level
# `build_synthetic_benchmark_section` and `run_synthetic_benchmark_scenario`
# below, placed after the real-data helpers above since this section is
# additive to, and clearly separated from, the real-data path.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Bootstrap replicate count for the primary treatment-effect estimate's
#: confidence interval (`adjusted_cox`, per
#: `uncertainty.bootstrap.patient_level_bootstrap`). Kept modest --
#: `adjusted_cox` refits a Cox model once per replicate.
TREATMENT_EFFECT_N_BOOTSTRAP = 100

#: Causal-recovery columns that must be suppressed (never released as a
#: real number) whenever the aggregate witness release decision is not
#: `passed` (`decisions.md` Section 16's "release" condition: `release =
#: overlap_valid AND balance_valid AND uncertainty_valid AND
#: temporal_witness == pass AND causal_witness == pass AND
#: clinical_witness == pass`). A `fail` or `uncertain` result on any
#: witness means the cohort-level estimate is not release-eligible -- the
#: frozen rule this constant enforces, not a judgment call this script
#: makes per run.
#: `risk_difference_error` is deliberately NOT listed here even though it
#: is also a causal-recovery column: it is unconditionally unsupported
#: (see `UNSUPPORTED_METRIC_REASONS`) regardless of the release gate's
#: outcome, since it requires unsealing the internal randomized reference
#: -- never done by this script at all, gate-passing or not. Listing it
#: here would let the release-gate reason overwrite its more precise
#: Section-8 reason when the gate happens to fail.
_SUPPRESSED_WHEN_RELEASE_GATE_FAILS: tuple[str, ...] = (
    "estimated_hr",
    "e_log_hr",
    "effect_direction_agreement",
    "ci_overlap",
)

#: Reason recorded (never a fabricated number) when the release gate has
#: not passed and `_SUPPRESSED_WHEN_RELEASE_GATE_FAILS` columns are
#: withheld.
_RELEASE_GATE_SUPPRESSION_REASON = (
    "aggregate witness release decision is not 'passed' (decisions.md "
    "Section 16); the cohort-level causal estimate is withheld from "
    "release per the frozen release condition, not computed as NaN for a "
    "missing-feature reason."
)

#: Reason recorded (task 22.1, Requirement 17.1-17.5) for every one of the
#: five headline comparisons `src/reporting/tables.py::HEADLINE_COMPARISON_NAMES`
#: names: all five compare at least one of the sequence-GRU,
#: intervention-aware, IPCW-AIPW (as a *paired*, per-patient value series,
#: not the single scalar `ipcw_aipw` estimate `run_all_baselines` already
#: produces), gated/ungated, or placebo-channel configurations. None of
#: those models are trained by this script or by `run_causal_analysis.py`
#: (see module docstring, "What this script can and cannot honestly
#: report") -- producing them requires `scripts/run_ablations.py` (the
#: GRU/intervention-aware/placebo/gated-ungated arms) and
#: `scripts/run_locked_evaluation.py` (the only call site that evaluates
#: the three frozen comparisons' paired bootstrap over the identical
#: locked test-split patient set, per Requirement 17.6's "paired bootstrap
#: comparison on identical patients"). Reporting a headline comparison
#: honestly requires that identical-patient pairing; this script's single
#: non-neural Cox-baseline row has no paired model-vs-model values to
#: compare at all.
_HEADLINE_COMPARISON_UNSUPPORTED_REASON = (
    "requires at least one of the sequence-GRU, intervention-aware, "
    "IPCW-AIPW (as a paired per-patient series), gated/ungated, or "
    "placebo-channel model outputs, paired on identical test patients; "
    "this script trains no neural model and holds no locked-evaluation "
    "test-split pairing -- see scripts/run_ablations.py and "
    "scripts/run_locked_evaluation.py."
)


def _unsupported_headline_comparisons() -> dict[str, dict[str, Any]]:
    """Build the honest "not computed here" placeholder for every one of
    the five headline comparisons (Requirement 17.1-17.5), named exactly
    `reporting.tables.HEADLINE_COMPARISON_NAMES` so a caller wiring this
    script's output into `reporting.tables.build_headline_comparisons_table()`
    sees the same five names, every one carrying `"supported": False` and
    the reason, with no fabricated point estimate or interval -- see
    `_HEADLINE_COMPARISON_UNSUPPORTED_REASON`.
    """
    from reporting.tables import HEADLINE_COMPARISON_NAMES

    return {
        name: {"supported": False, "reason": _HEADLINE_COMPARISON_UNSUPPORTED_REASON}
        for name in HEADLINE_COMPARISON_NAMES
    }


#: Columns from `reporting.tables.REQUIRED_COLUMNS` this script cannot
#: honestly compute on real data, and why. See module docstring.
UNSUPPORTED_METRIC_REASONS: dict[str, str] = {
    "ipcw_brier_score": (
        "requires calibrated absolute event probabilities; CoxBaseline "
        "only exposes the relative-hazard linear predictor (no Breslow "
        "baseline-hazard estimator is implemented)."
    ),
    "calibration_slope": "same reason as ipcw_brier_score.",
    "calibration_intercept": "same reason as ipcw_brier_score.",
    "cd4_cd8_mae": (
        "requires the sequence-GRU longitudinal model's held-out "
        "predictions; not trained by this script."
    ),
    "retained_coverage": (
        "requires the ensemble-uncertainty-based support gate "
        "(uncertainty.ensemble.DeepEnsemble); not trained by this script."
    ),
    "selective_risk": "same reason as retained_coverage.",
    "aurcc": "same reason as retained_coverage.",
    "incorrect_release_rate": "same reason as retained_coverage.",
    "unnecessary_abstention_rate": "same reason as retained_coverage.",
    "error_at_matched_coverage": "same reason as retained_coverage.",
    "risk_difference_error": (
        "requires unsealing the internal randomized reference "
        "(decisions.md Section 8); never computed before the locked "
        "external-evaluation stage, by design -- not a missing-feature gap."
    ),
}


def _elapsed_weeks(patient, assignment_time) -> float:
    return (patient.endpoint_result.event_time - assignment_time).days / 7.0


def _build_results_table_row(
    dev_patients: list,
    causal_estimates: dict[str, float],
    temporal_violation_count: int,
    protocol_violation_count: int,
    witness_detected_failure_count: int,
    release_gate_passed: bool,
) -> dict[str, Any]:
    """Build the one results-table row this script can honestly support.

    Factual-prediction metrics (concordance/AUROC/AUPRC) are computed from
    a freshly-fit `CoxBaseline`'s relative-hazard risk scores on the same
    dev-retained population `run_causal_analysis` already produced --
    these are rank-based and need only a risk ordering, unlike
    IPCW-Brier/calibration (see `UNSUPPORTED_METRIC_REASONS`).

    Args:
        release_gate_passed: `run_causal_analysis`'s
            `aggregate_decision_passed` -- the frozen witness-aggregation
            release decision (`decisions.md` Section 16). When `False`,
            every column in `_SUPPRESSED_WHEN_RELEASE_GATE_FAILS` (the
            causal-recovery estimate and everything derived from it) is
            withheld (`NaN`/`None`, never the real computed number) rather
            than reported alongside a failed/uncertain witness result.
    """
    from causal.baselines import HORIZON_WEEKS

    baseline = CoxBaseline().fit(dev_patients)
    fitted = baseline.fitted

    risk_scores = [fitted.risk_scores[patient.patient_id] for patient in dev_patients]
    event_times_weeks = []
    labels_96wk = []
    event_indicators = []
    for patient in dev_patients:
        assignment_time = min(
            event.observed_at for event in patient.timeline.treatment_events
        )
        elapsed = _elapsed_weeks(patient, assignment_time)
        event_times_weeks.append(elapsed)
        event_indicators.append(patient.endpoint_result.event_indicator)
        labels_96wk.append(
            patient.endpoint_result.event_indicator and elapsed <= HORIZON_WEEKS
        )

    concordance = concordance_index(risk_scores, event_times_weeks, event_indicators)
    auroc_value = auroc(risk_scores, labels_96wk)
    auprc_value = auprc(risk_scores, labels_96wk)

    adjusted_cox_log_hr = causal_estimates["adjusted_cox"]

    row: dict[str, Any] = {
        "label": "real_data_causal_analysis",
        "n": len(dev_patients),
        # Factual-prediction metrics are reported unconditionally: they
        # measure predictive accuracy, which decisions.md/Requirement 8
        # explicitly treats as separate from causal validity, so a failed
        # causal-witness gate does not withhold them.
        "concordance_index": concordance,
        "auroc_96wk": auroc_value,
        "auprc_96wk": auprc_value,
        "temporal_violations": temporal_violation_count,
        "protocol_violations": protocol_violation_count,
        # "impossible_transition_rate" specifically names impossible STATE
        # TRANSITIONS in a longitudinal model rollout (Requirement 9.9 /
        # decisions.md Section 18's consistency group) -- this script does
        # not train the sequence-GRU/intervention-aware model, so there is
        # no rollout to check transitions on. Distinct from
        # "protocol_violations"/"temporal_violations" above, which ARE
        # real counts from the clinical/temporal witnesses already run on
        # this cohort.
        "impossible_transition_rate": float("nan"),
        "witness_detected_failures": witness_detected_failure_count,
        "release_gate_passed": release_gate_passed,
    }

    if release_gate_passed:
        row["estimated_hr"] = estimated_hazard_ratio(adjusted_cox_log_hr)
        row["e_log_hr"] = absolute_log_hr_error(adjusted_cox_log_hr)
        row["effect_direction_agreement"] = effect_direction_agreement(adjusted_cox_log_hr)
        # ci_overlap needs a second interval to compare against (e.g. the
        # published reference's own CI, which decisions.md Section 8
        # explicitly does not define -- "no arbitrary clinical-equivalence
        # margin"); left unsupported rather than fabricating a comparison
        # interval, even when the release gate has passed.
        row["ci_overlap"] = None
    else:
        # decisions.md Section 16's release condition failed -- suppress
        # the causal-recovery estimate rather than reporting it alongside
        # a failed/uncertain witness result. Never a fabricated number.
        for column in _SUPPRESSED_WHEN_RELEASE_GATE_FAILS:
            row[column] = float("nan")

    for column, _reason in UNSUPPORTED_METRIC_REASONS.items():
        row[column] = float("nan")
    return row


def build_experiment_json(
    real_data_path: Path,
    trial_config_path: Path,
    results_root: Path,
    master_seed: int,
    negative_control_n_bootstrap: int = NEGATIVE_CONTROL_N_BOOTSTRAP,
    treatment_effect_n_bootstrap: int = TREATMENT_EFFECT_N_BOOTSTRAP,
) -> dict[str, Any]:
    """Run the causal analysis and build the `--experiment` JSON payload
    `scripts/run_paper.py --stage reproduction` expects.

    Returns the payload dict (not yet written to disk); `main()` writes it.
    """
    with trial_config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    spec = TargetTrialSpec.model_validate(raw)

    from cohorts.pipeline import run_observationalization_pipeline
    from cohorts.sealed_reference import unadjusted_cox_reference_estimator
    from data.loaders.actg175 import load_actg175
    from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
    from infra.seeding import derive_seed

    analysis_summary = run_causal_analysis(
        real_data_path,
        trial_config_path,
        results_root,
        master_seed,
        negative_control_n_bootstrap=negative_control_n_bootstrap,
    )

    # Rebuild the dev-retained population once more for the bootstrap CI
    # and the factual-prediction row -- run_causal_analysis() does not
    # return the patient objects themselves (only its summary dict), and
    # re-deriving them here (same inputs, same master_seed) is cheap
    # relative to the bootstrap itself and keeps run_causal_analysis's own
    # return contract unchanged for its existing callers/tests.
    tables = load_actg175(real_data_path)
    timelines, treatment_arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)
    pipeline_seed = derive_seed(master_seed, "causal_analysis_pipeline")
    pipeline_result = run_observationalization_pipeline(
        timelines,
        treatment_arms,
        spec,
        pipeline_seed,
        reference_estimator=unadjusted_cox_reference_estimator,
        precomputed_endpoint_results=endpoint_results,
    )
    dev_ids = set(pipeline_result.dev.retained_patient_ids)
    # The same estimation population `run_causal_analysis` used: the
    # retained dev split minus the patients its common-support trim removed
    # (`decisions.md` Section 26 amendment of 2026-10-12, item B), so the
    # interval and the results row describe the estimate actually reported.
    trimmed_ids = set(analysis_summary["common_support_trim"]["trimmed_patient_ids"])
    dev_patients = [
        patient
        for patient in pipeline_result.split_cohorts.dev
        if patient.patient_id in dev_ids and patient.patient_id not in trimmed_ids
    ]

    release_gate_passed = analysis_summary["aggregate_decision_passed"]

    # Skip the (expensive -- one Cox refit per replicate) bootstrap
    # entirely when the release gate has not passed: its result would be
    # withheld from the payload anyway (see _build_results_table_row /
    # treatment_effect_estimate below), so there is nothing to compute it
    # for.
    bootstrap_ci = None
    if release_gate_passed:
        from uncertainty.bootstrap import patient_level_bootstrap

        bootstrap_ci = patient_level_bootstrap(
            lambda patients: adjusted_cox(patients).estimate,
            dev_patients,
            n_replicates=treatment_effect_n_bootstrap,
            seed=derive_seed(master_seed, "treatment_effect_bootstrap"),
        )

    results_table_row = _build_results_table_row(
        dev_patients,
        analysis_summary["causal_estimates"],
        temporal_violation_count=analysis_summary["temporal_violation_count"],
        protocol_violation_count=analysis_summary["protocol_violation_count"],
        witness_detected_failure_count=analysis_summary["witness_detected_failure_count"],
        release_gate_passed=release_gate_passed,
    )

    # Task 22.2 (Requirement 19.1, 19.3): honest reporting of the real
    # six-way cohort-release decision and every executed sensitivity
    # variant, reusing `run_causal_analysis`'s own already-computed
    # `release_decision_reasons`/`sensitivity_variant_outcomes` rather than
    # recomputing anything. `release_decision_reasons` is from the real
    # `evaluation.release_gate.evaluate_release` call (not the
    # three-witness `aggregate_decision_passed` this row's suppression
    # above uses) -- it is always non-empty at this script's scope, since
    # `run_causal_analysis.py` conservatively supplies
    # `uncertainty_valid=False` (see that module's own docstring), so the
    # six-way gate never actually passes here. The real six-way result is
    # reported anyway, honestly, rather than substituting the
    # three-witness aggregate for it.
    release_report: dict[str, Any] | None = None
    if not analysis_summary["release_decision_released"]:
        release_report = build_blocked_outcome_report(
            released=False,
            failing_conditions=analysis_summary["release_decision_reasons"],
            supported_count=analysis_summary["supported_count"],
            abstained_count=analysis_summary["abstained_count"],
        )
    executed_results_report = build_executed_results_report(
        analysis_summary["sensitivity_variant_outcomes"]
    )

    # decisions.md Section 16's release condition governs the treatment-
    # effect estimate itself, not just the results-table row -- when the
    # gate has not passed, the bootstrap CI is withheld from the returned
    # payload entirely (run_paper.py's --experiment schema treats
    # "treatment_effect_estimate" as optional: omitting it simply skips
    # building the treatment-effect-vs-reference figure, rather than
    # plotting a suppressed estimate with a caveat).
    treatment_effect_estimate = (
        {
            "point_estimate": bootstrap_ci.point_estimate,
            "ci_low": bootstrap_ci.ci_low,
            "ci_high": bootstrap_ci.ci_high,
            "n_replicates": bootstrap_ci.n_replicates,
            "n_successful": bootstrap_ci.n_successful,
        }
        if release_gate_passed
        else None
    )

    unsupported_metrics = dict(UNSUPPORTED_METRIC_REASONS)
    if not release_gate_passed:
        for column in _SUPPRESSED_WHEN_RELEASE_GATE_FAILS:
            unsupported_metrics[column] = _RELEASE_GATE_SUPPRESSION_REASON

    payload: dict[str, Any] = {
        "results_table_rows": [results_table_row],
        # reference_value is deliberately omitted here: the only real
        # reference this script could supply is either the published
        # external HR (0.50, log-HR ~ -0.69) or the sealed internal
        # reference. The internal reference cannot be read (see module
        # docstring); using the published one instead would violate
        # decisions.md Section 8's "published reference is historical
        # corroboration only, never used as if it were the internal
        # reference" framing if labeled ambiguously. A caller wanting the
        # published-reference comparison plot should pass
        # reference_value=math.log(0.50) and reference_label="Published
        # external reference (Hammer et al. 1996)" explicitly, rather than
        # this script silently choosing that framing.
        "ablation_results": {},
        "unsupported_metrics": unsupported_metrics,
        "headline_comparisons": _unsupported_headline_comparisons(),
        "causal_analysis_summary": analysis_summary,
        "release_gate_passed": release_gate_passed,
        # Task 22.2 (Requirement 19.1, 19.3): the real six-way release
        # decision and every sensitivity variant's outcome, honestly
        # reported -- see the comment above where these are built.
        "executed_results": executed_results_report,
    }
    if release_report is not None:
        payload["blocked_outcome"] = release_report
    if treatment_effect_estimate is not None:
        payload["treatment_effect_estimate"] = treatment_effect_estimate

    # Task 19.2 (Requirement 16.5, 16.6): the synthetic benchmark's three
    # scenarios, run and reported alongside the real-data result. See
    # module docstring addendum above and `build_synthetic_benchmark_section`.
    payload["synthetic_benchmark"] = build_synthetic_benchmark_section(master_seed)
    return payload


def _default_synthetic_scenarios():
    """Default scenario parameters this reporting section runs
    (Requirement 16.1-16.2). These are this script's own choice -- see
    `evaluation.synthetic_benchmark`'s module docstring for why no
    `decisions.md` amendment is needed for them. A modest `n_patients`
    (200) keeps this section fast to compute on every reporting run while
    still giving the causal/overlap/witness machinery a genuine
    population to operate on.
    """
    from evaluation.synthetic_benchmark import SyntheticScenario

    return (
        SyntheticScenario(
            name="limited_overlap",
            n_patients=200,
            true_risk_difference=-0.10,
            overlap_severity=0.8,
            unmeasured_confounding_strength=0.1,
            missingness_rate=0.1,
        ),
        SyntheticScenario(
            name="unmeasured_confounding",
            n_patients=200,
            true_risk_difference=-0.10,
            overlap_severity=0.1,
            unmeasured_confounding_strength=0.8,
            missingness_rate=0.1,
        ),
        SyntheticScenario(
            name="increased_missingness",
            n_patients=200,
            true_risk_difference=-0.10,
            overlap_severity=0.1,
            unmeasured_confounding_strength=0.1,
            missingness_rate=0.7,
        ),
    )


def run_synthetic_benchmark_scenario(scenario, seed: int) -> dict[str, Any]:
    """Run one synthetic-benchmark scenario through the real causal
    pipeline and report its recovery error against the declared ground
    truth, alongside the real `ReleaseDecision` the gate produced
    (Requirement 16.6).

    This is the identical pipeline the real-data path runs (Requirement
    16.3): `cohorts.pipeline.run_observationalization_pipeline` ->
    cross-fitted propensity -> `causal.baselines.ipcw_aipw` (the primary
    causal estimator) -> `causal.overlap.OverlapDiagnostics.compute` ->
    `evaluation.release_gate.evaluate_release`. Witnesses other than the
    causal witness are not evaluated here (temporal/clinical witness
    evaluation needs the full timeline-adapter machinery
    `scripts/run_causal_analysis.py` already owns, and duplicating it for
    a synthetic smoke-level report would be the second-pipeline risk
    Requirement 16.3 warns against) -- `temporal_pass`/`clinical_pass` are
    conservatively supplied as `False`, so the six-way gate never passes
    by omission; only `causal_pass` and `overlap_valid`/`balance_valid`
    reflect this scenario's real, computed diagnostics.

    Args:
        scenario: an `evaluation.synthetic_benchmark.SyntheticScenario`.
        seed: forwarded to `evaluation.synthetic_benchmark.generate` and
            to every derived seed in this scenario's own pipeline run.

    Returns:
        A dict with `"scenario_name"`, `"true_risk_difference"`,
        `"realized_average_treatment_effect"` (the generator's own
        ground-truth-recovery diagnostic, Requirement 16.1),
        `"recovery_error"` (`abs(estimated - declared ground truth)`,
        computed from the real `ipcw_aipw` estimate -- the actual
        pipeline's recovery error, not the generator's internal
        bookkeeping), `"causal_estimate"`, `"release_decision"` (every
        one of the six named booleans plus `"reasons"`), and
        `"diagnostic_label"` fixed to the literal string this section's
        caller asserts on (Requirement 16.5).
    """
    from cohorts.pipeline import run_observationalization_pipeline
    from infra.seeding import derive_seed

    scenarios_module_path = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
    with scenarios_module_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    spec = TargetTrialSpec.model_validate(raw)

    from evaluation.synthetic_benchmark import generate

    cohort = generate(scenario, seed=seed)

    pipeline_seed = derive_seed(seed, f"synthetic_benchmark_{scenario.name}")
    pipeline_result = run_observationalization_pipeline(
        cohort.timelines,
        cohort.treatment_arms,
        spec,
        pipeline_seed,
    )
    dev_ids = set(pipeline_result.dev.retained_patient_ids)
    dev_patients = [
        patient for patient in pipeline_result.split_cohorts.dev if patient.patient_id in dev_ids
    ]

    from causal.adjustment_set import ADJUSTMENT_SET_NAMES
    from causal.baselines import ipcw_aipw
    from causal.overlap import OverlapDiagnostics, trim_to_common_support
    from causal.propensity import LogisticPropensityModel
    from evaluation.release_gate import evaluate_release
    from witnesses.causal import CausalWitness
    from witnesses.temporal import WitnessResult

    propensity_model = LogisticPropensityModel(
        seed=derive_seed(pipeline_seed, "synthetic_benchmark_propensity")
    )
    try:
        cross_fitted = propensity_model.fit_cross_fitted(dev_patients, n_folds=3)
    except Exception as exc:  # noqa: BLE001 -- report a failed scenario, never crash the section
        return {
            "scenario_name": scenario.name,
            "diagnostic_label": _SYNTHETIC_DIAGNOSTIC_LABEL,
            "true_risk_difference": scenario.true_risk_difference,
            "realized_average_treatment_effect": cohort.realized_average_treatment_effect,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # The same common-support trim the real-data path applies (Section 26
    # amendment of 2026-10-12, item B), so this stays the identical pipeline.
    support_trim = trim_to_common_support(dev_patients, cross_fitted.propensity_scores)
    dev_patients = list(support_trim.retained)

    try:
        estimate = ipcw_aipw(dev_patients, cross_fitted.propensity_scores)
    except Exception as exc:  # noqa: BLE001 -- e.g. trimming left too few patients to fit
        return {
            "scenario_name": scenario.name,
            "diagnostic_label": _SYNTHETIC_DIAGNOSTIC_LABEL,
            "true_risk_difference": scenario.true_risk_difference,
            "realized_average_treatment_effect": cohort.realized_average_treatment_effect,
            "common_support_trim": support_trim.to_dict(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    recovery_error = abs(estimate.estimate - scenario.true_risk_difference)

    overlap_report = OverlapDiagnostics.compute(
        dev_patients, cross_fitted.propensity_scores, ADJUSTMENT_SET_NAMES
    )

    from causal.negative_controls import (
        run_negative_control_exposure,
        run_negative_control_outcome,
    )

    exposure_result = run_negative_control_exposure(
        dev_patients, cross_fitted.propensity_scores, seed=pipeline_seed, n_bootstrap=20
    )
    outcome_result = run_negative_control_outcome(
        dev_patients, ADJUSTMENT_SET_NAMES, seed=pipeline_seed, n_bootstrap=20
    )
    # This scenario runner evaluates no sensitivity variants (see
    # docstring), so there is no real `SensitivityStabilityResult` to
    # derive from here -- but `sensitivity_stable` is still expressed as
    # a comparison against the `evaluation.robustness` convention
    # ("stable" only when a real result says so) rather than a bare
    # literal, exactly as `scripts/run_causal_analysis.py` and
    # `scripts/run_replication.py` both do (Requirement 8.9;
    # `tests/unit/test_sensitivity_stability.py`'s
    # `test_no_runner_passes_a_literal_stability_boolean`).
    sensitivity_overall_stability = "not_evaluated"
    causal_result = CausalWitness.evaluate(
        overlap_report=overlap_report,
        sensitivity_stable=(sensitivity_overall_stability == "stable"),
        negative_control_exposure_result=exposure_result,
        negative_control_outcome_result=outcome_result,
    )
    # Temporal/clinical witnesses are not evaluated here (see docstring) --
    # "uncertain" status so neither reads as a pass by omission.
    witness_results = [
        WitnessResult(witness="temporal", status="uncertain", evidence={"note": "not evaluated"}),
        causal_result,
        WitnessResult(witness="clinical", status="uncertain", evidence={"note": "not evaluated"}),
    ]
    release_decision = evaluate_release(overlap_report, witness_results, uncertainty_valid=False)

    return {
        "scenario_name": scenario.name,
        "diagnostic_label": _SYNTHETIC_DIAGNOSTIC_LABEL,
        "true_risk_difference": scenario.true_risk_difference,
        "realized_average_treatment_effect": cohort.realized_average_treatment_effect,
        "recovery_error": recovery_error,
        "causal_estimate": estimate.estimate,
        "release_decision": {
            "released": release_decision.released,
            "overlap_valid": release_decision.overlap_valid,
            "balance_valid": release_decision.balance_valid,
            "uncertainty_valid": release_decision.uncertainty_valid,
            "temporal_pass": release_decision.temporal_pass,
            "causal_pass": release_decision.causal_pass,
            "clinical_pass": release_decision.clinical_pass,
            "reasons": list(release_decision.reasons),
        },
    }


#: The frozen label every synthetic result's section carries (Requirement
#: 16.5: "label synthetic results as failure-mode diagnostics and ... the
#: real-data result as the primary empirical result").
_SYNTHETIC_DIAGNOSTIC_LABEL: str = "failure_mode_diagnostic"

#: The label the real-data result's own row carries, for symmetry with
#: `_SYNTHETIC_DIAGNOSTIC_LABEL` -- `build_experiment_json`'s real-data
#: `results_table_rows` entry is implicitly this (it is the only entry in
#: that list), but `build_synthetic_benchmark_section`'s payload also
#: states it explicitly so a reader of the synthetic section alone still
#: sees the labeling rule, not just the synthetic half of it.
_PRIMARY_EMPIRICAL_RESULT_LABEL: str = "primary_empirical_result"


def build_synthetic_benchmark_section(
    master_seed: int, scenarios=None
) -> dict[str, Any]:
    """Build the `"synthetic_benchmark"` reporting section: one entry per
    scenario, each honestly labeled as a failure-mode diagnostic, never
    as the primary empirical result (Requirement 16.5).

    Args:
        master_seed: forwarded to `run_synthetic_benchmark_scenario` for
            every scenario (through `derive_seed`, so scenarios do not
            share a raw random stream).
        scenarios: the scenarios to run; defaults to
            `_default_synthetic_scenarios()`'s three frozen-by-this-module
            parameterizations covering `evaluation.synthetic_benchmark.SCENARIOS`.

    Returns:
        `{"real_data_result_label": _PRIMARY_EMPIRICAL_RESULT_LABEL,
        "scenarios": {scenario_name: run_synthetic_benchmark_scenario(...)}}`.
    """
    resolved_scenarios = scenarios if scenarios is not None else _default_synthetic_scenarios()
    return {
        "real_data_result_label": _PRIMARY_EMPIRICAL_RESULT_LABEL,
        "scenarios": {
            scenario.name: run_synthetic_benchmark_scenario(scenario, master_seed)
            for scenario in resolved_scenarios
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-data causal analysis and regenerate paper "
            "tables/figures/model card from its results (bridges "
            "run_causal_analysis.py's output into run_paper.py's "
            "--experiment JSON schema)."
        )
    )
    parser.add_argument("--real-data-path", type=Path, default=DEFAULT_REAL_DATA_PATH)
    parser.add_argument("--trial-config", type=Path, default=DEFAULT_TRIAL_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--master-seed", type=int, default=20240101)
    parser.add_argument(
        "--negative-control-n-bootstrap", type=int, default=NEGATIVE_CONTROL_N_BOOTSTRAP
    )
    parser.add_argument(
        "--treatment-effect-n-bootstrap", type=int, default=TREATMENT_EFFECT_N_BOOTSTRAP
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "real_data_reporting",
        help="Where run_paper.py's regenerated tables/figures/model card are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        experiment_json = build_experiment_json(
            args.real_data_path,
            args.trial_config,
            args.results_root,
            args.master_seed,
            negative_control_n_bootstrap=args.negative_control_n_bootstrap,
            treatment_effect_n_bootstrap=args.treatment_effect_n_bootstrap,
        )
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"run_real_data_reporting.py failed: {exc}", file=sys.stderr)
        return 1

    experiment_json_path = args.results_root / "real_data_experiment.json"
    experiment_json_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_json_path.write_text(
        json.dumps(experiment_json, indent=2, sort_keys=True, default=str, allow_nan=True)
    )
    print(f"wrote experiment JSON to {experiment_json_path}")

    exit_code = run_reproduction(args.trial_config, experiment_json_path, args.output_dir)
    if exit_code != 0:
        return exit_code

    print("run_real_data_reporting.py succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
