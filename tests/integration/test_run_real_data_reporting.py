"""Integration test for `scripts/run_real_data_reporting.py` (bridges
`run_causal_analysis.py`'s real-data output into `run_paper.py`'s
`--experiment` JSON schema and reproduction stage).

Confirms:

- the built experiment JSON has real (non-NaN) values for every metric
  this pipeline can honestly compute (concordance/AUROC/AUPRC, estimated
  HR, log-HR error, direction agreement, real violation/witness counts),
  and explicit `NaN` for every metric it cannot (never a fabricated
  placeholder);
- `risk_difference_error` is always `NaN` -- this script must never read
  the sealed internal randomized reference, by design;
- `run_paper.py`'s reproduction stage runs successfully against the built
  JSON and writes real tables/figures/model-card artifacts;
- no sealed-reference-named file is created anywhere under the reporting
  output directory.

Skipped (not failed) when the real raw data is not present, matching the
other real-data integration tests' convention.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"
TRIAL_CONFIG_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

#: Small bootstrap replicate counts for this test module -- keeps runtime
#: reasonable; the logic under test does not depend on replicate count.
_TEST_N_BOOTSTRAP = 10

pytestmark = pytest.mark.skipif(
    not REAL_DATA_PATH.is_dir(),
    reason=(
        f"Real ACTG175 data not present at {REAL_DATA_PATH} (gitignored; see "
        "docs/data_access.md Section 2)."
    ),
)

from run_real_data_reporting import (  # noqa: E402
    UNSUPPORTED_METRIC_REASONS,
    build_experiment_json,
)


@pytest.fixture(scope="module")
def experiment_json(tmp_path_factory):
    results_root = tmp_path_factory.mktemp("reporting_results")
    return build_experiment_json(
        real_data_path=REAL_DATA_PATH,
        trial_config_path=TRIAL_CONFIG_PATH,
        results_root=results_root,
        master_seed=24680,
        negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
        treatment_effect_n_bootstrap=_TEST_N_BOOTSTRAP,
    )


def test_supported_factual_prediction_metrics_are_always_real_finite_numbers(experiment_json):
    """Factual-prediction metrics are reported regardless of the release
    gate -- predictive accuracy is separate from causal validity
    (Requirement 8) -- and real violation/witness counts are always
    integers, never NaN."""
    row = experiment_json["results_table_rows"][0]

    for column in ("concordance_index", "auroc_96wk", "auprc_96wk"):
        assert isinstance(row[column], float), column
        assert math.isfinite(row[column]), column

    assert isinstance(row["temporal_violations"], int)
    assert isinstance(row["protocol_violations"], int)
    assert isinstance(row["witness_detected_failures"], int)
    assert row["n"] > 0
    assert isinstance(row["release_gate_passed"], bool)


def test_causal_recovery_metrics_match_the_release_gate_exactly(experiment_json):
    """`estimated_hr`/`e_log_hr`/`effect_direction_agreement`/`ci_overlap`
    are real numbers when (and only when) `release_gate_passed` is True,
    and explicit NaN/None -- never a fabricated number -- when it is
    False (decisions.md Section 16's release condition)."""
    row = experiment_json["results_table_rows"][0]

    if row["release_gate_passed"]:
        assert isinstance(row["estimated_hr"], float)
        assert math.isfinite(row["estimated_hr"])
        assert isinstance(row["e_log_hr"], float)
        assert math.isfinite(row["e_log_hr"])
        assert isinstance(row["effect_direction_agreement"], bool)
        assert "treatment_effect_estimate" in experiment_json
    else:
        assert math.isnan(row["estimated_hr"])
        assert math.isnan(row["e_log_hr"])
        assert math.isnan(row["effect_direction_agreement"])
        assert "treatment_effect_estimate" not in experiment_json
        for column in ("estimated_hr", "e_log_hr", "effect_direction_agreement", "ci_overlap"):
            assert column in experiment_json["unsupported_metrics"]
            assert "release" in experiment_json["unsupported_metrics"][column].lower()


def test_unsupported_metrics_are_explicit_nan_not_fabricated(experiment_json):
    row = experiment_json["results_table_rows"][0]

    for column in UNSUPPORTED_METRIC_REASONS:
        assert math.isnan(row[column]), column

    assert set(UNSUPPORTED_METRIC_REASONS) <= set(experiment_json["unsupported_metrics"])
    for reason in experiment_json["unsupported_metrics"].values():
        assert isinstance(reason, str) and reason


def test_headline_comparisons_are_named_but_unsupported_here(experiment_json):
    """This script trains no neural model and holds no locked-evaluation
    test-split pairing, so all five headline comparisons (task 22.1,
    Requirement 17.1-17.5) are present and named but explicitly marked
    unsupported here -- `scripts/run_locked_evaluation.py` is the real
    call site that fills them in with paired-bootstrap results."""
    from reporting.tables import HEADLINE_COMPARISON_NAMES

    headline = experiment_json["headline_comparisons"]
    assert set(headline) == set(HEADLINE_COMPARISON_NAMES)
    for name, entry in headline.items():
        assert entry["supported"] is False, name
        assert isinstance(entry["reason"], str) and entry["reason"]


def test_executed_results_includes_every_sensitivity_variant(experiment_json):
    """Task 22.2 (Requirement 19.3): every one of the eight frozen
    sensitivity variants appears in the payload's `executed_results`,
    unfiltered."""
    executed = experiment_json["executed_results"]
    assert executed["n_arms"] == 8
    names = {arm["name"] for arm in executed["arms"]}
    assert len(names) == 8


def test_blocked_outcome_present_only_when_the_real_release_decision_blocks(experiment_json):
    """Task 22.2 (Requirement 19.1): since this script conservatively
    supplies `uncertainty_valid=False` to the real six-way release gate
    (see run_causal_analysis.py's own docstring), the gate never passes
    here, so `blocked_outcome` is always present with every failing
    condition and both population counts."""
    assert "blocked_outcome" in experiment_json
    blocked = experiment_json["blocked_outcome"]
    assert blocked["blocked"] is True
    assert len(blocked["failing_conditions"]) > 0
    assert blocked["supported_count"] > 0
    assert isinstance(blocked["abstained_count"], int)


def test_risk_difference_error_is_never_computed_from_the_sealed_reference(experiment_json):
    """The sealed internal randomized reference must never be read to
    produce this column -- it is always NaN, with a reason explaining why
    this is by design, not a missing feature. This holds unconditionally,
    regardless of the release gate's outcome."""
    row = experiment_json["results_table_rows"][0]
    assert math.isnan(row["risk_difference_error"])
    assert "risk_difference_error" in experiment_json["unsupported_metrics"]
    reason = experiment_json["unsupported_metrics"]["risk_difference_error"]
    assert "decisions.md Section 8" in reason


def test_release_gate_passing_forces_all_causal_columns_present(tmp_path):
    """Force the release-gate-passed branch explicitly (rather than relying
    on whatever the real cohort happens to produce for a given seed) by
    monkeypatching `run_causal_analysis` to report a passing decision, and
    confirm every causal-recovery column is then a real number and
    `treatment_effect_estimate` is included."""
    import run_real_data_reporting as reporting_module

    original = reporting_module.run_causal_analysis

    def _force_passing(*args, **kwargs):
        summary = original(*args, **kwargs)
        summary["aggregate_decision_passed"] = True
        return summary

    reporting_module.run_causal_analysis = _force_passing
    try:
        payload = reporting_module.build_experiment_json(
            real_data_path=REAL_DATA_PATH,
            trial_config_path=TRIAL_CONFIG_PATH,
            results_root=tmp_path,
            master_seed=98765,
            negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
            treatment_effect_n_bootstrap=_TEST_N_BOOTSTRAP,
        )
    finally:
        reporting_module.run_causal_analysis = original

    row = payload["results_table_rows"][0]
    assert row["release_gate_passed"] is True
    assert math.isfinite(row["estimated_hr"])
    assert math.isfinite(row["e_log_hr"])
    assert isinstance(row["effect_direction_agreement"], bool)
    assert "treatment_effect_estimate" in payload
    estimate = payload["treatment_effect_estimate"]
    assert math.isfinite(estimate["point_estimate"])
    assert estimate["ci_low"] <= estimate["point_estimate"] <= estimate["ci_high"]
    assert estimate["n_replicates"] == _TEST_N_BOOTSTRAP


def test_release_gate_failing_forces_all_causal_columns_suppressed(tmp_path):
    """Force the release-gate-failed branch explicitly and confirm every
    causal-recovery column is NaN/None and `treatment_effect_estimate` is
    absent -- the direct regression test for the bug where a failed/
    uncertain witness result did not suppress the published estimate."""
    import run_real_data_reporting as reporting_module

    original = reporting_module.run_causal_analysis

    def _force_failing(*args, **kwargs):
        summary = original(*args, **kwargs)
        summary["aggregate_decision_passed"] = False
        return summary

    reporting_module.run_causal_analysis = _force_failing
    try:
        payload = reporting_module.build_experiment_json(
            real_data_path=REAL_DATA_PATH,
            trial_config_path=TRIAL_CONFIG_PATH,
            results_root=tmp_path,
            master_seed=98765,
            negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
            treatment_effect_n_bootstrap=_TEST_N_BOOTSTRAP,
        )
    finally:
        reporting_module.run_causal_analysis = original

    row = payload["results_table_rows"][0]
    assert row["release_gate_passed"] is False
    assert math.isnan(row["estimated_hr"])
    assert math.isnan(row["e_log_hr"])
    assert math.isnan(row["effect_direction_agreement"])
    assert row["ci_overlap"] is None or (
        isinstance(row["ci_overlap"], float) and math.isnan(row["ci_overlap"])
    )
    assert "treatment_effect_estimate" not in payload


def test_run_paper_reproduction_succeeds_on_the_built_experiment_json(tmp_path):
    from run_paper import run_reproduction

    results_root = tmp_path / "results"
    output_dir = tmp_path / "reporting"

    payload = build_experiment_json(
        real_data_path=REAL_DATA_PATH,
        trial_config_path=TRIAL_CONFIG_PATH,
        results_root=results_root,
        master_seed=11111,
        negative_control_n_bootstrap=_TEST_N_BOOTSTRAP,
        treatment_effect_n_bootstrap=_TEST_N_BOOTSTRAP,
    )
    experiment_json_path = results_root / "real_data_experiment.json"
    experiment_json_path.write_text(json.dumps(payload, indent=2, default=str))

    # task 23.2: manuscript_dir defaults to the real paper/generated/;
    # pointed at tmp_path here so this test never writes into the real
    # manuscript package's working tree.
    manuscript_dir = tmp_path / "manuscript_generated"
    exit_code = run_reproduction(
        TRIAL_CONFIG_PATH, experiment_json_path, output_dir, manuscript_dir
    )

    assert exit_code == 0
    assert (output_dir / "tables" / "main_results_table.json").is_file()
    assert (output_dir / "model_card.json").is_file()

    # No sealed-reference-named file was created anywhere under the
    # reporting output directory.
    sealed_files = list(output_dir.rglob("*sealed*")) + list(
        output_dir.rglob("*internal_randomized_reference*")
    )
    assert sealed_files == []
