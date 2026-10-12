"""Smoke test for `scripts/run_real_data_reporting.py`'s synthetic-
benchmark reporting section (task 19.2; Requirement 16.5, 16.6).

Deliberately separate from `tests/integration/test_run_real_data_reporting.py`,
which is skipped when the real ACTG175 data is absent -- the synthetic
benchmark section needs no real data at all, so it is checkable
unconditionally, same as `test_run_real_data_reporting_headline.py`.

**Validates: Requirements 16.5, 16.6**
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest  # noqa: E402
from run_real_data_reporting import (  # noqa: E402
    _PRIMARY_EMPIRICAL_RESULT_LABEL,
    _SYNTHETIC_DIAGNOSTIC_LABEL,
    build_synthetic_benchmark_section,
)

from evaluation.synthetic_benchmark import SCENARIOS  # noqa: E402


@pytest.fixture(scope="module")
def section():
    return build_synthetic_benchmark_section(master_seed=4242)


def test_section_carries_the_real_data_result_label(section):
    assert section["real_data_result_label"] == _PRIMARY_EMPIRICAL_RESULT_LABEL
    assert _PRIMARY_EMPIRICAL_RESULT_LABEL != _SYNTHETIC_DIAGNOSTIC_LABEL


def test_every_frozen_scenario_name_is_present(section):
    assert set(section["scenarios"]) == set(SCENARIOS)


def test_every_scenario_carries_label_recovery_error_and_release_decision(section):
    for name, result in section["scenarios"].items():
        assert result["scenario_name"] == name
        assert result["diagnostic_label"] == _SYNTHETIC_DIAGNOSTIC_LABEL
        # Either a clean run (recovery_error + release_decision present)
        # or a reported failure (error present) -- never both silently
        # missing.
        if "error" in result:
            assert isinstance(result["error"], str) and result["error"]
            continue
        assert isinstance(result["true_risk_difference"], float)
        assert isinstance(result["realized_average_treatment_effect"], float)
        assert isinstance(result["recovery_error"], float)
        assert result["recovery_error"] >= 0.0
        release_decision = result["release_decision"]
        for key in (
            "released",
            "overlap_valid",
            "balance_valid",
            "uncertainty_valid",
            "temporal_pass",
            "causal_pass",
            "clinical_pass",
        ):
            assert isinstance(release_decision[key], bool), (name, key)
        assert isinstance(release_decision["reasons"], list)
