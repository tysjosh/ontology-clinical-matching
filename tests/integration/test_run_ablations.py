"""Integration test for `scripts/run_ablations.py` (task 12.1; Requirement 12.6).

Builds a small synthetic cohort (same hand-built-event pattern as
`tests/unit/test_pipeline.py`'s `_build_synthetic_inputs`, scaled down to
~25 patients per arm) and confirms `run_all_ablations(...)` runs all ten
frozen ablations (`evaluation.ablation_matrix.ABLATIONS`) without error
through the shared pipeline + evaluation harness, returning a dict with
exactly ten keys matching `ABLATIONS`' names, and that each ablation's
result file was actually written to disk under
`results_root/<experiment_id>/ablations/<name>/result.json` with the
expected JSON keys.

**Validates: Requirements 12.6**
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from evaluation.ablation_matrix import ABLATIONS
from protocols.schema import TargetTrialSpec

#: `scripts/` is not an installed package (see `tests/unit/test_scan_exports.py`
#: for the established convention this mirrors) -- add it to `sys.path`
#: directly so `run_ablations` can be imported here.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_ablations import _compute_harness_experiment_id, run_all_ablations  # noqa: E402

MAIN_SPEC_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = datetime(2020, 6, 4, 9, 0, 0)
END_OF_FOLLOWUP = datetime(2021, 1, 15, 9, 0, 0)


@pytest.fixture(scope="module")
def target_trial_spec() -> TargetTrialSpec:
    with MAIN_SPEC_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TargetTrialSpec.model_validate(raw)


def _event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="cd4_count",
        value=350.0,
        unit="cells/mm3",
        modality="lab",
        observed_at=WEEK_20,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


def _make_patient_events(
    patient_id: str,
    *,
    treatment_arm: int,
    baseline_cd4: float,
    karnofsky_score: float,
    age: float,
    symptomatic_status: bool,
    prior_opportunistic_infection: bool,
    prior_art_exposure_days: float,
) -> list[ClinicalEvent]:
    treatment_concept = "zidovudine" if treatment_arm == 0 else "zidovudine+didanosine"
    return [
        _event(
            patient_id,
            concept_id="hiv_positive",
            value=True,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="baseline_cd4",
            value=baseline_cd4,
            unit="cells/mm3",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="age",
            value=age,
            unit="years",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="karnofsky_score",
            value=karnofsky_score,
            unit="score",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="symptomatic_status",
            value=symptomatic_status,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="prior_opportunistic_infection",
            value=prior_opportunistic_infection,
            unit=None,
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="prior_art_exposure_days",
            value=prior_art_exposure_days,
            unit="days",
            modality="baseline",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id=treatment_concept,
            value=1.0,
            unit=None,
            modality="treatment",
            observed_at=ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="administrative_censoring",
            value=1.0,
            unit=None,
            modality="censoring",
            observed_at=END_OF_FOLLOWUP,
        ),
        _event(
            patient_id,
            concept_id="cd4_count",
            value=baseline_cd4 - 10.0,
            unit="cells/mm3",
            modality="lab",
            observed_at=WEEK_20,
        ),
    ]


def _build_small_synthetic_inputs(n_per_arm: int = 25):
    """`n_per_arm` eligible patients per treatment arm, varied enough to
    produce a genuine mix of retained/dropped patients under
    observationalization (same pattern as `tests/unit/test_pipeline.py`'s
    `_build_synthetic_inputs`, scaled down for a fast integration test)."""
    timelines = []
    treatment_arms: dict[str, int] = {}
    counter = 0
    for treatment_arm in (0, 1):
        for i in range(n_per_arm):
            counter += 1
            patient_id = f"P{counter:04d}"
            events = _make_patient_events(
                patient_id,
                treatment_arm=treatment_arm,
                baseline_cd4=200.0 + (i * 6) % 300,
                karnofsky_score=70.0 + (i % 30),
                age=18.0 + (i % 50),
                symptomatic_status=(i % 3 == 0),
                prior_opportunistic_infection=(i % 5 == 0),
                prior_art_exposure_days=float((i % 4) * 30),
            )
            timelines.append(build_patient_timeline(events))
            treatment_arms[patient_id] = treatment_arm
    return timelines, treatment_arms


@pytest.fixture
def synthetic_inputs():
    return _build_small_synthetic_inputs()


def test_run_all_ablations_runs_all_ten_without_error(
    synthetic_inputs, target_trial_spec, tmp_path
):
    timelines, treatment_arms = synthetic_inputs
    master_seed = 909090

    results = run_all_ablations(
        timelines, treatment_arms, target_trial_spec, master_seed, tmp_path
    )

    expected_names = {ablation.name for ablation in ABLATIONS}
    assert len(ABLATIONS) == 10
    assert set(results.keys()) == expected_names
    assert len(results) == 10


def test_result_files_written_with_expected_keys(synthetic_inputs, target_trial_spec, tmp_path):
    timelines, treatment_arms = synthetic_inputs
    master_seed = 424242

    results = run_all_ablations(
        timelines, treatment_arms, target_trial_spec, master_seed, tmp_path
    )

    experiment_id = _compute_harness_experiment_id(master_seed)

    for ablation in ABLATIONS:
        result_path = (
            tmp_path / experiment_id / "ablations" / ablation.name / "result.json"
        )
        assert result_path.is_file(), f"missing result file for ablation {ablation.name!r}"

        with result_path.open("r", encoding="utf-8") as handle:
            on_disk = json.load(handle)

        expected_keys = {"ablation_name", "config_diff", "concordance_index", "master_seed"}
        assert expected_keys <= set(on_disk.keys())
        assert on_disk["ablation_name"] == ablation.name
        assert on_disk["config_diff"] == ablation.config_diff
        assert on_disk["master_seed"] == master_seed

        # Matches the in-memory result returned by run_all_ablations.
        assert results[ablation.name]["ablation_name"] == on_disk["ablation_name"]
