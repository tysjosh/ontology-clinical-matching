"""Shared synthetic-cohort fixture helper for `scripts/run_stage.py`'s
non-`smoke` stages (task 18).

Every non-`smoke` stage in `run_stage.py` needs a small, varied synthetic
patient cohort to exercise real pipeline stages end to end. Rather than
duplicate the hand-built-event fixture pattern already established by
`tests/unit/test_pipeline.py`'s `_build_synthetic_inputs` and
`tests/integration/test_run_ablations.py`'s `_build_small_synthetic_inputs`
a third time inline in `run_stage.py`, this module factors that exact same
pattern out into one place both this script and any future caller can
reuse. This is deliberately NOT real ACTG175 data (`decisions.md`
Section 2) -- the dataset checksum implied by these fixtures is a
placeholder, matching `run_stage.py`'s existing `smoke` stage's own
documented convention.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import PatientTimeline, build_patient_timeline
from protocols.schema import TargetTrialSpec

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The frozen, complete target-trial spec (`configs/target_trials/main.yaml`)
#: -- the same file `tests/unit/test_pipeline.py` and
#: `tests/integration/test_run_ablations.py` validate against. Loading this
#: (rather than the lightweight `tests/fixtures/config/valid_target_trial.yaml`
#: the `smoke` stage uses) is required here because these stages call the
#: richly-typed `protocols.schema.TargetTrialSpec`-consuming pipeline
#: functions directly (`cohorts.pipeline.run_observationalization_pipeline`,
#: etc.), not just `infra.config.load_config`.
MAIN_TARGET_TRIAL_CONFIG = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = datetime(2020, 6, 4, 9, 0, 0)
END_OF_FOLLOWUP = datetime(2021, 1, 15, 9, 0, 0)


def load_main_target_trial_spec(path: Path = MAIN_TARGET_TRIAL_CONFIG) -> TargetTrialSpec:
    """Load and validate the frozen `configs/target_trials/main.yaml` spec."""
    with path.open("r", encoding="utf-8") as handle:
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
            concept_id="baseline_cd8",
            value=baseline_cd4 * 2.0,
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


def build_synthetic_pipeline_inputs(
    n_per_arm: int = 40,
) -> tuple[list[PatientTimeline], dict[str, int]]:
    """`n_per_arm` eligible synthetic patients per treatment arm, varied
    enough that observationalization retains a genuine mix rather than an
    all-or-nothing outcome.

    Identical fixture pattern to `tests/unit/test_pipeline.py`'s
    `_build_synthetic_inputs` / `tests/integration/test_run_ablations.py`'s
    `_build_small_synthetic_inputs` -- factored out here so `run_stage.py`
    and those tests share one source of truth rather than three
    hand-copied fixtures.
    """
    timelines: list[PatientTimeline] = []
    treatment_arms: dict[str, int] = {}
    counter = 0
    for treatment_arm in (0, 1):
        for i in range(n_per_arm):
            counter += 1
            patient_id = f"P{counter:04d}"
            events = _make_patient_events(
                patient_id,
                treatment_arm=treatment_arm,
                baseline_cd4=200.0 + (i * 6) % 300,  # stays within [200, 500]
                karnofsky_score=70.0 + (i % 30),
                age=18.0 + (i % 50),
                symptomatic_status=(i % 3 == 0),
                prior_opportunistic_infection=(i % 5 == 0),
                prior_art_exposure_days=float((i % 4) * 30),
            )
            timelines.append(build_patient_timeline(events))
            treatment_arms[patient_id] = treatment_arm
    return timelines, treatment_arms
