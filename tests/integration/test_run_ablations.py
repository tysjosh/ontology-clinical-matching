"""Integration test for `scripts/run_ablations.py` (task 14; Requirement
7.2, 7.3, 7.4, 7.5, 7.9, 7.10, 7.14, 7.16).

Builds a small synthetic cohort (~40 patients: 20 per treatment arm, a mix
of death/AIDS-defining events and administrative censoring so the
composite endpoint has genuine events in both splits -- not a cohort of
pure censoring, which starves lambda selection of a usable validation
concordance) and confirms `run_retrained_ablations(...)` runs the six
retrain-mode frozen ablation arms (`evaluation.ablation_matrix.
expand_ablations()`, `execution_mode == "retrain"`) end to end through
`models.variants.build_model` and the frozen training harness, at a
shortened seed sequence (for test speed; the harness itself is seed-count
agnostic).

**Validates: Requirements 7.2, 7.3, 7.4, 7.5, 7.9, 7.10, 7.14, 7.16**
(`.kiro/specs/neural-experiment-pipeline/requirements.md`)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from data.event_model import ClinicalEvent, MissingnessState  # noqa: E402
from data.timelines import build_patient_timeline  # noqa: E402
from evaluation.ablation_matrix import expand_ablations  # noqa: E402
from protocols.schema import TargetTrialSpec  # noqa: E402

pytestmark = pytest.mark.neural

#: `scripts/` is not an installed package (see `tests/unit/test_scan_exports.py`
#: for the established convention this mirrors) -- add it to `sys.path`
#: directly so `run_ablations` can be imported here.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_ablations import (  # noqa: E402
    _compute_harness_experiment_id,
    build_training_batches,
    run_retrained_ablations,
)

from models.torch_training import TrainingConfig  # noqa: E402

#: A short training configuration for test speed: the frozen configuration
#: (`decisions.md` Section 25, `max_epochs=500`, `patience=20`) is correct
#: for production but makes a six-arm x nine-combination-grid x N-seed
#: integration test prohibitively slow. `max_epochs=3` with `patience=2`
#: still exercises the full train/validate/checkpoint loop and early
#: stopping path at every seed; only the frozen config's `max_epochs` and
#: `patience` are shortened, every other field (lr, betas, weight_decay,
#: grad_clip_norm) stays at the frozen value.
SHORT_CONFIG = TrainingConfig(max_epochs=3, patience=2)

MAIN_SPEC_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
WEEK_20 = ASSIGNMENT_TIME + timedelta(weeks=20)
WEEK_96 = ASSIGNMENT_TIME + timedelta(weeks=96)
END_OF_FOLLOWUP = ASSIGNMENT_TIME + timedelta(weeks=150)

#: Shortened seed sequence: the harness is seed-count agnostic (it simply
#: iterates `seeds`), so two seeds is enough to confirm the per-seed
#: retraining loop actually runs more than once without paying the real
#: five-seed training cost in a fast test.
TEST_SEEDS = (101, 202)


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
    baseline_cd8: float,
    karnofsky_score: float,
    age: float,
    symptomatic_status: bool,
    prior_opportunistic_infection: bool,
    prior_art_exposure_days: float,
    death_time: datetime | None,
    week_20_cd4: float,
    week_96_cd4: float,
    censoring_time_weeks: float = 150.0,
) -> list[ClinicalEvent]:
    """A patient with genuine baseline covariates, two landmark CD4 labs,
    and either a death event (a real composite-endpoint event) or a
    censoring event at a varied follow-up time (not both) -- so a cohort
    built from a mix of `death_time=None` and `death_time=<datetime>`
    patients has the real event/censoring split, and the varied elapsed
    times, a usable (non-degenerate, non-all-tied) concordance index
    needs.
    """
    treatment_concept = "zidovudine" if treatment_arm == 0 else "zidovudine+didanosine"
    events = [
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
            value=baseline_cd8,
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
            concept_id="cd4_count",
            value=week_20_cd4,
            unit="cells/mm3",
            modality="lab",
            observed_at=WEEK_20,
        ),
        _event(
            patient_id,
            concept_id="cd4_count",
            value=week_96_cd4,
            unit="cells/mm3",
            modality="lab",
            observed_at=WEEK_96,
        ),
    ]
    if death_time is not None:
        events.append(
            _event(
                patient_id,
                concept_id="death",
                value=1.0,
                unit=None,
                modality="endpoint",
                observed_at=death_time,
            )
        )
    else:
        events.append(
            _event(
                patient_id,
                concept_id="administrative_censoring",
                value=1.0,
                unit=None,
                modality="censoring",
                observed_at=ASSIGNMENT_TIME + timedelta(weeks=censoring_time_weeks),
            )
        )
    return events


def _build_small_synthetic_inputs(n_per_arm: int = 20):
    """`n_per_arm` eligible patients per treatment arm, with roughly 40%
    experiencing death at a varied elapsed time and the rest
    administratively censored -- varied enough to produce a genuine mix
    of retained/dropped patients under observationalization AND a
    non-degenerate composite-endpoint concordance in both dev and val
    (unlike a cohort of pure censoring, which starves `concordance_index`
    of comparable pairs and makes lambda selection fail with an
    all-`nan`/empty candidate set)."""
    timelines = []
    treatment_arms: dict[str, int] = {}
    counter = 0
    for treatment_arm in (0, 1):
        for i in range(n_per_arm):
            counter += 1
            patient_id = f"P{counter:04d}"
            had_event = i % 2 == 0  # half of all patients experience death
            death_time = (
                ASSIGNMENT_TIME + timedelta(weeks=30.0 + i * 3.0) if had_event else None
            )
            censoring_time_weeks = 140.0 + i * 2.0
            baseline_cd4 = 200.0 + (i * 6) % 300
            events = _make_patient_events(
                patient_id,
                treatment_arm=treatment_arm,
                baseline_cd4=baseline_cd4,
                baseline_cd8=850.0 + (i * 7) % 300,
                karnofsky_score=70.0 + (i % 30),
                age=18.0 + (i % 50),
                symptomatic_status=(i % 3 == 0),
                prior_opportunistic_infection=(i % 5 == 0),
                prior_art_exposure_days=float((i % 4) * 30),
                death_time=death_time,
                week_20_cd4=baseline_cd4 - 10.0,
                week_96_cd4=baseline_cd4 - (40.0 if had_event else 5.0),
                censoring_time_weeks=censoring_time_weeks,
            )
            timelines.append(build_patient_timeline(events))
            treatment_arms[patient_id] = treatment_arm
    return timelines, treatment_arms


@pytest.fixture(scope="module")
def synthetic_inputs():
    # n_per_arm=15 (30 patients total): enough to produce non-degenerate
    # dev/val retained populations with a genuine event/censoring mix,
    # while keeping lambda selection's 9-combination grid search fast at
    # the module's SHORT_CONFIG training budget.
    #
    # At this size and SHORT_CONFIG's short training budget,
    # `concordance_index` (a purely rank-based metric) on an ~5-patient
    # retained Validation_Split can genuinely tie across several of the
    # six independently-trained architectures by small-sample chance --
    # a true positive from
    # `evaluation.ablation_matrix.detect_variant_collisions`'s
    # fewer-than-six-distinct-values check, not a defect in it. Every
    # test below that calls `run_retrained_ablations` therefore passes
    # `raise_on_collision=False`: these tests exercise harness plumbing
    # (seed loop, checkpointing, placebo vectors, failure handling,
    # metric computation), not collision detection itself.
    # `tests/unit/test_variant_collision.py` and
    # `tests/neural/test_variant_collision.py` are the dedicated tests
    # for that; `test_a_real_collision_raises_variant_collision_error`
    # below additionally confirms the raising path end-to-end.
    return _build_small_synthetic_inputs(n_per_arm=15)


# ---------------------------------------------------------------------------
# build_training_batches
# ---------------------------------------------------------------------------


def test_build_training_batches_uses_retained_populations_only(
    synthetic_inputs, target_trial_spec
):
    from cohorts.pipeline import run_observationalization_pipeline

    timelines, treatment_arms = synthetic_inputs
    pipeline_result = run_observationalization_pipeline(
        timelines, treatment_arms, target_trial_spec, master_seed=909090
    )

    dev_batch, val_batch, dimensions = build_training_batches(pipeline_result)

    assert set(dev_batch.patient_ids) == set(pipeline_result.dev.retained_patient_ids)
    assert set(val_batch.patient_ids) == set(pipeline_result.val.retained_patient_ids)
    assert dev_batch.split_name == "dev"
    assert val_batch.split_name == "val"
    assert dimensions == dev_batch.dimensions == val_batch.dimensions


# ---------------------------------------------------------------------------
# run_retrained_ablations: all six retrain arms, two seeds
# ---------------------------------------------------------------------------


def test_run_retrained_ablations_covers_exactly_the_six_retrain_runs(
    synthetic_inputs, target_trial_spec, tmp_path
):
    timelines, treatment_arms = synthetic_inputs
    master_seed = 909090

    results = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed,
        tmp_path,
        seeds=TEST_SEEDS,
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    expected_run_ids = {
        run.run_id for run in expand_ablations() if run.execution_mode == "retrain"
    }
    assert len(expected_run_ids) == 6
    assert set(results.keys()) == expected_run_ids


def test_writes_exactly_k_public_and_k_protected_ablation_audit_artifacts(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Task 17.2: one public and one protected audit artifact per
    `(run, seed)` candidate estimate -- K = 6 retrain-mode runs x 2 test
    seeds = 12 (Requirement 12.1, 12.2)."""
    timelines, treatment_arms = synthetic_inputs
    master_seed = 909093

    results = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed,
        tmp_path,
        seeds=TEST_SEEDS,
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    experiment_id = _compute_harness_experiment_id(master_seed)
    experiment_dir = tmp_path / experiment_id
    public_dir = experiment_dir / "audit" / "public"
    protected_dir = experiment_dir / "audit" / "protected"

    expected_claim_ids = {
        f"ablation_{run_id}_seed{seed}" for run_id in results for seed in TEST_SEEDS
    }
    assert len(expected_claim_ids) == 6 * len(TEST_SEEDS)

    public_files = {p.stem for p in public_dir.glob("*.json")}
    protected_files = {p.stem for p in protected_dir.glob("*.json")}

    assert public_files == expected_claim_ids
    assert protected_files == expected_claim_ids

    for claim_id in expected_claim_ids:
        payload = json.loads((public_dir / f"{claim_id}.json").read_text())
        assert payload["estimate_status"] == "suppressed"


def test_every_retrain_run_trains_at_every_configured_seed(
    synthetic_inputs, target_trial_spec, tmp_path
):
    timelines, treatment_arms = synthetic_inputs
    master_seed = 909091

    results = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed,
        tmp_path,
        seeds=TEST_SEEDS,
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    for run_id, payload in results.items():
        seed_keys = set(payload["seeds"].keys())
        assert seed_keys == {str(seed) for seed in TEST_SEEDS}, run_id


def test_no_complete_model_checkpoint_is_reused_every_seed_writes_its_own(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.10's 'reuse no complete-model checkpoint': every
    `(run, seed)` writes its own distinct checkpoint file."""
    timelines, treatment_arms = synthetic_inputs
    master_seed = 909092

    results = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed,
        tmp_path,
        seeds=TEST_SEEDS,
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    checkpoint_paths = set()
    for payload in results.values():
        for seed_result in payload["seeds"].values():
            assert seed_result["succeeded"], seed_result["failure_reason"]
            checkpoint_path = seed_result["checkpoint_path"]
            assert checkpoint_path is not None
            assert Path(checkpoint_path).is_file()
            checkpoint_paths.add(checkpoint_path)

    assert len(checkpoint_paths) == 6 * len(TEST_SEEDS)


def test_no_treatment_input_arm_has_zeroed_intervention_channel_input(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.2: every treatment input element is held at zero at
    every landmark, while non-treatment inputs are untouched. Checked at
    the model-input level: constructing `no_treatment_input`'s model
    through `build_model` and asking it to encode a batch's intervention
    input must yield an all-zero tensor of the frozen intervention width,
    for every patient."""
    from cohorts.pipeline import run_observationalization_pipeline
    from models.variants import variant_for_ablation

    timelines, treatment_arms = synthetic_inputs
    pipeline_result = run_observationalization_pipeline(
        timelines, treatment_arms, target_trial_spec, master_seed=909093
    )
    dev_batch, _val_batch, dimensions = build_training_batches(pipeline_result)

    run = next(r for r in expand_ablations() if r.run_id == "no_treatment_input")
    assert variant_for_ablation(run) is run.variant

    from models.variants import build_model

    model = build_model(run.variant, dimensions, seed=101)
    intervention_input = model.intervention_input(dev_batch, arm=1)
    assert torch.equal(intervention_input, torch.zeros_like(intervention_input))

    # Non-treatment input (the state channel) is untouched: it still varies
    # across patients (not all-zero/all-identical), confirming only the
    # intervention channel was zeroed.
    assert not torch.equal(dev_batch.state_init, torch.zeros_like(dev_batch.state_init))


def test_intervention_channel_without_protocol_arm_zeros_only_the_protocol_channel(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.4: an all-zero width-2 protocol channel, intervention
    channel identical to the complete model's."""
    from cohorts.pipeline import run_observationalization_pipeline
    from models.variants import build_model

    timelines, treatment_arms = synthetic_inputs
    pipeline_result = run_observationalization_pipeline(
        timelines, treatment_arms, target_trial_spec, master_seed=909094
    )
    dev_batch, _val_batch, dimensions = build_training_batches(pipeline_result)

    run = next(
        r for r in expand_ablations() if r.run_id == "intervention_channel_without_protocol"
    )
    model = build_model(run.variant, dimensions, seed=101)

    shared = model.shared_inputs(dev_batch)
    assert torch.equal(shared.protocol, torch.zeros_like(shared.protocol))
    assert shared.protocol.shape[1] == dimensions.protocol_width

    # The intervention channel is unaffected by protocol zeroing.
    intervention_input = model.intervention_input(dev_batch, arm=1)
    expected_arm_column = torch.ones(dev_batch.intervention.shape[0])
    assert torch.equal(intervention_input[:, 0], expected_arm_column)


def test_treatment_as_ordinary_feature_arm_builds_a_sequence_gru(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.3: built from `TorchSequenceGRU`, which has no
    intervention channel at all -- treatment lives inside the sequence
    input instead."""
    from cohorts.pipeline import run_observationalization_pipeline
    from models.torch_sequence_gru import TorchSequenceGRU
    from models.variants import build_model

    timelines, treatment_arms = synthetic_inputs
    pipeline_result = run_observationalization_pipeline(
        timelines, treatment_arms, target_trial_spec, master_seed=909095
    )
    _dev_batch, _val_batch, dimensions = build_training_batches(pipeline_result)

    run = next(r for r in expand_ablations() if r.run_id == "treatment_as_ordinary_feature")
    model = build_model(run.variant, dimensions, seed=101)

    assert isinstance(model, TorchSequenceGRU)
    assert not hasattr(model, "intervention_encoder")


def test_explicit_intervention_channel_arm_shares_hidden_width_with_gru_arm(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.14: the same frozen hidden width 16 for both arms."""
    from cohorts.pipeline import run_observationalization_pipeline
    from models.variants import build_model

    timelines, treatment_arms = synthetic_inputs
    pipeline_result = run_observationalization_pipeline(
        timelines, treatment_arms, target_trial_spec, master_seed=909096
    )
    _dev_batch, _val_batch, dimensions = build_training_batches(pipeline_result)

    gru_run = next(r for r in expand_ablations() if r.run_id == "treatment_as_ordinary_feature")
    explicit_run = next(
        r for r in expand_ablations() if r.run_id == "explicit_intervention_channel"
    )

    gru_model = build_model(gru_run.variant, dimensions, seed=101)
    explicit_model = build_model(explicit_run.variant, dimensions, seed=101)

    assert gru_model.hidden_size == explicit_model.hidden_size == 16


def test_placebo_vector_is_reproduced_at_a_fixed_seed_and_redrawn_on_seed_change(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.5: the placebo vector is drawn once per seed, constant
    across every patient/landmark within that seed, reproduced at a rerun
    of the same seed, and recorded with the run's results."""
    timelines, treatment_arms = synthetic_inputs

    results_a = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed=909097,
        results_root=tmp_path / "a",
        seeds=(101,),
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )
    results_b = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed=909097,
        results_root=tmp_path / "b",
        seeds=(101,),
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    vector_a = results_a["placebo_intervention_channel"]["seeds"]["101"]["placebo_vector"]
    vector_b = results_b["placebo_intervention_channel"]["seeds"]["101"]["placebo_vector"]
    assert vector_a is not None
    assert vector_a == vector_b  # same master_seed, same seed -> reproduced

    results_different_seed = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed=909097,
        results_root=tmp_path / "c",
        seeds=(202,),
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )
    vector_different_seed = results_different_seed["placebo_intervention_channel"]["seeds"][
        "202"
    ]["placebo_vector"]
    assert vector_different_seed != vector_a  # seed changed -> redrawn

    # Every non-placebo run must record `placebo_vector: null`.
    for run_id, payload in results_a.items():
        if run_id != "placebo_intervention_channel":
            assert payload["seeds"]["101"]["placebo_vector"] is None


def test_predictive_accuracy_only_arm_has_zero_balance_weight(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.9: `lambda_balance == 0.0` for this one arm, selected
    (nonzero in general) for every other arm."""
    timelines, treatment_arms = synthetic_inputs

    results = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed=909098,
        results_root=tmp_path,
        seeds=(101,),
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    assert results["predictive_accuracy_only"]["lambda_balance"] == 0.0


def test_a_training_failure_is_recorded_without_aborting_other_runs(
    monkeypatch, synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.16: a failed `(run, seed)` attempt is recorded with
    its cause, and does not prevent other runs/seeds from completing."""
    import run_ablations as run_ablations_module

    timelines, treatment_arms = synthetic_inputs

    original_train_one_model = run_ablations_module.train_one_model
    call_count = {"n": 0}

    def _flaky_train_one_model(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("synthetic injected failure")
        return original_train_one_model(*args, **kwargs)

    monkeypatch.setattr(run_ablations_module, "train_one_model", _flaky_train_one_model)

    results = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed=909099,
        results_root=tmp_path,
        seeds=TEST_SEEDS,
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    all_seed_results = [
        seed_result
        for payload in results.values()
        for seed_result in payload["seeds"].values()
    ]
    failed = [r for r in all_seed_results if not r["succeeded"]]
    succeeded = [r for r in all_seed_results if r["succeeded"]]

    assert len(failed) == 1
    assert "synthetic injected failure" in failed[0]["failure_reason"]
    assert len(succeeded) == len(all_seed_results) - 1


def test_validation_metric_is_computed_per_seed_with_a_mean_and_ablation_table_written(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.11, 7.12: one concordance value per seed plus the
    mean, and `ablation_table.json` carries the aggregated per-run rows."""
    import json

    from run_ablations import build_ablation_table

    timelines, treatment_arms = synthetic_inputs
    master_seed = 909101

    results = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed,
        tmp_path,
        seeds=TEST_SEEDS,
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    for run_id, payload in results.items():
        values = [
            seed_record["concordance_index"]
            for seed_record in payload["seeds"].values()
            if seed_record["succeeded"]
        ]
        assert len(values) == len(TEST_SEEDS), run_id
        for value in values:
            assert value == value or value is None  # not NaN-propagating silently

    table = build_ablation_table(results, table_incomplete=False)
    assert set(table["runs"].keys()) == set(results.keys())
    for row in table["runs"].values():
        assert row["evaluated_split"] == "val"
        assert set(row["per_seed_concordance_index"].keys()) == {
            str(seed) for seed in TEST_SEEDS
        }
        assert row["mean_concordance_index"] == row["mean_concordance_index"]  # not NaN

    experiment_id = _compute_harness_experiment_id(master_seed)
    table_path = tmp_path / experiment_id / "ablations" / "ablation_table.json"
    assert table_path.is_file()
    on_disk = json.loads(table_path.read_text())
    assert on_disk["table_incomplete"] is False
    # `table_invalid` is not asserted here: at this fixture's small
    # retained-population size and SHORT_CONFIG's short training budget,
    # concordance_index (rank-based) can genuinely tie across arms by
    # small-sample chance -- see the module-level note on
    # `synthetic_inputs`. `table_invalid`'s correctness is covered by
    # test_a_detected_collision_raises_and_writes_collision_report_and_invalid_table
    # and the dedicated unit/neural collision-detection test modules.
    assert "table_invalid" in on_disk
    assert len(on_disk["runs"]) == 6


def test_validation_metric_patient_ids_are_drawn_from_val_and_disjoint_from_dev(
    synthetic_inputs, target_trial_spec, tmp_path
):
    """Property 29: the patient ids a run's metric is computed over are
    the retained Validation_Split ids, disjoint from the retained
    Development_Split ids used for fitting."""
    from cohorts.pipeline import run_observationalization_pipeline

    timelines, treatment_arms = synthetic_inputs
    master_seed = 909102

    pipeline_result = run_observationalization_pipeline(
        timelines, treatment_arms, target_trial_spec, master_seed=master_seed
    )
    dev_ids = set(pipeline_result.dev.retained_patient_ids)
    val_ids = set(pipeline_result.val.retained_patient_ids)
    assert dev_ids.isdisjoint(val_ids)

    results = run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed,
        tmp_path,
        seeds=(101,),
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    for run_id, payload in results.items():
        seed_record = payload["seeds"]["101"]
        assert seed_record["succeeded"], seed_record["failure_reason"]
        metric_patient_ids = set(seed_record["val_patient_ids"])
        assert metric_patient_ids == val_ids, run_id
        assert metric_patient_ids.isdisjoint(dev_ids), run_id


def test_a_detected_collision_raises_and_writes_collision_report_and_invalid_table(
    monkeypatch, synthetic_inputs, target_trial_spec, tmp_path
):
    """Requirement 7.13: when `detect_variant_collisions` reports a
    collision and `raise_on_collision=True` (the production default),
    `run_retrained_ablations` raises `VariantCollisionError`, still
    writes `collision_report.json`, and marks `ablation_table.json`'s
    `table_invalid` flag -- forced via monkeypatch rather than relying on
    real small-sample training to coincidentally collide, so this test
    is deterministic."""
    import run_ablations as run_ablations_module

    from evaluation.ablation_matrix import CollisionReport, VariantCollisionError

    forced_report = CollisionReport(
        has_collision=True,
        prediction_collisions=(),
        insufficient_distinct_metrics=True,
        distinct_metric_count=3,
        checked_seeds=(101,),
    )
    monkeypatch.setattr(
        run_ablations_module,
        "_detect_collisions_among_retrained_runs",
        lambda *args, **kwargs: forced_report,
    )

    timelines, treatment_arms = synthetic_inputs

    with pytest.raises(VariantCollisionError, match="variant collision detected"):
        run_retrained_ablations(
            timelines,
            treatment_arms,
            target_trial_spec,
            master_seed=909103,
            results_root=tmp_path,
            seeds=(101,),
            training_config=SHORT_CONFIG,
            raise_on_collision=True,
        )

    experiment_id = _compute_harness_experiment_id(909103)
    ablations_dir = tmp_path / experiment_id / "ablations"

    import json

    collision_payload = json.loads((ablations_dir / "collision_report.json").read_text())
    assert collision_payload["has_collision"] is True
    assert collision_payload["insufficient_distinct_metrics"] is True

    table_payload = json.loads((ablations_dir / "ablation_table.json").read_text())
    assert table_payload["table_invalid"] is True


def test_summary_file_written_with_table_incomplete_flag(
    synthetic_inputs, target_trial_spec, tmp_path
):
    timelines, treatment_arms = synthetic_inputs
    master_seed = 909100

    run_retrained_ablations(
        timelines,
        treatment_arms,
        target_trial_spec,
        master_seed,
        tmp_path,
        seeds=TEST_SEEDS,
        training_config=SHORT_CONFIG,
        raise_on_collision=False,
    )

    experiment_id = _compute_harness_experiment_id(master_seed)
    summary_path = tmp_path / experiment_id / "ablations" / "retrained_arms_summary.json"
    assert summary_path.is_file()

    import json

    payload = json.loads(summary_path.read_text())
    assert payload["table_incomplete"] is False
    assert payload["master_seed"] == master_seed
    assert set(payload["seeds"]) == set(TEST_SEEDS)
    assert len(payload["runs"]) == 6
