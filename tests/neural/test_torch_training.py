"""Tests for `models.torch_training` (tasks 10, 10.1, and 10.2).

Covers the frozen training configuration, split isolation, checkpointing,
the frozen 3x3 lambda grid, validation-only lambda selection, and the
replica-seed derivation shared with `uncertainty.ensemble.DeepEnsemble`.
Gradient/oracle and determinism properties belong to task 10.3.

Runs use a shortened `max_epochs` so the nine-combination selection stays
fast; every other configuration field is the frozen value.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import models.torch_training as torch_training  # noqa: E402
from evaluation.metrics import concordance_index  # noqa: E402
from infra.manifest import GitProvenance, ManifestWriter  # noqa: E402
from infra.reference_guard import ReferenceAccessError  # noqa: E402
from infra.seeding import derive_seed  # noqa: E402
from models.dataset import PatientDatasetBuilder  # noqa: E402
from models.losses import (  # noqa: E402
    load_lambda_grid,
    select_lambda_by_validation_concordance,
)
from models.torch_intervention_aware import TorchInterventionAwareModel  # noqa: E402
from models.torch_sequence_gru import TorchSequenceGRU  # noqa: E402
from models.torch_training import (  # noqa: E402
    FROZEN_TRAINING_CONFIG,
    LAMBDA_GRID_SIZE,
    TrainingBatches,
    TrainingConfig,
    checkpoint_sha256,
    compute_loss_terms,
    load_torch_checkpoint,
    masked_huber_tensor,
    replica_seeds,
    select_lambdas,
    train_ensemble,
    train_one_model,
    validation_scoring_triple,
)
from tests.neural.test_dataset_builder import _cohort  # noqa: E402
from uncertainty.ensemble import DeepEnsemble  # noqa: E402

pytestmark = pytest.mark.neural

SHORT_CONFIG = TrainingConfig(max_epochs=4)
FROZEN_VALUES = {0.1, 0.3, 1.0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def batches() -> TrainingBatches:
    """Distinct dev (14) and val (9) populations, so a triple's length and
    content identify which split it was computed from."""
    dev_records = _cohort(14, prefix="D", seed=11)
    val_records = _cohort(9, prefix="V", seed=23)
    builder = PatientDatasetBuilder().fit(dev_records)
    return TrainingBatches(
        dev=builder.build(dev_records, "dev"), val=builder.build(val_records, "val")
    )


@pytest.fixture(scope="module")
def test_batch(batches):
    records = _cohort(6, prefix="T", seed=31)
    builder = PatientDatasetBuilder().fit(_cohort(14, prefix="D", seed=11))
    return builder.build(records, "test")


def _factory(kind: str, batches: TrainingBatches):
    dimensions = batches.dev.dimensions
    if kind == "gru":
        return lambda: TorchSequenceGRU(dimensions)
    return lambda: TorchInterventionAwareModel(dimensions)


# ---------------------------------------------------------------------------
# Frozen configuration (decisions.md Section 25)
# ---------------------------------------------------------------------------


def test_frozen_training_configuration_matches_section_25():
    assert FROZEN_TRAINING_CONFIG.learning_rate == 1e-3
    assert FROZEN_TRAINING_CONFIG.betas == (0.9, 0.999)
    assert FROZEN_TRAINING_CONFIG.weight_decay == 1e-4
    assert FROZEN_TRAINING_CONFIG.grad_clip_norm == 1.0
    assert FROZEN_TRAINING_CONFIG.patience == 20
    assert FROZEN_TRAINING_CONFIG.max_epochs == 500
    assert FROZEN_TRAINING_CONFIG.is_frozen
    assert not SHORT_CONFIG.is_frozen
    assert inspect.signature(train_one_model).parameters["config"].default is (
        FROZEN_TRAINING_CONFIG
    )


# ---------------------------------------------------------------------------
# train_one_model: curves, best epoch, checkpoint format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["gru", "intervention_aware"])
def test_train_one_model_checkpoints_the_validation_minimizing_epoch(kind, batches, tmp_path):
    torch_training.seed_everything(3)
    model = _factory(kind, batches)()
    path = tmp_path / "model.pt"

    result = train_one_model(
        model,
        batches,
        lambda_state=0.3,
        lambda_balance=0.1,
        seed=3,
        config=TrainingConfig(max_epochs=6),
        checkpoint_path=path,
    )

    assert result.epochs_run == len(result.train_loss_curve) == len(result.val_loss_curve)
    assert 1 <= result.epochs_run <= 6
    minimum = min(result.val_loss_curve)
    assert result.best_val_loss == minimum
    assert result.best_epoch == result.val_loss_curve.index(minimum) + 1

    checkpoint = load_torch_checkpoint(path)
    assert set(checkpoint) == {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "seed",
        "val_loss",
    }
    assert checkpoint["epoch"] == result.best_epoch
    assert checkpoint["seed"] == 3
    assert checkpoint["val_loss"] == result.best_val_loss
    for name, tensor in result.model.state_dict().items():
        assert torch.equal(checkpoint["model_state_dict"][name], tensor), name

    group = checkpoint["optimizer_state_dict"]["param_groups"][0]
    assert group["lr"] == 1e-3
    assert tuple(group["betas"]) == (0.9, 0.999)
    assert group["weight_decay"] == 1e-4

    assert result.checkpoint_sha256 == checkpoint_sha256(path)
    assert len(result.checkpoint_sha256) == 64


def test_training_reduces_the_development_loss(batches):
    torch_training.seed_everything(5)
    result = train_one_model(
        TorchSequenceGRU(batches.dev.dimensions),
        batches,
        lambda_state=0.3,
        lambda_balance=0.3,
        seed=5,
        config=TrainingConfig(max_epochs=30),
    )
    assert result.train_loss_curve[-1] < result.train_loss_curve[0]


def test_early_stopping_stops_patience_epochs_after_the_best(batches):
    """With `lr` effectively zero the validation loss never improves after
    epoch 1, so the run stops exactly `patience` epochs later."""
    config = TrainingConfig(learning_rate=0.0, weight_decay=0.0, patience=3, max_epochs=50)
    torch_training.seed_everything(7)
    result = train_one_model(
        TorchSequenceGRU(batches.dev.dimensions),
        batches,
        lambda_state=0.1,
        lambda_balance=0.1,
        seed=7,
        config=config,
    )
    assert result.stopped_early
    assert result.best_epoch == 1
    assert result.epochs_run == 1 + config.patience


def test_masked_entries_contribute_zero_to_state_loss_value_and_gradient(batches):
    batch = batches.dev
    prediction = torch.randn(*batch.cd4.shape, 2, requires_grad=True)
    loss = masked_huber_tensor(prediction, batch.cd4, batch.cd8, batch.cd4_mask, batch.cd8_mask)
    loss.backward()

    perturbed_cd4 = torch.where(batch.cd4_mask, batch.cd4, torch.full_like(batch.cd4, 1e3))
    perturbed_cd8 = torch.where(batch.cd8_mask, batch.cd8, torch.full_like(batch.cd8, -1e3))
    perturbed = masked_huber_tensor(
        prediction.detach(), perturbed_cd4, perturbed_cd8, batch.cd4_mask, batch.cd8_mask
    )

    assert (~batch.cd4_mask).any() and (~batch.cd8_mask).any()
    assert float(perturbed) == float(loss)
    assert torch.all(prediction.grad[..., 0][~batch.cd4_mask] == 0.0)
    assert torch.all(prediction.grad[..., 1][~batch.cd8_mask] == 0.0)


def test_loss_terms_combine_by_the_frozen_formula(batches):
    torch_training.seed_everything(1)
    model = TorchInterventionAwareModel(batches.dev.dimensions)
    terms = compute_loss_terms(model, batches.dev, 0.3, 1.0)
    expected = terms.survival + 0.3 * terms.state + 1.0 * terms.balance
    assert torch.allclose(terms.total, expected)


# ---------------------------------------------------------------------------
# Split isolation (Requirements 4.7, 4.8)
# ---------------------------------------------------------------------------


def test_training_batches_refuse_a_test_batch_in_either_slot(batches, test_batch):
    with pytest.raises(ReferenceAccessError):
        TrainingBatches(dev=test_batch, val=batches.val)
    with pytest.raises(ReferenceAccessError):
        TrainingBatches(dev=batches.dev, val=test_batch)
    with pytest.raises(ValueError):
        TrainingBatches(dev=batches.val, val=batches.dev)


def test_no_entry_point_has_a_parameter_for_a_test_batch():
    for function in (train_one_model, select_lambdas, train_ensemble):
        names = set(inspect.signature(function).parameters)
        assert not any("test" in name for name in names), (function.__name__, names)


def test_harness_never_calls_test_split_or_unseal_accessors():
    tree = ast.parse(Path(torch_training.__file__).read_text())
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    referenced |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    }
    for forbidden in ("require_test_split", "unseal", "read_sealed_artifact"):
        assert forbidden not in referenced | imported, forbidden


def test_validation_scoring_refuses_dev_and_test_batches(batches, test_batch):
    model = TorchSequenceGRU(batches.dev.dimensions)
    with pytest.raises(ValueError):
        validation_scoring_triple(model, batches.dev)
    with pytest.raises(ReferenceAccessError):
        validation_scoring_triple(model, test_batch)


# ---------------------------------------------------------------------------
# Lambda grid and selection (Requirement 4.3; task 10.1)
# ---------------------------------------------------------------------------


def test_lambda_grid_has_exactly_the_nine_frozen_combinations():
    grid = load_lambda_grid()
    assert len(grid) == LAMBDA_GRID_SIZE == 9
    assert set(grid) == set(itertools.product(sorted(FROZEN_VALUES), repeat=2))


def test_select_lambdas_rejects_a_grid_that_is_not_nine_combinations(batches):
    with pytest.raises(ValueError, match="9"):
        select_lambdas(
            _factory("gru", batches),
            batches,
            seed=1,
            config=SHORT_CONFIG,
            grid=load_lambda_grid()[:8],
        )


@pytest.fixture(scope="module")
def selection(batches):
    return select_lambdas(_factory("intervention_aware", batches), batches, 17, config=SHORT_CONFIG)


def test_selection_evaluates_all_nine_combinations(selection):
    assert selection.grid == tuple(load_lambda_grid())
    assert set(selection.candidate_triples) == set(selection.grid)
    assert selection.selected in selection.grid
    assert selection.seed == 17


def test_selection_reads_validation_split_values_only(selection, batches):
    val = batches.val
    assert selection.scored_split == "val"
    assert val.event_indicator.shape[0] != batches.dev.event_indicator.shape[0]
    expected_times = tuple(float(v) for v in val.event_time_weeks.tolist())
    expected_events = tuple(bool(v) for v in val.event_indicator.tolist())
    for risk, times, events in selection.candidate_triples.values():
        assert len(risk) == len(val.patient_ids)
        assert times == expected_times
        assert events == expected_events


def test_selection_is_the_validation_concordance_argmax(selection):
    assert select_lambda_by_validation_concordance(selection.candidate_triples) == (
        selection.selected
    )
    concordances = {
        pair: concordance_index(*triple) for pair, triple in selection.candidate_triples.items()
    }
    assert concordances[selection.selected] == max(concordances.values())


# ---------------------------------------------------------------------------
# Ensemble (Requirement 4.6; task 10.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("master_seed", [0, 101, 505, 20240101])
def test_replica_seeds_equal_the_deep_ensemble_derivation(master_seed):
    ensemble = DeepEnsemble(
        model_factory=lambda seed: seed,
        predict_fn=lambda model, x: 0.0,
        n_replicas=5,
        master_seed=master_seed,
    )
    assert list(replica_seeds(master_seed)) == ensemble.replica_seeds
    assert replica_seeds(master_seed) == tuple(
        derive_seed(master_seed, f"replica_{i}") for i in range(5)
    )


def test_train_ensemble_selects_once_on_replica_zero_and_trains_five(batches, tmp_path):
    master_seed = 101
    result = train_ensemble(
        _factory("gru", batches),
        batches,
        master_seed,
        config=SHORT_CONFIG,
        checkpoint_dir=tmp_path,
    )
    reference = DeepEnsemble(lambda s: s, lambda m, x: 0.0, 5, master_seed)
    expected_seeds = tuple(reference.replica_seeds)

    assert result.replica_seeds == expected_seeds
    assert result.lambda_selection.seed == expected_seeds[0]
    assert len(result.replicas) == 5
    assert tuple(replica.seed for replica in result.replicas) == expected_seeds
    for index, replica in enumerate(result.replicas):
        assert (replica.lambda_state, replica.lambda_balance) == result.lambda_selection.selected
        assert replica.checkpoint_path == tmp_path / f"replica_{index}.pt"
        assert replica.checkpoint_sha256 == checkpoint_sha256(replica.checkpoint_path)
    first = result.replicas[0].model.state_dict()
    second = result.replicas[1].model.state_dict()
    assert any(not torch.equal(first[name], second[name]) for name in first)


# ---------------------------------------------------------------------------
# Device resolution and seeding (Requirements 5.3, 5.4, 5.6; task 10.2)
# ---------------------------------------------------------------------------


def test_resolve_device_defaults_to_cpu_and_accepts_cuda_when_available(monkeypatch):
    assert inspect.signature(torch_training.resolve_device).parameters["requested"].default == (
        "cpu"
    )
    assert torch_training.resolve_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_training.resolve_device("cuda") == "cuda"


def test_resolve_device_names_cuda_when_it_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(torch_training.DeviceUnavailableError, match="'cuda'"):
        torch_training.resolve_device("cuda")


@pytest.mark.parametrize("requested", ["mps", "CUDA", "cuda:0", " cpu", "", "tpu", None, 0])
def test_resolve_device_rejects_every_other_value(requested):
    with pytest.raises(torch_training.UnsupportedDeviceError):
        torch_training.resolve_device(requested)


def test_unavailable_cuda_terminates_before_the_model_is_touched(monkeypatch, batches):
    """`train_one_model` fails at device resolution: no optimizer step, no
    parameter moved, no checkpoint written."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = TorchSequenceGRU(batches.dev.dimensions)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(torch_training.DeviceUnavailableError):
        train_one_model(model, batches, lambda_state=0.1, lambda_balance=0.1, seed=1, device="cuda")
    for name, tensor in model.state_dict().items():
        assert tensor.device.type == "cpu"
        assert torch.equal(tensor, before[name]), name


def test_seed_everything_seeds_torch_and_enables_deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch_training.seed_everything(42)
        first = torch.rand(4)
        assert torch.are_deterministic_algorithms_enabled()
        # Full fp32 on CUDA: TF32 off for both matmul and cuDNN (Req 5.5).
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        torch_training.seed_everything(42)
        assert torch.equal(torch.rand(4), first)
    finally:
        torch.use_deterministic_algorithms(previous)


# ---------------------------------------------------------------------------
# Manifest recording (Requirements 4.9, 5.7; task 10.2)
# ---------------------------------------------------------------------------


def _stub_provenance():
    return GitProvenance(commit="0" * 40)


def test_two_epoch_run_populates_all_five_manifest_fields(batches, tmp_path):
    manifest = ManifestWriter(
        experiment_id="exp",
        stage="train",
        allowed_path_prefixes=(str(tmp_path),),
        git_provenance_resolver=_stub_provenance,
    )
    manifest.set_model_config({"family": "gru"})
    torch_training.seed_everything(9)
    result = train_one_model(
        TorchSequenceGRU(batches.dev.dimensions),
        batches,
        lambda_state=0.3,
        lambda_balance=1.0,
        seed=9,
        config=TrainingConfig(max_epochs=2),
        checkpoint_path=tmp_path / "model.pt",
        manifest=manifest,
    )
    assert result.epochs_run == 2

    written = manifest.write(tmp_path / "manifest.json")
    recorded = json.loads(written.read_text())
    metrics = recorded["metrics"]
    config = recorded["model_config"]

    # 1. per-epoch loss curve
    assert metrics["training.train_loss_curve"] == list(result.train_loss_curve)
    assert metrics["training.val_loss_curve"] == list(result.val_loss_curve)
    assert len(metrics["training.val_loss_curve"]) == 2
    # 2. selected lambdas
    assert (config["training"]["lambda_state"], config["training"]["lambda_balance"]) == (0.3, 1.0)
    assert (metrics["training.lambda_state"], metrics["training.lambda_balance"]) == (0.3, 1.0)
    # 3. selected epoch
    assert config["training"]["selected_epoch"] == result.best_epoch
    assert metrics["training.selected_epoch"] == result.best_epoch in (1, 2)
    # 4. device
    assert recorded["device"] == config["training"]["device"] == "cpu"
    # 5. checkpoint hash (and path)
    digest = checkpoint_sha256(tmp_path / "model.pt")
    assert config["training"]["checkpoint_sha256"] == digest
    assert metrics["training.checkpoint_sha256"] == digest
    assert recorded["checkpoint_paths"] == [str(tmp_path / "model.pt")]

    # Existing model_config entries survive the merge.
    assert config["family"] == "gru"
    assert config["training"]["training_config"]["max_epochs"] == 2
