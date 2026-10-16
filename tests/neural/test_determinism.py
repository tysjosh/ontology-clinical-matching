"""Determinism from a frozen configuration (task 10.3).

**Correctness Property 17**'s four conjuncts, one test each so a failure
names the conjunct:

1. two seeded model constructions give bitwise-identical parameters;
2. two `eval()`-mode forward passes on one batch give bitwise-identical
   outputs;
3. `compute_experiment_id` is stable, and any single-field change to the
   resolved config changes it;
4. a tiny end-to-end run (dataset build -> seeded construction ->
   `train_one_model` -> checkpoint) repeated in two separate `tmp_path`
   roots yields identical artifact hashes.

Plus the CPU/CUDA agreement integration test (Requirement 5.5), skipped
unless `torch.cuda.is_available()`: it cannot run on the local M2 and is
intended to run once on the A100 host, recording its output in a run
manifest through `ManifestWriter`. Set `CWPWM_DEVICE_AGREEMENT_DIR` to keep
that manifest; otherwise it is written under `tmp_path`.

**Validates: Requirements 2.6, 5.4, 5.5, 6.5, 9.9, 18.7**
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import typing
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

torch = pytest.importorskip("torch")

from infra.config import ResolvedConfig, load_config  # noqa: E402
from infra.manifest import ManifestWriter  # noqa: E402
from infra.registry import compute_experiment_id  # noqa: E402
from models.dataset import PatientDatasetBuilder  # noqa: E402
from models.torch_intervention_aware import TorchInterventionAwareModel  # noqa: E402
from models.torch_sequence_gru import TorchSequenceGRU  # noqa: E402
from models.torch_training import (  # noqa: E402
    TrainingBatches,
    TrainingConfig,
    checkpoint_sha256,
    factual_outputs,
    record_training_manifest,
    seed_everything,
    train_one_model,
)
from tests.neural.test_dataset_builder import _cohort  # noqa: E402

pytestmark = pytest.mark.neural

MODEL_KINDS = ("gru", "intervention_aware")
CONFIG_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "config"
_seeds = st.integers(min_value=0, max_value=2**31 - 1)


@pytest.fixture(autouse=True, scope="module")
def _restore_determinism_flags():
    """`seed_everything` switches deterministic algorithms on globally."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _batches() -> TrainingBatches:
    dev_records = _cohort(12, prefix="D", seed=41)
    val_records = _cohort(8, prefix="V", seed=43)
    builder = PatientDatasetBuilder().fit(dev_records)
    return TrainingBatches(
        dev=builder.build(dev_records, "dev"), val=builder.build(val_records, "val")
    )


_BATCHES = _batches()


def _construct(kind: str, seed: int) -> torch.nn.Module:
    seed_everything(seed)
    if kind == "gru":
        return TorchSequenceGRU(_BATCHES.dev.dimensions)
    return TorchInterventionAwareModel(_BATCHES.dev.dimensions)


# ---------------------------------------------------------------------------
# Conjunct 1: seeded construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", MODEL_KINDS)
@settings(max_examples=15, deadline=None)
@given(seed=_seeds)
def test_property_17_seeded_constructions_are_bitwise_identical(kind, seed):
    """**Correctness Property 17** (conjunct 1), Requirements 2.6, 5.4."""
    first = _construct(kind, seed).state_dict()
    second = _construct(kind, seed).state_dict()
    assert first.keys() == second.keys()
    for name, tensor in first.items():
        assert torch.equal(tensor, second[name]), name


# ---------------------------------------------------------------------------
# Conjunct 2: eval-mode forwards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", MODEL_KINDS)
@settings(max_examples=15, deadline=None)
@given(seed=_seeds)
def test_property_17_eval_forwards_are_bitwise_identical(kind, seed):
    """**Correctness Property 17** (conjunct 2), Requirement 2.6."""
    model = _construct(kind, seed)
    model.eval()
    with torch.no_grad():
        first = factual_outputs(model, _BATCHES.dev)
        second = factual_outputs(model, _BATCHES.dev)
    for field in ("state_pred", "hazard_logits", "hazards", "representation"):
        assert torch.equal(getattr(first, field), getattr(second, field)), field

    if isinstance(model, TorchInterventionAwareModel):
        with torch.no_grad():
            pair_a = model.rollout_both_arms(_BATCHES.dev)
            pair_b = model.rollout_both_arms(_BATCHES.dev)
        for arm_a, arm_b in zip(pair_a, pair_b, strict=True):
            for field in ("state_pred", "hazard_logits", "hazards", "outcome_96w"):
                assert torch.equal(getattr(arm_a, field), getattr(arm_b, field)), field


# ---------------------------------------------------------------------------
# Conjunct 3: experiment identity
# ---------------------------------------------------------------------------


def _load_fixture_config() -> ResolvedConfig:
    return load_config(
        CONFIG_FIXTURES / "valid_dataset.yaml",
        CONFIG_FIXTURES / "valid_target_trial.yaml",
        CONFIG_FIXTURES / "valid_experiment.yaml",
    )


#: Values for fields whose allowed set is enforced by a validator rather
#: than a `Literal` annotation.
_VALIDATOR_ALTERNATIVES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("experiment", "witness_policy"): ("TWO_OF_THREE", "RISK_WEIGHTED"),
}

#: A valid model variant for the `model_variant: None` leaf.
_MODEL_VARIANT = {
    "name": "complete",
    "model_family": "intervention_aware",
    "treatment_representation": "explicit_channel",
    "protocol_channel": "active",
    "overlap_gate": True,
    "uncertainty_gate": True,
    "clinical_constraints": True,
    "enabled_witnesses": ["causal", "clinical", "temporal"],
    "causal_adjustment": True,
    "balance_penalty_enabled": True,
}


def _leaves(node, path=()):
    """Every field path in a canonical dict: scalars, and lists as wholes."""
    if isinstance(node, dict):
        for key in sorted(node):
            yield from _leaves(node[key], (*path, key))
    elif isinstance(node, list):
        yield path, node
        for index, item in enumerate(node):
            yield from _leaves(item, (*path, index))
    else:
        yield path, node


def _literal_alternatives(path: tuple) -> tuple:
    """The `Literal` values the schema allows at `path`, if it declares any."""
    model: type[BaseModel] = ResolvedConfig
    for key in path:
        if not isinstance(key, str) or key not in model.model_fields:
            return ()
        annotation = model.model_fields[key].annotation
        if typing.get_origin(annotation) is typing.Literal:
            return typing.get_args(annotation)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model = annotation
            continue
        return ()
    return ()


def _candidates(path: tuple, value) -> list:
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1]
    if isinstance(value, float):
        return [value + 0.5]
    if isinstance(value, str):
        swapped_last = value[:-1] + ("1" if value[-1:] != "1" else "2")
        return [
            f"{value}_mutated",
            swapped_last,
            *(v for v in _literal_alternatives(path) if v != value),
            *(v for v in _VALIDATOR_ALTERNATIVES.get(path, ()) if v != value),
        ]
    if isinstance(value, list):
        return [[*value, value[-1]], value[:-1]] if value else [["mutated"]]
    if value is None and path == ("experiment", "model_variant"):
        return [_MODEL_VARIANT]
    return []


def _set(tree, path: tuple, value):
    mutated = copy.deepcopy(tree)
    node = mutated
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return mutated


def test_property_17_experiment_id_is_stable_and_changes_under_any_single_field_mutation():
    """**Correctness Property 17** (conjunct 3), Requirements 6.5, 9.9.

    Every leaf field of the resolved config (and every list as a whole) is
    mutated once, in isolation, to a value the schema accepts; each
    mutation must change the identifier.

    The base has `exploratory=true` so that `witness_policy` -- which a
    non-exploratory config pins to `ALL_REQUIRED` -- also admits a valid
    single-field change; stability is checked on the unmodified fixture too.
    """
    fixture = _load_fixture_config()
    assert compute_experiment_id(fixture) == compute_experiment_id(fixture)
    assert compute_experiment_id(fixture) == compute_experiment_id(_load_fixture_config())

    base_tree = _set(fixture.canonical_dict(), ("experiment", "exploratory"), True)
    base = ResolvedConfig.model_validate(base_tree)
    base_id = compute_experiment_id(base)
    assert base_id == compute_experiment_id(ResolvedConfig.model_validate(base_tree))

    unmutated: list[tuple] = []
    checked = 0
    for path, value in _leaves(base.canonical_dict()):
        mutated_id = None
        for candidate in _candidates(path, value):
            try:
                mutated = ResolvedConfig.model_validate(_set(base_tree, path, candidate))
            except ValidationError:
                continue
            if mutated.canonical_dict() == base.canonical_dict():
                continue  # normalized back to the base value: not a change
            mutated_id = compute_experiment_id(mutated)
            break
        if mutated_id is None:
            unmutated.append(path)
            continue
        checked += 1
        assert mutated_id != base_id, f"mutating {path!r} left experiment_id unchanged"

    assert not unmutated, f"no valid single-field mutation found for {unmutated!r}"
    assert checked >= 30


# ---------------------------------------------------------------------------
# Conjunct 4: end-to-end artifact hashes
# ---------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _end_to_end_run(root: Path, kind: str, seed: int) -> dict[str, str]:
    """Build the dataset, construct, train briefly, checkpoint; hash every artifact."""
    batches = _batches()
    model = _construct(kind, seed)
    result = train_one_model(
        model,
        batches,
        lambda_state=0.3,
        lambda_balance=0.1,
        seed=seed,
        config=TrainingConfig(max_epochs=3),
        checkpoint_path=root / "checkpoints" / "model.pt",
    )
    curves_path = root / "curves.json"
    curves_path.write_text(
        json.dumps(
            {
                "train": list(result.train_loss_curve),
                "val": list(result.val_loss_curve),
                "best_epoch": result.best_epoch,
            },
            sort_keys=True,
        )
    )
    return {
        "checkpoint": checkpoint_sha256(result.checkpoint_path),
        "reported_checkpoint": result.checkpoint_sha256,
        "curves": _sha256_bytes(curves_path.read_bytes()),
    }


@pytest.mark.parametrize("seed", [0, 7919])
@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_property_17_end_to_end_runs_in_two_roots_have_identical_artifact_hashes(
    kind, seed, tmp_path
):
    """**Correctness Property 17** (conjunct 4), Requirement 18.7."""
    first = _end_to_end_run(tmp_path / "run_a", kind, seed)
    second = _end_to_end_run(tmp_path / "run_b", kind, seed)
    assert first["checkpoint"] == first["reported_checkpoint"]
    assert first == second


# ---------------------------------------------------------------------------
# CPU/CUDA agreement (Requirement 5.5) -- A100 host only
# ---------------------------------------------------------------------------

#: "Floating-point accumulation tolerance" for a few full-batch Adam steps.
DEVICE_RTOL = 1e-4
DEVICE_ATOL = 1e-5


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable; Requirement 5.5 is verified once on the A100 host",
)
@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_cpu_and_cuda_runs_agree_and_are_recorded_in_the_manifest(kind, tmp_path):
    """Requirement 5.5: the same seeded configuration on `cpu` and `cuda`
    gives loss curves and final weights that agree within tolerance. The
    comparison is recorded in a run manifest (Requirement 5.7)."""
    seed = 20240101
    config = TrainingConfig(max_epochs=5)
    results = {}
    for device in ("cpu", "cuda"):
        results[device] = train_one_model(
            _construct(kind, seed),
            _BATCHES,
            lambda_state=0.3,
            lambda_balance=0.3,
            seed=seed,
            device=device,
            config=config,
        )
    cpu, cuda = results["cpu"], results["cuda"]

    def max_abs_diff(a, b) -> float:
        return float((torch.as_tensor(a) - torch.as_tensor(b)).abs().max())

    curve_diff = max(
        max_abs_diff(cpu.train_loss_curve, cuda.train_loss_curve),
        max_abs_diff(cpu.val_loss_curve, cuda.val_loss_curve),
    )
    # Empty tensors (e.g. the intervention-aware model's zero-length
    # `placebo_vector` buffer outside placebo mode) hold nothing to compare,
    # and `max()` over zero elements raises; they are skipped.
    cpu_weights = {n: t for n, t in cpu.model.state_dict().items() if t.numel() > 0}
    cuda_weights = {
        name: t.detach().cpu() for name, t in cuda.model.state_dict().items() if t.numel() > 0
    }
    assert cpu_weights.keys() == cuda_weights.keys()
    weight_diff = max(max_abs_diff(cpu_weights[n], cuda_weights[n]) for n in cpu_weights)
    curves_agree = torch.allclose(
        torch.tensor(cpu.train_loss_curve + cpu.val_loss_curve, dtype=torch.float64),
        torch.tensor(cuda.train_loss_curve + cuda.val_loss_curve, dtype=torch.float64),
        rtol=DEVICE_RTOL,
        atol=DEVICE_ATOL,
    )
    weights_agree = all(
        torch.allclose(cpu_weights[n], cuda_weights[n], rtol=DEVICE_RTOL, atol=DEVICE_ATOL)
        for n in cpu_weights
    )

    output_dir = Path(os.environ.get("CWPWM_DEVICE_AGREEMENT_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(
        experiment_id=f"device_agreement_{kind}",
        stage="device_agreement",
        allowed_path_prefixes=(str(output_dir),),
    )
    record_training_manifest(manifest, cpu, label=f"{kind}.cpu", config=config)
    record_training_manifest(manifest, cuda, label=f"{kind}.cuda", config=config)
    manifest.add_metric(f"{kind}.device_agreement.max_abs_loss_curve_diff", curve_diff)
    manifest.add_metric(f"{kind}.device_agreement.max_abs_weight_diff", weight_diff)
    manifest.add_metric(f"{kind}.device_agreement.rtol", DEVICE_RTOL)
    manifest.add_metric(f"{kind}.device_agreement.atol", DEVICE_ATOL)
    manifest.add_metric(f"{kind}.device_agreement.curves_agree", curves_agree)
    manifest.add_metric(f"{kind}.device_agreement.weights_agree", weights_agree)
    manifest.add_metric(f"{kind}.device_agreement.cuda_device_name", torch.cuda.get_device_name())
    written = manifest.write(output_dir / f"device_agreement_{kind}_manifest.json")

    recorded = json.loads(written.read_text())
    assert recorded["device"] == "cuda"
    assert recorded["metrics"][f"{kind}.device_agreement.curves_agree"] is True
    assert curves_agree, f"loss curves diverge across devices (max |diff| {curve_diff})"
    assert weights_agree, f"final weights diverge across devices (max |diff| {weight_diff})"
