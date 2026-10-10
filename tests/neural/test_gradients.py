"""Gradient-flow, oracle-agreement, and masking tests for the training loss
(task 10.3).

- **Gradient flow** (Requirement 2.5): after `loss.backward()` every
  parameter of both torch models has a non-`None`, finite, non-all-zero
  `.grad`. That proves autograd is in use and that no submodule is detached
  or saturated out of the graph.
- **Oracle agreement** (Requirement 4.1): the torch objective equals
  `models.losses.total_loss` applied to the pure-Python oracles
  (`models.training.discrete_time_censored_nll`, `models.losses.masked_huber`,
  `models.losses.mmd`) evaluated on the same predictions, within `1e-6`.
- **Correctness Property 15** (Requirements 1.5, 4.2): masked-missing
  CD4/CD8 observations are inert.

Oracle scope. `masked_huber` marks a whole `(cd4, cd8)` pair missing with
`None`; it cannot express a position where only one component is observed.
The oracle comparison therefore runs on a batch whose two masks are made to
agree (their intersection; newly masked entries reset to the placeholder).
The torch term's handling of single-component positions is exercised by
Property 15 and by `test_torch_training.py`. The comparison runs in float64
so that `1e-6` measures agreement of definitions rather than float32
accumulation error over 23 bins.

Property 15 scope. `TorchSequenceGRU` consumes CD4/CD8 values as *inputs*
(the placeholder is fed in by design, Requirements 1.5 and 2.2), so for that
model the property is stated on the loss with the model's predictions held
fixed. `TorchInterventionAwareModel` reads no CD4/CD8 values as inputs, so
for it the property also holds end to end through `compute_loss_terms`,
including every parameter gradient.

**Validates: Requirements 1.5, 2.5, 4.1, 4.2**
"""

from __future__ import annotations

import dataclasses
import math
from functools import cache

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from models.dataset import MISSING_PLACEHOLDER, PatientDatasetBuilder, TensorBatch  # noqa: E402
from models.losses import load_lambda_grid, masked_huber, mmd, total_loss  # noqa: E402
from models.torch_intervention_aware import TorchInterventionAwareModel  # noqa: E402
from models.torch_sequence_gru import TorchSequenceGRU  # noqa: E402
from models.torch_training import (  # noqa: E402
    compute_loss_terms,
    factual_outputs,
    masked_huber_tensor,
    mmd_tensor,
    seed_everything,
    survival_nll_tensor,
)
from models.training import discrete_time_censored_nll  # noqa: E402
from tests.neural.test_dataset_builder import _cohort  # noqa: E402

pytestmark = pytest.mark.neural

MODEL_KINDS = ("gru", "intervention_aware")
ORACLE_TOLERANCE = 1e-6
LAMBDA_GRID = tuple(load_lambda_grid())


# ---------------------------------------------------------------------------
# Shared inputs (module-level caches rather than fixtures, so Hypothesis
# tests can use them without function-scoped fixture interplay)
# ---------------------------------------------------------------------------


@cache
def _dev_batch() -> TensorBatch:
    records = _cohort(16, prefix="G", seed=5)
    return PatientDatasetBuilder().fit(records).build(records, "dev")


def _model(kind: str, seed: int = 0) -> torch.nn.Module:
    seed_everything(seed)
    dimensions = _dev_batch().dimensions
    if kind == "gru":
        return TorchSequenceGRU(dimensions)
    return TorchInterventionAwareModel(dimensions)


@cache
def _fixed_outputs(kind: str):
    """One model's factual outputs on the dev batch, detached."""
    model = _model(kind, seed=13)
    model.eval()
    with torch.no_grad():
        return factual_outputs(model, _dev_batch())


def _float_fields(batch: TensorBatch) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in dataclasses.fields(TensorBatch)
        if isinstance(getattr(batch, field.name), torch.Tensor)
        and getattr(batch, field.name).is_floating_point()
    )


def _as_float64(batch: TensorBatch) -> TensorBatch:
    return dataclasses.replace(
        batch, **{name: getattr(batch, name).double() for name in _float_fields(batch)}
    )


def _with_agreeing_masks(batch: TensorBatch) -> TensorBatch:
    """`batch` with both masks set to their intersection, the case
    `masked_huber`'s per-pair `None` can express."""
    joint = batch.cd4_mask & batch.cd8_mask
    placeholder = torch.full_like(batch.cd4, MISSING_PLACEHOLDER)
    return dataclasses.replace(
        batch,
        cd4_mask=joint.clone(),
        cd8_mask=joint.clone(),
        cd4=torch.where(joint, batch.cd4, placeholder),
        cd8=torch.where(joint, batch.cd8, placeholder),
    )


# ---------------------------------------------------------------------------
# Gradient flow (Requirement 2.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_every_parameter_receives_a_finite_nonzero_gradient(kind):
    model = _model(kind, seed=3)
    model.train()
    loss = compute_loss_terms(model, _dev_batch(), lambda_state=1.0, lambda_balance=1.0).total
    loss.backward()

    parameters = dict(model.named_parameters())
    assert parameters, "model has no parameters"
    for name, parameter in parameters.items():
        assert parameter.grad is not None, f"{name}: .grad is None (detached from the graph)"
        assert torch.isfinite(parameter.grad).all(), f"{name}: non-finite gradient"
        assert bool((parameter.grad != 0).any()), f"{name}: gradient is all zero"


# ---------------------------------------------------------------------------
# Oracle agreement with the pure-Python loss (Requirement 4.1)
# ---------------------------------------------------------------------------


def _oracle_survival(hazards: list[list[float]], bins: list[int], events: list[bool]) -> float:
    """Batch mean of `discrete_time_censored_nll` under the module's
    censoring convention (censored patients credited survival for bins < b)."""
    per_patient = []
    for patient_hazards, bin_index, event in zip(hazards, bins, events, strict=True):
        if event:
            per_patient.append(discrete_time_censored_nll(patient_hazards, bin_index))
        elif bin_index == 0:
            per_patient.append(0.0)
        else:
            per_patient.append(discrete_time_censored_nll(patient_hazards[:bin_index], None))
    return sum(per_patient) / len(per_patient)


def _oracle_state(state_pred, batch: TensorBatch) -> float:
    predicted: list[tuple[float, float]] = []
    observed: list[tuple[float, float] | None] = []
    pred = state_pred.tolist()
    cd4, cd8 = batch.cd4.tolist(), batch.cd8.tolist()
    cd4_mask, cd8_mask = batch.cd4_mask.tolist(), batch.cd8_mask.tolist()
    for i, patient_pred in enumerate(pred):
        for landmark, (pred_cd4, pred_cd8) in enumerate(patient_pred):
            assert cd4_mask[i][landmark] == cd8_mask[i][landmark], "oracle needs agreeing masks"
            predicted.append((pred_cd4, pred_cd8))
            observed.append((cd4[i][landmark], cd8[i][landmark]) if cd4_mask[i][landmark] else None)
    return masked_huber(predicted, observed)


def _oracle_balance(representation, treatment_arm) -> float:
    rows = representation.tolist()
    arms = treatment_arm.tolist()
    treated = [row for row, arm in zip(rows, arms, strict=True) if arm == 1.0]
    control = [row for row, arm in zip(rows, arms, strict=True) if arm != 1.0]
    return mmd(treated, control)


@pytest.mark.parametrize("lambdas", LAMBDA_GRID)
@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_torch_total_loss_matches_the_pure_python_oracle(kind, lambdas):
    lambda_state, lambda_balance = lambdas
    batch = _as_float64(_with_agreeing_masks(_dev_batch()))
    assert (~batch.cd4_mask).any() and batch.cd4_mask.any()
    assert bool((batch.treatment_arm == 1.0).any()) and bool((batch.treatment_arm == 0.0).any())

    model = _model(kind, seed=21).double()
    terms = compute_loss_terms(model, batch, lambda_state, lambda_balance)
    with torch.no_grad():
        outputs = factual_outputs(model, batch)

    survival = _oracle_survival(
        outputs.hazards.tolist(), batch.bin_index.tolist(), batch.event_indicator.tolist()
    )
    state = _oracle_state(outputs.state_pred, batch)
    balance = _oracle_balance(outputs.representation, batch.treatment_arm)
    expected = total_loss(survival, state, balance, lambda_state, lambda_balance)

    assert float(terms.survival) == pytest.approx(survival, abs=ORACLE_TOLERANCE)
    assert float(terms.state) == pytest.approx(state, abs=ORACLE_TOLERANCE)
    assert float(terms.balance) == pytest.approx(balance, abs=ORACLE_TOLERANCE)
    assert abs(float(terms.total) - expected) <= ORACLE_TOLERANCE


# ---------------------------------------------------------------------------
# Correctness Property 15: masked-missing observations are inert
# ---------------------------------------------------------------------------

_finite_float32 = st.floats(width=32, allow_nan=False, allow_infinity=False)


def test_emitted_observation_tensors_are_finite_with_the_fixed_placeholder():
    """Property 15's emission clause, on the batch the properties below use."""
    batch = _dev_batch()
    for values, mask in ((batch.cd4, batch.cd4_mask), (batch.cd8, batch.cd8_mask)):
        assert torch.isfinite(values).all()
        assert (~mask).any(), "fixture must contain masked-missing entries"
        assert torch.all(values[~mask] == MISSING_PLACEHOLDER)


def _perturbed(values, mask, replacements: list[float]):
    """`values` with its masked-missing entries overwritten, as a grad leaf."""
    perturbed = values.clone()
    perturbed[~mask] = torch.tensor(replacements, dtype=values.dtype)
    return perturbed.requires_grad_(True)


def _masked_counts() -> tuple[int, int]:
    batch = _dev_batch()
    return int((~batch.cd4_mask).sum()), int((~batch.cd8_mask).sum())


def _replacement_values():
    n_cd4, n_cd8 = _masked_counts()
    return st.tuples(
        st.lists(_finite_float32, min_size=n_cd4, max_size=n_cd4),
        st.lists(_finite_float32, min_size=n_cd8, max_size=n_cd8),
    )


@pytest.mark.parametrize("kind", MODEL_KINDS)
@settings(max_examples=60, deadline=None)
@given(replacements=_replacement_values(), lambdas=st.sampled_from(LAMBDA_GRID))
def test_property_15_masked_positions_are_inert_in_the_loss(kind, replacements, lambdas):
    """**Correctness Property 15**, Requirements 1.5, 4.2.

    With the model's predictions held fixed, substituting any finite value
    into every masked-missing CD4/CD8 target leaves the total loss value
    bitwise unchanged, and the gradient at every masked position -- with
    respect to both the prediction and the substituted target -- is exactly
    `0.0`.
    """
    batch = _dev_batch()
    outputs = _fixed_outputs(kind)
    lambda_state, lambda_balance = lambdas
    survival = survival_nll_tensor(outputs.hazard_logits, batch.bin_index, batch.event_indicator)
    balance = mmd_tensor(outputs.representation, batch.treatment_arm)

    def loss_for(cd4, cd8):
        prediction = outputs.state_pred.clone().requires_grad_(True)
        state = masked_huber_tensor(prediction, cd4, cd8, batch.cd4_mask, batch.cd8_mask)
        return total_loss(survival, state, balance, lambda_state, lambda_balance), prediction

    reference, _ = loss_for(batch.cd4, batch.cd8)
    cd4 = _perturbed(batch.cd4, batch.cd4_mask, replacements[0])
    cd8 = _perturbed(batch.cd8, batch.cd8_mask, replacements[1])
    perturbed, prediction = loss_for(cd4, cd8)
    perturbed.backward()

    assert torch.equal(perturbed.detach(), reference.detach())
    assert torch.all(prediction.grad[..., 0][~batch.cd4_mask] == 0.0)
    assert torch.all(prediction.grad[..., 1][~batch.cd8_mask] == 0.0)
    assert torch.all(cd4.grad[~batch.cd4_mask] == 0.0)
    assert torch.all(cd8.grad[~batch.cd8_mask] == 0.0)


@settings(max_examples=30, deadline=None)
@given(replacements=_replacement_values(), lambdas=st.sampled_from(LAMBDA_GRID))
def test_property_15_masked_positions_are_inert_end_to_end(replacements, lambdas):
    """**Correctness Property 15**, Requirements 1.5, 4.2, through the whole
    intervention-aware model and `compute_loss_terms`.

    The model reads no CD4/CD8 values as inputs, so substituting masked
    targets must leave the loss value and every parameter gradient bitwise
    unchanged, with exactly-zero gradient at the substituted positions.
    """
    batch = _dev_batch()
    lambda_state, lambda_balance = lambdas

    def run(candidate: TensorBatch):
        model = _model("intervention_aware", seed=17)
        loss = compute_loss_terms(model, candidate, lambda_state, lambda_balance).total
        loss.backward()
        return loss.detach(), {name: p.grad.clone() for name, p in model.named_parameters()}

    reference_loss, reference_grads = run(batch)
    cd4 = _perturbed(batch.cd4, batch.cd4_mask, replacements[0])
    cd8 = _perturbed(batch.cd8, batch.cd8_mask, replacements[1])
    perturbed_loss, perturbed_grads = run(dataclasses.replace(batch, cd4=cd4, cd8=cd8))

    assert torch.equal(perturbed_loss, reference_loss)
    for name, gradient in reference_grads.items():
        assert torch.equal(perturbed_grads[name], gradient), name
    assert torch.all(cd4.grad[~batch.cd4_mask] == 0.0)
    assert torch.all(cd8.grad[~batch.cd8_mask] == 0.0)
    assert math.isfinite(float(perturbed_loss))
