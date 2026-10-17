"""Verification of `models.torch_sequence_gru` (task 9.1).

Scope. Three claims the task names, plus the two structural claims that
make them mean something:

- **shapes** -- `state_pred` is `(N, 3, 2)` and `hazards` is `(N, 23)`, with
  both trailing widths traced back to the constants they are supposed to
  come from rather than to literals repeated here;
- **eval-mode determinism** -- two `eval()`-mode forwards on identical
  inputs are bitwise identical (Requirement 2.6), asserted with
  `torch.equal` on every emitted tensor;
- **frozen widths** -- the hidden width is `decisions.md` Section 25's 16
  and the landmark count is `sequence_baseline.LANDMARK_WEEKS`', and both
  are load-bearing: a dataset reporting a different landmark count is a
  construction error, not a silent reshape;
- **structural determinism** -- no dropout and no batch-coupled
  normalization exists anywhere in the module tree, so eval-mode
  determinism holds *by construction* rather than because a global seed
  happened to be set. This is the difference between the determinism test
  above passing for the right reason and passing for the wrong one;
- **treatment as an ordinary column** -- the third column of the
  per-landmark input is `treatment_arm`, and the module contains no
  separate intervention encoder (Requirement 2.2). That absence is what
  makes this the comparison arm for `torch_intervention_aware`.

Batches come from `models.dataset.PatientDatasetBuilder` over hand-built
`PatientRecord`s, in the fixture style the other `tests/neural/` modules
use, so the model is exercised against the real emitted `TensorBatch` --
including its placeholder-filled missing landmark entries -- rather than
against tensors invented to suit it.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**
"""

from __future__ import annotations

import dataclasses
import random
from datetime import datetime, timedelta

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from causal.propensity import COVARIATE_FIELDS  # noqa: E402
from cohorts.cohort_construction import PatientRecord  # noqa: E402
from data.event_model import ClinicalEvent, MissingnessState  # noqa: E402
from data.timelines import build_patient_timeline  # noqa: E402
from models.dataset import (  # noqa: E402
    SURVIVAL_BIN_COUNT,
    DatasetDimensions,
    PatientDatasetBuilder,
)
from models.sequence_baseline import (  # noqa: E402
    CD4_LAB_CONCEPT_ID,
    CD8_LAB_CONCEPT_ID,
    INPUT_SIZE,
    LANDMARK_WEEKS,
    N_LANDMARKS,
    STATE_DIM,
)
from models.torch_sequence_gru import (  # noqa: E402
    FROZEN_HIDDEN_SIZE,
    TorchSequenceGRU,
    assemble_sequence_input,
    stochastic_or_batch_coupled_layers,
)
from protocols.eligibility_dsl import EligibilityResult  # noqa: E402
from protocols.endpoints import EndpointResult  # noqa: E402

pytestmark = pytest.mark.neural

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


# ---------------------------------------------------------------------------
# Fixtures: hand-built PatientRecords -> real TensorBatch
# ---------------------------------------------------------------------------


def _event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="baseline_cd4",
        value=350.0,
        unit="cells/mm3",
        modality="baseline",
        observed_at=ASSIGNMENT_TIME,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


def _make_patient_record(
    patient_id: str,
    *,
    treatment_arm: int = 1,
    baseline_cd4: float = 300.0,
    baseline_cd8: float = 900.0,
    karnofsky_score: float = 90.0,
    age: float = 35.0,
    event_time_weeks: float = 40.0,
    event_indicator: bool = True,
    cd4_by_landmark: tuple[float | None, ...] = (300.0, 340.0, 380.0),
    cd8_by_landmark: tuple[float | None, ...] = (900.0, 940.0, 980.0),
) -> PatientRecord:
    """A `PatientRecord` in the established `tests/neural/` fixture style.

    Landmark labs sit at exactly `ASSIGNMENT_TIME + weeks(w)` for each `w`
    in `LANDMARK_WEEKS`, which is how `data.loaders.actg175_adapter` writes
    them; a `None` entry becomes a `MissingnessState.MISSING` event rather
    than a dropped one, also matching the loader.
    """
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=baseline_cd4),
        _event(patient_id, concept_id="baseline_cd8", value=baseline_cd8),
        _event(patient_id, concept_id="karnofsky_score", value=karnofsky_score, unit="score"),
        _event(patient_id, concept_id="age", value=age, unit="years"),
        _event(patient_id, concept_id="symptomatic_status", value=False, unit=None),
        _event(patient_id, concept_id="prior_opportunistic_infection", value=False, unit=None),
        _event(patient_id, concept_id="prior_art_exposure_days", value=0.0, unit="days"),
        _event(
            patient_id,
            concept_id="zidovudine" if treatment_arm == 0 else "zidovudine+didanosine",
            value=1.0,
            unit=None,
            modality="treatment",
        ),
    ]
    for landmark_index, landmark_week in enumerate(LANDMARK_WEEKS):
        observed_at = ASSIGNMENT_TIME + timedelta(weeks=landmark_week)
        for concept_id, per_landmark in (
            (CD4_LAB_CONCEPT_ID, cd4_by_landmark),
            (CD8_LAB_CONCEPT_ID, cd8_by_landmark),
        ):
            value = per_landmark[landmark_index]
            events.append(
                _event(
                    patient_id,
                    concept_id=concept_id,
                    value=value,
                    modality="lab",
                    observed_at=observed_at,
                    missingness=(
                        MissingnessState.MISSING if value is None else MissingnessState.OBSERVED
                    ),
                )
            )

    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=build_patient_timeline(events),
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME + timedelta(weeks=event_time_weeks),
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="death" if event_indicator else None,
        ),
    )


def _cohort(n: int = 11, *, seed: int = 5) -> list[PatientRecord]:
    """`n` records with varied covariates and some missing landmark labs."""
    rng = random.Random(seed)
    return [
        _make_patient_record(
            f"P{i:03d}",
            treatment_arm=i % 2,
            baseline_cd4=200.0 + rng.uniform(-80.0, 80.0),
            baseline_cd8=850.0 + rng.uniform(-150.0, 150.0),
            karnofsky_score=85.0 + rng.uniform(-10.0, 10.0),
            age=36.0 + rng.uniform(-12.0, 12.0),
            event_time_weeks=rng.uniform(1.0, 200.0),
            event_indicator=rng.random() < 0.6,
            cd4_by_landmark=(250.0 + i, 275.0 + i, None if i % 3 == 0 else 300.0 + i),
            cd8_by_landmark=(880.0 + i, None if i % 4 == 0 else 900.0 + i, 950.0 + i),
        )
        for i in range(n)
    ]


@pytest.fixture
def batch():
    records = _cohort()
    builder = PatientDatasetBuilder().fit(records)
    return builder.build(records, "dev")


@pytest.fixture
def model(batch):
    torch.manual_seed(0)
    return TorchSequenceGRU(batch.dimensions)


# ---------------------------------------------------------------------------
# Output shapes (Requirement 2.3)
# ---------------------------------------------------------------------------


def test_forward_output_shapes(model, batch):
    """`state_pred` is `(N, 3, 2)` and `hazards` is `(N, 23)`.

    The expected trailing widths are the imported constants, not literals:
    `N_LANDMARKS` / `STATE_DIM` from the prototype and `SURVIVAL_BIN_COUNT`
    from `models.dataset` (`decisions.md` Section 25).
    """
    n_patients = len(batch.patient_ids)
    output = model(batch)

    assert output.state_pred.shape == (n_patients, N_LANDMARKS, STATE_DIM)
    assert output.hazards.shape == (n_patients, SURVIVAL_BIN_COUNT)
    assert output.hazard_logits.shape == (n_patients, SURVIVAL_BIN_COUNT)
    assert output.hidden_states.shape == (n_patients, N_LANDMARKS, FROZEN_HIDDEN_SIZE)

    # The design's literal shapes, stated once here as a cross-check that the
    # constants above still hold the values decisions.md froze.
    assert output.state_pred.shape[1:] == (3, 2)
    assert output.hazards.shape[1:] == (23,)


def test_hazards_are_finite_probabilities(model, batch):
    """The hazard head emits per-bin probabilities, the prototype's
    convention and what `training.discrete_time_censored_nll` consumes."""
    output = model(batch)

    assert torch.isfinite(output.hazards).all()
    assert bool((output.hazards > 0.0).all())
    assert bool((output.hazards < 1.0).all())
    assert torch.allclose(output.hazards, torch.sigmoid(output.hazard_logits))
    assert torch.isfinite(output.state_pred).all()


# ---------------------------------------------------------------------------
# Determinism (Requirement 2.6)
# ---------------------------------------------------------------------------


def test_two_eval_mode_forwards_are_bitwise_identical(model, batch):
    """Requirement 2.6, asserted with `torch.equal` -- bitwise, not
    `allclose`."""
    model.eval()
    with torch.no_grad():
        first = model(batch)
        second = model(batch)

    for field_name in ("state_pred", "hazards", "hazard_logits", "hidden_states"):
        left = getattr(first, field_name)
        right = getattr(second, field_name)
        assert torch.equal(left, right), f"{field_name} differed between eval() forwards"


def test_train_and_eval_mode_compute_the_same_function(model, batch):
    """A stronger reading of "deterministic by construction": with no
    dropout and no batch-coupled normalization, `train()` and `eval()` are
    not merely each self-consistent, they agree with each other."""
    model.train()
    with torch.no_grad():
        train_mode = model(batch)
    model.eval()
    with torch.no_grad():
        eval_mode = model(batch)

    assert torch.equal(train_mode.hazards, eval_mode.hazards)
    assert torch.equal(train_mode.state_pred, eval_mode.state_pred)


def test_no_dropout_or_batch_coupled_normalization_anywhere(model):
    """The structural half of Requirement 2.6.

    Walks the whole module tree and asserts nothing stochastic or
    batch-coupled exists, so eval-mode determinism cannot be an accident of
    a seeded global RNG. `nn.GRU` holds dropout as a float attribute rather
    than a child module, so it is checked directly as well.
    """
    assert stochastic_or_batch_coupled_layers(model) == ()

    # Belt and braces: the same claim expressed without the helper, so a
    # defect in the helper cannot hide a defect in the model.
    forbidden = (
        nn.Dropout,
        nn.AlphaDropout,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.InstanceNorm1d,
    )
    for submodule in model.modules():
        assert not isinstance(submodule, forbidden)
    assert float(model.gru.dropout) == 0.0


def test_stochastic_layer_detector_names_offenders():
    """The detector is only reassuring if it can fail: a tree that does
    carry dropout and batchnorm is reported, naming both."""
    offending = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.BatchNorm1d(4))
    offenders = stochastic_or_batch_coupled_layers(offending)

    assert len(offenders) == 2
    assert any("Dropout" in name for name in offenders)
    assert any("BatchNorm1d" in name for name in offenders)

    dropout_gru = nn.GRU(input_size=2, hidden_size=3, num_layers=2, dropout=0.5)
    assert any("GRU" in name for name in stochastic_or_batch_coupled_layers(dropout_gru))


# ---------------------------------------------------------------------------
# Frozen widths come from the imported constants (Requirements 2.1, 2.4)
# ---------------------------------------------------------------------------


def test_hidden_width_is_the_frozen_sixteen(model):
    """`decisions.md` Section 25's `hidden_size = 16`, reaching every layer
    that touches the hidden state (Requirement 2.4)."""
    assert FROZEN_HIDDEN_SIZE == 16
    assert model.hidden_size == FROZEN_HIDDEN_SIZE
    assert model.gru.hidden_size == FROZEN_HIDDEN_SIZE
    assert model.state_init_projection.out_features == FROZEN_HIDDEN_SIZE
    assert model.state_head.in_features == FROZEN_HIDDEN_SIZE
    assert model.hazard_head.in_features == FROZEN_HIDDEN_SIZE


def test_layer_widths_trace_to_the_prototype_constants(model):
    """Every width is the prototype's constant, so prototype/production
    drift is a shape error rather than a silent divergence (task 9's parent
    body, Requirement 2.1)."""
    assert model.landmark_count == N_LANDMARKS == len(LANDMARK_WEEKS)
    # h_0 is seeded from the Section 13 adjustment set (decisions.md Section
    # 26 amendment of 2026-10-16), not the prototype's four fields.
    assert model.state_init_projection.in_features == len(COVARIATE_FIELDS)
    assert model.gru.input_size == INPUT_SIZE
    assert model.state_head.out_features == STATE_DIM
    assert model.hazard_head.out_features == SURVIVAL_BIN_COUNT


def test_landmark_count_and_widths_are_load_bearing(batch):
    """A dataset reporting a layout the prototype does not describe is a
    construction error, not a reshape.

    This is what makes the constants above load-bearing rather than
    decorative: if `models.dataset` and `models.sequence_baseline` ever
    disagree, construction fails here with both numbers named.
    """
    real = batch.dimensions

    def _with(**overrides) -> DatasetDimensions:
        fields = dict(
            baseline_covariate_width=real.baseline_covariate_width,
            state_width=real.state_width,
            intervention_width=real.intervention_width,
            protocol_width=real.protocol_width,
            landmark_count=real.landmark_count,
            survival_bin_count=real.survival_bin_count,
        )
        fields.update(overrides)
        return DatasetDimensions(**fields)

    with pytest.raises(ValueError, match="landmark_count"):
        TorchSequenceGRU(_with(landmark_count=N_LANDMARKS + 1))
    with pytest.raises(ValueError, match="baseline_covariate_width"):
        TorchSequenceGRU(_with(baseline_covariate_width=0))
    with pytest.raises(ValueError, match="survival_bin_count"):
        TorchSequenceGRU(_with(survival_bin_count=SURVIVAL_BIN_COUNT - 1))
    with pytest.raises(ValueError, match="hidden_size"):
        TorchSequenceGRU(real, hidden_size=0)


# ---------------------------------------------------------------------------
# Treatment as an ordinary input column (Requirement 2.2)
# ---------------------------------------------------------------------------


def test_treatment_arm_is_an_ordinary_input_column(batch):
    """`[cd4, cd8, treatment_arm]` per landmark, no separate encoder."""
    sequence_input = assemble_sequence_input(batch)
    n_patients = len(batch.patient_ids)

    assert sequence_input.shape == (n_patients, N_LANDMARKS, INPUT_SIZE)
    assert torch.equal(sequence_input[:, :, 0], batch.cd4)
    assert torch.equal(sequence_input[:, :, 1], batch.cd8)
    for landmark_index in range(N_LANDMARKS):
        assert torch.equal(sequence_input[:, landmark_index, 2], batch.treatment_arm)


def test_module_has_no_separate_intervention_encoder(model):
    """Requirement 2.2's negative half: the treatment value reaches the GRU
    only as an input column, so no submodule consumes it separately.

    The model's parameterized submodules are exactly the baseline-state-init
    projection, the GRU, and the two heads. Any additional encoder would
    appear here, which is what distinguishes this baseline from
    `torch_intervention_aware`'s explicit-channel design.
    """
    parameterized = {
        name
        for name, submodule in model.named_modules()
        if name and any(True for _ in submodule.parameters(recurse=False))
    }
    assert parameterized == {"state_init_projection", "gru", "state_head", "hazard_head"}


# ---------------------------------------------------------------------------
# Autograd is the gradient mechanism (Requirement 2.5)
# ---------------------------------------------------------------------------


def test_gradients_reach_every_parameter_through_autograd(model, batch):
    """The point of the torch transcription: real autograd rather than the
    prototype's central-difference numerical gradient."""
    output = model(batch)
    (output.hazard_logits.sum() + output.state_pred.sum()).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} received a non-finite gradient"


def test_forward_from_tensors_matches_forward(model, batch):
    """The loose-tensor entry point computes the same function as the
    `TensorBatch` one, so the documented batch-first interface is a
    convenience over the tensor path, not a different model."""
    model.eval()
    with torch.no_grad():
        from_batch = model(batch)
        from_tensors = model.forward_from_tensors(batch.baseline, assemble_sequence_input(batch))

    assert torch.equal(from_batch.hazards, from_tensors.hazards)
    assert torch.equal(from_batch.state_pred, from_tensors.state_pred)


def test_malformed_batch_shapes_are_rejected(model, batch):
    """A width disagreement is a named error, not a broadcast."""
    wrong_baseline = dataclasses.replace(batch, baseline=batch.baseline[:, :-1])
    with pytest.raises(ValueError, match="baseline"):
        model(wrong_baseline)

    wrong_landmarks = dataclasses.replace(batch, cd4=batch.cd4[:, :-1])
    with pytest.raises(ValueError, match="cd4"):
        model(wrong_landmarks)

    fewer_rows = dataclasses.replace(batch, treatment_arm=batch.treatment_arm[:-1])
    with pytest.raises(ValueError, match="row count"):
        model(fewer_rows)


def test_state_prediction_for_a_landmark_never_sees_that_landmarks_measurement(model, batch):
    """Next-state regression (`decisions.md` Section 14.2): changing the
    CD4/CD8 observed at landmark k leaves the predictions for landmarks
    0..k unchanged, and moves only the predictions for later landmarks."""
    import dataclasses

    with torch.no_grad():
        reference = model(batch).state_pred
        for landmark in range(N_LANDMARKS):
            cd4 = batch.cd4.clone()
            cd8 = batch.cd8.clone()
            cd4[:, landmark] += 5.0
            cd8[:, landmark] -= 5.0
            perturbed = model(dataclasses.replace(batch, cd4=cd4, cd8=cd8)).state_pred

            assert torch.equal(perturbed[:, : landmark + 1], reference[:, : landmark + 1])
            if landmark + 1 < N_LANDMARKS:
                assert not torch.equal(perturbed[:, landmark + 1 :], reference[:, landmark + 1 :])


def test_first_landmark_prediction_comes_from_the_baseline_initialized_state(model, batch):
    """Landmark 0 is predicted from h_0 alone, which the baseline
    covariates set (Requirement 2.1)."""
    with torch.no_grad():
        output = model(batch)
        h0 = torch.tanh(model.state_init_projection(batch.baseline))
        assert torch.allclose(output.state_pred[:, 0], model.state_head(h0))


def test_survival_hazards_never_see_post_baseline_labs(model, batch):
    """The outcome is predicted from time zero (decisions.md Section 26
    amendment of 2026-10-16): changing the week-20 or week-96 CD4/CD8 leaves
    the hazards bitwise unchanged; changing a baseline covariate or the
    treatment moves them."""
    with torch.no_grad():
        reference = model(batch).hazard_logits
        for landmark in range(1, N_LANDMARKS):
            cd4 = batch.cd4.clone()
            cd8 = batch.cd8.clone()
            cd4[:, landmark] += 7.0
            cd8[:, landmark] -= 7.0
            perturbed = model(dataclasses.replace(batch, cd4=cd4, cd8=cd8)).hazard_logits
            assert torch.equal(perturbed, reference), f"landmark {landmark} leaked"

        shifted_baseline = dataclasses.replace(batch, baseline=batch.baseline + 1.0)
        assert not torch.equal(model(shifted_baseline).hazard_logits, reference)
        flipped = dataclasses.replace(batch, treatment_arm=1.0 - batch.treatment_arm)
        assert not torch.equal(model(flipped).hazard_logits, reference)


def test_gru_reads_every_adjustment_set_column(model, batch):
    """Each of the Section 13 columns, perturbed alone, moves the GRU's
    hazards, so no confounder is silently dropped."""
    with torch.no_grad():
        reference = model(batch).hazard_logits
        for column in range(batch.baseline.shape[1]):
            baseline = batch.baseline.clone()
            baseline[:, column] += 1.0
            perturbed = model(dataclasses.replace(batch, baseline=baseline)).hazard_logits
            assert not torch.equal(perturbed, reference), COVARIATE_FIELDS[column]
