"""Structural verification of `models.dataset` (task 8).

Scope note. This module covers what task 8 built, so the implementation is
verified as it lands: the `design.md` Section 4.1 field/shape/dtype
contract, delegation of every semantic decision to the module that owns it,
ascending-`patient_id` row ordering and input-order independence, the
fit-on-development-only split between `fit` and `build`, and the error paths
that must emit nothing.

Deliberately left to the two follow-on tasks, so this file does not
pre-empt them:

- `tests/neural/test_dataset.py` (task 8.1): Properties 18, 19, and 20 as
  `hypothesis` properties over 1-50 generated records with randomized
  elapsed times, randomized missingness patterns, and shuffled input order,
  with the explicit boundary examples 0.0 / 7.999 / 8.0 / 175.999 / 176.0 /
  1e6. The boundary assertions below are single-point checks of the same
  edges, not the generated property.
- `tests/neural/test_dataset.py` (task 8.2): the full error-path matrix
  (absent / NaN / inf event times) and the real-data fully-missing-week-96
  CD8 condition across all three splits. Covered here: empty input,
  duplicate id, negative elapsed weeks, and a fully missing CD8 week-96
  column in the development split.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 18.5**
"""

from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta

import pytest

torch = pytest.importorskip("torch")

from causal.baselines import _elapsed_weeks  # noqa: E402
from causal.propensity import COVARIATE_FIELDS  # noqa: E402
from cohorts.cohort_construction import PatientRecord  # noqa: E402
from data.event_model import ClinicalEvent, MissingnessState  # noqa: E402
from data.timelines import build_patient_timeline  # noqa: E402
from models.dataset import (  # noqa: E402
    BIN_WIDTH_WEEKS,
    FINAL_BIN_START_WEEKS,
    LANDMARK_COUNT,
    MISSING_PLACEHOLDER,
    SURVIVAL_BIN_COUNT,
    DatasetBuildError,
    PatientDatasetBuilder,
    survival_bin_index,
)
from models.intervention_aware import (  # noqa: E402
    INTERVENTION_INPUT_DIM,
    PROTOCOL_INPUT_DIM,
    STATE_INPUT_DIM,
)
from models.sequence_baseline import (  # noqa: E402
    CD4_LAB_CONCEPT_ID,
    CD8_LAB_CONCEPT_ID,
    LANDMARK_WEEKS,
)
from protocols.eligibility_dsl import EligibilityResult  # noqa: E402
from protocols.endpoints import EndpointResult  # noqa: E402

pytestmark = pytest.mark.neural

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)


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
    censoring_time_weeks: float | None = None,
    eligible: bool = True,
) -> PatientRecord:
    """A hand-built `PatientRecord` in the established fixture style.

    Landmark labs are written at exactly `ASSIGNMENT_TIME + weeks(w)` for
    each `w` in `LANDMARK_WEEKS`, which is how
    `data.loaders.actg175_adapter` writes them; a `None` entry is emitted as
    a `MissingnessState.MISSING` event rather than dropped, also matching the
    loader.
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
    if censoring_time_weeks is not None:
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

    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=build_patient_timeline(events),
        eligibility_result=EligibilityResult(eligible=eligible, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME + timedelta(weeks=event_time_weeks),
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="death" if event_indicator else None,
        ),
    )


def _cohort(n: int = 12, *, prefix: str = "P", seed: int = 7) -> list[PatientRecord]:
    rng = random.Random(seed)
    return [
        _make_patient_record(
            f"{prefix}{i:03d}",
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


# ---------------------------------------------------------------------------
# Structure: fields, shapes, dtypes, reported dimensions (Req 1.1, 1.7)
# ---------------------------------------------------------------------------


def test_batch_shapes_dtypes_and_dimensions_match_design_section_4_1():
    records = _cohort(9)
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    n = len(records)
    dimensions = builder.dimensions

    assert batch.split_name == "dev"
    assert len(batch.patient_ids) == n

    assert batch.baseline.shape == (n, dimensions.baseline_covariate_width)
    assert batch.state_init.shape == (n, dimensions.state_width)
    assert batch.cd4.shape == (n, dimensions.landmark_count)
    assert batch.cd8.shape == (n, dimensions.landmark_count)
    assert batch.cd4_mask.shape == (n, dimensions.landmark_count)
    assert batch.cd8_mask.shape == (n, dimensions.landmark_count)
    assert batch.treatment_arm.shape == (n,)
    assert batch.protocol.shape == (n, dimensions.protocol_width)
    assert batch.intervention.shape == (n, dimensions.intervention_width)
    assert batch.bin_index.shape == (n,)
    assert batch.censoring_mask.shape == (n,)

    for float_tensor in (
        batch.baseline,
        batch.state_init,
        batch.cd4,
        batch.cd8,
        batch.treatment_arm,
        batch.protocol,
        batch.intervention,
    ):
        assert float_tensor.dtype == torch.float32
    for bool_tensor in (batch.cd4_mask, batch.cd8_mask, batch.censoring_mask):
        assert bool_tensor.dtype == torch.bool
    assert batch.bin_index.dtype == torch.int64

    assert batch.dimensions == dimensions


def test_reported_dimensions_are_the_imported_widths_not_local_literals():
    """Requirement 1.7: widths are data, and they are the *imported* widths."""
    dimensions = PatientDatasetBuilder().dimensions

    assert dimensions.baseline_covariate_width == len(COVARIATE_FIELDS)
    assert dimensions.state_width == STATE_INPUT_DIM
    assert dimensions.intervention_width == INTERVENTION_INPUT_DIM
    assert dimensions.protocol_width == PROTOCOL_INPUT_DIM
    assert dimensions.landmark_count == LANDMARK_COUNT == len(LANDMARK_WEEKS)
    assert dimensions.survival_bin_count == SURVIVAL_BIN_COUNT


def test_treatment_arm_and_intervention_channel_agree_on_assignment():
    records = _cohort(8)
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    expected = torch.tensor(
        [float(r.treatment_arm) for r in sorted(records, key=lambda r: r.patient_id)],
        dtype=torch.float32,
    )
    assert torch.equal(batch.treatment_arm, expected)
    # `[treatment_arm, dose_or_default, adherence_or_default]`: ACTG175
    # records neither dose nor adherence, so both default to 0.0 while the
    # channel keeps its frozen width.
    assert torch.equal(batch.intervention[:, 0], expected)
    assert torch.equal(batch.intervention[:, 1:], torch.zeros(len(records), 2))


# ---------------------------------------------------------------------------
# Ordering and input-order independence (Requirement 1.6)
# ---------------------------------------------------------------------------


def test_rows_are_ordered_by_ascending_patient_id():
    records = _cohort(15)
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    assert list(batch.patient_ids) == sorted(batch.patient_ids)
    assert list(batch.patient_ids) == sorted(r.patient_id for r in records)


def test_shuffled_input_order_produces_identical_tensors():
    records = _cohort(15)
    shuffled = list(records)
    random.Random(99).shuffle(shuffled)
    assert [r.patient_id for r in shuffled] != [r.patient_id for r in records]

    builder = PatientDatasetBuilder().fit(records)
    from_original = builder.build(records, "dev")
    from_shuffled = builder.build(shuffled, "dev")

    assert from_original.patient_ids == from_shuffled.patient_ids
    for field_name in (
        "baseline",
        "state_init",
        "cd4",
        "cd8",
        "cd4_mask",
        "cd8_mask",
        "treatment_arm",
        "protocol",
        "intervention",
        "bin_index",
        "censoring_mask",
    ):
        assert torch.equal(
            getattr(from_original, field_name), getattr(from_shuffled, field_name)
        ), field_name


def test_row_index_refers_to_the_same_patient_in_every_tensor():
    records = _cohort(10)
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")
    by_id = {record.patient_id: record for record in records}

    for row, patient_id in enumerate(batch.patient_ids):
        record = by_id[patient_id]
        assert batch.treatment_arm[row].item() == float(record.treatment_arm)
        assert batch.bin_index[row].item() == survival_bin_index(_elapsed_weeks(record))


# ---------------------------------------------------------------------------
# Survival bins (Requirements 1.2, 1.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("elapsed_weeks", "expected_index"),
    [
        (0.0, 0),
        (7.999, 0),
        (BIN_WIDTH_WEEKS, 1),
        (175.999, 21),
        (FINAL_BIN_START_WEEKS, SURVIVAL_BIN_COUNT - 1),
        (1e6, SURVIVAL_BIN_COUNT - 1),
    ],
)
def test_survival_bin_index_at_the_frozen_boundaries(elapsed_weeks, expected_index):
    assert survival_bin_index(elapsed_weeks) == expected_index


def test_survival_bin_index_rejects_non_finite_and_negative_input():
    for bad in (float("nan"), float("inf"), -0.5):
        with pytest.raises(ValueError):
            survival_bin_index(bad)


def test_bin_index_delegates_elapsed_weeks_to_causal_baselines():
    """Requirement 1.3: the elapsed weeks driving bin assignment are exactly
    what `causal.baselines._elapsed_weeks` returns for the same record."""
    records = _cohort(14)
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    expected = torch.tensor(
        [
            survival_bin_index(_elapsed_weeks(record))
            for record in sorted(records, key=lambda r: r.patient_id)
        ],
        dtype=torch.int64,
    )
    assert torch.equal(batch.bin_index, expected)
    assert int(batch.bin_index.min()) >= 0
    assert int(batch.bin_index.max()) <= SURVIVAL_BIN_COUNT - 1


def test_censoring_mask_marks_only_censoring_before_the_assigned_bin_start():
    """Requirement 1.2's two-part condition, one record per outcome.

    `late_censored`'s censoring event coincides with its endpoint time, so it
    falls inside its own bin, not before it -- the real-data case, and
    correctly unmarked.
    """
    early_censored = _make_patient_record(
        "A_early_censored",
        event_time_weeks=100.0,  # bin 12, starting at week 96
        event_indicator=False,
        censoring_time_weeks=50.0,  # before week 96
    )
    late_censored = _make_patient_record(
        "B_late_censored",
        event_time_weeks=100.0,
        event_indicator=False,
        censoring_time_weeks=100.0,  # inside bin 12
    )
    had_event = _make_patient_record(
        "C_event",
        event_time_weeks=100.0,
        event_indicator=True,
        censoring_time_weeks=50.0,  # not censored, so irrelevant
    )
    records = [early_censored, late_censored, had_event]

    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    assert batch.patient_ids == ("A_early_censored", "B_late_censored", "C_event")
    assert batch.censoring_mask.tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# Missingness (Requirement 1.5)
# ---------------------------------------------------------------------------


def test_masks_are_true_for_observed_and_missing_entries_hold_the_placeholder():
    records = _cohort(12)
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    assert bool((~batch.cd4_mask).any()), "fixture should contain missing CD4 entries"
    assert bool((~batch.cd8_mask).any()), "fixture should contain missing CD8 entries"

    for values, mask in ((batch.cd4, batch.cd4_mask), (batch.cd8, batch.cd8_mask)):
        assert torch.equal(
            values[~mask], torch.full_like(values[~mask], MISSING_PLACEHOLDER)
        )


def test_no_emitted_tensor_contains_a_non_finite_value():
    records = _cohort(12)
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    for field_name in (
        "baseline",
        "state_init",
        "cd4",
        "cd8",
        "treatment_arm",
        "protocol",
        "intervention",
    ):
        tensor = getattr(batch, field_name)
        assert bool(torch.isfinite(tensor).all()), field_name


def test_landmark_column_missing_for_every_development_patient_is_never_fitted():
    """Requirement 1.9's real-data condition, narrowed to what task 8 owns.

    Task 8.2 extends this to all three splits; here the point is that a
    fully missing column produces no fitted parameter, an all-`False` mask,
    placeholder fill, and no exception.
    """
    records = [
        _make_patient_record(
            f"P{i:03d}",
            treatment_arm=i % 2,
            cd8_by_landmark=(900.0 + i, 940.0 + i, None),  # week-96 CD8 absent
        )
        for i in range(6)
    ]
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    week_96_index = LANDMARK_WEEKS.index(96)
    assert (CD8_LAB_CONCEPT_ID, week_96_index) not in builder.fitted_landmark_params
    assert (CD8_LAB_CONCEPT_ID, 0) in builder.fitted_landmark_params
    assert not bool(batch.cd8_mask[:, week_96_index].any())
    assert torch.equal(
        batch.cd8[:, week_96_index],
        torch.full((len(records),), MISSING_PLACEHOLDER),
    )


# ---------------------------------------------------------------------------
# fit / build separation (Requirements 1.4, 18.5)
# ---------------------------------------------------------------------------


def test_fit_is_idempotent_and_replaces_rather_than_accumulates():
    dev = _cohort(10, prefix="D", seed=1)
    first = PatientDatasetBuilder().fit(dev).fitted_landmark_params
    second = PatientDatasetBuilder().fit(dev).fitted_landmark_params
    refit = PatientDatasetBuilder().fit(dev).fit(dev).fitted_landmark_params

    assert first == second == refit


def test_building_validation_and_test_batches_leaves_parameters_unchanged():
    """Requirement 1.4 / 18.5: `build` never refits, whatever split it sees."""
    dev = _cohort(10, prefix="D", seed=1)
    # Validation/test rows are drawn from a deliberately different CD4/CD8
    # distribution, so a refit would be visible as changed parameters.
    val = [
        _make_patient_record(
            f"V{i:03d}",
            cd4_by_landmark=(900.0 + i, 950.0 + i, 990.0 + i),
            cd8_by_landmark=(50.0 + i, 60.0 + i, 70.0 + i),
        )
        for i in range(6)
    ]
    test = [
        _make_patient_record(
            f"T{i:03d}",
            cd4_by_landmark=(10.0 + i, 20.0 + i, 30.0 + i),
            cd8_by_landmark=(5000.0 + i, 5100.0 + i, 5200.0 + i),
        )
        for i in range(6)
    ]

    builder = PatientDatasetBuilder().fit(dev)
    before = builder.fitted_landmark_params

    builder.build(dev, "dev")
    builder.build(val, "val")
    builder.build(test, "test")

    assert builder.fitted_landmark_params == before


def test_development_fitted_parameters_use_observed_entries_only():
    """Requirement 1.4: masked-missing entries enter no mean or stdev."""
    observed_week_20 = [310.0, 330.0, 350.0]
    dev = [
        _make_patient_record("D000", cd4_by_landmark=(300.0, observed_week_20[0], 400.0)),
        _make_patient_record("D001", cd4_by_landmark=(300.0, observed_week_20[1], 400.0)),
        _make_patient_record("D002", cd4_by_landmark=(300.0, observed_week_20[2], 400.0)),
        _make_patient_record("D003", cd4_by_landmark=(300.0, None, 400.0)),
        _make_patient_record("D004", cd4_by_landmark=(300.0, None, 400.0)),
    ]
    params = PatientDatasetBuilder().fit(dev).fitted_landmark_params
    week_20_index = LANDMARK_WEEKS.index(20)
    fitted = params[(CD4_LAB_CONCEPT_ID, week_20_index)]

    assert fitted.mean == pytest.approx(statistics.fmean(observed_week_20))
    assert fitted.stdev == pytest.approx(statistics.pstdev(observed_week_20))


def test_validation_rows_are_transformed_with_the_development_parameters():
    dev = [
        _make_patient_record(f"D{i:03d}", cd4_by_landmark=(300.0 + 10.0 * i, 320.0, 340.0))
        for i in range(5)
    ]
    val = [_make_patient_record("V000", cd4_by_landmark=(300.0, 320.0, 340.0))]

    builder = PatientDatasetBuilder().fit(dev)
    fitted = builder.fitted_landmark_params[(CD4_LAB_CONCEPT_ID, 0)]
    val_batch = builder.build(val, "val")

    expected = (300.0 - fitted.mean) / fitted.stdev
    assert val_batch.cd4[0, 0].item() == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Error paths (Requirements 1.8, 1.10)
# ---------------------------------------------------------------------------


def test_build_before_fit_is_refused():
    records = _cohort(4)
    builder = PatientDatasetBuilder()
    with pytest.raises(DatasetBuildError, match="before fit"):
        builder.build(records, "dev")


def test_empty_input_raises_and_emits_nothing():
    with pytest.raises(DatasetBuildError, match="empty"):
        PatientDatasetBuilder().fit([])


def test_duplicate_patient_id_is_named():
    records = _cohort(4)
    duplicated = [*records, records[1]]
    builder = PatientDatasetBuilder().fit(records)

    with pytest.raises(DatasetBuildError) as excinfo:
        builder.build(duplicated, "dev")
    assert records[1].patient_id in str(excinfo.value)


def test_negative_elapsed_weeks_names_the_patient_and_the_reason():
    good = _make_patient_record("P000")
    bad = _make_patient_record("P001", event_time_weeks=-3.0)
    builder = PatientDatasetBuilder().fit([good])

    with pytest.raises(DatasetBuildError) as excinfo:
        builder.build([good, bad], "dev")
    message = str(excinfo.value)
    assert "P001" in message
    assert "negative elapsed weeks" in message


def test_unrecognized_split_name_is_refused():
    records = _cohort(4)
    builder = PatientDatasetBuilder().fit(records)
    with pytest.raises(DatasetBuildError, match="unrecognized split_name"):
        builder.build(records, "holdout")
