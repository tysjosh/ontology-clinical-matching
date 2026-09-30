"""Error paths and the fully-missing-column condition of `models.dataset` (task 8.2).

Why a third dataset test module. `tests/neural/test_dataset_builder.py`
(task 8) covers the structural contract plus a first cut at the error paths;
`tests/neural/test_dataset.py` (task 8.1) carries the `hypothesis`
properties. This file is task 8.2's scope only, and deliberately does not
re-derive either of theirs.

What is new here versus `test_dataset_builder.py`
-------------------------------------------------------------------------

That module asserts the *exception* on empty input, a duplicate
`patient_id`, and negative elapsed weeks. It does not assert the second half
of Requirements 1.8 and 1.10 -- "SHALL emit no tensors for that split rather
than a partially populated batch". Raising is necessary but not sufficient:
a builder could construct a `TensorBatch` over the good rows, then raise. So
every error test below runs under `tensor_batch_spy`, which wraps
`models.dataset.TensorBatch` and counts constructions, and additionally
asserts the builder's own fitted state is unchanged by the failed call.

Also new: the absent / non-datetime (`nan`, `inf`) event-time cases, and the
fully-missing-week-96-CD8 condition across all three splits -- including the
case where validation and test rows *do* carry week-96 CD8 values and are
still masked missing, because no development parameter exists to transform
them with (Requirement 1.9).

On the non-finite-elapsed branch. `_validated_elapsed_weeks` guards
`math.isfinite(elapsed)` after calling `causal.baselines._elapsed_weeks`.
That guard is unreachable through the public API and is documented as such
by `test_elapsed_weeks_from_a_real_datetime_is_always_finite` below rather
than exercised through a faked return value: `_elapsed_weeks` computes
`(event_time - assignment_time).days / 7.0`, and `timedelta.days` is always
an `int`, so a genuine `datetime` event time can never produce a NaN or an
infinity. A `float("nan")` written into `EndpointResult.event_time` -- the
only way to get a non-finite value in there, since the field is declared
`datetime` -- fails at the subtraction with a `TypeError` and is rejected by
the `except (ValueError, TypeError)` branch instead. The `isfinite` check is
therefore dead defensive code today; it is kept because it is the branch
that would catch a future change to the elapsed-week helper's return type,
and removing it would trade a named error for a wrong bin index.

**Validates: Requirements 1.5, 1.8, 1.9, 1.10**
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

torch = pytest.importorskip("torch")

import models.dataset as dataset_module  # noqa: E402
from causal.baselines import _elapsed_weeks  # noqa: E402
from cohorts.cohort_construction import PatientRecord  # noqa: E402
from data.event_model import ClinicalEvent, MissingnessState  # noqa: E402
from data.timelines import build_patient_timeline  # noqa: E402
from models.dataset import (  # noqa: E402
    MISSING_PLACEHOLDER,
    DatasetBuildError,
    PatientDatasetBuilder,
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

WEEK_96_INDEX = LANDMARK_WEEKS.index(96)


# ---------------------------------------------------------------------------
# Fixture records
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
    event_time_weeks: float = 40.0,
    event_time_override: object = None,
    use_event_time_override: bool = False,
    event_indicator: bool = True,
    cd4_by_landmark: tuple[float | None, ...] = (300.0, 340.0, 380.0),
    cd8_by_landmark: tuple[float | None, ...] = (900.0, 940.0, 980.0),
    with_treatment_event: bool = True,
) -> PatientRecord:
    """A hand-built `PatientRecord`, in `test_dataset_builder.py`'s style.

    `use_event_time_override` writes `event_time_override` straight into
    `EndpointResult.event_time`, which is how the absent / non-datetime
    event-time cases below are constructed: `EndpointResult` is a plain
    frozen dataclass and enforces no runtime type, so a malformed record is
    reachable exactly this way and no other.
    """
    events = [
        _event(patient_id, concept_id="baseline_cd4", value=baseline_cd4),
        _event(patient_id, concept_id="baseline_cd8", value=baseline_cd8),
        _event(patient_id, concept_id="karnofsky_score", value=90.0, unit="score"),
        _event(patient_id, concept_id="age", value=35.0, unit="years"),
        _event(patient_id, concept_id="symptomatic_status", value=False, unit=None),
        _event(patient_id, concept_id="prior_opportunistic_infection", value=False, unit=None),
        _event(patient_id, concept_id="prior_art_exposure_days", value=0.0, unit="days"),
    ]
    if with_treatment_event:
        events.append(
            _event(
                patient_id,
                concept_id="zidovudine" if treatment_arm == 0 else "zidovudine+didanosine",
                value=1.0,
                unit=None,
                modality="treatment",
            )
        )
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

    event_time: object
    if use_event_time_override:
        event_time = event_time_override
    else:
        event_time = ASSIGNMENT_TIME + timedelta(weeks=event_time_weeks)

    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=build_patient_timeline(events),
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=event_time,  # type: ignore[arg-type]
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="death" if event_indicator else None,
        ),
    )


def _healthy_cohort(n: int = 5, *, prefix: str = "D") -> list[PatientRecord]:
    return [
        _make_patient_record(
            f"{prefix}{i:03d}",
            treatment_arm=i % 2,
            baseline_cd4=280.0 + 5.0 * i,
            baseline_cd8=880.0 + 7.0 * i,
            event_time_weeks=20.0 + 9.0 * i,
            cd4_by_landmark=(300.0 + i, 330.0 + i, 360.0 + i),
            cd8_by_landmark=(900.0 + i, 930.0 + i, 960.0 + i),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# "Nothing partial is emitted" instrumentation
# ---------------------------------------------------------------------------


@pytest.fixture
def tensor_batch_spy(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Counts every `TensorBatch` construction inside `models.dataset`.

    Delegates to the real class, so behaviour is unchanged and a successful
    build still returns a genuine `TensorBatch`; the list is what lets an
    error test assert *zero* were constructed rather than only that an
    exception escaped.
    """
    real_tensor_batch = dataset_module.TensorBatch
    constructed: list[object] = []

    def recording(*args, **kwargs):
        batch = real_tensor_batch(*args, **kwargs)
        constructed.append(batch)
        return batch

    monkeypatch.setattr(dataset_module, "TensorBatch", recording)
    return constructed


def test_tensor_batch_spy_sees_a_successful_build(tensor_batch_spy):
    """The instrumentation is load-bearing, so prove it can observe a build.

    Without this, a spy that silently failed to patch would make every
    "no `TensorBatch` was constructed" assertion below vacuously true.
    """
    records = _healthy_cohort(4)
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")

    assert len(tensor_batch_spy) == 1
    assert tensor_batch_spy[0] is batch


# ---------------------------------------------------------------------------
# Requirement 1.8: empty input and duplicate identifiers
# ---------------------------------------------------------------------------


def test_empty_input_to_build_says_empty_and_emits_no_tensor_batch(tensor_batch_spy):
    records = _healthy_cohort(4)
    builder = PatientDatasetBuilder().fit(records)
    before = builder.fitted_landmark_params

    with pytest.raises(DatasetBuildError) as excinfo:
        builder.build([], "dev")

    message = str(excinfo.value)
    assert "empty" in message
    assert "duplicate" not in message, "the message must say which of the two defects it was"
    assert tensor_batch_spy == []
    assert builder.fitted_landmark_params == before


def test_empty_input_to_fit_leaves_the_builder_unfitted(tensor_batch_spy):
    builder = PatientDatasetBuilder()

    with pytest.raises(DatasetBuildError, match="empty"):
        builder.fit([])

    assert not builder.is_fitted
    assert tensor_batch_spy == []
    with pytest.raises(DatasetBuildError, match="before fit"):
        _ = builder.fitted_landmark_params


def test_duplicate_patient_id_names_the_identifier_and_emits_no_tensor_batch(tensor_batch_spy):
    records = _healthy_cohort(4)
    builder = PatientDatasetBuilder().fit(records)
    before = builder.fitted_landmark_params
    duplicated = [*records, records[2]]

    with pytest.raises(DatasetBuildError) as excinfo:
        builder.build(duplicated, "dev")

    message = str(excinfo.value)
    assert "duplicate" in message
    assert records[2].patient_id in message
    assert "empty" not in message, "the message must say which of the two defects it was"
    assert tensor_batch_spy == []
    assert builder.fitted_landmark_params == before


def test_duplicate_patient_id_at_fit_time_leaves_previous_parameters_intact(tensor_batch_spy):
    """A failed refit must not half-replace the fitted parameters."""
    first_cohort = _healthy_cohort(4, prefix="D")
    builder = PatientDatasetBuilder().fit(first_cohort)
    before = builder.fitted_landmark_params

    second_cohort = _healthy_cohort(3, prefix="E")
    with pytest.raises(DatasetBuildError, match="duplicate"):
        builder.fit([*second_cohort, second_cohort[0]])

    assert builder.is_fitted
    assert builder.fitted_landmark_params == before
    assert tensor_batch_spy == []


# ---------------------------------------------------------------------------
# Requirement 1.10: absent / non-datetime / negative-elapsed event times
# ---------------------------------------------------------------------------

# `(case name, EndpointResult.event_time override, substrings the message must
# carry)`. The reason phrase differs per case because Requirement 1.10 asks for
# "the reason the event time was rejected", not a single generic sentence.
_BAD_EVENT_TIME_CASES = [
    pytest.param(
        {"use_event_time_override": True, "event_time_override": None},
        ["event_time is absent"],
        id="absent",
    ),
    pytest.param(
        {"use_event_time_override": True, "event_time_override": float("nan")},
        ["elapsed weeks could not be computed"],
        id="nan",
    ),
    pytest.param(
        {"use_event_time_override": True, "event_time_override": float("inf")},
        ["elapsed weeks could not be computed"],
        id="inf",
    ),
    pytest.param(
        {"use_event_time_override": True, "event_time_override": float("-inf")},
        ["elapsed weeks could not be computed"],
        id="negative-inf",
    ),
    pytest.param(
        {"event_time_weeks": -3.0},
        ["negative elapsed weeks"],
        id="negative-elapsed",
    ),
    pytest.param(
        {"with_treatment_event": False},
        ["elapsed weeks could not be computed", "no treatment_events"],
        id="no-time-zero",
    ),
]


@pytest.mark.parametrize(("overrides", "expected_phrases"), _BAD_EVENT_TIME_CASES)
def test_bad_event_time_in_build_names_the_patient_and_the_reason(
    overrides, expected_phrases, tensor_batch_spy
):
    """Requirement 1.10, both halves: a named error *and* no tensors at all.

    The bad record is placed last among several good ones so that a
    row-by-row implementation would have had to build most of the batch
    before reaching it -- which is exactly what `tensor_batch_spy == []`
    rules out.
    """
    good = _healthy_cohort(4, prefix="D")
    bad = _make_patient_record("Z999", **overrides)
    builder = PatientDatasetBuilder().fit(good)
    before = builder.fitted_landmark_params

    with pytest.raises(DatasetBuildError) as excinfo:
        builder.build([*good, bad], "dev")

    message = str(excinfo.value)
    assert "Z999" in message, message
    for phrase in expected_phrases:
        assert phrase in message, message
    assert tensor_batch_spy == []
    assert builder.fitted_landmark_params == before


@pytest.mark.parametrize(("overrides", "expected_phrases"), _BAD_EVENT_TIME_CASES)
def test_bad_event_time_in_fit_is_rejected_before_any_parameter_is_stored(
    overrides, expected_phrases, tensor_batch_spy
):
    """The same matrix at fit time: validation precedes parameter storage."""
    good = _healthy_cohort(4, prefix="D")
    bad = _make_patient_record("Z999", **overrides)
    builder = PatientDatasetBuilder()

    with pytest.raises(DatasetBuildError) as excinfo:
        builder.fit([*good, bad])

    message = str(excinfo.value)
    assert "Z999" in message, message
    for phrase in expected_phrases:
        assert phrase in message, message
    assert not builder.is_fitted
    assert tensor_batch_spy == []


def test_elapsed_weeks_from_a_real_datetime_is_always_finite():
    """Why `_validated_elapsed_weeks`' `isfinite` guard is unreachable.

    `causal.baselines._elapsed_weeks` returns `timedelta.days / 7.0`, and
    `timedelta.days` is an `int` for every representable pair of
    `datetime`s -- including the extremes below. So no `PatientRecord`
    carrying an actual `datetime` event time can reach the non-finite
    branch, and the `float("nan")` / `float("inf")` cases in the matrix
    above are rejected one branch earlier, at the subtraction. This test
    records that reasoning as an executable claim instead of leaving the
    branch looking untested-by-oversight.
    """
    for event_time_weeks in (0.0, 1.0, 52.0 * 200.0, -52.0 * 200.0):
        record = _make_patient_record("P000", event_time_weeks=event_time_weeks)
        elapsed = _elapsed_weeks(record)
        assert isinstance(elapsed, float)
        assert elapsed == elapsed  # not NaN
        assert abs(elapsed) != float("inf")

    for extreme in (datetime.min, datetime.max):
        record = _make_patient_record(
            "P001", use_event_time_override=True, event_time_override=extreme
        )
        elapsed = _elapsed_weeks(record)
        assert isinstance(elapsed, float)
        assert elapsed == elapsed
        assert abs(elapsed) != float("inf")


# ---------------------------------------------------------------------------
# Requirement 1.9 / 1.5: the real-data fully missing week-96 CD8 column
# ---------------------------------------------------------------------------


def _three_splits(
    *,
    dev_week_96_cd8: float | None,
    val_week_96_cd8: float | None,
    test_week_96_cd8: float | None,
) -> tuple[list[PatientRecord], list[PatientRecord], list[PatientRecord]]:
    """Dev/val/test cohorts differing only in their week-96 CD8 entry."""

    def cohort(prefix: str, n: int, week_96: float | None) -> list[PatientRecord]:
        return [
            _make_patient_record(
                f"{prefix}{i:03d}",
                treatment_arm=i % 2,
                event_time_weeks=20.0 + 7.0 * i,
                cd4_by_landmark=(300.0 + i, 330.0 + i, 360.0 + i),
                cd8_by_landmark=(
                    900.0 + i,
                    930.0 + i,
                    None if week_96 is None else week_96 + i,
                ),
            )
            for i in range(n)
        ]

    return (
        cohort("D", 6, dev_week_96_cd8),
        cohort("V", 4, val_week_96_cd8),
        cohort("T", 3, test_week_96_cd8),
    )


def test_fully_missing_week_96_cd8_is_masked_in_dev_val_and_test():
    """Requirement 1.9 across all three splits, the real ACTG175 v3.2 case.

    Week-96 CD8 is 100% missing in the delivered dataset, so the column is
    absent in every split. Extends `test_dataset_builder.py`'s
    development-only version of this check to `val` and `test`, which is
    where "SHALL set that column's missingness mask to missing for every
    patient in the Development_Split, the Validation_Split, and the
    Test_Split" actually bites.
    """
    dev, val, test = _three_splits(
        dev_week_96_cd8=None, val_week_96_cd8=None, test_week_96_cd8=None
    )

    builder = PatientDatasetBuilder().fit(dev)  # raises nothing

    # No preprocessing transformation was fitted for that column, while the
    # two observed CD8 columns and every CD4 column were.
    params = builder.fitted_landmark_params
    assert (CD8_LAB_CONCEPT_ID, WEEK_96_INDEX) not in params
    assert (CD8_LAB_CONCEPT_ID, 0) in params
    assert (CD8_LAB_CONCEPT_ID, 1) in params
    assert (CD4_LAB_CONCEPT_ID, WEEK_96_INDEX) in params

    for split_name, records in (("dev", dev), ("val", val), ("test", test)):
        batch = builder.build(records, split_name)  # raises nothing

        column = batch.cd8[:, WEEK_96_INDEX]
        mask_column = batch.cd8_mask[:, WEEK_96_INDEX]

        assert mask_column.shape == (len(records),), split_name
        assert not bool(mask_column.any()), f"{split_name}: cd8_mask[:, 2] must be all False"
        assert torch.equal(
            column, torch.full((len(records),), MISSING_PLACEHOLDER, dtype=column.dtype)
        ), split_name
        assert bool(torch.isfinite(batch.cd8).all()), split_name
        # The neighbouring columns are unaffected: only week 96 is missing.
        assert bool(batch.cd8_mask[:, 0].all()), split_name
        assert bool(batch.cd8_mask[:, 1].all()), split_name
        assert bool(batch.cd4_mask.all()), split_name

    # Building three splits never fits anything, so the absence persists.
    assert builder.fitted_landmark_params == params


def test_column_missing_in_development_only_is_still_masked_in_val_and_test():
    """The sharper reading of Requirement 1.9.

    Development has no week-96 CD8 at all, so no parameter is fitted; the
    validation and test rows here *do* carry week-96 CD8 values. They must
    still come out masked missing and placeholder-filled, because
    transforming them would require a parameter that development never
    produced -- and fitting one from validation or test rows would be the
    leak Requirement 1.4 forbids.
    """
    dev, val, test = _three_splits(
        dev_week_96_cd8=None, val_week_96_cd8=970.0, test_week_96_cd8=1010.0
    )

    builder = PatientDatasetBuilder().fit(dev)
    assert (CD8_LAB_CONCEPT_ID, WEEK_96_INDEX) not in builder.fitted_landmark_params

    for split_name, records in (("val", val), ("test", test)):
        batch = builder.build(records, split_name)
        column = batch.cd8[:, WEEK_96_INDEX]

        assert not bool(batch.cd8_mask[:, WEEK_96_INDEX].any()), split_name
        assert torch.equal(
            column, torch.full((len(records),), MISSING_PLACEHOLDER, dtype=column.dtype)
        ), split_name

    assert (CD8_LAB_CONCEPT_ID, WEEK_96_INDEX) not in builder.fitted_landmark_params


def test_every_landmark_column_missing_everywhere_still_builds():
    """The degenerate end of Requirement 1.9: no CD4 or CD8 value anywhere.

    Nothing is fitted for any landmark column, both mask tensors are
    entirely `False`, both observation tensors are entirely the placeholder,
    and the build completes without raising -- which is what "SHALL complete
    the build without raising an error" has to mean at its limit.
    """
    records = [
        _make_patient_record(
            f"D{i:03d}",
            treatment_arm=i % 2,
            event_time_weeks=20.0 + 7.0 * i,
            cd4_by_landmark=(None, None, None),
            cd8_by_landmark=(None, None, None),
        )
        for i in range(5)
    ]

    builder = PatientDatasetBuilder().fit(records)
    assert builder.fitted_landmark_params == {}

    batch = builder.build(records, "dev")
    n = len(records)

    assert not bool(batch.cd4_mask.any())
    assert not bool(batch.cd8_mask.any())
    for values in (batch.cd4, batch.cd8):
        assert torch.equal(values, torch.full((n, 3), MISSING_PLACEHOLDER, dtype=values.dtype))
        assert bool(torch.isfinite(values).all())
