"""Correctness Properties 18, 19, and 20 for `models.dataset` (task 8.1).

Generated counterpart to `tests/neural/test_dataset_builder.py`. That module
pins the Section 4.1 contract at hand-built examples; this one quantifies the
three named properties over `hypothesis`-generated cohorts of 1-50
`PatientRecord` objects with randomized elapsed times, randomized missingness
patterns (including landmark columns forced missing for *every* generated
patient), and shuffled input order:

- **Property 18** -- bin assignment is total, ordered, and delegated. Split
  across two tests because the property has two quantifiers: one over
  elapsed-week *values* (totality, monotonicity, the open final bin, with the
  frozen boundary values 0.0 / 7.999 / 8.0 / 175.999 / 176.0 / 1e6 as explicit
  examples), one over *records* (the elapsed weeks driving assignment equal
  `causal.baselines._elapsed_weeks` on the same record).
- **Property 19** -- batch structure and ordering are consistent and
  deterministic.
- **Property 20** -- preprocessing is fitted once, on development data only.

Two constraints this module works within, both properties of the code under
test rather than of the tests:

- `survival_bin_index` *raises* on non-finite or negative input, so the
  elapsed-week strategy is bounded to finite non-negative floats. The rejected
  inputs are covered as error paths in the sibling module, not here.
- `TensorBatch` is `@dataclass(frozen=True, eq=False)`, so batches are
  compared field by field with `torch.equal` (`_assert_identical_tensors`)
  and never with `==`.

The record builder is imported from the sibling module rather than restated:
generated records must follow exactly the loader's landmark-placement
convention, and there should be one definition of that fixture shape.

**Validates: Requirements 1.2, 1.3, 1.4, 1.6, 1.7, 18.5**
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from causal.baselines import _elapsed_weeks  # noqa: E402
from causal.propensity import COVARIATE_FIELDS  # noqa: E402
from cohorts.cohort_construction import PatientRecord  # noqa: E402
from models.dataset import (  # noqa: E402
    BIN_WIDTH_WEEKS,
    FINAL_BIN_START_WEEKS,
    LANDMARK_COUNT,
    MISSING_PLACEHOLDER,
    SURVIVAL_BIN_COUNT,
    PatientDatasetBuilder,
    TensorBatch,
    landmark_values,
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
from tests.neural.test_dataset_builder import _make_patient_record  # noqa: E402

pytestmark = pytest.mark.neural

#: Scalar properties are cheap; cohort properties build up to 50 records with
#: ~14 events each per example, so they run at a lower budget. Both follow the
#: suite's `deadline=None` convention (see `tests/unit/test_readiness.py`).
_MAX_EXAMPLES = 50
_COHORT_MAX_EXAMPLES = 20

#: Cohort generation is deliberately expensive (50 records of 14 events), and
#: the size of a 50-record draw is what the property is about, so neither the
#: timing nor the data-size health check carries information here.
_COHORT_SETTINGS = settings(
    max_examples=_COHORT_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

#: Every tensor field of a `TensorBatch`, for field-wise comparison.
_TENSOR_FIELDS: tuple[str, ...] = (
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
)

#: `(field name, DatasetDimensions attribute, independently imported width)`
#: for every 2-D tensor. The third element is imported from the module that
#: owns the width, so this table checks reported dimensions against the
#: authority rather than against `models.dataset`'s own view.
_TRAILING_WIDTHS: tuple[tuple[str, str, int], ...] = (
    ("baseline", "baseline_covariate_width", len(COVARIATE_FIELDS)),
    ("state_init", "state_width", STATE_INPUT_DIM),
    ("cd4", "landmark_count", LANDMARK_COUNT),
    ("cd8", "landmark_count", LANDMARK_COUNT),
    ("cd4_mask", "landmark_count", LANDMARK_COUNT),
    ("cd8_mask", "landmark_count", LANDMARK_COUNT),
    ("protocol", "protocol_width", PROTOCOL_INPUT_DIM),
    ("intervention", "intervention_width", INTERVENTION_INPUT_DIM),
)

#: Fields that are one value per patient, so `(N,)`.
_ROW_VECTOR_FIELDS: tuple[str, ...] = ("treatment_arm", "bin_index", "censoring_mask")

#: The six CD4/CD8 landmark columns, as `(concept_id, landmark_index)` keys.
_LANDMARK_COLUMNS: tuple[tuple[str, int], ...] = tuple(
    (concept_id, landmark_index)
    for concept_id in (CD4_LAB_CONCEPT_ID, CD8_LAB_CONCEPT_ID)
    for landmark_index in range(LANDMARK_COUNT)
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

#: A landmark measurement, or `None` for a missing one.
_landmark_value = st.one_of(
    st.none(),
    st.floats(min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
)

#: Elapsed weeks to the endpoint. The upper bound is well past
#: `FINAL_BIN_START_WEEKS`, so the open final bin is generated, not only
#: reached through the explicit boundary examples.
_event_time_weeks = st.floats(min_value=0.0, max_value=260.0, allow_nan=False, allow_infinity=False)

#: Finite non-negative elapsed weeks -- `survival_bin_index`'s whole domain.
#: It raises on anything outside this, which the sibling module covers.
_elapsed_weeks_value = st.floats(
    min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False
)


@st.composite
def _cohorts(draw, min_size: int = 1, max_size: int = 50) -> list[PatientRecord]:
    """1-50 unique `PatientRecord`s with randomized values and missingness.

    `forced_missing` columns are set missing for *every* generated patient,
    which is Requirement 1.9's fully-missing-column condition arising inside
    the generated cohort rather than only in a hand-built fixture.

    The cohort size is drawn as an integer and the identifiers are then a
    *permutation* of `P0..P{size-1}`, rather than a `unique=True` list of
    drawn identifiers: uniqueness is then structural instead of enforced by a
    filter that exhausts and discards examples, so cohorts across the whole
    1-50 range are actually generated. Identifier *values* are not what the
    properties quantify over; the count and the input order are.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Not zero-padded, so string order differs from numeric order (`P10`
    # sorts before `P9`) and the ascending-`patient_id` assertions cannot
    # pass on a batch that merely preserved insertion order.
    patient_ids = draw(st.permutations([f"P{number}" for number in range(size)]))
    forced_missing = draw(st.sets(st.sampled_from(_LANDMARK_COLUMNS), max_size=2))

    records: list[PatientRecord] = []
    for patient_id in patient_ids:
        per_concept: dict[str, list[float | None]] = {}
        for concept_id in (CD4_LAB_CONCEPT_ID, CD8_LAB_CONCEPT_ID):
            values = list(draw(st.tuples(*(_landmark_value for _ in LANDMARK_WEEKS))))
            for landmark_index in range(LANDMARK_COUNT):
                if (concept_id, landmark_index) in forced_missing:
                    values[landmark_index] = None
            per_concept[concept_id] = values

        records.append(
            _make_patient_record(
                patient_id,
                treatment_arm=draw(st.sampled_from((0, 1))),
                baseline_cd4=draw(st.floats(min_value=1.0, max_value=1500.0)),
                baseline_cd8=draw(st.floats(min_value=1.0, max_value=3000.0)),
                karnofsky_score=draw(st.floats(min_value=0.0, max_value=100.0)),
                age=draw(st.floats(min_value=18.0, max_value=90.0)),
                event_time_weeks=draw(_event_time_weeks),
                event_indicator=draw(st.booleans()),
                cd4_by_landmark=tuple(per_concept[CD4_LAB_CONCEPT_ID]),
                cd8_by_landmark=tuple(per_concept[CD8_LAB_CONCEPT_ID]),
                censoring_time_weeks=draw(st.one_of(st.none(), _event_time_weeks)),
                eligible=draw(st.booleans()),
            )
        )
    return records


@st.composite
def _shuffled_cohorts(draw) -> tuple[list[PatientRecord], list[PatientRecord]]:
    """A cohort and a permutation of it -- the same records, different order."""
    records = draw(_cohorts())
    return records, list(draw(st.permutations(records)))


@st.composite
def _three_splits(
    draw,
) -> tuple[list[PatientRecord], list[PatientRecord], list[PatientRecord]]:
    """Development, validation, and test cohorts, each non-empty and disjoint.

    Striding one generated cohort keeps identifiers unique across the three
    splits while letting their CD4/CD8 distributions differ, so a refit on
    validation or test rows would be visible as changed parameters.
    """
    records = draw(_cohorts(min_size=3, max_size=30))
    return records[0::3], records[1::3], records[2::3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_identical_tensors(left: TensorBatch, right: TensorBatch) -> None:
    """Field-wise equality. `TensorBatch` is `eq=False`; see module docstring."""
    assert left.patient_ids == right.patient_ids
    for field_name in _TENSOR_FIELDS:
        assert torch.equal(getattr(left, field_name), getattr(right, field_name)), field_name


def _observed_development_values(
    records: Sequence[PatientRecord], concept_id: str, landmark_index: int
) -> list[float]:
    """One landmark column's *unmasked* values across `records`.

    Uses the module's own public `landmark_values`, so the expectation differs
    from the implementation in exactly the thing under test -- which entries
    are aggregated -- and not in the landmark-matching rule.
    """
    return [
        values[landmark_index]
        for values in (landmark_values(record, concept_id) for record in records)
        if values[landmark_index] is not None
    ]


# ---------------------------------------------------------------------------
# Property 18: bin assignment is total, ordered, and delegated
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(elapsed_weeks=_elapsed_weeks_value, increment=_elapsed_weeks_value)
@example(elapsed_weeks=0.0, increment=0.0)
@example(elapsed_weeks=7.999, increment=0.0)
@example(elapsed_weeks=8.0, increment=0.0)
@example(elapsed_weeks=175.999, increment=0.0)
@example(elapsed_weeks=176.0, increment=0.0)
@example(elapsed_weeks=1e6, increment=0.0)
@example(elapsed_weeks=175.999, increment=0.001)  # crosses into the open bin
def test_property_18_bin_index_is_total_ordered_and_open_at_the_end(
    elapsed_weeks: float, increment: float
) -> None:
    """**Correctness Property 18** (value quantifier), Requirement 1.2.

    For any finite non-negative elapsed-week value the index lies in 0..22,
    is non-decreasing in elapsed weeks, and equals 22 exactly at or beyond
    176 weeks -- however far beyond.
    """
    bin_index = survival_bin_index(elapsed_weeks)

    assert bin_index in range(SURVIVAL_BIN_COUNT)
    assert (bin_index == SURVIVAL_BIN_COUNT - 1) == (elapsed_weeks >= FINAL_BIN_START_WEEKS)
    if elapsed_weeks < FINAL_BIN_START_WEEKS:
        assert bin_index == math.floor(elapsed_weeks / BIN_WIDTH_WEEKS)

    assert survival_bin_index(elapsed_weeks + increment) >= bin_index


@_COHORT_SETTINGS
@given(records=_cohorts())
def test_property_18_bin_assignment_delegates_elapsed_weeks(
    records: list[PatientRecord],
) -> None:
    """**Correctness Property 18** (record quantifier), Requirement 1.3.

    The elapsed weeks behind every emitted bin index are exactly what
    `causal.baselines._elapsed_weeks` returns for that same record; this
    module re-derives nothing.
    """
    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")
    by_id = {record.patient_id: record for record in records}

    for row, patient_id in enumerate(batch.patient_ids):
        expected = survival_bin_index(_elapsed_weeks(by_id[patient_id]))
        assert int(batch.bin_index[row]) == expected

    assert int(batch.bin_index.min()) >= 0
    assert int(batch.bin_index.max()) <= SURVIVAL_BIN_COUNT - 1


# ---------------------------------------------------------------------------
# Property 19: batch structure and ordering are consistent and deterministic
# ---------------------------------------------------------------------------


@_COHORT_SETTINGS
@given(cohorts=_shuffled_cohorts())
def test_property_19_batch_structure_alignment_and_order_independence(
    cohorts: tuple[list[PatientRecord], list[PatientRecord]],
) -> None:
    """**Correctness Property 19**, Requirements 1.1, 1.6, 1.7.

    Six conjuncts over any non-empty cohort of unique records: N rows
    everywhere, frozen trailing widths, row `i` of every tensor referring to
    `patient_ids[i]`, reported `DatasetDimensions` equal to those trailing
    dimensions, ascending identifiers, and identical tensors from a shuffled
    input order.
    """
    records, shuffled = cohorts
    n = len(records)

    builder = PatientDatasetBuilder().fit(records)
    batch = builder.build(records, "dev")
    dimensions = batch.dimensions

    # N rows, and the frozen trailing widths, which the reported dimensions
    # must agree with (Requirement 1.7).
    assert len(batch.patient_ids) == n
    for field_name, dimension_name, imported_width in _TRAILING_WIDTHS:
        tensor = getattr(batch, field_name)
        assert tensor.shape == (n, imported_width), field_name
        assert getattr(dimensions, dimension_name) == imported_width, dimension_name
    for field_name in _ROW_VECTOR_FIELDS:
        assert getattr(batch, field_name).shape == (n,), field_name
    assert dimensions.survival_bin_count == SURVIVAL_BIN_COUNT
    assert set(int(value) for value in batch.bin_index) <= set(range(dimensions.survival_bin_count))

    # Ascending identifiers, and exactly the identifiers handed in
    # (Requirement 1.6).
    assert list(batch.patient_ids) == sorted(record.patient_id for record in records)

    # Row `i` of every tensor refers to `patient_ids[i]` (Requirement 1.1).
    by_id = {record.patient_id: record for record in records}
    for row, patient_id in enumerate(batch.patient_ids):
        record = by_id[patient_id]
        assert float(batch.treatment_arm[row]) == float(record.treatment_arm)
        assert float(batch.intervention[row, 0]) == float(record.treatment_arm)
        assert float(batch.protocol[row, 0]) == (1.0 if record.eligibility_result.eligible else 0.0)
        assert int(batch.bin_index[row]) == survival_bin_index(_elapsed_weeks(record))
        for concept_id, values, mask in (
            (CD4_LAB_CONCEPT_ID, batch.cd4, batch.cd4_mask),
            (CD8_LAB_CONCEPT_ID, batch.cd8, batch.cd8_mask),
        ):
            for landmark_index, value in enumerate(landmark_values(record, concept_id)):
                if value is None:
                    assert not bool(mask[row, landmark_index])
                    assert float(values[row, landmark_index]) == MISSING_PLACEHOLDER

    # Shuffled input order, identical output (Requirement 1.6).
    _assert_identical_tensors(batch, builder.build(shuffled, "dev"))
    # And repeating the same build is identical too, which is the same
    # requirement's determinism half.
    _assert_identical_tensors(batch, builder.build(records, "dev"))


# ---------------------------------------------------------------------------
# Property 20: preprocessing is fitted once, on development data only
# ---------------------------------------------------------------------------


@_COHORT_SETTINGS
@given(splits=_three_splits())
def test_property_20_preprocessing_is_fitted_once_on_development_only(
    splits: tuple[list[PatientRecord], list[PatientRecord], list[PatientRecord]],
) -> None:
    """**Correctness Property 20**, Requirements 1.4, 18.5.

    Three conjuncts: fitting twice yields identical parameters; building
    validation or test batches leaves them unchanged; and the parameters come
    from unmasked development entries only.
    """
    dev, val, test = splits

    # Fitting twice is identical -- both as a second `fit` on the same
    # builder and as an independent builder. Landmark parameters compare
    # directly; the baseline block's parameters are private, so they are
    # compared through the tensors they produce.
    first = PatientDatasetBuilder().fit(dev)
    second = PatientDatasetBuilder().fit(dev)
    refit = PatientDatasetBuilder().fit(dev).fit(dev)
    assert first.fitted_landmark_params == second.fitted_landmark_params
    assert first.fitted_landmark_params == refit.fitted_landmark_params
    _assert_identical_tensors(first.build(dev, "dev"), second.build(dev, "dev"))
    _assert_identical_tensors(first.build(dev, "dev"), refit.build(dev, "dev"))

    # Building the held-out splits changes nothing: same parameters, and the
    # development batch is bit-for-bit what it was before those builds.
    before_params = first.fitted_landmark_params
    dev_batch_before = first.build(dev, "dev")
    first.build(val, "val")
    first.build(test, "test")
    assert first.fitted_landmark_params == before_params
    _assert_identical_tensors(dev_batch_before, first.build(dev, "dev"))

    # Parameters are the mean/stdev of the *unmasked development* entries of
    # each column, and a column with no unmasked development entry is absent
    # rather than fabricated (Requirement 1.9).
    for concept_id, landmark_index in _LANDMARK_COLUMNS:
        observed = _observed_development_values(dev, concept_id, landmark_index)
        key = (concept_id, landmark_index)
        if not observed:
            assert key not in before_params
            continue
        fitted = before_params[key]
        assert fitted.mean == pytest.approx(statistics.fmean(observed), rel=1e-9)
        expected_stdev = statistics.pstdev(observed) if len(observed) > 1 else 0.0
        assert fitted.stdev == pytest.approx(expected_stdev, rel=1e-9, abs=1e-12)
