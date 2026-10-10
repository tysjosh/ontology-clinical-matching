"""Patient_Dataset_Builder: `PatientRecord` -> model-ready torch tensors.

This is the single conversion point between the cohort pipeline's
`cohorts.cohort_construction.PatientRecord` and the PyTorch models of
`models.torch_sequence_gru` / `models.torch_intervention_aware`
(`.kiro/specs/neural-experiment-pipeline/design.md` Section 3.1,
Requirement 1). Nothing else in the repository converts patient records to
tensors -- the pure-Python prototypes accept plain Python lists assembled
ad hoc in their tests.

Torch dependency scope (read before extending this module)
-------------------------------------------------------------------------

`torch` is a **leaf** dependency confined to `src/models/torch_*.py` and
this module (`design.md` Section 2.4). Consequences, both enforced by
`tests/unit/test_torch_isolation.py`:

- `src/models/__init__.py` must not import this module, so importing
  anything else under `models.` stays torch-free;
- nothing under `src/causal/`, `src/cohorts/`, `src/evaluation/`,
  `src/witnesses/`, `src/infra/`, or `scripts/` may import this module.

Tests for this module therefore live in `tests/neural/` (see that
package's `__init__.py`), behind `pytest.importorskip("torch")`.

Every semantic decision is delegated, never re-derived
-------------------------------------------------------------------------

This module owns tensor *layout*. It owns no clinical or statistical
definition. Each one is imported from the module that already defines and
tests it:

- **elapsed weeks** (bin assignment): `causal.baselines._elapsed_weeks`,
  the same treatment-assignment-to-endpoint elapsed-weeks definition every
  conventional estimator uses (Requirement 1.3). This module never
  subtracts an endpoint timestamp itself.
- **baseline covariate ordering**: `causal.propensity.COVARIATE_FIELDS`,
  with per-patient extraction and development-fold-safe standardization by
  that module's own `_fit_standardization` / `_feature_vector` -- so the
  baseline tensor's columns are, in order and in value, exactly the
  features the propensity models and `adjusted_cox` see.
- **landmark timepoints**: `models.sequence_baseline.LANDMARK_WEEKS`
  (`(0, 20, 96)`), with the CD4/CD8 `concept_id`s from that same module.
- **channel widths**: `models.intervention_aware.STATE_INPUT_DIM`
  (4, and equal to `len(BASELINE_STATE_INIT_FIELDS)`),
  `INTERVENTION_INPUT_DIM` (3), `PROTOCOL_INPUT_DIM` (2), and the
  per-channel feature conventions documented in that module's docstring.
- **baseline-state-init vector**:
  `models.sequence_baseline.extract_baseline_vector`, z-scored per column
  with development-fitted parameters (raw clinical scales saturate the
  models' `tanh` input projections; see `_fit_state_init_standardization`).

The frozen values this module carries
-------------------------------------------------------------------------

`decisions.md` Section 25 ("Neural-training implementation freeze") fixes
the discrete-time survival discretization: 8-week half-open bins
`[0,8), [8,16), ..., [168,176)` plus one final open bin `[176, inf)` --
**23 bins total**. `SURVIVAL_BIN_COUNT`, `BIN_WIDTH_WEEKS`, and
`FINAL_BIN_START_WEEKS` below are that value read from that document, not
re-derived here; the module-level consistency assertion at the end of the
constants block fails loudly if the three ever stop agreeing.
The same section's "Hidden width" entry (`hidden_size = 16`, shared by the
sequence-GRU baseline and the intervention-aware model's `GRUTransition`)
is `FROZEN_HIDDEN_SIZE` below. It is canonical here rather than in either
torch model because it is *one* frozen value used by both, and this module
is the shared torch-layer home both of them already import.

`MISSING_PLACEHOLDER = 0.0` is this spec's own choice (`design.md` Section
4.1): one fixed finite fill for every missing CD4/CD8 entry, in every
build. Zero because it is the additive identity of the mask-multiplied
Huber state term, which makes the "masked-missing entries contribute
exactly zero gradient" guarantee exact rather than approximate
(Requirement 1.5).

Standardization and the fit/build split
-------------------------------------------------------------------------

`fit(dev_records)` computes standardization parameters from
Development_Split rows only -- for CD4/CD8 landmarks, only from entries
whose mask is `observed`; for the baseline-state-init vector, per column.
`build(records, split_name)` never fits: it applies the frozen
parameters unchanged to development, validation, and test rows alike
(Requirement 1.4). `fit` is deterministic, so refitting the same records
reproduces the same parameters bit for bit.

A CD4/CD8 landmark column with no observed development entry at all --
which is the real ACTG175 v3.2 situation for week-96 CD8 -- gets **no**
fitted parameter, an all-`False` mask in every split, and the placeholder
fill, and raises nothing (Requirement 1.9).

Missingness convention: masks are `True = observed`, `False = missing`, and
every `False` position holds exactly `MISSING_PLACEHOLDER`. A landmark
entry counts as missing when the observation is absent, carries
`MissingnessState.MISSING`, or has a non-numeric or non-finite value. No
emitted tensor contains a NaN or an infinity.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import torch
from torch import Tensor

from causal.baselines import _elapsed_weeks
from causal.propensity import (
    COVARIATE_FIELDS,
    _feature_vector,
    _fit_standardization,
    _StandardizationParams,
)
from cohorts.cohort_construction import PatientRecord
from data.event_model import ClinicalEvent, MissingnessState
from models.intervention_aware import (
    INTERVENTION_INPUT_DIM,
    PROTOCOL_INPUT_DIM,
    STATE_INPUT_DIM,
)
from models.sequence_baseline import (
    BASELINE_STATE_INIT_FIELDS,
    CD4_LAB_CONCEPT_ID,
    CD8_LAB_CONCEPT_ID,
    LANDMARK_WEEKS,
    extract_baseline_vector,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of landmark timesteps per patient -- `len(LANDMARK_WEEKS)`, not an
#: independent value (`decisions.md` Section 5: baseline, week 20, week 96).
LANDMARK_COUNT: int = 3

#: Discrete-time survival bins (`decisions.md` Section 25).
SURVIVAL_BIN_COUNT: int = 23

#: GRU hidden-state width, frozen in `decisions.md` Section 25 ("Hidden
#: width: `hidden_size = 16` for both the sequence-GRU baseline and the
#: intervention-aware model's `GRUTransition`"). Requirement 2.4.
#:
#: This lives here, in the module both torch models already import, because
#: the frozen decision is that the two models share *one* width: the
#: treatment-as-feature/explicit-channel ablation contrast (`decisions.md`
#: Section 17 items 2-3) has to isolate the representational difference, not
#: a capacity difference. `torch_sequence_gru.FROZEN_HIDDEN_SIZE` and
#: `torch_intervention_aware.HIDDEN_SIZE` are aliases of this name, not
#: independent literals, so the two models cannot silently diverge to
#: different capacities with every test still passing.
FROZEN_HIDDEN_SIZE: int = 16

#: Width of each closed survival bin, in weeks (`decisions.md` Section 25).
BIN_WIDTH_WEEKS: float = 8.0

#: Start of the final, open bin `[176, inf)` (`decisions.md` Section 25).
#: Any elapsed time at or beyond this receives the last bin index.
FINAL_BIN_START_WEEKS: float = 176.0

#: The single fixed finite value filling every masked-missing CD4/CD8 entry,
#: identically in every build (`design.md` Section 4.1, Requirement 1.5).
MISSING_PLACEHOLDER: float = 0.0

#: Split names `build` accepts, matching `design.md` Section 4.1.
SPLIT_NAMES: tuple[str, ...] = ("dev", "val", "test")

#: `dtype` of every float tensor in a `TensorBatch` (`design.md` Section 4.1).
FLOAT_DTYPE: torch.dtype = torch.float32

#: Static-covariate `concept_id`s consulted for the intervention channel's
#: `dose` / `adherence` columns. Absent for ACTG175, which does not record
#: either -- `models.intervention_aware`'s documented `0.0` default then
#: applies (see that module's `InterventionEncoder` convention).
DOSE_FIELD: str = "dose"
ADHERENCE_FIELD: str = "adherence"

# The three frozen survival-discretization values are three views of one
# decision. Assert they agree at import rather than letting a future edit to
# one of them silently produce a 24th bin or a shifted final bin.
if LANDMARK_COUNT != len(LANDMARK_WEEKS):  # pragma: no cover - import-time guard
    raise AssertionError(
        f"LANDMARK_COUNT={LANDMARK_COUNT} disagrees with "
        f"len(sequence_baseline.LANDMARK_WEEKS)={len(LANDMARK_WEEKS)}"
    )
if STATE_INPUT_DIM != len(BASELINE_STATE_INIT_FIELDS):  # pragma: no cover
    raise AssertionError(
        f"intervention_aware.STATE_INPUT_DIM={STATE_INPUT_DIM} disagrees with "
        f"len(sequence_baseline.BASELINE_STATE_INIT_FIELDS)="
        f"{len(BASELINE_STATE_INIT_FIELDS)}"
    )
if (SURVIVAL_BIN_COUNT - 1) * BIN_WIDTH_WEEKS != FINAL_BIN_START_WEEKS:  # pragma: no cover
    raise AssertionError(
        f"decisions.md Section 25's bin definition is inconsistent here: "
        f"({SURVIVAL_BIN_COUNT} - 1) * {BIN_WIDTH_WEEKS} != {FINAL_BIN_START_WEEKS}"
    )


class DatasetBuildError(Exception):
    """Raised when a batch cannot be built from the records handed in.

    Every raise site rejects the input *before* any tensor is constructed,
    so a caller never receives a partially populated `TensorBatch`
    (Requirements 1.8, 1.10). The message always names the reason and, where
    a single record is at fault, that record's `patient_id`.
    """


# ---------------------------------------------------------------------------
# Survival bin assignment (Requirement 1.2)
# ---------------------------------------------------------------------------


def survival_bin_index(elapsed_weeks: float) -> int:
    """The discrete-hazard bin index for `elapsed_weeks`.

    `decisions.md` Section 25's 23 bins: `[0,8), [8,16), ..., [168,176)`
    are indices 0-21, and `[176, inf)` is index 22. So the returned index is
    always in `0..22` inclusive, is non-decreasing in `elapsed_weeks`, and
    is exactly `SURVIVAL_BIN_COUNT - 1` for every elapsed time at or beyond
    `FINAL_BIN_START_WEEKS` -- however far beyond (Requirement 1.2).

    Args:
        elapsed_weeks: weeks since treatment assignment, as returned by
            `causal.baselines._elapsed_weeks`. Must be finite and
            non-negative.

    Returns:
        The bin index, in `range(SURVIVAL_BIN_COUNT)`.

    Raises:
        ValueError: if `elapsed_weeks` is not finite, or is negative. Both
            are input defects rather than bins: this function has no
            sensible index for them, and `PatientDatasetBuilder.build`
            rejects such records with a `DatasetBuildError` naming the
            patient before ever calling here (Requirement 1.10).
    """
    if not math.isfinite(elapsed_weeks):
        raise ValueError(f"elapsed_weeks must be finite, got {elapsed_weeks!r}")
    if elapsed_weeks < 0.0:
        raise ValueError(f"elapsed_weeks must be non-negative, got {elapsed_weeks!r}")
    if elapsed_weeks >= FINAL_BIN_START_WEEKS:
        return SURVIVAL_BIN_COUNT - 1
    return int(math.floor(elapsed_weeks / BIN_WIDTH_WEEKS))


def survival_bin_start_weeks(bin_index: int) -> float:
    """The elapsed-week start of `bin_index` (its half-open lower edge)."""
    if not 0 <= bin_index < SURVIVAL_BIN_COUNT:
        raise ValueError(
            f"bin_index must be in 0..{SURVIVAL_BIN_COUNT - 1}, got {bin_index!r}"
        )
    return bin_index * BIN_WIDTH_WEEKS


# ---------------------------------------------------------------------------
# Emitted data models (design.md Section 4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetDimensions:
    """Every trailing tensor width, as integer data (Requirement 1.7).

    `models.variants.build_model` reads these instead of hard-coding a
    width, so a layout change is an import or shape error rather than a
    silent divergence. Each field equals the corresponding trailing
    dimension of the tensors `PatientDatasetBuilder.build` emits.

    Attributes:
        baseline_covariate_width: columns of `TensorBatch.baseline`
            (`len(causal.propensity.COVARIATE_FIELDS)`).
        state_width: columns of `TensorBatch.state_init`
            (`intervention_aware.STATE_INPUT_DIM`).
        intervention_width: columns of `TensorBatch.intervention`
            (`intervention_aware.INTERVENTION_INPUT_DIM`).
        protocol_width: columns of `TensorBatch.protocol`
            (`intervention_aware.PROTOCOL_INPUT_DIM`).
        landmark_count: columns of the CD4/CD8 and mask tensors
            (`LANDMARK_COUNT`).
        survival_bin_count: size of the hazard target range, so
            `TensorBatch.bin_index` values lie in
            `0..survival_bin_count - 1` (`SURVIVAL_BIN_COUNT`).
    """

    baseline_covariate_width: int
    state_width: int
    intervention_width: int
    protocol_width: int
    landmark_count: int
    survival_bin_count: int


@dataclass(frozen=True, eq=False)
class TensorBatch:
    """One split's tensors, with row `i` referring to `patient_ids[i]`.

    Field names, shapes, and dtypes are exactly `design.md` Section 4.1 --
    tasks 9.x and 10.x consume them by name.

    `eq=False`: a generated `__eq__` would compare `Tensor` fields with
    `==`, which yields an elementwise tensor and then raises on the
    truthiness test. Compare batches field by field with `torch.equal`
    instead, which is what the ordering/determinism assertions want anyway.

    Attributes:
        split_name: one of `SPLIT_NAMES`.
        patient_ids: length `N`, ascending (Requirement 1.6).
        baseline: `(N, baseline_covariate_width)` float32, columns in
            `causal.propensity.COVARIATE_FIELDS` order.
        state_init: `(N, 4)` float32,
            `sequence_baseline.BASELINE_STATE_INIT_FIELDS` order, z-scored
            with development-fitted per-column parameters.
        cd4: `(N, 3)` float32, landmark weeks 0/20/96 in that order.
        cd8: `(N, 3)` float32, same landmark order.
        cd4_mask: `(N, 3)` bool, `True = observed`.
        cd8_mask: `(N, 3)` bool, `True = observed`.
        treatment_arm: `(N,)` float32, `0.0` or `1.0`.
        protocol: `(N, 2)` float32, `[is_eligible, is_time_zero_valid]`.
        intervention: `(N, 3)` float32,
            `[treatment_arm, dose_or_default, adherence_or_default]`.
        bin_index: `(N,)` int64, in `0..SURVIVAL_BIN_COUNT - 1`.
        censoring_mask: `(N,)` bool, `True` for patients whose
            `endpoint_result.censoring_indicator` is set *and* whose
            censoring time falls before the start of their assigned bin.
        event_indicator: `(N,)` bool, `endpoint_result.event_indicator`
            (`True` = composite event observed, `False` = censored). Added
            for task 10: the discrete-time survival NLL needs to know
            whether bin `bin_index[i]` holds an event or a censoring, and
            `censoring_mask` (a narrower condition) cannot tell it.
        event_time_weeks: `(N,)` float32, the same elapsed weeks
            `bin_index` was computed from (`causal.baselines._elapsed_weeks`).
            Added for task 10: validation concordance
            (`evaluation.metrics.concordance_index`) is computed on
            continuous event times, not on bin indices, so it does not
            collapse every within-bin pair into an uninformative tie.
        dimensions: the widths above, as data.
    """

    split_name: str
    patient_ids: tuple[str, ...]
    baseline: Tensor
    state_init: Tensor
    cd4: Tensor
    cd8: Tensor
    cd4_mask: Tensor
    cd8_mask: Tensor
    treatment_arm: Tensor
    protocol: Tensor
    intervention: Tensor
    bin_index: Tensor
    censoring_mask: Tensor
    event_indicator: Tensor
    event_time_weeks: Tensor
    dimensions: DatasetDimensions


@dataclass(frozen=True)
class LandmarkStandardization:
    """One CD4/CD8 landmark column's development-fitted `(mean, stdev)`.

    Only columns with at least one observed development entry get one; a
    fully missing column has no entry at all (Requirement 1.9).
    """

    mean: float
    stdev: float


# ---------------------------------------------------------------------------
# Per-patient extraction helpers
# ---------------------------------------------------------------------------


def _assignment_time(patient: PatientRecord) -> datetime | None:
    """Time zero: the earliest recorded treatment event, or `None`.

    Same definition `causal.baselines._elapsed_weeks` and
    `cohorts.cohort_construction._earliest_treatment_time` use.
    """
    if not patient.timeline.treatment_events:
        return None
    return min(event.observed_at for event in patient.timeline.treatment_events)


def _weeks_between(later: datetime, assignment: datetime) -> float:
    """Whole days between two timestamps, in weeks.

    Mirrors `causal.baselines._elapsed_weeks`' day-resolution convention
    (`.days / 7.0`) so timeline timestamps and endpoint timestamps are
    placed on one time scale. Used only for *censoring* events, which
    `causal.baselines` exposes no elapsed-time helper for; the endpoint's
    own elapsed weeks always come from `_elapsed_weeks` itself.
    """
    return (later - assignment).days / 7.0


def _numeric_observation_value(event: ClinicalEvent) -> float | None:
    """`event.value` as a finite float, or `None` if it is not usable.

    `None` covers all four ways a landmark entry can be missing: the value
    is absent, the event is flagged `MissingnessState.MISSING`, the value is
    non-numeric (`ClinicalEvent.value` admits `str`), or it is a NaN /
    infinity. Each becomes a `False` mask entry and a placeholder fill,
    which is how the emitted tensors stay free of non-finite values
    (Requirement 1.5).
    """
    if event.missingness is MissingnessState.MISSING:
        return None
    if event.value is None:
        return None
    try:
        value = float(event.value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def landmark_values(patient: PatientRecord, concept_id: str) -> tuple[float | None, ...]:
    """One patient's `LANDMARK_WEEKS`-ordered values for `concept_id`.

    Each element is the observed value at that landmark, or `None` when the
    landmark measurement is missing (see `_numeric_observation_value` for
    what counts as missing).

    Matching rule: a `timestamped_observations` event belongs to landmark
    week `w` when it carries `concept_id` and its `observed_at` is exactly
    `w` weeks (day resolution) after time zero -- which is how
    `data.loaders.actg175_adapter` writes landmark measurements
    (`ASSIGNMENT_TIME_ANCHOR + timedelta(weeks=w)`). No fuzzy window is
    applied: a widened match radius would be an unfrozen semantic choice
    about which measurement represents a landmark, and this module makes no
    such choice. If several events match one landmark, the first observed
    value in `timestamped_observations`' existing sort order wins, and the
    landmark counts as missing only when none of them carries a usable
    value.

    Exposed (not private) because tests and the neural runners need to build
    landmark inputs under exactly this rule rather than a re-guessed one.
    """
    assignment = _assignment_time(patient)
    if assignment is None:
        return tuple(None for _ in LANDMARK_WEEKS)

    per_landmark: list[float | None] = []
    for landmark_week in LANDMARK_WEEKS:
        found: float | None = None
        for event in patient.timeline.timestamped_observations:
            if event.concept_id != concept_id:
                continue
            if (event.observed_at - assignment).days != landmark_week * 7:
                continue
            value = _numeric_observation_value(event)
            if value is not None:
                found = value
                break
        per_landmark.append(found)
    return tuple(per_landmark)


def _intervention_features(patient: PatientRecord) -> list[float]:
    """`[treatment_arm, dose_or_default, adherence_or_default]`.

    The per-channel convention `models.intervention_aware`'s
    `InterventionEncoder` documents, with its documented `0.0` default for
    `dose`/`adherence`: ACTG175 records neither, and the primary analysis
    does not require them, so the channel carries treatment assignment plus
    two structural zeros rather than being narrowed -- the width has to stay
    `INTERVENTION_INPUT_DIM` for the capacity-matched placebo ablation to
    mean anything.
    """
    static_covariates = patient.timeline.static_covariates
    features = [float(patient.treatment_arm)]
    for field_name in (DOSE_FIELD, ADHERENCE_FIELD):
        raw_value = static_covariates.get(field_name)
        try:
            value = 0.0 if raw_value is None else float(raw_value)
        except (TypeError, ValueError):
            value = 0.0
        features.append(value if math.isfinite(value) else 0.0)
    return features


def _protocol_features(patient: PatientRecord) -> list[float]:
    """`[is_eligible, is_time_zero_valid]`, the placeholder protocol feature
    set `models.intervention_aware`'s `ProtocolEncoder` documents.

    `is_time_zero_valid` is `1.0` for every record `build` emits, because
    `build` rejects a record with no resolvable time zero before reaching
    here (Requirement 1.10). It is carried anyway rather than dropped: the
    column is part of the frozen `PROTOCOL_INPUT_DIM` width, and a richer
    protocol feature set (visit windows, deviation flags) replaces these
    two without changing the encoder interface.
    """
    is_eligible = 1.0 if patient.eligibility_result.eligible else 0.0
    is_time_zero_valid = 1.0 if _assignment_time(patient) is not None else 0.0
    return [is_eligible, is_time_zero_valid]


def _censored_before_bin_start(patient: PatientRecord, bin_index: int) -> bool:
    """Requirement 1.2's censoring mask entry for one patient.

    `True` only when both halves hold: the patient's
    `endpoint_result.censoring_indicator` is set, *and* their censoring time
    falls strictly before the start of their assigned bin.

    The censoring time is the earliest `timeline.censoring_events` entry.
    That is deliberately not `endpoint_result.event_time`: for a censored
    patient the endpoint time is `_last_observed_time`, the latest timestamp
    across *every* timeline bucket (see
    `cohorts.cohort_construction._last_observed_time`), so a patient
    censored at week 50 whose record still carries a week-96 lab observation
    has an endpoint time of week 96 and a censoring time of week 50. This
    mask is exactly the patients for whom those two diverge across a bin
    boundary. Where they coincide -- the real ACTG175 v3.2 case, where the
    censoring table and the outcome table agree -- no patient is marked,
    which is the correct answer, not a degenerate one.

    A censored patient with no censoring event at all is not marked: there
    is no censoring time to place before the bin start.
    """
    if not patient.endpoint_result.censoring_indicator:
        return False
    censoring_events = patient.timeline.censoring_events
    if not censoring_events:
        return False
    assignment = _assignment_time(patient)
    if assignment is None:
        return False
    censoring_weeks = _weeks_between(
        min(event.observed_at for event in censoring_events), assignment
    )
    return censoring_weeks < survival_bin_start_weeks(bin_index)


# ---------------------------------------------------------------------------
# Validation (Requirements 1.8, 1.10)
# ---------------------------------------------------------------------------


def _ordered_unique_records(records: Sequence[PatientRecord]) -> tuple[PatientRecord, ...]:
    """`records` sorted by ascending `patient_id`, rejecting empty/duplicate input.

    Ordering happens here, once, so every emitted tensor's row order and the
    returned identifier tuple come from the same sort and cannot drift apart
    (Requirement 1.6).
    """
    if not records:
        raise DatasetBuildError(
            "received an empty record list; a split must contain at least one "
            "PatientRecord (Requirement 1.8)."
        )
    seen: set[str] = set()
    for record in records:
        if record.patient_id in seen:
            raise DatasetBuildError(
                f"duplicate patient_id={record.patient_id!r} in the input records; "
                "each patient must appear exactly once in a split "
                "(Requirement 1.8)."
            )
        seen.add(record.patient_id)
    return tuple(sorted(records, key=lambda record: record.patient_id))


def _validated_elapsed_weeks(records: Iterable[PatientRecord]) -> dict[str, float]:
    """`{patient_id: elapsed weeks}` for every record, or raise.

    Runs over *all* records before `build` constructs anything, so a single
    bad event time yields an exception and no tensors rather than a
    partially populated batch (Requirement 1.10). Elapsed weeks come from
    `causal.baselines._elapsed_weeks`; the only work here is turning its
    rejections, and the values it cannot represent, into a
    `DatasetBuildError` that names the patient and the reason.
    """
    elapsed_by_id: dict[str, float] = {}
    for record in records:
        patient_id = record.patient_id
        if record.endpoint_result.event_time is None:
            raise DatasetBuildError(
                f"patient_id={patient_id!r}: endpoint_result.event_time is absent; "
                "cannot assign a survival bin (Requirement 1.10)."
            )
        try:
            elapsed = _elapsed_weeks(record)
        except (ValueError, TypeError) as error:
            raise DatasetBuildError(
                f"patient_id={patient_id!r}: elapsed weeks could not be computed "
                f"from endpoint_result.event_time ({error})."
            ) from error
        if not math.isfinite(elapsed):
            raise DatasetBuildError(
                f"patient_id={patient_id!r}: elapsed weeks is non-finite "
                f"({elapsed!r}); cannot assign a survival bin (Requirement 1.10)."
            )
        if elapsed < 0.0:
            raise DatasetBuildError(
                f"patient_id={patient_id!r}: endpoint_result.event_time precedes "
                f"treatment assignment, giving negative elapsed weeks ({elapsed!r}) "
                "(Requirement 1.10)."
            )
        elapsed_by_id[patient_id] = elapsed
    return elapsed_by_id


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


class PatientDatasetBuilder:
    """Converts `PatientRecord`s into `TensorBatch`es (Requirement 1).

    Usage is fit-once, build-many:

    ```python
    builder = PatientDatasetBuilder().fit(dev_records)
    dev_batch = builder.build(dev_records, "dev")
    val_batch = builder.build(val_records, "val")
    ```

    `fit` reads Development_Split rows only; `build` applies the frozen
    parameters and never refits, so validation and test rows cannot
    influence any transformation (Requirement 1.4, Requirement 18.5).
    """

    def __init__(self) -> None:
        self._baseline_params: _StandardizationParams | None = None
        self._landmark_params: dict[tuple[str, int], LandmarkStandardization] | None = None
        self._state_init_params: tuple[LandmarkStandardization, ...] | None = None

    # -- fitting ---------------------------------------------------------

    def fit(self, dev_records: Sequence[PatientRecord]) -> PatientDatasetBuilder:
        """Fit standardization parameters on development records only.

        Two groups of parameters:

        - the baseline covariate block, fitted by
          `causal.propensity._fit_standardization` so the baseline tensor's
          continuous columns are standardized exactly as the propensity
          models standardize theirs;
        - one `(mean, stdev)` per CD4/CD8 landmark column, over that
          column's *observed* development entries only -- masked-missing
          entries never enter a mean or a standard deviation
          (Requirement 1.4). A column with no observed development entry
          gets no parameter at all (Requirement 1.9);
        - one `(mean, stdev)` per `BASELINE_STATE_INIT_FIELDS` column of the
          baseline-state-init vector (see `_fit_state_init_standardization`).

        Deterministic, hence idempotent: refitting the same records
        reproduces identical parameters, and refitting replaces rather than
        accumulates.

        Args:
            dev_records: the Development_Split records. Validated the same
                way `build` validates, so a defect surfaces at fit time
                rather than at first build.

        Returns:
            `self`, for chaining.

        Raises:
            DatasetBuildError: on empty input, a duplicate `patient_id`, or
                an unusable event time.
        """
        ordered = _ordered_unique_records(dev_records)
        _validated_elapsed_weeks(ordered)

        baseline_params = _fit_standardization(ordered)
        landmark_params: dict[tuple[str, int], LandmarkStandardization] = {}
        for concept_id in (CD4_LAB_CONCEPT_ID, CD8_LAB_CONCEPT_ID):
            per_patient = [landmark_values(record, concept_id) for record in ordered]
            for landmark_index in range(LANDMARK_COUNT):
                observed = [
                    values[landmark_index]
                    for values in per_patient
                    if values[landmark_index] is not None
                ]
                if not observed:
                    # Fully missing column: no parameter is fitted at all,
                    # so `fitted_landmark_params` shows its absence rather
                    # than a fabricated (0.0, 0.0) entry (Requirement 1.9).
                    continue
                mean = statistics.fmean(observed)
                stdev = statistics.pstdev(observed) if len(observed) > 1 else 0.0
                landmark_params[(concept_id, landmark_index)] = LandmarkStandardization(
                    mean=mean, stdev=stdev
                )

        state_init_params = _fit_state_init_standardization(ordered)

        self._baseline_params = baseline_params
        self._landmark_params = landmark_params
        self._state_init_params = state_init_params
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether `fit` has run."""
        return (
            self._baseline_params is not None
            and self._landmark_params is not None
            and self._state_init_params is not None
        )

    @property
    def fitted_state_init_params(self) -> tuple[LandmarkStandardization, ...]:
        """The development-fitted `(mean, stdev)` per `BASELINE_STATE_INIT_FIELDS` column."""
        if self._state_init_params is None:
            raise DatasetBuildError(
                "fitted_state_init_params was read before fit(); call fit(dev_records) first."
            )
        return self._state_init_params

    @property
    def fitted_landmark_params(self) -> Mapping[tuple[str, int], LandmarkStandardization]:
        """The fitted `(concept_id, landmark_index) -> (mean, stdev)` map.

        A fully missing column is *absent* from this mapping, which is how
        "no preprocessing transformation was fitted for that column"
        (Requirement 1.9) is observable rather than asserted.
        """
        if self._landmark_params is None:
            raise DatasetBuildError(
                "fitted_landmark_params was read before fit(); call "
                "fit(dev_records) first."
            )
        return dict(self._landmark_params)

    @property
    def dimensions(self) -> DatasetDimensions:
        """The trailing widths of every emitted tensor (Requirement 1.7).

        Available before `fit`: every width is a frozen constant or an
        imported one, never a property of the data.
        """
        return DatasetDimensions(
            baseline_covariate_width=len(COVARIATE_FIELDS),
            state_width=STATE_INPUT_DIM,
            intervention_width=INTERVENTION_INPUT_DIM,
            protocol_width=PROTOCOL_INPUT_DIM,
            landmark_count=LANDMARK_COUNT,
            survival_bin_count=SURVIVAL_BIN_COUNT,
        )

    # -- building --------------------------------------------------------

    def build(self, records: Sequence[PatientRecord], split_name: str) -> TensorBatch:
        """Build one split's `TensorBatch`, applying frozen parameters.

        Never fits: the parameters `fit` computed on the Development_Split
        are applied unchanged here, whichever split `records` belongs to
        (Requirement 1.4). Rows are ordered by ascending `patient_id`, so
        the same records in any input order produce identical tensors
        (Requirement 1.6).

        Args:
            records: the split's records, in any order.
            split_name: one of `SPLIT_NAMES`.

        Returns:
            A `TensorBatch` whose row `i` refers to `patient_ids[i]`.

        Raises:
            DatasetBuildError: if `fit` has not run, if `split_name` is not
                a recognized split, on empty input or a duplicate
                `patient_id` (Requirement 1.8), or on an unusable event time
                (Requirement 1.10). Nothing partial is emitted in any case.
        """
        if (
            self._baseline_params is None
            or self._landmark_params is None
            or self._state_init_params is None
        ):
            raise DatasetBuildError(
                f"build(split_name={split_name!r}) was called before fit(); "
                "call fit(dev_records) on the Development_Split first. build() "
                "never fits parameters itself (Requirement 1.4)."
            )
        if split_name not in SPLIT_NAMES:
            raise DatasetBuildError(
                f"unrecognized split_name={split_name!r}; expected one of {SPLIT_NAMES!r}."
            )

        ordered = _ordered_unique_records(records)
        elapsed_by_id = _validated_elapsed_weeks(ordered)

        patient_ids: list[str] = []
        baseline_rows: list[list[float]] = []
        state_init_rows: list[list[float]] = []
        cd4_rows: list[list[float]] = []
        cd8_rows: list[list[float]] = []
        cd4_mask_rows: list[list[bool]] = []
        cd8_mask_rows: list[list[bool]] = []
        treatment_arm_values: list[float] = []
        protocol_rows: list[list[float]] = []
        intervention_rows: list[list[float]] = []
        bin_indices: list[int] = []
        censoring_flags: list[bool] = []
        event_flags: list[bool] = []
        event_times_weeks: list[float] = []

        for record in ordered:
            patient_ids.append(record.patient_id)
            baseline_rows.append(_sanitized(_feature_vector(record, self._baseline_params)))
            state_init_rows.append(_standardized_state_init(record, self._state_init_params))

            cd4_row, cd4_mask_row = self._standardized_landmarks(record, CD4_LAB_CONCEPT_ID)
            cd8_row, cd8_mask_row = self._standardized_landmarks(record, CD8_LAB_CONCEPT_ID)
            cd4_rows.append(cd4_row)
            cd8_rows.append(cd8_row)
            cd4_mask_rows.append(cd4_mask_row)
            cd8_mask_rows.append(cd8_mask_row)

            treatment_arm_values.append(float(record.treatment_arm))
            protocol_rows.append(_protocol_features(record))
            intervention_rows.append(_intervention_features(record))

            bin_index = survival_bin_index(elapsed_by_id[record.patient_id])
            bin_indices.append(bin_index)
            censoring_flags.append(_censored_before_bin_start(record, bin_index))
            event_flags.append(bool(record.endpoint_result.event_indicator))
            event_times_weeks.append(elapsed_by_id[record.patient_id])

        return TensorBatch(
            split_name=split_name,
            patient_ids=tuple(patient_ids),
            baseline=torch.tensor(baseline_rows, dtype=FLOAT_DTYPE),
            state_init=torch.tensor(state_init_rows, dtype=FLOAT_DTYPE),
            cd4=torch.tensor(cd4_rows, dtype=FLOAT_DTYPE),
            cd8=torch.tensor(cd8_rows, dtype=FLOAT_DTYPE),
            cd4_mask=torch.tensor(cd4_mask_rows, dtype=torch.bool),
            cd8_mask=torch.tensor(cd8_mask_rows, dtype=torch.bool),
            treatment_arm=torch.tensor(treatment_arm_values, dtype=FLOAT_DTYPE),
            protocol=torch.tensor(protocol_rows, dtype=FLOAT_DTYPE),
            intervention=torch.tensor(intervention_rows, dtype=FLOAT_DTYPE),
            bin_index=torch.tensor(bin_indices, dtype=torch.int64),
            censoring_mask=torch.tensor(censoring_flags, dtype=torch.bool),
            event_indicator=torch.tensor(event_flags, dtype=torch.bool),
            event_time_weeks=torch.tensor(event_times_weeks, dtype=FLOAT_DTYPE),
            dimensions=self.dimensions,
        )

    def _standardized_landmarks(
        self, record: PatientRecord, concept_id: str
    ) -> tuple[list[float], list[bool]]:
        """One patient's standardized landmark row and its observed-mask row.

        Observed entries are z-scored with the development-fitted parameters
        for that column; an entry is left at `MISSING_PLACEHOLDER` when it is
        missing, when the column has no fitted parameter (fully missing in
        development, Requirement 1.9), or when the fitted standard deviation
        is zero -- the last mirroring
        `causal.propensity._standardized_value`'s documented "standardizes to
        0.0 rather than dividing by zero" convention. In every one of those
        cases the mask entry is `False`, so no fabricated value is ever
        presented as observed.
        """
        assert self._landmark_params is not None  # guarded by build()
        values = landmark_values(record, concept_id)
        row: list[float] = []
        mask_row: list[bool] = []
        for landmark_index, value in enumerate(values):
            params = self._landmark_params.get((concept_id, landmark_index))
            if value is None or params is None:
                row.append(MISSING_PLACEHOLDER)
                mask_row.append(False)
                continue
            standardized = (
                (value - params.mean) / params.stdev if params.stdev > 0.0 else 0.0
            )
            if not math.isfinite(standardized):  # pragma: no cover - defensive
                row.append(MISSING_PLACEHOLDER)
                mask_row.append(False)
                continue
            row.append(standardized)
            mask_row.append(True)
        return row, mask_row


def _sanitized(values: Sequence[float]) -> list[float]:
    """Replace any non-finite entry with `MISSING_PLACEHOLDER`.

    Applied to the baseline covariate block and the baseline-state-init
    vector. Both sources already default absent and non-numeric fields to
    `0.0` (`causal.propensity._standardized_value`,
    `sequence_baseline.extract_baseline_vector`), but neither screens a
    stored NaN or infinity. "No non-finite values" is a property of the
    *emitted tensors*, so it is enforced here at the emission point rather
    than assumed of upstream extraction. Note this is the same `0.0` those
    two modules already use for an unusable covariate -- not a new sentinel.
    """
    return [value if math.isfinite(value) else MISSING_PLACEHOLDER for value in values]


def _fit_state_init_standardization(
    dev_records: Sequence[PatientRecord],
) -> tuple[LandmarkStandardization, ...]:
    """Development-fitted `(mean, stdev)` per baseline-state-init column.

    `extract_baseline_vector` returns raw clinical scales (baseline CD4 in
    the hundreds, CD8 near a thousand, Karnofsky near 90, age in years).
    Fed unscaled into `tanh(Linear(state_init))`, those saturate the
    activation to exactly `+/-1.0` in float32, so the intervention-aware
    model's state encoder receives an exactly-zero gradient and cannot
    learn (found by task 10.3's gradient-flow test). The columns are
    therefore z-scored with parameters fitted here, on Development_Split
    rows only, and applied unchanged by `build` to every split
    (Requirement 1.4; `decisions.md` Section 10 step 4 "fit all
    preprocessing transformations ... on the development split only").

    Fitted over the vectors exactly as they would otherwise be emitted
    (after `_sanitized`), so what is standardized is what the model used to
    see. Population standard deviation, as the landmark columns use.
    """
    rows = [_sanitized(extract_baseline_vector(record)) for record in dev_records]
    params: list[LandmarkStandardization] = []
    for column_index in range(len(BASELINE_STATE_INIT_FIELDS)):
        column = [row[column_index] for row in rows]
        mean = statistics.fmean(column)
        stdev = statistics.pstdev(column) if len(column) > 1 else 0.0
        params.append(LandmarkStandardization(mean=mean, stdev=stdev))
    return tuple(params)


def _standardized_state_init(
    record: PatientRecord, params: tuple[LandmarkStandardization, ...] | None
) -> list[float]:
    """One patient's z-scored baseline-state-init vector.

    A zero-stdev column standardizes to `0.0` rather than dividing by zero,
    mirroring `causal.propensity._standardized_value` and
    `_standardized_landmarks`.
    """
    assert params is not None  # guarded by build()
    raw = _sanitized(extract_baseline_vector(record))
    standardized = [
        (value - column.mean) / column.stdev if column.stdev > 0.0 else 0.0
        for value, column in zip(raw, params, strict=True)
    ]
    return _sanitized(standardized)
