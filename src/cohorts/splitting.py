"""Patient-level stratified dev/val/test splitting (Requirements 6.2, 6.7;
`design.md` Stage 5, Section 3.4; `decisions.md` Section 9 step 3, Section 10).

`decisions.md` Section 9's frozen order of operations places the patient-level
split (step 3) strictly *after* the internal randomized reference is sealed
(step 2, task 6.1) and *before* any preprocessing is fit (step 4, task 6.3).
`design.md` Section 3.4 says this ordering must be enforced "structurally
rather than by convention." Task 6.1 enforces the *reference-seal-before-
split* half of that ordering only in the sense that `SealedReference` can
only be constructed from the pre-split `RandomizedCohort` type -- it does not
by itself stop a caller from splitting a cohort whose reference was never
sealed at all.

This module closes that remaining gap for the *split* side: `stratified_patient_split`
requires a genuine `cohorts.sealed_reference.SealedReference` instance as an
argument (raising `TypeError` for anything else, including a bare
`RandomizedCohort` or `None`), even though the splitting logic itself never
reads the sealed value. A caller cannot call this function without having
first constructed (and therefore already computed/sealed) the internal
reference for that cohort -- the reference's *existence*, not its unsealed
*value*, is what this function's signature demands. This mirrors how
`SealedReference.unseal()` requires a `LockedEvaluationToken` it does not
otherwise need to "use": the type-level dependency is the enforcement
mechanism, not a runtime read of the argument's contents.

Stratification
--------------

The stratum key for a patient is the tuple of values returned by
`_STRATUM_FIELD_EXTRACTORS` for each name in `strata` (default:
`treatment_arm`, `primary_event_indicator`, `prior_art_status`,
per `decisions.md` Section 10 -- "Stratified on: treatment arm, primary-event
indicator, prior-antiretroviral-treatment stratum."):

- `treatment_arm`: `PatientRecord.treatment_arm` (`0` or `1`).
- `primary_event_indicator`: `PatientRecord.endpoint_result.event_indicator`
  (`bool`), reusing the composite endpoint already computed at cohort
  construction (task 3.3/6) rather than recomputing it.
- `prior_art_status`: derived from `PatientRecord.timeline.static_covariates`,
  since `PatientRecord` has no dedicated `prior_art_status` field (prior
  antiretroviral exposure is recorded as a raw/normalized concept_id --
  `"prior_art_exposure_days"` in `data.loaders.actg175.BaselineCovariateRecord`
  and the hand-built-event test fixtures, or `"prior_antiretroviral_exposure_duration"`
  once `data.normalization.normalize_concept` has been applied). The first
  matching concept_id present in `static_covariates` is read and thresholded:
  `"experienced"` if the recorded exposure is `> 0`, `"naive"` if it is `<= 0`.
  A patient with *neither* concept_id present (or a non-numeric value) gets
  its own explicit stratum, `"unknown"`, rather than raising -- a missing
  covariate must not crash the split, and lumping "unknown" patients into
  either real bucket would misrepresent them.

Deterministic within-stratum splitting
---------------------------------------

Within each stratum (patients grouped by the composite key above, always
sorted by `patient_id` first -- never relying on `dict`/`set` iteration
order):

1. A stratum-specific seed is derived via
   `infra.seeding.derive_seed(derive_seed(seed, "split"), stratum_label)`,
   where `stratum_label` is a stable string rendering of the stratum key.
   Deriving a *per-stratum* sub-seed (rather than reusing one RNG across all
   strata) means adding, removing, or reordering strata never perturbs any
   other stratum's assignment.
2. The stratum's patients (already sorted by `patient_id`) are shuffled
   using a `random.Random(stratum_seed)` instance seeded from that derived
   seed -- deterministic and reproducible for a fixed `seed`, but not merely
   "first N in patient_id order" (which could otherwise introduce bias if
   patient IDs correlate with anything).
3. Target per-split counts are computed from `fractions` via the largest-
   remainder method (see `_largest_remainder_counts`): each split's count is
   `floor(fraction * stratum_size)`, and any leftover patients (from
   rounding) are distributed one at a time to the splits with the largest
   fractional remainder, ties broken by split order (dev, then val, then
   test). This is a standard, deterministic apportionment rule -- see
   module docstring note below on small strata.
4. The shuffled list is sliced into exactly those counts, in order
   (dev, val, test) -- a partition, not an overlapping sample: every patient
   in the stratum lands in exactly one split.

Every patient in the cohort belongs to exactly one stratum and is assigned to
exactly one split, so `len(dev) + len(val) + len(test) == len(cohort)` and no
`patient_id` appears in more than one of `dev`/`val`/`test` -- this holds
regardless of `fractions` or stratum sizes.

Small strata
------------

The largest-remainder rounding above guarantees the split counts within a
stratum always sum to the stratum's size, but for a very small stratum (e.g.
1-2 patients) it cannot hit the target fractions exactly -- a 1-patient
stratum, for instance, always places its one patient in the split with the
largest fraction (dev, under the default `(0.6, 0.2, 0.2)`), and a 2-patient
stratum can only ever realize a `{1,1,0}`-style split, never a fractional
one. Exact proportions are therefore only guaranteed asymptotically, as
per-stratum sizes grow; callers evaluating "did this stratum hit 60/20/20"
on a handful of patients should expect visible rounding noise, not an exact
match.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from math import floor

from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from cohorts.sealed_reference import SealedReference
from infra.seeding import derive_seed

#: `concept_id`s checked, in order, when deriving `prior_art_status` from
#: `PatientRecord.timeline.static_covariates`. The first one present (with a
#: non-`None`, numeric value) wins. See module docstring.
_PRIOR_ART_CONCEPT_IDS: tuple[str, ...] = (
    "prior_art_exposure_days",
    "prior_antiretroviral_exposure_duration",
)

#: Stratum value used when neither `_PRIOR_ART_CONCEPT_IDS` entry is present
#: (or present with a non-numeric value) for a patient -- its own explicit
#: bucket, never silently merged into "naive" or "experienced".
_PRIOR_ART_UNKNOWN = "unknown"


def _prior_art_status(patient: PatientRecord) -> str:
    """Derive `"experienced"` / `"naive"` / `"unknown"` for one patient.

    See module docstring ("Stratification") for the full rule.
    """
    static_covariates = patient.timeline.static_covariates
    for concept_id in _PRIOR_ART_CONCEPT_IDS:
        if concept_id not in static_covariates:
            continue
        raw_value = static_covariates[concept_id]
        if raw_value is None:
            continue
        try:
            exposure_days = float(raw_value)
        except (TypeError, ValueError):
            continue
        return "experienced" if exposure_days > 0 else "naive"
    return _PRIOR_ART_UNKNOWN


#: Registry of stratification-field extractors, keyed by the field name a
#: caller may pass in `strata`. Adding a new stratifiable field means adding
#: one entry here -- `stratified_patient_split` itself never needs to change.
_STRATUM_FIELD_EXTRACTORS: dict[str, Callable[[PatientRecord], Hashable]] = {
    "treatment_arm": lambda patient: patient.treatment_arm,
    "primary_event_indicator": lambda patient: patient.endpoint_result.event_indicator,
    "prior_art_status": _prior_art_status,
}

#: The frozen default stratification fields (`decisions.md` Section 10).
DEFAULT_STRATA: tuple[str, ...] = (
    "treatment_arm",
    "primary_event_indicator",
    "prior_art_status",
)

#: Tolerance used when validating that `fractions` sums to `1.0`, to absorb
#: ordinary floating-point representation error (e.g. `0.6 + 0.2 + 0.2` is
#: not bit-exact `1.0` in IEEE-754 double precision).
_FRACTION_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PatientSet:
    """An immutable, ordered collection of `PatientRecord`s -- one dev/val/test
    split's worth of patients (`design.md` Section 3.4).

    Always constructed with `patients` sorted by `patient_id` by this
    module's callers, so two `PatientSet`s built from the same membership are
    equal (`==`) regardless of construction order.
    """

    patients: tuple[PatientRecord, ...]

    def __len__(self) -> int:
        return len(self.patients)

    def __iter__(self):
        return iter(self.patients)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        """`patient_id`s in the same order as `patients`."""
        return tuple(patient.patient_id for patient in self.patients)


@dataclass(frozen=True)
class SplitCohorts:
    """Stage-5 output: the cohort partitioned into development, validation,
    and held-out test `PatientSet`s (`design.md` Section 3.4).

    Constructed only by `stratified_patient_split` below.
    """

    dev: PatientSet
    val: PatientSet
    test: PatientSet


def _validate_fractions(fractions: Sequence[float]) -> None:
    if len(fractions) != 3:
        raise ValueError(
            f"fractions must have exactly 3 entries (dev, val, test), got {len(fractions)}: "
            f"{fractions!r}"
        )
    if any(fraction < 0 for fraction in fractions):
        raise ValueError(f"fractions must all be non-negative, got {fractions!r}")
    total = sum(fractions)
    if abs(total - 1.0) > _FRACTION_SUM_TOLERANCE:
        raise ValueError(
            f"fractions must sum to 1.0 (within {_FRACTION_SUM_TOLERANCE}), "
            f"got {fractions!r} summing to {total}"
        )


def _validate_strata(strata: Sequence[str]) -> None:
    unknown = [name for name in strata if name not in _STRATUM_FIELD_EXTRACTORS]
    if unknown:
        raise ValueError(
            f"Unknown stratification field(s) {unknown!r}; known fields are "
            f"{sorted(_STRATUM_FIELD_EXTRACTORS)!r}."
        )


def _stratum_key(patient: PatientRecord, strata: Sequence[str]) -> tuple[Hashable, ...]:
    return tuple(_STRATUM_FIELD_EXTRACTORS[field_name](patient) for field_name in strata)


def _largest_remainder_counts(
    stratum_size: int, fractions: Sequence[float]
) -> tuple[int, int, int]:
    """Deterministically apportion `stratum_size` patients into `(dev, val, test)`
    counts that sum exactly to `stratum_size`, following `fractions` as closely
    as integer counts allow.

    Uses the largest-remainder (Hare-quota) apportionment method: each split
    gets `floor(fraction * stratum_size)` first, then any remaining, unassigned
    patients (from truncation) are handed out one at a time to the splits
    with the largest fractional remainder, ties broken by split order
    (dev, then val, then test). This is a standard, deterministic rounding
    rule -- not a source of nondeterminism -- and is exact whenever
    `fraction * stratum_size` is already an integer for every split.
    """
    raw_counts = [fraction * stratum_size for fraction in fractions]
    floor_counts = [floor(raw) for raw in raw_counts]
    remainder = stratum_size - sum(floor_counts)

    remainders_by_index = sorted(
        range(len(fractions)),
        key=lambda index: (-(raw_counts[index] - floor_counts[index]), index),
    )

    counts = list(floor_counts)
    for index in remainders_by_index[:remainder]:
        counts[index] += 1

    dev_count, val_count, test_count = counts
    return dev_count, val_count, test_count


def stratified_patient_split(
    cohort: RandomizedCohort,
    sealed_reference: SealedReference,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    strata: Sequence[str] | None = None,
    seed: int = 0,
) -> SplitCohorts:
    """Partition `cohort` into stratified development/validation/test splits.

    Requires `sealed_reference` to be a genuine
    `cohorts.sealed_reference.SealedReference` -- not because this function
    reads its value, but because its mere existence is proof that the
    internal randomized reference for this cohort has already been computed
    and sealed (task 6.1), which is what `decisions.md` Section 9's frozen
    order of operations ("split strictly after the reference is sealed")
    requires. See module docstring.

    Args:
        cohort: the complete eligible arm-0/arm-1 `RandomizedCohort`
            (Stage 3 output) to split.
        sealed_reference: the `SealedReference` already constructed from
            `cohort` (task 6.1). Required for its type, not its value --
            calling this function without one raises `TypeError`.
        fractions: `(dev, val, test)` target proportions. Must be
            non-negative and sum to `1.0` (within floating-point tolerance).
            Defaults to the frozen `(0.6, 0.2, 0.2)` (`decisions.md`
            Section 10).
        strata: names of the stratification fields to use, drawn from
            `treatment_arm`, `primary_event_indicator`, `prior_art_status`.
            Defaults to `DEFAULT_STRATA` (all three, `decisions.md`
            Section 10). An empty sequence is allowed (a single, whole-
            cohort "stratum").
        seed: the master seed. A stratum-specific sub-seed is derived from
            it (via `infra.seeding.derive_seed`) for each stratum's
            deterministic shuffle, so the same `seed` always reproduces the
            same split, and a different `seed` reliably produces a
            different one.

    Returns:
        A `SplitCohorts` whose `dev`, `val`, `test` `PatientSet`s partition
        `cohort` -- no `patient_id` appears in more than one of them, and
        their combined size equals `len(cohort)`.

    Raises:
        TypeError: if `sealed_reference` is not a `SealedReference` instance.
        ValueError: if `fractions` does not have exactly 3 non-negative
            entries summing to `1.0`, or if `strata` contains an unknown
            field name.
    """
    if not isinstance(sealed_reference, SealedReference):
        raise TypeError(
            "stratified_patient_split requires a genuine "
            "cohorts.sealed_reference.SealedReference instance as "
            "sealed_reference (proof the internal randomized reference has "
            "already been sealed for this cohort, per decisions.md Section 9's "
            f"frozen order of operations); got {type(sealed_reference).__name__!r}."
        )

    resolved_strata: tuple[str, ...] = tuple(strata) if strata is not None else DEFAULT_STRATA
    _validate_strata(resolved_strata)
    _validate_fractions(fractions)

    split_seed = derive_seed(seed, "split")

    groups: dict[tuple[Hashable, ...], list[PatientRecord]] = {}
    for patient in sorted(cohort.patients, key=lambda record: record.patient_id):
        key = _stratum_key(patient, resolved_strata)
        groups.setdefault(key, []).append(patient)

    dev_patients: list[PatientRecord] = []
    val_patients: list[PatientRecord] = []
    test_patients: list[PatientRecord] = []

    # Iterate strata in a stable, seed-independent order (by their string
    # rendering) so the *order in which strata are processed* never affects
    # the result -- each stratum's assignment depends only on its own
    # derived sub-seed, not on iteration order over `groups`.
    for stratum_key in sorted(groups, key=repr):
        stratum_patients = groups[stratum_key]  # already patient_id-sorted (see loop above)
        stratum_label = "|".join(str(part) for part in stratum_key)
        stratum_seed = derive_seed(split_seed, stratum_label)

        shuffled = list(stratum_patients)
        random.Random(stratum_seed).shuffle(shuffled)

        dev_count, val_count, _test_count = _largest_remainder_counts(
            len(shuffled), fractions
        )
        dev_patients.extend(shuffled[:dev_count])
        val_patients.extend(shuffled[dev_count : dev_count + val_count])
        test_patients.extend(shuffled[dev_count + val_count :])

    dev_patients.sort(key=lambda record: record.patient_id)
    val_patients.sort(key=lambda record: record.patient_id)
    test_patients.sort(key=lambda record: record.patient_id)

    return SplitCohorts(
        dev=PatientSet(tuple(dev_patients)),
        val=PatientSet(tuple(val_patients)),
        test=PatientSet(tuple(test_patients)),
    )
