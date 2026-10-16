"""The frozen order-of-operations pipeline (`design.md` Section 3.4;
`decisions.md` Section 9's frozen order of operations, steps 1-8).

This module is the single place that chains, in the mandatory order:

    1. build_randomized_cohort           (construct the complete eligible cohort)
    2. SealedReference(cohort, ...)      (seal the internal randomized reference)
    3. stratified_patient_split(...)     (60/20/20 dev/val/test split)
    4. FittedPreprocessing().fit(...)    (fit standardization on dev only)
    5-6-7-8. observationalize(dev/val/test independently, distinct seeds)

Every earlier module in `cohorts/` already enforces its own piece of this
order structurally (e.g. `SealedReference` only accepts a pre-split
`RandomizedCohort`; `stratified_patient_split` requires a genuine
`SealedReference`; `FittedPreprocessing.fit` requires a `DevelopmentSet`).
This module does not re-implement any of those guarantees -- it simply
calls each step's function in the only order those type constraints permit,
so there is exactly one place a reader (or the go/no-go checklist) needs to
look to confirm the full Section 9 order is followed end to end.

Per-split seeds (`decisions.md` Section 9 step 7)
----------------------------------------------------

Exactly one `master_seed` is accepted. Every derived seed --  the split
seed and each of the three per-split observationalization seeds -- comes
from `infra.seeding.derive_seed(master_seed, label)`, with a distinct
`label` per use (`"split"`, `"dev"`, `"val"`, `"test"`), so no
random-number stream is ever reused across the split step or across
dev/val/test observationalization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cohorts.cohort_construction import RandomizedCohort, build_randomized_cohort
from cohorts.preprocessing import DEFAULT_CONTINUOUS_FIELDS, DevelopmentSet, FittedPreprocessing
from cohorts.sealed_reference import ReferenceEstimator, SealedReference, placeholder_estimator
from cohorts.selection import ObservationalizedSplit, SelectionSeverity, observationalize
from cohorts.splitting import SplitCohorts, stratified_patient_split
from data.timelines import PatientTimeline as DataPatientTimeline
from infra.seeding import derive_seed
from protocols.endpoints import EndpointResult
from protocols.schema import TargetTrialSpec


@dataclass(frozen=True)
class ObservationalizationPipelineResult:
    """The complete, bundled output of running the frozen Section-9 pipeline.

    Attributes:
        sealed_reference: the sealed internal randomized reference (Stage
            4), computed once over the complete pre-split cohort.
        split_cohorts: the dev/val/test patient-level split (Stage 5).
        fitted_preprocessing: the development-fit standardization
            parameters (Stage 6), applied unchanged to every split when
            computing observationalization risk scores.
        dev: the observationalized development split (Stage 7).
        val: the observationalized validation split (Stage 7).
        test: the observationalized test split (Stage 7).
        master_seed: the top-level seed every derived seed in this result
            traces back to (`infra.seeding.derive_seed`), recorded for
            provenance/manifest purposes.
    """

    sealed_reference: SealedReference
    split_cohorts: SplitCohorts
    fitted_preprocessing: FittedPreprocessing
    dev: ObservationalizedSplit
    val: ObservationalizedSplit
    test: ObservationalizedSplit
    master_seed: int


def run_observationalization_pipeline(
    raw_timelines: Sequence[DataPatientTimeline] | Mapping[str, DataPatientTimeline],
    treatment_arms: Mapping[str, int],
    spec: TargetTrialSpec,
    master_seed: int,
    *,
    reference_estimator: ReferenceEstimator = placeholder_estimator,
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    preprocessing_fields: tuple[str, ...] | None = None,
    precomputed_endpoint_results: Mapping[str, EndpointResult] | None = None,
    selection_severity: SelectionSeverity = "primary",
) -> ObservationalizationPipelineResult:
    """Run the complete frozen Section-9 order-of-operations pipeline.

    Steps, in the mandatory order (`decisions.md` Section 9):

    1. `build_randomized_cohort(raw_timelines, treatment_arms, spec)` --
       construct the complete eligible arm-0/arm-1 randomized cohort.
    2. `SealedReference(cohort, estimator=reference_estimator)` -- compute
       and seal the internal randomized reference on that complete cohort,
       before any splitting or selection.
    3. `stratified_patient_split(cohort, sealed_reference, fractions=
       split_fractions, seed=derive_seed(master_seed, "split"))` -- the
       60/20/20 (by default) patient-level dev/val/test split.
    4. `FittedPreprocessing(fields=preprocessing_fields).fit(
       DevelopmentSet(split_cohorts.dev))` -- fit standardization on the
       development split only.
    5-6-7-8. `observationalize(split, fitted_preprocessing,
       seed=derive_seed(master_seed, split_name))` for each of dev, val,
       test independently, using the SAME development-fitted
       `fitted_preprocessing` and the SAME frozen selection formula, but a
       DISTINCT, split-specific seed for each split's Bernoulli draws.

    Args:
        raw_timelines: candidate patients' `data.timelines.PatientTimeline`s
            (see `cohort_construction.build_randomized_cohort`).
        treatment_arms: `{patient_id: arm}` raw treatment-arm assignments.
        spec: the target-trial specification.
        master_seed: the single top-level seed every derived seed in this
            pipeline traces back to.
        reference_estimator: the `RandomizedCohort -> InternalReference`
            callable passed to `SealedReference`. Defaults to
            `cohorts.sealed_reference.placeholder_estimator` (no real
            causal-estimator machinery is wired in until task 7.x).
        split_fractions: `(dev, val, test)` target proportions, forwarded
            to `stratified_patient_split`. Defaults to the frozen
            `(0.6, 0.2, 0.2)` (`decisions.md` Section 10).
        preprocessing_fields: continuous covariate fields to standardize,
            forwarded to `FittedPreprocessing`. Defaults to
            `cohorts.preprocessing.DEFAULT_CONTINUOUS_FIELDS` (CD4,
            Karnofsky, Age -- exactly the three risk-score inputs
            `FittedPreprocessing` standardizes) when `None`.
        precomputed_endpoint_results: forwarded unchanged to
            `build_randomized_cohort`'s `precomputed_endpoint_results`
            parameter. `None` (the default) preserves the original
            recompute-from-timeline-buckets behavior every synthetic-
            fixture-based caller already relies on. A real-data caller
            (`data.loaders.actg175_adapter.raw_tables_to_pipeline_inputs`'s
            third return value) passes its `{patient_id: EndpointResult}`
            mapping here, since the real ACTG175 data provides only a
            composite outcome with no per-component breakdown (see that
            module's docstring).
        selection_severity: which frozen selection mechanism steps 5-8
            apply, forwarded to every `observationalize` call. `"primary"`
            (the default) is the `decisions.md` Section 9 mechanism used
            by every primary result; `"stronger"` is the Section 26
            amendment's mechanism, used only by the
            `stronger_selection_severity` sensitivity variant. Steps 1-4
            (cohort, sealed reference, split, preprocessing) and every
            seed are identical under both, so the two pipelines differ
            only in which patients selection retains.

    Returns:
        An `ObservationalizationPipelineResult` bundling every stage's
        output.
    """
    # Step 1: construct the complete eligible arm-0/arm-1 randomized cohort.
    cohort: RandomizedCohort = build_randomized_cohort(
        raw_timelines, treatment_arms, spec, precomputed_endpoint_results
    )

    # Step 2: compute and seal the internal randomized reference, before any
    # splitting or selection.
    sealed_reference = SealedReference(cohort, estimator=reference_estimator)

    # Step 3: the 60/20/20 (by default) patient-level dev/val/test split,
    # using the frozen split seed derived from master_seed.
    split_seed = derive_seed(master_seed, "split")
    split_cohorts: SplitCohorts = stratified_patient_split(
        cohort,
        sealed_reference,
        fractions=split_fractions,
        seed=split_seed,
    )

    # Step 4: fit preprocessing (z-scoring) on the development split only.
    resolved_fields = (
        preprocessing_fields if preprocessing_fields is not None else DEFAULT_CONTINUOUS_FIELDS
    )
    fitted_preprocessing = FittedPreprocessing(fields=resolved_fields)
    fitted_preprocessing.fit(DevelopmentSet(split_cohorts.dev))

    # Steps 5-8: observationalize each split independently, using the SAME
    # development-fitted preprocessing and formula, but a DISTINCT
    # split-specific seed per split (no random-number stream reused across
    # splits).
    dev_result = observationalize(
        split_cohorts.dev,
        fitted_preprocessing,
        seed=derive_seed(master_seed, "dev"),
        selection_severity=selection_severity,
    )
    val_result = observationalize(
        split_cohorts.val,
        fitted_preprocessing,
        seed=derive_seed(master_seed, "val"),
        selection_severity=selection_severity,
    )
    test_result = observationalize(
        split_cohorts.test,
        fitted_preprocessing,
        seed=derive_seed(master_seed, "test"),
        selection_severity=selection_severity,
    )

    return ObservationalizationPipelineResult(
        sealed_reference=sealed_reference,
        split_cohorts=split_cohorts,
        fitted_preprocessing=fitted_preprocessing,
        dev=dev_result,
        val=val_result,
        test=test_result,
        master_seed=master_seed,
    )
