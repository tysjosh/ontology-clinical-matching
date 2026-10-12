"""Controlled semi-synthetic benchmark with a known ground-truth
counterfactual effect (task 19.1; Requirement 16.1-16.4; `design.md`
Section 3.14; Correctness Property 30).

`decisions.md` has no frozen formula for a synthetic data-generating
process -- Requirement 16 only fixes *behavior* (a known ground-truth
effect, three named failure-mode scenarios, the same data contract as the
real cohort, and seed-reproducibility), not numeric parameters. Per
`design.md`'s own "Open questions" note: "the exact confounding
mechanism, and the default cohort size are implementation choices,
constrained by Property 30. Because these values affect no reported
real-data number, they need no `decisions.md` amendment." Every scale
constant below is therefore this module's own documented choice, not a
value transcribed from `decisions.md`.

Why this module does not import `scripts/_synthetic_fixtures.py`
-------------------------------------------------------------------------

`scripts/_synthetic_fixtures.py` builds a similarly-shaped synthetic
cohort for `scripts/run_stage.py`'s non-`smoke` stages, and the task text
describes this module as "extending" it. `scripts/` is this project's
layer of CLI entry points that import from `src/`; a `src/evaluation`
module importing back from `scripts/` would invert that dependency
direction (`scripts` already imports `evaluation.*` throughout this
codebase). This module therefore reimplements the same event-emission
*pattern* `_synthetic_fixtures.py` already established (one
`data.event_model.ClinicalEvent` per baseline covariate/treatment/lab/
endpoint fact, assembled via `data.timelines.build_patient_timeline`) --
"extending" it in spirit (the DGP, the ground-truth effect, and the three
severity knobs) rather than in the Python `import` sense.

The data-generating process
-------------------------------------------------------------------------

For patient `i`, drawn from a `random.Random` seeded via
`derive_seed(seed, scenario.name)` (Requirement 16.4):

1. Observed baseline covariates (`baseline_cd4`, `age`, `karnofsky_score`,
   `symptomatic_status`, `prior_opportunistic_infection`,
   `prior_art_exposure_days`) are drawn independently, within ACTG175's
   real eligibility ranges, so every generated patient satisfies
   `protocols.eligibility_dsl.ACTG175_ELIGIBILITY` (Requirement 16.3).
2. A **latent** confounder `U_i ~ Uniform(0, 1)` is drawn but never
   emitted as an observed covariate -- it is the "unmeasured" part of
   `unmeasured_confounding_strength`.
3. Treatment assignment is confounded on both the observed baseline risk
   index `R_i` (scaled by `scenario.overlap_severity`) and the latent
   `U_i` (scaled by `scenario.unmeasured_confounding_strength`):
   `e_i = sigmoid(OVERLAP_LOGIT_SCALE * overlap_severity * (R_i - 0.5) +
   CONFOUNDING_LOGIT_SCALE * unmeasured_confounding_strength * (U_i -
   0.5))`. Higher `overlap_severity` pushes `e_i` toward the extremes
   (worse common support); higher `unmeasured_confounding_strength`
   correlates treatment assignment with the very latent variable that
   also drives the outcome in step 4, which is exactly what makes it
   unmeasured confounding rather than an adjustable one.
4. Both potential outcomes are drawn independently:
   `Y_i(0) ~ Bernoulli(p0_i)`, `Y_i(1) ~ Bernoulli(clip(p0_i +
   scenario.true_risk_difference, 0, 1))`, where `p0_i` is itself a
   function of `R_i` and `U_i` (the same latent confounder). The observed
   outcome is `Y_i(A_i)`. Because `scenario.true_risk_difference` is
   added directly to the arm-1 potential-outcome probability for every
   patient, the realized population-average treatment effect
   (`mean(Y_i(1)) - mean(Y_i(0))`, computed in `generate()` from the
   drawn-but-not-necessarily-observed potential outcomes) converges to
   `scenario.true_risk_difference` as `n_patients` grows (Requirement
   16.1; Property 30's "realized effect matches the declared ground truth
   within sampling tolerance").
5. The composite endpoint is realized as a `"death"` event at a sampled
   pre-horizon time when `Y_i(A_i) == 1`, or `"administrative_censoring"`
   at the horizon otherwise -- so `protocols.endpoints.CompositeEndpoint`
   (reached via `cohorts.cohort_construction.build_randomized_cohort`'s
   existing adapter, never re-derived here) produces an `EndpointResult`
   consistent with the drawn outcome.
6. The week-96 CD4 lab observation is independently marked
   `MissingnessState.MISSING` with probability `scenario.missingness_rate`.

No second pipeline: steps 1-6 only decide which `ClinicalEvent`s to emit.
Every emitted record uses the identical `concept_id`/`modality` vocabulary
`data.loaders.actg175_adapter`/`cohorts.cohort_construction` already
expect, so a caller feeds `SyntheticCohort.timelines`/`.treatment_arms`
into the exact same `cohorts.pipeline.run_observationalization_pipeline`
every real-data caller uses (Requirement 16.3).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import PatientTimeline, build_patient_timeline
from infra.seeding import derive_seed

__all__ = [
    "CONFOUNDING_LOGIT_SCALE",
    "HORIZON_WEEKS",
    "OUTCOME_CONFOUNDING_SCALE",
    "OVERLAP_LOGIT_SCALE",
    "SCENARIOS",
    "SyntheticCohort",
    "SyntheticScenario",
    "generate",
]

#: The three frozen (by this module's own choice, see module docstring)
#: failure-mode scenario names (Requirement 16.2).
SCENARIOS: tuple[str, ...] = ("limited_overlap", "unmeasured_confounding", "increased_missingness")

#: The time-to-event horizon this benchmark's endpoint realization uses,
#: matching `causal.baselines.HORIZON_WEEKS` (96-week composite endpoint)
#: without importing that module (this one is deliberately dependency-light
#: -- see module docstring on layering).
HORIZON_WEEKS: float = 96.0

#: How strongly `overlap_severity` pushes the propensity logit away from
#: 0 (i.e. toward extreme, poorly-overlapping propensities). An
#: implementation choice (see module docstring); not a `decisions.md`
#: value. Chosen so `overlap_severity=1.0` pushes a meaningful fraction of
#: patients outside the primary `[0.10, 0.90]` common-support band
#: (`sigmoid(+-5) ~= 0.0067/0.9933`), while `overlap_severity=0.0` leaves
#: every patient's propensity exactly at `sigmoid(0) = 0.5`.
OVERLAP_LOGIT_SCALE: float = 10.0

#: How strongly `unmeasured_confounding_strength` ties treatment
#: assignment to the latent confounder `U_i`. An implementation choice.
CONFOUNDING_LOGIT_SCALE: float = 4.0

#: How strongly the latent confounder `U_i` shifts the baseline outcome
#: probability `p0_i` -- the other half of "unmeasured confounding" (the
#: same `U_i` that drove assignment in step 3 also drives the outcome
#: here). An implementation choice.
OUTCOME_CONFOUNDING_SCALE: float = 0.5

#: The reference wall-clock time step 5's event timestamps are offset
#: from -- arbitrary, consistent with `scripts/_synthetic_fixtures.py`'s
#: own fixed `ASSIGNMENT_TIME`.
_ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
_WEEK_20 = _ASSIGNMENT_TIME + timedelta(weeks=20)
_WEEK_96 = _ASSIGNMENT_TIME + timedelta(weeks=96)


@dataclass(frozen=True)
class SyntheticScenario:
    """One failure-mode scenario's generation parameters (Requirement 16.2).

    Attributes:
        name: one of `SCENARIOS`.
        n_patients: total synthetic cohort size (both arms combined).
        true_risk_difference: the declared ground-truth 96-week risk
            difference (`P(Y(1)=1) - P(Y(0)=1)`), added directly to every
            patient's arm-1 potential-outcome probability (Requirement
            16.1).
        overlap_severity: in `[0, 1]`. `0` means treatment assignment is
            unconfounded by the observed baseline risk index (good
            common support); `1` means heavily confounded by it (poor
            common support). See module docstring step 3.
        unmeasured_confounding_strength: in `[0, 1]`. `0` means the
            latent confounder `U_i` has no effect on either assignment or
            outcome (no unmeasured confounding); `1` means it strongly
            drives both. See module docstring steps 3-4.
        missingness_rate: in `[0, 1]`, the probability a patient's
            week-96 CD4 lab observation is marked
            `MissingnessState.MISSING`.
    """

    name: str
    n_patients: int
    true_risk_difference: float
    overlap_severity: float
    unmeasured_confounding_strength: float
    missingness_rate: float

    def __post_init__(self) -> None:
        if self.name not in SCENARIOS:
            raise ValueError(f"name must be one of {SCENARIOS!r}, got {self.name!r}")
        if self.n_patients <= 0:
            raise ValueError(f"n_patients must be positive, got {self.n_patients!r}")
        knob_fields = ("overlap_severity", "unmeasured_confounding_strength", "missingness_rate")
        for field_name in knob_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True)
class SyntheticCohort:
    """`generate()`'s output: a benchmark cohort plus everything needed to
    both feed the real pipeline and verify the generator itself
    (Correctness Property 30).

    Attributes:
        scenario: the `SyntheticScenario` this cohort was generated from.
        timelines: one `data.timelines.PatientTimeline` per patient,
            satisfying the same data contract the real cohort satisfies
            (Requirement 16.3) -- feed directly into
            `cohorts.pipeline.run_observationalization_pipeline` or
            `cohorts.cohort_construction.build_randomized_cohort`.
        treatment_arms: `{patient_id: observed_arm}`, the observed
            (confounded) assignment `A_i`.
        potential_outcome_arm0 / potential_outcome_arm1: `{patient_id:
            Y_i(0)}` / `{patient_id: Y_i(1)}` -- BOTH potential outcomes
            for every patient, drawn independently regardless of which
            arm was actually assigned. Never fed into the emitted
            `ClinicalEvent`s (only `Y_i(A_i)`, the observed outcome, is);
            exposed here purely so a caller/test can verify the
            generator's own internal consistency (Property 30) without
            running the full observational pipeline.
        realized_average_treatment_effect: `mean(potential_outcome_arm1)
            - mean(potential_outcome_arm0)` over all patients -- the
            benchmark's own sampling-noise-affected realization of
            `scenario.true_risk_difference` (Requirement 16.1; Property
            30's ground-truth-matches-within-tolerance check).
        realized_extreme_propensity_fraction: fraction of patients whose
            confounded propensity `e_i` falls outside the primary
            `[0.10, 0.90]` common-support band (`causal.overlap`'s
            own primary band, reused here as a plain float constant to
            avoid importing that module) -- the overlap diagnostic that
            should worsen (increase) as `overlap_severity` increases.
        realized_naive_bias: `abs(naive_difference -
            scenario.true_risk_difference)`, where `naive_difference` is
            the unadjusted `mean(Y_i | A_i=1) - mean(Y_i | A_i=0)` over
            the OBSERVED outcomes -- the confounding diagnostic that
            should worsen (increase) as `unmeasured_confounding_strength`
            increases.
        realized_missingness_fraction: the fraction of patients whose
            week-96 CD4 lab was marked missing -- should track
            `scenario.missingness_rate` directly.
    """

    scenario: SyntheticScenario
    timelines: tuple[PatientTimeline, ...]
    treatment_arms: dict[str, int]
    potential_outcome_arm0: dict[str, int] = field(default_factory=dict)
    potential_outcome_arm1: dict[str, int] = field(default_factory=dict)
    realized_average_treatment_effect: float = 0.0
    realized_extreme_propensity_fraction: float = 0.0
    realized_naive_bias: float = 0.0
    realized_missingness_fraction: float = 0.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _event(
    patient_id: str,
    *,
    concept_id: str,
    value: float | bool | None,
    unit: str | None,
    modality: str,
    observed_at: datetime,
    missingness: MissingnessState = MissingnessState.OBSERVED,
) -> ClinicalEvent:
    return ClinicalEvent(
        patient_id=patient_id,
        encounter_id=None,
        concept_id=concept_id,
        value=value,
        unit=unit,
        modality=modality,
        observed_at=observed_at,
        valid_from=None,
        valid_to=None,
        source="synthetic_benchmark",
        missingness=missingness,
        provenance="evaluation.synthetic_benchmark.generate",
    )


def _build_patient_events(
    patient_id: str,
    *,
    baseline_cd4: float,
    age: float,
    karnofsky_score: float,
    symptomatic_status: bool,
    prior_opportunistic_infection: bool,
    prior_art_exposure_days: float,
    treatment_arm: int,
    week_96_missing: bool,
    event_occurred: bool,
    event_time_weeks: float,
) -> list[ClinicalEvent]:
    treatment_concept = "zidovudine" if treatment_arm == 0 else "zidovudine+didanosine"
    events = [
        _event(
            patient_id,
            concept_id="hiv_positive",
            value=True,
            unit=None,
            modality="baseline",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="baseline_cd4",
            value=baseline_cd4,
            unit="cells/mm3",
            modality="baseline",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="baseline_cd8",
            value=baseline_cd4 * 2.0,
            unit="cells/mm3",
            modality="baseline",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="age",
            value=age,
            unit="years",
            modality="baseline",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="karnofsky_score",
            value=karnofsky_score,
            unit="score",
            modality="baseline",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="symptomatic_status",
            value=symptomatic_status,
            unit=None,
            modality="baseline",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="prior_opportunistic_infection",
            value=prior_opportunistic_infection,
            unit=None,
            modality="baseline",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="prior_art_exposure_days",
            value=prior_art_exposure_days,
            unit="days",
            modality="baseline",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id=treatment_concept,
            value=1.0,
            unit=None,
            modality="treatment",
            observed_at=_ASSIGNMENT_TIME,
        ),
        _event(
            patient_id,
            concept_id="cd4_count",
            value=baseline_cd4 - 10.0,
            unit="cells/mm3",
            modality="lab",
            observed_at=_WEEK_20,
        ),
        _event(
            patient_id,
            concept_id="cd4_count",
            value=None if week_96_missing else max(0.0, baseline_cd4 - 30.0),
            unit="cells/mm3",
            modality="lab",
            observed_at=_WEEK_96,
            missingness=(
                MissingnessState.MISSING if week_96_missing else MissingnessState.OBSERVED
            ),
        ),
    ]
    if event_occurred:
        events.append(
            _event(
                patient_id,
                concept_id="death",
                value=1.0,
                unit=None,
                modality="endpoint",
                observed_at=_ASSIGNMENT_TIME + timedelta(weeks=event_time_weeks),
            )
        )
    else:
        events.append(
            _event(
                patient_id,
                concept_id="administrative_censoring",
                value=1.0,
                unit=None,
                modality="censoring",
                observed_at=_ASSIGNMENT_TIME + timedelta(weeks=HORIZON_WEEKS),
            )
        )
    return events


def generate(scenario: SyntheticScenario, seed: int) -> SyntheticCohort:
    """Generate one synthetic benchmark cohort for `scenario`, fully
    determined by `derive_seed(seed, scenario.name)` (Requirement 16.4).

    Args:
        scenario: the scenario parameters (Requirement 16.1-16.2).
        seed: the top-level seed; the actual generation seed is
            `infra.seeding.derive_seed(seed, scenario.name)`, so the same
            `seed` reproduces identical cohorts across scenarios without
            their random streams colliding, and a different `seed`
            produces a different cohort (Requirement 16.4).

    Returns:
        A `SyntheticCohort`.
    """
    rng = random.Random(derive_seed(seed, scenario.name))

    timelines: list[PatientTimeline] = []
    treatment_arms: dict[str, int] = {}
    potential_outcome_arm0: dict[str, int] = {}
    potential_outcome_arm1: dict[str, int] = {}
    extreme_propensity_count = 0
    observed_sum_by_arm = {0: 0.0, 1: 0.0}
    observed_count_by_arm = {0: 0, 1: 0}
    missing_count = 0

    for i in range(scenario.n_patients):
        patient_id = f"SYN{scenario.name[:4].upper()}{i:05d}"

        baseline_cd4 = rng.uniform(200.0, 500.0)
        age = rng.uniform(18.0, 65.0)
        karnofsky_score = rng.uniform(70.0, 100.0)
        symptomatic_status = rng.random() < 0.3
        prior_opportunistic_infection = rng.random() < 0.2
        prior_art_exposure_days = float(rng.choice([0, 30, 60, 90]))

        # Observed baseline risk index in [0, 1]; lower CD4 -> higher risk.
        risk_index = _clip01((500.0 - baseline_cd4) / 300.0)
        # Latent confounder, never emitted as an observed covariate.
        latent_confounder = rng.random()

        propensity_logit = OVERLAP_LOGIT_SCALE * scenario.overlap_severity * (
            risk_index - 0.5
        ) + CONFOUNDING_LOGIT_SCALE * scenario.unmeasured_confounding_strength * (
            latent_confounder - 0.5
        )
        propensity = _sigmoid(propensity_logit)
        if not 0.10 <= propensity <= 0.90:
            extreme_propensity_count += 1

        treatment_arm = 1 if rng.random() < propensity else 0

        baseline_outcome_probability = _clip01(
            0.15
            + 0.30 * risk_index
            + OUTCOME_CONFOUNDING_SCALE
            * scenario.unmeasured_confounding_strength
            * (latent_confounder - 0.5)
        )
        arm1_outcome_probability = _clip01(
            baseline_outcome_probability + scenario.true_risk_difference
        )

        outcome_arm0 = 1 if rng.random() < baseline_outcome_probability else 0
        outcome_arm1 = 1 if rng.random() < arm1_outcome_probability else 0
        observed_outcome = outcome_arm1 if treatment_arm == 1 else outcome_arm0

        week_96_missing = rng.random() < scenario.missingness_rate
        event_time_weeks = rng.uniform(1.0, HORIZON_WEEKS - 1.0)

        events = _build_patient_events(
            patient_id,
            baseline_cd4=baseline_cd4,
            age=age,
            karnofsky_score=karnofsky_score,
            symptomatic_status=symptomatic_status,
            prior_opportunistic_infection=prior_opportunistic_infection,
            prior_art_exposure_days=prior_art_exposure_days,
            treatment_arm=treatment_arm,
            week_96_missing=week_96_missing,
            event_occurred=bool(observed_outcome),
            event_time_weeks=event_time_weeks,
        )

        timelines.append(build_patient_timeline(events))
        treatment_arms[patient_id] = treatment_arm
        potential_outcome_arm0[patient_id] = outcome_arm0
        potential_outcome_arm1[patient_id] = outcome_arm1
        observed_sum_by_arm[treatment_arm] += observed_outcome
        observed_count_by_arm[treatment_arm] += 1
        if week_96_missing:
            missing_count += 1

    n = scenario.n_patients
    realized_ate = (
        sum(potential_outcome_arm1.values()) / n - sum(potential_outcome_arm0.values()) / n
    )
    naive_mean_1 = (
        observed_sum_by_arm[1] / observed_count_by_arm[1]
        if observed_count_by_arm[1] > 0
        else float("nan")
    )
    naive_mean_0 = (
        observed_sum_by_arm[0] / observed_count_by_arm[0]
        if observed_count_by_arm[0] > 0
        else float("nan")
    )
    naive_difference = naive_mean_1 - naive_mean_0
    naive_bias = abs(naive_difference - scenario.true_risk_difference)

    return SyntheticCohort(
        scenario=scenario,
        timelines=tuple(timelines),
        treatment_arms=treatment_arms,
        potential_outcome_arm0=potential_outcome_arm0,
        potential_outcome_arm1=potential_outcome_arm1,
        realized_average_treatment_effect=realized_ate,
        realized_extreme_propensity_fraction=extreme_propensity_count / n,
        realized_naive_bias=naive_bias,
        realized_missingness_fraction=missing_count / n,
    )
