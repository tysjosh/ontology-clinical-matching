"""Property 30: synthetic cohorts are contract-valid, ground-truthed, and
reproducible (task 19.1; Requirement 16.1-16.4).

**Validates: Requirements 16.1, 16.2, 16.3, 16.4; Correctness Property 30**

Property 30, verbatim (`design.md`): "*For any* scenario parameters and
seed, the generated cohort satisfies the same data contract the real
cohort satisfies (accepted by the existing eligibility, endpoint, and
contract validators), its realized treatment effect matches its declared
ground truth within sampling tolerance, increasing a failure-mode severity
knob monotonically worsens the corresponding diagnostic, and two
generations at the same seed are identical while different seeds differ."
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from evaluation.synthetic_benchmark import SCENARIOS, SyntheticScenario, generate
from protocols.eligibility_dsl import ACTG175_ELIGIBILITY
from protocols.eligibility_dsl import PatientTimeline as EligibilityPatientTimeline
from protocols.endpoints import CompositeEndpoint, PatientEndpointRecord

# ---------------------------------------------------------------------------
# Shared strategies.
# ---------------------------------------------------------------------------

_scenario_name_strategy = st.sampled_from(SCENARIOS)
_knob_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_true_risk_difference_strategy = st.floats(
    min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False
)
_seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)


def _scenario(
    name: str,
    *,
    n_patients: int = 120,
    true_risk_difference: float = -0.1,
    overlap_severity: float = 0.2,
    unmeasured_confounding_strength: float = 0.2,
    missingness_rate: float = 0.1,
) -> SyntheticScenario:
    return SyntheticScenario(
        name=name,
        n_patients=n_patients,
        true_risk_difference=true_risk_difference,
        overlap_severity=overlap_severity,
        unmeasured_confounding_strength=unmeasured_confounding_strength,
        missingness_rate=missingness_rate,
    )


# ---------------------------------------------------------------------------
# Contract validity: eligibility + endpoint construction accept every
# generated patient (Requirement 16.3).
# ---------------------------------------------------------------------------


@given(
    name=_scenario_name_strategy,
    overlap_severity=_knob_strategy,
    unmeasured_confounding_strength=_knob_strategy,
    missingness_rate=_knob_strategy,
    true_risk_difference=_true_risk_difference_strategy,
    seed=_seed_strategy,
)
@settings(max_examples=25, deadline=None)
def test_generated_patients_are_accepted_by_the_real_eligibility_and_endpoint_validators(
    name, overlap_severity, unmeasured_confounding_strength, missingness_rate,
    true_risk_difference, seed,
):
    scenario = _scenario(
        name,
        n_patients=40,
        true_risk_difference=true_risk_difference,
        overlap_severity=overlap_severity,
        unmeasured_confounding_strength=unmeasured_confounding_strength,
        missingness_rate=missingness_rate,
    )
    cohort = generate(scenario, seed=seed)

    assert len(cohort.timelines) == 40
    for timeline in cohort.timelines:
        eligibility_timeline = EligibilityPatientTimeline(
            baseline_covariates=dict(timeline.static_covariates)
        )
        result = ACTG175_ELIGIBILITY.evaluate(eligibility_timeline)
        assert result.eligible, (timeline.patient_id, result)

        # Endpoint construction: every generated patient carries enough
        # information (baseline CD4, a treatment-start event, at least one
        # recorded post-baseline timestamp) for CompositeEndpoint to run
        # without error, exactly as the real cohort's adapter requires.
        assignment_time = min(event.observed_at for event in timeline.treatment_events)
        last_observed_time = max(
            event.observed_at
            for bucket in (
                timeline.timestamped_observations,
                timeline.treatment_events,
                timeline.endpoint_events,
                timeline.censoring_events,
            )
            for event in bucket
        )
        cd4_observations = [
            (event.observed_at, event.value)
            for event in timeline.timestamped_observations
            if event.concept_id == "cd4_count" and event.value is not None
        ]
        aids_defining_event_time = next(
            (
                event.observed_at
                for event in timeline.endpoint_events
                if event.concept_id == "aids_defining_event"
            ),
            None,
        )
        death_time = next(
            (
                event.observed_at
                for event in timeline.endpoint_events
                if event.concept_id == "death"
            ),
            None,
        )
        record = PatientEndpointRecord(
            baseline_cd4=timeline.static_covariates["baseline_cd4"],
            last_observed_time=last_observed_time,
            assignment_time=assignment_time,
            cd4_observations=cd4_observations,
            aids_defining_event_time=aids_defining_event_time,
            death_time=death_time,
        )
        endpoint_result = CompositeEndpoint.compute(record)
        assert endpoint_result.event_indicator in (True, False)


# ---------------------------------------------------------------------------
# Ground-truth recovery: the realized ATE matches the declared ground
# truth within sampling tolerance at a large-ish n (Requirement 16.1).
# ---------------------------------------------------------------------------


def test_realized_effect_matches_declared_ground_truth_within_sampling_tolerance():
    for true_risk_difference in (-0.2, -0.05, 0.0, 0.1, 0.25):
        scenario = _scenario(
            "limited_overlap", n_patients=4000, true_risk_difference=true_risk_difference
        )
        cohort = generate(scenario, seed=12345)

        assert abs(cohort.realized_average_treatment_effect - true_risk_difference) < 0.03


# ---------------------------------------------------------------------------
# Monotonicity: increasing each severity knob never improves (and, over a
# wide enough range, worsens) its own diagnostic.
# ---------------------------------------------------------------------------


def test_increasing_overlap_severity_never_improves_the_extreme_propensity_fraction():
    severities = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    fractions = []
    for severity in severities:
        scenario = _scenario("limited_overlap", n_patients=800, overlap_severity=severity)
        cohort = generate(scenario, seed=99)
        fractions.append(cohort.realized_extreme_propensity_fraction)

    assert all(fractions[i] <= fractions[i + 1] for i in range(len(fractions) - 1))
    assert fractions[-1] > fractions[0]


def test_increasing_unmeasured_confounding_strength_never_improves_the_naive_bias():
    """Averaged over several seeds to smooth the sampling noise that
    otherwise dominates near `unmeasured_confounding_strength=0` (where
    the true naive bias is close to zero, so a single draw's sign can
    flip either way) -- the monotonic *trend* this property asserts is in
    the expectation, not in every single draw."""
    strengths = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    seeds = (100, 101, 102, 103, 104)
    biases = []
    for strength in strengths:
        scenario = _scenario(
            "unmeasured_confounding",
            n_patients=1500,
            overlap_severity=0.1,
            unmeasured_confounding_strength=strength,
        )
        per_seed_biases = [generate(scenario, seed=seed).realized_naive_bias for seed in seeds]
        biases.append(sum(per_seed_biases) / len(per_seed_biases))

    assert all(biases[i] <= biases[i + 1] for i in range(len(biases) - 1))
    assert biases[-1] > biases[0]


def test_increasing_missingness_rate_never_improves_the_missingness_fraction():
    rates = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    fractions = []
    for rate in rates:
        scenario = _scenario("increased_missingness", n_patients=300, missingness_rate=rate)
        cohort = generate(scenario, seed=101)
        fractions.append(cohort.realized_missingness_fraction)

    assert all(fractions[i] <= fractions[i + 1] for i in range(len(fractions) - 1))
    assert fractions[-1] > fractions[0]


# ---------------------------------------------------------------------------
# Reproducibility: same seed -> identical cohort; different seed -> differs.
# ---------------------------------------------------------------------------


@given(name=_scenario_name_strategy, seed=_seed_strategy)
@settings(max_examples=15, deadline=None)
def test_same_seed_reproduces_identical_cohorts_different_seeds_differ(name, seed):
    scenario = _scenario(name, n_patients=30)

    cohort_a = generate(scenario, seed=seed)
    cohort_b = generate(scenario, seed=seed)
    cohort_c = generate(scenario, seed=seed + 1)

    assert cohort_a.treatment_arms == cohort_b.treatment_arms
    assert cohort_a.potential_outcome_arm0 == cohort_b.potential_outcome_arm0
    assert cohort_a.potential_outcome_arm1 == cohort_b.potential_outcome_arm1
    for timeline_a, timeline_b in zip(cohort_a.timelines, cohort_b.timelines, strict=True):
        assert timeline_a.static_covariates == timeline_b.static_covariates

    assert (
        cohort_a.treatment_arms != cohort_c.treatment_arms
        or cohort_a.potential_outcome_arm0 != cohort_c.potential_outcome_arm0
        or cohort_a.potential_outcome_arm1 != cohort_c.potential_outcome_arm1
    )
