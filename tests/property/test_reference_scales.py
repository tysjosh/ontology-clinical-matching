"""Property-based scale-safety tests for
`cohorts.sealed_reference.compare_to_reference` (`design.md` Section 2.6
"Tension 4", scale safety; "Correctness Property 24: Scale-matched
reference comparison").

The sealed reference carries four quantities on three different scales
(`log_hazard_ratio`, `risk`, `risk_difference`), only one of which is
comparable with any given estimate. `compare_to_reference` is the single
place that pairing is checked, and it is checked by a plain `==` between two
strings drawn from one shared vocabulary --
`causal.baselines.CausalEstimate.scale` and
`cohorts.sealed_reference.ReferenceQuantity.scale` -- rather than by
convention. These tests generate every scale pairing and confirm the check
is exactly "equal scales succeed, unequal scales raise", with no exception
carved out for the pairing that would look most plausible to a reader: a
risk difference against a log-hazard ratio.

No mocks or stand-in doubles: the estimates are real
`causal.baselines.CausalEstimate` values and the references are real
`cohorts.sealed_reference.InternalReference` values, including one built by
the actual `extended_randomized_reference_estimator` over a synthetic
cohort.

**Validates: Requirements 14.3, 14.4**
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causal.baselines import HORIZON_WEEKS, CausalEstimate
from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from cohorts.sealed_reference import (
    InternalReference,
    ReferenceQuantity,
    ScaleMismatchError,
    compare_to_reference,
    extended_randomized_reference_estimator,
    unadjusted_cox_reference_estimator,
)
from data.event_model import ClinicalEvent, MissingnessState
from data.timelines import build_patient_timeline
from protocols.eligibility_dsl import EligibilityResult
from protocols.endpoints import EndpointResult

_MAX_EXAMPLES = 50

#: The complete scale vocabulary, shared by `CausalEstimate.scale` and
#: `ReferenceQuantity.scale` (see `cohorts.sealed_reference.ReferenceScale`).
SCALES = ("log_hazard_ratio", "risk", "risk_difference")

ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)

_finite_value = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)


def _make_estimate(scale: str, value: float) -> CausalEstimate:
    return CausalEstimate(
        name=f"estimate_on_{scale}",
        estimate=value,
        se=None,
        is_primary=False,
        scale=scale,
    )


def _make_reference(quantity_scale: str, value: float) -> InternalReference:
    return InternalReference(
        value=value,
        cohort_size=100,
        computed_at=datetime.now(UTC),
        quantities={
            "the_quantity": ReferenceQuantity(
                value=value,
                scale=quantity_scale,
                horizon_weeks=None if quantity_scale == "log_hazard_ratio" else HORIZON_WEEKS,
            )
        },
    )


# ---------------------------------------------------------------------------
# Property 24
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(
    estimate_scale=st.sampled_from(SCALES),
    reference_scale=st.sampled_from(SCALES),
    estimate_value=_finite_value,
    reference_value=_finite_value,
)
def test_property_24_scale_matched_reference_comparison(
    estimate_scale: str,
    reference_scale: str,
    estimate_value: float,
    reference_value: float,
):
    """**Correctness Property 24: Scale-matched reference comparison.**

    *For any* pair of an estimate scale and a sealed-reference quantity
    scale, `compare_to_reference` succeeds exactly when the two scales are
    equal and raises `ScaleMismatchError` otherwise.

    **Validates: Requirements 14.3, 14.4**
    """
    estimate = _make_estimate(estimate_scale, estimate_value)
    reference = _make_reference(reference_scale, reference_value)

    if estimate_scale == reference_scale:
        error = compare_to_reference(estimate, reference, "the_quantity")
        assert math.isclose(error, abs(estimate_value - reference_value), abs_tol=1e-12)
    else:
        with pytest.raises(ScaleMismatchError) as excinfo:
            compare_to_reference(estimate, reference, "the_quantity")
        message = str(excinfo.value)
        assert estimate_scale in message
        assert reference_scale in message


# ---------------------------------------------------------------------------
# The explicit case Requirement 14.4 names: a risk-difference estimate
# against the log-hazard-ratio reference quantity.
# ---------------------------------------------------------------------------


def _patient_record(
    patient_id: str, *, treatment_arm: int, event_time_weeks: float, event_indicator: bool
) -> PatientRecord:
    def _event(concept_id: str, value: object, unit: str | None, modality: str) -> ClinicalEvent:
        return ClinicalEvent(
            patient_id=patient_id,
            encounter_id=None,
            concept_id=concept_id,
            value=value,
            unit=unit,
            modality=modality,
            observed_at=ASSIGNMENT_TIME,
            valid_from=None,
            valid_to=None,
            source="actg175",
            missingness=MissingnessState.OBSERVED,
            provenance="loader:actg175",
        )

    events = [
        _event("baseline_cd4", 350.0, "cells/mm3", "baseline"),
        _event("age", 34.0, "years", "baseline"),
        _event(
            "zidovudine" if treatment_arm == 0 else "zidovudine+didanosine",
            1.0,
            None,
            "treatment",
        ),
    ]
    return PatientRecord(
        patient_id=patient_id,
        treatment_arm=treatment_arm,
        timeline=build_patient_timeline(events),
        eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
        endpoint_result=EndpointResult(
            event_time=ASSIGNMENT_TIME + timedelta(weeks=event_time_weeks),
            event_indicator=event_indicator,
            censoring_indicator=not event_indicator,
            event_type="aids_defining_event" if event_indicator else None,
        ),
    )


@pytest.fixture
def real_extended_reference() -> InternalReference:
    """A reference produced by the actual `extended_randomized_reference_estimator`
    over a small deterministic cohort -- so the scale tags under test are the
    ones real sealing code writes, not test-authored strings."""
    patients = tuple(
        _patient_record(
            f"p{index:03d}",
            treatment_arm=index % 2,
            event_time_weeks=30.0 + (index % 7) * 15.0,
            event_indicator=index % 3 == 0,
        )
        for index in range(24)
    )
    return extended_randomized_reference_estimator(RandomizedCohort(patients=patients))


def test_risk_difference_estimate_against_log_hazard_ratio_reference_raises(
    real_extended_reference: InternalReference,
):
    """Requirement 14.4, stated literally: pairing a risk-difference
    estimate with the log-hazard-ratio reference quantity raises."""
    risk_difference_estimate = _make_estimate("risk_difference", -0.08)

    with pytest.raises(ScaleMismatchError):
        compare_to_reference(risk_difference_estimate, real_extended_reference, "log_hazard_ratio")


def test_risk_difference_estimate_against_risk_difference_reference_succeeds(
    real_extended_reference: InternalReference,
):
    risk_difference_estimate = _make_estimate("risk_difference", -0.08)

    error = compare_to_reference(
        risk_difference_estimate, real_extended_reference, "risk_difference"
    )

    assert error == pytest.approx(
        abs(-0.08 - real_extended_reference.quantities["risk_difference"].value)
    )


def test_every_real_quantity_matches_only_its_own_scale(
    real_extended_reference: InternalReference,
):
    """The whole quantity mapping, not just the named pair: each of the four
    sealed quantities accepts an estimate on its own scale and refuses the
    other two scales."""
    for name, quantity in real_extended_reference.quantities.items():
        for scale in SCALES:
            estimate = _make_estimate(scale, 0.1)
            if scale == quantity.scale:
                compare_to_reference(estimate, real_extended_reference, name)
            else:
                with pytest.raises(ScaleMismatchError):
                    compare_to_reference(estimate, real_extended_reference, name)


def test_version_1_reference_has_no_quantity_to_compare_against():
    """A version-1-shaped reference carries no `quantities`, so a
    comparison-by-name has nothing to select and says so, naming the
    re-seal path rather than silently falling back to `value` (whose scale
    is unchecked)."""
    patients = tuple(
        _patient_record(
            f"p{index:03d}",
            treatment_arm=index % 2,
            event_time_weeks=40.0 + index,
            event_indicator=index % 2 == 0,
        )
        for index in range(10)
    )
    v1_reference = unadjusted_cox_reference_estimator(RandomizedCohort(patients=patients))
    estimate = _make_estimate("risk_difference", 0.05)

    with pytest.raises(KeyError) as excinfo:
        compare_to_reference(estimate, v1_reference, "risk_difference")

    assert "risk_difference" in str(excinfo.value)


def test_unknown_quantity_name_names_the_available_quantities(
    real_extended_reference: InternalReference,
):
    estimate = _make_estimate("risk_difference", 0.05)

    with pytest.raises(KeyError) as excinfo:
        compare_to_reference(estimate, real_extended_reference, "not_a_quantity")

    message = str(excinfo.value)
    assert "not_a_quantity" in message
    assert "risk_arm0" in message
