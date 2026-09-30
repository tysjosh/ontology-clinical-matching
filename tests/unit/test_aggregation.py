"""Unit tests for `src/models/aggregation.py`.

Covers:

1. `PatientLevelEstimate.kind`'s type annotation is exactly
   `Literal["model_estimate"]` with no `"observed"` (or any other)
   variant possible per the type system -- Requirement 9.7, Correctness
   Property 5.
2. `estimate_ate()` matches a hand-computed value on synthetic
   `PatientLevelEstimate`s.
3. `estimate_cate()` matches hand-computed per-subgroup values, both for
   a `dict` and a callable `subgroup_of`.
4. `build_patient_level_estimate()` correctly extracts `event_hazard`
   from `BranchOutput`s into `y1_hat`/`y0_hat`.

**Validates: Requirements 9.6, 9.7**
"""

import typing

import pytest

from models.aggregation import (
    PatientLevelEstimate,
    build_patient_level_estimate,
    estimate_ate,
    estimate_cate,
)
from models.branch_heads import BranchOutput

# ---------------------------------------------------------------------------
# kind: Literal["model_estimate"] -- no "observed" variant (Requirement 9.7,
# Correctness Property 5)
# ---------------------------------------------------------------------------


def test_kind_field_type_has_no_observed_variant():
    hints = typing.get_type_hints(PatientLevelEstimate, include_extras=True)
    kind_hint = hints["kind"]

    assert typing.get_origin(kind_hint) is typing.Literal
    assert typing.get_args(kind_hint) == ("model_estimate",)
    assert "observed" not in typing.get_args(kind_hint)


def test_patient_level_estimate_constructs_with_model_estimate_kind():
    estimate = PatientLevelEstimate(patient_id="p1", y1_hat=0.3, y0_hat=0.5, kind="model_estimate")
    assert estimate.kind == "model_estimate"


# ---------------------------------------------------------------------------
# estimate_ate() -- hand-computed value (Requirement 9.6)
# ---------------------------------------------------------------------------


def _estimate(patient_id: str, y1_hat: float, y0_hat: float) -> PatientLevelEstimate:
    return PatientLevelEstimate(
        patient_id=patient_id, y1_hat=y1_hat, y0_hat=y0_hat, kind="model_estimate"
    )


def test_estimate_ate_matches_hand_computed_value():
    estimates = [
        _estimate("p1", y1_hat=0.2, y0_hat=0.5),  # diff = -0.3
        _estimate("p2", y1_hat=0.4, y0_hat=0.3),  # diff = +0.1
        _estimate("p3", y1_hat=0.1, y0_hat=0.6),  # diff = -0.5
        _estimate("p4", y1_hat=0.5, y0_hat=0.5),  # diff = 0.0
        _estimate("p5", y1_hat=0.3, y0_hat=0.2),  # diff = +0.1
    ]
    # hand-computed mean: (-0.3 + 0.1 - 0.5 + 0.0 + 0.1) / 5 = -0.6 / 5 = -0.12
    expected_ate = -0.12

    ate = estimate_ate(estimates)

    assert ate == pytest.approx(expected_ate)


def test_estimate_ate_raises_on_empty_sequence():
    try:
        estimate_ate([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty patient_estimates")


# ---------------------------------------------------------------------------
# estimate_cate() -- per-subgroup hand-computed values (Requirement 9.6)
# ---------------------------------------------------------------------------


def test_estimate_cate_matches_hand_computed_values_with_dict_subgroup():
    estimates = [
        _estimate("p1", y1_hat=0.2, y0_hat=0.5),  # diff = -0.3, group A
        _estimate("p2", y1_hat=0.4, y0_hat=0.2),  # diff = +0.2, group A
        _estimate("p3", y1_hat=0.1, y0_hat=0.6),  # diff = -0.5, group B
        _estimate("p4", y1_hat=0.3, y0_hat=0.1),  # diff = +0.2, group B
    ]
    subgroup_of = {"p1": "A", "p2": "A", "p3": "B", "p4": "B"}
    # A: (-0.3 + 0.2) / 2 = -0.05
    # B: (-0.5 + 0.2) / 2 = -0.15

    cate = estimate_cate(estimates, subgroup_of)

    assert cate == {"A": pytest.approx(-0.05), "B": pytest.approx(-0.15)}


def test_estimate_cate_matches_hand_computed_values_with_callable_subgroup():
    estimates = [
        _estimate("p1", y1_hat=0.2, y0_hat=0.5),  # diff = -0.3, group even? p1 -> odd
        _estimate("p2", y1_hat=0.4, y0_hat=0.2),  # diff = +0.2, p2 -> even
        _estimate("p3", y1_hat=0.1, y0_hat=0.6),  # diff = -0.5, p3 -> odd
        _estimate("p4", y1_hat=0.3, y0_hat=0.1),  # diff = +0.2, p4 -> even
    ]

    def subgroup_of(patient_id: str) -> str:
        index = int(patient_id[1:])
        return "even" if index % 2 == 0 else "odd"

    # odd (p1, p3): (-0.3 + -0.5) / 2 = -0.4
    # even (p2, p4): (0.2 + 0.2) / 2 = 0.2

    cate = estimate_cate(estimates, subgroup_of)

    assert cate == {"odd": pytest.approx(-0.4), "even": pytest.approx(0.2)}


# ---------------------------------------------------------------------------
# build_patient_level_estimate() -- extracts event_hazard (Requirement 9.6/9.7
# bridging from task 9.2's BranchOutput)
# ---------------------------------------------------------------------------


def test_build_patient_level_estimate_extracts_event_hazard():
    branch_1_output = BranchOutput(event_hazard=0.35, next_cd4=100.0, next_cd8=200.0, tag="factual")
    branch_0_output = BranchOutput(
        event_hazard=0.55, next_cd4=90.0, next_cd8=190.0, tag="counterfactual"
    )

    estimate = build_patient_level_estimate("patient-42", branch_1_output, branch_0_output)

    assert estimate.patient_id == "patient-42"
    assert estimate.y1_hat == 0.35
    assert estimate.y0_hat == 0.55
    assert estimate.kind == "model_estimate"


def test_build_patient_level_estimate_ignores_tag_ordering():
    """y1_hat/y0_hat always come from branch_1_output/branch_0_output
    respectively, regardless of which one is tagged "factual"."""
    branch_1_output = BranchOutput(
        event_hazard=0.7, next_cd4=1.0, next_cd8=1.0, tag="counterfactual"
    )
    branch_0_output = BranchOutput(event_hazard=0.2, next_cd4=1.0, next_cd8=1.0, tag="factual")

    estimate = build_patient_level_estimate("patient-7", branch_1_output, branch_0_output)

    assert estimate.y1_hat == 0.7
    assert estimate.y0_hat == 0.2
