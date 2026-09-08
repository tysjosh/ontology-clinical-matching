"""Unit tests for `src/witnesses/causal.py`.

Covers:

- the causal witness fails on insufficient overlap even when both
  negative controls pass (i.e., "factual prediction is accurate" stands
  in for "the negative-control checks look fine");
- the three-way pass/fail/uncertain classification against synthetic
  fixtures for each branch: all-pass -> pass; hard-requirement-violated
  (overlap invalid) -> fail; hard-requirement-violated (sensitivity
  unstable) -> fail; negative-control-fail (overlap otherwise fine) ->
  fail; negative-control-uncertain (overlap otherwise fine, no hard
  violation) -> uncertain.

**Validates: Requirements 11.2, 11.3, 11.4, 11.5**
"""

from causal.negative_controls import NegativeControlResult
from causal.overlap import ArmPropensitySummary, CommonSupportSummary, OverlapReport
from witnesses.causal import CausalWitness


def _overlap_report(*, support_valid: bool) -> OverlapReport:
    """A minimal `OverlapReport`-like fixture with `support_valid` set directly.

    The individual boolean conditions are set consistently with
    `support_valid` for auditability, but `CausalWitness.evaluate` only
    reads `support_valid` per the reuse note in `decisions.md` Section 12.
    """
    arm_summary = ArmPropensitySummary(n=120, minimum=0.4, maximum=0.6, mean=0.5, median=0.5)
    common_support = CommonSupportSummary(
        primary_count=240,
        primary_fraction=1.0 if support_valid else 0.5,
        sensitivity_count=240,
        sensitivity_fraction=1.0,
    )
    return OverlapReport(
        arm_counts={0: 120, 1: 120} if support_valid else {0: 20, 1: 20},
        arm_propensity_summary={0: arm_summary, 1: arm_summary},
        common_support=common_support,
        extreme_weight_count=0 if support_valid else 40,
        extreme_weight_fraction=0.0 if support_valid else 1.0,
        ess_by_arm={0: 110.0, 1: 110.0} if support_valid else {0: 10.0, 1: 10.0},
        smd_by_covariate={"baseline_cd4": 0.02} if support_valid else {"baseline_cd4": 0.5},
        arm_counts_sufficient=support_valid,
        ess_sufficient=support_valid,
        smd_sufficient=support_valid,
        no_unresolved_positivity_warning=support_valid,
        support_valid=support_valid,
    )


def _negative_control(
    *, name: str, classification: str
) -> NegativeControlResult:
    return NegativeControlResult(
        name=name,
        classification=classification,
        point_estimate=0.0,
        ci_low=-0.05,
        ci_high=0.05,
        evidence={},
    )


def _passing_exposure() -> NegativeControlResult:
    return _negative_control(name="negative_control_exposure", classification="pass")


def _passing_outcome() -> NegativeControlResult:
    return _negative_control(name="negative_control_outcome", classification="pass")


class TestFailsOnInsufficientOverlapDespiteAccurateFactualPrediction:
    def test_fails_when_overlap_invalid_even_if_both_negative_controls_pass(self):
        """Insufficient overlap must fail the witness even when both
        negative controls (standing in for accurate factual/pipeline
        behavior) pass -- Requirement 11.3."""
        overlap_report = _overlap_report(support_valid=False)

        result = CausalWitness.evaluate(
            overlap_report=overlap_report,
            sensitivity_stable=True,
            negative_control_exposure_result=_passing_exposure(),
            negative_control_outcome_result=_passing_outcome(),
        )

        assert result.witness == "causal"
        assert result.status == "fail"
        assert result.evidence["overlap_support_valid"] is False
        assert result.evidence["hard_requirement_violated"] is True


class TestThreeWayClassification:
    def test_all_pass_yields_pass(self):
        overlap_report = _overlap_report(support_valid=True)

        result = CausalWitness.evaluate(
            overlap_report=overlap_report,
            sensitivity_stable=True,
            negative_control_exposure_result=_passing_exposure(),
            negative_control_outcome_result=_passing_outcome(),
        )

        assert result.status == "pass"

    def test_overlap_invalid_yields_fail(self):
        overlap_report = _overlap_report(support_valid=False)

        result = CausalWitness.evaluate(
            overlap_report=overlap_report,
            sensitivity_stable=True,
            negative_control_exposure_result=_passing_exposure(),
            negative_control_outcome_result=_passing_outcome(),
        )

        assert result.status == "fail"

    def test_sensitivity_unstable_yields_fail(self):
        """`sensitivity_stable=False` is a hard-requirement violation even
        when overlap and both negative controls are otherwise fine."""
        overlap_report = _overlap_report(support_valid=True)

        result = CausalWitness.evaluate(
            overlap_report=overlap_report,
            sensitivity_stable=False,
            negative_control_exposure_result=_passing_exposure(),
            negative_control_outcome_result=_passing_outcome(),
        )

        assert result.status == "fail"
        assert result.evidence["hard_requirement_violated"] is True

    def test_negative_control_fail_with_overlap_fine_yields_fail(self):
        overlap_report = _overlap_report(support_valid=True)
        failing_exposure = _negative_control(
            name="negative_control_exposure", classification="fail"
        )

        result = CausalWitness.evaluate(
            overlap_report=overlap_report,
            sensitivity_stable=True,
            negative_control_exposure_result=failing_exposure,
            negative_control_outcome_result=_passing_outcome(),
        )

        assert result.status == "fail"
        assert result.evidence["either_negative_control_fail"] is True

    def test_negative_control_uncertain_with_no_hard_violation_yields_uncertain(self):
        overlap_report = _overlap_report(support_valid=True)
        uncertain_outcome = _negative_control(
            name="negative_control_outcome", classification="uncertain"
        )

        result = CausalWitness.evaluate(
            overlap_report=overlap_report,
            sensitivity_stable=True,
            negative_control_exposure_result=_passing_exposure(),
            negative_control_outcome_result=uncertain_outcome,
        )

        assert result.status == "uncertain"
        assert result.evidence["either_negative_control_uncertain"] is True
        assert result.evidence["hard_requirement_violated"] is False

    def test_fail_takes_precedence_over_uncertain(self):
        """If one negative control fails and the other is uncertain, the
        witness must report `fail`, not `uncertain`."""
        overlap_report = _overlap_report(support_valid=True)
        failing_exposure = _negative_control(
            name="negative_control_exposure", classification="fail"
        )
        uncertain_outcome = _negative_control(
            name="negative_control_outcome", classification="uncertain"
        )

        result = CausalWitness.evaluate(
            overlap_report=overlap_report,
            sensitivity_stable=True,
            negative_control_exposure_result=failing_exposure,
            negative_control_outcome_result=uncertain_outcome,
        )

        assert result.status == "fail"
