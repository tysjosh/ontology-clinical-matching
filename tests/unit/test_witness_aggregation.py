"""Unit tests for `src/witnesses/aggregation.py`.

Covers:

- `ALL_REQUIRED` policy: blocks (`passed=False`) if any single witness
  result is `"fail"` or `"uncertain"` (tested independently), and returns
  `passed=True` only when all three witnesses are `"pass"`.
- `TWO_OF_THREE` policy: 2-of-3 pass -> `passed=True`; 1-of-3 or 0-of-3
  pass -> `passed=False`.
- `RISK_WEIGHTED` policy: raises `NotImplementedError`.
- Cross-reference check that `infra.config.ExperimentConfig` (the existing
  validator from task 1.1) rejects a `witness_policy="TWO_OF_THREE"`
  config that is not tagged `exploratory=true`, confirming
  `WitnessPolicy`'s string values are interoperable with it.

**Validates: Requirements 11.9; Correctness Property 6**
"""

import pytest
from pydantic import ValidationError

from infra.config import ExperimentConfig
from witnesses.aggregation import WitnessPolicy, aggregate
from witnesses.temporal import WitnessResult


def _result(witness: str, status: str) -> WitnessResult:
    return WitnessResult(witness=witness, status=status, evidence={})  # type: ignore[arg-type]


def _all_pass():
    return [
        _result("temporal", "pass"),
        _result("causal", "pass"),
        _result("clinical", "pass"),
    ]


class TestAllRequiredPolicy:
    def test_all_pass_yields_passed_true(self):
        decision = aggregate(_all_pass(), WitnessPolicy.ALL_REQUIRED)

        assert decision.passed is True
        assert decision.policy is WitnessPolicy.ALL_REQUIRED

    @pytest.mark.parametrize("bad_status", ["fail", "uncertain"])
    def test_any_single_fail_or_uncertain_blocks_release(self, bad_status):
        results = [
            _result("temporal", bad_status),
            _result("causal", "pass"),
            _result("clinical", "pass"),
        ]

        decision = aggregate(results, WitnessPolicy.ALL_REQUIRED)

        assert decision.passed is False
        assert decision.evidence["statuses"]["temporal"] == bad_status

    @pytest.mark.parametrize("bad_status", ["fail", "uncertain"])
    def test_any_of_the_three_witnesses_individually_blocks_release(self, bad_status):
        for index in range(3):
            results = _all_pass()
            witness_name = results[index].witness
            results[index] = _result(witness_name, bad_status)

            decision = aggregate(results, WitnessPolicy.ALL_REQUIRED)

            assert decision.passed is False, (
                f"expected policy to block release when {witness_name}={bad_status}"
            )


class TestTwoOfThreePolicy:
    def test_two_of_three_pass_yields_passed_true(self):
        results = [
            _result("temporal", "pass"),
            _result("causal", "pass"),
            _result("clinical", "fail"),
        ]

        decision = aggregate(results, WitnessPolicy.TWO_OF_THREE)

        assert decision.passed is True

    def test_one_of_three_pass_yields_passed_false(self):
        results = [
            _result("temporal", "pass"),
            _result("causal", "fail"),
            _result("clinical", "uncertain"),
        ]

        decision = aggregate(results, WitnessPolicy.TWO_OF_THREE)

        assert decision.passed is False

    def test_zero_of_three_pass_yields_passed_false(self):
        results = [
            _result("temporal", "fail"),
            _result("causal", "fail"),
            _result("clinical", "uncertain"),
        ]

        decision = aggregate(results, WitnessPolicy.TWO_OF_THREE)

        assert decision.passed is False


class TestRiskWeightedPolicy:
    def test_risk_weighted_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            aggregate(_all_pass(), WitnessPolicy.RISK_WEIGHTED)


class TestWitnessPolicyInteroperableWithExperimentConfig:
    """`WitnessPolicy`'s string values must match `ExperimentConfig`'s
    validated set, and its config-level gating (task 1.1) must reject
    `TWO_OF_THREE` on a non-exploratory config."""

    def test_witness_policy_string_values_match_experiment_config_validation(self):
        for policy in WitnessPolicy:
            config = ExperimentConfig(
                experiment_name="test",
                seed=1,
                exploratory=True,
                witness_policy=policy.value,
            )
            assert config.witness_policy == policy.value

    def test_experiment_config_rejects_two_of_three_when_not_exploratory(self):
        with pytest.raises(ValidationError, match="exploratory"):
            ExperimentConfig(
                experiment_name="test",
                seed=1,
                exploratory=False,
                witness_policy=WitnessPolicy.TWO_OF_THREE.value,
            )

    def test_experiment_config_accepts_two_of_three_when_exploratory(self):
        config = ExperimentConfig(
            experiment_name="test",
            seed=1,
            exploratory=True,
            witness_policy=WitnessPolicy.TWO_OF_THREE.value,
        )
        assert config.witness_policy == "TWO_OF_THREE"
