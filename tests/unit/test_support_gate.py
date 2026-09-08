"""Unit tests for `src/uncertainty/support_gate.py` (Requirements 10.9,
10.10, 10.11).

One test per frozen condition, constructing inputs where exactly one
condition fails (the other three pass), confirming `supported == False`
and that `reasons` contains a fail-indicating string for the failing
condition and pass-indicating strings for the other three. Plus one test
where all four conditions pass.

**Validates: Requirements 10.9, 10.10, 10.11**
"""

from __future__ import annotations

from uncertainty.support_gate import PatientSupportGate, SupportDecision

# A baseline set of inputs where all four conditions pass.
_PASSING_PROPENSITY = 0.5
_PASSING_ENSEMBLE_UNCERTAINTY = 0.2
_PASSING_VALIDATION_90TH_PERCENTILE = 0.4
_PASSING_PROTOCOL_VALID = True
_PASSING_NO_HARD_VIOLATION = True


def _fail_reasons(reasons: list[str]) -> list[str]:
    return [reason for reason in reasons if ": fail" in reason]


def _pass_reasons(reasons: list[str]) -> list[str]:
    return [reason for reason in reasons if ": pass" in reason]


def test_all_four_conditions_pass_supported_true():
    decision = PatientSupportGate.evaluate(
        propensity=_PASSING_PROPENSITY,
        ensemble_uncertainty=_PASSING_ENSEMBLE_UNCERTAINTY,
        validation_uncertainty_90th_percentile=_PASSING_VALIDATION_90TH_PERCENTILE,
        protocol_valid=_PASSING_PROTOCOL_VALID,
        no_hard_violation=_PASSING_NO_HARD_VIOLATION,
    )

    assert isinstance(decision, SupportDecision)
    assert decision.supported is True
    assert len(decision.reasons) == 4
    assert len(_fail_reasons(decision.reasons)) == 0
    assert len(_pass_reasons(decision.reasons)) == 4


def test_propensity_outside_common_support_fails_alone():
    decision = PatientSupportGate.evaluate(
        propensity=0.98,  # outside [0.10, 0.90]
        ensemble_uncertainty=_PASSING_ENSEMBLE_UNCERTAINTY,
        validation_uncertainty_90th_percentile=_PASSING_VALIDATION_90TH_PERCENTILE,
        protocol_valid=_PASSING_PROTOCOL_VALID,
        no_hard_violation=_PASSING_NO_HARD_VIOLATION,
    )

    assert decision.supported is False
    fail_reasons = _fail_reasons(decision.reasons)
    assert len(fail_reasons) == 1
    assert "propensity_in_common_support" in fail_reasons[0]
    pass_reasons = _pass_reasons(decision.reasons)
    assert len(pass_reasons) == 3
    assert any("ensemble_uncertainty_ok" in reason for reason in pass_reasons)
    assert any("protocol_valid" in reason for reason in pass_reasons)
    assert any("no_hard_violation" in reason for reason in pass_reasons)


def test_ensemble_uncertainty_above_validation_percentile_fails_alone():
    decision = PatientSupportGate.evaluate(
        propensity=_PASSING_PROPENSITY,
        ensemble_uncertainty=0.9,  # above the validation 90th percentile
        validation_uncertainty_90th_percentile=_PASSING_VALIDATION_90TH_PERCENTILE,
        protocol_valid=_PASSING_PROTOCOL_VALID,
        no_hard_violation=_PASSING_NO_HARD_VIOLATION,
    )

    assert decision.supported is False
    fail_reasons = _fail_reasons(decision.reasons)
    assert len(fail_reasons) == 1
    assert "ensemble_uncertainty_ok" in fail_reasons[0]
    pass_reasons = _pass_reasons(decision.reasons)
    assert len(pass_reasons) == 3
    assert any("propensity_in_common_support" in reason for reason in pass_reasons)
    assert any("protocol_valid" in reason for reason in pass_reasons)
    assert any("no_hard_violation" in reason for reason in pass_reasons)


def test_protocol_invalid_fails_alone():
    decision = PatientSupportGate.evaluate(
        propensity=_PASSING_PROPENSITY,
        ensemble_uncertainty=_PASSING_ENSEMBLE_UNCERTAINTY,
        validation_uncertainty_90th_percentile=_PASSING_VALIDATION_90TH_PERCENTILE,
        protocol_valid=False,
        no_hard_violation=_PASSING_NO_HARD_VIOLATION,
    )

    assert decision.supported is False
    fail_reasons = _fail_reasons(decision.reasons)
    assert len(fail_reasons) == 1
    assert "protocol_valid" in fail_reasons[0]
    pass_reasons = _pass_reasons(decision.reasons)
    assert len(pass_reasons) == 3
    assert any("propensity_in_common_support" in reason for reason in pass_reasons)
    assert any("ensemble_uncertainty_ok" in reason for reason in pass_reasons)
    assert any("no_hard_violation" in reason for reason in pass_reasons)


def test_hard_violation_present_fails_alone():
    decision = PatientSupportGate.evaluate(
        propensity=_PASSING_PROPENSITY,
        ensemble_uncertainty=_PASSING_ENSEMBLE_UNCERTAINTY,
        validation_uncertainty_90th_percentile=_PASSING_VALIDATION_90TH_PERCENTILE,
        protocol_valid=_PASSING_PROTOCOL_VALID,
        no_hard_violation=False,
    )

    assert decision.supported is False
    fail_reasons = _fail_reasons(decision.reasons)
    assert len(fail_reasons) == 1
    assert "no_hard_violation" in fail_reasons[0]
    pass_reasons = _pass_reasons(decision.reasons)
    assert len(pass_reasons) == 3
    assert any("propensity_in_common_support" in reason for reason in pass_reasons)
    assert any("ensemble_uncertainty_ok" in reason for reason in pass_reasons)
    assert any("protocol_valid" in reason for reason in pass_reasons)
