"""Unit tests for `evaluation.release_gate.evaluate_release` (task 16;
Requirement 11.1-11.6; `decisions.md` Section 16).

Builds a minimal synthetic `OverlapReport` directly (its support/balance
booleans are the only fields `evaluate_release` reads) rather than
constructing a full cohort, since this module's whole job is the pure
six-way conjunction, not overlap computation itself.

**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6**
"""

from __future__ import annotations

from causal.overlap import OverlapReport
from evaluation.release_gate import ReleaseDecision, evaluate_release
from witnesses.temporal import WitnessResult


def _overlap(support_valid: bool = True, smd_sufficient: bool = True) -> OverlapReport:
    return OverlapReport(
        arm_counts={0: 150, 1: 150},
        arm_propensity_summary={},
        common_support={},
        extreme_weight_count=0,
        extreme_weight_fraction=0.0,
        ess_by_arm={0: 140.0, 1: 140.0},
        smd_by_covariate={},
        arm_counts_sufficient=True,
        ess_sufficient=True,
        smd_sufficient=smd_sufficient,
        no_unresolved_positivity_warning=True,
        support_valid=support_valid,
    )


def _witness(name: str, status: str) -> WitnessResult:
    return WitnessResult(witness=name, status=status, evidence={})


ALL_PASS_WITNESSES = (
    _witness("temporal", "pass"),
    _witness("causal", "pass"),
    _witness("clinical", "pass"),
)


def test_all_six_conditions_passing_releases():
    decision = evaluate_release(_overlap(), ALL_PASS_WITNESSES, uncertainty_valid=True)

    assert decision.released is True
    assert decision.overlap_valid is True
    assert decision.balance_valid is True
    assert decision.uncertainty_valid is True
    assert decision.temporal_pass is True
    assert decision.causal_pass is True
    assert decision.clinical_pass is True
    assert decision.reasons == ()


def test_overlap_invalid_blocks_release_and_is_named():
    decision = evaluate_release(
        _overlap(support_valid=False), ALL_PASS_WITNESSES, uncertainty_valid=True
    )

    assert decision.released is False
    assert decision.overlap_valid is False
    assert any("overlap_valid" in reason for reason in decision.reasons)


def test_balance_invalid_blocks_release_and_is_named():
    decision = evaluate_release(
        _overlap(smd_sufficient=False), ALL_PASS_WITNESSES, uncertainty_valid=True
    )

    assert decision.released is False
    assert decision.balance_valid is False
    assert any("balance_valid" in reason for reason in decision.reasons)


def test_uncertainty_invalid_blocks_release_and_is_named():
    decision = evaluate_release(_overlap(), ALL_PASS_WITNESSES, uncertainty_valid=False)

    assert decision.released is False
    assert decision.uncertainty_valid is False
    assert any("uncertainty_valid" in reason for reason in decision.reasons)


def test_a_causal_witness_uncertain_is_not_a_pass_and_blocks_release():
    """Requirement 11.3: 'uncertain' is explicitly not a pass."""
    witnesses = (
        _witness("temporal", "pass"),
        _witness("causal", "uncertain"),
        _witness("clinical", "pass"),
    )
    decision = evaluate_release(_overlap(), witnesses, uncertainty_valid=True)

    assert decision.released is False
    assert decision.causal_pass is False
    assert any("causal_pass" in reason for reason in decision.reasons)


def test_a_causal_witness_fail_blocks_release_identically_to_uncertain():
    witnesses_uncertain = (
        _witness("temporal", "pass"),
        _witness("causal", "uncertain"),
        _witness("clinical", "pass"),
    )
    witnesses_fail = (
        _witness("temporal", "pass"),
        _witness("causal", "fail"),
        _witness("clinical", "pass"),
    )

    decision_uncertain = evaluate_release(_overlap(), witnesses_uncertain, uncertainty_valid=True)
    decision_fail = evaluate_release(_overlap(), witnesses_fail, uncertainty_valid=True)

    assert decision_uncertain.released is False
    assert decision_fail.released is False
    assert decision_uncertain.causal_pass is False
    assert decision_fail.causal_pass is False


def test_a_missing_witness_is_treated_as_not_passing():
    """A caller that forgets to supply, say, the clinical witness must not
    be treated as though it silently passed."""
    witnesses = (
        _witness("temporal", "pass"),
        _witness("causal", "pass"),
        # No "clinical" entry at all.
    )
    decision = evaluate_release(_overlap(), witnesses, uncertainty_valid=True)

    assert decision.released is False
    assert decision.clinical_pass is False


def test_every_condition_status_is_always_populated():
    """Requirement 11.2: all six condition statuses are always returned,
    even on a wholesale failure."""
    decision = evaluate_release(
        _overlap(support_valid=False, smd_sufficient=False), (), uncertainty_valid=False
    )

    assert isinstance(decision, ReleaseDecision)
    assert decision.overlap_valid is False
    assert decision.balance_valid is False
    assert decision.uncertainty_valid is False
    assert decision.temporal_pass is False
    assert decision.causal_pass is False
    assert decision.clinical_pass is False


def test_every_failing_condition_is_enumerated_in_reasons():
    """Requirement 11.5: every failing condition, not just the first."""
    decision = evaluate_release(
        _overlap(support_valid=False, smd_sufficient=False), (), uncertainty_valid=False
    )

    assert len(decision.reasons) == 6


def test_multiple_witnesses_with_the_same_name_use_the_first_match():
    """Robustness: evaluate_release does not error on a malformed
    witness_results sequence carrying a duplicate witness name; it uses
    the first match it finds."""
    witnesses = (
        _witness("temporal", "pass"),
        _witness("temporal", "fail"),
        _witness("causal", "pass"),
        _witness("clinical", "pass"),
    )
    decision = evaluate_release(_overlap(), witnesses, uncertainty_valid=True)

    assert decision.temporal_pass is True
