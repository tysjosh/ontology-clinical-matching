"""Unit tests for `src/uncertainty/release_gate.py`.

Covers:

- release is blocked if any of the six conditions is false/not-pass
  (tested independently for each of the six);
- release is granted only when all six conditions hold;
- `reasons` is always populated with exactly six entries.

**Validates: Requirements 11.9; Correctness Property 6**
"""

from uncertainty.release_gate import CohortReleaseGate
from witnesses.temporal import WitnessResult


def _witness_result(witness: str, status: str) -> WitnessResult:
    return WitnessResult(witness=witness, status=status, evidence={})  # type: ignore[arg-type]


def _all_pass_kwargs() -> dict:
    return {
        "overlap_valid": True,
        "balance_valid": True,
        "uncertainty_valid": True,
        "temporal_result": _witness_result("temporal", "pass"),
        "causal_result": _witness_result("causal", "pass"),
        "clinical_result": _witness_result("clinical", "pass"),
    }


class TestReleaseGrantedWhenAllSixHold:
    def test_all_six_conditions_true_grants_release(self):
        decision = CohortReleaseGate.evaluate(**_all_pass_kwargs())

        assert decision.release is True
        assert len(decision.reasons) == 6


class TestReleaseBlockedWhenAnyConditionFails:
    def test_overlap_invalid_blocks_release(self):
        kwargs = _all_pass_kwargs()
        kwargs["overlap_valid"] = False

        decision = CohortReleaseGate.evaluate(**kwargs)

        assert decision.release is False
        assert len(decision.reasons) == 6

    def test_balance_invalid_blocks_release(self):
        kwargs = _all_pass_kwargs()
        kwargs["balance_valid"] = False

        decision = CohortReleaseGate.evaluate(**kwargs)

        assert decision.release is False
        assert len(decision.reasons) == 6

    def test_uncertainty_invalid_blocks_release(self):
        kwargs = _all_pass_kwargs()
        kwargs["uncertainty_valid"] = False

        decision = CohortReleaseGate.evaluate(**kwargs)

        assert decision.release is False
        assert len(decision.reasons) == 6

    def test_temporal_witness_not_pass_blocks_release(self):
        for bad_status in ("fail", "uncertain"):
            kwargs = _all_pass_kwargs()
            kwargs["temporal_result"] = _witness_result("temporal", bad_status)

            decision = CohortReleaseGate.evaluate(**kwargs)

            assert decision.release is False
            assert len(decision.reasons) == 6

    def test_causal_witness_not_pass_blocks_release(self):
        for bad_status in ("fail", "uncertain"):
            kwargs = _all_pass_kwargs()
            kwargs["causal_result"] = _witness_result("causal", bad_status)

            decision = CohortReleaseGate.evaluate(**kwargs)

            assert decision.release is False
            assert len(decision.reasons) == 6

    def test_clinical_witness_not_pass_blocks_release(self):
        for bad_status in ("fail", "uncertain"):
            kwargs = _all_pass_kwargs()
            kwargs["clinical_result"] = _witness_result("clinical", bad_status)

            decision = CohortReleaseGate.evaluate(**kwargs)

            assert decision.release is False
            assert len(decision.reasons) == 6


class TestReasonsAlwaysPopulated:
    def test_reasons_has_six_entries_even_when_release_true(self):
        decision = CohortReleaseGate.evaluate(**_all_pass_kwargs())

        assert len(decision.reasons) == 6
        assert all(isinstance(reason, str) and reason for reason in decision.reasons)

    def test_reasons_has_six_entries_when_all_conditions_fail(self):
        kwargs = {
            "overlap_valid": False,
            "balance_valid": False,
            "uncertainty_valid": False,
            "temporal_result": _witness_result("temporal", "fail"),
            "causal_result": _witness_result("causal", "fail"),
            "clinical_result": _witness_result("clinical", "fail"),
        }

        decision = CohortReleaseGate.evaluate(**kwargs)

        assert decision.release is False
        assert len(decision.reasons) == 6
