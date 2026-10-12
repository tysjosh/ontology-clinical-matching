"""Confirms `reporting.tables.build_blocked_outcome_report` accepts a real
`evaluation.release_gate.ReleaseDecision`'s own fields directly (task
22.2; Requirement 19.1), not just a hand-built dict -- i.e. the thin
adapter shape (`released`, `failing_conditions`) matches
`ReleaseDecision.released`/`.reasons` exactly.

**Validates: Requirements 19.1**
"""

from __future__ import annotations

from evaluation.release_gate import ReleaseDecision
from reporting.tables import build_blocked_outcome_report


def test_build_blocked_outcome_report_from_a_real_blocked_release_decision():
    decision = ReleaseDecision(
        released=False,
        overlap_valid=True,
        balance_valid=False,
        uncertainty_valid=True,
        temporal_pass=True,
        causal_pass=False,
        clinical_pass=True,
        reasons=(
            "balance_valid: fail (OverlapReport.smd_sufficient is False)",
            "causal_pass: fail (status='uncertain')",
        ),
    )

    report = build_blocked_outcome_report(
        released=decision.released,
        failing_conditions=decision.reasons,
        supported_count=100,
        abstained_count=0,
    )

    assert report["blocked"] is True
    assert any("balance_valid" in reason for reason in report["failing_conditions"])
    assert any("causal_pass" in reason for reason in report["failing_conditions"])
    assert report["supported_count"] == 100
    assert report["abstained_count"] == 0


def test_build_blocked_outcome_report_rejects_a_passing_release_decision():
    decision = ReleaseDecision(
        released=True,
        overlap_valid=True,
        balance_valid=True,
        uncertainty_valid=True,
        temporal_pass=True,
        causal_pass=True,
        clinical_pass=True,
        reasons=(),
    )

    import pytest

    with pytest.raises(ValueError):
        build_blocked_outcome_report(
            released=decision.released,
            failing_conditions=decision.reasons,
            supported_count=100,
            abstained_count=0,
        )
