"""The clinical(-constraint) witness (`decisions.md` Section 16, "Clinical
witness -- passes when"; Requirement 11.6-11.8).

`decisions.md` Section 16 freezes the clinical witness's passing condition
as the conjunction of six checks:

    - eligibility is satisfied (Section 4);
    - treatment assignment is permitted;
    - dose/treatment identity is valid;
    - event timing is possible;
    - endpoint construction follows the protocol (Section 6);
    - zero hard protocol violations remain.

`src/protocols/consistency_checker.py::ProtocolChecker.check()` already
implements exactly these six categories as a flat list of
`ProtocolViolation(patient_id, rule, timestamp, evidence)` records (see that
module's docstring for the `rule` values each check produces:
``"ineligible"`` (eligibility), ``"invalid_assignment"`` (treatment
assignment permitted / dose-treatment identity), ``"treatment_before_eligibility"``,
``"time_zero_violation"``, ``"outcome_before_intervention"``,
``"impossible_sequence"`` (event timing possible), and
``"endpoint_window_violation"`` (endpoint construction follows the
protocol)). This witness is a thin wrapper around that checker: it does not
duplicate any of the six checks.

Pass/fail interpretation
-------------------------
`decisions.md` does not describe an "uncertain" branch for this witness the
way it does for the causal witness (Section 16, "Causal witness -- pass /
fail / uncertain"). "Zero hard protocol violations remain" reads as a plain
pass/fail rule, so this module treats any non-empty violation list as
``"fail"`` and an empty list as ``"pass"``. There is no ``"uncertain"``
status for this witness.
"""

from __future__ import annotations

from typing import Literal

from protocols.consistency_checker import PatientProtocolRecord, ProtocolChecker
from protocols.schema import TargetTrialSpec
from witnesses.temporal import WitnessResult


class ClinicalWitness:
    """Evaluates the clinical(-constraint) witness per `decisions.md` Section 16.

    Stateless -- `evaluate()` is a `@staticmethod` to mirror the
    `SomeThing.evaluate(...)` convention used elsewhere (e.g.
    `witnesses.causal.CausalWitness.evaluate()`,
    `witnesses.temporal.TemporalWitness.evaluate()`).
    """

    @staticmethod
    def evaluate(
        cohort: list[PatientProtocolRecord], spec: TargetTrialSpec
    ) -> WitnessResult:
        """Classifies the clinical witness as pass/fail.

        Runs `ProtocolChecker().check(cohort, spec)` -- the same six
        eligibility/assignment/dose-identity/event-timing/endpoint-
        construction checks this witness is defined over -- and passes iff
        zero violations are returned.

        Args:
            cohort: the patients to check, one `PatientProtocolRecord` per
                patient (see `consistency_checker.py` for its shape).
            spec: the target-trial specification, forwarded to
                `ProtocolChecker.check()` unchanged.

        Returns:
            A `WitnessResult` with `witness="clinical"`, `status="pass"` if
            no violations were found or `status="fail"` otherwise, and
            `evidence` containing the total violation count, violations
            grouped by rule category (so the audit trail shows exactly
            which category -- eligibility, assignment, dose/treatment
            identity, event timing, or endpoint construction -- contributed
            violations), and the full flat violation list.
        """
        violations = ProtocolChecker.check(cohort, spec)

        violations_by_rule: dict[str, int] = {}
        for violation in violations:
            violations_by_rule[violation.rule] = violations_by_rule.get(violation.rule, 0) + 1

        status: Literal["pass", "fail", "uncertain"] = "fail" if violations else "pass"

        evidence = {
            "violation_count": len(violations),
            "violations_by_rule": violations_by_rule,
            "violations": [
                {
                    "patient_id": violation.patient_id,
                    "rule": violation.rule,
                    "timestamp": violation.timestamp.isoformat()
                    if violation.timestamp is not None
                    else None,
                    "evidence": violation.evidence,
                }
                for violation in violations
            ],
        }

        return WitnessResult(witness="clinical", status=status, evidence=evidence)
