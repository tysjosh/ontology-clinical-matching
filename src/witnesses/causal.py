"""The causal witness (`decisions.md` Section 16, "Causal witness — pass / fail / uncertain").

Implements the frozen three-way classification exactly:

- **Pass** when all hold: positivity/balance/effective-sample-size
  requirements hold (Section 12, bundled into
  `causal.overlap.OverlapReport.support_valid`); estimates are stable
  under the prespecified propensity and clipping sensitivities
  (Section 11, represented here by the caller-supplied `sensitivity_stable`
  placeholder -- see note below); both negative controls (Section 11.3)
  return `pass`.
- **Fail** when any hard requirement above is violated, or either
  negative control returns `fail`.
- **Uncertain** when no hard requirement is violated but either negative
  control returns `uncertain`.

Placeholder note on `sensitivity_stable`
-----------------------------------------
`decisions.md` Section 16 requires "estimates are stable under the
prespecified propensity and clipping sensitivities" (Section 11's
sensitivity variants: boosted-propensity model, alternative weight
clipping bands). No module in this codebase computes that stability
check yet -- it is task 13.1 (sensitivity-analysis robustness variants),
which does not exist at the time this task was implemented. Rather than
block on machinery that doesn't exist, `CausalWitness.evaluate` accepts a
simple caller-supplied `sensitivity_stable: bool` as a stand-in. The
caller (a later pipeline stage, once task 13.1 exists) is responsible for
computing this correctly -- e.g. by comparing the primary estimate
against each sensitivity variant's estimate under a prespecified
tolerance and reducing to a single boolean. This function does not
attempt to derive that boolean itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from causal.negative_controls import NegativeControlResult
    from causal.overlap import OverlapReport

try:
    # `src/witnesses/temporal.py` is being built by a parallel task and may
    # already define the shared `WitnessResult` type described in
    # `design.md`. Reuse it if present to avoid two divergent definitions.
    from witnesses.temporal import WitnessResult  # type: ignore
except ImportError:
    # `temporal.py` does not exist yet (parallel task still in progress).
    # Define a local copy matching `design.md`'s exact spec. NOTE: this is
    # a follow-up deduplication concern -- once `witnesses.temporal`
    # exists, this local definition should be removed and this module
    # should import from there instead.
    @dataclass(frozen=True)
    class WitnessResult:
        witness: Literal["temporal", "causal", "clinical"]
        status: Literal["pass", "fail", "uncertain"]
        evidence: dict


class CausalWitness:
    """Evaluates the causal witness per `decisions.md` Section 16."""

    @staticmethod
    def evaluate(
        overlap_report: OverlapReport,
        sensitivity_stable: bool,
        negative_control_exposure_result: NegativeControlResult,
        negative_control_outcome_result: NegativeControlResult,
    ) -> WitnessResult:
        """Classifies the causal witness as pass/fail/uncertain.

        Args:
            overlap_report: the result of `causal.overlap.OverlapDiagnostics.compute()`.
                `overlap_report.support_valid` bundles the frozen
                positivity/balance/effective-sample-size/positivity-warning
                conditions from `decisions.md` Section 12.
            sensitivity_stable: caller-supplied placeholder for "estimates
                are stable under the prespecified propensity and clipping
                sensitivities" (Section 11) -- see module docstring.
            negative_control_exposure_result: the sham-treatment
                negative-control result (`causal.negative_controls.run_negative_control_exposure`).
            negative_control_outcome_result: the baseline-CD8
                negative-control result (`causal.negative_controls.run_negative_control_outcome`).

        Returns:
            A `WitnessResult` with `witness="causal"`, the classified
            `status`, and an `evidence` dict containing the underlying
            booleans/classifications for auditability.
        """
        hard_requirement_violated = (not overlap_report.support_valid) or (not sensitivity_stable)
        either_negative_control_fail = (
            negative_control_exposure_result.classification == "fail"
            or negative_control_outcome_result.classification == "fail"
        )
        either_negative_control_uncertain = (
            negative_control_exposure_result.classification == "uncertain"
            or negative_control_outcome_result.classification == "uncertain"
        )

        if hard_requirement_violated or either_negative_control_fail:
            status: Literal["pass", "fail", "uncertain"] = "fail"
        elif either_negative_control_uncertain:
            status = "uncertain"
        else:
            status = "pass"

        evidence = {
            "overlap_support_valid": overlap_report.support_valid,
            "sensitivity_stable": sensitivity_stable,
            "hard_requirement_violated": hard_requirement_violated,
            "negative_control_exposure": negative_control_exposure_result.classification,
            "negative_control_outcome": negative_control_outcome_result.classification,
            "either_negative_control_fail": either_negative_control_fail,
            "either_negative_control_uncertain": either_negative_control_uncertain,
        }

        return WitnessResult(witness="causal", status=status, evidence=evidence)
