"""Cohort-level release gate (`decisions.md` Section 16, "Cohort-level
release condition"; Requirement 11.9, Correctness Property 6).

`decisions.md` Section 16 freezes the following (copied verbatim, not
re-derived):

    release = overlap_valid AND balance_valid AND uncertainty_valid
              AND temporal_witness==pass AND causal_witness==pass
              AND clinical_witness==pass

Mapping `overlap_valid` / `balance_valid` onto `OverlapReport`
-----------------------------------------------------------------

`causal.overlap.OverlapReport` (task 7.1) does not expose separate
`overlap_valid` / `balance_valid` booleans; per `decisions.md` Section 12
those two concepts are bundled into a single `support_valid` field
alongside two other conditions (`arm_counts_sufficient`, `ess_sufficient`,
`smd_sufficient`, `no_unresolved_positivity_warning`). `OverlapReport` does
expose all four individual booleans, so this module decomposes them along
the most natural, defensible split rather than treating `support_valid` as
an undifferentiated blob:

    - `overlap_valid`  = `arm_counts_sufficient AND ess_sufficient AND
                          no_unresolved_positivity_warning`
                          (i.e. "is there overlap/positivity" -- arm size,
                          effective sample size, and no extreme-weight/
                          positivity warning are all about the *support*
                          of the propensity distribution, not covariate
                          balance).
    - `balance_valid`  = `smd_sufficient`
                          (i.e. "are the arms balanced" -- the standardized
                          mean difference check is specifically a
                          covariate-*balance* condition).

This mapping is a decomposition of `decisions.md` Section 12's bundled
`support_valid`, not a new rule: `overlap_valid AND balance_valid` is
logically equivalent to `support_valid` (both are the conjunction of the
same four booleans), so this gate's behavior is identical whether the
caller passes decomposed values or `support_valid` itself for both. This
module accepts the two as separate `bool` parameters (rather than a single
`OverlapReport`) to mirror `decisions.md`'s two-name convention and to
keep this module decoupled from `causal.overlap`, consistent with
`PatientSupportGate`'s and `CausalWitness`'s own caller-supplies-signals
design (see `src/uncertainty/support_gate.py`, `src/witnesses/causal.py`).
A caller holding an `OverlapReport` passes
`overlap_valid=(report.arm_counts_sufficient and report.ess_sufficient and
report.no_unresolved_positivity_warning)` and
`balance_valid=report.smd_sufficient`.

`uncertainty_valid` is likewise a caller-supplied `bool` -- `decisions.md`
Section 16 does not define a single frozen formula for cohort-level
"uncertainty valid" the way it does for the patient-level
`patient_supported` condition (`uncertainty.support_gate.PatientSupportGate`),
so this gate does not attempt to derive it and instead accepts it directly,
consistent with the "decoupled inputs, not raw recomputation" pattern
established by `PatientSupportGate` and `CausalWitness`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from witnesses.temporal import WitnessResult


@dataclass(frozen=True)
class ReleaseDecision:
    """The result of `CohortReleaseGate.evaluate()`.

    Attributes:
        release: `True` only if all six frozen conditions hold.
        reasons: always populated (mirrors `SupportDecision.reasons` /
            `WitnessResult.evidence` always being populated) with exactly
            one descriptive pass/fail string per condition (six total),
            regardless of whether `release` is `True` or `False`.
    """

    release: bool
    reasons: list[str] = field(default_factory=list)


class CohortReleaseGate:
    """Evaluates the frozen six-part cohort-level `release` condition.

    Stateless -- `evaluate()` is kept as a `staticmethod` to mirror the
    `SomeThing.evaluate(...)` naming convention already used elsewhere in
    `src/uncertainty` and `src/witnesses` (e.g.
    `PatientSupportGate.evaluate()`, `CausalWitness.evaluate()`).
    """

    @staticmethod
    def evaluate(
        overlap_valid: bool,
        balance_valid: bool,
        uncertainty_valid: bool,
        temporal_result: WitnessResult,
        causal_result: WitnessResult,
        clinical_result: WitnessResult,
    ) -> ReleaseDecision:
        """Evaluate the frozen cohort-level `release` condition.

        Args:
            overlap_valid: `arm_counts_sufficient AND ess_sufficient AND
                no_unresolved_positivity_warning` from a
                `causal.overlap.OverlapReport` (see module docstring for
                the decomposition rationale).
            balance_valid: `smd_sufficient` from the same `OverlapReport`.
            uncertainty_valid: caller-supplied cohort-level uncertainty
                validity (see module docstring -- no single frozen formula
                exists for this at the cohort level).
            temporal_result: the temporal witness's `WitnessResult`
                (`witnesses.temporal.TemporalWitness.evaluate()`).
            causal_result: the causal witness's `WitnessResult`
                (`witnesses.causal.CausalWitness.evaluate()`).
            clinical_result: the clinical witness's `WitnessResult`
                (`witnesses.clinical.ClinicalWitness.evaluate()`).

        Returns:
            A `ReleaseDecision` with `release` set to the AND of all six
            conditions, and `reasons` always populated with one pass/fail
            string per condition (six total).
        """
        temporal_pass = temporal_result.status == "pass"
        causal_pass = causal_result.status == "pass"
        clinical_pass = clinical_result.status == "pass"

        reasons: list[str] = []

        reasons.append(
            f"overlap_valid: {'pass' if overlap_valid else 'fail'} ({overlap_valid})"
        )
        reasons.append(
            f"balance_valid: {'pass' if balance_valid else 'fail'} ({balance_valid})"
        )
        reasons.append(
            f"uncertainty_valid: {'pass' if uncertainty_valid else 'fail'} ({uncertainty_valid})"
        )
        reasons.append(
            f"temporal_witness: {temporal_result.status} "
            f"({'pass' if temporal_pass else 'fail'})"
        )
        reasons.append(
            f"causal_witness: {causal_result.status} "
            f"({'pass' if causal_pass else 'fail'})"
        )
        reasons.append(
            f"clinical_witness: {clinical_result.status} "
            f"({'pass' if clinical_pass else 'fail'})"
        )

        release = (
            overlap_valid
            and balance_valid
            and uncertainty_valid
            and temporal_pass
            and causal_pass
            and clinical_pass
        )

        return ReleaseDecision(release=release, reasons=reasons)
