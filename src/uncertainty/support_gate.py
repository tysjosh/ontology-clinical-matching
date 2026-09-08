"""Patient-support gate (Requirements 10.9, 10.10, 10.11; `decisions.md`
Section 16, `design.md` Section 3.7).

`decisions.md` Section 16 freezes the following boolean (copied verbatim,
not re-derived):

    patient_supported =
        propensity_in_common_support
        AND ensemble_uncertainty <= validation_90th_percentile
        AND protocol_valid
        AND no_hard_clinical_or_temporal_violation

Decoupling from raw patient objects / whole-cohort context
------------------------------------------------------------

Two of the four conditions cannot be computed from a single patient in
isolation:

- `validation_90th_percentile` is a property of the *entire* validation
  set's uncertainty distribution, not of one patient.
- `no_hard_clinical_or_temporal_violation` is naturally sourced from the
  (not-yet-built, as of this task) temporal/clinical witnesses in
  `src/witnesses/`.

Rather than have `PatientSupportGate` reach into a cohort-wide context or
import a not-yet-existing module, it is designed as a **pure decision
function over already-derived per-patient signals**: every value it needs
is passed in directly by the caller, who is responsible for computing (or
looking up) each one from whatever pipeline stage owns it. This mirrors
`causal.overlap.OverlapDiagnostics`'s own approach of accepting plain
values/mappings rather than importing sibling modules.

Condition mapping
--------------------

- `propensity_in_common_support`: `PRIMARY_SUPPORT_LOWER <= propensity <=
  PRIMARY_SUPPORT_UPPER`, reusing the frozen primary common-support band
  constants from `causal.overlap` (`decisions.md` Section 12) rather than
  redefining a second band here.
- `ensemble_uncertainty_ok`: `ensemble_uncertainty <=
  validation_uncertainty_90th_percentile`. "Ensemble uncertainty" is taken
  to be the `variance` half of `DeepEnsemble.predict(x) -> (mean,
  variance)`'s return value (not a transformed value like standard
  deviation); the caller is expected to pass the same statistic
  (`variance`) for both `ensemble_uncertainty` and the percentile it is
  compared against, so the comparison is self-consistent regardless of
  which statistic is chosen upstream.
- `protocol_valid` and `no_hard_violation`: both are caller-supplied
  booleans rather than being computed here. A natural (but not enforced by
  this module) way for a caller to derive `protocol_valid` is "no
  `protocols.consistency_checker.ProtocolViolation` for this patient from
  `ProtocolChecker.check(...)`", and to derive `no_hard_violation` from the
  clinical/temporal witnesses once they exist. Keeping both as plain
  booleans here avoids importing `witnesses/` (not yet built) and keeps
  this gate a pure function of its five scalar/boolean inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from causal.overlap import PRIMARY_SUPPORT_LOWER, PRIMARY_SUPPORT_UPPER


@dataclass(frozen=True)
class SupportDecision:
    """The result of `PatientSupportGate.evaluate()`.

    Attributes:
        supported: `True` only if all four frozen conditions pass.
        reasons: always populated (Requirement 10.10) with exactly one
            descriptive string per condition (four total), each stating
            whether that specific condition passed or failed and why --
            even when `supported=True`, so downstream audit reports can
            show "why," not just "what" (`design.md` Section 3.7).
    """

    supported: bool
    reasons: list[str] = field(default_factory=list)


class PatientSupportGate:
    """Evaluates the frozen four-part `patient_supported` condition.

    Stateless -- `evaluate()` is kept as a method (rather than a bare
    module function) to mirror the `SomeThing.evaluate(...)` naming
    convention already used elsewhere in `src/uncertainty` and
    `src/causal` (e.g. `OverlapDiagnostics.compute()`).
    """

    @staticmethod
    def evaluate(
        propensity: float,
        ensemble_uncertainty: float,
        validation_uncertainty_90th_percentile: float,
        protocol_valid: bool,
        no_hard_violation: bool,
    ) -> SupportDecision:
        """Evaluate the frozen `patient_supported` condition for one patient.

        Args:
            propensity: this patient's estimated propensity score.
            ensemble_uncertainty: this patient's ensemble uncertainty
                (e.g. the `variance` returned by
                `uncertainty.ensemble.DeepEnsemble.predict(x)`).
            validation_uncertainty_90th_percentile: the 90th percentile of
                ensemble uncertainty over the validation set, computed
                elsewhere (this gate does not compute cohort-wide
                statistics itself).
            protocol_valid: `True` if this patient has no protocol
                violation (e.g. no
                `protocols.consistency_checker.ProtocolViolation` from
                `ProtocolChecker.check(...)` for this patient).
            no_hard_violation: `True` if this patient has no hard clinical
                or temporal violation (e.g. from the clinical/temporal
                witnesses).

        Returns:
            A `SupportDecision` with `supported` set to the AND of all
            four conditions, and `reasons` always populated with one
            pass/fail string per condition.
        """
        propensity_in_common_support = (
            PRIMARY_SUPPORT_LOWER <= propensity <= PRIMARY_SUPPORT_UPPER
        )
        ensemble_uncertainty_ok = (
            ensemble_uncertainty <= validation_uncertainty_90th_percentile
        )

        reasons: list[str] = []

        if propensity_in_common_support:
            reasons.append(
                "propensity_in_common_support: pass "
                f"({propensity} in [{PRIMARY_SUPPORT_LOWER}, {PRIMARY_SUPPORT_UPPER}])"
            )
        else:
            reasons.append(
                "propensity_in_common_support: fail "
                f"({propensity} not in [{PRIMARY_SUPPORT_LOWER}, {PRIMARY_SUPPORT_UPPER}])"
            )

        if ensemble_uncertainty_ok:
            reasons.append(
                "ensemble_uncertainty_ok: pass "
                f"({ensemble_uncertainty} <= {validation_uncertainty_90th_percentile})"
            )
        else:
            reasons.append(
                "ensemble_uncertainty_ok: fail "
                f"({ensemble_uncertainty} > {validation_uncertainty_90th_percentile})"
            )

        if protocol_valid:
            reasons.append("protocol_valid: pass")
        else:
            reasons.append("protocol_valid: fail")

        if no_hard_violation:
            reasons.append("no_hard_violation: pass")
        else:
            reasons.append("no_hard_violation: fail")

        supported = (
            propensity_in_common_support
            and ensemble_uncertainty_ok
            and protocol_valid
            and no_hard_violation
        )

        return SupportDecision(supported=supported, reasons=reasons)
