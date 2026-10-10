"""Audit report (Requirement 11.10; `design.md` Section 3.8).

`design.md` Section 3.8 specifies:

    `audit_report.py`: `build_audit_report(protocol, population, treatment,
    estimate, overlap, witness_results, abstentions, violations,
    provenance) -> AuditReport`, serialized as the per-claim JSON described
    in Requirement 11.10.

Requirement 11.10 freezes the report's required content (copied verbatim,
not re-derived):

    Every released cohort estimate SHALL receive an audit report containing
    protocol version, population, treatment, estimate and interval, overlap
    diagnostics, witness outcomes, support restrictions, violations, and
    provenance.

Design choices
--------------

- **Pure bundling, not computation.** `build_audit_report` does not compute
  anything -- every input (the protocol spec, the bootstrap CI, the overlap
  report, the witness results, the support decisions, the protocol
  violations) is already built by an earlier pipeline stage. This module's
  only job is to reshape those already-computed artifacts into one flat,
  JSON-serializable structure meant to be written to
  `results/<experiment_id>/reporting/audit_reports/` (`design.md` Section
  2.3).
- **Accept real objects, not re-typed dicts.** Since every input is already
  constructed by the time an audit report is generated, `build_audit_report`
  accepts the actual dataclass/Pydantic types (`TargetTrialSpec`,
  `BootstrapCI`, `OverlapReport`, `WitnessResult`, `SupportDecision`,
  `ProtocolViolation`) rather than requiring the caller to pre-flatten them
  into dicts. `population` and `treatment` have no dedicated existing type
  in this codebase (they are descriptive, caller-assembled summaries, not a
  structured model), so they are accepted as plain `dict`s.
- **`abstentions` accepts either `SupportDecision`s or plain strings.**
  Requirement 11.10 calls for "support restrictions" (i.e. abstention
  reasons). The natural per-patient source is
  `uncertainty.support_gate.SupportDecision.reasons`, but a caller may also
  have already reduced abstentions to a flat list of reason strings (e.g. a
  cohort-level summary). Both are accepted so callers are not forced to
  wrap already-flattened reasons back into fake `SupportDecision`s.
- **`AuditReport`'s own shape is a flat, independent dataclass.** Per
  `design.md`'s guidance for this task, the report's *output* structure is
  its own simple, JSON-serializable dataclass (with a `.to_dict()` method)
  rather than a nested bag of the input types -- the report is written to
  disk and read by humans/downstream tooling, so it should not require
  knowledge of every upstream module's internal representation to consume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from causal.overlap import OverlapReport
from protocols.consistency_checker import ProtocolViolation
from protocols.schema import TargetTrialSpec
from uncertainty.bootstrap import BootstrapCI
from uncertainty.support_gate import SupportDecision
from witnesses.temporal import WitnessResult


def _violation_to_dict(violation: ProtocolViolation) -> dict:
    """Flatten one `ProtocolViolation` into a JSON-serializable dict."""
    return {
        "patient_id": violation.patient_id,
        "rule": violation.rule,
        "timestamp": violation.timestamp.isoformat() if violation.timestamp else None,
        "evidence": violation.evidence,
    }


def _abstention_reasons(
    abstentions: Sequence[SupportDecision] | Sequence[str],
) -> list[str]:
    """Flatten `abstentions` (either `SupportDecision`s or plain strings) into
    a single flat list of reason strings.

    Only the reasons associated with an *unsupported* `SupportDecision` are
    included -- a passing `SupportDecision`'s reasons describe why a patient
    *was* supported, which is not a "support restriction" for the purposes
    of this report. Plain strings are passed through unchanged (the caller
    has already decided they are abstention reasons).
    """
    reasons: list[str] = []
    for abstention in abstentions:
        if isinstance(abstention, SupportDecision):
            if not abstention.supported:
                reasons.extend(abstention.reasons)
        else:
            reasons.append(abstention)
    return reasons


@dataclass(frozen=True)
class AuditReport:
    """The audit report required by Requirement 11.10 for every released
    cohort estimate.

    Attributes:
        protocol_version: the locked `TargetTrialSpec.version`.
        population: a description of the (supported) target population
            this estimate pertains to.
        treatment: a description of the treatment/comparator contrast.
        estimate: the point estimate (`BootstrapCI.point_estimate`).
        estimate_ci_low: the lower confidence bound (`BootstrapCI.ci_low`).
        estimate_ci_high: the upper confidence bound (`BootstrapCI.ci_high`).
        overlap_diagnostics: a flattened summary of the `OverlapReport`.
        witness_outcomes: `{witness name: status}`, e.g.
            `{"temporal": "pass", "causal": "fail", "clinical": "pass"}`.
        abstention_reasons: the flattened support-restriction reasons for
            every unsupported patient (or caller-supplied reason strings).
        violations: one flattened dict per `ProtocolViolation`.
        provenance: caller-supplied provenance (e.g. `experiment_id`,
            `seed`, `commit`, `timestamp`).
    """

    protocol_version: str
    population: dict
    treatment: dict
    estimate: float
    estimate_ci_low: float
    estimate_ci_high: float
    overlap_diagnostics: dict
    witness_outcomes: dict[str, str]
    abstention_reasons: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable `dict` of this report."""
        return asdict(self)


def build_audit_report(
    protocol: TargetTrialSpec,
    population: dict,
    treatment: dict,
    estimate: BootstrapCI,
    overlap: OverlapReport,
    witness_results: Sequence[WitnessResult],
    abstentions: Sequence[SupportDecision] | Sequence[str],
    violations: Sequence[ProtocolViolation],
    provenance: dict,
) -> AuditReport:
    """Bundle already-computed artifacts from every pipeline stage into one
    flat `AuditReport` (Requirement 11.10).

    Args:
        protocol: the locked target-trial specification; only `.version` is
            surfaced directly (the full spec is not duplicated into the
            report -- it is identified by version instead).
        population: a caller-assembled description of the (supported)
            target population this estimate pertains to.
        treatment: a caller-assembled description of the treatment and
            comparator.
        estimate: the cohort-level treatment-effect estimate and interval
            (`uncertainty.bootstrap.BootstrapCI`).
        overlap: the overlap/balance diagnostics
            (`causal.overlap.OverlapReport`).
        witness_results: the temporal/causal/clinical witnesses' results.
        abstentions: either per-patient `SupportDecision`s or an
            already-flattened list of abstention/support-restriction reason
            strings.
        violations: the protocol/clinical-constraint violations found for
            this cohort (`protocols.consistency_checker.ProtocolViolation`).
        provenance: e.g. `{"experiment_id": ..., "seed": ..., "commit": ...,
            "timestamp": ...}`.

    Returns:
        An `AuditReport` containing every field required by Requirement
        11.10.
    """
    overlap_diagnostics = {
        "arm_counts": dict(overlap.arm_counts),
        "common_support": {
            "primary_count": overlap.common_support.primary_count,
            "primary_fraction": overlap.common_support.primary_fraction,
            "sensitivity_count": overlap.common_support.sensitivity_count,
            "sensitivity_fraction": overlap.common_support.sensitivity_fraction,
        },
        "extreme_weight_count": overlap.extreme_weight_count,
        "extreme_weight_fraction": overlap.extreme_weight_fraction,
        "ess_by_arm": dict(overlap.ess_by_arm),
        "smd_by_covariate": dict(overlap.smd_by_covariate),
        "smd_basis": overlap.smd_basis,
        "raw_smd_by_covariate_descriptive_only": dict(overlap.raw_smd_by_covariate),
        "smd_exclusions": dict(overlap.smd_exclusions),
        "arm_counts_sufficient": overlap.arm_counts_sufficient,
        "ess_sufficient": overlap.ess_sufficient,
        "smd_sufficient": overlap.smd_sufficient,
        "no_unresolved_positivity_warning": overlap.no_unresolved_positivity_warning,
        "support_valid": overlap.support_valid,
    }

    witness_outcomes = {result.witness: result.status for result in witness_results}

    return AuditReport(
        protocol_version=protocol.version,
        population=dict(population),
        treatment=dict(treatment),
        estimate=estimate.point_estimate,
        estimate_ci_low=estimate.ci_low,
        estimate_ci_high=estimate.ci_high,
        overlap_diagnostics=overlap_diagnostics,
        witness_outcomes=witness_outcomes,
        abstention_reasons=_abstention_reasons(abstentions),
        violations=[_violation_to_dict(violation) for violation in violations],
        provenance=dict(provenance),
    )
