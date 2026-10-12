"""Audit report (Requirement 11.10; `design.md` Section 3.8; extended by
task 17 for Requirement 12 / `design.md` Section 3.10).

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

Task 17 extension (Requirement 12.3, 12.4, 12.8)
-------------------------------------------------------------------------

`build_audit_report` existed and was called by no runner (`design.md`
Section 3.10: "exists and is called by no runner"). This task extends its
signature with the keyword-only fields `design.md` Section 3.10 lists --
`release_decision`, `estimate_status`, `dataset_checksum`, `split_hash`,
`resolved_config_hash`, `checkpoint_hashes`, `seed`, `sensitivity_variant`,
`supported_count`, `abstained_count`, and the locked-evaluation-only
`locked_evaluation_id` / `reference_access_timestamp` -- and extends
`AuditReport.to_dict()` with those fields plus `release_reasons` and the
six condition booleans (Requirement 12.3).

Every one of the eleven required keyword fields (everything except the two
locked-evaluation-only fields, which are optional and `None` outside a
locked evaluation) is validated present before the report is built:
`AuditFieldError` names the first missing one rather than this function
emitting a partial report (Requirement 12.8). "Unavailable" here means
`None` for any required field other than `estimate`/its interval (which are
legitimately `None` for a suppressed estimate -- see
`estimate_status`/`estimate` below) and `sensitivity_variant` (legitimately
`None` for an estimate not tied to any one sensitivity variant, e.g. the
primary complete-model estimate). `checkpoint_hashes` only rejects `None`,
not an empty mapping: a candidate estimate from a purely causal/statistical
estimator (e.g. `scripts/run_causal_analysis.py`'s propensity model and Cox
baselines) has no neural checkpoint at all to name, and an empty
`checkpoint_hashes={}` -- supplied deliberately, not omitted -- correctly
records that there is nothing checkpointed for this estimate, which is
different from the field never having been passed.

`estimate`'s own type widens from `BootstrapCI` to `BootstrapCI | None`
(`design.md` Section 3.10's signature) because a suppressed candidate may
genuinely have no computed interval at all. The `AuditReport` this
function returns is the single, internal, *protected*-shaped record: it
retains whatever real `estimate`/interval was passed, unconditionally
(Requirement 12.6, "retain the numeric estimate in the protected artifact
... only" -- there is exactly one `AuditReport` object, and it is the
protected one). Task 17.1's writer is where the public/protected split
actually happens: it writes this report's full `to_dict()` to
`audit/protected/<claim_id>.json`, and a derived copy with
`estimate`/`estimate_ci_low`/`estimate_ci_high` forced to `None` whenever
`estimate_status == "suppressed"` to `audit/public/<claim_id>.json`
(Requirement 12.5). This module does not itself null anything -- nulling
the one `AuditReport` object here would make the protected artifact
exactly as blind as the public one, defeating Requirement 12.6 entirely.

Design choices (original, task 11)
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

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from causal.overlap import OverlapReport
from protocols.consistency_checker import ProtocolViolation
from protocols.schema import TargetTrialSpec
from uncertainty.bootstrap import BootstrapCI
from uncertainty.support_gate import SupportDecision
from witnesses.temporal import WitnessResult

if TYPE_CHECKING:
    from evaluation.release_gate import ReleaseDecision


class AuditFieldError(Exception):
    """Raised by `build_audit_report` when a required field (Requirement
    12.3) is unavailable, naming it rather than emitting a partial report
    (Requirement 12.8)."""


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
    estimate: float | None
    estimate_ci_low: float | None
    estimate_ci_high: float | None
    overlap_diagnostics: dict
    witness_outcomes: dict[str, str]
    abstention_reasons: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    # --- Task 17 fields (Requirement 12.3, 12.4) ---
    released: bool = True
    release_reasons: tuple[str, ...] = field(default_factory=tuple)
    overlap_valid: bool = True
    balance_valid: bool = True
    uncertainty_valid: bool = True
    temporal_pass: bool = True
    causal_pass: bool = True
    clinical_pass: bool = True
    estimate_status: str = "released"
    dataset_checksum: str = ""
    split_hash: str = ""
    resolved_config_hash: str = ""
    checkpoint_hashes: dict[str, str] = field(default_factory=dict)
    seed: int = 0
    sensitivity_variant: str | None = None
    supported_count: int = 0
    abstained_count: int = 0
    locked_evaluation_id: str | None = None
    reference_access_timestamp: str | None = None

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable `dict` of this report."""
        as_dict = asdict(self)
        as_dict["release_reasons"] = list(self.release_reasons)
        return as_dict


def build_audit_report(
    protocol: TargetTrialSpec,
    population: dict,
    treatment: dict,
    estimate: BootstrapCI | None,
    overlap: OverlapReport,
    witness_results: Sequence[WitnessResult],
    abstentions: Sequence[SupportDecision] | Sequence[str],
    violations: Sequence[ProtocolViolation],
    provenance: dict,
    *,
    release_decision: ReleaseDecision,
    estimate_status: str,
    dataset_checksum: str,
    split_hash: str,
    resolved_config_hash: str,
    checkpoint_hashes: Mapping[str, str],
    seed: int,
    sensitivity_variant: str | None,
    supported_count: int,
    abstained_count: int,
    locked_evaluation_id: str | None = None,
    reference_access_timestamp: datetime | None = None,
) -> AuditReport:
    """Bundle already-computed artifacts from every pipeline stage into one
    flat `AuditReport` (Requirement 11.10, extended by Requirement 12 /
    `design.md` Section 3.10).

    Args:
        protocol: the locked target-trial specification; only `.version` is
            surfaced directly (the full spec is not duplicated into the
            report -- it is identified by version instead).
        population: a caller-assembled description of the (supported)
            target population this estimate pertains to.
        treatment: a caller-assembled description of the treatment and
            comparator.
        estimate: the cohort-level treatment-effect estimate and interval
            (`uncertainty.bootstrap.BootstrapCI`), or `None` for a
            suppressed candidate estimate that was never computed to a
            reportable interval. Regardless of what is passed here, a
            `estimate_status="suppressed"` report always records
            `estimate=None`/`estimate_ci_low=None`/`estimate_ci_high=None`
            (Requirement 12.5) -- see module docstring.
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
        release_decision: the `evaluation.release_gate.ReleaseDecision` for
            this candidate estimate; its six condition booleans and
            `reasons` are surfaced directly on the report (Requirement
            12.3).
        estimate_status: `"released"` or `"suppressed"`. Required (not
            inferred from `release_decision.released`) because a caller
            building a report for a locked-evaluation estimate, or any
            other context where suppression policy is decided one level
            up, should not have this module silently re-derive it.
        dataset_checksum: the frozen dataset contract's checksum.
        split_hash: the content hash of the split this estimate was
            evaluated on.
        resolved_config_hash: `infra.registry.canonical_json`'s hash of the
            `ResolvedConfig` this run used.
        checkpoint_hashes: `{model_name: checkpoint_sha256}` for every
            checkpoint this estimate depends on, or `{}` if this estimate
            has no neural checkpoint at all (e.g. a purely causal/
            statistical estimate). Must not be `None` (see module
            docstring).
        seed: the master seed this run used.
        sensitivity_variant: the sensitivity-variant name this estimate
            belongs to, or `None` if this estimate is not tied to any one
            sensitivity variant (e.g. the primary complete-model estimate).
        supported_count: how many patients were retained (supported) for
            this estimate.
        abstained_count: how many patients were not retained (abstained)
            for this estimate.
        locked_evaluation_id: the locked-evaluation identifier, if this
            estimate originates from a locked evaluation (Requirement
            12.4); `None` otherwise.
        reference_access_timestamp: the sealed-reference access timestamp,
            if this estimate originates from a locked evaluation
            (Requirement 12.4); `None` otherwise.

    Returns:
        An `AuditReport` containing every field required by Requirement
        11.10 and Requirement 12.3.

    Raises:
        AuditFieldError: if any required field above is unavailable (`None`
            where not explicitly permitted, or an empty
            `checkpoint_hashes`), naming the first missing field found,
            rather than this function emitting a partial report
            (Requirement 12.8).
    """
    _REQUIRED_FIELDS: tuple[tuple[str, object], ...] = (
        ("release_decision", release_decision),
        ("estimate_status", estimate_status),
        ("dataset_checksum", dataset_checksum),
        ("split_hash", split_hash),
        ("resolved_config_hash", resolved_config_hash),
        ("seed", seed),
        ("supported_count", supported_count),
        ("abstained_count", abstained_count),
    )
    for field_name, value in _REQUIRED_FIELDS:
        if value is None or value == "":
            raise AuditFieldError(f"required audit field {field_name!r} is unavailable (None)")
    if checkpoint_hashes is None:
        raise AuditFieldError(
            "required audit field 'checkpoint_hashes' is unavailable (None); "
            "pass an empty mapping explicitly if this estimate has no checkpoint"
        )
    if estimate_status not in ("released", "suppressed"):
        raise AuditFieldError(
            f"estimate_status must be 'released' or 'suppressed', got {estimate_status!r}"
        )

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
        estimate=estimate.point_estimate if estimate is not None else None,
        estimate_ci_low=estimate.ci_low if estimate is not None else None,
        estimate_ci_high=estimate.ci_high if estimate is not None else None,
        overlap_diagnostics=overlap_diagnostics,
        witness_outcomes=witness_outcomes,
        abstention_reasons=_abstention_reasons(abstentions),
        violations=[_violation_to_dict(violation) for violation in violations],
        provenance=dict(provenance),
        released=release_decision.released,
        release_reasons=release_decision.reasons,
        overlap_valid=release_decision.overlap_valid,
        balance_valid=release_decision.balance_valid,
        uncertainty_valid=release_decision.uncertainty_valid,
        temporal_pass=release_decision.temporal_pass,
        causal_pass=release_decision.causal_pass,
        clinical_pass=release_decision.clinical_pass,
        estimate_status=estimate_status,
        dataset_checksum=dataset_checksum,
        split_hash=split_hash,
        resolved_config_hash=resolved_config_hash,
        checkpoint_hashes=dict(checkpoint_hashes),
        seed=seed,
        sensitivity_variant=sensitivity_variant,
        supported_count=supported_count,
        abstained_count=abstained_count,
        locked_evaluation_id=locked_evaluation_id,
        reference_access_timestamp=(
            reference_access_timestamp.isoformat() if reference_access_timestamp else None
        ),
    )
