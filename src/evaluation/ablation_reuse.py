"""The six prediction-reuse ablation arms: `no_overlap_gate`,
`no_uncertainty_gate`, `no_clinical_constraints`, and the three
`witness_removed` sub-runs (task 14.1; Requirement 7.6, 7.7, 7.8, 7.15;
`design.md` Section 3.6).

These six runs (`evaluation.ablation_matrix.expand_ablations()`,
`execution_mode == "reuse_predictions"`) never retrain. They reuse the
complete model's saved patient-level predictions unmodified and bypass
only the named gate, constraint, or witness, then report the count of
Validation_Split patients whose *support decision* differs from the
complete model's own recorded decision.

What "bypass only the named gate/constraint/witness" means, concretely
-------------------------------------------------------------------------

`uncertainty.support_gate.PatientSupportGate` computes `patient_supported`
as the AND of four named conditions: `propensity_in_common_support`,
`ensemble_uncertainty_ok`, `protocol_valid`, `no_hard_violation`. Each
reuse run forces exactly one of those four conditions to `True`
(effectively removing it from the conjunction) while leaving the other
three exactly as recorded on the complete model's prediction, then
recomputes `supported` and compares against the complete model's own
recorded `supported` value:

| Run | Forced condition |
|---|---|
| `no_overlap_gate` | `propensity_in_common_support = True` |
| `no_uncertainty_gate` | `ensemble_uncertainty_ok = True` |
| `no_clinical_constraints` | `protocol_valid = True` |
| `witness_removed::clinical` | `protocol_valid = True` |
| `witness_removed::temporal` | `no_hard_violation = True` |
| `witness_removed::causal` | *(none -- see below)* |

`protocol_valid` and `no_hard_violation`'s witness provenance (documented
simplification, read before extending this module)
-------------------------------------------------------------------------

`PatientSupportGate`'s own module docstring documents `protocol_valid` as
sourced from "no `ProtocolChecker.ProtocolViolation` for this patient" --
exactly what `witnesses.clinical.ClinicalWitness` wraps -- and
`no_hard_violation` as sourced "from the clinical/temporal witnesses once
they exist." This module resolves that ambiguity concretely:
`protocol_valid` is attributed to the **clinical** witness (eligibility,
assignment, dose/treatment identity, endpoint construction -- exactly
`decisions.md` Section 16's "clinical witness" checks, which
`ProtocolChecker` already implements) and `no_hard_violation` is
attributed to the **temporal** witness (baseline-precedes-treatment,
outcome-after-time-zero, chronological ordering -- `decisions.md` Section
16's "temporal witness" checks). `no_clinical_constraints` and
`witness_removed::clinical` therefore force the *same* condition
(`protocol_valid`), which is consistent: disabling clinical constraints
entirely is a superset of removing only the clinical witness's
contribution to the gate.

`witness_removed::causal` has **no per-patient effect** under this
attribution: the causal witness (`decisions.md` Section 16) evaluates
cohort-level positivity, balance, effective-sample-size, sensitivity, and
negative-control conditions -- none of which `PatientSupportGate`'s four
per-patient conditions read from. Removing the causal witness therefore
changes zero patients' support decisions by this module's accounting, and
`ReusePredictionResult.differing_patient_ids` is correctly empty for that
run. This is reported explicitly in the result rather than silently
produced, since a reader comparing all six reuse arms' differing-counts
should not have to guess why one is exactly zero.

Checkpoint-hash verification (Requirement 7.15)
-------------------------------------------------------------------------

Every reuse run's prediction source carries the complete model's recorded
checkpoint hash (`CompleteModelPredictions.checkpoint_sha256`). `run_reuse_ablation`
compares it against the caller-supplied `expected_checkpoint_sha256`
(normally read from wherever the complete model's own training recorded
its checkpoint hash) and raises `ReusePredictionSourceError` on any
mismatch, on a missing file, or on an unreadable/malformed file --
writing no metrics for that run and leaving every other run's
already-written results untouched (this function never writes another
run's output; the caller controls per-run result-file placement, matching
`scripts/run_ablations.py`'s per-run-file convention).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: The six reuse-mode run ids this module handles, and which support
#: condition each one forces to True. `None` for `witness_removed::causal`
#: means "force nothing" (see module docstring).
_FORCED_CONDITION_BY_RUN_ID: dict[str, str | None] = {
    "no_overlap_gate": "propensity_in_common_support",
    "no_uncertainty_gate": "ensemble_uncertainty_ok",
    "no_clinical_constraints": "protocol_valid",
    "witness_removed::clinical": "protocol_valid",
    "witness_removed::temporal": "no_hard_violation",
    "witness_removed::causal": None,
}

#: Every run id this module is defined over -- the keys of the mapping
#: above. Exposed so a caller/test can assert the reuse-mode run set this
#: module handles equals `evaluation.ablation_matrix`'s own reuse-mode
#: subset.
REUSE_RUN_IDS: frozenset[str] = frozenset(_FORCED_CONDITION_BY_RUN_ID)


class ReusePredictionSourceError(Exception):
    """Raised when a reuse run's complete-model prediction source is
    absent, unreadable, or carries a checkpoint hash that does not match
    the complete model's recorded checkpoint hash (Requirement 7.15).
    """


@dataclass(frozen=True)
class PatientPrediction:
    """One patient's complete-model prediction record and the four
    support-gate condition inputs it was computed from.

    Attributes:
        patient_id: this patient's id.
        predicted_risk: the complete model's factual-arm predicted risk
            (e.g. the 96-week cumulative event probability) for this
            patient -- carried through unmodified by every reuse run.
        propensity_in_common_support: the complete model's recorded value
            of this `PatientSupportGate` condition.
        ensemble_uncertainty_ok: likewise.
        protocol_valid: likewise.
        no_hard_violation: likewise.
        supported: the complete model's own recorded overall
            `patient_supported` decision (the AND of the four conditions
            above, as actually computed when the complete model's
            predictions were saved).
    """

    patient_id: str
    predicted_risk: float
    propensity_in_common_support: bool
    ensemble_uncertainty_ok: bool
    protocol_valid: bool
    no_hard_violation: bool
    supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "patient_id": self.patient_id,
            "predicted_risk": self.predicted_risk,
            "propensity_in_common_support": self.propensity_in_common_support,
            "ensemble_uncertainty_ok": self.ensemble_uncertainty_ok,
            "protocol_valid": self.protocol_valid,
            "no_hard_violation": self.no_hard_violation,
            "supported": self.supported,
        }

    @staticmethod
    def from_dict(payload: dict[str, object]) -> PatientPrediction:
        try:
            return PatientPrediction(
                patient_id=str(payload["patient_id"]),
                predicted_risk=float(payload["predicted_risk"]),  # type: ignore[arg-type]
                propensity_in_common_support=bool(payload["propensity_in_common_support"]),
                ensemble_uncertainty_ok=bool(payload["ensemble_uncertainty_ok"]),
                protocol_valid=bool(payload["protocol_valid"]),
                no_hard_violation=bool(payload["no_hard_violation"]),
                supported=bool(payload["supported"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReusePredictionSourceError(
                f"malformed patient prediction record: {payload!r} ({exc})"
            ) from exc


@dataclass(frozen=True)
class CompleteModelPredictions:
    """The complete model's saved Validation_Split (or Development_Split)
    patient-level predictions -- the reuse source every reuse run reads
    unmodified (`design.md` Section 2.3:
    `predictions/complete_model/{val,dev}_patient_predictions.json`).

    Attributes:
        split_name: `"val"` or `"dev"`.
        checkpoint_sha256: the complete model's recorded checkpoint hash,
            verified against on every reuse (Requirement 7.15).
        patients: one `PatientPrediction` per patient, in a stable order.
    """

    split_name: str
    checkpoint_sha256: str
    patients: tuple[PatientPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "patients": [patient.to_dict() for patient in self.patients],
        }

    def write(self, path: str | Path) -> Path:
        """Write this prediction set as JSON to `path`, creating parent
        directories as needed. Returns the written path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return target


def load_complete_model_predictions(
    path: str | Path, expected_checkpoint_sha256: str
) -> CompleteModelPredictions:
    """Load and verify the complete model's saved predictions.

    Args:
        path: path to a `CompleteModelPredictions`-shaped JSON file
            (typically `predictions/complete_model/val_patient_predictions.json`).
        expected_checkpoint_sha256: the complete model's recorded
            checkpoint hash this file is expected to carry.

    Returns:
        The loaded `CompleteModelPredictions`.

    Raises:
        ReusePredictionSourceError: if `path` does not exist, is not
            readable as JSON, is malformed, or carries a
            `checkpoint_sha256` that does not equal
            `expected_checkpoint_sha256`.
    """
    target = Path(path)
    if not target.is_file():
        raise ReusePredictionSourceError(
            f"complete-model prediction source is missing: {target}"
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReusePredictionSourceError(
            f"complete-model prediction source is unreadable: {target} ({exc})"
        ) from exc

    try:
        split_name = str(raw["split_name"])
        checkpoint_sha256 = str(raw["checkpoint_sha256"])
        patient_payloads = raw["patients"]
    except (KeyError, TypeError) as exc:
        raise ReusePredictionSourceError(
            f"complete-model prediction source is malformed: {target} ({exc})"
        ) from exc

    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ReusePredictionSourceError(
            f"complete-model prediction source checkpoint hash mismatch at {target}: "
            f"expected {expected_checkpoint_sha256!r}, found {checkpoint_sha256!r}"
        )

    patients = tuple(PatientPrediction.from_dict(payload) for payload in patient_payloads)
    return CompleteModelPredictions(
        split_name=split_name, checkpoint_sha256=checkpoint_sha256, patients=patients
    )


@dataclass(frozen=True)
class ReusePredictionResult:
    """One reuse run's outcome: the complete model's predictions,
    unmodified, plus the recomputed support decision under the named
    bypass, and the resulting differing-patient count.

    Attributes:
        run_id: the reuse run id this result belongs to.
        forced_condition: which `PatientSupportGate` condition this run
            forced to `True`, or `None` for `witness_removed::causal`
            (see module docstring).
        source_checkpoint_sha256: the verified complete-model checkpoint
            hash this run's predictions were reused from.
        predicted_risks: `{patient_id: predicted_risk}`, carried through
            unmodified from the complete model's predictions -- the
            structural evidence that no training occurred (Property 28:
            "reused predictions reproduced byte-identically").
        bypassed_supported: `{patient_id: supported}`, the recomputed
            support decision under the forced condition.
        differing_patient_ids: patient ids whose `bypassed_supported`
            differs from the complete model's own recorded `supported`.
    """

    run_id: str
    forced_condition: str | None
    source_checkpoint_sha256: str
    predicted_risks: dict[str, float]
    bypassed_supported: dict[str, bool]
    differing_patient_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "forced_condition": self.forced_condition,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "predicted_risks": self.predicted_risks,
            "bypassed_supported": self.bypassed_supported,
            "differing_patient_ids": list(self.differing_patient_ids),
            "differing_patient_count": len(self.differing_patient_ids),
        }


def _recompute_supported(patient: PatientPrediction, forced_condition: str | None) -> bool:
    propensity_in_common_support = patient.propensity_in_common_support
    ensemble_uncertainty_ok = patient.ensemble_uncertainty_ok
    protocol_valid = patient.protocol_valid
    no_hard_violation = patient.no_hard_violation

    if forced_condition == "propensity_in_common_support":
        propensity_in_common_support = True
    elif forced_condition == "ensemble_uncertainty_ok":
        ensemble_uncertainty_ok = True
    elif forced_condition == "protocol_valid":
        protocol_valid = True
    elif forced_condition == "no_hard_violation":
        no_hard_violation = True
    elif forced_condition is not None:  # pragma: no cover - defensive
        raise ValueError(f"unknown forced_condition {forced_condition!r}")

    return (
        propensity_in_common_support
        and ensemble_uncertainty_ok
        and protocol_valid
        and no_hard_violation
    )


def run_reuse_ablation(
    run_id: str,
    source: CompleteModelPredictions,
    expected_checkpoint_sha256: str,
) -> ReusePredictionResult:
    """Run one prediction-reuse ablation arm.

    Performs no training. Verifies `source.checkpoint_sha256` against
    `expected_checkpoint_sha256` (redundant with
    `load_complete_model_predictions`'s own check when `source` was
    produced by that function, but re-checked here so this function is
    safe to call directly with a hand-built `source` too), recomputes
    each patient's support decision under `run_id`'s forced condition,
    and reports which patients' decisions changed.

    Args:
        run_id: one of `REUSE_RUN_IDS`.
        source: the complete model's loaded predictions.
        expected_checkpoint_sha256: the complete model's recorded
            checkpoint hash.

    Returns:
        A `ReusePredictionResult`.

    Raises:
        ValueError: if `run_id` is not one of `REUSE_RUN_IDS`.
        ReusePredictionSourceError: if `source.checkpoint_sha256` does not
            equal `expected_checkpoint_sha256`.
    """
    if run_id not in _FORCED_CONDITION_BY_RUN_ID:
        raise ValueError(
            f"run_id={run_id!r} is not a reuse-mode run; expected one of "
            f"{sorted(REUSE_RUN_IDS)!r}"
        )
    if source.checkpoint_sha256 != expected_checkpoint_sha256:
        raise ReusePredictionSourceError(
            f"source.checkpoint_sha256={source.checkpoint_sha256!r} does not match "
            f"expected_checkpoint_sha256={expected_checkpoint_sha256!r}"
        )

    forced_condition = _FORCED_CONDITION_BY_RUN_ID[run_id]

    predicted_risks: dict[str, float] = {}
    bypassed_supported: dict[str, bool] = {}
    differing: list[str] = []

    for patient in source.patients:
        predicted_risks[patient.patient_id] = patient.predicted_risk
        recomputed = _recompute_supported(patient, forced_condition)
        bypassed_supported[patient.patient_id] = recomputed
        if recomputed != patient.supported:
            differing.append(patient.patient_id)

    return ReusePredictionResult(
        run_id=run_id,
        forced_condition=forced_condition,
        source_checkpoint_sha256=source.checkpoint_sha256,
        predicted_risks=predicted_risks,
        bypassed_supported=bypassed_supported,
        differing_patient_ids=tuple(differing),
    )


def run_all_reuse_ablations(
    source: CompleteModelPredictions,
    expected_checkpoint_sha256: str,
    run_ids: Sequence[str] = tuple(sorted(REUSE_RUN_IDS)),
) -> dict[str, ReusePredictionResult]:
    """Run every reuse-mode ablation arm named in `run_ids` against the
    same loaded `source`.

    If any run raises `ReusePredictionSourceError` (only possible here if
    `source` was constructed by a caller that bypassed
    `load_complete_model_predictions`'s own check), the exception
    propagates and no further run in `run_ids` executes -- but, per
    Requirement 7.15, any run already written to disk by the caller
    before this call is unaffected, since this function returns an
    in-memory mapping and performs no file I/O itself.

    Returns:
        `{run_id: ReusePredictionResult}`, one entry per `run_ids` entry.
    """
    return {
        run_id: run_reuse_ablation(run_id, source, expected_checkpoint_sha256)
        for run_id in run_ids
    }
