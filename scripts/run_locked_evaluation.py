#!/usr/bin/env python
"""Locked_Evaluation_Runner: the one entry point that opens the sealed
reference and the test split, exactly once per minted token (task 21;
`neural-experiment-pipeline` Requirements 13.1-13.11, 14.1-14.5).

Why a separate script (Requirement 13.1)
-------------------------------------------------------------------------

A flag on a training script cannot satisfy Requirement 13.1's "exposes no
option to fit, tune, or select any parameter or threshold," because the
fitting code would still be *reachable* even behind a flag that defaults
to off. This script instead imports `models.training.load_checkpoint` /
`models.torch_training.load_torch_checkpoint` and the evaluation functions
only, and never imports `models.torch_training.select_lambdas`,
`train_one_model`, `train_ensemble`, or `uncertainty.calibration.Calibrator.fit`
(any calibration-fitting function). `tests/neural/test_run_locked_evaluation.py`
asserts this module's import graph excludes every one of those names by
AST-walking the source rather than trusting a docstring (Correctness
Property 16, Requirement 13.1).

Rejected invocation options (Requirement 13.2)
-------------------------------------------------------------------------

`REJECTED_OPTION_NAMES` below lists every CLI flag name that would resemble
fitting, tuning, or selection, if this script ever grew one. `build_parser`
does not define any of them -- argparse's own "unrecognized arguments"
behavior would already reject them, but `main()` additionally checks `argv`
by name *before* `parser.parse_args` runs, so the rejection is explicit, by
name, and happens before anything else, including before any file is
opened (Requirement 13.2's "terminate before opening the test split, the
sealed reference, or any result artifact").

Execution order (`design.md` Section 3.12), chosen so nothing irreversible
happens before verification completes
-------------------------------------------------------------------------

1. Parse args; reject any fitting/tuning/selection-shaped option by name
   (Requirement 13.2).
2. Load the token artifact (JSON written by `scripts/check_go_no_go.py`
   via `--token-output`). Reconstruct a real `LockedEvaluationToken`
   through `_mint`-adjacent validation (never a bare dataclass call that
   could construct a partially-bound token).
3. Check `evaluation_id` (and the loaded token) against `consumed.jsonl`;
   refuse before opening the test split or sealed reference if already
   consumed (Requirement 13.6).
4. Recompute and compare every bound hash -- resolved config, dataset
   checksum, split hash, each checkpoint -- against the artifacts actually
   present. Terminate non-zero on the FIRST mismatch, naming the field,
   writing nothing (Requirement 13.5).
5. Verify the target paths under `results/locked_evaluation/<evaluation_id>/`
   are unoccupied; refuse non-zero and change nothing if occupied
   (Requirement 13.10).
6. (task 21.1) Open the test split and unseal the reference; compute
   treatment-effect recovery, run the paired bootstrap + Holm correction,
   evaluate the witnesses and the release gate, build audit reports.
7. (task 21.2) Atomic commit: stage everything under
   `results/locked_evaluation/.staging_<evaluation_id>/`, `os.rename` it
   into place, then append to `consumed.jsonl`.

Any failure after step 4 (hash verification passed) leaves the results
tree byte-identical, the token and evaluation id unconsumed, and reports
the failing step by name (Requirement 13.11) -- see
`run_locked_evaluation`'s docstring and `_atomic_commit`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from cohorts.sealed_reference import LockedEvaluationToken, TokenFieldError
from infra.consumption_ledger import append_consumed, is_consumed
from models.training import load_checkpoint  # noqa: F401 -- see module docstring; torch-free

#: Repository root, resolved relative to this file's location (matches
#: every other `scripts/` entry point's convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default top-level results root (Requirement 13.6's `consumed.jsonl` and
#: Requirement 13.10's target-path check both live under this root).
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

#: Every CLI option name that would resemble fitting, tuning, or selection
#: (Requirement 13.2). None of these are ever defined by `build_parser` --
#: this list exists so `main()` can refuse them BY NAME, with a clear
#: message, before `argparse` even runs (whose own "unrecognized
#: arguments" message would not name *why* the option is rejected).
REJECTED_OPTION_NAMES: tuple[str, ...] = (
    "--fit",
    "--train",
    "--tune",
    "--select-lambdas",
    "--select-lambda",
    "--calibrate",
    "--fit-calibration",
    "--retrain",
    "--learning-rate",
    "--lr",
    "--epochs",
    "--max-epochs",
    "--patience",
    "--threshold",
    "--set-threshold",
)


class RejectedOptionError(Exception):
    """Raised by `check_for_rejected_options` when `argv` names an option
    resembling fitting, tuning, or selection (Requirement 13.2)."""


def check_for_rejected_options(argv: list[str]) -> None:
    """Raise `RejectedOptionError` naming the first rejected option found
    in `argv`, before any other argument parsing happens.

    Matches on the option name only (the part before `=`, if any), so
    `--learning-rate=0.01` is caught exactly like `--learning-rate 0.01`.
    """
    rejected = set(REJECTED_OPTION_NAMES)
    for arg in argv:
        name = arg.split("=", 1)[0]
        if name in rejected:
            raise RejectedOptionError(
                f"rejected option {name!r}: this entry point exposes no option to "
                "fit, tune, or select any parameter or threshold (Requirement 13.2). "
                "Checkpoints are loaded read-only; training/selection happens only "
                "in the development-stage entry points."
            )


# ---------------------------------------------------------------------------
# Step 2: load and reconstruct the token artifact
# ---------------------------------------------------------------------------


class TokenArtifactError(Exception):
    """Raised when a token artifact file cannot be loaded or does not
    describe a validly-bound `LockedEvaluationToken`."""


def load_token_artifact(path: Path) -> LockedEvaluationToken:
    """Load a token JSON artifact (written by
    `scripts/check_go_no_go.py --token-output`) and reconstruct a real,
    validated `LockedEvaluationToken`.

    This is NOT a bare dataclass construction: every field is re-validated
    by `LockedEvaluationToken.__post_init__` (absent/empty/malformed
    bindings raise `TokenFieldError`, surfaced here as
    `TokenArtifactError` naming every offender), so a hand-edited or
    corrupted artifact cannot produce a token this runner treats as valid
    merely because the JSON parses.

    Args:
        path: path to the token JSON artifact.

    Returns:
        A fully-bound, re-validated `LockedEvaluationToken`.

    Raises:
        TokenArtifactError: if `path` does not exist, is not valid JSON,
            is missing a required key, or if the reconstructed fields fail
            `LockedEvaluationToken`'s own validation.
    """
    if not path.exists():
        raise TokenArtifactError(f"token artifact not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TokenArtifactError(f"token artifact at {path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise TokenArtifactError(f"token artifact at {path} must be a JSON object")

    required_keys = (
        "issued_by",
        "issued_at",
        "resolved_config_hash",
        "dataset_checksum",
        "split_hash",
        "checkpoint_hashes",
        "git_commit",
        "statistical_plan_version",
        "evaluation_id",
    )
    missing_keys = sorted(key for key in required_keys if key not in payload)
    if missing_keys:
        raise TokenArtifactError(
            f"token artifact at {path} is missing key(s): {', '.join(missing_keys)}"
        )

    from datetime import datetime

    try:
        issued_at = datetime.fromisoformat(payload["issued_at"])
    except (TypeError, ValueError) as exc:
        raise TokenArtifactError(
            f"token artifact at {path}: 'issued_at' is not a valid ISO-8601 "
            f"timestamp: {payload['issued_at']!r}"
        ) from exc

    checkpoint_hashes_raw = payload["checkpoint_hashes"]
    try:
        checkpoint_hashes = tuple(tuple(pair) for pair in checkpoint_hashes_raw)
    except TypeError as exc:
        raise TokenArtifactError(
            f"token artifact at {path}: 'checkpoint_hashes' is not a list of pairs"
        ) from exc

    try:
        return LockedEvaluationToken(
            issued_by=payload["issued_by"],
            issued_at=issued_at,
            resolved_config_hash=payload["resolved_config_hash"],
            dataset_checksum=payload["dataset_checksum"],
            split_hash=payload["split_hash"],
            checkpoint_hashes=checkpoint_hashes,
            git_commit=payload["git_commit"],
            statistical_plan_version=payload["statistical_plan_version"],
            evaluation_id=payload["evaluation_id"],
        )
    except TokenFieldError as exc:
        raise TokenArtifactError(
            f"token artifact at {path} does not describe a validly-bound token: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Step 3: consumption check
# ---------------------------------------------------------------------------


class AlreadyConsumedError(Exception):
    """Raised when the token's `evaluation_id` is already recorded as
    consumed (Requirement 13.6)."""


def check_not_consumed(token: LockedEvaluationToken, results_root: Path) -> None:
    """Refuse if `token.evaluation_id` is already in `consumed.jsonl`.

    Called before the test split or the sealed reference is opened
    (Requirement 13.6) -- this function touches only the consumption
    ledger, never the protected resources themselves.

    Raises:
        AlreadyConsumedError: if already consumed.
    """
    if is_consumed(token.evaluation_id, results_root):
        from infra.consumption_ledger import consumed_ledger_path

        raise AlreadyConsumedError(
            f"evaluation_id {token.evaluation_id!r} is already recorded as consumed "
            f"in {consumed_ledger_path(results_root)}. A locked evaluation is "
            "one-shot; refusing before opening the test split or the sealed "
            "reference."
        )


# ---------------------------------------------------------------------------
# Step 4: hash recomputation (Requirement 13.5)
# ---------------------------------------------------------------------------


class HashMismatchError(Exception):
    """Raised by `verify_bound_hashes` on the FIRST mismatch between a
    token-bound hash and the recomputed value for the same artifact
    (Requirement 13.5)."""


def _file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes.

    Deliberately NOT `models.torch_training.checkpoint_sha256`, even
    though the two are byte-for-byte the same algorithm over the same
    kind of file: that function lives in a module that imports torch at
    module scope, and this runner's hash-verification step (Requirement
    13.5) must stay reachable before torch is ever touched, matching
    `models.training.load_checkpoint`'s own torch-free import at this
    module's top (see module docstring, "Why a separate script"). A
    checkpoint's hash is a property of its bytes, not of which hasher
    happened to compute it, so there is no risk of the two functions
    disagreeing on a real checkpoint file.
    """
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bound_hashes(
    token: LockedEvaluationToken,
    *,
    resolved_config_hash: str,
    dataset_checksum: str,
    split_hash: str,
    checkpoint_paths: dict[str, Path],
) -> None:
    """Recompute every bound hash from the artifacts actually present and
    compare each against the token's binding, terminating on the FIRST
    mismatch (Requirement 13.5).

    Fields are compared in `LockedEvaluationToken`'s declaration order
    (resolved config, dataset checksum, split hash, then checkpoints in
    sorted-name order) so two runs against the same mismatched inputs
    always report the identical first offender.

    Args:
        token: the verified, loaded token.
        resolved_config_hash: recomputed from the `ResolvedConfig` this
            run actually composes (`infra.registry.canonical_json`).
        dataset_checksum: recomputed/read from the loaded dataset
            contract.
        split_hash: recomputed from the actual dev/val/test assignment
            present.
        checkpoint_paths: `{logical_name: path}` for every checkpoint this
            run would load. Must cover exactly the logical names bound in
            `token.checkpoint_hashes`.

    Raises:
        HashMismatchError: naming the first field (or missing/extra
            checkpoint name) that does not match.
    """
    if resolved_config_hash != token.resolved_config_hash:
        raise HashMismatchError(
            "resolved_config_hash mismatch: token binds "
            f"{token.resolved_config_hash!r}, recomputed {resolved_config_hash!r}"
        )
    if dataset_checksum != token.dataset_checksum:
        raise HashMismatchError(
            f"dataset_checksum mismatch: token binds {token.dataset_checksum!r}, "
            f"recomputed {dataset_checksum!r}"
        )
    if split_hash != token.split_hash:
        raise HashMismatchError(
            f"split_hash mismatch: token binds {token.split_hash!r}, recomputed "
            f"{split_hash!r}"
        )

    bound_checkpoints = dict(token.checkpoint_hashes)
    missing_names = sorted(set(bound_checkpoints) - set(checkpoint_paths))
    if missing_names:
        raise HashMismatchError(
            f"checkpoint_hashes mismatch: token binds checkpoint(s) not present "
            f"among the loaded checkpoints: {', '.join(missing_names)}"
        )
    extra_names = sorted(set(checkpoint_paths) - set(bound_checkpoints))
    if extra_names:
        raise HashMismatchError(
            f"checkpoint_hashes mismatch: loaded checkpoint(s) not bound by the "
            f"token: {', '.join(extra_names)}"
        )

    for logical_name in sorted(bound_checkpoints):
        path = checkpoint_paths[logical_name]
        recomputed = _file_sha256(path)
        bound = bound_checkpoints[logical_name]
        if recomputed != bound:
            raise HashMismatchError(
                f"checkpoint_hashes mismatch for {logical_name!r}: token binds "
                f"{bound!r}, recomputed {recomputed!r} from {path}"
            )


# ---------------------------------------------------------------------------
# Step 5: target-path occupancy (Requirement 13.10)
# ---------------------------------------------------------------------------


class OccupiedPathError(Exception):
    """Raised when `results/locked_evaluation/<evaluation_id>/` already
    exists and is non-empty (Requirement 13.10)."""


def locked_evaluation_result_dir(results_root: Path, evaluation_id: str) -> Path:
    """`results_root/locked_evaluation/<evaluation_id>` -- the committed
    result directory for one evaluation id."""
    return Path(results_root) / "locked_evaluation" / evaluation_id


def staging_dir(results_root: Path, evaluation_id: str) -> Path:
    """`results_root/locked_evaluation/.staging_<evaluation_id>` -- the
    atomic-commit staging directory (task 21.2; `design.md` Section 2.3's
    two-phase-commit convention, matching the sealed reference's
    `write_sealed_artifact`-adjacent 0o600 convention for defense in
    depth, though staging itself is about atomicity, not secrecy)."""
    return Path(results_root) / "locked_evaluation" / f".staging_{evaluation_id}"


def verify_target_path_unoccupied(results_root: Path, evaluation_id: str) -> None:
    """Refuse if `locked_evaluation_result_dir` already exists and holds
    anything (Requirement 13.10).

    Raises:
        OccupiedPathError: if occupied.
    """
    target = locked_evaluation_result_dir(results_root, evaluation_id)
    if target.exists() and any(target.iterdir()):
        raise OccupiedPathError(
            f"target path {target} is already occupied; refusing to run a "
            f"locked evaluation for evaluation_id={evaluation_id!r} a second time "
            "at this path. Nothing has been changed."
        )


# ---------------------------------------------------------------------------
# Task 21.1: the evaluation steps under a verified token (design.md
# Section 3.12 steps 5-9).
# ---------------------------------------------------------------------------


class LockedEvaluationResult:
    """The complete output of `run_locked_evaluation` (task 21.1).

    A plain holder (not a frozen dataclass with validation, since this is
    an internal result object threaded straight into task 21.2's atomic
    commit, not a value with its own invariants to enforce) carrying:

    Attributes:
        reference_access_timestamp: when the sealed reference was
            unsealed (Requirement 14.5, 12.4).
        recovery_errors: `{quantity_name: compare_to_reference(...)}` for
            every candidate causal estimate compared against the sealed
            reference (Requirement 13.7, 14.3-14.4).
        comparison_results: `{comparison_name: BootstrapCI}` for the
            three frozen primary comparisons (Requirement 13.8).
        holm_adjusted_p_values: `{comparison_name: adjusted_p_value}`
            (Requirement 13.8).
        witness_results: the three witnesses' results over the identical
            test patients (Requirement 13.9).
        release_decision: `evaluate_release(...)` over the identical test
            patients (Requirement 13.9).
        checkpoint_hashes_after: `{logical_name: sha256}` recomputed AFTER
            the run, for the unchanged-checkpoint confirmation
            (Requirement 13.1).
        audit_reports: `{claim_id: AuditReport}`, one per candidate
            estimate (every recovery quantity plus every frozen
            comparison), released and suppressed alike (Requirement
            12.1, 12.2). Written to disk by task 21.2's atomic commit via
            `witnesses.audit_io.write_audit_artifacts`.
    """

    def __init__(
        self,
        *,
        reference_access_timestamp: Any,
        recovery_errors: dict[str, float],
        comparison_results: dict[str, Any],
        holm_adjusted_p_values: dict[str, float],
        witness_results: list[Any],
        release_decision: Any,
        checkpoint_hashes_after: dict[str, str],
        audit_reports: dict[str, Any],
    ) -> None:
        self.reference_access_timestamp = reference_access_timestamp
        self.recovery_errors = recovery_errors
        self.comparison_results = comparison_results
        self.holm_adjusted_p_values = holm_adjusted_p_values
        self.witness_results = witness_results
        self.release_decision = release_decision
        self.checkpoint_hashes_after = checkpoint_hashes_after
        self.audit_reports = audit_reports


def run_locked_evaluation(
    token: LockedEvaluationToken,
    pipeline_result: Any,
    *,
    target_trial_spec: Any,
    recovery_estimates: dict[str, Any],
    comparison_values: dict[str, tuple[list[float], list[float]]],
    witness_results: list[Any],
    release_decision: Any,
    checkpoint_paths: dict[str, Path],
    results_root: Path,
    bootstrap_seed: int = 0,
) -> LockedEvaluationResult:
    """Run the evaluation steps under an already-verified token
    (`design.md` Section 3.12 steps 5-9; Requirements 13.7, 13.8, 13.9,
    13.1, 14.3, 14.4, 14.5).

    Called only after `main()`'s steps 1-4 (option rejection, token
    loading, consumption check, hash verification) and step 5 (occupied-
    path check) have all passed -- this function itself opens the two
    protected resources (test split, sealed reference), so it must never
    be reachable except behind that verification.

    This function accepts already-computed per-patient recovery estimates
    and comparison values rather than running the models itself: the
    model-forward-pass plumbing (building `TensorBatch`es, running
    `rollout_both_arms`, extracting factual outcomes) is identical to what
    `scripts/run_replication.py::run_seed`'s own helpers already do, and
    duplicating that plumbing a second time here -- the one place in the
    whole pipeline where it runs against the real test split -- is exactly
    the kind of divergent second implementation Requirement 13.7's
    "compute treatment-effect recovery over that identical patient set"
    should not have two authors. The caller (this module's `main()`, once
    wired to a real data-loading CLI surface) is responsible for producing
    these values FROM the test split `require_test_split` returns, using
    the identical patient id ordering throughout.

    Args:
        token: the already-verified `LockedEvaluationToken`.
        pipeline_result: the `cohorts.pipeline.ObservationalizationPipelineResult`
            whose `.test` split `require_test_split` will open.
        target_trial_spec: the locked `protocols.schema.TargetTrialSpec`,
            forwarded to every built audit report's `protocol` argument
            (only `.version` is read).
        recovery_estimates: `{quantity_name: causal.baselines.CausalEstimate}`
            -- one candidate estimate per sealed-reference quantity to
            compare against (e.g. `"risk_difference"`), computed over the
            test split's patients.
        comparison_values: `{comparison_name: (values_a, values_b)}` for
            exactly `evaluation.statistics.PRIMARY_COMPARISONS`, each a
            pair of same-length, index-paired per-patient value sequences
            on the identical test patient set.
        witness_results: the three witnesses' results, already evaluated
            over the identical test patients.
        release_decision: `evaluation.release_gate.evaluate_release(...)`,
            already evaluated over the identical test patients.
        checkpoint_paths: `{logical_name: path}` for every checkpoint
            this run loaded -- recomputed again at the end to confirm
            unchanged (Requirement 13.1).
        results_root: the top-level results directory, forwarded to
            `require_test_split`'s token guard path and to
            `SealedReference.unseal`'s access-ledger append.
        bootstrap_seed: seed for both the paired-bootstrap CI and p-value
            calls, applied per comparison via `derive_seed(bootstrap_seed,
            comparison_name)` so the three comparisons do not share one
            raw seed.

    Returns:
        A `LockedEvaluationResult`.

    Raises:
        ReferenceAccessError: if `token` is not a valid token (defensive;
            `main()`'s step 2 should already guarantee this).
        cohorts.sealed_reference.ScaleMismatchError: if a recovery
            estimate's scale does not match its sealed-reference quantity.
        ValueError: from `evaluation.statistics.paired_bootstrap_test`/
            `holm_correction` if `comparison_values` does not cover
            exactly `PRIMARY_COMPARISONS`.
    """
    from evaluation.statistics import (
        PRIMARY_COMPARISONS,
        holm_correction,
        paired_bootstrap_p_value,
        paired_bootstrap_test,
    )
    from infra.reference_guard import require_test_split
    from infra.seeding import derive_seed

    # Step 5 (design.md numbering): open the test split and unseal the
    # reference. Both guards raise before anything is touched if `token`
    # is not a valid, minted `LockedEvaluationToken`.
    require_test_split(pipeline_result, token)
    sealed_reference = pipeline_result.sealed_reference.unseal(token, results_root=results_root)

    # `unseal` already appended this access to `reference_access.jsonl`;
    # read it back so the result carries the real timestamp rather than a
    # second, independently-computed one (Requirement 14.5, 12.4).
    from infra.consumption_ledger import reference_access_timestamp as _read_access_timestamp

    reference_access_time = _read_access_timestamp(token.evaluation_id, results_root)

    # Step 6: treatment-effect recovery, scale-checked against the sealed
    # reference.
    from cohorts.sealed_reference import compare_to_reference

    recovery_errors = {
        quantity_name: compare_to_reference(estimate, sealed_reference, quantity_name)
        for quantity_name, estimate in recovery_estimates.items()
    }

    # Step 7: paired bootstrap + Holm correction across exactly the three
    # frozen comparisons.
    missing_comparisons = sorted(set(PRIMARY_COMPARISONS) - set(comparison_values))
    extra_comparisons = sorted(set(comparison_values) - set(PRIMARY_COMPARISONS))
    if missing_comparisons or extra_comparisons:
        raise ValueError(
            "comparison_values must cover exactly the three frozen primary "
            f"comparisons {PRIMARY_COMPARISONS!r}; missing={missing_comparisons!r}, "
            f"extra={extra_comparisons!r}"
        )

    comparison_results: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for comparison_name in PRIMARY_COMPARISONS:
        values_a, values_b = comparison_values[comparison_name]
        comparison_seed = derive_seed(bootstrap_seed, comparison_name)
        comparison_results[comparison_name] = paired_bootstrap_test(
            comparison_name, values_a, values_b, n_replicates=1000, seed=comparison_seed
        )
        raw_p_values[comparison_name] = paired_bootstrap_p_value(
            comparison_name, values_a, values_b, n_replicates=1000, seed=comparison_seed
        )
    holm_adjusted_p_values = holm_correction(raw_p_values)

    # Step 8: witnesses and release gate are already evaluated by the
    # caller over the identical test patients (see docstring); this
    # function records them unchanged in the result.

    # Step 9: build an audit report for every candidate estimate --
    # recovery estimates plus the three frozen comparisons, released and
    # suppressed alike (Requirement 12.1-12.4). Reuses the existing
    # `witnesses.audit_report.build_audit_report` /
    # `witnesses.audit_io.write_audit_artifacts` pair exactly as
    # `scripts/run_causal_analysis.py`/`scripts/run_ablations.py` already
    # do, with `locked_evaluation_id`/`reference_access_timestamp`
    # populated this time (the one context where those two fields are
    # non-`None`).
    audit_reports = _build_locked_evaluation_audit_reports(
        token=token,
        target_trial_spec=target_trial_spec,
        recovery_estimates=recovery_estimates,
        comparison_results=comparison_results,
        witness_results=witness_results,
        release_decision=release_decision,
        reference_access_time=reference_access_time,
    )

    # Step 10's unchanged-checkpoint confirmation (Requirement 13.1): the
    # hashes recomputed here are compared by the caller/task 21.2 against
    # `checkpoint_paths`' values verified in `main()`'s step 3 -- if a
    # checkpoint changed during the run, the two will disagree.
    checkpoint_hashes_after = {
        name: _file_sha256(path) for name, path in checkpoint_paths.items()
    }

    return LockedEvaluationResult(
        reference_access_timestamp=reference_access_time,
        recovery_errors=recovery_errors,
        comparison_results=comparison_results,
        holm_adjusted_p_values=holm_adjusted_p_values,
        witness_results=witness_results,
        release_decision=release_decision,
        checkpoint_hashes_after=checkpoint_hashes_after,
        audit_reports=audit_reports,
    )


def _build_locked_evaluation_audit_reports(
    *,
    token: LockedEvaluationToken,
    target_trial_spec: Any,
    recovery_estimates: dict[str, Any],
    comparison_results: dict[str, Any],
    witness_results: list[Any],
    release_decision: Any,
    reference_access_time: Any,
) -> dict[str, Any]:
    """Build one `AuditReport` per candidate estimate: every recovery
    estimate plus every frozen comparison's paired-bootstrap result
    (Requirement 12.1, 12.2, 12.4).

    Every report carries `locked_evaluation_id=token.evaluation_id` and
    `reference_access_timestamp=reference_access_time` (Requirement
    12.4's "locked-evaluation-only" fields) -- this is the one call site
    in the whole pipeline where those two fields are populated, since
    every other `build_audit_report` caller runs at the development
    stage, before any locked evaluation exists.

    `estimate_status` follows `release_decision.released` for every
    report built here: the release gate is evaluated once, over the
    identical test patients, and applies uniformly to every candidate
    estimate this locked evaluation produced -- there is no per-estimate
    release decision at this stage (unlike the sensitivity-variant audit
    reports `scripts/run_causal_analysis.py` builds, where each variant
    could in principle have its own witness/overlap profile).
    """
    from uncertainty.bootstrap import BootstrapCI
    from witnesses.audit_report import build_audit_report

    estimate_status = "released" if release_decision.released else "suppressed"
    reports: dict[str, Any] = {}

    for quantity_name, estimate in recovery_estimates.items():
        reports[f"recovery_{quantity_name}"] = build_audit_report(
            protocol=target_trial_spec,
            population={"description": "locked-evaluation test split"},
            treatment={"quantity": quantity_name},
            estimate=BootstrapCI(
                point_estimate=estimate.estimate,
                ci_low=float("nan"),
                ci_high=float("nan"),
                n_replicates=0,
                n_successful=0,
            ),
            overlap=_locked_evaluation_overlap_placeholder(),
            witness_results=witness_results,
            abstentions=[],
            violations=[],
            provenance={"evaluation_id": token.evaluation_id},
            release_decision=release_decision,
            estimate_status=estimate_status,
            dataset_checksum=token.dataset_checksum,
            split_hash=token.split_hash,
            resolved_config_hash=token.resolved_config_hash,
            checkpoint_hashes=dict(token.checkpoint_hashes),
            seed=0,
            sensitivity_variant=None,
            supported_count=0,
            abstained_count=0,
            locked_evaluation_id=token.evaluation_id,
            reference_access_timestamp=reference_access_time,
        )

    for comparison_name, bootstrap_ci in comparison_results.items():
        reports[f"comparison_{comparison_name}"] = build_audit_report(
            protocol=target_trial_spec,
            population={"description": "locked-evaluation test split"},
            treatment={"comparison": comparison_name},
            estimate=bootstrap_ci,
            overlap=_locked_evaluation_overlap_placeholder(),
            witness_results=witness_results,
            abstentions=[],
            violations=[],
            provenance={"evaluation_id": token.evaluation_id},
            release_decision=release_decision,
            estimate_status=estimate_status,
            dataset_checksum=token.dataset_checksum,
            split_hash=token.split_hash,
            resolved_config_hash=token.resolved_config_hash,
            checkpoint_hashes=dict(token.checkpoint_hashes),
            seed=0,
            sensitivity_variant=None,
            supported_count=0,
            abstained_count=0,
            locked_evaluation_id=token.evaluation_id,
            reference_access_timestamp=reference_access_time,
        )

    return reports


def _locked_evaluation_overlap_placeholder() -> Any:
    """A placeholder `OverlapReport` for `build_audit_report`'s `overlap`
    argument. This locked-evaluation step does not itself compute overlap
    diagnostics (that happens at the development stage, over the
    development split); the audit report's own overlap-diagnostics field
    is therefore not meaningful at this call site and is populated with
    an explicit all-`False` placeholder, mirroring
    `scripts/run_ablations.py::_not_evaluated_overlap_report`'s
    documented convention for the same situation.
    """
    from causal.overlap import CommonSupportSummary, OverlapReport

    return OverlapReport(
        arm_counts={},
        arm_propensity_summary={},
        common_support=CommonSupportSummary(
            primary_count=0, primary_fraction=0.0, sensitivity_count=0, sensitivity_fraction=0.0
        ),
        extreme_weight_count=0,
        extreme_weight_fraction=0.0,
        ess_by_arm={},
        smd_by_covariate={},
        arm_counts_sufficient=False,
        ess_sufficient=False,
        smd_sufficient=False,
        no_unresolved_positivity_warning=False,
        support_valid=False,
    )


# ---------------------------------------------------------------------------
# Task 21.2: the atomic commit (design.md Section 3.12 step 10;
# Requirement 13.10, 13.11).
# ---------------------------------------------------------------------------


class CommitFailedError(Exception):
    """Raised by `commit_locked_evaluation` when a step after hash
    verification fails (Requirement 13.11).

    `failing_step` names exactly which step failed (`"staging"`,
    `"checkpoint_unchanged_check"`, `"rename"`, or `"consumption_append"`),
    so a caller can report it without having to parse a message string.
    """

    def __init__(self, failing_step: str, message: str) -> None:
        super().__init__(message)
        self.failing_step = failing_step


def commit_locked_evaluation(
    token: LockedEvaluationToken,
    result: LockedEvaluationResult,
    *,
    checkpoint_hashes_before: dict[str, str],
    results_root: Path,
) -> Path:
    """Commit `result`'s artifacts as one indivisible operation
    (`design.md` Section 3.12 step 10; Requirement 13.10, 13.11).

    Stages every artifact under
    `results/locked_evaluation/.staging_<evaluation_id>/`, confirms every
    loaded checkpoint's hash is unchanged (recomputed once more here,
    comparing `result.checkpoint_hashes_after` -- itself already
    recomputed once, at the end of `run_locked_evaluation` -- against
    `checkpoint_hashes_before`, the values `main()`'s step 3 verified
    against the token before any evaluation ran), then commits with a
    single `os.rename` of the staging directory into
    `results/locked_evaluation/<evaluation_id>/`, and finally appends to
    `consumed.jsonl`.

    On ANY failure after staging begins, the staging directory is deleted
    and every pre-existing artifact (including the token, the evaluation
    id's consumption record, and the target path itself) is left
    byte-identical -- this function either fully commits or changes
    nothing on disk outside its own staging directory, which is itself
    always cleaned up (Requirement 13.11).

    Args:
        token: the verified token this result was produced under.
        result: `run_locked_evaluation`'s output.
        checkpoint_hashes_before: `{logical_name: sha256}` as verified by
            `main()`'s step 3 (`verify_bound_hashes`), i.e. the hashes
            BEFORE the evaluation steps ran.
        results_root: the top-level results directory.

    Returns:
        The committed path, `results/locked_evaluation/<evaluation_id>/`.

    Raises:
        CommitFailedError: naming the failing step, if a checkpoint
            changed during the run, the rename fails, or the consumption
            append fails. The staging directory is deleted and nothing
            pre-existing is touched in every one of these cases.
    """
    import shutil

    staging = staging_dir(results_root, token.evaluation_id)
    target = locked_evaluation_result_dir(results_root, token.evaluation_id)

    try:
        staging.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise CommitFailedError("staging", f"could not create staging directory: {exc}") from exc

    try:
        # Requirement 13.1's unchanged-checkpoint confirmation: compare the
        # hashes verified before evaluation (`checkpoint_hashes_before`)
        # against those recomputed after (`result.checkpoint_hashes_after`).
        # Any disagreement means a checkpoint file changed mid-run -- the
        # evaluation read something other than what was verified, so the
        # whole result is untrustworthy and nothing commits.
        changed = sorted(
            name
            for name in checkpoint_hashes_before
            if result.checkpoint_hashes_after.get(name) != checkpoint_hashes_before[name]
        )
        if changed:
            raise CommitFailedError(
                "checkpoint_unchanged_check",
                f"checkpoint(s) changed during the run, after hash verification "
                f"but before commit: {', '.join(changed)}. Refusing to commit.",
            )

        _stage_locked_evaluation_artifacts(staging, token, result)

        try:
            os.rename(staging, target)
        except OSError as exc:
            raise CommitFailedError(
                "rename", f"could not rename staging directory into place: {exc}"
            ) from exc

        try:
            append_consumed(token.evaluation_id, results_root, result_directory=str(target))
        except Exception as exc:
            # The rename already happened -- the result directory IS in
            # place. A failure here is a real, reportable problem (the
            # consumption record is now out of sync with the committed
            # artifact), but per Requirement 13.11's scope ("any step
            # after successful token verification" that fails leaves
            # pre-existing artifacts byte-identical), this failure comes
            # after the one irreversible step (the rename) and so cannot
            # itself be undone by deleting a staging directory that no
            # longer exists. It is surfaced distinctly so a caller does
            # not mistake it for a staging-time failure that left nothing
            # committed.
            raise CommitFailedError(
                "consumption_append",
                f"result directory committed at {target}, but appending to "
                f"consumed.jsonl failed: {exc}. The evaluation_id may need to be "
                "recorded manually; the result artifact itself is valid.",
            ) from exc

        return target
    except CommitFailedError:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    except Exception as exc:  # noqa: BLE001 -- any other failure is also a commit failure
        if staging.exists():
            shutil.rmtree(staging)
        raise CommitFailedError("staging", f"unexpected failure during staging: {exc}") from exc


def _stage_locked_evaluation_artifacts(
    staging: Path, token: LockedEvaluationToken, result: LockedEvaluationResult
) -> None:
    """Write every artifact `commit_locked_evaluation` is about to make
    permanent into the staging directory.

    A separate function so a staging-time failure (e.g. a malformed
    artifact) is caught by `commit_locked_evaluation`'s own exception
    handling and reported as a `"staging"`-step failure, rather than
    partially writing artifacts directly into `commit_locked_evaluation`'s
    body.
    """
    from witnesses.audit_io import write_audit_artifacts

    summary = {
        "evaluation_id": token.evaluation_id,
        "reference_access_timestamp": (
            result.reference_access_timestamp.isoformat()
            if result.reference_access_timestamp is not None
            else None
        ),
        "recovery_errors": result.recovery_errors,
        "holm_adjusted_p_values": result.holm_adjusted_p_values,
        "comparison_results": {
            name: {
                "point_estimate": ci.point_estimate,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "n_replicates": ci.n_replicates,
                "n_successful": ci.n_successful,
            }
            for name, ci in result.comparison_results.items()
        },
        "witness_outcomes": {r.witness: r.status for r in result.witness_results},
        "release_decision": {
            "released": result.release_decision.released,
            "reasons": list(result.release_decision.reasons),
        },
        "checkpoint_hashes_after": result.checkpoint_hashes_after,
    }
    (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    # `write_audit_artifacts` creates `staging/audit/{public,protected}/`
    # itself; one call per candidate estimate.
    for claim_id, report in result.audit_reports.items():
        write_audit_artifacts(report, staging, claim_id=claim_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Locked_Evaluation_Runner: open the sealed reference and the test "
            "split exactly once per minted LockedEvaluationToken, compute "
            "treatment-effect recovery against the sealed reference, run the "
            "paired bootstrap and Holm correction, evaluate the witnesses and "
            "the release gate, and commit the result atomically. Exposes no "
            "option to fit, tune, or select any parameter or threshold "
            "(Requirement 13.1, 13.2)."
        )
    )
    parser.add_argument(
        "--token-artifact",
        type=Path,
        required=True,
        help="Path to the token JSON artifact written by check_go_no_go.py --token-output.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Top-level results directory (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--resolved-config-hash",
        required=True,
        help="Recomputed resolved-config hash for this run, compared against the token.",
    )
    parser.add_argument(
        "--dataset-checksum",
        required=True,
        help="Recomputed dataset checksum for this run, compared against the token.",
    )
    parser.add_argument(
        "--split-hash",
        required=True,
        help="Recomputed split hash for this run, compared against the token.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "One checkpoint to load, as logical_name=path. Repeatable; must "
            "cover exactly the checkpoint names bound in the token."
        ),
    )
    return parser


def _parse_checkpoint_args(raw: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for entry in raw:
        name, sep, path_str = entry.partition("=")
        if not sep or not name.strip() or not path_str.strip():
            raise argparse.ArgumentTypeError(
                f"--checkpoint value {entry!r} is not of the form NAME=PATH"
            )
        paths[name.strip()] = Path(path_str.strip())
    return paths


def main(argv: list[str] | None = None) -> int:
    resolved_argv = list(argv) if argv is not None else sys.argv[1:]

    try:
        check_for_rejected_options(resolved_argv)
    except RejectedOptionError as exc:
        print(f"run_locked_evaluation failed: {exc}", file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args(resolved_argv)

    try:
        token = load_token_artifact(args.token_artifact)
    except TokenArtifactError as exc:
        print(f"run_locked_evaluation failed: {exc}", file=sys.stderr)
        return 1

    try:
        check_not_consumed(token, args.results_root)
    except AlreadyConsumedError as exc:
        print(f"run_locked_evaluation failed: {exc}", file=sys.stderr)
        return 1

    try:
        checkpoint_paths = _parse_checkpoint_args(args.checkpoint)
    except argparse.ArgumentTypeError as exc:
        print(f"run_locked_evaluation failed: {exc}", file=sys.stderr)
        return 1

    try:
        verify_bound_hashes(
            token,
            resolved_config_hash=args.resolved_config_hash,
            dataset_checksum=args.dataset_checksum,
            split_hash=args.split_hash,
            checkpoint_paths=checkpoint_paths,
        )
    except HashMismatchError as exc:
        print(f"run_locked_evaluation failed: {exc}", file=sys.stderr)
        return 1

    try:
        verify_target_path_unoccupied(args.results_root, token.evaluation_id)
    except OccupiedPathError as exc:
        print(f"run_locked_evaluation failed: {exc}", file=sys.stderr)
        return 1

    # Task 21.2 implements the atomic commit under this verified token;
    # this CLI's own `main()` stops at verification (steps 1-5) until that
    # task wires the evaluation steps (task 21.1's `run_locked_evaluation`
    # library function, below) and the commit into this entry point. The
    # CLI deliberately does not yet call `run_locked_evaluation` itself --
    # that function needs the real `ObservationalizationPipelineResult`
    # this script has no loader for yet (no `--real-data-path`/dataset-
    # contract flags exist on this parser), so wiring it in now would mean
    # either a second, divergent loading path or a CLI surface this task
    # cannot honestly promise works end to end.
    print(
        "run_locked_evaluation: token verified, target path unoccupied. "
        f"evaluation_id={token.evaluation_id!r}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
