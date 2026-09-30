"""The sealed internal randomized reference (`design.md` Stage 4; Section 3.4;
"Correctness Property 1: Sealed reference isolation"; `decisions.md`
Sections 8-9).

`decisions.md` Section 8 defines the internal randomized reference as "the
effect estimated from the complete eligible arm-0/arm-1 dataset, prior to
observationalization," computed exactly once, before any splitting or
observationalization step, and withheld from model development until the
locked external-evaluation stage (`requirements.md` Section 9). Section 9's
frozen order of operations makes step 2 ("compute and seal the internal
randomized reference effect ... on that complete cohort, before any
splitting or selection") come strictly before step 3 (the dev/val/test
split). `design.md` Section 3.4 enforces this order structurally, not just
by convention: `SealedReference.__init__` only accepts a
`cohorts.cohort_construction.RandomizedCohort` (the Stage-3 pre-split
output), so it "cannot compile against a post-split dataset."

What "internal reference" means at this point in the build
-------------------------------------------------------------

The actual causal-estimator machinery this reference will eventually wrap
(Cox baselines, `src/causal/*`) does not exist yet — that is task 7.x. This
module does not block on that. Instead, `SealedReference` accepts an
`estimator: ReferenceEstimator` callable at construction time: a
`RandomizedCohort -> InternalReference` function computed once, immediately,
inside `__init__`, over the complete pre-split cohort. A caller with no real
causal estimator yet may pass a placeholder estimator (see
`placeholder_estimator` below); a later task (7.x) wires in a real one (e.g.
a Cox-model hazard-ratio estimate) by passing a different `estimator`
argument — `SealedReference`'s public contract (`__init__(cohort, estimator)`,
`.unseal(unlock_token)`) does not change either way.

The estimator runs eagerly, at construction time, not lazily inside
`.unseal()`: `decisions.md` Section 8 requires the reference to be
"computed exactly once, on the complete eligible arm-0/arm-1 cohort, before
any splitting or observationalization step." Computing it at construction
time (when only the pre-split `RandomizedCohort` could possibly be in
scope) rather than deferring computation to whenever `.unseal()` happens to
first be called removes any risk of the estimator accidentally being
handed a later, post-split or post-observationalization object.

Token minting
--------------

`LockedEvaluationToken` is only mintable via its `_mint()` classmethod.
Nothing about `_mint()`'s current implementation is meant to be the real
go/no-go enforcement mechanism — that arrives in task 16
(`scripts/check_go_no_go.py`, minted only "after every checklist item
passes," `decisions.md` Section 8/Section 21). Today, `_mint()` exists only
so that:

1. a caller cannot construct a valid token by accident (e.g. by passing
   `None` or a bare string where a token is expected — see
   `src/infra/reference_guard.py`);
2. every minted token records who minted it and when, which task 16's real
   implementation will populate meaningfully (e.g. `issued_by="scripts/
   check_go_no_go.py"`), and which tests can use to construct a
   deliberately-minted token today.

`LockedEvaluationToken.__init__` is not private (Python has no true private
constructors), but every one of its nine fields is required with no default
(`neural-experiment-pipeline/design.md` Section 4.6; Requirement 13.3), so
there is no accidental, argument-free — or even partially-argued — way to
produce one. `_mint()` is documented as the intended call site so task 16
has an obvious extension point.

What the nine bound fields buy (Requirement 13.3-13.4)
-------------------------------------------------------

`neural-experiment-pipeline/design.md` Section 2.6 tension 5 records the
problem the original two-field token had: it "carries only `issued_by` and
`issued_at`, so it authorizes nothing in particular." A token that names
its minter but not *what* it authorizes cannot support Requirement 13.5's
"recompute and compare every hash bound in the token against the artifacts
present," because nothing is bound.

The fix deliberately does **not** touch
`infra.reference_guard.assert_reference_unlocked`, which stays an
`isinstance(token, LockedEvaluationToken)` check against this exact class.
Adding required fields to the same class strictly *strengthens* that
check — every value that passes it now carries nine validated bindings —
whereas duck typing an "extended token" protocol, or accepting a second
token class, would have weakened it. Class identity and the
`@dataclass(frozen=True)` decorator are unchanged for exactly that reason.

`__post_init__` raises a single `TokenFieldError` naming *every* absent or
empty field at once rather than the first one it meets (Requirement 13.4:
"SHALL name every offending field"). `TokenFieldError` subclasses
`ValueError`, so the original empty-`issued_by` rejection — which raised a
bare `ValueError` — is subsumed rather than replaced, and callers that
caught `ValueError` keep working.

Consumption is *not* a token field: the token is frozen and may be
serialized to `token_verification.json`, so "has this evaluation already
run?" lives in an append-only ledger instead
(`infra.consumption_ledger`, `results/locked_evaluation/consumed.jsonl`).
`_mint()` requires a `results_root` and refuses to mint an
`evaluation_id` that ledger already records (Requirement 13.4's
"duplicated"); task 21's locked-evaluation runner checks the same ledger
again before opening anything (Requirement 13.6).

Recording reference access (Requirements 14.5, 12.4)
-----------------------------------------------------

Every authorized read of the sealed reference --- `SealedReference.unseal`
and `read_sealed_artifact` --- appends `{evaluation_id, accessed_at}` to
`<results_root>/locked_evaluation/reference_access.jsonl` immediately after
its existing `assert_reference_unlocked` call, which is what supplies the
audit report's reference-access timestamp. Both take an optional
keyword-only `results_root`, falling back to `CWPWM_RESULTS_ROOT`; when
neither is available they *refuse the access* rather than return the
reference unrecorded. `infra.consumption_ledger`'s docstring records why
the root is resolved that way rather than being bound to the token or
derived from the artifact path.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from causal.baselines import (
    HORIZON_WEEKS,
    CausalEstimate,
    _censoring_kaplan_meier_weeks,
    _elapsed_weeks,
    _survival_at_weeks,
    unadjusted_cox,
)
from cohorts.cohort_construction import PatientRecord, RandomizedCohort
from infra.consumption_ledger import (
    append_reference_access,
    consumed_ledger_path,
    is_consumed,
)
from infra.reference_guard import assert_reference_unlocked

#: The current sealed-artifact format version (`design.md` Section 2.6
#: "Tension 4"; Section 4.5; Requirement 14.6). Version 1 artifacts carried
#: only the log-hazard-ratio `value` and recorded no version field at all;
#: version 2 additionally carries the `quantities` mapping (log-HR plus the
#: two arm-wise 96-week risks and their difference).
#:
#: Bumping this constant is *not* an in-place migration: see
#: `SealedFormatVersionError` and `RESEAL_INSTRUCTIONS` for why the only
#: supported upgrade path is a re-seal onto a new path.
SEALED_FORMAT_VERSION: int = 2

#: The documented re-seal step named by `SealedFormatVersionError`
#: (Requirement 14.7). `design.md` Section 2.6 "Tension 4" spells out why
#: this is a re-seal and not an upgrade: a version-1 artifact does not
#: contain the arm-wise risks and they cannot be recovered from a log-HR,
#: so there is deliberately no code path that reads a v1 artifact and
#: writes a v2 one. The v1 artifact is left byte-identical and unread.
RESEAL_INSTRUCTIONS: str = (
    "Re-seal (there is no in-place upgrade path; the version-1 artifact is "
    "left byte-identical and unread): "
    "(1) re-run Stage 3 to rebuild the complete eligible arm-0/arm-1 cohort "
    "(pre-split, pre-observationalization, per decisions.md Section 9's "
    "frozen order of operations); "
    "(2) call SealedReference(cohort, extended_randomized_reference_estimator), "
    "which computes all three quantities in one pass; "
    "(3) write to the NEW path "
    "reference/internal_randomized_reference.v2.sealed.json. "
    "The re-seal entry point is "
    "`scripts/run_stage.py --stage seal-reference --reseal`, which refuses "
    "to run if a version-2 artifact already exists at the target path."
)


class SealedFormatVersionError(ValueError):
    """A loaded sealed artifact carries a format version older than
    `SEALED_FORMAT_VERSION` (`design.md` Error Handling: "Old-format sealed
    artifact loaded -> `SealedFormatVersionError` naming the documented
    re-seal step; raised after authorization, before payload parsing";
    Requirement 14.7).

    A missing `format_version` field is treated as version 1 (the format
    that predates the field), not as "unknown, assume current".
    """


class ScaleMismatchError(ValueError):
    """A reference comparison paired an estimate with a sealed-reference
    quantity on a different scale (`design.md` Error Handling: "Scale
    mismatch in a reference comparison -> `ScaleMismatchError` from
    `compare_to_reference`"; Requirements 14.3, 14.4).

    The canonical instance is comparing a risk-difference estimate against
    the log-hazard-ratio reference quantity, which is not a smaller number
    on the same scale -- it is a different quantity entirely.
    """


class TokenFieldError(ValueError):
    """One or more `LockedEvaluationToken` fields are absent, empty, or
    malformed (`neural-experiment-pipeline/design.md` Section 4.6;
    Requirement 13.4).

    Always names *every* offending field in a single message, never just
    the first one met, so a minter's operator learns the full set of
    unresolved bindings from one failed run.

    Subclasses `ValueError` deliberately: before the token grew to nine
    bound fields, `__post_init__` raised a bare `ValueError` for an empty
    `issued_by`. That rejection is subsumed by this error rather than
    replaced, so existing `pytest.raises(ValueError)` call sites and any
    `except ValueError` handler keep behaving identically.
    """


#: The `LockedEvaluationToken` fields that must be present and non-empty,
#: in declaration order (Requirement 13.3). Kept as data rather than as a
#: hand-written sequence of `if` statements so `__post_init__` cannot drift
#: out of sync with the dataclass, and so tests can enumerate the bound
#: fields (e.g. task 20's Property 26 over offending subsets) instead of
#: hardcoding a parallel list.
BOUND_TOKEN_FIELDS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class LockedEvaluationToken:
    """An unforgeable-by-accident credential, bound to the exact artifacts
    it authorizes, required to unseal the internal randomized reference or
    open the test split.

    Real minting logic lives in `scripts/check_go_no_go.py`, the sole real
    caller of `_mint()`, which mints only once every go/no-go checklist item
    passes and the readiness ledger is complete. Do not construct this
    directly outside of `_mint()`/tests: every field is required with no
    default, so there is no constructor call that produces a partially bound
    token by accident (`design.md` Section 4.6; Requirement 13.3).

    Attributes:
        issued_by: identifies the minting process (e.g.
            `"scripts/check_go_no_go.py"`). Never empty.
        issued_at: when the token was minted (UTC).
        resolved_config_hash: `infra.registry`-style hash of the
            `ResolvedConfig` this evaluation is bound to.
        dataset_checksum: checksum of the dataset the evaluation may read.
        split_hash: hash of the frozen dev/val/test assignment.
        checkpoint_hashes: one `(logical_name, sha256)` pair per frozen
            checkpoint the locked evaluation loads — non-empty (an
            evaluation that loads no checkpoint has nothing to evaluate)
            and sorted, so the bound set has one canonical ordering
            regardless of the order the minter discovered the files in.
        git_commit: the commit the evaluation is bound to.
        statistical_plan_version: version of `decisions.md` Section 19's
            frozen statistical plan the evaluation follows.
        evaluation_id: unique across every previously minted token; checked
            against `infra.consumption_ledger` at mint time.

    Every one of these is a *binding*, not provenance decoration:
    Requirement 13.5 has the locked-evaluation runner recompute and compare
    each hash against the artifacts actually present before it opens the
    test split or the sealed reference.

    Raises:
        TokenFieldError: from `__post_init__`, naming every field that is
            absent, empty, or malformed — all of them in one message.
    """

    issued_by: str
    issued_at: datetime
    resolved_config_hash: str
    dataset_checksum: str
    split_hash: str
    checkpoint_hashes: tuple[tuple[str, str], ...]
    git_commit: str
    statistical_plan_version: str
    evaluation_id: str

    def __post_init__(self) -> None:
        """Reject any absent, empty, or malformed binding, naming all of
        them at once (Requirement 13.4).

        Subsumes the original empty-`issued_by` check: `issued_by` is one
        of the eight string fields validated below, and `TokenFieldError`
        is a `ValueError`.
        """
        offenders = _token_field_offenses(
            {name: getattr(self, name) for name in BOUND_TOKEN_FIELDS}
        )
        if offenders:
            raise TokenFieldError(_format_token_field_error(offenders))

    @classmethod
    def _mint(
        cls,
        issued_by: str,
        *,
        resolved_config_hash: str,
        dataset_checksum: str,
        split_hash: str,
        checkpoint_hashes: Sequence[tuple[str, str]],
        git_commit: str,
        statistical_plan_version: str,
        evaluation_id: str,
        results_root: str | Path,
    ) -> LockedEvaluationToken:
        """Mint a new token bound to the nine fields Requirement 13.3 names.

        Intended sole real caller: `scripts/check_go_no_go.py`, once every
        go/no-go checklist item has passed and the readiness ledger is
        complete. Any other caller (including tests) must go through this
        same method — there is no other path to a valid token that does not
        have to supply every binding itself.

        Every new field is keyword-only. A minter binding nine hashes,
        several of which are same-shaped hex strings, must not be able to
        transpose `dataset_checksum` and `split_hash` positionally.

        Args:
            issued_by: identifies the minting process. Non-empty.
            resolved_config_hash: hash of the bound `ResolvedConfig`.
            dataset_checksum: checksum of the bound dataset.
            split_hash: hash of the bound dev/val/test assignment.
            checkpoint_hashes: `(logical_name, sha256)` pairs, one per
                frozen checkpoint. Sorted here so the caller does not have
                to; must be non-empty and every element must be a pair of
                non-empty strings.
            git_commit: the bound commit identifier.
            statistical_plan_version: the bound statistical-plan version.
            evaluation_id: must not already appear in the consumption
                ledger under `results_root`.
            results_root: the top-level results directory holding
                `locked_evaluation/consumed.jsonl`. Required, not
                defaulted: Requirement 13.4's duplicate check is not
                something a caller should be able to skip by omitting an
                argument.

        Returns:
            The minted, fully bound `LockedEvaluationToken`.

        Raises:
            TokenFieldError: if any binding is absent, empty, or malformed
                (every offender named at once), or if `evaluation_id` is
                already recorded as consumed.
            infra.consumption_ledger.ConsumptionLedgerError: if the
                consumption ledger exists but cannot be read.
        """
        normalized_checkpoints = (
            tuple(sorted(tuple(pair) for pair in checkpoint_hashes))
            if isinstance(checkpoint_hashes, Sequence)
            and not isinstance(checkpoint_hashes, str | bytes)
            else checkpoint_hashes
        )

        candidate: dict[str, Any] = {
            "issued_by": issued_by,
            "issued_at": datetime.now(UTC),
            "resolved_config_hash": resolved_config_hash,
            "dataset_checksum": dataset_checksum,
            "split_hash": split_hash,
            "checkpoint_hashes": normalized_checkpoints,
            "git_commit": git_commit,
            "statistical_plan_version": statistical_plan_version,
            "evaluation_id": evaluation_id,
        }

        # Field validation runs here as well as in `__post_init__` so a
        # refusal to mint reports *both* kinds of offense in one message:
        # a malformed field and a duplicated `evaluation_id`. Letting
        # `__post_init__` raise first would hide the duplicate behind the
        # field errors, and checking the ledger first would hide the field
        # errors behind the duplicate.
        offenders = _token_field_offenses(candidate)
        if (
            isinstance(evaluation_id, str)
            and evaluation_id
            and is_consumed(evaluation_id, results_root)
        ):
            offenders["evaluation_id"] = (
                f"already recorded as consumed in "
                f"{consumed_ledger_path(results_root)}; an evaluation "
                "identifier is single-use across every token ever minted"
            )
        if offenders:
            raise TokenFieldError(_format_token_field_error(offenders))

        return cls(**candidate)


def _token_field_offenses(values: Mapping[str, Any]) -> dict[str, str]:
    """Every absent/empty/malformed binding in `values`, name -> reason.

    Shared by `LockedEvaluationToken.__post_init__` and `_mint` so the two
    cannot disagree about what "absent or empty" means. Returns a mapping
    rather than raising, so a caller (`_mint`) can fold in offenses this
    function cannot see — the duplicated-`evaluation_id` ledger check — and
    still report everything in one error (Requirement 13.4).

    `checkpoint_hashes` is the only structurally interesting field: it must
    be a non-empty, sorted tuple of `(logical_name, sha256)` pairs of
    non-empty strings. Sortedness is checked rather than silently repaired
    because the dataclass is frozen and `_mint` already sorts on the way
    in; a directly constructed token that skipped `_mint` should be told
    its bound set has no canonical ordering, not quietly accepted.
    """
    offenses: dict[str, str] = {}

    for name in BOUND_TOKEN_FIELDS:
        if name in ("issued_at", "checkpoint_hashes"):
            continue
        value = values.get(name)
        if value is None:
            offenses[name] = "absent (None)"
        elif not isinstance(value, str):
            offenses[name] = f"not a string (got {type(value).__name__})"
        elif not value.strip():
            offenses[name] = "empty"

    issued_at = values.get("issued_at")
    if issued_at is None:
        offenses["issued_at"] = "absent (None)"
    elif not isinstance(issued_at, datetime):
        offenses["issued_at"] = f"not a datetime (got {type(issued_at).__name__})"

    checkpoint_hashes = values.get("checkpoint_hashes")
    if checkpoint_hashes is None:
        offenses["checkpoint_hashes"] = "absent (None)"
    elif not isinstance(checkpoint_hashes, tuple):
        offenses["checkpoint_hashes"] = (
            f"not a tuple of (logical_name, sha256) pairs (got "
            f"{type(checkpoint_hashes).__name__})"
        )
    elif not checkpoint_hashes:
        offenses["checkpoint_hashes"] = (
            "empty; a locked evaluation must bind at least one checkpoint hash"
        )
    else:
        malformed = [
            index
            for index, pair in enumerate(checkpoint_hashes)
            if not (
                isinstance(pair, tuple)
                and len(pair) == 2
                and all(isinstance(part, str) and part.strip() for part in pair)
            )
        ]
        if malformed:
            offenses["checkpoint_hashes"] = (
                "entries at index "
                + ", ".join(str(index) for index in malformed)
                + " are not (logical_name, sha256) pairs of non-empty strings"
            )
        elif list(checkpoint_hashes) != sorted(checkpoint_hashes):
            offenses["checkpoint_hashes"] = (
                "not sorted; the bound checkpoint set must have one canonical "
                "ordering (LockedEvaluationToken._mint sorts for you)"
            )

    return offenses


def _format_token_field_error(offenses: Mapping[str, str]) -> str:
    """Render `offenses` as a single message naming every offending field.

    Ordered by `BOUND_TOKEN_FIELDS` (declaration order) rather than by
    discovery or alphabetically, so two runs with the same offending set
    produce the same message.
    """
    ordered = [name for name in BOUND_TOKEN_FIELDS if name in offenses]
    ordered += [name for name in offenses if name not in BOUND_TOKEN_FIELDS]
    detail = "; ".join(f"{name}: {offenses[name]}" for name in ordered)
    return (
        "Refusing to bind a LockedEvaluationToken: "
        f"{len(ordered)} offending field(s) -- {detail}. Every field listed "
        "in Requirement 13.3 must be present, non-empty, and unique; no "
        "partially bound token can exist (design.md Section 4.6)."
    )


#: The scale vocabulary, taken verbatim from
#: `causal.baselines.CausalEstimate.scale` (`"log_hazard_ratio"` for the Cox
#: variants, `"risk_difference"` for the outcome-regression/IPCW variants)
#: plus `"risk"` for a single-arm absolute risk, which no `CausalEstimate`
#: currently produces but the sealed reference does (Requirement 14.1).
#: Deliberately the *same* strings rather than a parallel enum, so
#: `compare_to_reference` can compare `CausalEstimate.scale` against
#: `ReferenceQuantity.scale` with a plain `==` (`design.md` Section 2.6
#: "Tension 4", scale safety).
ReferenceScale = Literal["log_hazard_ratio", "risk", "risk_difference"]


@dataclass(frozen=True)
class ReferenceQuantity:
    """One scale-tagged quantity carried by the extended sealed reference
    (`design.md` Section 4.5; Requirement 14.1).

    Attributes:
        value: the point estimate.
        scale: the scale `value` lives on, from `ReferenceScale` (the same
            vocabulary as `causal.baselines.CausalEstimate.scale`). This is
            what makes a comparison checkable rather than a matter of
            convention -- see `compare_to_reference`.
        horizon_weeks: the risk horizon `value` was evaluated at, or `None`
            for a horizon-free quantity such as a Cox log-hazard-ratio
            (which is a whole-follow-up quantity, not a horizon one).
    """

    value: float
    scale: ReferenceScale
    horizon_weeks: float | None


@dataclass(frozen=True)
class InternalReference:
    """The unsealed internal randomized reference value (`decisions.md`
    Section 8, item 2).

    Generic on purpose: task 7.x's causal baselines will eventually
    populate `value`/`metadata` with a real hazard-ratio estimate (and
    whatever standard error/CI accompanies it) computed over the complete
    pre-split `RandomizedCohort`. Until then, callers may populate this
    with any placeholder statistic without this type, or
    `SealedReference`'s contract, needing to change.

    Attributes:
        value: the computed statistic (e.g. a hazard ratio). `Any` because
            the concrete estimator is not fixed yet.
        cohort_size: number of patients the estimate was computed over —
            recorded so downstream consumers/audits don't need to
            recompute it or re-open the cohort.
        computed_at: when the estimate was computed (UTC) — provenance,
            not a substitute for the manifest's own timestamps.
        metadata: free-form extra provenance (e.g. per-arm counts,
            standard error, confidence interval, estimator name). Empty by
            default.
        format_version: the sealed-artifact format version this reference
            conforms to. Defaults to `SEALED_FORMAT_VERSION`.
        quantities: the scale-tagged quantities (Requirement 14.1),
            populated by `extended_randomized_reference_estimator`:
            `"log_hazard_ratio"` (scale `log_hazard_ratio`, horizon
            `None`), `"risk_arm0"` and `"risk_arm1"` (scale `risk`, horizon
            `HORIZON_WEEKS`), and `"risk_difference"` (scale
            `risk_difference`, horizon `HORIZON_WEEKS`, value `risk_arm1 -
            risk_arm0`). Empty for the v1-shaped estimators
            (`placeholder_estimator`, `unadjusted_cox_reference_estimator`),
            which are left unchanged.

    `value` deliberately keeps its existing v1 meaning (the unadjusted Cox
    log-hazard-ratio, when a real estimator produced it) rather than being
    repurposed into a mapping: `design.md` Section 4.5 requires existing v1
    consumers to be unaffected, so new consumers read `quantities` and old
    ones keep reading `value`.
    """

    value: Any
    cohort_size: int
    computed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: int = SEALED_FORMAT_VERSION
    quantities: Mapping[str, ReferenceQuantity] = field(default_factory=dict)


#: A `RandomizedCohort -> InternalReference` callable, computed exactly once
#: at `SealedReference` construction time. Task 7.x's causal baselines will
#: supply a real implementation (e.g. one that fits a Cox model and returns
#: the hazard ratio); see module docstring.
ReferenceEstimator = Callable[[RandomizedCohort], InternalReference]


def placeholder_estimator(cohort: RandomizedCohort) -> InternalReference:
    """A placeholder `ReferenceEstimator` for use before a real
    causal-estimator is wired in for a given call site.

    Records only the cohort size and per-arm counts — no actual treatment
    effect is computed. This exists so `SealedReference` can be constructed
    and exercised (including in tests) without inventing a fake numeric
    estimate that could be mistaken for a real one. Real pipeline entry
    points (`cohorts.pipeline.run_observationalization_pipeline`'s default,
    real-data CLI stages) should pass `unadjusted_cox_reference_estimator`
    (or another real `ReferenceEstimator`) instead of this function; tests
    that only need *a* `SealedReference` to exist (not a real estimate) may
    continue to use this placeholder.
    """
    return InternalReference(
        value=None,
        cohort_size=len(cohort),
        computed_at=datetime.now(UTC),
        metadata={
            "estimator": "placeholder_estimator",
            "note": (
                "No real causal estimator wired in for this call site. This "
                "InternalReference carries no treatment-effect value."
            ),
            "arm_0_count": len(cohort.by_arm(0)),
            "arm_1_count": len(cohort.by_arm(1)),
        },
    )


def unadjusted_cox_reference_estimator(cohort: RandomizedCohort) -> InternalReference:
    """The real internal-randomized-reference estimator: unadjusted Cox PH
    (`causal.baselines.unadjusted_cox`) on the complete, pre-split
    `RandomizedCohort` (`decisions.md` Section 8, item 2: "the effect
    estimated from the complete eligible arm-0/arm-1 dataset, prior to
    observationalization").

    Deliberately the *unadjusted* Cox baseline, not `adjusted_cox` or any
    propensity-weighted variant: the internal randomized reference's whole
    point is to capture the effect randomization alone identifies, on the
    complete randomized (pre-selection) cohort, before any covariate
    adjustment for confounding that only observationalization introduces.
    `InternalReference.value` is the fitted `treatment_arm` log-hazard-ratio
    coefficient (`causal.baselines.CausalEstimate.estimate`,
    `scale="log_hazard_ratio"`); `metadata` additionally records the
    standard error (if computed) and per-arm counts for provenance, exactly
    like `placeholder_estimator`.

    This function has the exact `ReferenceEstimator` signature
    (`RandomizedCohort -> InternalReference`) `SealedReference.__init__`
    requires, so it can be passed directly as the `estimator` argument
    wherever a real (non-placeholder) internal reference is needed --- e.g.
    `cohorts.pipeline.run_observationalization_pipeline`'s
    `reference_estimator` parameter for a real-data run.
    """
    estimate = unadjusted_cox(list(cohort.patients))
    return InternalReference(
        value=estimate.estimate,
        cohort_size=len(cohort),
        computed_at=datetime.now(UTC),
        metadata={
            "estimator": "unadjusted_cox_reference_estimator",
            "scale": estimate.scale,
            "se": estimate.se,
            "arm_0_count": len(cohort.by_arm(0)),
            "arm_1_count": len(cohort.by_arm(1)),
        },
    )


def _arm_risk_at_horizon(
    patients: Sequence[PatientRecord], horizon_weeks: float = HORIZON_WEEKS
) -> float:
    """Kaplan-Meier estimate of the cumulative endpoint risk by
    `horizon_weeks` for one arm (`design.md` Section 2.6 "Tension 4": "a
    Kaplan-Meier evaluation at `HORIZON_WEEKS=96.0` per arm via
    `cohort.by_arm(arm)`").

    Composed entirely from `causal.baselines`' existing, tested pieces
    rather than a new survival implementation:
    `causal.baselines._elapsed_weeks` for the time scale (the same
    treatment-assignment-to-event elapsed-weeks definition every baseline
    uses), `causal.baselines._censoring_kaplan_meier_weeks` for the
    product-limit curve over `(elapsed_weeks, is_event)` rows, and
    `causal.baselines._survival_at_weeks` to read the curve at the horizon.

    `_censoring_kaplan_meier_weeks` is named for its usual caller (the
    reverse-KM censoring curve of the IPCW baselines) but is a plain
    Kaplan-Meier estimator over `(time, is_event)` rows: passing
    `endpoint_result.event_indicator` as `is_event` (rather than
    `censoring_indicator`) yields the *endpoint* survival curve, so
    `1 - S(horizon)` is the arm's cumulative endpoint risk. Patients
    censored before the horizon are handled by the product-limit estimator
    itself (they leave the risk set at their censoring time), which is why
    this is a Kaplan-Meier risk and not the cruder binary
    `_horizon_outcome` proportion the outcome-regression baseline uses.

    Returns:
        `1 - S_hat(horizon_weeks)` in `[0, 1]`, or `nan` for an empty arm
        (no risk set at all -- deliberately `nan` rather than `0.0`, which
        would read as "nobody had the endpoint", matching
        `causal.baselines.ipcw_ipw`'s `nan`-on-empty-denominator
        convention).
    """
    if not patients:
        return float("nan")
    rows = [
        (_elapsed_weeks(patient), bool(patient.endpoint_result.event_indicator))
        for patient in patients
    ]
    times, survival = _censoring_kaplan_meier_weeks(rows)
    return 1.0 - _survival_at_weeks(times, survival, horizon_weeks)


def extended_randomized_reference_estimator(cohort: RandomizedCohort) -> InternalReference:
    """The version-2 internal-randomized-reference estimator: the same
    unadjusted Cox log-hazard-ratio as `unadjusted_cox_reference_estimator`
    *plus* the two arm-wise 96-week Kaplan-Meier risks and their difference
    (`design.md` Section 2.6 "Tension 4", Section 4.5; Requirements 14.1,
    14.2).

    All three quantities are computed in one pass over `cohort` --- which,
    by `SealedReference.__init__`'s signature, can only be the complete
    eligible arm-0/arm-1 `RandomizedCohort`: the pre-split,
    pre-observationalization Stage-3 output required by `decisions.md`
    Section 9's frozen order of operations (Requirement 14.2). There is no
    parameter through which a post-split or post-observationalization
    dataset could reach this function.

    Composition, not a new estimator family (`design.md` Section 2.6
    "Tension 4"): `causal.baselines.unadjusted_cox` supplies the log-HR
    (bit-for-bit the same call `unadjusted_cox_reference_estimator` makes,
    over the same `list(cohort.patients)`), and `_arm_risk_at_horizon`
    supplies each arm's risk from the existing Kaplan-Meier helpers.

    `InternalReference.value` keeps its version-1 meaning exactly --- the
    fitted `treatment_arm` log-hazard-ratio coefficient, byte-identical to
    what `unadjusted_cox_reference_estimator` produces for the same cohort
    --- so existing version-1 consumers are unaffected; new consumers read
    `quantities` (`design.md` Section 4.5).

    Returns:
        An `InternalReference` with `format_version=SEALED_FORMAT_VERSION`
        and four `quantities` keys: `"log_hazard_ratio"` (scale
        `log_hazard_ratio`, horizon `None` --- a Cox coefficient is a
        whole-follow-up quantity, not a horizon one), `"risk_arm0"` and
        `"risk_arm1"` (scale `risk`, horizon `HORIZON_WEEKS`), and
        `"risk_difference"` (scale `risk_difference`, horizon
        `HORIZON_WEEKS`, value `risk_arm1 - risk_arm0`).
    """
    estimate = unadjusted_cox(list(cohort.patients))

    arm_0 = cohort.by_arm(0)
    arm_1 = cohort.by_arm(1)
    risk_arm0 = _arm_risk_at_horizon(arm_0, HORIZON_WEEKS)
    risk_arm1 = _arm_risk_at_horizon(arm_1, HORIZON_WEEKS)

    quantities: dict[str, ReferenceQuantity] = {
        "log_hazard_ratio": ReferenceQuantity(
            value=estimate.estimate, scale="log_hazard_ratio", horizon_weeks=None
        ),
        "risk_arm0": ReferenceQuantity(value=risk_arm0, scale="risk", horizon_weeks=HORIZON_WEEKS),
        "risk_arm1": ReferenceQuantity(value=risk_arm1, scale="risk", horizon_weeks=HORIZON_WEEKS),
        "risk_difference": ReferenceQuantity(
            value=risk_arm1 - risk_arm0,
            scale="risk_difference",
            horizon_weeks=HORIZON_WEEKS,
        ),
    }

    return InternalReference(
        value=estimate.estimate,
        cohort_size=len(cohort),
        computed_at=datetime.now(UTC),
        metadata={
            "estimator": "extended_randomized_reference_estimator",
            "scale": estimate.scale,
            "se": estimate.se,
            "arm_0_count": len(arm_0),
            "arm_1_count": len(arm_1),
            "horizon_weeks": HORIZON_WEEKS,
        },
        format_version=SEALED_FORMAT_VERSION,
        quantities=quantities,
    )


def compare_to_reference(
    estimate: CausalEstimate, reference: InternalReference, quantity: str
) -> float:
    """Compare `estimate` against the sealed reference's `quantity`, on a
    checked-matching scale (`design.md` Section 2.6 "Tension 4", scale
    safety; Requirements 14.3, 14.4; Correctness Property 24).

    The scale check is a plain `==` between two strings drawn from the same
    vocabulary --- `causal.baselines.CausalEstimate.scale` and
    `ReferenceQuantity.scale` deliberately share it (see `ReferenceScale`)
    --- rather than a convention a reviewer has to trust. The canonical
    misuse this exists to stop is comparing a risk-difference estimate
    against the log-hazard-ratio reference quantity: those are different
    quantities, not a bigger and smaller number on one scale
    (Requirement 14.4).

    Args:
        estimate: the model/baseline estimate, carrying its own `scale`.
        reference: the unsealed `InternalReference` (version 2 or later ---
            a version-1-shaped reference has no `quantities` and cannot be
            compared by name).
        quantity: the `reference.quantities` key to compare against, e.g.
            `"risk_difference"` or `"log_hazard_ratio"`.

    Returns:
        `abs(estimate.estimate - reference.quantities[quantity].value)`,
        the same absolute-error convention
        `evaluation.metrics.risk_difference_error_vs_internal_reference`
        reports.

    Raises:
        KeyError: if `quantity` is not among `reference.quantities` (which
            includes every version-1-shaped reference, whose `quantities`
            is empty). The message names the available keys.
        ScaleMismatchError: if `estimate.scale` differs from the selected
            quantity's `scale`.
    """
    if quantity not in reference.quantities:
        available = ", ".join(sorted(reference.quantities)) or "<none>"
        raise KeyError(
            f"Sealed reference carries no quantity named {quantity!r}; "
            f"available quantities: {available}. A format-version-1 "
            "reference carries no quantities at all -- see "
            "SealedFormatVersionError and RESEAL_INSTRUCTIONS."
        )

    reference_quantity = reference.quantities[quantity]
    if estimate.scale != reference_quantity.scale:
        raise ScaleMismatchError(
            f"Scale mismatch: estimate {estimate.name!r} is on scale "
            f"{estimate.scale!r} but sealed-reference quantity {quantity!r} "
            f"is on scale {reference_quantity.scale!r}. These are different "
            "quantities, not the same quantity at a different magnitude; "
            "select a reference quantity on the estimate's own scale "
            "(decisions.md Section 8; Requirements 14.3-14.4)."
        )

    return abs(estimate.estimate - reference_quantity.value)


class SealedReference:
    """The sealed internal randomized reference (`design.md` Stage 4).

    Constructible only from a `cohorts.cohort_construction.RandomizedCohort`
    (the pre-split, pre-observationalization Stage-3 output) — there is no
    constructor path that accepts a post-split or post-observationalization
    object, which is what makes `decisions.md` Section 9's frozen ordering
    ("compute and seal ... before any splitting or selection") a structural
    guarantee rather than a convention.

    The reference value is computed exactly once, eagerly, inside
    `__init__` (see module docstring for why), by calling `estimator(cohort)`.
    It is stored only in the private `_internal_reference` attribute; the
    only way to read it back out is `.unseal()`, which requires a valid
    `LockedEvaluationToken`.
    """

    def __init__(
        self,
        cohort: RandomizedCohort,
        estimator: ReferenceEstimator = placeholder_estimator,
    ) -> None:
        """Compute and seal the internal randomized reference.

        Args:
            cohort: the complete eligible arm-0/arm-1
                `RandomizedCohort` (Stage 3 output), before any splitting
                or observationalization.
            estimator: a `RandomizedCohort -> InternalReference` callable,
                invoked exactly once, immediately, over `cohort`. Defaults
                to `placeholder_estimator` since no real causal-estimator
                machinery exists yet (task 7.x will supply a real one).
        """
        self._internal_reference: InternalReference = estimator(cohort)

    def unseal(
        self, unlock_token: object, *, results_root: str | Path | None = None
    ) -> InternalReference:
        """Return the sealed `InternalReference`, if `unlock_token` is valid,
        recording the access (Requirement 14.5).

        Delegates the validity check to
        `infra.reference_guard.assert_reference_unlocked` rather than
        duplicating it here, so there is exactly one implementation of "is
        this a genuine token" (`design.md` Error Handling: "Attempt to read
        sealed reference before Stage 17 -> `ReferenceAccessError`, raised
        at the function-call boundary, not just logged").

        Immediately after that guard, and before the sealed value is
        returned, the access is appended to
        `<results_root>/locked_evaluation/reference_access.jsonl` as
        `{evaluation_id, accessed_at}` --- the record that supplies the
        audit report's reference-access timestamp (Requirements 14.5,
        12.4). The order matters in both directions: only an authorized
        access is recorded (the guard runs first), and no authorized access
        goes unrecorded (the append runs before the `return`, and its
        failure propagates instead of being swallowed).

        Args:
            unlock_token: must be a valid, minted `LockedEvaluationToken`
                (see `LockedEvaluationToken._mint`). Any other value
                (including `None` or a plain string) is rejected.
            results_root: the top-level results directory holding the
                reference-access ledger. Keyword-only and optional so every
                existing `unseal(token)` call site keeps working; when
                omitted the root comes from `CWPWM_RESULTS_ROOT`, and when
                that is unset too the access is *refused*, not performed
                unrecorded (see
                `infra.consumption_ledger.resolve_results_root` and that
                module's docstring for why this is not an optional
                convenience).

        Returns:
            The sealed `InternalReference`.

        Raises:
            ReferenceAccessError: if `unlock_token` is not a valid
                `LockedEvaluationToken` --- raised before anything is
                recorded.
            infra.consumption_ledger.ResultsRootNotConfiguredError: if the
                access cannot be recorded because no results root is
                resolvable. The reference is not returned.
        """
        assert_reference_unlocked(unlock_token)
        # `assert_reference_unlocked` has already established that this is a
        # `LockedEvaluationToken` (and `__post_init__` that its
        # `evaluation_id` is a non-empty string). `cast` states that for a
        # reader/type checker without adding a second runtime check that
        # could drift out of agreement with the guard.
        token = cast(LockedEvaluationToken, unlock_token)
        append_reference_access(token.evaluation_id, results_root)
        return self._internal_reference

    def write_sealed_artifact(self, path: str | Path) -> Path:
        """Persist this sealed reference to disk as `design.md` Section
        2.3's `reference/internal_randomized_reference.sealed.json`.

        This writes the REAL `InternalReference` value (not a placeholder,
        not metadata-only) to `path`, because `design.md` Section 2.3 names
        this file as the actual sealed-reference artifact later stages
        (Stage 17's locked-evaluation, `scripts/run_paper.py --stage
        locked-evaluation`) read once a valid `LockedEvaluationToken`
        exists -- a metadata-only stand-in would not satisfy that later
        requirement. What keeps this artifact "sealed" is NOT secrecy of
        the file's bytes (this is a research-benchmark reproducibility
        artifact stored in `results/<experiment_id>/reference/` on a
        filesystem the analyst already controls, not a security boundary
        against a determined local attacker) -- it is the *process-level*
        access-control convention every reader of this file must follow:

        1. this method is the ONLY writer, called exactly once, immediately
           after `__init__` computes `self._internal_reference` (see
           `write_sealed_artifact`'s caller in
           `cohorts.pipeline.run_observationalization_pipeline`, if wired
           in there, or a real-data CLI stage) -- there is no code path
           that re-derives or overwrites this file from a different
           (post-split/post-observationalization) cohort;
        2. `read_sealed_artifact` (the only reader) requires a valid
           `LockedEvaluationToken` argument and calls
           `assert_reference_unlocked` BEFORE opening `path` at all --
           mirroring `scripts/run_paper.py`'s "refuse before touching any
           file" convention -- so no pre-Stage-17 code path can read this
           file back even though it is not filesystem-encrypted;
        3. the file is written with `0o600` permissions (owner
           read/write only) as an additional, defense-in-depth OS-level
           speed bump against a different local user reading it -- not a
           substitute for check 2 above, which is the real gate.

        The written payload records `format_version:
        SEALED_FORMAT_VERSION` (Requirement 14.6) plus the scale-tagged
        `quantities` mapping, alongside the version-1 keys (`value`,
        `cohort_size`, `computed_at`, `metadata`) whose meanings are
        unchanged. A reference produced by a version-1-shaped estimator
        (`placeholder_estimator`, `unadjusted_cox_reference_estimator`)
        simply writes an empty `quantities` object.

        Args:
            path: destination file path (e.g.
                `results/<experiment_id>/reference/internal_randomized_reference.sealed.json`).

        Returns:
            The `Path` written to.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            # `format_version` is recorded on every write (Requirement
            # 14.6) so `read_sealed_artifact` never has to guess: an
            # artifact with no version field is, unambiguously, one written
            # before this field existed (version 1).
            "format_version": SEALED_FORMAT_VERSION,
            "value": self._internal_reference.value,
            "cohort_size": self._internal_reference.cohort_size,
            "computed_at": self._internal_reference.computed_at.isoformat(),
            "metadata": self._internal_reference.metadata,
            "quantities": {
                name: {
                    "value": quantity.value,
                    "scale": quantity.scale,
                    "horizon_weeks": quantity.horizon_weeks,
                }
                for name, quantity in self._internal_reference.quantities.items()
            },
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        # Owner read/write only (0o600) -- defense-in-depth, not the real
        # gate; see docstring point 3.
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
        return target


def read_sealed_artifact(
    path: str | Path, unlock_token: object, *, results_root: str | Path | None = None
) -> InternalReference:
    """Read a sealed-reference artifact written by
    `SealedReference.write_sealed_artifact`, given a valid
    `LockedEvaluationToken`.

    Mirrors `scripts/run_paper.py`'s locked-evaluation-gating convention
    exactly: `assert_reference_unlocked(unlock_token)` is called FIRST,
    before `path` is opened at all. A caller without a valid token gets
    `ReferenceAccessError` and this function never touches the filesystem.

    The access is appended to
    `<results_root>/locked_evaluation/reference_access.jsonl` immediately
    after that guard and *before* `path` is opened (Requirement 14.5). That
    ordering is deliberate: this function records the authorized attempt to
    open the sealed artifact, so an attempt that then fails on a missing
    file or an old format version still leaves a trace. Recording only
    successful parses would let a reader open the file, learn that it exists
    and at what version, and leave no audit record --- the direction of
    error that Requirement 14.5 exists to close.

    Args:
        path: the sealed-reference artifact's path (see
            `SealedReference.write_sealed_artifact`).
        unlock_token: must be a valid, minted `LockedEvaluationToken`.
        results_root: the top-level results directory holding the
            reference-access ledger. Keyword-only and optional so every
            existing call site keeps working; when omitted the root comes
            from `CWPWM_RESULTS_ROOT`, and when that is unset too the read
            is *refused* rather than performed unrecorded (see
            `SealedReference.unseal` and
            `infra.consumption_ledger.resolve_results_root`).

    Returns:
        The `InternalReference` reconstructed from the artifact.

    Raises:
        ReferenceAccessError: if `unlock_token` is not a valid
            `LockedEvaluationToken` -- raised before `path` is opened and
            before anything is recorded.
        infra.consumption_ledger.ResultsRootNotConfiguredError: if the
            access cannot be recorded because no results root is
            resolvable -- raised before `path` is opened.
        FileNotFoundError: if `path` does not exist (only checked after the
            token has already been validated).
        SealedFormatVersionError: if the artifact records a `format_version`
            below `SEALED_FORMAT_VERSION`, or records none at all (treated
            as version 1) -- raised after authorization and before the
            payload is parsed into an `InternalReference` (Requirement
            14.7; `design.md` Section 2.6 "Tension 4").
    """
    assert_reference_unlocked(unlock_token)
    # See `SealedReference.unseal` for why this is a `cast` and not a second
    # `isinstance` check.
    append_reference_access(cast(LockedEvaluationToken, unlock_token).evaluation_id, results_root)

    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    # Version gate first: nothing below is parsed until the format is known
    # to be readable. A missing field means "written before the field
    # existed", i.e. version 1 -- never "unknown, assume current".
    recorded_version = payload.get("format_version", 1)
    if not isinstance(recorded_version, int) or isinstance(recorded_version, bool):
        raise SealedFormatVersionError(
            f"Sealed artifact at {target} records a non-integer "
            f"format_version ({recorded_version!r}); it cannot be read. " + RESEAL_INSTRUCTIONS
        )
    if recorded_version < SEALED_FORMAT_VERSION:
        raise SealedFormatVersionError(
            f"Sealed artifact at {target} is format version "
            f"{recorded_version}"
            + (" (no format_version field recorded)" if "format_version" not in payload else "")
            + f", older than the required version {SEALED_FORMAT_VERSION}. "
            + RESEAL_INSTRUCTIONS
        )

    quantities = {
        name: ReferenceQuantity(
            value=entry["value"],
            scale=entry["scale"],
            horizon_weeks=entry["horizon_weeks"],
        )
        for name, entry in payload.get("quantities", {}).items()
    }

    return InternalReference(
        value=payload["value"],
        cohort_size=payload["cohort_size"],
        computed_at=datetime.fromisoformat(payload["computed_at"]),
        metadata=payload["metadata"],
        format_version=recorded_version,
        quantities=quantities,
    )
