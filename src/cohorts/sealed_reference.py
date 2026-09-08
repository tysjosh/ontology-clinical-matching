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
constructors), but it requires two explicit, non-defaulted arguments
(`issued_by`, `issued_at`) with no shortcut — there is no accidental,
argument-free way to produce one. `_mint()` is documented as the intended
call site so task 16 has an obvious extension point.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from causal.baselines import unadjusted_cox
from cohorts.cohort_construction import RandomizedCohort
from infra.reference_guard import assert_reference_unlocked


@dataclass(frozen=True)
class LockedEvaluationToken:
    """An unforgeable-by-accident credential required to unseal the internal
    randomized reference.

    Real minting logic arrives in task 16 (`scripts/check_go_no_go.py`),
    which will be the sole real caller of `_mint()` (or whatever minting
    mechanism it ultimately requires — e.g. requiring every checklist item
    to have already passed). Do not construct this directly outside of
    `_mint()`/tests; there is deliberately no default constructor call that
    produces a valid token by accident.

    Attributes:
        issued_by: identifies the minting process (e.g.
            `"scripts/check_go_no_go.py"`). Never empty.
        issued_at: when the token was minted (UTC).
    """

    issued_by: str
    issued_at: datetime

    def __post_init__(self) -> None:
        if not self.issued_by:
            raise ValueError("LockedEvaluationToken.issued_by must be non-empty.")

    @classmethod
    def _mint(cls, issued_by: str) -> LockedEvaluationToken:
        """Mint a new token. Intended sole real caller: task 16's
        `scripts/check_go_no_go.py`, once every go/no-go checklist item has
        passed. Any other caller (including tests) must go through this
        same method — there is no other path to a valid token.
        """
        return cls(issued_by=issued_by, issued_at=datetime.now(UTC))


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
    """

    value: Any
    cohort_size: int
    computed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def unseal(self, unlock_token: object) -> InternalReference:
        """Return the sealed `InternalReference`, if `unlock_token` is valid.

        Delegates the validity check to
        `infra.reference_guard.assert_reference_unlocked` rather than
        duplicating it here, so there is exactly one implementation of "is
        this a genuine token" (`design.md` Error Handling: "Attempt to read
        sealed reference before Stage 17 -> `ReferenceAccessError`, raised
        at the function-call boundary, not just logged").

        Args:
            unlock_token: must be a valid, minted `LockedEvaluationToken`
                (see `LockedEvaluationToken._mint`). Any other value
                (including `None` or a plain string) is rejected.

        Returns:
            The sealed `InternalReference`.

        Raises:
            ReferenceAccessError: if `unlock_token` is not a valid
                `LockedEvaluationToken`.
        """
        assert_reference_unlocked(unlock_token)
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

        Args:
            path: destination file path (e.g.
                `results/<experiment_id>/reference/internal_randomized_reference.sealed.json`).

        Returns:
            The `Path` written to.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "value": self._internal_reference.value,
            "cohort_size": self._internal_reference.cohort_size,
            "computed_at": self._internal_reference.computed_at.isoformat(),
            "metadata": self._internal_reference.metadata,
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        # Owner read/write only (0o600) -- defense-in-depth, not the real
        # gate; see docstring point 3.
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
        return target


def read_sealed_artifact(path: str | Path, unlock_token: object) -> InternalReference:
    """Read a sealed-reference artifact written by
    `SealedReference.write_sealed_artifact`, given a valid
    `LockedEvaluationToken`.

    Mirrors `scripts/run_paper.py`'s locked-evaluation-gating convention
    exactly: `assert_reference_unlocked(unlock_token)` is called FIRST,
    before `path` is opened at all. A caller without a valid token gets
    `ReferenceAccessError` and this function never touches the filesystem.

    Args:
        path: the sealed-reference artifact's path (see
            `SealedReference.write_sealed_artifact`).
        unlock_token: must be a valid, minted `LockedEvaluationToken`.

    Returns:
        The `InternalReference` reconstructed from the artifact.

    Raises:
        ReferenceAccessError: if `unlock_token` is not a valid
            `LockedEvaluationToken` -- raised before `path` is opened.
        FileNotFoundError: if `path` does not exist (only checked after the
            token has already been validated).
    """
    assert_reference_unlocked(unlock_token)

    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return InternalReference(
        value=payload["value"],
        cohort_size=payload["cohort_size"],
        computed_at=datetime.fromisoformat(payload["computed_at"]),
        metadata=payload["metadata"],
    )
