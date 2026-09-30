"""Access guards for the sealed internal randomized reference and the test
split (`design.md` "Correctness Property 1: Sealed reference isolation" and
"Property 9: Test-split isolation under the strengthened token";
`decisions.md` Section 8).

`decisions.md` Section 8 requires the internal randomized reference to be
"withheld from model development" and opened only "at the locked
external-evaluation stage." `design.md` makes this a structural guarantee
rather than a policy: `cohorts.sealed_reference.SealedReference.unseal()`
must call `assert_reference_unlocked()` below before returning the
unsealed value, and this function raises `ReferenceAccessError` for
anything other than a genuine, minted
`cohorts.sealed_reference.LockedEvaluationToken`.

Per `design.md`'s Error Handling table: "Attempt to read sealed reference
before Stage 17 -> `ReferenceAccessError`, raised at the function-call
boundary, not just logged." This module is that function-call boundary.

Why the deferred import
------------------------

`cohorts.sealed_reference.SealedReference.unseal()` calls
`assert_reference_unlocked()` (this module), so `reference_guard` cannot
import `cohorts.sealed_reference` at module scope without creating an
import cycle (`sealed_reference` -> `reference_guard` -> `sealed_reference`).
`LockedEvaluationToken` is therefore imported lazily, inside the function
body, only at call time. This keeps the check a real `isinstance` check
against the actual token type (not a duck-typed attribute probe that a
caller could spoof by accident) while avoiding the cycle.

Minting, and why these checks did not change when the token grew
-----------------------------------------------------------------

`scripts/check_go_no_go.py` is the sole intended real-world caller of
`LockedEvaluationToken._mint()` (see `cohorts.sealed_reference`), which
mints only once every go/no-go checklist item passes and the readiness
ledger is complete. This module's job is unaffected by how minting evolves:
it only ever asks "is this a genuine `LockedEvaluationToken` instance?" and
delegates the *how it was minted* question to `cohorts.sealed_reference`.

That indifference is what let the token go from two fields to the nine
Requirement 13.3 binds (`neural-experiment-pipeline/design.md` Section 4.6)
without a single line changing here. Both checks below are still
`isinstance(unlock_token, LockedEvaluationToken)` against the same class,
and because that class now refuses to exist unless all nine bindings are
present and non-empty, the check is strictly *stronger* than it was: every
value that passes it carries nine validated bindings. Duck typing an
"extended token" protocol, or accepting a second token class alongside this
one, would have been the weakening this design explicitly rules out.

The second guard: the test split
---------------------------------

`neural-experiment-pipeline/design.md` Section 2.6 tension 7 adds a second,
independent guard for defense in depth. Requirement 18.3 ("WHILE any
development-stage seed, ablation arm, or sensitivity variant remains
unexecuted ... keep the Test_Split closed and the Sealed_Reference_Store
sealed") is a temporal condition that conventions cannot enforce, so the
test split gets the same structural treatment the sealed reference already
has: `assert_test_split_unlocked(unlock_token)` mirrors
`assert_reference_unlocked` exactly, using the same
`isinstance(token, LockedEvaluationToken)` check against the same token
type, and `require_test_split(pipeline_result, unlock_token)` is the only
accessor any runner uses to reach
`cohorts.pipeline.ObservationalizationPipelineResult.test`.

Two functions rather than one shared implementation is deliberate. They are
separate call boundaries with separate messages, so a traceback says which
protected resource was reached for, and a future change to one guard's
policy (e.g. an extra ledger check on the test split) does not silently
change the other's. They share the token type, which is the part that must
not diverge.

`ObservationalizationPipelineResult.test` is deliberately *not* removed or
renamed: `design.md` records that existing tests depend on the attribute,
and a rename would trade a working suite for a guarantee that
`require_test_split` already provides. The guarantee instead comes from
`require_test_split` being the only accessor in `src/` that touches the
attribute (asserted mechanically in
`tests/unit/test_token_and_guards.py`), and from development-stage runners
having no token to pass it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    # Import-time cycle avoidance again: `cohorts.pipeline` reaches
    # `infra.reference_guard` transitively through
    # `cohorts.sealed_reference`, so these names exist for annotations only
    # (resolved lazily by `from __future__ import annotations`).
    from cohorts.pipeline import ObservationalizationPipelineResult
    from cohorts.selection import ObservationalizedSplit


class ReferenceAccessError(Exception):
    """Raised when the sealed internal randomized reference is accessed
    without a valid, minted `LockedEvaluationToken`.

    This is an engineering/process failure, not a domain outcome (see
    `design.md`'s Error Handling table) — it signals a bug or a protocol
    violation (a stage running before the reference should be readable),
    so it halts execution rather than being logged and swallowed.
    """


def assert_reference_unlocked(unlock_token: object) -> None:
    """Raise `ReferenceAccessError` unless `unlock_token` is a valid token.

    Returns silently (no exception, no return value of interest) when
    `unlock_token` is a genuine `cohorts.sealed_reference.LockedEvaluationToken`
    instance. Raises `ReferenceAccessError` for anything else, including
    `None`, a plain string, or any other wrong-typed value — there is no
    code path from an accidental/forgeable value to a successful unlock.

    Args:
        unlock_token: the value a caller is attempting to unlock the sealed
            reference with. Typed as `object` (not
            `LockedEvaluationToken`) precisely because this function's job
            is to validate that type at runtime, not assume it.

    Raises:
        ReferenceAccessError: if `unlock_token` is not a valid
            `LockedEvaluationToken`.
    """
    # Deferred import: see module docstring ("Why the deferred import").
    from cohorts.sealed_reference import LockedEvaluationToken

    if not isinstance(unlock_token, LockedEvaluationToken):
        raise ReferenceAccessError(
            "Attempted to unseal the internal randomized reference without a "
            "valid LockedEvaluationToken (got "
            f"{type(unlock_token).__name__!r}). This token is only mintable "
            "by scripts/check_go_no_go.py (task 16) after every go/no-go "
            "checklist item passes (decisions.md Section 8; design.md "
            "'Correctness Property 1: Sealed reference isolation')."
        )


def assert_test_split_unlocked(unlock_token: object) -> None:
    """Raise `ReferenceAccessError` unless `unlock_token` is a valid token.

    The test-split mirror of `assert_reference_unlocked`: same token type,
    same `isinstance` check, same exception, different protected resource
    (`design.md` Section 2.6 tension 7; Requirements 4.8, 18.6). Returns
    silently for a genuine
    `cohorts.sealed_reference.LockedEvaluationToken` and raises for
    everything else, including an object that merely exposes the same
    attribute names with valid-looking values — the check is on type
    identity, never on duck-typed attribute presence ("Property 9:
    Test-split isolation under the strengthened token").

    Args:
        unlock_token: the value a caller is attempting to open the test
            split with. Typed as `object` because validating that type at
            runtime is this function's entire job.

    Raises:
        ReferenceAccessError: if `unlock_token` is not a valid
            `LockedEvaluationToken`.
    """
    # Deferred import: see module docstring ("Why the deferred import").
    from cohorts.sealed_reference import LockedEvaluationToken

    if not isinstance(unlock_token, LockedEvaluationToken):
        raise ReferenceAccessError(
            "Attempted to open the Test_Split without a valid "
            f"LockedEvaluationToken (got {type(unlock_token).__name__!r}). "
            "The test split stays closed while any development-stage seed, "
            "ablation arm, or sensitivity variant remains unexecuted; the "
            "token is only mintable by scripts/check_go_no_go.py once the "
            "readiness ledger is complete and every go/no-go checklist item "
            "passes (neural-experiment-pipeline requirements 4.8, 18.3, "
            "18.6; design.md 'Property 9: Test-split isolation under the "
            "strengthened token')."
        )


def require_test_split(
    pipeline_result: ObservationalizationPipelineResult,
    unlock_token: object,
) -> ObservationalizedSplit:
    """Return the observationalized test split, if `unlock_token` is valid.

    The single accessor every runner uses to reach
    `cohorts.pipeline.ObservationalizationPipelineResult.test`
    (Requirements 4.8, 18.6). Development-stage code calls nothing of the
    sort and therefore cannot reach test outcomes; the locked-evaluation
    runner calls this with the token `scripts/check_go_no_go.py` minted.

    The guard runs *before* `pipeline_result` is touched at all — no
    attribute is read, no length taken, nothing — mirroring
    `cohorts.sealed_reference.read_sealed_artifact`'s "refuse before
    opening the file" convention. A caller without a valid token learns
    nothing about the result object, not even whether it has a `.test`
    attribute.

    Args:
        pipeline_result: the frozen Section-9 pipeline output whose test
            split is being requested. Not inspected until after the token
            check passes.
        unlock_token: must be a valid, minted `LockedEvaluationToken`.

    Returns:
        The `cohorts.selection.ObservationalizedSplit` for the test split.

    Raises:
        ReferenceAccessError: if `unlock_token` is not a valid
            `LockedEvaluationToken` — raised before `pipeline_result` is
            read.
    """
    assert_test_split_unlocked(unlock_token)
    return pipeline_result.test
