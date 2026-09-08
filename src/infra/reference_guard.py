"""Access guard for the sealed internal randomized reference (`design.md`
"Correctness Property 1: Sealed reference isolation"; `decisions.md`
Section 8).

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

Real minting arrives in task 16
---------------------------------

Today, `LockedEvaluationToken._mint()` (see `cohorts.sealed_reference`) is
a minimal, deliberately-not-trivially-constructible token type. Task 16
(`scripts/check_go_no_go.py`) is the sole intended real-world caller of
whatever minting mechanism ends up guarding token issuance once every
go/no-go checklist item passes. This module's job is unaffected by how
minting evolves: it only ever asks "is this a genuine
`LockedEvaluationToken` instance?" and delegates the *how it was minted*
question to `cohorts.sealed_reference`.
"""

from __future__ import annotations


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
