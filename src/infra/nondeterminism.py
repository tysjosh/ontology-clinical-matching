"""Nondeterminism registry.

`design.md` Section 3.1 requires every run to record operations known to be
nondeterministic (Requirement 1.6), so that the manifest (`src/infra/manifest.py`,
not yet implemented) can surface them alongside the rest of a run's provenance.

`mark_nondeterministic` is the single decorator used to tag such an operation
(for example, a GPU op that is only bitwise-reproducible when mixed precision
is disabled). Every call to a decorated function is recorded in a simple,
importable, module-level registry — not written to disk or logged directly —
so that `manifest.py` can later pull the recorded invocations via
`get_recorded_invocations()` and fold them into `manifest.json` at the end of
a stage.

The registry is process-global by design (a plain module-level list), which
keeps the decorator itself trivial and keeps `manifest.py`'s future dependency
on this module to two calls: read the invocations, then reset the registry at
a stage boundary. `reset_registry()` exists for exactly that purpose, and is
also what test isolation relies on between test cases.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True)
class NondeterminismInvocation:
    """A single recorded call to a `@mark_nondeterministic`-decorated function."""

    function: str
    reason: str
    timestamp: str


#: Process-global, append-only record of every call to a
#: `@mark_nondeterministic`-decorated function made since the last
#: `reset_registry()`. Not intended to be mutated directly by callers — use
#: `get_recorded_invocations()` to read and `reset_registry()` to clear.
_REGISTRY: list[NondeterminismInvocation] = []


def mark_nondeterministic(reason: str) -> Callable[[_F], _F]:
    """Decorator marking a function as a known source of nondeterminism.

    Every call to the decorated function is recorded in the module-level
    registry (function's qualified name, `reason`, and a UTC timestamp)
    before the wrapped function executes. The wrapped function's return
    value and signature are otherwise unaffected.

    Args:
        reason: a short, human-readable explanation of why this operation is
            nondeterministic (e.g. `"cuDNN convolution backward pass is only
            bitwise-deterministic with deterministic algorithms disabled"`).
            Recorded verbatim alongside every invocation, so the manifest can
            explain *why* a given call was flagged, not just *that* it was.

    Returns:
        A decorator that wraps the target function.
    """

    def decorator(func: _F) -> _F:
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _REGISTRY.append(
                NondeterminismInvocation(
                    function=qualname,
                    reason=reason,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
            return func(*args, **kwargs)

        wrapper.__name__ = getattr(func, "__name__", qualname)
        wrapper.__qualname__ = qualname
        wrapper.__doc__ = func.__doc__
        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def get_recorded_invocations() -> tuple[NondeterminismInvocation, ...]:
    """Return every invocation recorded since the last `reset_registry()`.

    Returns an immutable snapshot (a tuple), so callers — including
    `manifest.py` — cannot accidentally mutate the live registry through the
    returned value.
    """
    return tuple(_REGISTRY)


def reset_registry() -> None:
    """Clear all recorded invocations.

    Intended to be called at stage boundaries once a manifest has consumed
    the recorded invocations for that stage, and by tests to ensure
    isolation between test cases (the registry is process-global module
    state, so an unreset registry would otherwise leak recorded invocations
    from one test into the next).
    """
    _REGISTRY.clear()
