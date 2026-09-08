"""Unit tests for `src/infra/nondeterminism.py`.

**Validates: Requirements 1.6**
"""

import pytest

from infra.nondeterminism import (
    get_recorded_invocations,
    mark_nondeterministic,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    # The registry is process-global module state; reset before and after
    # every test so recorded invocations never leak across test cases.
    reset_registry()
    yield
    reset_registry()


def test_decorated_call_is_recorded():
    @mark_nondeterministic("test reason")
    def flaky_op(x, y):
        return x + y

    result = flaky_op(2, 3)

    assert result == 5
    invocations = get_recorded_invocations()
    assert len(invocations) == 1
    assert invocations[0].function.endswith("flaky_op")
    assert invocations[0].reason == "test reason"
    assert invocations[0].timestamp


def test_multiple_calls_are_each_recorded():
    @mark_nondeterministic("nondeterministic op")
    def op():
        return None

    op()
    op()
    op()

    assert len(get_recorded_invocations()) == 3


def test_undecorated_calls_are_not_recorded():
    def plain_op(x):
        return x

    plain_op(1)

    assert get_recorded_invocations() == ()


def test_reset_registry_clears_invocations():
    @mark_nondeterministic("reason")
    def op():
        return None

    op()
    assert len(get_recorded_invocations()) == 1

    reset_registry()

    assert get_recorded_invocations() == ()


def test_get_recorded_invocations_returns_immutable_snapshot():
    @mark_nondeterministic("reason")
    def op():
        return None

    op()
    snapshot = get_recorded_invocations()

    assert isinstance(snapshot, tuple)
