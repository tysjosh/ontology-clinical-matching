"""Token and guard refusal (task 4; `design.md` "Property 9: Test-split
isolation under the strengthened token").

Property 9, verbatim: "*For any* object that is not an instance of
`LockedEvaluationToken`, including objects that expose identical attribute
names and values, both `assert_reference_unlocked` and the new
`assert_test_split_unlocked` raise `ReferenceAccessError`, and no
development-stage code path reaches Test_Split outcomes or the sealed
reference without such an instance. Extending the token from two to nine
required fields strengthens this guarantee, because no partially-bound token
can be constructed."

The look-alike case is the load-bearing one. Any guard that probed for
attributes (`hasattr(token, "issued_by")`) instead of checking the type
would pass every other case in this module and fail only that one, so the
look-alike is what proves the `isinstance` check was not softened into duck
typing. The look-alike's fields are read off `LockedEvaluationToken` with
`dataclasses.fields` rather than hard-coded, so when task 7 extends the
token from two fields to nine the look-alike grows with it and keeps
testing the same thing.

Property 26 (the multi-field `TokenFieldError`) also lands in this file per
`design.md`'s test-mapping section, but it belongs to task 7, which is what
introduces the seven additional bound fields. Only Property 9 is
implemented here.

The last test in this module is the structural half of Property 9's "no
development-stage code path reaches Test_Split outcomes": an AST scan
asserting `require_test_split` is the only place in `src/` that reads a
`.test` attribute off a pipeline result. That is a claim about the whole
source tree, so it is checked against the source tree rather than argued in
a docstring.

**Validates: Requirements 4.8, 18.6**
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
from dataclasses import fields, make_dataclass
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cohorts.pipeline import ObservationalizationPipelineResult
from cohorts.sealed_reference import LockedEvaluationToken
from infra.reference_guard import (
    ReferenceAccessError,
    assert_reference_unlocked,
    assert_test_split_unlocked,
    require_test_split,
)
from tests.token_fixtures import synthetic_token

_MAX_EXAMPLES = 50

#: Both guards, exercised identically. Property 9 quantifies over both, and
#: parameterizing over the pair is what keeps them from drifting apart: a
#: change that softened one guard's check would fail here even if the other
#: stayed strict.
GUARDS = (assert_reference_unlocked, assert_test_split_unlocked)
GUARD_IDS = ("assert_reference_unlocked", "assert_test_split_unlocked")


def _valid_looking_value(field: dataclasses.Field) -> object:
    """Produce a plausible value for one `LockedEvaluationToken` field.

    "Valid-looking" is the point: the look-alike must not be rejectable on
    the grounds that its values are obviously junk. Values are chosen by
    the field's declared annotation so that task 7's seven additional
    fields (hashes, git commit, evaluation id, checkpoint-hash tuple) get
    plausible content without this helper needing to know their names.
    """
    annotation = str(field.type)
    if "datetime" in annotation:
        return datetime.now(UTC)
    if "tuple" in annotation:
        return (("torch_sequence_gru", "a" * 64),)
    if "int" in annotation:
        return 1
    return f"valid-looking-{field.name}"


def _look_alike_fields() -> tuple[tuple[str, type, object], ...]:
    return tuple(
        (field.name, object, _valid_looking_value(field))
        for field in fields(LockedEvaluationToken)
    )


def _make_look_alike(class_name: str) -> object:
    """Build a frozen dataclass instance carrying every token field name
    with a valid-looking value, sharing nothing with the real token but its
    shape.
    """
    field_specs = _look_alike_fields()
    look_alike_cls = make_dataclass(
        class_name,
        [(name, annotation) for name, annotation, _ in field_specs],
        frozen=True,
    )
    return look_alike_cls(**{name: value for name, _, value in field_specs})


def look_alike_tokens() -> tuple[object, ...]:
    """Every flavour of near-miss worth distinguishing from the real token.

    The second entry shares the real token's *class name* as well as its
    field names, which is the strongest accidental-forgery a non-malicious
    caller could plausibly produce (e.g. a duplicated definition imported
    from the wrong module after a refactor).
    """
    return (
        _make_look_alike("TokenLookAlike"),
        _make_look_alike("LockedEvaluationToken"),
    )


def non_token_objects() -> st.SearchStrategy[object]:
    """Arbitrary objects that are not `LockedEvaluationToken` instances.

    Includes the wrong-typed values a caller might pass by accident
    (`None`, a bare string, a number), structured stand-ins (mappings and
    sequences carrying token-shaped content), and the look-alike
    dataclasses.
    """
    token_field_names = tuple(field.name for field in fields(LockedEvaluationToken))
    token_shaped_mappings = st.fixed_dictionaries(
        {name: st.text() for name in token_field_names}
    )
    return st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.binary(),
        st.datetimes(),
        st.lists(st.text(), max_size=3),
        st.tuples(st.text(), st.datetimes()),
        token_shaped_mappings,
        st.sampled_from(look_alike_tokens()),
    )


class _ExplodingResult:
    """A stand-in pipeline result that fails loudly if it is read at all.

    Used to prove `require_test_split` refuses *before* touching the result
    object. `__getattribute__` (not `__getattr__`) so that even reading an
    attribute that would otherwise exist is caught.
    """

    def __getattribute__(self, name: str) -> object:
        raise AssertionError(
            f"require_test_split read .{name} before validating the unlock token"
        )


def _attribute_reads_of_test() -> dict[pathlib.Path, list[ast.Attribute]]:
    """Every `<expr>.test` attribute read in `src/`, found by AST.

    AST rather than text search on purpose: `.test` appears in several
    module docstrings (`cohorts/preprocessing.py` explains why
    `SplitCohorts.test` cannot be rejected by type alone), and a grep-based
    assertion would either flag those or need fragile exclusions. Only real
    attribute-access nodes count.
    """
    src_root = pathlib.Path(__file__).resolve().parents[2] / "src"
    found: dict[pathlib.Path, list[ast.Attribute]] = {}
    for module_path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "test"
            and isinstance(node.ctx, ast.Load)
        ]
        if reads:
            found[module_path.relative_to(src_root)] = reads
    return found


# --------------------------------------------------------------------------
# Property 9
# --------------------------------------------------------------------------


@pytest.mark.parametrize("guard", GUARDS, ids=GUARD_IDS)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(candidate=non_token_objects())
def test_property_9_both_guards_refuse_every_non_token(guard, candidate: object) -> None:
    """Property 9: every non-token object is refused by both guards."""
    assert not isinstance(candidate, LockedEvaluationToken)

    with pytest.raises(ReferenceAccessError):
        guard(candidate)


@pytest.mark.parametrize("guard", GUARDS, ids=GUARD_IDS)
@pytest.mark.parametrize(
    "look_alike", look_alike_tokens(), ids=("field_names_only", "field_names_and_class_name")
)
def test_property_9_look_alikes_are_refused_by_both_guards(guard, look_alike: object) -> None:
    """The look-alike case, stated explicitly rather than left to the
    generator: an object carrying every token field name with valid-looking
    values is still refused, because the guards check the type.
    """
    for field in fields(LockedEvaluationToken):
        assert hasattr(look_alike, field.name)
        assert getattr(look_alike, field.name)

    with pytest.raises(ReferenceAccessError):
        guard(look_alike)


@pytest.mark.parametrize("guard", GUARDS, ids=GUARD_IDS)
def test_property_9_a_genuine_token_is_accepted_by_both_guards(guard) -> None:
    """The other side of Property 9: a real minted token passes both
    guards, so the refusals above are discrimination rather than blanket
    rejection.
    """
    token = synthetic_token(issued_by="tests/unit/test_token_and_guards.py")

    assert guard(token) is None


@pytest.mark.parametrize("guard", GUARDS, ids=GUARD_IDS)
def test_guard_messages_name_the_offending_type(guard) -> None:
    """A refusal has to be diagnosable: the message names what was passed
    instead of a token.
    """
    with pytest.raises(ReferenceAccessError, match="str"):
        guard("not-a-token")


# --------------------------------------------------------------------------
# require_test_split
# --------------------------------------------------------------------------


def test_require_test_split_refuses_before_touching_the_result() -> None:
    """`require_test_split` validates the token before reading anything off
    the pipeline result, so a caller without a token learns nothing about
    the result object — not even whether it has a `.test` attribute.
    """
    with pytest.raises(ReferenceAccessError):
        require_test_split(_ExplodingResult(), unlock_token=None)


def test_the_exploding_result_would_detect_a_read() -> None:
    """Guards the test above from passing vacuously: with a valid token,
    `require_test_split` does read the result, and `_ExplodingResult`
    catches it.
    """
    token = synthetic_token(issued_by="tests/unit/test_token_and_guards.py")

    with pytest.raises(AssertionError, match="before validating the unlock token"):
        require_test_split(_ExplodingResult(), unlock_token=token)


def test_require_test_split_returns_the_test_split_with_a_valid_token() -> None:
    """The accessor's happy path, and a check that it targets a field that
    actually exists on the real result type rather than a misspelling that
    only a stub would tolerate.
    """
    assert "test" in {field.name for field in fields(ObservationalizationPipelineResult)}

    token = synthetic_token(issued_by="tests/unit/test_token_and_guards.py")
    sentinel = object()
    stub_result = type("StubResult", (), {"test": sentinel})()

    assert require_test_split(stub_result, unlock_token=token) is sentinel


def test_require_test_split_is_the_only_reader_of_dot_test_in_src() -> None:
    """The structural half of Property 9: no development-stage module in
    `src/` reads a `.test` attribute directly.

    Exactly two sites are permitted, and each is checked for what it reads
    rather than merely being allowlisted by filename:

    - `cohorts/pipeline.py` reads `split_cohorts.test` while *constructing*
      the result (it must observationalize the test split to build it); it
      never reads `.test` off a pipeline result.
    - `infra/reference_guard.py` reads `pipeline_result.test` inside
      `require_test_split`, after the guard has run.
    """
    reads = _attribute_reads_of_test()

    assert set(reads) == {
        pathlib.Path("cohorts/pipeline.py"),
        pathlib.Path("infra/reference_guard.py"),
    }, (
        "a new direct .test read appeared in src/; route it through "
        f"infra.reference_guard.require_test_split instead: {sorted(map(str, reads))}"
    )

    pipeline_reads = reads[pathlib.Path("cohorts/pipeline.py")]
    assert [ast.unparse(node) for node in pipeline_reads] == ["split_cohorts.test"]

    guard_reads = reads[pathlib.Path("infra/reference_guard.py")]
    assert [ast.unparse(node) for node in guard_reads] == ["pipeline_result.test"]


def test_no_src_module_imports_the_test_split_accessor_except_locked_evaluation() -> None:
    """`require_test_split` exists to be called by the locked-evaluation
    runner only. Today nothing in `src/` calls it, which is the expected
    state while the test split stays closed (Requirement 18.3): the
    accessor is in place ahead of the runner that will use it.
    """
    src_root = pathlib.Path(__file__).resolve().parents[2] / "src"
    callers = []
    for module_path in sorted(src_root.rglob("*.py")):
        if module_path.name == "reference_guard.py":
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "require_test_split"
            ):
                callers.append(str(module_path.relative_to(src_root)))

    assert callers == [], (
        "require_test_split is called from src/ before the locked-evaluation "
        f"stage exists: {callers}"
    )
