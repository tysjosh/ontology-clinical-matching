"""Unit tests for the Dependency_Manifest (task 2.1; Requirements 5.1, 5.2, 20.5).

Three claims are pinned down here, all read from `pyproject.toml` itself
rather than from the installed environment, because the manifest is what a
fresh install reproduces from:

1. **Requirement 5.1-5.2** -- torch and numpy are declared at pinned
   versions. `design.md` Section 2.4 satisfies "declare pinned versions"
   with an *optional* `neural` extra rather than a base dependency, so that
   the existing suite keeps running where torch is not installed. That makes
   the placement assertion as load-bearing as the pin assertion: torch
   appearing in `[project] dependencies` would be a regression even though
   the version would still be pinned.
2. **Requirement 20.5** -- the manifest names ICLR 2027 main track as the
   submission venue, in place of the parent spec's NeurIPS 2026 WMHS line.
3. The `neural` pytest marker is registered and `tests/neural/` is the
   declared collection root, so torch-requiring tests are addressable as a
   group (`-m neural` / `-m "not neural"`) without warnings.

The base-dependency assertions are written over the parsed requirement
*names*, not over a substring search of the file, so a comment mentioning
torch cannot make them pass or fail spuriously.

**Validates: Requirements 5.1, 5.2, 20.5**
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

PINNED_TORCH = "torch==2.4.1"
PINNED_NUMPY = "numpy==1.26.4"
VENUE = "ICLR 2027 main track"
SUPERSEDED_VENUE = "NeurIPS 2026"

BASE_DEPENDENCY_NAMES = {"pydantic", "pyyaml"}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _requirement_name(requirement: str) -> str:
    """Lowercased distribution name of a PEP 508 requirement string."""
    return re.split(r"[\[<>=!~;\s]", requirement, maxsplit=1)[0].strip().lower()


# --------------------------------------------------------------------------
# Requirement 5.1-5.2: pinned torch/numpy, declared as an optional extra
# --------------------------------------------------------------------------


def test_neural_extra_exists(pyproject):
    extras = pyproject["project"]["optional-dependencies"]

    assert "neural" in extras, (
        "pyproject.toml declares no `neural` extra; design.md Section 2.4 "
        "requires torch to be declared as an optional extra"
    )


def test_neural_extra_pins_the_torch_version(pyproject):
    """Requirement 5.2: the declared torch version is the environment's, exactly pinned."""
    neural = pyproject["project"]["optional-dependencies"]["neural"]

    assert PINNED_TORCH in neural, (
        f"expected {PINNED_TORCH!r} in the `neural` extra, found {neural!r}"
    )


def test_neural_extra_declares_numpy_at_a_pinned_version(pyproject):
    """Requirement 5.1: numpy is declared alongside torch, and pinned, not floating."""
    neural = pyproject["project"]["optional-dependencies"]["neural"]
    numpy_requirements = [r for r in neural if _requirement_name(r) == "numpy"]

    assert numpy_requirements, f"numpy is absent from the `neural` extra: {neural!r}"
    assert numpy_requirements == [PINNED_NUMPY], (
        f"numpy must be pinned with `==` to {PINNED_NUMPY.split('==')[1]}, "
        f"found {numpy_requirements!r}"
    )


@pytest.mark.parametrize("name", ["torch", "numpy"])
def test_tensor_stack_is_absent_from_the_base_dependencies(pyproject, name):
    """design.md Section 2.4: a base install stays torch-free.

    This is the assertion that keeps the existing suite runnable without
    torch. If it fails, `pip install .` starts dragging in a ~900 MB wheel
    and the torch-isolation guarantee is gone.
    """
    base_names = {_requirement_name(r) for r in pyproject["project"]["dependencies"]}

    assert name not in base_names, (
        f"{name} appears in [project] dependencies; it belongs to the "
        f"optional `neural` extra only (Requirements 5.1-5.2)"
    )


def test_base_dependencies_are_exactly_the_existing_pinned_pair(pyproject):
    """The base install is unchanged: pinned pydantic + PyYAML, nothing added."""
    base = pyproject["project"]["dependencies"]
    base_names = {_requirement_name(r) for r in base}

    assert base_names == BASE_DEPENDENCY_NAMES, (
        f"base dependencies changed; expected {sorted(BASE_DEPENDENCY_NAMES)}, "
        f"found {sorted(base_names)}"
    )
    for requirement in base:
        assert "==" in requirement, f"base dependency {requirement!r} is not pinned"


def test_dev_extra_includes_the_neural_extra(pyproject):
    """A development install gets the tensor stack; a base install does not."""
    extras = pyproject["project"]["optional-dependencies"]
    project_name = pyproject["project"]["name"]

    dev_self_references = [
        r
        for r in extras["dev"]
        if _requirement_name(r) == project_name.lower() and "neural" in r
    ]

    assert dev_self_references, (
        f"the `dev` extra does not include the `neural` extra; expected an "
        f"entry like {project_name}[neural], found {extras['dev']!r}"
    )


# --------------------------------------------------------------------------
# Requirement 20.5: submission venue
# --------------------------------------------------------------------------


def test_manifest_names_iclr_2027_main_track_as_the_venue(pyproject):
    description = pyproject["project"]["description"]

    assert VENUE in description, (
        f"project description must name {VENUE!r} as the submission venue "
        f"(Requirement 20.5), found {description!r}"
    )


def test_manifest_no_longer_names_the_parent_specs_venue(pyproject):
    """Requirement 20.5 replaces the parent's venue line rather than adding to it."""
    description = pyproject["project"]["description"]

    assert SUPERSEDED_VENUE not in description, (
        f"the superseded venue {SUPERSEDED_VENUE!r} is still in the project "
        f"description: {description!r}"
    )


# --------------------------------------------------------------------------
# Marker registration and collection root
# --------------------------------------------------------------------------


def test_neural_pytest_marker_is_registered(pyproject):
    markers = pyproject["tool"]["pytest"]["ini_options"].get("markers", [])
    marker_names = {m.split(":", 1)[0].strip() for m in markers}

    assert "neural" in marker_names, (
        f"the `neural` marker is not registered in pyproject.toml, so "
        f"`@pytest.mark.neural` raises an unknown-marker warning: {markers!r}"
    )


def test_tests_neural_is_collected_by_a_declared_testpath(pyproject):
    """`tests/neural` is reached by the configured testpaths.

    It is asserted as *covered by* a testpath rather than as a testpath
    entry of its own. Listing both `tests` and `tests/neural` looks
    harmless and is not: pytest de-duplicates overlapping arguments by
    keeping the narrower path, which drops `tests` and collapses the suite
    to the neural directory alone. This test pins the containment property
    that actually matters -- torch-requiring tests run by default -- without
    inviting that configuration.
    """
    testpaths = [Path(p) for p in pyproject["tool"]["pytest"]["ini_options"]["testpaths"]]
    neural_root = Path("tests/neural")

    covered = any(
        neural_root == p or neural_root.is_relative_to(p) for p in testpaths
    )

    assert covered, (
        f"tests/neural is not reachable from any configured testpath: {testpaths!r}"
    )


def test_testpaths_contain_no_overlapping_entries(pyproject):
    """Guards the collection-collapsing configuration described above."""
    testpaths = [Path(p) for p in pyproject["tool"]["pytest"]["ini_options"]["testpaths"]]

    overlaps = [
        (str(a), str(b))
        for a in testpaths
        for b in testpaths
        if a != b and b.is_relative_to(a)
    ]

    assert overlaps == [], (
        f"overlapping testpaths {overlaps} make pytest discard the broader "
        f"entry and collect only the narrower one"
    )


def test_tests_neural_collection_root_exists_on_disk():
    """The collection root is a real package, so torch tests have a home."""
    neural_dir = REPO_ROOT / "tests" / "neural"

    assert neural_dir.is_dir(), f"{neural_dir} does not exist"
    assert (neural_dir / "__init__.py").is_file(), (
        f"{neural_dir}/__init__.py is missing; the rest of tests/ is a "
        f"package, and mixing styles breaks module-name resolution"
    )
