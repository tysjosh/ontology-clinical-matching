"""Unit tests for the repository's ignore rules (task 2; Requirement 15.1).

Requirement 15.1 reads: "THE Repository SHALL track this project's source
tree at the project root and SHALL exclude raw patient data and generated
artifacts from tracked content, per the existing ignore rules."

The three excluded families this repository must never track are:

- `data/raw/`  -- the real ACTG175 v3.2 patient records;
- `reference/` -- sealed internal randomized reference artifacts, whose
  access is process-gated by a `LockedEvaluationToken` and which must not
  become readable simply by cloning the repository;
- `results/`   -- every generated experiment output (`results/.gitkeep` is
  the one deliberate exception, so the directory survives a clone).

These assertions are made against `git check-ignore` rather than against a
hand-parsed `.gitignore`, so what is tested is the rule *as git applies
it* -- pattern-precedence bugs and later-line negations included.

Both the ignore-rule assertions and the tracked-content assertions are
made, because the two failure modes are different: a missing ignore rule
lets a future `git add` stage patient data, while an already-tracked path
stays tracked regardless of what the ignore rules say.

**Validates: Requirements 15.1**
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Representative paths under each excluded family. They need not exist on
# disk: `git check-ignore` matches patterns against path names, which is
# exactly the property under test (the rule must hold for artifacts not
# yet generated, too).
IGNORED_PATHS = [
    # Raw patient data.
    "data/raw/actg175_v3_2/demographics.csv",
    "data/raw/actg175_v3_2/outcomes.csv",
    "data/raw/actg175_v3_2/longitudinal_cd4_cd8.csv",
    "data/raw/source_packages/LongCART_ACTG175.rda",
    "data/raw/some_future_extract/anything.csv",
    # Sealed reference artifacts.
    "reference/internal_randomized_reference.sealed.json",
    "reference/internal_randomized_reference.v2.sealed.json",
    # Generated results.
    "results/abc123/manifest.json",
    "results/abc123/reference/internal_randomized_reference.sealed.json",
    "results/locked_evaluation/consumed.jsonl",
    "results/diagnostics/overlap_smd_raw_vs_weighted.json",
]

# The environment and tool caches the task also requires excluded.
IGNORED_CACHE_PATHS = [
    ".venv/bin/python",
    ".pytest_cache/CACHEDIR.TAG",
    ".ruff_cache/CACHEDIR.TAG",
    ".mypy_cache/anything",
    ".hypothesis/unicode_data/14.0.0/charmap.json.gz",
    "src/__pycache__/__init__.cpython-311.pyc",
]

# Source-tree paths that must remain trackable, so the ignore rules cannot
# be "fixed" by over-broadly ignoring the project.
TRACKED_PATHS = [
    "pyproject.toml",
    ".gitignore",
    "src/cohorts/sealed_reference.py",
    "src/infra/manifest.py",
    "scripts/run_causal_analysis.py",
    "tests/unit/test_repository_provenance.py",
    "configs/target_trials/main.yaml",
    ".kiro/specs/causal-witness-pwm/decisions.md",
    "results/.gitkeep",
]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _is_repository() -> bool:
    return _git("rev-parse", "--is-inside-work-tree").returncode == 0


requires_repository = pytest.mark.skipif(
    not _is_repository(),
    reason="project root is not a git repository; task 2's `git init` has not run",
)


@requires_repository
@pytest.mark.parametrize("path", IGNORED_PATHS)
def test_patient_data_and_generated_artifacts_are_ignored(path: str):
    """`data/raw`, `reference`, and `results` paths are excluded by the ignore rules."""
    result = _git("check-ignore", "-q", "--no-index", path)

    assert result.returncode == 0, (
        f"{path!r} is NOT excluded by .gitignore. Raw patient data, sealed "
        f"reference artifacts, and generated results must never be stageable."
    )


@requires_repository
@pytest.mark.parametrize("path", IGNORED_CACHE_PATHS)
def test_environment_and_caches_are_ignored(path: str):
    result = _git("check-ignore", "-q", "--no-index", path)

    assert result.returncode == 0, f"{path!r} is NOT excluded by .gitignore"


@requires_repository
@pytest.mark.parametrize("path", TRACKED_PATHS)
def test_source_tree_paths_remain_trackable(path: str):
    """The ignore rules exclude artifacts without swallowing the source tree."""
    result = _git("check-ignore", "-q", "--no-index", path)

    assert result.returncode == 1, (
        f"{path!r} is excluded by .gitignore but belongs to the tracked "
        f"source tree (Requirement 15.1)"
    )


@requires_repository
def test_no_patient_data_or_sealed_artifact_is_tracked():
    """The index itself is clean, not merely the ignore rules."""
    tracked = _git("ls-files").stdout.splitlines()

    offending = [
        p
        for p in tracked
        if p.startswith(("data/", "reference/"))
        or (p.startswith("results/") and p != "results/.gitkeep")
    ]

    assert offending == [], f"tracked paths that must not be tracked: {offending}"


@requires_repository
def test_repository_tracks_the_project_source_tree():
    """Requirement 15.1's positive half: the source tree is actually tracked."""
    tracked = set(_git("ls-files").stdout.splitlines())

    assert tracked, "repository has no tracked files; the initial commit is missing"
    for expected in ("pyproject.toml", "src/cohorts/sealed_reference.py"):
        assert expected in tracked, f"{expected} is not tracked"


@requires_repository
def test_nothing_excluded_is_stageable_in_the_current_working_tree():
    """`git status --porcelain` surfaces no excluded path as stageable content.

    Run with `--untracked-files=all` so a directory entry cannot hide an
    excluded file inside it.
    """
    status = _git("status", "--porcelain", "--untracked-files=all").stdout.splitlines()

    paths = [line[3:].strip().strip('"') for line in status]
    offending = [
        p
        for p in paths
        if p.startswith(("data/", "reference/"))
        or (p.startswith("results/") and p != "results/.gitkeep")
    ]

    assert offending == [], (
        f"excluded paths appear as stageable working-tree content: {offending}"
    )
