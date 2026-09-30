#!/usr/bin/env python
"""CI / pre-commit check: fail when a staged or tracked file contains a secret.

Requirement 2.4 requires secrets and credentials to be loaded through an
approved secret mechanism (environment variables, per `docs/secrets.md`) and
excluded from version control. This script is the enforcement check: it runs
the same shared detection heuristics used by `src/infra/manifest.py`'s
`PrivacyScanner` and `scripts/scan_exports.py`
(`src/infra/privacy_heuristics.py`) against either the currently staged diff
(`git diff --cached`, the default — appropriate for a pre-commit hook) or
every tracked file in the repository (`--all-tracked` — appropriate for a
periodic CI job), and exits non-zero if a secret-shaped string is found.

Only the `category == "secret"` subset of violations blocks this check
(direct/quasi-identifier and unauthorized-path findings are out of scope for
a *secrets* check and are handled by `scripts/scan_exports.py` instead at
export time).

Usage:

    python scripts/check_no_secrets.py                # staged changes only
    python scripts/check_no_secrets.py --all-tracked   # every tracked file
    python scripts/check_no_secrets.py --repo-root DIR # check a different tree

Wire into pre-commit via `.pre-commit-config.yaml` (see that file in the
repository root) or into CI as a standalone step; exit code 0 means clean,
1 means a secret-shaped string was found, or the tree could not be scanned
at all.

Pre-flight (Requirement 15.1)
-----------------------------
Every scan mode here needs a working git repository: the staged mode reads
the index, the `--all-tracked` mode reads `git ls-files`. Before task 2 the
project was not a repository, and this script's very first git call
(`git diff --cached --name-only --diff-filter=ACMR`) failed with git's own
``unknown option `cached` `` line followed by a multi-line usage dump, which
the script surfaced verbatim. That output says nothing about what the
*operator* should do about it.

`preflight_repository()` now runs first and turns every "cannot scan"
condition — git missing from PATH, the directory not existing, the path not
being inside a git work tree, or being inside some *other* repository rather
than the one asked about — into one clear sentence naming the condition and
the path. The exit code stays non-zero in all of those cases: a checker that
cannot inspect what is about to be committed must fail closed, because
"scanned nothing successfully" and "found no secrets" are not the same
result and must never report the same way.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from infra.privacy_heuristics import Violation, scan_string_leaf

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Seconds any single `git` invocation may take before it is treated as a
#: failure. A pre-commit hook must not hang on a wedged git process.
_GIT_TIMEOUT_SECONDS = 10


class PreflightError(RuntimeError):
    """The tree cannot be scanned at all: raised by `preflight_repository()`.

    Carries a single self-contained operator-facing sentence. Distinct from
    `GitCommandError` so that `main()` can label a pre-flight failure
    differently from a git command that failed *after* the repository was
    confirmed present.
    """


class GitCommandError(RuntimeError):
    """A `git` invocation failed after the pre-flight check passed."""


def _run_git(args: list[str], repo_root: Path) -> str:
    """Run `git <args>` in `repo_root`, returning stdout.

    Raises `GitCommandError` with a one-line, first-line-only summary of
    git's stderr. Truncating to the first line is deliberate: it keeps a
    multi-line usage dump out of the operator's terminal, which is the
    failure mode this script previously had.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitCommandError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitCommandError(
            f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s"
        ) from exc
    if result.returncode != 0:
        detail = _first_line(result.stderr) or _first_line(result.stdout)
        message = f"git {' '.join(args)} failed"
        raise GitCommandError(f"{message}: {detail}" if detail else message)
    return result.stdout


def _first_line(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def preflight_repository(repo_root: Path) -> None:
    """Confirm `repo_root` is a git work tree this script can scan.

    Raises `PreflightError` with one clear sentence when it is not. Checked
    conditions, in order:

    1. `repo_root` exists and is a directory.
    2. `git` is runnable at all.
    3. `repo_root` is inside a git work tree.
    4. That work tree's top level *is* `repo_root`, not an unrelated
       ancestor repository. Without this check a directory nested inside
       some other checkout would report "is a repository" and then scan that
       outer repository's index, silently reporting a clean result for a
       tree nobody asked about.
    """
    resolved = repo_root.resolve()
    if not resolved.exists():
        raise PreflightError(f"path does not exist: {resolved}")
    if not resolved.is_dir():
        raise PreflightError(f"path is not a directory: {resolved}")

    try:
        toplevel = _run_git(["rev-parse", "--show-toplevel"], resolved).strip()
    except GitCommandError as exc:
        if "not found on PATH" in str(exc):
            raise PreflightError(
                "git executable not found on PATH, so staged content cannot be "
                "inspected"
            ) from exc
        raise PreflightError(f"not a git repository: {resolved}") from exc

    if not toplevel:
        raise PreflightError(f"not a git repository: {resolved}")

    if Path(toplevel).resolve() != resolved:
        raise PreflightError(
            f"not a git repository: {resolved} (it sits inside the repository "
            f"rooted at {Path(toplevel).resolve()}, which is not the tree "
            "requested)"
        )


def _staged_files(repo_root: Path) -> list[Path]:
    """Filenames staged for commit (added/copied/modified/renamed) only."""
    output = _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"], repo_root)
    return [repo_root / line for line in output.splitlines() if line.strip()]


def _staged_content(path: Path, repo_root: Path) -> str | None:
    """The staged (index) version of `path`'s content, not the working-tree version.

    Scanning the index rather than the working tree matters because a
    pre-commit hook must judge exactly what is about to be committed, not
    whatever happens to currently be on disk (which could differ from the
    staged version).
    """
    relative = path.relative_to(repo_root)
    result = subprocess.run(
        ["git", "show", f":{relative.as_posix()}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _all_tracked_files(repo_root: Path) -> list[Path]:
    output = _run_git(["ls-files"], repo_root)
    return [repo_root / line for line in output.splitlines() if line.strip()]


def _scan_file(path: Path, content: str, repo_root: Path) -> list[Violation]:
    relative = path.relative_to(repo_root)
    return [v for v in scan_string_leaf(content, str(relative)) if v.category == "secret"]


def check_staged(repo_root: Path = REPO_ROOT) -> list[Violation]:
    violations: list[Violation] = []
    for path in _staged_files(repo_root):
        content = _staged_content(path, repo_root)
        if content is None:
            continue
        violations.extend(_scan_file(path, content, repo_root))
    return violations


def check_all_tracked(repo_root: Path = REPO_ROOT) -> list[Violation]:
    violations: list[Violation] = []
    for path in _all_tracked_files(repo_root):
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        violations.extend(_scan_file(path, content, repo_root))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if any staged (or, with --all-tracked, any tracked) file "
            "contains a secret-shaped string (Requirement 2.4)."
        )
    )
    parser.add_argument(
        "--all-tracked",
        action="store_true",
        help="Scan every git-tracked file instead of only the staged diff.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Repository work tree to scan. Defaults to this project's root; "
            "override to check another checkout."
        ),
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()

    try:
        preflight_repository(repo_root)
    except PreflightError as exc:
        # Fail closed: an unscannable tree is not a clean tree.
        print(f"check_no_secrets: cannot scan for secrets: {exc}", file=sys.stderr)
        return 1

    try:
        violations = (
            check_all_tracked(repo_root) if args.all_tracked else check_staged(repo_root)
        )
    except GitCommandError as exc:
        print(f"check_no_secrets failed: {exc}", file=sys.stderr)
        return 1

    if not violations:
        print("check_no_secrets: no secrets detected")
        return 0

    print(f"check_no_secrets: found {len(violations)} potential secret(s):", file=sys.stderr)
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)
    print(
        "Remove the secret and load it via an environment variable instead "
        "(see docs/secrets.md).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
