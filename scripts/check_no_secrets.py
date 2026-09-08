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

Wire into pre-commit via `.pre-commit-config.yaml` (see that file in the
repository root) or into CI as a standalone step; exit code 0 means clean,
1 means a secret-shaped string was found (or the git command itself failed).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from infra.privacy_heuristics import Violation, scan_string_leaf

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _staged_files() -> list[Path]:
    """Filenames staged for commit (added/copied/modified/renamed), added or changed only."""
    output = _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    return [REPO_ROOT / line for line in output.splitlines() if line.strip()]


def _staged_content(path: Path) -> str | None:
    """The staged (index) version of `path`'s content, not the working-tree version.

    Scanning the index rather than the working tree matters because a
    pre-commit hook must judge exactly what is about to be committed, not
    whatever happens to currently be on disk (which could differ from the
    staged version).
    """
    relative = path.relative_to(REPO_ROOT)
    result = subprocess.run(
        ["git", "show", f":{relative.as_posix()}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _all_tracked_files() -> list[Path]:
    output = _run_git(["ls-files"])
    return [REPO_ROOT / line for line in output.splitlines() if line.strip()]


def _scan_file(path: Path, content: str) -> list[Violation]:
    relative = path.relative_to(REPO_ROOT)
    return [v for v in scan_string_leaf(content, str(relative)) if v.category == "secret"]


def check_staged() -> list[Violation]:
    violations: list[Violation] = []
    for path in _staged_files():
        content = _staged_content(path)
        if content is None:
            continue
        violations.extend(_scan_file(path, content))
    return violations


def check_all_tracked() -> list[Violation]:
    violations: list[Violation] = []
    for path in _all_tracked_files():
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        violations.extend(_scan_file(path, content))
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
    args = parser.parse_args(argv)

    try:
        violations = check_all_tracked() if args.all_tracked else check_staged()
    except RuntimeError as exc:
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
