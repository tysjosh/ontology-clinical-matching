"""Unit tests for `scripts/check_no_secrets.py`.

Two behaviours are covered, both required by task 2.4 of the
`neural-experiment-pipeline` spec (Requirement 15.1):

1. **Pre-flight.** Run against a directory that is not a git repository, the
   checker must say so in one clear sentence and still exit non-zero. Before
   the repair it invoked `git diff --cached ...` blind, and git answered with
   ``unknown option `cached` `` plus a multi-line usage dump — output that
   tells the operator nothing about the actual problem. Failing *closed*
   matters as much as the wording: a checker that could not read the index has
   not established that the index is clean, so it must never exit 0.

2. **Detection.** A file staged in a real (temporary) repository containing a
   credential-shaped string must be flagged.

`scripts/` is not an installed package, so this test appends it to `sys.path`
the same way `tests/unit/test_scan_exports.py` does.

**Validates: Requirement 15.1**
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_no_secrets  # noqa: E402  (import after sys.path manipulation, by design)

#: A fake AWS-access-key-shaped secret — not a real credential.
#:
#: Assembled from two fragments rather than written as one literal on purpose:
#: this test file is itself tracked content that `check_no_secrets.py` scans
#: when it runs as a pre-commit hook, and a whole-key literal here would make
#: the checker flag the very test that exercises it. Neither fragment matches
#: any heuristic in `infra.privacy_heuristics` alone (`AKIA` is far too short,
#: and the 16-character tail is below the 20-character high-entropy floor and
#: carries no `sk_`/`api_`-style prefix), while the concatenation matches
#: `AWS_ACCESS_KEY_REGEX` exactly.
FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def staging_repo(tmp_path: Path) -> Path:
    """An initialized, commit-free git repository. Staging needs no identity."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "--quiet"], repo)
    return repo


# ---------------------------------------------------------------------------
# 1. Pre-flight: an unscannable tree reports clearly and fails closed.
# ---------------------------------------------------------------------------


class TestPreflightOutsideARepository:
    """**Validates: Requirement 15.1**"""

    def test_main_exits_non_zero_outside_a_repository(self, tmp_path, capsys):
        not_a_repo = tmp_path / "plain_directory"
        not_a_repo.mkdir()

        exit_code = check_no_secrets.main(["--repo-root", str(not_a_repo)])

        assert exit_code != 0
        captured = capsys.readouterr()
        assert "not a git repository" in captured.err
        assert str(not_a_repo.resolve()) in captured.err
        # The clean-result line must not appear: "could not scan" is not "clean".
        assert "no secrets detected" not in captured.out

    def test_message_does_not_surface_a_raw_git_usage_dump(self, tmp_path, capsys):
        not_a_repo = tmp_path / "plain_directory"
        not_a_repo.mkdir()

        check_no_secrets.main(["--repo-root", str(not_a_repo)])

        stderr = capsys.readouterr().err
        assert "usage: git" not in stderr
        assert "unknown option" not in stderr
        # One diagnostic line, not a wall of git output.
        assert len([line for line in stderr.splitlines() if line.strip()]) == 1

    def test_all_tracked_mode_also_fails_closed_outside_a_repository(self, tmp_path, capsys):
        not_a_repo = tmp_path / "plain_directory"
        not_a_repo.mkdir()

        exit_code = check_no_secrets.main(["--all-tracked", "--repo-root", str(not_a_repo)])

        assert exit_code != 0
        assert "not a git repository" in capsys.readouterr().err

    def test_missing_path_is_reported_as_such(self, tmp_path, capsys):
        missing = tmp_path / "does_not_exist"

        exit_code = check_no_secrets.main(["--repo-root", str(missing)])

        assert exit_code != 0
        assert "does not exist" in capsys.readouterr().err

    def test_preflight_raises_a_single_sentence_for_a_non_repository(self, tmp_path):
        not_a_repo = tmp_path / "plain_directory"
        not_a_repo.mkdir()

        with pytest.raises(check_no_secrets.PreflightError) as excinfo:
            check_no_secrets.preflight_repository(not_a_repo)

        assert "not a git repository" in str(excinfo.value)

    def test_preflight_accepts_a_real_repository(self, staging_repo):
        # No exception: the happy path the four cases above are contrasted with.
        check_no_secrets.preflight_repository(staging_repo)

    def test_nested_subdirectory_is_not_accepted_as_its_own_repository(self, staging_repo):
        """A directory *inside* a repository is not the tree that was asked for.

        Accepting it would scan the enclosing repository's index and report a
        result for a tree the caller never named.
        """
        nested = staging_repo / "subdir"
        nested.mkdir()

        with pytest.raises(check_no_secrets.PreflightError, match="not a git repository"):
            check_no_secrets.preflight_repository(nested)


# ---------------------------------------------------------------------------
# 2. Detection: a staged credential-shaped string is flagged.
# ---------------------------------------------------------------------------


class TestStagedSecretDetection:
    """**Validates: Requirement 15.1**"""

    def test_staged_file_with_credential_shaped_string_is_flagged(self, staging_repo, capsys):
        (staging_repo / "settings.yaml").write_text(f"aws_access_key_id: {FAKE_AWS_KEY}\n")
        _git(["add", "settings.yaml"], staging_repo)

        exit_code = check_no_secrets.main(["--repo-root", str(staging_repo)])

        assert exit_code != 0
        stderr = capsys.readouterr().err
        assert "potential secret(s)" in stderr
        assert "settings.yaml" in stderr
        assert "docs/secrets.md" in stderr

    def test_flagged_output_does_not_echo_the_secret_itself(self, staging_repo, capsys):
        (staging_repo / "settings.yaml").write_text(f"aws_access_key_id: {FAKE_AWS_KEY}\n")
        _git(["add", "settings.yaml"], staging_repo)

        check_no_secrets.main(["--repo-root", str(staging_repo)])

        captured = capsys.readouterr()
        assert FAKE_AWS_KEY not in captured.err
        assert FAKE_AWS_KEY not in captured.out

    def test_check_staged_returns_only_secret_category_violations(self, staging_repo):
        (staging_repo / "settings.yaml").write_text(f"aws_access_key_id: {FAKE_AWS_KEY}\n")
        _git(["add", "settings.yaml"], staging_repo)

        violations = check_no_secrets.check_staged(staging_repo)

        assert violations
        assert {v.category for v in violations} == {"secret"}

    def test_clean_staged_file_passes(self, staging_repo, capsys):
        (staging_repo / "notes.md").write_text("cd4 counts at weeks 0, 20, and 96\n")
        _git(["add", "notes.md"], staging_repo)

        exit_code = check_no_secrets.main(["--repo-root", str(staging_repo)])

        assert exit_code == 0
        assert "no secrets detected" in capsys.readouterr().out

    def test_unstaged_secret_in_the_working_tree_is_not_scanned(self, staging_repo, capsys):
        """The index is the subject, not the working tree.

        A pre-commit hook must judge exactly what is about to be committed.
        """
        (staging_repo / "scratch.yaml").write_text(f"api_key: {FAKE_AWS_KEY}\n")
        # Deliberately not staged.

        exit_code = check_no_secrets.main(["--repo-root", str(staging_repo)])

        assert exit_code == 0
        assert "no secrets detected" in capsys.readouterr().out

    def test_all_tracked_mode_finds_a_committed_secret(self, staging_repo, capsys):
        (staging_repo / "settings.yaml").write_text(f"aws_access_key_id: {FAKE_AWS_KEY}\n")
        _git(["add", "settings.yaml"], staging_repo)

        exit_code = check_no_secrets.main(["--all-tracked", "--repo-root", str(staging_repo)])

        assert exit_code != 0
        assert "settings.yaml" in capsys.readouterr().err
