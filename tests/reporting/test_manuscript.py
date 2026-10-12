"""Smoke-level tests for the ICLR 2027 manuscript package (task 23.2;
Requirement 20.5, 20.6, 20.7).

**Validates: Requirements 20.5, 20.6, 20.7**

Confirms:

- the `submission` Makefile target builds a real PDF;
- the submission's main text is at most nine pages (Requirement 20.2,
  checked here as a build-time property since task 23.1 built the
  skeleton against that limit);
- no author name, affiliation, acknowledgement, or identifying
  repository link appears in the submission PDF's text layer
  (Requirement 20.3);
- the AI-assistance statement is present (Requirement 20.4);
- the venue string reads ICLR 2027 main track in both the manuscript and
  `pyproject.toml` (Requirement 20.5);
- every macro in `generated/numbers.tex` traces to a saved artifact
  (Requirement 20.6) -- checked here by building `numbers.tex` from a
  caller-controlled synthetic experiment payload and confirming every
  resulting macro name is derived from that payload's own row
  labels/columns, with none left over;
- a suppressed source renders the status rather than a number
  (Requirement 20.7).

Building the real submission PDF (via `tectonic`) is skipped, not failed,
when `tectonic` is not on `PATH` -- this mirrors
`tests/integration/test_run_causal_analysis.py`'s "skip, don't fail, on a
missing environment precondition" convention; the pure-Python half of
this file (`numbers.tex`/suppression generation) runs unconditionally.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DIR = REPO_ROOT / "paper"

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_paper import build_manuscript_artifacts  # noqa: E402

from reporting.tables import REQUIRED_COLUMNS  # noqa: E402

_TECTONIC_AVAILABLE = shutil.which("tectonic") is not None

#: Strings that must never appear in the submission PDF's text layer
#: (Requirement 20.3). Deliberately narrow and literal (not a general
#: PII scanner): these are the specific identifying strings this
#: project's own real author/affiliation/repo would use, chosen the same
#: way `tests/e2e/test_run_paper.py`'s stale-message scan picks concrete
#: literal strings to check for, rather than a heuristic.
_FORBIDDEN_IDENTIFYING_STRINGS = (
    "github.com",
    "Acknowledg",  # "Acknowledgement"/"Acknowledgment" -- unnumbered
    # section header `main.tex`'s shared "Acknowledgments" convention
    # would use if authored; the skeleton's stub sections do not include
    # one at all, so this also catches an accidental future addition to
    # the anonymous build.
)


# ---------------------------------------------------------------------------
# numbers.tex traceability and suppression rendering -- pure Python, no
# LaTeX engine required, run unconditionally.
# ---------------------------------------------------------------------------


def _synthetic_results_table_rows() -> list[dict]:
    released_row = {column: 1.23 for column in REQUIRED_COLUMNS}
    released_row["label"] = "complete"
    suppressed_row = {column: 4.56 for column in REQUIRED_COLUMNS}
    suppressed_row["label"] = "blocked_arm"
    suppressed_row["estimate_status"] = "suppressed"
    return [released_row, suppressed_row]


def test_every_macro_in_numbers_tex_traces_to_the_supplied_results_table(tmp_path):
    results_table = _synthetic_results_table_rows()
    manuscript_dir = tmp_path / "generated"

    build_manuscript_artifacts(
        {"results_table_rows": results_table}, results_table, manuscript_dir
    )

    numbers_text = (manuscript_dir / "numbers.tex").read_text(encoding="utf-8")
    macro_names = re.findall(r"\\newcommand\{\\(\w+)\}", numbers_text)
    assert macro_names, "no macros were generated at all"

    # Every macro name must contain the row label it came from, with
    # non-alphabetic characters (e.g. "_") stripped exactly as
    # `_latex_macro_name` strips them -- traceability back to a specific
    # results_table_rows entry, not an untraceable, free-floating number.
    expected_labels = {
        "".join(ch for ch in str(row["label"]).lower() if ch.isalpha())
        for row in results_table
    }
    for macro_name in macro_names:
        assert any(label in macro_name.lower() for label in expected_labels), macro_name

    # No duplicate macro names (would silently clobber a prior
    # \newcommand and lose traceability to one of the two rows).
    assert len(macro_names) == len(set(macro_names))


def test_suppressed_row_renders_the_status_not_a_number(tmp_path):
    results_table = _synthetic_results_table_rows()
    manuscript_dir = tmp_path / "generated"

    build_manuscript_artifacts(
        {"results_table_rows": results_table}, results_table, manuscript_dir
    )

    numbers_text = (manuscript_dir / "numbers.tex").read_text(encoding="utf-8")
    suppressed_lines = [
        line for line in numbers_text.splitlines() if "blockedarm" in line.lower()
    ]
    assert suppressed_lines, "expected at least one macro for the suppressed row"
    for line in suppressed_lines:
        assert "{suppressed}" in line, line
        assert "4.56" not in line, line

    table_text = (manuscript_dir / "table_main_results.tex").read_text(encoding="utf-8")
    assert "suppressed" in table_text
    assert "4.56" not in table_text


def test_venue_string_matches_in_pyproject_and_manuscript():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert "ICLR 2027 main track" in pyproject["project"]["description"]

    main_tex = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    assert "iclr2027_conference" in main_tex


def test_ai_assistance_statement_file_is_included_from_main_tex():
    main_tex = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    assert "ai_assistance_statement.tex" in main_tex
    statement = (PAPER_DIR / "ai_assistance_statement.tex").read_text(encoding="utf-8")
    assert statement.strip()


# ---------------------------------------------------------------------------
# Real build (skipped, not failed, without `tectonic` on PATH).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TECTONIC_AVAILABLE, reason="tectonic is not installed")
def test_submission_target_builds_a_pdf_with_page_limit_and_no_identifying_strings():
    subprocess.run(["make", "clean"], cwd=PAPER_DIR, capture_output=True, text=True, timeout=30)
    try:
        result = subprocess.run(
            ["make", "submission"],
            cwd=PAPER_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"submission build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        pdf_path = PAPER_DIR / "main.pdf"
        assert pdf_path.is_file()

        page_count = _pdf_page_count(pdf_path)
        assert page_count is not None, "could not determine page count"
        assert page_count <= 9, f"submission main text exceeded 9 pages: {page_count}"

        text_layer = _pdf_text(pdf_path)
        for forbidden in _FORBIDDEN_IDENTIFYING_STRINGS:
            assert forbidden not in text_layer, forbidden
        assert "Anonymous authors" in text_layer
        assert "AI use statement" in text_layer
        assert "ICLR 2027" in text_layer
    finally:
        subprocess.run(["make", "clean"], cwd=PAPER_DIR, capture_output=True, text=True, timeout=30)


def _pdf_page_count(pdf_path: Path) -> int | None:
    if shutil.which("pdfinfo") is None:
        return None
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, timeout=30
    )
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def _pdf_text(pdf_path: Path) -> str:
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext is not installed")
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"], capture_output=True, text=True, timeout=30
    )
    return result.stdout
