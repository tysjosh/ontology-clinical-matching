"""Unit tests for `scripts/scan_exports.py`.

Injects a fake SSN-like direct identifier and a fake AWS-key-shaped secret
into files under a temporary export directory, then confirms
`scan_directory()` flags both (Requirement 2.7).

`scripts/` is not an installed package (it holds CLI entry points invoked
directly, e.g. `python scripts/scan_exports.py <dir>`), so this test adds it
to `sys.path` directly rather than relying on `pythonpath` in
`pyproject.toml` — mirroring how `tests/e2e/test_smoke.py` exercises
`scripts/run_stage.py` as a subprocess, except here we import the module
directly since we are unit-testing its `scan_directory()` function rather
than the full CLI process.

**Validates: Requirements 2.7**
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCAN_EXPORTS_SCRIPT = SCRIPTS_DIR / "scan_exports.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scan_exports  # noqa: E402  (import after sys.path manipulation, by design)

#: A fake SSN-like direct identifier — not a real patient identifier.
FAKE_SSN = "123-45-6789"

#: A fake AWS-access-key-shaped secret — not a real credential.
FAKE_AWS_KEY = "AKIAABCDEFGHIJKLMNOP"


def test_scan_directory_flags_injected_fake_ssn_and_aws_key(tmp_path):
    (tmp_path / "notes.txt").write_text(f"patient contact on file: {FAKE_SSN}\n")
    (tmp_path / "config.yaml").write_text(f"aws_key: {FAKE_AWS_KEY}\n")

    violations = scan_exports.scan_directory(tmp_path)

    assert any(
        v.category == "direct_identifier" and v.path == "notes.txt" for v in violations
    ), f"expected an SSN-like direct-identifier violation in notes.txt, got: {violations}"
    assert any(
        v.category == "secret" and "config.yaml" in v.path for v in violations
    ), f"expected an AWS-key-shaped secret violation in config.yaml, got: {violations}"


def test_scan_directory_returns_no_violations_for_clean_directory(tmp_path):
    (tmp_path / "results.json").write_text('{"cohort": "arm0", "concordance": 0.71}')

    violations = scan_exports.scan_directory(tmp_path)

    assert violations == []


def test_scan_directory_flags_field_name_inside_structured_json(tmp_path):
    (tmp_path / "manifest.json").write_text('{"patient": {"ssn": "' + FAKE_SSN + '"}}')

    violations = scan_exports.scan_directory(tmp_path)

    assert any(v.category == "direct_identifier" for v in violations)


def test_scan_directory_skips_unreadable_binary_files(tmp_path):
    (tmp_path / "checkpoint.bin").write_bytes(bytes(range(256)))

    # Must not raise on a binary file that fails UTF-8 decoding.
    violations = scan_exports.scan_directory(tmp_path)

    assert violations == []


def test_cli_exits_nonzero_and_reports_violations(tmp_path):
    (tmp_path / "secret.env").write_text(f"AWS_SECRET_ACCESS_KEY={FAKE_AWS_KEY}\n")

    result = subprocess.run(
        [sys.executable, str(SCAN_EXPORTS_SCRIPT), str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "secret" in result.stderr.lower()


def test_cli_exits_zero_for_clean_directory(tmp_path):
    (tmp_path / "results.json").write_text('{"cohort": "arm0"}')

    result = subprocess.run(
        [sys.executable, str(SCAN_EXPORTS_SCRIPT), str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "no violations" in result.stdout.lower()
