#!/usr/bin/env python
"""CLI: automated identifier/secret scan over an export directory.

Requirement 2.7 requires exported artifacts to pass an automated identifier
and secret scan before release. This script is that scan: given a directory
path, it walks every file under it, reads text-shaped files (`.json`,
`.yaml`/`.yml`, `.txt`, `.md`, `.csv`, `.log`, and any other file that
decodes as UTF-8 text), and runs the same detection heuristics used by
`src/infra/manifest.py`'s `PrivacyScanner` — both call into
`src/infra/privacy_heuristics.py`'s `scan_tree()`/`scan_string_leaf()`, so
export scanning and manifest scanning can never drift out of sync
(`design.md`'s "Existing context" note for this task).

`.json` and `.yaml`/`.yml` files are parsed and scanned structurally (so
field-name checks such as `ssn:` or `api_key:` fire on the parsed key, not
just on the raw text), falling back to a plain-text scan if parsing fails.
Every other readable text file is scanned as one string. Binary files (that
fail UTF-8 decoding) are skipped, since this scanner has no way to
meaningfully inspect their content and would otherwise raise on every
checkpoint/model-weights file in a typical export.

Filenames and relative paths themselves are also checked (Requirement 1.11 /
2.2's "filenames" clause): a filename matching a direct-identifier or
secret field-name pattern is flagged even if the file's content is clean.

Export paths this scanner is pointed at include
`results/<experiment_id>/report/` (`src/reporting/`) and
`results/<experiment_id>/audit/public/` (`src/witnesses/audit_io.py`, task
17.1) -- the per-claim audit artifacts a suppressed estimate's redacted
copy lives in. The sibling `results/<experiment_id>/audit/protected/`
directory retains the real numeric value for a suppressed estimate and is
never scanned for export, exactly like `reference/` (see `docs/data_access.md`
Section 2).

Usage:

    python scripts/scan_exports.py <directory>
    python scripts/scan_exports.py results/abc123deadbeef00/reporting/
    python scripts/scan_exports.py results/abc123deadbeef00/audit/public/

Exit code is 0 if no violations are found, 1 if any are found (or if the
given path is not a directory), so this script is safe to wire into CI as a
release-blocking check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from infra.privacy_heuristics import Violation, normalize_key, scan_string_leaf, scan_tree

#: File extensions attempted as structured JSON/YAML first (falling back to
#: plain-text scanning on a parse error). Every other extension is scanned
#: as plain text directly, without a structured-parse attempt.
_STRUCTURED_EXTENSIONS = {".json", ".yaml", ".yml"}


def _parse_structured(text: str, suffix: str) -> object | None:
    """Attempt to parse `text` as JSON/YAML. Returns `None` on failure."""
    import json

    try:
        if suffix == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (ValueError, yaml.YAMLError):
        return None


def _scan_filename(relative_path: Path) -> list[Violation]:
    """Flag a filename/path segment that itself looks like a PII or secret field.

    Reuses `normalize_key()` from the shared heuristics module so a filename
    such as `patient_ssn.csv` or `api_key.txt` is judged by the exact same
    field-name denylist logic as a manifest's dict keys, rather than a
    separate, potentially-drifting filename-specific rule.
    """
    from infra.privacy_heuristics import (
        DIRECT_IDENTIFIER_FIELD_NAMES,
        SECRET_FIELD_NAMES,
    )

    violations: list[Violation] = []
    stem = relative_path.stem
    normalized = normalize_key(stem)
    tokens = set(normalized.split("_")) | {normalized}

    if tokens & DIRECT_IDENTIFIER_FIELD_NAMES:
        violations.append(
            Violation(
                category="direct_identifier",
                path=str(relative_path),
                message="filename resembles a direct-identifier field name",
                evidence=f"[redacted filename, length={len(str(relative_path))}]",
            )
        )
    if tokens & SECRET_FIELD_NAMES:
        violations.append(
            Violation(
                category="secret",
                path=str(relative_path),
                message="filename resembles a secret/credential field name",
                evidence=f"[redacted filename, length={len(str(relative_path))}]",
            )
        )
    return violations


def scan_directory(root: Path) -> list[Violation]:
    """Walk every file under `root` and return every violation found.

    Files are scanned relative to `root`, so the reported `path` in each
    `Violation` is stable across machines/export locations rather than
    embedding an absolute, machine-specific prefix.
    """
    violations: list[Violation] = []

    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative_path = file_path.relative_to(root)
        violations.extend(_scan_filename(relative_path))

        try:
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Binary or unreadable file: filename check above already ran;
            # content cannot be meaningfully scanned as text.
            continue

        suffix = file_path.suffix.lower()
        parsed = _parse_structured(text, suffix) if suffix in _STRUCTURED_EXTENSIONS else None
        if parsed is not None:
            violations.extend(scan_tree(parsed, root_path=str(relative_path)))
        else:
            violations.extend(scan_string_leaf(text, str(relative_path)))

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan every file under a directory for direct identifiers, "
            "quasi-identifiers, secrets, and unauthorized filesystem paths "
            "(Requirement 2.7)."
        )
    )
    parser.add_argument("directory", type=Path, help="Directory to scan, e.g. an export folder.")
    args = parser.parse_args(argv)

    root: Path = args.directory
    if not root.is_dir():
        print(f"scan_exports failed: {root} is not a directory", file=sys.stderr)
        return 1

    violations = scan_directory(root)

    if not violations:
        print(f"scan_exports: no violations found under {root}")
        return 0

    print(f"scan_exports: found {len(violations)} violation(s) under {root}:", file=sys.stderr)
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
