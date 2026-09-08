"""Shared privacy/secret detection heuristics.

Requirement 2.4 and 2.7 both depend on the same underlying detection logic:
direct identifiers, quasi-identifiers, secrets, and unauthorized filesystem
paths must be caught consistently wherever they might appear — inside an
artifact manifest (`src/infra/manifest.py`'s `PrivacyScanner`, Requirement
1.11 / 2.2-2.4 / 2.7) and inside exported directories of files
(`scripts/scan_exports.py`, Requirement 2.7) or files about to be committed
(`scripts/check_no_secrets.py`, Requirement 2.4).

This module is the single place those heuristics live. `manifest.py`'s
`PrivacyScanner` and `scripts/scan_exports.py` both call into
`scan_tree()`/`scan_text()` here rather than reimplementing regexes or
field-name denylists separately, so the two call sites can never drift out
of sync with each other.

This is a synthetic/structural check, not an ML NER system. It catches the
categories of leakage the pipeline is realistically at risk of introducing
(a stray raw identifier field, a copy-pasted credential, an absolute
developer-machine path) using regexes and field-name denylists. It will not
catch identifiers embedded in unstructured free text in formats not covered
below, or secrets that don't match a known shape. It is defense-in-depth,
not a substitute for the manual privacy review required by Requirement 2.9.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Violation:
    """One privacy/secret violation found by a scan.

    `evidence` is always a redacted description (never the raw offending
    value) - a `Violation` may end up inside a raised error or a printed CLI
    report, which could propagate into logs, so the scanner must not itself
    become a secondary leakage path.
    """

    category: str  # "direct_identifier" | "quasi_identifier" | "secret" | "unauthorized_path"
    path: str  # dotted/bracketed location, or file path, of the offending value
    message: str
    evidence: str

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"[{self.category}] {self.path}: {self.message} ({self.evidence})"


def redact(value: Any) -> str:
    """Redact a value for inclusion in a `Violation`'s evidence field.

    Never returns the raw value - only its type and length - so that raising
    or logging a `Violation` cannot itself leak the sensitive content it
    describes.
    """
    text = str(value)
    return f"[redacted {type(value).__name__}, length={len(text)}]"


def normalize_key(key: str) -> str:
    return re.sub(r"[\s\-]+", "_", key.strip().lower())


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list | dict | tuple | set):
        return len(value) > 0
    return True


# ---------------------------------------------------------------------------
# Heuristic pattern tables.
# ---------------------------------------------------------------------------

#: Field names (normalized: lowercased, whitespace/hyphens -> underscore)
#: that, if present with a non-empty value anywhere in a scanned structure,
#: are treated as direct identifiers regardless of their value's shape.
DIRECT_IDENTIFIER_FIELD_NAMES = {
    "name",
    "full_name",
    "first_name",
    "last_name",
    "patient_name",
    "patient_full_name",
    "ssn",
    "social_security_number",
    "mrn",
    "medical_record_number",
    "email",
    "email_address",
    "phone",
    "phone_number",
    "address",
    "street_address",
}

#: Field names treated as quasi-identifiers on their own (Requirement 2.2-2.3).
QUASI_IDENTIFIER_FIELD_NAMES = {
    "zip",
    "zip_code",
    "zipcode",
    "postal_code",
    "dob",
    "date_of_birth",
    "birth_date",
    "birthdate",
    "age_exact",
    "exact_age",
    "city",
    "county",
}

#: The subset of `QUASI_IDENTIFIER_FIELD_NAMES` representing birth-date-like
#: fields, used for the birth-date + zip combination check.
DOB_LIKE_FIELDS = {"dob", "date_of_birth", "birth_date", "birthdate"}

#: The subset of `QUASI_IDENTIFIER_FIELD_NAMES` representing zip-like fields.
ZIP_LIKE_FIELDS = {"zip", "zip_code", "zipcode", "postal_code"}

#: Field names that, if present with a non-empty value, are treated as
#: secrets regardless of the value's shape (common secret/credential
#: field and environment-variable names).
SECRET_FIELD_NAMES = {
    "password",
    "passwd",
    "secret",
    "secret_key",
    "api_key",
    "apikey",
    "access_key",
    "access_key_id",
    "secret_access_key",
    "aws_secret_access_key",
    "aws_access_key_id",
    "private_key",
    "token",
    "auth_token",
    "client_secret",
    "credential",
    "credentials",
}

#: SSN-like digit pattern: NNN-NN-NNNN, not preceded/followed by another digit.
SSN_REGEX = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

#: AWS access-key-ID shape (`AKIA...`/`ASIA...` + 16 upper/digit chars).
AWS_ACCESS_KEY_REGEX = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")

#: Common "prefix_restofkey" secret-token shapes (e.g. `sk_live_...`,
#: `api-key-...`), case-insensitive.
PREFIXED_TOKEN_REGEX = re.compile(
    r"\b(?:sk|pk|api|apikey|token|secret)[_-][A-Za-z0-9]{16,}\b", re.IGNORECASE
)

#: Generic high-entropy token heuristic: a long run of mixed upper/lower/digit
#: characters. Deliberately requires *all three* character classes so that
#: plain lowercase-hex content (sha256 hashes, git commit SHAs - which a
#: manifest legitimately and intentionally contains, per Requirement 1.10 and
#: 2.8) never matches: hex digests only use `0-9a-f`, never uppercase.
#:
#: `/` is deliberately excluded from the character class (even though it is
#: common in base64 alphabets) because including it lets the pattern span an
#: entire filesystem path as one "token" - a path's directory separators
#: would otherwise stitch unrelated, individually-short path segments into a
#: single long candidate and trigger false positives on ordinary absolute
#: paths. Path values are still checked separately by the absolute-path
#: rules below; this pattern only needs to catch tokens *within* a single
#: path segment or a non-path string.
_HIGH_ENTROPY_CANDIDATE_REGEX = re.compile(r"\b[A-Za-z0-9+_-]{20,}\b")


def looks_like_high_entropy_token(candidate: str) -> bool:
    has_upper = any(c.isupper() for c in candidate)
    has_lower = any(c.islower() for c in candidate)
    has_digit = any(c.isdigit() for c in candidate)
    return has_upper and has_lower and has_digit


#: Absolute filesystem path shapes (POSIX root, `~`, or a Windows drive
#: letter). Only absolute paths are ever flagged - relative paths such as
#: `results/checkpoints/model.pt` are the expected, safe shape for manifest
#: path fields and are never absolute-path violations.
_ABSOLUTE_PATH_REGEX = re.compile(r"^(?:/|~(?:/|$)|[A-Za-z]:\\)")

#: Home-directory path shapes specifically - flagged with a more specific
#: message because they are the primary way a local username leaks into a
#: public artifact.
_HOME_DIR_PATH_REGEX = re.compile(r"^(?:/Users/|/home/|~(?:/|$)|[A-Za-z]:\\Users\\)")


def is_allowed_path(path_value: str, allowed_path_prefixes: tuple[str, ...]) -> bool:
    if not allowed_path_prefixes:
        return False
    try:
        resolved = str(Path(path_value).resolve())
    except (OSError, ValueError):  # pragma: no cover - defensive
        return False
    return any(
        resolved == prefix or resolved.startswith(prefix.rstrip("/") + "/")
        for prefix in allowed_path_prefixes
    )


def scan_string_leaf(
    value: str,
    path: str,
    allowed_path_prefixes: tuple[str, ...] = (),
) -> list[Violation]:
    """Run every string-level (regex-based) check against one leaf value.

    Checks for an SSN-like digit pattern, a known secret/API-key shape, a
    generic high-entropy token, and an unauthorized absolute filesystem
    path. Does not look at field/key names - see `scan_field_name()` for
    that half of the detection surface.
    """
    violations: list[Violation] = []

    if SSN_REGEX.search(value):
        violations.append(
            Violation(
                category="direct_identifier",
                path=path,
                message="value matches an SSN-like digit pattern (NNN-NN-NNNN)",
                evidence=redact(value),
            )
        )

    if AWS_ACCESS_KEY_REGEX.search(value) or PREFIXED_TOKEN_REGEX.search(value):
        violations.append(
            Violation(
                category="secret",
                path=path,
                message="value matches a known secret/API-key shape",
                evidence=redact(value),
            )
        )
    else:
        for candidate in _HIGH_ENTROPY_CANDIDATE_REGEX.findall(value):
            if looks_like_high_entropy_token(candidate):
                violations.append(
                    Violation(
                        category="secret",
                        path=path,
                        message=(
                            "value contains a long mixed-case alphanumeric token "
                            "resembling a high-entropy secret"
                        ),
                        evidence=redact(value),
                    )
                )
                break

    if _ABSOLUTE_PATH_REGEX.match(value) and not is_allowed_path(value, allowed_path_prefixes):
        if _HOME_DIR_PATH_REGEX.match(value):
            message = (
                "absolute path is under a home directory and outside the allowed "
                "path prefixes; it may leak a local username"
            )
        else:
            message = "absolute filesystem path is outside the allowed path prefixes"
        violations.append(
            Violation(
                category="unauthorized_path",
                path=path,
                message=message,
                evidence=redact(value),
            )
        )

    return violations


def scan_field_name(
    key: str,
    value: Any,
    path: str,
    quasi_hits: dict[str, list[str]] | None = None,
) -> list[Violation]:
    """Run every field-name-based (denylist) check for one key/value pair.

    `quasi_hits` (if provided) is mutated in place to record dob-like and
    zip-like field paths, so a caller walking a whole tree can later raise
    the combined dob+zip violation via `combined_quasi_identifier_violation()`.
    """
    violations: list[Violation] = []
    if not is_present(value):
        return violations

    normalized_key = normalize_key(str(key))

    if normalized_key in DIRECT_IDENTIFIER_FIELD_NAMES:
        violations.append(
            Violation(
                category="direct_identifier",
                path=path,
                message=f"field name '{key}' is a direct-identifier field",
                evidence=redact(value),
            )
        )

    if normalized_key in QUASI_IDENTIFIER_FIELD_NAMES:
        violations.append(
            Violation(
                category="quasi_identifier",
                path=path,
                message=f"field name '{key}' is a quasi-identifier field",
                evidence=redact(value),
            )
        )
        if quasi_hits is not None:
            if normalized_key in DOB_LIKE_FIELDS:
                quasi_hits.setdefault("dob", []).append(path)
            if normalized_key in ZIP_LIKE_FIELDS:
                quasi_hits.setdefault("zip", []).append(path)

    if normalized_key in SECRET_FIELD_NAMES:
        violations.append(
            Violation(
                category="secret",
                path=path,
                message=f"field name '{key}' is a secret/credential field",
                evidence=redact(value),
            )
        )

    return violations


def combined_quasi_identifier_violation(
    quasi_hits: dict[str, list[str]],
) -> Violation | None:
    """Build the combined dob+zip violation, if both were recorded in `quasi_hits`."""
    if quasi_hits.get("dob") and quasi_hits.get("zip"):
        dob_paths = ", ".join(quasi_hits["dob"])
        zip_paths = ", ".join(quasi_hits["zip"])
        return Violation(
            category="quasi_identifier",
            path=f"{dob_paths} + {zip_paths}",
            message=(
                "birth-date-like and zip-like fields both present; this "
                "combination is substantially more re-identifying than either "
                "field alone"
            ),
            evidence="[redacted combination]",
        )
    return None


def scan_tree(
    node: Any,
    allowed_path_prefixes: tuple[str, ...] = (),
    root_path: str = "",
) -> list[Violation]:
    """Recursively scan a JSON-like structure (dicts, lists/tuples, strings).

    This walks the full structure and, for every leaf, applies both halves
    of the detection surface: field-name denylist checks (for dict keys) and
    string-level regex checks (for string values), plus the combined dob+zip
    check across the whole structure. This is the routine both
    `PrivacyScanner.scan()` (over an assembled manifest dict) and
    `scripts/scan_exports.py` (over a parsed JSON/YAML file, or a bare
    string of file content) call into, so the two never implement the walk
    twice.
    """
    violations: list[Violation] = []
    quasi_hits: dict[str, list[str]] = {}
    _scan_node(node, root_path, allowed_path_prefixes, violations, quasi_hits)
    combined = combined_quasi_identifier_violation(quasi_hits)
    if combined is not None:
        violations.append(combined)
    return violations


def _scan_node(
    node: Any,
    path: str,
    allowed_path_prefixes: tuple[str, ...],
    violations: list[Violation],
    quasi_hits: dict[str, list[str]],
) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{path}.{key}" if path else str(key)
            violations.extend(scan_field_name(str(key), value, key_path, quasi_hits))
            _scan_node(value, key_path, allowed_path_prefixes, violations, quasi_hits)
    elif isinstance(node, list | tuple):
        for index, item in enumerate(node):
            _scan_node(item, f"{path}[{index}]", allowed_path_prefixes, violations, quasi_hits)
    elif isinstance(node, str):
        violations.extend(scan_string_leaf(node, path, allowed_path_prefixes))
