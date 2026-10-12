"""Write the public/protected audit-artifact pair for one `AuditReport`
(task 17.1; Requirement 12.5, 12.6, 12.7, 19.5; `design.md` Section 3.10).

`build_audit_report` (`src/witnesses/audit_report.py`) returns a single,
*protected*-shaped `AuditReport` object: it always retains whatever real
`estimate`/interval was computed, even for a suppressed candidate (see
that module's docstring). This module is where the public/protected split
`design.md` Section 3.10 describes actually happens, at write time:

- `results/<experiment_id>/audit/protected/<claim_id>.json` -- the full
  report, unredacted, written at owner-only `0o600` permissions (mirroring
  `cohorts.sealed_reference.SealedReference.write_sealed_artifact`'s
  convention for the sealed reference it sits alongside -- Requirement
  12.6's "protected ... alongside `reference/`").
- `results/<experiment_id>/audit/public/<claim_id>.json` -- a derived
  copy with `estimate`, `estimate_ci_low`, and `estimate_ci_high` forced
  to `None` whenever `report.estimate_status == "suppressed"` (Requirement
  12.5). A `"released"` report's public copy is identical to its
  protected copy (there is nothing to redact).

Why a derived copy, not a different code path
-------------------------------------------------------------------------

Both artifacts are built from the exact same `report.to_dict()` call, so
every other field (witness outcomes, overlap diagnostics, hashes, seed,
population counts, release reasons, ...) is guaranteed to agree between
the two artifacts. Only the three numeric estimate fields differ, and only
when `estimate_status == "suppressed"` -- this keeps "what gets redacted"
a single, obviously-correct `if` rather than two independently-maintained
serialization paths that could silently drift apart.

The exact numeric value's *string form* is checked to never leak into the
public artifact's full serialized text in `tests/unit/test_suppression.py`
(Property 16), since a drive-by field in `population`/`treatment`/
`provenance` could in principle still echo it if a caller passed the raw
number through one of those caller-assembled dicts; this module's own
redaction only has visibility into the three dedicated estimate fields.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from witnesses.audit_report import AuditReport


def _public_dict(report: AuditReport) -> dict:
    """`report.to_dict()`, with the three numeric estimate fields forced
    to `None` whenever `report.estimate_status == "suppressed"`
    (Requirement 12.5)."""
    as_dict = report.to_dict()
    if report.estimate_status == "suppressed":
        as_dict["estimate"] = None
        as_dict["estimate_ci_low"] = None
        as_dict["estimate_ci_high"] = None
    return as_dict


def audit_artifact_paths(experiment_dir: str | Path, claim_id: str) -> tuple[Path, Path]:
    """Return `(public_path, protected_path)` for one claim, without
    writing anything -- used by both `write_audit_artifacts` and any
    caller/test that needs to know the destination ahead of time."""
    experiment_dir = Path(experiment_dir)
    public_path = experiment_dir / "audit" / "public" / f"{claim_id}.json"
    protected_path = experiment_dir / "audit" / "protected" / f"{claim_id}.json"
    return public_path, protected_path


def write_audit_artifacts(
    report: AuditReport, experiment_dir: str | Path, claim_id: str
) -> tuple[Path, Path]:
    """Write both the public and protected audit artifacts for one
    `AuditReport` (task 17.1).

    Args:
        report: the already-built `AuditReport` (the protected-shaped
            record; see module docstring).
        experiment_dir: `results/<experiment_id>`.
        claim_id: a stable, unique identifier for this candidate estimate
            (e.g. `f"{sensitivity_variant or 'primary'}_seed{seed}"`),
            used as both artifacts' filename stem.

    Returns:
        `(public_path, protected_path)`, both written.
    """
    public_path, protected_path = audit_artifact_paths(experiment_dir, claim_id)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    protected_path.parent.mkdir(parents=True, exist_ok=True)

    public_text = json.dumps(_public_dict(report), indent=2, sort_keys=True, default=str)
    public_path.write_text(public_text)

    protected_text = json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str)
    protected_path.write_text(protected_text)
    # Owner read/write only (0o600) -- defense-in-depth, mirroring
    # `cohorts.sealed_reference.SealedReference.write_sealed_artifact`'s
    # convention for `reference/`, which this directory sits alongside
    # (Requirement 12.6).
    os.chmod(protected_path, stat.S_IRUSR | stat.S_IWUSR)

    return public_path, protected_path
