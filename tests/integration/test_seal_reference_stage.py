"""Integration test for `scripts/run_stage.py --stage seal-reference --reseal`
(task 6.2; Requirements 14.2, 14.6, 14.7).

This is the re-seal entry point `cohorts.sealed_reference.RESEAL_INSTRUCTIONS`
names verbatim, so what is asserted here is the behavior that string promises a
reader: a version-2 artifact written to the new path, a refusal when one is
already there, and a version-1 artifact left byte-identical and unread
(`design.md` Section 2.6 "Tension 4": "there is no code path that upgrades
it").

Invoked as a subprocess, matching `tests/integration/test_full_pipeline_stage.py`
and `tests/integration/test_real_data_pipeline_stage.py`, so the exit code being
asserted is a real process exit code. The stage runs against the tiny synthetic
fixture cohort (`--synthetic-cohort`), never real ACTG175 data, and writes into
`tmp_path`.

One deliberate divergence from the task text, recorded rather than papered
over: the task says "the three quantities", while the version-2 `quantities`
mapping written by `extended_randomized_reference_estimator` has FOUR keys ---
the three named risk-scale quantities (`risk_arm0`, `risk_arm1`,
`risk_difference`) plus the retained `log_hazard_ratio`, which Requirement 14.1
lists first ("the randomized Cox log-hazard-ratio, the randomized 96-week risk
for each arm, and the randomized 96-week risk difference"). The assertions below
pin the mapping as it actually is: four keys.

**Validates: Requirements 14.2, 14.6, 14.7**
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_STAGE_SCRIPT = REPO_ROOT / "scripts" / "run_stage.py"

V2_ARTIFACT_NAME = "internal_randomized_reference.v2.sealed.json"
V1_ARTIFACT_NAME = "internal_randomized_reference.sealed.json"

#: The four version-2 quantity names (see module docstring on "three" vs four).
EXPECTED_QUANTITY_NAMES = {
    "log_hazard_ratio",
    "risk_arm0",
    "risk_arm1",
    "risk_difference",
}

#: A deliberately CORRUPT version-1 payload: valid-looking enough to be
#: recognizably the v1 artifact by name, but not parseable as JSON at all. If
#: the re-seal stage read, parsed, or migrated the v1 file, the stage would
#: fail; it succeeding is evidence the file was never opened, which
#: byte-identity alone would not establish (an idempotent read-and-rewrite
#: would also leave the bytes equal).
CORRUPT_V1_BYTES = b'{"value": -0.212, "cohort_size": 855, "computed_at": <<TRUNCATED'


def _run_seal_reference(
    reference_dir: Path | None,
    *,
    reseal: bool = True,
    synthetic: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(RUN_STAGE_SCRIPT), "--stage", "seal-reference"]
    if reseal:
        command.append("--reseal")
    if synthetic:
        command.append("--synthetic-cohort")
    if reference_dir is not None:
        command += ["--reference-dir", str(reference_dir)]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_reseal_writes_a_v2_artifact_with_a_version_field_and_the_quantities(tmp_path):
    """Requirement 14.6: the written artifact records the sealed-artifact
    format version, alongside the risk-scale quantities of Requirement 14.1.

    Read back as raw JSON rather than through
    `cohorts.sealed_reference.read_sealed_artifact`, so the assertion is about
    the bytes the stage put on disk (and so this test needs no
    `LockedEvaluationToken`).
    """
    reference_dir = tmp_path / "reference"

    result = _run_seal_reference(reference_dir)

    assert result.returncode == 0, (
        f"seal-reference stage failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    artifact = reference_dir / V2_ARTIFACT_NAME
    assert artifact.exists()

    payload = json.loads(artifact.read_text())
    assert payload["format_version"] == 2
    assert set(payload["quantities"]) == EXPECTED_QUANTITY_NAMES
    assert payload["quantities"]["log_hazard_ratio"]["scale"] == "log_hazard_ratio"
    assert payload["quantities"]["risk_arm0"]["scale"] == "risk"
    assert payload["quantities"]["risk_arm1"]["scale"] == "risk"
    assert payload["quantities"]["risk_difference"]["scale"] == "risk_difference"
    assert payload["quantities"]["risk_difference"]["horizon_weeks"] == 96.0
    assert payload["metadata"]["estimator"] == "extended_randomized_reference_estimator"

    # The stage reports where it wrote and at what version, and never echoes a
    # sealed value -- the only sanctioned reader is `read_sealed_artifact`,
    # which requires a token.
    assert "seal-reference stage succeeded" in result.stdout
    assert "format version 2" in result.stdout
    assert str(artifact) in result.stdout
    assert "values withheld until the locked external evaluation" in result.stdout


def test_a_second_reseal_attempt_is_refused_and_leaves_the_artifact_untouched(tmp_path):
    """Requirement 14.7's write-side counterpart, and the promise
    `RESEAL_INSTRUCTIONS` makes verbatim: the entry point "refuses to run if a
    version-2 artifact already exists at the target path".
    """
    reference_dir = tmp_path / "reference"

    first = _run_seal_reference(reference_dir)
    assert first.returncode == 0, first.stderr

    artifact = reference_dir / V2_ARTIFACT_NAME
    bytes_before = artifact.read_bytes()
    mtime_before = artifact.stat().st_mtime_ns

    second = _run_seal_reference(reference_dir)

    assert second.returncode != 0
    assert "already exists" in second.stderr
    assert V2_ARTIFACT_NAME in second.stderr
    assert artifact.read_bytes() == bytes_before
    assert artifact.stat().st_mtime_ns == mtime_before


def test_a_preexisting_v1_artifact_is_left_byte_identical_and_unread(tmp_path):
    """`design.md` Section 2.6 "Tension 4": the migration is a re-seal onto a
    new path, so "the v1 artifact is left byte-identical and unread; there is
    no code path that upgrades it".

    The v1 file planted here is unparseable JSON (see `CORRUPT_V1_BYTES`): the
    re-seal succeeding proves the stage never opened it, and the bytes and
    mtime being unchanged prove it never wrote to it.
    """
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(parents=True)
    v1_artifact = reference_dir / V1_ARTIFACT_NAME
    v1_artifact.write_bytes(CORRUPT_V1_BYTES)
    mtime_before = v1_artifact.stat().st_mtime_ns

    result = _run_seal_reference(reference_dir)

    assert result.returncode == 0, (
        f"seal-reference stage failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert (reference_dir / V2_ARTIFACT_NAME).exists()
    assert v1_artifact.read_bytes() == CORRUPT_V1_BYTES
    assert v1_artifact.stat().st_mtime_ns == mtime_before
    # The v1 artifact is not named anywhere in the stage's output either: the
    # stage has no notion of it at all.
    assert V1_ARTIFACT_NAME not in result.stdout


def test_the_stage_refuses_without_the_reseal_flag(tmp_path):
    """Sealing is a once-per-dataset act, so `--reseal` is a required
    affirmation and nothing is written without it.
    """
    reference_dir = tmp_path / "reference"

    result = _run_seal_reference(reference_dir, reseal=False)

    assert result.returncode != 0
    assert "--reseal" in result.stderr
    assert not (reference_dir / V2_ARTIFACT_NAME).exists()
    assert not reference_dir.exists()


def test_a_synthetic_cohort_may_not_be_sealed_into_the_default_reference_dir():
    """A synthetic-fixture artifact must never be able to land at the path the
    locked external evaluation reads as the real internal randomized reference
    (`decisions.md` Section 2: nothing synthetic is ever mistaken for real
    ACTG175 data).

    Passes no `--reference-dir`, so the refusal is what keeps this test from
    writing into the repository's own `reference/` tree.
    """
    result = _run_seal_reference(None)

    assert result.returncode != 0
    assert "--synthetic-cohort" in result.stderr
    assert "--reference-dir" in result.stderr
