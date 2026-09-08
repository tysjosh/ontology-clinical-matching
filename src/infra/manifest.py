"""Artifact manifests and the privacy scanner.

`design.md` Section 3.1 requires every completed stage to write a privacy-safe
`manifest.json` capturing input-version hashes, the cohort definition, the
feature list, treatment/outcome definitions, model configuration, checkpoint
paths, predictions references, metric summaries, and environment versions
(Requirement 1.10). `ManifestWriter` is the single accumulator for that
content; it also folds in the recorded nondeterminism invocations from
`src/infra/nondeterminism.py` (Requirement 1.6), so a manifest's provenance is
never assembled by hand at more than one call site.

Requirement 1.11 / 2.2-2.4 / 2.7 require that a manifest never expose direct
identifiers, quasi-identifiers, secrets, or unauthorized filesystem paths.
`PrivacyScanner` is the single enforcement point for that rule:
`ManifestWriter.write()` always runs the scanner on the assembled manifest
*before* touching the filesystem, and raises `ManifestPrivacyError` — without
writing anything — if any violation is found. There is deliberately no way to
call `write()` and skip the scan.

Environment-version capture (the `pip freeze` requirement) uses
`importlib.metadata.distributions()` rather than shelling out to `pip freeze`
as a subprocess. `importlib.metadata` is stdlib (no extra dependency), reads
installed-package metadata directly from the environment without spawning a
process, and returns the same name/version information `pip freeze` would for
installed distributions - which is all Requirement 1.10 asks for ("environment
versions"). Shelling out to `pip` would add a subprocess dependency, a
platform-specific `pip` executable lookup, and a slower call, for no
additional information in this context.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infra.nondeterminism import get_recorded_invocations
from infra.privacy_heuristics import Violation, scan_tree

#: Default location a manifest is written to, if the caller does not resolve
#: a stage-specific path itself (`design.md` Section 2.3:
#: `results/<experiment_id>/manifest.json`).
DEFAULT_RESULTS_ROOT = Path("results")

#: Re-exported for backwards compatibility - `Violation` used to be defined
#: in this module. It now lives in `src/infra/privacy_heuristics.py` so
#: `scripts/scan_exports.py` and `scripts/check_no_secrets.py` can reuse the
#: exact same type and detection heuristics (see that module's docstring).
__all__ = [
    "ManifestPrivacyError",
    "ManifestWriter",
    "PrivacyScanner",
    "Violation",
]


class ManifestPrivacyError(Exception):
    """Raised when `PrivacyScanner.scan()` finds one or more violations.

    Carries the full list of `Violation` records so callers (and tests) can
    inspect what was found. `ManifestWriter.write()` raises this *before*
    creating or writing to the target file — a privacy violation always
    blocks the write entirely (Requirement 1.11, 2.2-2.4, 2.7).
    """

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        summary = "; ".join(str(v) for v in violations)
        super().__init__(f"Manifest privacy scan found {len(violations)} violation(s): {summary}")


class PrivacyScanner:
    """Scans an assembled manifest dict for privacy/security violations.

    This is a heuristic, structural scanner (see the module-level pattern
    tables above for scope and limitations), not an ML-based identifier
    detector. It walks the full manifest (dicts and lists, arbitrarily
    nested) and, for every leaf, checks:

    - **direct identifiers**: a dict key matching a denylist of PII field
      names (e.g. `name`, `ssn`, `mrn`, `email`) with a non-empty value, or a
      string value matching an SSN-like digit pattern.
    - **quasi-identifiers**: a dict key matching a denylist (e.g. `zip`,
      `dob`, `city`) with a non-empty value; a birth-date-like field and a
      zip-like field both present anywhere in the manifest additionally
      raise a combined violation, since the pair is more re-identifying than
      either alone.
    - **secrets**: a dict key matching a denylist of credential/secret field
      or environment-variable names with a non-empty value; a string value
      matching an AWS-access-key-ID shape, a common prefixed-token shape
      (e.g. `sk_live_...`), or the generic high-entropy heuristic (a long
      run of mixed-case alphanumeric characters).
    - **unauthorized filesystem paths**: a string value that is an absolute
      filesystem path and is not located under one of `allowed_path_prefixes`.
      Home-directory-shaped paths (`/Users/...`, `/home/...`, `~/...`) are
      still subject to the same `allowed_path_prefixes` exemption - a project
      whose own results root happens to live under a user's home directory
      is not itself a violation - but they receive a more specific message
      pointing out the username-leakage risk, since that is the common case
      this check exists to catch.
    """

    def __init__(self, allowed_path_prefixes: tuple[str, ...] = ()) -> None:
        self._allowed_path_prefixes = tuple(
            str(Path(prefix).resolve()) for prefix in allowed_path_prefixes
        )

    def scan(self, manifest: dict[str, Any]) -> list[Violation]:
        """Scan an assembled manifest dict and return every violation found.

        Returns an empty list if the manifest is clean. Never raises on its
        own - raising `ManifestPrivacyError` on a non-empty result is the
        caller's responsibility (`ManifestWriter.write()` does this).

        Delegates the actual walk and every detection rule to
        `infra.privacy_heuristics.scan_tree()` - the single shared
        implementation also used by `scripts/scan_exports.py` and
        `scripts/check_no_secrets.py`, so manifest scanning and export/commit
        scanning can never drift apart.
        """
        return scan_tree(manifest, allowed_path_prefixes=self._allowed_path_prefixes)


def _collect_environment_versions() -> dict[str, str]:
    """Snapshot installed-package name -> version pairs.

    Uses `importlib.metadata.distributions()` (stdlib, in-process) rather
    than shelling out to `pip freeze` - see the module docstring for the
    rationale. Distributions with unreadable metadata are skipped rather than
    raising, since a manifest write should not fail because of an unrelated
    malformed package's metadata.
    """
    versions: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata.get("Name") if dist.metadata is not None else None
        except Exception:  # pragma: no cover - defensive against odd metadata
            name = None
        if not name:
            continue
        try:
            version = dist.version
        except Exception:  # pragma: no cover - defensive
            continue
        versions.setdefault(name, version)
    return dict(sorted(versions.items()))


@dataclass
class ManifestWriter:
    """Accumulates one stage's artifact provenance and writes `manifest.json`.

    Every field below corresponds directly to a required element of the
    artifact manifest (Requirement 1.10): input-version hashes, the cohort
    definition, the feature list, treatment/outcome definitions, model
    configuration, checkpoint paths, predictions references, metric
    summaries, and environment versions. Nondeterminism invocations
    (Requirement 1.6) are pulled automatically from
    `src/infra/nondeterminism.get_recorded_invocations()` at `build()` time
    rather than being set explicitly, since that registry is the single
    source of truth for them.

    Accumulation is via the `add_*`/`set_*` methods below rather than direct
    field assignment, so every mutation site is easy to find and so future
    validation (e.g. shape-checking a checkpoint path) has one place to live.
    """

    experiment_id: str
    stage: str
    input_hashes: dict[str, str] = field(default_factory=dict)
    cohort_definition: dict[str, Any] = field(default_factory=dict)
    feature_list: list[str] = field(default_factory=list)
    treatment_definition: dict[str, Any] = field(default_factory=dict)
    outcome_definition: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    checkpoint_paths: list[str] = field(default_factory=list)
    predictions: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    allowed_path_prefixes: tuple[str, ...] = ()

    def add_input_hash(self, name: str, sha256_hex: str) -> None:
        """Record one input artifact's frozen SHA-256 hash (Requirement 2.8).

        Only the stable identifier/hash is stored here, never the underlying
        patient-level data.
        """
        self.input_hashes[name] = sha256_hex

    def set_cohort_definition(self, cohort_definition: dict[str, Any]) -> None:
        self.cohort_definition = dict(cohort_definition)

    def set_feature_list(self, feature_list: list[str]) -> None:
        self.feature_list = list(feature_list)

    def set_treatment_definition(self, treatment_definition: dict[str, Any]) -> None:
        self.treatment_definition = dict(treatment_definition)

    def set_outcome_definition(self, outcome_definition: dict[str, Any]) -> None:
        self.outcome_definition = dict(outcome_definition)

    def set_model_config(self, model_config: dict[str, Any]) -> None:
        self.model_config = dict(model_config)

    def add_checkpoint_path(self, path: str) -> None:
        self.checkpoint_paths.append(path)

    def set_predictions(self, predictions: dict[str, Any]) -> None:
        """Record a *reference* to predictions (e.g. a path or aggregate
        summary), never patient-level prediction values themselves
        (Requirement 2.6: patient-level predictions remain in approved
        protected storage, not in this public-facing manifest).
        """
        self.predictions = dict(predictions)

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics = dict(metrics)

    def build(self) -> dict[str, Any]:
        """Assemble the full manifest dict for the current accumulated state.

        Environment versions and nondeterminism invocations are computed
        fresh on every call, so `build()` (and therefore `write()`) always
        reflects the environment and nondeterminism registry as of the call,
        not as of `ManifestWriter` construction.
        """
        invocations = [
            {
                "function": invocation.function,
                "reason": invocation.reason,
                "timestamp": invocation.timestamp,
            }
            for invocation in get_recorded_invocations()
        ]
        return {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "input_hashes": dict(self.input_hashes),
            "cohort_definition": dict(self.cohort_definition),
            "feature_list": list(self.feature_list),
            "treatment_definition": dict(self.treatment_definition),
            "outcome_definition": dict(self.outcome_definition),
            "model_config": dict(self.model_config),
            "checkpoint_paths": list(self.checkpoint_paths),
            "predictions": dict(self.predictions),
            "metrics": dict(self.metrics),
            "environment_versions": _collect_environment_versions(),
            "nondeterminism_invocations": invocations,
        }

    def write(self, path: str | Path) -> Path:
        """Scan the assembled manifest and write it to `path` if it is clean.

        Runs `PrivacyScanner.scan()` on the freshly-built manifest first.
        If any violation is found, raises `ManifestPrivacyError` and does
        **not** create or write to `path` at all (Requirement 1.11,
        2.2-2.4, 2.7). Only on a clean scan does this method touch the
        filesystem: it creates parent directories as needed and writes the
        manifest as indented, sorted-key JSON.

        Args:
            path: destination file path for `manifest.json`.

        Returns:
            The `Path` the manifest was written to.

        Raises:
            ManifestPrivacyError: if the scanner finds any violation.
        """
        manifest = self.build()
        scanner = PrivacyScanner(allowed_path_prefixes=self.allowed_path_prefixes)
        violations = scanner.scan(manifest)
        if violations:
            raise ManifestPrivacyError(violations)

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        return target
