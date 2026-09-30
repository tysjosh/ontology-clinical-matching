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

Git provenance (Requirement 15.2, 15.4, 15.5) is recorded on *every* manifest
write, not at selected call sites: `build()` calls a git-commit resolver and
folds the resulting commit identifier, working-tree cleanliness state, and
per-artifact commit bindings into the manifest. The resolver is an injectable
callable (`ManifestWriter.git_provenance_resolver`) so tests can stub the
clean, dirty, and unavailable cases without touching the real repository. When
resolution fails the state is recorded *explicitly* as
`git_commit_unavailable` with a reason, never by omitting the field — an
absent field is indistinguishable from a field nobody bothered to write,
which is exactly the ambiguity Requirement 15.5 exists to remove.

Requirement 5.7 (the device a run executed on) is recorded through the same
accumulator via `set_device()`, and follows the same no-silent-omission rule:
an unset device records `device_unspecified` rather than dropping the key.

Nothing in this module imports torch, and nothing here may: the resolver
shells out to `git` only, so manifests remain writable in the torch-free base
environment (`design.md` Section 2.4).
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infra.nondeterminism import get_recorded_invocations
from infra.privacy_heuristics import Violation, scan_tree

#: Default location a manifest is written to, if the caller does not resolve
#: a stage-specific path itself (`design.md` Section 2.3:
#: `results/<experiment_id>/manifest.json`).
DEFAULT_RESULTS_ROOT = Path("results")

#: Project root, used as the working directory for the default git resolver.
#: `src/infra/manifest.py` -> `src/infra` -> `src` -> project root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Recorded working-tree state when the commit identifier could not be
#: resolved (Requirement 15.5). Also the value bound to every artifact in
#: that case, so a binding is still *present* and explicitly marked
#: unavailable rather than silently missing.
GIT_COMMIT_UNAVAILABLE = "git_commit_unavailable"

#: Recorded working-tree states when the commit identifier *was* resolved.
GIT_STATE_CLEAN = "clean"
GIT_STATE_DIRTY = "dirty"

#: Recorded device value when no caller set one (Requirement 5.7). Stages
#: with no tensor computation (cohort construction, causal analysis) legitimately
#: have no device; they record this sentinel rather than dropping the key.
DEVICE_UNSPECIFIED = "device_unspecified"

#: Seconds any single `git` invocation in the default resolver may take before
#: it is treated as unresolvable. A manifest write must not hang on a wedged
#: git process.
_GIT_TIMEOUT_SECONDS = 10

#: Re-exported for backwards compatibility - `Violation` used to be defined
#: in this module. It now lives in `src/infra/privacy_heuristics.py` so
#: `scripts/scan_exports.py` and `scripts/check_no_secrets.py` can reuse the
#: exact same type and detection heuristics (see that module's docstring).
__all__ = [
    "DEVICE_UNSPECIFIED",
    "GIT_COMMIT_UNAVAILABLE",
    "GIT_STATE_CLEAN",
    "GIT_STATE_DIRTY",
    "GitProvenance",
    "ManifestPrivacyError",
    "ManifestWriter",
    "PrivacyScanner",
    "Violation",
    "resolve_git_provenance",
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


@dataclass(frozen=True)
class GitProvenance:
    """The source state a manifest write happened from (Requirement 15.2).

    Three states are representable, and exactly three:

    - **clean**: `commit` is a non-empty identifier, `dirty` is `False`.
    - **dirty**: `commit` is a non-empty identifier, `dirty` is `True`, and
      `uncommitted_paths` names what was outstanding, so the token minter can
      report those paths when it declines to mint (Requirement 15.3).
    - **unavailable**: `commit` is `None` and `unavailable_reason` explains
      why (Requirement 15.5). Construct this case through
      `GitProvenance.unavailable()`.

    `__post_init__` refuses the dishonest combinations outright: an empty-string
    commit (which would serialize as a present-but-meaningless identifier), an
    unavailable state carrying no reason, and an unavailable state simultaneously
    claiming to know the working tree is dirty. A stub resolver in a test is held
    to the same rules as the real one.
    """

    commit: str | None
    dirty: bool = False
    uncommitted_paths: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.commit is not None and not self.commit.strip():
            raise ValueError(
                "GitProvenance.commit must be a non-empty identifier, or None for the "
                f"{GIT_COMMIT_UNAVAILABLE} state; got an empty string"
            )
        if self.commit is None:
            if not (self.unavailable_reason or "").strip():
                raise ValueError(
                    "GitProvenance with no commit must carry a non-empty "
                    "unavailable_reason so the unavailable state is explicit "
                    "(Requirement 15.5)"
                )
            if self.dirty or self.uncommitted_paths:
                raise ValueError(
                    "GitProvenance with no commit cannot also report working-tree "
                    "cleanliness; the cleanliness state is unknown when the commit "
                    "is unavailable"
                )

    @classmethod
    def unavailable(cls, reason: str) -> GitProvenance:
        """Build the explicit unavailable state with `reason` (Requirement 15.5)."""
        return cls(commit=None, unavailable_reason=_summarize_reason(reason))

    @property
    def available(self) -> bool:
        return self.commit is not None

    @property
    def state(self) -> str:
        """`"clean"`, `"dirty"`, or `"git_commit_unavailable"`."""
        if self.commit is None:
            return GIT_COMMIT_UNAVAILABLE
        return GIT_STATE_DIRTY if self.dirty else GIT_STATE_CLEAN

    @property
    def commit_binding_value(self) -> str:
        """The identifier an artifact is bound to (Requirement 15.4).

        Never `None`: an unresolvable commit binds the artifact to the explicit
        `git_commit_unavailable` marker, so every written artifact carries a
        binding and none is left unexplained.
        """
        return self.commit if self.commit is not None else GIT_COMMIT_UNAVAILABLE

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the manifest. Every key is always present."""
        return {
            "state": self.state,
            "commit": self.commit_binding_value,
            "dirty": self.dirty,
            "uncommitted_paths": list(self.uncommitted_paths),
            "unavailable_reason": self.unavailable_reason,
        }


#: An injectable git-commit resolver: any zero-argument callable returning a
#: `GitProvenance`. `resolve_git_provenance` is the production implementation;
#: tests pass stubs for the clean, dirty, and unavailable cases.
GitProvenanceResolver = Callable[[], GitProvenance]


def _summarize_reason(reason: str) -> str:
    """Reduce a resolver failure message to one short, path-free-ish line.

    Failure messages reach the manifest, which is privacy-scanned and may be
    published, so only the first line is kept and it is length-capped. This
    keeps a multi-line `git` usage dump or a long traceback string out of the
    artifact while still recording *why* the commit was unavailable.
    """
    first_line = (reason or "").strip().splitlines()[0].strip() if (reason or "").strip() else ""
    if not first_line:
        return "git commit could not be resolved (no reason reported)"
    return first_line[:200]


class _GitCommandError(Exception):
    """Internal: a `git` invocation in the default resolver did not succeed."""


def _run_git(args: list[str], repo_root: Path) -> str:
    """Run `git <args>` in `repo_root` and return stripped stdout.

    Raises `_GitCommandError` with a short message for every failure mode the
    resolver must survive: no `git` executable on PATH, a non-zero exit (not a
    repository, no commits yet), or a hung process.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise _GitCommandError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise _GitCommandError(f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS}s") from exc
    except OSError as exc:  # pragma: no cover - defensive
        raise _GitCommandError(f"git {args[0]} could not be executed: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        message = f"git {args[0]} exited {completed.returncode}"
        raise _GitCommandError(f"{message}: {detail}" if detail else message)
    return completed.stdout.strip()


def resolve_git_provenance(repo_root: Path | None = None) -> GitProvenance:
    """Resolve the current commit and working-tree cleanliness from `git`.

    Uses `git rev-parse HEAD` for the commit identifier and
    `git status --porcelain` for cleanliness, both run in `repo_root`
    (defaulting to this project's root).

    Returns the explicit unavailable state — never raises — when either call
    fails. Note that a *successful* commit lookup followed by a *failed*
    status lookup is also reported as unavailable: Requirement 15.2 asks for
    the commit and the cleanliness state together, and recording a commit
    while silently implying "clean" would be an attestation the resolver
    cannot make.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        commit = _run_git(["rev-parse", "HEAD"], root)
    except _GitCommandError as exc:
        return GitProvenance.unavailable(str(exc))
    if not commit:
        return GitProvenance.unavailable("git rev-parse HEAD returned no commit identifier")

    try:
        status = _run_git(["status", "--porcelain"], root)
    except _GitCommandError as exc:
        return GitProvenance.unavailable(
            f"commit resolved but working-tree cleanliness could not be determined: {exc}"
        )

    uncommitted = tuple(sorted(line[3:].strip() for line in status.splitlines() if line.strip()))
    return GitProvenance(
        commit=commit,
        dirty=bool(uncommitted),
        uncommitted_paths=uncommitted,
    )


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

    Git provenance and the run device are *not* accumulated per call site:
    `build()` resolves provenance through `git_provenance_resolver` on every
    call and emits `git_provenance` plus one `artifact_commit_bindings` entry
    per recorded checkpoint path and result directory (Requirement 15.2,
    15.4, 15.5). `device` is set once by whichever runner knows it
    (Requirement 5.7) and defaults to the `device_unspecified` sentinel.
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
    result_directories: list[str] = field(default_factory=list)
    predictions: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    allowed_path_prefixes: tuple[str, ...] = ()
    #: Injectable git-commit resolver (Requirement 15.2). `None` means the
    #: production `resolve_git_provenance`; tests pass a stub returning a
    #: clean, dirty, or unavailable `GitProvenance`.
    git_provenance_resolver: GitProvenanceResolver | None = None

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
        """Record a written checkpoint path.

        Every recorded path is bound to the resolved commit in
        `artifact_commit_bindings` at `build()` time (Requirement 15.4).
        """
        self.checkpoint_paths.append(str(path))

    def add_result_directory(self, path: str | Path) -> None:
        """Record a written result directory (Requirement 15.4).

        Same binding treatment as checkpoints: the directory appears in
        `artifact_commit_bindings` tied to the commit the run executed from.
        """
        self.result_directories.append(str(path))

    def set_device(self, device: str) -> None:
        """Record the device this run executed on (Requirement 5.7).

        Refuses an empty value rather than recording a blank device, so a
        caller cannot half-satisfy the requirement.
        """
        if not device or not device.strip():
            raise ValueError("device must be a non-empty string (e.g. 'cpu' or 'cuda')")
        self.device = device.strip()

    def resolve_git_provenance(self) -> GitProvenance:
        """Resolve git provenance through the injected or default resolver.

        A resolver that raises, or returns something that is not a
        `GitProvenance`, is folded into the explicit unavailable state rather
        than being allowed to abort the manifest write: a completed run must
        still produce a manifest, and Requirement 15.5 already specifies what
        that manifest should say when the commit cannot be established.
        """
        resolver = self.git_provenance_resolver or resolve_git_provenance
        try:
            provenance = resolver()
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            return GitProvenance.unavailable(
                f"git provenance resolver raised {type(exc).__name__}: {exc}"
            )
        if not isinstance(provenance, GitProvenance):
            return GitProvenance.unavailable(
                "git provenance resolver returned "
                f"{type(provenance).__name__}, not a GitProvenance"
            )
        return provenance

    def _artifact_commit_bindings(self, provenance: GitProvenance) -> list[dict[str, str]]:
        """Bind every written checkpoint path and result directory to the commit.

        Requirement 15.4. Bindings are emitted in a stable order (checkpoints
        in recorded order, then result directories) and carry the working-tree
        state alongside the identifier, so a binding to a dirty tree is never
        mistaken for a binding to a reproducible source state.
        """
        commit = provenance.commit_binding_value
        bindings: list[dict[str, str]] = []
        for path in self.checkpoint_paths:
            bindings.append(
                {
                    "artifact": path,
                    "artifact_kind": "checkpoint",
                    "git_commit": commit,
                    "git_state": provenance.state,
                }
            )
        for path in self.result_directories:
            bindings.append(
                {
                    "artifact": path,
                    "artifact_kind": "result_directory",
                    "git_commit": commit,
                    "git_state": provenance.state,
                }
            )
        return bindings

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

        Environment versions, git provenance, and nondeterminism invocations
        are computed fresh on every call, so `build()` (and therefore
        `write()`) always reflects the environment, the source state, and the
        nondeterminism registry as of the call, not as of `ManifestWriter`
        construction.
        """
        provenance = self.resolve_git_provenance()
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
            "result_directories": list(self.result_directories),
            "predictions": dict(self.predictions),
            "metrics": dict(self.metrics),
            "device": self.device if self.device else DEVICE_UNSPECIFIED,
            "git_provenance": provenance.as_dict(),
            "artifact_commit_bindings": self._artifact_commit_bindings(provenance),
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
