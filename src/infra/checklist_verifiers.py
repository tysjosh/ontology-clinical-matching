"""Machine verifiers for the twelve go/no-go checklist items (task 2.5;
`decisions.md` Section 21; `design.md` Section 2.6 tension 8; Requirements
13.13, 18.4).

Why this module exists
----------------------

`scripts/check_go_no_go.py` was written as pure gating logic: it read
`configs/go_no_go_checklist.yaml`, where each of the twelve items is a
boolean that "some *other* process (a human reviewer ticking a box, or an
earlier automated pipeline stage writing its own result here) has already
determined." That leaves one hole big enough to drive a paper through: a
human can attest `leakage_tests_passed: true` while the leakage tests are
red, and nothing in the repository notices. The attestation is the only
evidence, so the attestation cannot be wrong.

This module closes that hole for the items where evidence exists. Each of
the twelve items gets a verifier that returns one of three outcomes:

- `verified` — evidence was found and it supports the item;
- `failed` — evidence was found and it *contradicts* the item;
- `unverifiable` — no evidence exists that a machine can read, so the
  human attestation is the only signal available.

`check_go_no_go.py` reports every verifier result next to the attested
boolean and refuses to mint a `LockedEvaluationToken` when a
machine-verifiable item is attested `true` but its verifier returns
`failed`. An attestation may still be the *only* evidence (the three
`unverifiable` items below), but it may never contradict evidence that
exists.

The `verified` / `failed` / `unverifiable` boundary
--------------------------------------------------

One rule, applied uniformly: **a missing evidence source is
`unverifiable`; a present evidence source that disagrees with the item is
`failed`.** So a deleted `decisions.md` (e.g. a deployment that ships
`src/` without `.kiro/`) does not fail the protocol-freeze items — it
merely means there is nothing to check. A *present* `decisions.md` whose
frozen-status header is gone, or a present `configs/target_trials/main.yaml`
whose adjustment set no longer matches `decisions.md` Section 13, is a
failure: the evidence is there and it says the item is untrue.

The three groups
----------------

1. **Document-checkable** (`frozen_protocol`, `frozen_causal_diagram`,
   `frozen_adjustment_set`, `target_population_defined`,
   `statistical_procedures_locked`) — checked against `decisions.md` and
   `configs/target_trials/main.yaml`, cross-checking the config against the
   document's own prose wherever the document states the value in a
   machine-readable form (the adjustment set is the cleanest case: the
   Section 13 prose list is normalized and compared name-for-name against
   the config's `adjustment_set`, so editing either one alone fails).

2. **Test/scan-checkable** (`leakage_tests_passed`,
   `splits_cross_fitting_verified`, `privacy_checks_passed`,
   `one_command_reproduction_verified`) — checked by invoking the existing
   test selections that already encode those guarantees, plus
   `scripts/scan_exports.py` for the privacy item. These verifiers do not
   re-implement any check; they run the ones the repository already has.

3. **Attestation-only** (`estimand_alignment_checked`,
   `reference_inaccessible_to_model_selection`, `thresholds_frozen`) —
   permanently `unverifiable`, and deliberately so. Each asserts something
   about *human process ordering* that leaves no artifact: that a report was
   read and understood, that nobody looked at the sealed reference while
   choosing models, that thresholds were fixed *before* rather than *after*
   seeing test-split behavior. A repository can hold identical bytes in both
   the honest and the dishonest world, so a machine check here would be
   theater. These stay human-set (task 2.5's explicit instruction).

Recursion safety
----------------

Group 2's verifiers shell out to `pytest`. If a verifier ran a selection
that included this module's own tests, `pytest` would run a test that runs
`pytest` that runs the test, forever. Two independent guards prevent that:

- **Selection discipline.** Every selection in `TEST_SELECTIONS` names
  specific test modules, none of which is `tests/unit/test_checklist_verifiers.py`
  or `tests/unit/test_check_go_no_go.py`. `SELF_TEST_MODULES` records those
  two, `_assert_no_self_selection()` enforces the invariant at import time,
  and `tests/unit/test_checklist_verifiers.py` asserts it as well.
- **Environment flag.** `run_pytest_selection()` sets
  `CHECKLIST_VERIFIER_ACTIVE=1` in the subprocess environment and refuses
  (returning `unavailable`) if that variable is already set. Even if a
  future edit adds a self-referential selection, the nested invocation stops
  at depth one instead of forking without bound.

Torch-free by construction: this module imports only the standard library,
`yaml`, and (lazily, inside verifiers) `protocols.schema` and
`evaluation.statistics`, none of which pull in torch. The selections it runs
are torch-free test modules.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

#: Repository root, resolved from this file's location (`src/infra/` ->
#: repo root), matching `scripts/check_go_no_go.py`'s convention.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DECISIONS_PATH = REPO_ROOT / ".kiro" / "specs" / "causal-witness-pwm" / "decisions.md"
DEFAULT_TARGET_TRIAL_PATH = REPO_ROOT / "configs" / "target_trials" / "main.yaml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_EXPORT_SCANNER = REPO_ROOT / "scripts" / "scan_exports.py"
DEFAULT_REPRODUCTION_SCRIPT = REPO_ROOT / "scripts" / "run_paper.py"

#: Set in the environment of every `pytest` subprocess a verifier spawns.
#: Its presence means "a checklist verifier is already running above me", and
#: `run_pytest_selection()` refuses to spawn another (see "Recursion safety"
#: in the module docstring).
NESTED_VERIFICATION_ENV_VAR = "CHECKLIST_VERIFIER_ACTIVE"

#: Wall-clock ceiling for any subprocess a verifier spawns. A verifier that
#: hangs must degrade to `unverifiable`, never block the gate forever.
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 600.0

#: The two test modules that exercise the verifiers and the gate. No test
#: selection below may name either one -- see "Recursion safety".
SELF_TEST_MODULES: tuple[str, ...] = (
    "tests/unit/test_checklist_verifiers.py",
    "tests/unit/test_check_go_no_go.py",
)


class VerifierOutcome(str, Enum):
    """The three outcomes a checklist verifier may return."""

    VERIFIED = "verified"
    UNVERIFIABLE = "unverifiable"
    FAILED = "failed"

    def __str__(self) -> str:  # pragma: no cover - display convenience
        return self.value


@dataclass(frozen=True)
class VerifierResult:
    """A verifier's outcome for one checklist item, with its evidence.

    Attributes:
        item: the checklist item name (a member of `CHECKLIST_ITEM_NAMES`).
        outcome: `verified`, `failed`, or `unverifiable`.
        evidence: human-readable description of what was inspected and what
            it showed. Always populated -- an outcome without evidence is
            exactly the unaccountable attestation this module replaces.
    """

    item: str
    outcome: VerifierOutcome
    evidence: str

    @property
    def failed(self) -> bool:
        return self.outcome is VerifierOutcome.FAILED

    def __str__(self) -> str:
        return f"{self.item}: {self.outcome.value} ({self.evidence})"


class CommandStatus(str, Enum):
    """Outcome of invoking an external check (a test selection or a scan)."""

    PASSED = "passed"
    FAILED = "failed"
    #: The check could not be run at all (files absent, tool missing,
    #: timeout, nested invocation) -- distinct from "ran and failed".
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CommandOutcome:
    """Result of one external check invocation."""

    status: CommandStatus
    detail: str


#: Injection points on `VerificationContext`. Both receive the context so a
#: replacement runner sees the same paths the verifiers do.
TestSelectionRunner = Callable[["VerificationContext", tuple[str, ...]], CommandOutcome]
ExportScanRunner = Callable[["VerificationContext", Path], CommandOutcome]


# ---------------------------------------------------------------------------
# Frozen values transcribed from `decisions.md`.
#
# Same convention as `tests/frozen/test_decisions_match.py`: where a check
# needs a literal, the literal lives here with a citation, and a mismatch is
# resolved by amending `decisions.md` -- never by editing the constant to
# match whatever the code or config currently says.
# ---------------------------------------------------------------------------

#: `decisions.md` header: "**Status:** Frozen. Changes after this point
#: require a documented amendment (see Section 26)..."
FROZEN_STATUS_MARKER = "**Status:** Frozen"

#: Section headings that must exist for the protocol to count as frozen
#: (`decisions.md` Sections 3-7: treatment comparison, target population,
#: time zero/follow-up, endpoints, estimands).
PROTOCOL_SECTION_HEADINGS: tuple[str, ...] = (
    "## 3. Treatment comparison",
    "## 4. Target population",
    "## 5. Time zero and follow-up",
    "## 6. Endpoints",
    "## 7. Estimands",
)

TARGET_POPULATION_HEADING = "## 4. Target population"
CAUSAL_GRAPH_HEADING = "## 13. Causal graph and adjustment set"
STATISTICAL_PLAN_HEADING = "## 19. Statistical plan"
AMENDMENT_LOG_HEADING = "## 26. Amendment log"

#: `decisions.md` Section 13's causal graph, as nodes and directed edges.
FROZEN_CAUSAL_DIAGRAM_NODES: frozenset[str] = frozenset({"X", "A", "L", "Y", "S", "C"})
FROZEN_CAUSAL_DIAGRAM_EDGES: frozenset[str] = frozenset(
    {
        "X->A",
        "A->L",
        "L->Y",
        "X->Y",
        "X->S",
        "A->S",
        "A->C",
        "X->C",
        "L->C",
    }
)

#: The two relations Section 13 states in prose beside its ASCII diagram
#: (which draws only the X/A/L/Y core). Compared after `_normalize()`, so the
#: document's arrow glyphs and this transcription's ASCII arrows agree.
FROZEN_SECTION_13_RELATIONS: tuple[str, ...] = (
    "X, A -> Observational-selection indicator S",
    "A, X, L -> Censoring C",
)

#: The prose lead-in to Section 13's frozen adjustment-set list. The list
#: itself is *not* transcribed here: `verify_frozen_adjustment_set()` parses
#: it out of the document and compares it against the config, so the document
#: stays the single source of truth.
ADJUSTMENT_SET_PROSE_MARKER = "**Baseline adjustment set (frozen):**"

#: `decisions.md` Section 4's prohibition, which the config encodes as
#: `population.post_hoc_outcome_exclusion_permitted: false`.
POST_HOC_EXCLUSION_PROHIBITION = (
    "No post hoc exclusion based on observed outcomes is permitted at any stage."
)

#: `decisions.md` Section 19's locked statistical procedures, checked as
#: prose markers within that section.
STATISTICAL_PLAN_MARKERS: tuple[str, ...] = (
    "Holm correction",
    "1,000 patient-level bootstrap replicates",
    "Paired bootstrap comparisons",
    "95% confidence intervals",
)

#: `decisions.md` Section 19's three primary model comparisons, as encoded in
#: `evaluation.statistics.PRIMARY_COMPARISONS`.
FROZEN_PRIMARY_COMPARISONS: frozenset[str] = frozenset(
    {
        "baseline_vs_intervention_aware",
        "intervention_aware_vs_ipcw_aipw",
        "gated_vs_ungated",
    }
)

#: `decisions.md` Sections 15/19: 1,000 patient-level bootstrap replicates.
FROZEN_BOOTSTRAP_REPLICATES = 1000

#: `decisions.md` Section 19 / Requirement 14.6-14.7: minimum independent
#: configurations before a predictive-causal correlation claim.
FROZEN_MIN_CONFIGURATIONS = 5


# ---------------------------------------------------------------------------
# Test selections.
#
# Each selection names the existing test modules that already encode the
# guarantee the checklist item asserts. Selections are deliberately narrow:
# they must run in seconds (the gate is interactive), must not require the
# optional `neural` extra (torch), and must not name `SELF_TEST_MODULES`.
# ---------------------------------------------------------------------------

#: `leakage_tests_passed`: `src/data/leakage_audit.py`'s suite (the auditor
#: that enforces `decisions.md` Section 5's permanent exclusion of
#: follow-up-derived variables from the baseline adjustment set) together
#: with the time-zero tests, since a leakage claim is exactly a claim about
#: what was knowable at time zero.
LEAKAGE_TEST_SELECTION: tuple[str, ...] = (
    "tests/unit/test_leakage_audit.py",
    "tests/unit/test_time_zero.py",
)

#: `splits_cross_fitting_verified`: the stratified-split tests and the
#: propensity tests, which are where fold construction and cross-fitted
#: (zero train/prediction overlap) propensity estimation are asserted.
SPLITS_TEST_SELECTION: tuple[str, ...] = (
    "tests/unit/test_splitting.py",
    "tests/unit/test_propensity.py",
)

#: `privacy_checks_passed`: the export scanner's own tests, the
#: minimum-cell-size suppression tests, and the secret-scan tests. Combined
#: with an actual `scan_exports` run over the results root.
PRIVACY_TEST_SELECTION: tuple[str, ...] = (
    "tests/unit/test_scan_exports.py",
    "tests/unit/test_min_cell_size.py",
    "tests/unit/test_check_no_secrets.py",
)

#: `one_command_reproduction_verified`: the end-to-end `run_paper.py` tests,
#: which are the existing mechanical statement of "one documented command
#: reproduces tables/figures from saved predictions".
REPRODUCTION_TEST_SELECTION: tuple[str, ...] = ("tests/e2e/test_run_paper.py",)

TEST_SELECTIONS: dict[str, tuple[str, ...]] = {
    "leakage_tests_passed": LEAKAGE_TEST_SELECTION,
    "splits_cross_fitting_verified": SPLITS_TEST_SELECTION,
    "privacy_checks_passed": PRIVACY_TEST_SELECTION,
    "one_command_reproduction_verified": REPRODUCTION_TEST_SELECTION,
}


def _assert_no_self_selection() -> None:
    """Fail loudly at import time if a selection would recurse.

    See "Recursion safety" in the module docstring. This is a structural
    invariant, not a runtime condition, so it is checked once at import
    rather than on every verifier call.
    """
    for item, selection in TEST_SELECTIONS.items():
        overlap = sorted(set(selection) & set(SELF_TEST_MODULES))
        if overlap:
            raise RuntimeError(
                f"checklist verifier test selection for {item!r} names its own "
                f"test module(s) {overlap} -- this would make pytest run a test "
                "that runs pytest. Choose a narrower selection."
            )


_assert_no_self_selection()


# ---------------------------------------------------------------------------
# External check runners (the default, real implementations).
# ---------------------------------------------------------------------------


def run_pytest_selection(
    context: VerificationContext, selection: tuple[str, ...]
) -> CommandOutcome:
    """Run `pytest` over `selection` and classify the result.

    Args:
        context: supplies `repo_root` (the subprocess's working directory, so
            `pyproject.toml`'s `pythonpath = ["src"]` applies) and the
            subprocess timeout.
        selection: repo-root-relative test paths / node ids.

    Returns:
        `CommandOutcome` with `PASSED` when pytest exits 0, `FAILED` when
        pytest reports test failures, and `UNAVAILABLE` when the run could
        not happen or could not be interpreted: a nested invocation, missing
        test files, pytest not installed, an internal/usage error, no tests
        collected, or a timeout.
    """
    if os.environ.get(NESTED_VERIFICATION_ENV_VAR):
        return CommandOutcome(
            CommandStatus.UNAVAILABLE,
            "refused to run a nested pytest invocation "
            f"({NESTED_VERIFICATION_ENV_VAR} is already set)",
        )

    missing = [path for path in selection if not (context.repo_root / path).exists()]
    if missing:
        return CommandOutcome(
            CommandStatus.UNAVAILABLE,
            f"test path(s) not present: {', '.join(missing)}",
        )

    command = [
        sys.executable,
        "-m",
        "pytest",
        *selection,
        "-q",
        "--no-header",
        # Never touch the developer's cache from inside a gate check.
        "-p",
        "no:cacheprovider",
    ]
    environment = dict(os.environ)
    environment[NESTED_VERIFICATION_ENV_VAR] = "1"

    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            cwd=context.repo_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=context.subprocess_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandOutcome(
            CommandStatus.UNAVAILABLE,
            f"pytest {' '.join(selection)} exceeded "
            f"{context.subprocess_timeout_seconds:g}s and was terminated",
        )
    except OSError as exc:
        return CommandOutcome(CommandStatus.UNAVAILABLE, f"could not start pytest: {exc}")

    output = f"{completed.stdout}\n{completed.stderr}"
    if "No module named pytest" in output:
        return CommandOutcome(
            CommandStatus.UNAVAILABLE, "pytest is not installed in this environment"
        )

    # pytest exit codes: 0 = all passed, 1 = tests failed, 2 = interrupted,
    # 3 = internal error, 4 = usage error, 5 = no tests collected.
    if completed.returncode == 0:
        return CommandOutcome(CommandStatus.PASSED, f"pytest {' '.join(selection)} passed")
    if completed.returncode == 1:
        return CommandOutcome(
            CommandStatus.FAILED,
            f"pytest {' '.join(selection)} reported failures: " f"{_last_meaningful_line(output)}",
        )
    return CommandOutcome(
        CommandStatus.UNAVAILABLE,
        f"pytest {' '.join(selection)} exited {completed.returncode} without a "
        f"pass/fail verdict: {_last_meaningful_line(output)}",
    )


def run_export_scan(context: VerificationContext, directory: Path) -> CommandOutcome:
    """Invoke `scripts/scan_exports.py` over `directory`.

    Returns `PASSED` on exit 0 (no violations), `FAILED` on exit 1
    (violations found), and `UNAVAILABLE` when the scanner script or the
    directory is absent, the scan times out, or the process cannot start.
    Violation *details* are deliberately not echoed: `scan_exports` already
    redacts its evidence, and re-printing its output through the gate would
    be a second place for that redaction to go wrong.
    """
    if not context.export_scanner_path.exists():
        scanner_location = _relative_to_repo(context, context.export_scanner_path)
        return CommandOutcome(
            CommandStatus.UNAVAILABLE,
            f"export scanner not present at {scanner_location}",
        )
    if not directory.is_dir():
        return CommandOutcome(
            CommandStatus.UNAVAILABLE,
            f"no export directory to scan at {_relative_to_repo(context, directory)}",
        )

    command = [sys.executable, str(context.export_scanner_path), str(directory)]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            cwd=context.repo_root,
            capture_output=True,
            text=True,
            timeout=context.subprocess_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandOutcome(
            CommandStatus.UNAVAILABLE,
            f"scan_exports over {_relative_to_repo(context, directory)} exceeded "
            f"{context.subprocess_timeout_seconds:g}s and was terminated",
        )
    except OSError as exc:
        return CommandOutcome(CommandStatus.UNAVAILABLE, f"could not start scan_exports: {exc}")

    scanned = _relative_to_repo(context, directory)
    if completed.returncode == 0:
        return CommandOutcome(
            CommandStatus.PASSED, f"scan_exports found no violations under {scanned}"
        )
    if completed.returncode == 1:
        return CommandOutcome(
            CommandStatus.FAILED,
            f"scan_exports reported violation(s) under {scanned} "
            "(details withheld; re-run scripts/scan_exports.py to inspect)",
        )
    return CommandOutcome(
        CommandStatus.UNAVAILABLE,
        f"scan_exports over {scanned} exited {completed.returncode} without a verdict",
    )


@dataclass(frozen=True)
class VerificationContext:
    """Everything a verifier is allowed to look at.

    Every evidence source is a field, so a verifier never reaches for a
    global path: pointing the context at a temporary directory is enough to
    exercise the `failed` and `unverifiable` branches against real files.
    The two runners are fields for the same reason -- they are the only parts
    of a verifier that spawn a process.
    """

    repo_root: Path = REPO_ROOT
    decisions_path: Path = DEFAULT_DECISIONS_PATH
    target_trial_path: Path = DEFAULT_TARGET_TRIAL_PATH
    results_root: Path = DEFAULT_RESULTS_ROOT
    export_scanner_path: Path = DEFAULT_EXPORT_SCANNER
    reproduction_script_path: Path = DEFAULT_REPRODUCTION_SCRIPT
    subprocess_timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    test_runner: TestSelectionRunner = field(default=run_pytest_selection)
    export_scan_runner: ExportScanRunner = field(default=run_export_scan)

    def run_test_selection(self, selection: Sequence[str]) -> CommandOutcome:
        return self.test_runner(self, tuple(selection))

    def scan_export_directory(self, directory: Path) -> CommandOutcome:
        return self.export_scan_runner(self, directory)


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _relative_to_repo(context: VerificationContext, path: Path) -> str:
    """Render `path` relative to the repo root when possible.

    Keeps evidence strings free of machine-specific absolute prefixes (the
    same motivation as `scan_exports.scan_directory`'s relative paths).
    """
    try:
        return str(path.resolve().relative_to(context.repo_root.resolve()))
    except (ValueError, OSError):
        return str(path)


def _last_meaningful_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def _normalize(label: str) -> str:
    """Normalize a human label to a comparable identifier.

    `"race/ethnicity"`, `"Race/Ethnicity"`, and `"race_ethnicity"` all
    normalize to `race_ethnicity`, which is what lets the Section 13 prose
    list be compared name-for-name against the YAML config's snake_case
    entries without transcribing either one.
    """
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", label.strip().lower())).strip("_")


def _read_decisions(context: VerificationContext) -> str | None:
    """Return `decisions.md`'s text, or `None` if it is not present/readable."""
    try:
        return context.decisions_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _section_text(document: str, heading: str) -> str | None:
    """Return the body of the `## `-level section introduced by `heading`."""
    if heading not in document:
        return None
    after = document.split(heading, 1)[1]
    match = re.search(r"^## ", after, flags=re.MULTILINE)
    return after[: match.start()] if match else after


def _load_target_trial_raw(context: VerificationContext) -> tuple[dict | None, str | None]:
    """Load the target-trial YAML.

    Returns:
        `(mapping, None)` on success; `(None, reason)` when the file is
        absent (an `unverifiable` reason) or unparseable/not a mapping (a
        `failed` reason -- the file is there and it is broken).
    """
    if not context.target_trial_path.exists():
        return None, "absent"
    try:
        raw = yaml.safe_load(context.target_trial_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"unparseable: {exc}"
    if not isinstance(raw, dict):
        return None, "top-level content is not a mapping"
    return raw, None


def _verified(item: str, evidence: str) -> VerifierResult:
    return VerifierResult(item, VerifierOutcome.VERIFIED, evidence)


def _failed(item: str, evidence: str) -> VerifierResult:
    return VerifierResult(item, VerifierOutcome.FAILED, evidence)


def _unverifiable(item: str, evidence: str) -> VerifierResult:
    return VerifierResult(item, VerifierOutcome.UNVERIFIABLE, evidence)


def _missing_config_result(item: str, context: VerificationContext, reason: str) -> VerifierResult:
    """Map `_load_target_trial_raw`'s failure reason onto an outcome."""
    location = _relative_to_repo(context, context.target_trial_path)
    if reason == "absent":
        return _unverifiable(item, f"target-trial config not present at {location}")
    return _failed(item, f"target-trial config {location} is {reason}")


def _from_command(item: str, outcome: CommandOutcome) -> VerifierResult:
    """Translate an external check's status into a verifier outcome."""
    if outcome.status is CommandStatus.PASSED:
        return _verified(item, outcome.detail)
    if outcome.status is CommandStatus.FAILED:
        return _failed(item, outcome.detail)
    return _unverifiable(item, outcome.detail)


# ---------------------------------------------------------------------------
# Group 1: document-checkable items.
# ---------------------------------------------------------------------------


def verify_frozen_protocol(context: VerificationContext) -> VerifierResult:
    """`frozen_protocol`: the protocol is frozen and the config encodes it.

    Checks, in order: `decisions.md` carries its frozen-status header and an
    amendment log; Sections 3-7 (the protocol sections) are present; the
    target-trial config validates against `protocols.schema.TargetTrialSpec`
    with a non-empty `version`; and the config's `time_zero.definition`
    matches `decisions.md` Section 5's time-zero row after normalization
    (`randomized_treatment_assignment` vs. "randomized treatment
    assignment"), which is the one protocol value the document states in a
    directly comparable form.
    """
    item = "frozen_protocol"
    document = _read_decisions(context)
    if document is None:
        return _unverifiable(
            item,
            f"decisions.md not present at {_relative_to_repo(context, context.decisions_path)}",
        )

    if FROZEN_STATUS_MARKER not in document:
        return _failed(
            item,
            f"decisions.md carries no {FROZEN_STATUS_MARKER!r} header, so the "
            "protocol is not recorded as frozen",
        )
    if AMENDMENT_LOG_HEADING not in document:
        return _failed(
            item,
            f"decisions.md has no {AMENDMENT_LOG_HEADING!r} section, so a "
            "post-freeze change would have nowhere to be disclosed",
        )

    missing_sections = [heading for heading in PROTOCOL_SECTION_HEADINGS if heading not in document]
    if missing_sections:
        return _failed(item, f"decisions.md is missing section(s): {', '.join(missing_sections)}")

    raw, reason = _load_target_trial_raw(context)
    if raw is None:
        return _missing_config_result(item, context, reason or "unreadable")

    from pydantic import ValidationError  # local import: keeps module import cheap

    from protocols.schema import TargetTrialSpec

    try:
        spec = TargetTrialSpec.model_validate(raw)
    except ValidationError as exc:
        return _failed(
            item,
            f"target-trial config {_relative_to_repo(context, context.target_trial_path)} "
            f"does not validate against TargetTrialSpec: {exc.error_count()} error(s), "
            f"first at {exc.errors()[0]['loc']}",
        )

    if not spec.version.strip():
        return _failed(item, "target-trial config has an empty version string")

    section_5 = _section_text(document, "## 5. Time zero and follow-up") or ""
    if _normalize(spec.time_zero.definition) not in _normalize(section_5):
        return _failed(
            item,
            f"config time_zero.definition {spec.time_zero.definition!r} does not "
            "appear in decisions.md Section 5's time-zero definition",
        )

    return _verified(
        item,
        f"decisions.md is marked frozen with Sections 3-7 and an amendment log; "
        f"target-trial config version {spec.version} validates against "
        f"TargetTrialSpec and its time zero matches Section 5",
    )


def verify_frozen_causal_diagram(context: VerificationContext) -> VerifierResult:
    """`frozen_causal_diagram`: the config's DAG is Section 13's DAG.

    The config's `causal_diagram.nodes`/`edges` must equal the frozen node
    and edge sets, and `decisions.md` Section 13 must still contain the two
    prose arrow lines that the ASCII diagram cannot express structurally
    (`X, A -> S` and `A, X, L -> C`).
    """
    item = "frozen_causal_diagram"
    document = _read_decisions(context)
    if document is None:
        return _unverifiable(
            item,
            f"decisions.md not present at {_relative_to_repo(context, context.decisions_path)}",
        )

    section_13 = _section_text(document, CAUSAL_GRAPH_HEADING)
    if section_13 is None:
        return _failed(
            item, f"decisions.md has no {CAUSAL_GRAPH_HEADING!r} section to check against"
        )

    raw, reason = _load_target_trial_raw(context)
    if raw is None:
        return _missing_config_result(item, context, reason or "unreadable")

    diagram = raw.get("causal_diagram")
    if not isinstance(diagram, dict):
        return _failed(item, "target-trial config has no causal_diagram mapping")

    nodes = diagram.get("nodes")
    edges = diagram.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return _failed(item, "causal_diagram must provide list-valued nodes and edges")

    config_nodes = frozenset(str(node) for node in nodes)
    config_edges = frozenset(str(edge).replace(" ", "") for edge in edges)

    if config_nodes != FROZEN_CAUSAL_DIAGRAM_NODES:
        return _failed(
            item,
            "causal_diagram.nodes "
            f"{sorted(config_nodes)} != decisions.md Section 13's "
            f"{sorted(FROZEN_CAUSAL_DIAGRAM_NODES)}",
        )
    if config_edges != FROZEN_CAUSAL_DIAGRAM_EDGES:
        missing = sorted(FROZEN_CAUSAL_DIAGRAM_EDGES - config_edges)
        extra = sorted(config_edges - FROZEN_CAUSAL_DIAGRAM_EDGES)
        return _failed(
            item,
            "causal_diagram.edges differ from decisions.md Section 13 "
            f"(missing: {missing or 'none'}; unexpected: {extra or 'none'})",
        )

    normalized_section = _normalize(section_13)
    for relation in FROZEN_SECTION_13_RELATIONS:
        if _normalize(relation) not in normalized_section:
            return _failed(
                item,
                f"decisions.md Section 13 no longer states the relation " f"{relation!r}",
            )

    return _verified(
        item,
        f"config causal_diagram matches decisions.md Section 13's "
        f"{len(FROZEN_CAUSAL_DIAGRAM_NODES)} nodes and "
        f"{len(FROZEN_CAUSAL_DIAGRAM_EDGES)} edges, and the section still states "
        "the selection and censoring relations",
    )


def _adjustment_set_from_prose(section_13: str) -> list[str] | None:
    """Parse Section 13's semicolon-separated frozen adjustment-set list."""
    if ADJUSTMENT_SET_PROSE_MARKER not in section_13:
        return None
    after = section_13.split(ADJUSTMENT_SET_PROSE_MARKER, 1)[1]
    # The list is the first non-empty paragraph after the marker, ending at
    # the sentence-terminating period.
    paragraph = next((block.strip() for block in after.split("\n\n") if block.strip()), "")
    if not paragraph:
        return None
    paragraph = paragraph.rstrip().rstrip(".")
    entries = [entry.strip() for entry in paragraph.split(";") if entry.strip()]
    return entries or None


def verify_frozen_adjustment_set(context: VerificationContext) -> VerifierResult:
    """`frozen_adjustment_set`: config `adjustment_set` == Section 13's list.

    Neither list is transcribed into this module. The document's prose list
    is parsed and normalized, the config's list is normalized, and the two
    must match as sets with the same length -- so editing either one alone
    fails, which is the whole point of calling the set frozen.
    """
    item = "frozen_adjustment_set"
    document = _read_decisions(context)
    if document is None:
        return _unverifiable(
            item,
            f"decisions.md not present at {_relative_to_repo(context, context.decisions_path)}",
        )

    section_13 = _section_text(document, CAUSAL_GRAPH_HEADING)
    if section_13 is None:
        return _failed(
            item, f"decisions.md has no {CAUSAL_GRAPH_HEADING!r} section to check against"
        )

    prose_entries = _adjustment_set_from_prose(section_13)
    if prose_entries is None:
        return _failed(
            item,
            "decisions.md Section 13 no longer states a " f"{ADJUSTMENT_SET_PROSE_MARKER!r} list",
        )

    raw, reason = _load_target_trial_raw(context)
    if raw is None:
        return _missing_config_result(item, context, reason or "unreadable")

    config_entries = raw.get("adjustment_set")
    if not isinstance(config_entries, list) or not config_entries:
        return _failed(item, "target-trial config has no non-empty adjustment_set list")

    prose_normalized = {_normalize(entry) for entry in prose_entries}
    config_normalized = {_normalize(str(entry)) for entry in config_entries}

    if prose_normalized != config_normalized:
        only_document = sorted(prose_normalized - config_normalized)
        only_config = sorted(config_normalized - prose_normalized)
        return _failed(
            item,
            "config adjustment_set does not match decisions.md Section 13 "
            f"(only in decisions.md: {only_document or 'none'}; only in config: "
            f"{only_config or 'none'})",
        )
    if len(config_entries) != len(prose_entries):
        return _failed(
            item,
            f"config adjustment_set has {len(config_entries)} entries but "
            f"decisions.md Section 13 lists {len(prose_entries)}",
        )

    return _verified(
        item,
        f"all {len(prose_entries)} config adjustment_set covariates match "
        "decisions.md Section 13's frozen list name-for-name",
    )


def verify_target_population_defined(context: VerificationContext) -> VerifierResult:
    """`target_population_defined`: Section 4 is final and the config encodes it.

    Checks that `decisions.md` Section 4 exists, still carries its five
    numbered inclusion criteria and its no-post-hoc-exclusion prohibition,
    and that the config's population block encodes the same commitments: a
    non-empty criteria list, every criterion referenced by the eligibility
    expression, `post_hoc_outcome_exclusion_permitted: false`, and the
    excluded arms named.
    """
    item = "target_population_defined"
    document = _read_decisions(context)
    if document is None:
        return _unverifiable(
            item,
            f"decisions.md not present at {_relative_to_repo(context, context.decisions_path)}",
        )

    section_4 = _section_text(document, TARGET_POPULATION_HEADING)
    if section_4 is None:
        return _failed(item, f"decisions.md has no {TARGET_POPULATION_HEADING!r} section")

    numbered = re.findall(r"^\s*\d+\.\s+\S", section_4, flags=re.MULTILINE)
    if len(numbered) < 5:
        return _failed(
            item,
            f"decisions.md Section 4 lists {len(numbered)} numbered inclusion "
            "criteria; the frozen definition has five",
        )
    if _normalize(POST_HOC_EXCLUSION_PROHIBITION) not in _normalize(section_4):
        return _failed(
            item,
            "decisions.md Section 4 no longer states the prohibition on post hoc "
            "outcome-based exclusion",
        )

    raw, reason = _load_target_trial_raw(context)
    if raw is None:
        return _missing_config_result(item, context, reason or "unreadable")

    population = raw.get("population")
    if not isinstance(population, dict):
        return _failed(item, "target-trial config has no population mapping")

    criteria = population.get("eligibility_criteria")
    if not isinstance(criteria, list) or not criteria:
        return _failed(item, "population.eligibility_criteria is missing or empty")

    expression = str(population.get("eligibility_expression") or "")
    unreferenced = [
        str(criterion)
        for criterion in criteria
        if _normalize(str(criterion)) not in _normalize(expression)
    ]
    if unreferenced:
        return _failed(
            item,
            "population.eligibility_expression does not reference criteria: "
            f"{', '.join(unreferenced)}",
        )

    if population.get("post_hoc_outcome_exclusion_permitted") is not False:
        return _failed(
            item,
            "population.post_hoc_outcome_exclusion_permitted is not false, "
            "contradicting decisions.md Section 4",
        )

    excluded_arms = population.get("excluded_arms")
    if not isinstance(excluded_arms, list) or not excluded_arms:
        return _failed(
            item,
            "population.excluded_arms is missing or empty; decisions.md Section 3 "
            "excludes arms 2 and 3 before any modeling",
        )
    if not str(population.get("source_population") or "").strip():
        return _failed(item, "population.source_population is missing or empty")

    return _verified(
        item,
        f"decisions.md Section 4 states five inclusion criteria and the post hoc "
        f"exclusion prohibition; config encodes {len(criteria)} criteria, all "
        f"referenced by the eligibility expression, with {len(excluded_arms)} "
        "excluded arm(s) and post hoc outcome exclusion forbidden",
    )


def verify_statistical_procedures_locked(context: VerificationContext) -> VerifierResult:
    """`statistical_procedures_locked`: Section 19's plan matches the code.

    Checks that `decisions.md` Section 19 exists and still names its locked
    procedures, then that the implementing constants agree with it:
    `evaluation.statistics.PRIMARY_COMPARISONS` (exactly the three frozen
    comparisons), `DEFAULT_MIN_CONFIGURATIONS`, and
    `paired_bootstrap_test`'s default replicate count.
    """
    item = "statistical_procedures_locked"
    document = _read_decisions(context)
    if document is None:
        return _unverifiable(
            item,
            f"decisions.md not present at {_relative_to_repo(context, context.decisions_path)}",
        )

    section_19 = _section_text(document, STATISTICAL_PLAN_HEADING)
    if section_19 is None:
        return _failed(item, f"decisions.md has no {STATISTICAL_PLAN_HEADING!r} section")

    missing_markers = [marker for marker in STATISTICAL_PLAN_MARKERS if marker not in section_19]
    if missing_markers:
        return _failed(
            item,
            "decisions.md Section 19 no longer states: "
            f"{', '.join(repr(marker) for marker in missing_markers)}",
        )

    try:
        from evaluation.statistics import (
            DEFAULT_MIN_CONFIGURATIONS,
            PRIMARY_COMPARISONS,
            paired_bootstrap_test,
        )
    except ImportError as exc:
        return _unverifiable(
            item, f"evaluation.statistics is not importable, so nothing to compare: {exc}"
        )

    if frozenset(PRIMARY_COMPARISONS) != FROZEN_PRIMARY_COMPARISONS:
        return _failed(
            item,
            f"evaluation.statistics.PRIMARY_COMPARISONS {sorted(PRIMARY_COMPARISONS)} "
            f"!= decisions.md Section 19's {sorted(FROZEN_PRIMARY_COMPARISONS)}",
        )
    if DEFAULT_MIN_CONFIGURATIONS != FROZEN_MIN_CONFIGURATIONS:
        return _failed(
            item,
            f"DEFAULT_MIN_CONFIGURATIONS is {DEFAULT_MIN_CONFIGURATIONS}, not "
            f"{FROZEN_MIN_CONFIGURATIONS}",
        )

    replicate_default = inspect.signature(paired_bootstrap_test).parameters["n_replicates"].default
    if replicate_default != FROZEN_BOOTSTRAP_REPLICATES:
        return _failed(
            item,
            f"paired_bootstrap_test defaults to {replicate_default} replicates, not "
            f"{FROZEN_BOOTSTRAP_REPLICATES}",
        )

    return _verified(
        item,
        "decisions.md Section 19 states the locked procedures and the "
        "implementing constants agree: three primary comparisons, "
        f"{FROZEN_BOOTSTRAP_REPLICATES} bootstrap replicates, minimum "
        f"{FROZEN_MIN_CONFIGURATIONS} configurations",
    )


# ---------------------------------------------------------------------------
# Group 2: test/scan-checkable items.
# ---------------------------------------------------------------------------


def verify_leakage_tests_passed(context: VerificationContext) -> VerifierResult:
    """`leakage_tests_passed`: run `LEAKAGE_TEST_SELECTION`."""
    return _from_command("leakage_tests_passed", context.run_test_selection(LEAKAGE_TEST_SELECTION))


def verify_splits_cross_fitting_verified(context: VerificationContext) -> VerifierResult:
    """`splits_cross_fitting_verified`: run `SPLITS_TEST_SELECTION`."""
    return _from_command(
        "splits_cross_fitting_verified", context.run_test_selection(SPLITS_TEST_SELECTION)
    )


def verify_privacy_checks_passed(context: VerificationContext) -> VerifierResult:
    """`privacy_checks_passed`: run the privacy tests *and* `scan_exports`.

    Both must succeed for `verified`. Either one failing is `failed` (the
    test failure takes precedence in the evidence string, since a red privacy
    suite invalidates the scan's meaning). A missing export directory leaves
    the item `unverifiable` even when the tests pass: nothing was scanned, so
    "the intended release artifacts passed the scan" has not been shown.
    """
    item = "privacy_checks_passed"
    test_outcome = context.run_test_selection(PRIVACY_TEST_SELECTION)
    if test_outcome.status is CommandStatus.FAILED:
        return _failed(item, test_outcome.detail)

    scan_outcome = context.scan_export_directory(context.results_root)
    if scan_outcome.status is CommandStatus.FAILED:
        return _failed(item, scan_outcome.detail)

    if test_outcome.status is CommandStatus.UNAVAILABLE:
        return _unverifiable(item, test_outcome.detail)
    if scan_outcome.status is CommandStatus.UNAVAILABLE:
        return _unverifiable(item, f"{test_outcome.detail}, but {scan_outcome.detail}")

    return _verified(item, f"{test_outcome.detail}; {scan_outcome.detail}")


def verify_one_command_reproduction_verified(
    context: VerificationContext,
) -> VerifierResult:
    """`one_command_reproduction_verified`: run the end-to-end paper tests.

    The reproduction entry point itself must exist first -- a passing test
    selection would say nothing about reproduction if the documented command
    were gone.
    """
    item = "one_command_reproduction_verified"
    if not context.reproduction_script_path.exists():
        return _unverifiable(
            item,
            "reproduction entry point not present at "
            f"{_relative_to_repo(context, context.reproduction_script_path)}",
        )
    return _from_command(item, context.run_test_selection(REPRODUCTION_TEST_SELECTION))


# ---------------------------------------------------------------------------
# Group 3: attestation-only items (permanently `unverifiable`).
# ---------------------------------------------------------------------------

#: Why each attestation-only item cannot be machine-checked. These strings
#: are the verifier's evidence: the point is that the *absence* of evidence is
#: itself stated, item by item, rather than left implicit.
ATTESTATION_ONLY_REASONS: dict[str, str] = {
    "estimand_alignment_checked": (
        "human attestation: the estimand-alignment report having been read and "
        "accepted by a reviewer leaves no artifact distinguishable from the "
        "report merely having been generated"
    ),
    "reference_inaccessible_to_model_selection": (
        "human attestation: the repository cannot show that nobody consulted the "
        "sealed reference while choosing models -- reference_guard blocks "
        "in-process reads, but not a human reading the artifact out of band"
    ),
    "thresholds_frozen": (
        "human attestation: the frozen threshold *values* are checked by "
        "tests/frozen/test_decisions_match.py, but whether they were fixed "
        "before rather than after observing test-split behavior is an ordering "
        "claim about people, and both worlds leave identical bytes"
    ),
}


def _attestation_only_verifier(item: str) -> Callable[[VerificationContext], VerifierResult]:
    def verifier(context: VerificationContext) -> VerifierResult:  # noqa: ARG001
        return _unverifiable(item, ATTESTATION_ONLY_REASONS[item])

    verifier.__name__ = f"verify_{item}"
    verifier.__doc__ = f"`{item}`: always `unverifiable`. {ATTESTATION_ONLY_REASONS[item]}"
    return verifier


verify_estimand_alignment_checked = _attestation_only_verifier("estimand_alignment_checked")
verify_reference_inaccessible_to_model_selection = _attestation_only_verifier(
    "reference_inaccessible_to_model_selection"
)
verify_thresholds_frozen = _attestation_only_verifier("thresholds_frozen")


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------

#: Every checklist item, in `configs/go_no_go_checklist.yaml` order, mapped to
#: its verifier. `scripts/check_go_no_go.py` owns the authoritative item
#: tuple; `tests/unit/test_checklist_verifiers.py` asserts the two agree, so
#: a thirteenth item (task 20's `neural_readiness_verified`) cannot be added
#: on one side only.
VERIFIERS: dict[str, Callable[[VerificationContext], VerifierResult]] = {
    "frozen_protocol": verify_frozen_protocol,
    "frozen_causal_diagram": verify_frozen_causal_diagram,
    "frozen_adjustment_set": verify_frozen_adjustment_set,
    "estimand_alignment_checked": verify_estimand_alignment_checked,
    "reference_inaccessible_to_model_selection": (verify_reference_inaccessible_to_model_selection),
    "leakage_tests_passed": verify_leakage_tests_passed,
    "splits_cross_fitting_verified": verify_splits_cross_fitting_verified,
    "thresholds_frozen": verify_thresholds_frozen,
    "target_population_defined": verify_target_population_defined,
    "statistical_procedures_locked": verify_statistical_procedures_locked,
    "privacy_checks_passed": verify_privacy_checks_passed,
    "one_command_reproduction_verified": verify_one_command_reproduction_verified,
}

#: All twelve item names, in checklist order.
CHECKLIST_ITEM_NAMES: tuple[str, ...] = tuple(VERIFIERS)

#: Items whose attestation can be contradicted by evidence. `check_go_no_go.py`
#: refuses to mint when one of these is attested `true` and its verifier
#: returns `failed`.
MACHINE_VERIFIABLE_ITEMS: frozenset[str] = frozenset(CHECKLIST_ITEM_NAMES) - frozenset(
    ATTESTATION_ONLY_REASONS
)

#: Items that stay human-set, by design (task 2.5).
ATTESTATION_ONLY_ITEMS: frozenset[str] = frozenset(ATTESTATION_ONLY_REASONS)


def verify_item(item: str, context: VerificationContext | None = None) -> VerifierResult:
    """Run one item's verifier.

    Raises:
        KeyError: if `item` is not a known checklist item.
    """
    if item not in VERIFIERS:
        raise KeyError(f"unknown checklist item: {item!r}")
    return VERIFIERS[item](context if context is not None else VerificationContext())


def verify_all(
    context: VerificationContext | None = None, items: Iterable[str] | None = None
) -> dict[str, VerifierResult]:
    """Run every verifier (or just `items`) and return results by item name.

    Results are returned in `CHECKLIST_ITEM_NAMES` order so the gate's report
    reads in the same order as the checklist config.
    """
    resolved_context = context if context is not None else VerificationContext()
    selected = set(items) if items is not None else set(CHECKLIST_ITEM_NAMES)
    unknown = sorted(selected - set(CHECKLIST_ITEM_NAMES))
    if unknown:
        raise KeyError(f"unknown checklist item(s): {', '.join(unknown)}")
    return {
        name: VERIFIERS[name](resolved_context) for name in CHECKLIST_ITEM_NAMES if name in selected
    }


def contradicted_items(checklist: dict[str, bool], results: dict[str, VerifierResult]) -> list[str]:
    """Return every item attested `true` whose verifier returned `failed`.

    Order follows `CHECKLIST_ITEM_NAMES`. Only `MACHINE_VERIFIABLE_ITEMS` can
    appear: attestation-only verifiers never return `failed`.
    """
    return [
        name
        for name in CHECKLIST_ITEM_NAMES
        if checklist.get(name, False) and name in results and results[name].failed
    ]
