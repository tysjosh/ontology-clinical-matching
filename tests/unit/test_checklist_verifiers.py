"""Unit tests for `src/infra/checklist_verifiers.py` (task 2.5; Requirements
13.13, 18.4).

Every verifier is exercised for all three outcomes it can produce:

- `verified` against the real repository (`decisions.md`,
  `configs/target_trials/main.yaml`, the real test selections);
- `failed` against a *mutated copy* of the real evidence -- a temporary
  `decisions.md` with its frozen-status header stripped, a temporary
  target-trial config with an adjustment-set covariate dropped, and so on --
  so the failure path is driven by real parsing of real files rather than by
  a stubbed return value;
- `unverifiable` by pointing the context at an evidence source that does not
  exist, which is the boundary the module documents (missing evidence source
  = `unverifiable`; present-but-contradicting evidence = `failed`).

The four test/scan-invoking verifiers additionally get a recording runner
substituted into the context (`VerificationContext.test_runner` is an
injection point precisely for this) so their three outcomes and their
documented selections can be checked without spawning twelve pytest
subprocesses. The default runners themselves are then tested for real:
`run_pytest_selection` against a genuinely passing and a genuinely failing
throwaway test file, and `run_export_scan` against a clean directory and one
holding a fake credential.

**Validates: Requirements 13.13, 18.4**
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import yaml

from infra.checklist_verifiers import (
    ATTESTATION_ONLY_ITEMS,
    CHECKLIST_ITEM_NAMES,
    LEAKAGE_TEST_SELECTION,
    MACHINE_VERIFIABLE_ITEMS,
    NESTED_VERIFICATION_ENV_VAR,
    PRIVACY_TEST_SELECTION,
    REPRODUCTION_TEST_SELECTION,
    SELF_TEST_MODULES,
    SPLITS_TEST_SELECTION,
    TEST_SELECTIONS,
    CommandOutcome,
    CommandStatus,
    VerificationContext,
    VerifierOutcome,
    contradicted_items,
    run_export_scan,
    run_pytest_selection,
    verify_all,
    verify_frozen_adjustment_set,
    verify_frozen_causal_diagram,
    verify_frozen_protocol,
    verify_item,
    verify_leakage_tests_passed,
    verify_one_command_reproduction_verified,
    verify_privacy_checks_passed,
    verify_splits_cross_fitting_verified,
    verify_statistical_procedures_locked,
    verify_target_population_defined,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_go_no_go  # noqa: E402  (import after sys.path manipulation, by design)

#: A fake AWS-access-key-shaped secret -- not a real credential. Same shape as
#: `tests/unit/test_scan_exports.py` uses.
FAKE_AWS_KEY = "AKIAABCDEFGHIJKLMNOP"


# ---------------------------------------------------------------------------
# Helpers: real evidence, and mutated copies of it.
# ---------------------------------------------------------------------------


def _real_context() -> VerificationContext:
    """A context pointed at the actual repository evidence."""
    return VerificationContext()


def _copy_decisions(tmp_path: Path, *, remove: str | None = None) -> Path:
    """Copy the real `decisions.md`, optionally deleting one line of it."""
    text = VerificationContext().decisions_path.read_text(encoding="utf-8")
    if remove is not None:
        assert remove in text, f"{remove!r} not present in decisions.md to remove"
        text = text.replace(remove, "")
    destination = tmp_path / "decisions.md"
    destination.write_text(text, encoding="utf-8")
    return destination


def _copy_target_trial(tmp_path: Path, mutate=None) -> Path:
    """Copy the real target-trial config, optionally mutating the mapping."""
    raw = yaml.safe_load(VerificationContext().target_trial_path.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(raw)
    destination = tmp_path / "main.yaml"
    destination.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return destination


def _recording_runner(status: CommandStatus, detail: str = "stub"):
    """A test runner that records its selection and returns a fixed status."""
    seen: list[tuple[str, ...]] = []

    def runner(context: VerificationContext, selection: tuple[str, ...]) -> CommandOutcome:
        seen.append(selection)
        return CommandOutcome(status, detail)

    return runner, seen


def _context_with_test_runner(status: CommandStatus, **overrides):
    runner, seen = _recording_runner(status)
    return dataclasses.replace(_real_context(), test_runner=runner, **overrides), seen


# ---------------------------------------------------------------------------
# Registry and structural invariants.
# ---------------------------------------------------------------------------


def test_registry_covers_exactly_the_twelve_gate_items_in_the_same_order():
    """The verifier registry and the gate's item tuple cannot drift apart.

    Task 20 adds a thirteenth item (`neural_readiness_verified`); this
    assertion is what makes adding it on one side only a test failure.
    """
    assert CHECKLIST_ITEM_NAMES == check_go_no_go.CHECKLIST_ITEMS
    assert len(CHECKLIST_ITEM_NAMES) == 12


def test_machine_verifiable_and_attestation_only_items_partition_the_checklist():
    assert MACHINE_VERIFIABLE_ITEMS | ATTESTATION_ONLY_ITEMS == set(CHECKLIST_ITEM_NAMES)
    assert not (MACHINE_VERIFIABLE_ITEMS & ATTESTATION_ONLY_ITEMS)
    assert ATTESTATION_ONLY_ITEMS == {
        "estimand_alignment_checked",
        "reference_inaccessible_to_model_selection",
        "thresholds_frozen",
    }
    assert len(MACHINE_VERIFIABLE_ITEMS) == 9


def test_no_test_selection_names_the_verifier_or_gate_test_modules():
    """Recursion guard: a selection naming this file would make pytest run a
    test that runs pytest that runs the test, without bound.
    """
    for item, selection in TEST_SELECTIONS.items():
        assert not set(selection) & set(SELF_TEST_MODULES), item


def test_every_selected_test_path_exists_in_the_repository():
    for item, selection in TEST_SELECTIONS.items():
        for path in selection:
            assert (REPO_ROOT / path).exists(), f"{item}: {path}"


def test_verify_item_rejects_an_unknown_item():
    with pytest.raises(KeyError):
        verify_item("neural_readiness_verified")


# ---------------------------------------------------------------------------
# Group 1: document-checkable verifiers.
# ---------------------------------------------------------------------------


def test_frozen_protocol_verified_against_the_real_repository():
    result = verify_frozen_protocol(_real_context())

    assert result.outcome is VerifierOutcome.VERIFIED
    assert "TargetTrialSpec" in result.evidence


def test_frozen_protocol_failed_when_the_frozen_status_header_is_gone(tmp_path):
    context = dataclasses.replace(
        _real_context(),
        decisions_path=_copy_decisions(tmp_path, remove="**Status:** Frozen"),
    )

    result = verify_frozen_protocol(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "frozen" in result.evidence.lower()


def test_frozen_protocol_failed_when_the_config_no_longer_validates(tmp_path):
    def drop_estimand(raw: dict) -> None:
        del raw["estimand"]

    context = dataclasses.replace(
        _real_context(), target_trial_path=_copy_target_trial(tmp_path, drop_estimand)
    )

    result = verify_frozen_protocol(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "TargetTrialSpec" in result.evidence


def test_frozen_protocol_unverifiable_when_decisions_is_absent(tmp_path):
    context = dataclasses.replace(_real_context(), decisions_path=tmp_path / "no_decisions.md")

    result = verify_frozen_protocol(context)

    assert result.outcome is VerifierOutcome.UNVERIFIABLE
    assert "not present" in result.evidence


def test_frozen_causal_diagram_verified_against_the_real_repository():
    result = verify_frozen_causal_diagram(_real_context())

    assert result.outcome is VerifierOutcome.VERIFIED


def test_frozen_causal_diagram_failed_when_an_edge_is_dropped(tmp_path):
    def drop_edge(raw: dict) -> None:
        raw["causal_diagram"]["edges"].remove("L->C")

    context = dataclasses.replace(
        _real_context(), target_trial_path=_copy_target_trial(tmp_path, drop_edge)
    )

    result = verify_frozen_causal_diagram(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "L->C" in result.evidence


def test_frozen_causal_diagram_failed_when_decisions_drops_the_censoring_relation(
    tmp_path,
):
    """Section 13's ASCII diagram draws only the X/A/L/Y core; the selection
    and censoring relations live in prose beside it, so they are checked as
    prose.
    """
    context = dataclasses.replace(
        _real_context(),
        decisions_path=_copy_decisions(tmp_path, remove="A, X, L \u2500\u2500\u2192 Censoring C"),
    )

    result = verify_frozen_causal_diagram(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "Censoring C" in result.evidence


def test_frozen_causal_diagram_unverifiable_when_decisions_is_absent(tmp_path):
    context = dataclasses.replace(_real_context(), decisions_path=tmp_path / "no_decisions.md")

    assert verify_frozen_causal_diagram(context).outcome is VerifierOutcome.UNVERIFIABLE


def test_frozen_adjustment_set_verified_against_the_real_repository():
    """The config's thirteen covariates match `decisions.md` Section 13's
    prose list name-for-name, with neither list transcribed into the
    verifier.
    """
    result = verify_frozen_adjustment_set(_real_context())

    assert result.outcome is VerifierOutcome.VERIFIED
    assert "13" in result.evidence


def test_frozen_adjustment_set_failed_when_a_covariate_is_dropped(tmp_path):
    def drop_covariate(raw: dict) -> None:
        raw["adjustment_set"].remove("baseline_cd8")

    context = dataclasses.replace(
        _real_context(), target_trial_path=_copy_target_trial(tmp_path, drop_covariate)
    )

    result = verify_frozen_adjustment_set(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "baseline_cd8" in result.evidence


def test_frozen_adjustment_set_failed_when_a_covariate_is_added(tmp_path):
    """Symmetry matters: an *extra* covariate in the config is as much a
    break of the frozen set as a missing one.
    """

    def add_covariate(raw: dict) -> None:
        raw["adjustment_set"].append("week_20_cd4")

    context = dataclasses.replace(
        _real_context(), target_trial_path=_copy_target_trial(tmp_path, add_covariate)
    )

    result = verify_frozen_adjustment_set(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "week_20_cd4" in result.evidence


def test_frozen_adjustment_set_unverifiable_when_decisions_is_absent(tmp_path):
    context = dataclasses.replace(_real_context(), decisions_path=tmp_path / "no_decisions.md")

    assert verify_frozen_adjustment_set(context).outcome is VerifierOutcome.UNVERIFIABLE


def test_target_population_defined_verified_against_the_real_repository():
    result = verify_target_population_defined(_real_context())

    assert result.outcome is VerifierOutcome.VERIFIED


def test_target_population_defined_failed_when_post_hoc_exclusion_is_permitted(tmp_path):
    def permit_post_hoc(raw: dict) -> None:
        raw["population"]["post_hoc_outcome_exclusion_permitted"] = True

    context = dataclasses.replace(
        _real_context(), target_trial_path=_copy_target_trial(tmp_path, permit_post_hoc)
    )

    result = verify_target_population_defined(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "post_hoc_outcome_exclusion_permitted" in result.evidence


def test_target_population_defined_failed_when_the_prohibition_leaves_decisions(tmp_path):
    context = dataclasses.replace(
        _real_context(),
        decisions_path=_copy_decisions(
            tmp_path,
            remove=("No post hoc exclusion based on observed outcomes is permitted at any stage."),
        ),
    )

    result = verify_target_population_defined(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "post hoc" in result.evidence


def test_target_population_defined_unverifiable_when_decisions_is_absent(tmp_path):
    context = dataclasses.replace(_real_context(), decisions_path=tmp_path / "no_decisions.md")

    assert verify_target_population_defined(context).outcome is VerifierOutcome.UNVERIFIABLE


def test_statistical_procedures_locked_verified_against_the_real_repository():
    result = verify_statistical_procedures_locked(_real_context())

    assert result.outcome is VerifierOutcome.VERIFIED
    assert "1000 bootstrap replicates" in result.evidence


def test_statistical_procedures_locked_failed_when_holm_correction_leaves_decisions(
    tmp_path,
):
    context = dataclasses.replace(
        _real_context(), decisions_path=_copy_decisions(tmp_path, remove="Holm correction")
    )

    result = verify_statistical_procedures_locked(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "Holm correction" in result.evidence


def test_statistical_procedures_locked_unverifiable_when_decisions_is_absent(tmp_path):
    context = dataclasses.replace(_real_context(), decisions_path=tmp_path / "no_decisions.md")

    assert verify_statistical_procedures_locked(context).outcome is VerifierOutcome.UNVERIFIABLE


# ---------------------------------------------------------------------------
# Group 2: test/scan-invoking verifiers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (CommandStatus.PASSED, VerifierOutcome.VERIFIED),
        (CommandStatus.FAILED, VerifierOutcome.FAILED),
        (CommandStatus.UNAVAILABLE, VerifierOutcome.UNVERIFIABLE),
    ],
)
def test_leakage_verifier_maps_each_test_status_and_runs_its_selection(status, expected):
    context, seen = _context_with_test_runner(status)

    result = verify_leakage_tests_passed(context)

    assert result.outcome is expected
    assert seen == [LEAKAGE_TEST_SELECTION]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (CommandStatus.PASSED, VerifierOutcome.VERIFIED),
        (CommandStatus.FAILED, VerifierOutcome.FAILED),
        (CommandStatus.UNAVAILABLE, VerifierOutcome.UNVERIFIABLE),
    ],
)
def test_splits_verifier_maps_each_test_status_and_runs_its_selection(status, expected):
    context, seen = _context_with_test_runner(status)

    result = verify_splits_cross_fitting_verified(context)

    assert result.outcome is expected
    assert seen == [SPLITS_TEST_SELECTION]


def test_privacy_verifier_verified_when_tests_pass_and_the_scan_is_clean(tmp_path):
    (tmp_path / "results.json").write_text('{"cohort": "arm0", "concordance": 0.71}')
    context, seen = _context_with_test_runner(CommandStatus.PASSED, results_root=tmp_path)

    result = verify_privacy_checks_passed(context)

    assert result.outcome is VerifierOutcome.VERIFIED
    assert seen == [PRIVACY_TEST_SELECTION]
    assert "no violations" in result.evidence


def test_privacy_verifier_failed_when_the_export_scan_finds_a_secret(tmp_path):
    (tmp_path / "config.yaml").write_text(f"aws_key: {FAKE_AWS_KEY}\n")
    context, _ = _context_with_test_runner(CommandStatus.PASSED, results_root=tmp_path)

    result = verify_privacy_checks_passed(context)

    assert result.outcome is VerifierOutcome.FAILED
    assert "violation" in result.evidence


def test_privacy_verifier_failed_when_the_privacy_tests_fail(tmp_path):
    (tmp_path / "results.json").write_text("{}")
    context, _ = _context_with_test_runner(CommandStatus.FAILED, results_root=tmp_path)

    assert verify_privacy_checks_passed(context).outcome is VerifierOutcome.FAILED


def test_privacy_verifier_unverifiable_when_there_is_no_export_directory(tmp_path):
    """Passing tests alone do not show that the intended release artifacts
    passed the scan -- nothing was scanned.
    """
    context, _ = _context_with_test_runner(CommandStatus.PASSED, results_root=tmp_path / "absent")

    result = verify_privacy_checks_passed(context)

    assert result.outcome is VerifierOutcome.UNVERIFIABLE
    assert "no export directory" in result.evidence


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (CommandStatus.PASSED, VerifierOutcome.VERIFIED),
        (CommandStatus.FAILED, VerifierOutcome.FAILED),
    ],
)
def test_reproduction_verifier_maps_each_test_status_and_runs_its_selection(status, expected):
    context, seen = _context_with_test_runner(status)

    result = verify_one_command_reproduction_verified(context)

    assert result.outcome is expected
    assert seen == [REPRODUCTION_TEST_SELECTION]


def test_reproduction_verifier_unverifiable_without_the_documented_entry_point(tmp_path):
    context, seen = _context_with_test_runner(
        CommandStatus.PASSED, reproduction_script_path=tmp_path / "run_paper.py"
    )

    result = verify_one_command_reproduction_verified(context)

    assert result.outcome is VerifierOutcome.UNVERIFIABLE
    assert seen == []


# ---------------------------------------------------------------------------
# Group 3: attestation-only verifiers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("item", sorted(ATTESTATION_ONLY_ITEMS))
def test_attestation_only_items_are_always_unverifiable_with_a_stated_reason(item):
    """These three stay human-set by design (task 2.5). Their evidence string
    is the *reason* no machine check exists, so the gap is documented at the
    gate rather than implied.
    """
    result = verify_item(item, _real_context())

    assert result.outcome is VerifierOutcome.UNVERIFIABLE
    assert "human attestation" in result.evidence


def test_attestation_only_verifiers_cannot_block_minting():
    """`contradicted_items()` can only ever name machine-verifiable items."""
    results = verify_all(_real_context(), items=ATTESTATION_ONLY_ITEMS)
    checklist = dict.fromkeys(CHECKLIST_ITEM_NAMES, True)

    assert contradicted_items(checklist, results) == []


# ---------------------------------------------------------------------------
# The default runners, tested for real.
# ---------------------------------------------------------------------------


def test_run_pytest_selection_passes_on_a_genuinely_passing_selection(tmp_path):
    (tmp_path / "test_green.py").write_text("def test_green():\n    assert True\n")
    context = dataclasses.replace(_real_context(), repo_root=tmp_path)

    outcome = run_pytest_selection(context, ("test_green.py",))

    assert outcome.status is CommandStatus.PASSED


def test_run_pytest_selection_fails_on_a_genuinely_failing_selection(tmp_path):
    (tmp_path / "test_red.py").write_text("def test_red():\n    assert False\n")
    context = dataclasses.replace(_real_context(), repo_root=tmp_path)

    outcome = run_pytest_selection(context, ("test_red.py",))

    assert outcome.status is CommandStatus.FAILED
    assert "test_red.py" in outcome.detail


def test_run_pytest_selection_unavailable_when_the_selection_does_not_exist(tmp_path):
    context = dataclasses.replace(_real_context(), repo_root=tmp_path)

    outcome = run_pytest_selection(context, ("tests/unit/test_absent.py",))

    assert outcome.status is CommandStatus.UNAVAILABLE
    assert "not present" in outcome.detail


def test_run_pytest_selection_refuses_a_nested_invocation(tmp_path, monkeypatch):
    """Second recursion guard: even a self-referential selection stops at
    depth one, because the flag the outer subprocess sets is already in the
    environment.
    """
    (tmp_path / "test_green.py").write_text("def test_green():\n    assert True\n")
    monkeypatch.setenv(NESTED_VERIFICATION_ENV_VAR, "1")
    context = dataclasses.replace(_real_context(), repo_root=tmp_path)

    outcome = run_pytest_selection(context, ("test_green.py",))

    assert outcome.status is CommandStatus.UNAVAILABLE
    assert "nested" in outcome.detail


def test_run_pytest_selection_unavailable_on_timeout(tmp_path):
    (tmp_path / "test_slow.py").write_text(
        "import time\n\n\ndef test_slow():\n    time.sleep(30)\n"
    )
    context = dataclasses.replace(
        _real_context(), repo_root=tmp_path, subprocess_timeout_seconds=2.0
    )

    outcome = run_pytest_selection(context, ("test_slow.py",))

    assert outcome.status is CommandStatus.UNAVAILABLE
    assert "exceeded" in outcome.detail


def test_run_export_scan_passes_on_a_clean_directory(tmp_path):
    (tmp_path / "tables.json").write_text('{"concordance": 0.71}')

    outcome = run_export_scan(_real_context(), tmp_path)

    assert outcome.status is CommandStatus.PASSED


def test_run_export_scan_fails_on_a_directory_holding_a_fake_credential(tmp_path):
    (tmp_path / "config.yaml").write_text(f"aws_key: {FAKE_AWS_KEY}\n")

    outcome = run_export_scan(_real_context(), tmp_path)

    assert outcome.status is CommandStatus.FAILED
    # The scanner's own redacted evidence is not re-echoed through the gate.
    assert FAKE_AWS_KEY not in outcome.detail


def test_run_export_scan_unavailable_without_a_directory(tmp_path):
    outcome = run_export_scan(_real_context(), tmp_path / "absent")

    assert outcome.status is CommandStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# verify_all over the real repository.
# ---------------------------------------------------------------------------


def test_verify_all_covers_every_item_and_currently_contradicts_nothing():
    """The repository's own evidence must not contradict any item: today
    every machine-verifiable item verifies and the three attestation-only
    items report `unverifiable`.

    This doubles as the guard that keeps the gate usable -- if a verifier ever
    starts returning `failed` on a clean checkout, `check_go_no_go.py` would
    refuse to mint no matter what a reviewer attests, and that would show up
    here first.
    """
    results = verify_all(_real_context())

    assert set(results) == set(CHECKLIST_ITEM_NAMES)
    assert [name for name, result in results.items() if result.failed] == []
    for name in MACHINE_VERIFIABLE_ITEMS:
        assert results[name].outcome is VerifierOutcome.VERIFIED, name
    for name in ATTESTATION_ONLY_ITEMS:
        assert results[name].outcome is VerifierOutcome.UNVERIFIABLE, name
