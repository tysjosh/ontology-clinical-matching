"""Unit tests for `scripts/check_go_no_go.py` (task 16; `decisions.md`
Section 21; `design.md` "Correctness Property 1: Sealed reference
isolation").

Confirms the gating logic in isolation:

- a checklist with exactly one item `false` (rest `true`) is reported as
  failed, listing exactly that one unmet item;
- a checklist with every item `false` lists all twelve unmet items;
- a checklist with every item `true` causes `main()` to mint a
  `LockedEvaluationToken` (serialized to `--token-output`) and exit 0;
- a checklist with any item unmet causes `main()` to exit non-zero and
  never mint a token.

Task 2.5 added evidence to the gate (`src/infra/checklist_verifiers.py`), so
this file also covers the attestation-vs-evidence rules:

- an item attested `true` whose verifier returns `failed` blocks minting and
  is named in the refusal (an attestation may not contradict evidence);
- an item whose verifier returns `unverifiable` does not block (absence of
  evidence is not evidence against);
- the report prints every item's attested boolean beside its verifier
  outcome.

`scripts/` is not an installed package, so this test adds it to `sys.path`
directly, mirroring `tests/unit/test_scan_exports.py`.

**Validates: Correctness Property 1; Requirements 13.13, 18.4; decisions.md
Section 21**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_go_no_go  # noqa: E402  (import after sys.path manipulation, by design)

from cohorts.sealed_reference import BOUND_TOKEN_FIELDS  # noqa: E402
from tests.token_fixtures import synthetic_token  # noqa: E402


def _all_true() -> dict[str, bool]:
    return dict.fromkeys(check_go_no_go.CHECKLIST_ITEMS, True)


def _all_false() -> dict[str, bool]:
    return dict.fromkeys(check_go_no_go.CHECKLIST_ITEMS, False)


def _write_checklist(path: Path, items: dict[str, bool]) -> Path:
    checklist_path = path / "checklist.yaml"
    checklist_path.write_text(yaml.safe_dump({"items": items}))
    return checklist_path


def test_check_go_no_go_all_pass_returns_true_and_empty_unmet():
    all_passed, unmet = check_go_no_go.check_go_no_go(_all_true())

    assert all_passed is True
    assert unmet == []


def test_check_go_no_go_single_unmet_item_reports_exactly_that_item():
    items = _all_true()
    items["leakage_tests_passed"] = False

    all_passed, unmet = check_go_no_go.check_go_no_go(items)

    assert all_passed is False
    assert unmet == ["leakage_tests_passed"]


def test_check_go_no_go_all_false_lists_all_twelve_items():
    all_passed, unmet = check_go_no_go.check_go_no_go(_all_false())

    assert all_passed is False
    assert len(check_go_no_go.CHECKLIST_ITEMS) == 12
    assert unmet == list(check_go_no_go.CHECKLIST_ITEMS)


def test_load_checklist_round_trips_the_template_config():
    """The shipped template (`configs/go_no_go_checklist.yaml`) is loadable
    and, unmodified, reports every item unmet (its documented default
    state).
    """
    checklist = check_go_no_go.load_checklist(REPO_ROOT / "configs" / "go_no_go_checklist.yaml")

    assert set(checklist) == set(check_go_no_go.CHECKLIST_ITEMS)
    all_passed, unmet = check_go_no_go.check_go_no_go(checklist)
    assert all_passed is False
    assert len(unmet) == 12


def test_load_checklist_raises_on_missing_item(tmp_path):
    items = _all_true()
    del items["thresholds_frozen"]
    checklist_path = _write_checklist(tmp_path, items)

    with pytest.raises(check_go_no_go.ChecklistError, match="thresholds_frozen"):
        check_go_no_go.load_checklist(checklist_path)


def test_load_checklist_raises_on_non_boolean_value(tmp_path):
    items = _all_true()
    items["thresholds_frozen"] = "yes"
    checklist_path = _write_checklist(tmp_path, items)

    with pytest.raises(check_go_no_go.ChecklistError):
        check_go_no_go.load_checklist(checklist_path)


def test_main_exits_nonzero_and_does_not_mint_when_one_item_unmet(tmp_path, capsys):
    items = _all_true()
    items["privacy_checks_passed"] = False
    checklist_path = _write_checklist(tmp_path, items)
    token_output = tmp_path / "token.json"

    exit_code = check_go_no_go.main(
        ["--checklist", str(checklist_path), "--token-output", str(token_output)]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "privacy_checks_passed" in captured.err
    assert not token_output.exists()


def test_main_mints_and_writes_token_when_all_items_pass(tmp_path, capsys):
    checklist_path = _write_checklist(tmp_path, _all_true())
    token_output = tmp_path / "token.json"

    exit_code = check_go_no_go.main(
        ["--checklist", str(checklist_path), "--token-output", str(token_output)]
    )

    assert exit_code == 0
    assert token_output.exists()

    token_data = json.loads(token_output.read_text())
    assert token_data["issued_by"] == "scripts/check_go_no_go.py"
    assert "issued_at" in token_data


def test_token_artifact_records_every_bound_field_but_they_are_not_wired_yet(tmp_path):
    """The artifact serializes all nine bindings (task 21's runner recomputes
    them per Requirement 13.5), and the seven artifact bindings are still the
    loudly-labeled placeholder task 20 replaces.

    Asserting the placeholder explicitly is deliberate: it makes "these are
    not real hashes yet" a checked fact rather than a comment, and this test
    is the one task 20 must change when it wires the real values in.
    """
    checklist_path = _write_checklist(tmp_path, _all_true())
    token_output = tmp_path / "token.json"

    assert (
        check_go_no_go.main(
            ["--checklist", str(checklist_path), "--token-output", str(token_output)]
        )
        == 0
    )

    token_data = json.loads(token_output.read_text())
    assert set(token_data) == set(BOUND_TOKEN_FIELDS)

    placeholder = check_go_no_go.UNWIRED_BINDING_PLACEHOLDER
    for name in (
        "resolved_config_hash",
        "dataset_checksum",
        "split_hash",
        "git_commit",
        "statistical_plan_version",
        "evaluation_id",
    ):
        assert token_data[name] == placeholder
    assert token_data["checkpoint_hashes"] == [[placeholder, placeholder]]


def test_main_without_token_output_still_succeeds_when_all_pass(tmp_path):
    checklist_path = _write_checklist(tmp_path, _all_true())

    exit_code = check_go_no_go.main(["--checklist", str(checklist_path)])

    assert exit_code == 0


def _target_trial_copy_permitting_post_hoc_exclusion(tmp_path: Path) -> Path:
    """Write a target-trial config that contradicts `target_population_defined`.

    `decisions.md` Section 4 forbids post hoc outcome-based exclusion, and the
    real config encodes that as
    `population.post_hoc_outcome_exclusion_permitted: false`. Flipping it to
    `true` in a copy makes that item's verifier return `failed` against real
    evidence -- no stubbing involved.
    """
    from infra.checklist_verifiers import VerificationContext

    raw = yaml.safe_load(VerificationContext().target_trial_path.read_text(encoding="utf-8"))
    raw["population"]["post_hoc_outcome_exclusion_permitted"] = True
    destination = tmp_path / "contradicting_target_trial.yaml"
    destination.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return destination


def test_main_reports_each_items_attestation_beside_its_verifier_outcome(tmp_path, capsys):
    checklist_path = _write_checklist(tmp_path, _all_true())

    check_go_no_go.main(["--checklist", str(checklist_path)])

    captured = capsys.readouterr()
    for name in check_go_no_go.CHECKLIST_ITEMS:
        assert name in captured.out
    assert "attested=true" in captured.out
    assert "verifier=verified" in captured.out
    # The three process attestations report their absence of evidence.
    assert captured.out.count("verifier=unverifiable") == 3


def test_main_refuses_to_mint_when_an_attestation_contradicts_evidence(tmp_path, capsys):
    """Attested `true`, verifier `failed` -> no token, item named, exit non-zero."""
    checklist_path = _write_checklist(tmp_path, _all_true())
    token_output = tmp_path / "token.json"

    exit_code = check_go_no_go.main(
        [
            "--checklist",
            str(checklist_path),
            "--token-output",
            str(token_output),
            "--target-trial-path",
            str(_target_trial_copy_permitting_post_hoc_exclusion(tmp_path)),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert not token_output.exists()
    assert "target_population_defined" in captured.err
    assert "attested true but its verifier returned failed" in captured.err
    assert "post_hoc_outcome_exclusion_permitted" in captured.err


def test_main_still_mints_when_evidence_is_merely_unverifiable(tmp_path):
    """Absence of evidence is not evidence against.

    With `decisions.md` unavailable, the five document-checkable verifiers
    return `unverifiable` rather than `failed`, so the attestations remain the
    only signal and the gate mints as before.
    """
    checklist_path = _write_checklist(tmp_path, _all_true())
    token_output = tmp_path / "token.json"

    exit_code = check_go_no_go.main(
        [
            "--checklist",
            str(checklist_path),
            "--token-output",
            str(token_output),
            "--decisions-path",
            str(tmp_path / "absent_decisions.md"),
        ]
    )

    assert exit_code == 0
    assert token_output.exists()


def test_minted_token_unlocks_a_sealed_reference(tmp_path):
    """End-to-end confirmation that the token minted by this script's gating
    path is a genuine `LockedEvaluationToken` usable by
    `SealedReference.unseal()` (`infra.reference_guard.assert_reference_unlocked`).
    """
    from datetime import datetime

    from cohorts.cohort_construction import PatientRecord, RandomizedCohort
    from cohorts.sealed_reference import SealedReference, placeholder_estimator
    from data.event_model import ClinicalEvent, MissingnessState
    from data.timelines import build_patient_timeline
    from protocols.eligibility_dsl import EligibilityResult
    from protocols.endpoints import EndpointResult

    checklist_path = _write_checklist(tmp_path, _all_true())
    check_go_no_go.main(["--checklist", str(checklist_path)])

    token = synthetic_token(issued_by="scripts/check_go_no_go.py")

    event = ClinicalEvent(
        patient_id="p1",
        encounter_id=None,
        concept_id="baseline_cd4",
        value=350.0,
        unit="cells/mm3",
        modality="baseline",
        observed_at=datetime(2020, 1, 15, 9, 0, 0),
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    timeline = build_patient_timeline([event])
    cohort = RandomizedCohort(
        patients=(
            PatientRecord(
                patient_id="p1",
                treatment_arm=0,
                timeline=timeline,
                eligibility_result=EligibilityResult(eligible=True, reason="", missing_fields=[]),
                endpoint_result=EndpointResult(
                    event_time=datetime(2020, 1, 15, 9, 0, 0),
                    event_indicator=False,
                    censoring_indicator=True,
                    event_type=None,
                ),
            ),
        )
    )
    sealed = SealedReference(cohort, estimator=placeholder_estimator)

    internal_reference = sealed.unseal(token)

    assert internal_reference.cohort_size == 1
