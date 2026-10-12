"""Unit tests for `scripts/run_locked_evaluation.py` (task 21; Requirement
13.1, 13.2, 13.5, 13.6, 13.10).

Covers the pieces of the Locked_Evaluation_Runner that do not require a
real token/sealed-reference/checkpoint fixture: the rejected-option check,
the import-graph exclusion (Correctness Property 16), the consumed-id
refusal, the hash-mismatch refusal, and the occupied-path refusal -- each
confirmed to leave the results tree byte-identical.

`scripts/` is not an installed package, so this test adds it to
`sys.path` directly, mirroring `tests/unit/test_scan_exports.py`.

**Validates: Requirements 13.1, 13.2, 13.5, 13.6, 13.10**
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_locked_evaluation  # noqa: E402  (import after sys.path manipulation, by design)

from cohorts.sealed_reference import LockedEvaluationToken  # noqa: E402

# ---------------------------------------------------------------------------
# Requirement 13.2: rejected invocation options.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("option", run_locked_evaluation.REJECTED_OPTION_NAMES)
def test_check_for_rejected_options_raises_for_every_named_option(option):
    with pytest.raises(run_locked_evaluation.RejectedOptionError, match=option):
        run_locked_evaluation.check_for_rejected_options([option, "value"])


def test_check_for_rejected_options_catches_the_equals_form():
    with pytest.raises(run_locked_evaluation.RejectedOptionError, match="--learning-rate"):
        run_locked_evaluation.check_for_rejected_options(["--learning-rate=0.01"])


def test_check_for_rejected_options_accepts_a_clean_argv():
    run_locked_evaluation.check_for_rejected_options(
        ["--token-artifact", "x.json", "--resolved-config-hash", "abc"]
    )  # must not raise


def test_main_refuses_a_rejected_option_before_opening_anything(tmp_path, capsys):
    exit_code = run_locked_evaluation.main(
        ["--learning-rate", "0.01", "--token-artifact", str(tmp_path / "does_not_exist.json")]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "--learning-rate" in captured.err
    # The nonexistent token artifact was never even checked for existence,
    # confirming the rejection happened first.
    assert "token artifact not found" not in captured.err


# ---------------------------------------------------------------------------
# Correctness Property 16 / Requirement 13.1: import graph excludes every
# fitting entry point.
# ---------------------------------------------------------------------------

#: Names this module's import graph must never reach, per `design.md`
#: Section 3.12.
_FORBIDDEN_IMPORT_NAMES: tuple[str, ...] = (
    "select_lambdas",
    "train_one_model",
    "train_ensemble",
)


def test_module_source_imports_none_of_the_fitting_entry_points():
    """AST-based (not a live import graph walk, which would require
    executing every transitively-imported module): confirms this
    module's own `import`/`from ... import` statements never name a
    fitting entry point, directly or via an alias."""
    source = Path(run_locked_evaluation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)

    forbidden_found = imported_names & set(_FORBIDDEN_IMPORT_NAMES)
    assert forbidden_found == set()


def test_module_does_not_import_calibrator_fit_or_torch_training_module_wholesale():
    """`from models.torch_training import *` or `import models.torch_training`
    would make every fitting entry point reachable through this module's
    own namespace even without naming one directly; confirm neither
    import form appears."""
    source = Path(run_locked_evaluation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "models.torch_training":
            names = {alias.name for alias in node.names}
            assert "*" not in names
            assert names.isdisjoint(_FORBIDDEN_IMPORT_NAMES)
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "models.torch_training"


def test_module_imports_no_calibrator_and_calls_no_fit_method():
    """Checks real `import`/`from...import` statements and real method-call
    sites via AST, not the docstring text -- the module docstring itself
    discusses `Calibrator.fit` by name as part of explaining what this
    module must NOT reach, which a plain substring check would wrongly
    flag."""
    source = Path(run_locked_evaluation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
    assert "Calibrator" not in imported_names

    fit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]
    assert fit_calls == []


# ---------------------------------------------------------------------------
# Token artifact loading.
# ---------------------------------------------------------------------------


def _write_token_artifact(path: Path, **overrides) -> Path:
    payload = {
        "issued_by": "scripts/check_go_no_go.py",
        "issued_at": datetime(2026, 9, 6, 12, 0, 0).isoformat(),
        "resolved_config_hash": "resolved-config-hash-abc",
        "dataset_checksum": "dataset-checksum-abc",
        "split_hash": "split-hash-abc",
        "checkpoint_hashes": [["intervention_aware", "checkpoint-sha256-abc"]],
        "git_commit": "git-commit-abc",
        "statistical_plan_version": "plan-version-abc",
        "evaluation_id": "eval-abc-123",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_load_token_artifact_reconstructs_a_real_token(tmp_path):
    path = _write_token_artifact(tmp_path / "token.json")

    token = run_locked_evaluation.load_token_artifact(path)

    assert isinstance(token, LockedEvaluationToken)
    assert token.evaluation_id == "eval-abc-123"
    assert token.checkpoint_hashes == (("intervention_aware", "checkpoint-sha256-abc"),)


def test_load_token_artifact_raises_on_missing_file(tmp_path):
    with pytest.raises(run_locked_evaluation.TokenArtifactError, match="not found"):
        run_locked_evaluation.load_token_artifact(tmp_path / "absent.json")


def test_load_token_artifact_raises_on_invalid_json(tmp_path):
    path = tmp_path / "token.json"
    path.write_text("not valid json {")

    with pytest.raises(run_locked_evaluation.TokenArtifactError, match="not valid JSON"):
        run_locked_evaluation.load_token_artifact(path)


def test_load_token_artifact_raises_on_missing_key(tmp_path):
    path = tmp_path / "token.json"
    payload = {
        "issued_by": "x",
        "issued_at": datetime(2026, 1, 1).isoformat(),
        # resolved_config_hash intentionally absent
        "dataset_checksum": "x",
        "split_hash": "x",
        "checkpoint_hashes": [["a", "b"]],
        "git_commit": "x",
        "statistical_plan_version": "x",
        "evaluation_id": "x",
    }
    path.write_text(json.dumps(payload))

    with pytest.raises(run_locked_evaluation.TokenArtifactError, match="resolved_config_hash"):
        run_locked_evaluation.load_token_artifact(path)


def test_load_token_artifact_raises_on_an_empty_field(tmp_path):
    path = _write_token_artifact(tmp_path / "token.json", split_hash="")

    with pytest.raises(run_locked_evaluation.TokenArtifactError):
        run_locked_evaluation.load_token_artifact(path)


# ---------------------------------------------------------------------------
# Requirement 13.6: consumed-id refusal.
# ---------------------------------------------------------------------------


def test_check_not_consumed_passes_for_a_fresh_evaluation_id(tmp_path):
    path = _write_token_artifact(tmp_path / "token.json")
    token = run_locked_evaluation.load_token_artifact(path)

    run_locked_evaluation.check_not_consumed(token, tmp_path / "results")  # must not raise


def test_check_not_consumed_raises_when_already_consumed(tmp_path):
    from infra.consumption_ledger import append_consumed

    results_root = tmp_path / "results"
    path = _write_token_artifact(tmp_path / "token.json")
    token = run_locked_evaluation.load_token_artifact(path)
    append_consumed(token.evaluation_id, results_root, result_directory="somewhere")

    with pytest.raises(run_locked_evaluation.AlreadyConsumedError, match=token.evaluation_id):
        run_locked_evaluation.check_not_consumed(token, results_root)


def test_main_refuses_a_consumed_evaluation_id_leaving_the_tree_byte_identical(tmp_path, capsys):
    from infra.consumption_ledger import append_consumed, consumed_ledger_path

    results_root = tmp_path / "results"
    token_path = _write_token_artifact(tmp_path / "token.json")
    token = run_locked_evaluation.load_token_artifact(token_path)
    append_consumed(token.evaluation_id, results_root, result_directory="somewhere")
    ledger_path = consumed_ledger_path(results_root)
    before = ledger_path.read_bytes()

    exit_code = run_locked_evaluation.main(
        [
            "--token-artifact",
            str(token_path),
            "--results-root",
            str(results_root),
            "--resolved-config-hash",
            "resolved-config-hash-abc",
            "--dataset-checksum",
            "dataset-checksum-abc",
            "--split-hash",
            "split-hash-abc",
            "--checkpoint",
            "intervention_aware=/does/not/matter",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert token.evaluation_id in captured.err
    assert ledger_path.read_bytes() == before


# ---------------------------------------------------------------------------
# Requirement 13.5: hash mismatch, first offender named.
# ---------------------------------------------------------------------------


def _write_checkpoint_file(path: Path, content: bytes = b"checkpoint-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_verify_bound_hashes_passes_when_everything_matches(tmp_path):
    import hashlib

    checkpoint_path = _write_checkpoint_file(tmp_path / "ckpt.pt")
    real_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    token_path = _write_token_artifact(
        tmp_path / "token.json", checkpoint_hashes=[["intervention_aware", real_hash]]
    )
    token = run_locked_evaluation.load_token_artifact(token_path)

    run_locked_evaluation.verify_bound_hashes(
        token,
        resolved_config_hash="resolved-config-hash-abc",
        dataset_checksum="dataset-checksum-abc",
        split_hash="split-hash-abc",
        checkpoint_paths={"intervention_aware": checkpoint_path},
    )  # must not raise


def test_verify_bound_hashes_raises_on_resolved_config_hash_mismatch(tmp_path):
    checkpoint_path = _write_checkpoint_file(tmp_path / "ckpt.pt")
    token_path = _write_token_artifact(tmp_path / "token.json")
    token = run_locked_evaluation.load_token_artifact(token_path)

    with pytest.raises(run_locked_evaluation.HashMismatchError, match="resolved_config_hash"):
        run_locked_evaluation.verify_bound_hashes(
            token,
            resolved_config_hash="WRONG",
            dataset_checksum="dataset-checksum-abc",
            split_hash="split-hash-abc",
            checkpoint_paths={"intervention_aware": checkpoint_path},
        )


def test_verify_bound_hashes_raises_on_dataset_checksum_mismatch(tmp_path):
    checkpoint_path = _write_checkpoint_file(tmp_path / "ckpt.pt")
    token_path = _write_token_artifact(tmp_path / "token.json")
    token = run_locked_evaluation.load_token_artifact(token_path)

    with pytest.raises(run_locked_evaluation.HashMismatchError, match="dataset_checksum"):
        run_locked_evaluation.verify_bound_hashes(
            token,
            resolved_config_hash="resolved-config-hash-abc",
            dataset_checksum="WRONG",
            split_hash="split-hash-abc",
            checkpoint_paths={"intervention_aware": checkpoint_path},
        )


def test_verify_bound_hashes_raises_on_split_hash_mismatch(tmp_path):
    checkpoint_path = _write_checkpoint_file(tmp_path / "ckpt.pt")
    token_path = _write_token_artifact(tmp_path / "token.json")
    token = run_locked_evaluation.load_token_artifact(token_path)

    with pytest.raises(run_locked_evaluation.HashMismatchError, match="split_hash"):
        run_locked_evaluation.verify_bound_hashes(
            token,
            resolved_config_hash="resolved-config-hash-abc",
            dataset_checksum="dataset-checksum-abc",
            split_hash="WRONG",
            checkpoint_paths={"intervention_aware": checkpoint_path},
        )


def test_verify_bound_hashes_raises_on_a_checkpoint_content_mismatch(tmp_path):
    checkpoint_path = _write_checkpoint_file(tmp_path / "ckpt.pt", content=b"different-bytes")
    token_path = _write_token_artifact(tmp_path / "token.json")  # bound to "checkpoint-sha256-abc"
    token = run_locked_evaluation.load_token_artifact(token_path)

    with pytest.raises(run_locked_evaluation.HashMismatchError, match="checkpoint_hashes"):
        run_locked_evaluation.verify_bound_hashes(
            token,
            resolved_config_hash="resolved-config-hash-abc",
            dataset_checksum="dataset-checksum-abc",
            split_hash="split-hash-abc",
            checkpoint_paths={"intervention_aware": checkpoint_path},
        )


def test_verify_bound_hashes_raises_when_a_bound_checkpoint_is_missing(tmp_path):
    token_path = _write_token_artifact(tmp_path / "token.json")
    token = run_locked_evaluation.load_token_artifact(token_path)

    with pytest.raises(run_locked_evaluation.HashMismatchError, match="checkpoint_hashes"):
        run_locked_evaluation.verify_bound_hashes(
            token,
            resolved_config_hash="resolved-config-hash-abc",
            dataset_checksum="dataset-checksum-abc",
            split_hash="split-hash-abc",
            checkpoint_paths={},
        )


def test_verify_bound_hashes_raises_when_an_extra_unbound_checkpoint_is_supplied(tmp_path):
    checkpoint_path = _write_checkpoint_file(tmp_path / "ckpt.pt")
    extra_path = _write_checkpoint_file(tmp_path / "extra.pt")
    import hashlib

    real_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    token_path = _write_token_artifact(
        tmp_path / "token.json", checkpoint_hashes=[["intervention_aware", real_hash]]
    )
    token = run_locked_evaluation.load_token_artifact(token_path)

    with pytest.raises(run_locked_evaluation.HashMismatchError, match="checkpoint_hashes"):
        run_locked_evaluation.verify_bound_hashes(
            token,
            resolved_config_hash="resolved-config-hash-abc",
            dataset_checksum="dataset-checksum-abc",
            split_hash="split-hash-abc",
            checkpoint_paths={
                "intervention_aware": checkpoint_path,
                "sequence_gru": extra_path,
            },
        )


def test_main_refuses_on_a_hash_mismatch_leaving_the_results_tree_byte_identical(tmp_path, capsys):
    results_root = tmp_path / "results"
    results_root.mkdir()
    marker = results_root / "marker.txt"
    marker.write_text("untouched")
    before = marker.read_bytes()

    token_path = _write_token_artifact(tmp_path / "token.json")

    exit_code = run_locked_evaluation.main(
        [
            "--token-artifact",
            str(token_path),
            "--results-root",
            str(results_root),
            "--resolved-config-hash",
            "WRONG",
            "--dataset-checksum",
            "dataset-checksum-abc",
            "--split-hash",
            "split-hash-abc",
            "--checkpoint",
            "intervention_aware=/does/not/matter",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "resolved_config_hash" in captured.err
    assert marker.read_bytes() == before
    assert not (results_root / "locked_evaluation").exists()


# ---------------------------------------------------------------------------
# Requirement 13.10: occupied-path refusal.
# ---------------------------------------------------------------------------


def test_verify_target_path_unoccupied_passes_for_a_fresh_path(tmp_path):
    run_locked_evaluation.verify_target_path_unoccupied(tmp_path, "eval-1")  # must not raise


def test_verify_target_path_unoccupied_raises_when_occupied(tmp_path):
    target = run_locked_evaluation.locked_evaluation_result_dir(tmp_path, "eval-1")
    target.mkdir(parents=True)
    (target / "existing.json").write_text("{}")

    with pytest.raises(run_locked_evaluation.OccupiedPathError, match="eval-1"):
        run_locked_evaluation.verify_target_path_unoccupied(tmp_path, "eval-1")


def test_verify_target_path_unoccupied_passes_for_an_empty_existing_directory(tmp_path):
    target = run_locked_evaluation.locked_evaluation_result_dir(tmp_path, "eval-1")
    target.mkdir(parents=True)

    run_locked_evaluation.verify_target_path_unoccupied(tmp_path, "eval-1")  # must not raise


def test_main_succeeds_through_verification_when_everything_resolves(tmp_path, capsys):
    import hashlib

    checkpoint_path = _write_checkpoint_file(tmp_path / "ckpt.pt")
    real_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    token_path = _write_token_artifact(
        tmp_path / "token.json", checkpoint_hashes=[["intervention_aware", real_hash]]
    )

    exit_code = run_locked_evaluation.main(
        [
            "--token-artifact",
            str(token_path),
            "--results-root",
            str(tmp_path / "results"),
            "--resolved-config-hash",
            "resolved-config-hash-abc",
            "--dataset-checksum",
            "dataset-checksum-abc",
            "--split-hash",
            "split-hash-abc",
            "--checkpoint",
            f"intervention_aware={checkpoint_path}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "eval-abc-123" in captured.out
