"""Shared test factory for fully bound `LockedEvaluationToken`s
(`neural-experiment-pipeline/design.md` Section 4.6; Requirement 13.3).

Why this module exists
-----------------------

`cohorts.sealed_reference.LockedEvaluationToken` binds nine fields, all
required with no defaults, and `_mint()` is the only construction path. That
is the point of Requirement 13.3 — but it means every test that merely needs
*a* valid token to hand to `assert_reference_unlocked`,
`SealedReference.unseal`, or `require_test_split` would otherwise have to
spell out nine synthetic hashes inline.

The alternative — giving the new fields defaults so `_mint(issued_by=...)`
keeps working — was rejected: a default is exactly the "partially bound
token" Requirement 13.3 forbids, and a default that only tests use is still
a default the minter can reach. So the defaults live here, in test code,
where they are unmistakably synthetic (every value is prefixed
`synthetic-`) and cannot leak into `src/` or `scripts/`.

Guard tests keep testing the guard
-----------------------------------

Tests that assert `assert_reference_unlocked`/`assert_test_split_unlocked`
behavior are unchanged in *substance*: they still pass a genuine, minted
token and still assert the `isinstance` check accepts it, and still pass
`None`/wrong-typed values and assert it refuses. Only the token's
*construction* moved here. The guard itself was not touched by task 7.

Not a `conftest.py` fixture
----------------------------

Deliberately a plain callable rather than a pytest fixture: the three test
modules that need it use it inside `with pytest.raises(...)` blocks, inside
`hypothesis` strategies, and several times per test with different
overrides. A fixture would give one token per test; a factory gives as many
as a test needs, each independently overridable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cohorts.sealed_reference import LockedEvaluationToken

#: Results root used when a caller does not supply one. Deliberately a
#: directory that does not exist: `infra.consumption_ledger.is_consumed`
#: treats a missing `locked_evaluation/consumed.jsonl` as an empty ledger,
#: so the default reads nothing, writes nothing, and — importantly — never
#: consults the repository's real `results/` tree, which would make token
#: minting in tests depend on whatever a previous real run had consumed.
EMPTY_LEDGER_ROOT: Path = Path(__file__).parent / "_nonexistent_results_root"

#: The synthetic bindings. Every value is visibly not a real hash, so a
#: token that escapes into an artifact is recognizable on sight.
SYNTHETIC_BINDINGS: dict[str, Any] = {
    "resolved_config_hash": "synthetic-resolved-config-hash",
    "dataset_checksum": "synthetic-dataset-checksum",
    "split_hash": "synthetic-split-hash",
    "checkpoint_hashes": (
        ("baseline_gru", "synthetic-baseline-gru-sha256"),
        ("intervention_aware", "synthetic-intervention-aware-sha256"),
    ),
    "git_commit": "synthetic-git-commit",
    "statistical_plan_version": "synthetic-statistical-plan-version",
    "evaluation_id": "synthetic-evaluation-id",
}


def synthetic_token(
    *,
    issued_by: str = "tests/token_fixtures.py::synthetic_token",
    results_root: str | Path = EMPTY_LEDGER_ROOT,
    **overrides: Any,
) -> LockedEvaluationToken:
    """Mint a fully bound token with synthetic values for every binding.

    Routes through `LockedEvaluationToken._mint`, not the dataclass
    constructor, so tests exercise the real minting path — including the
    field validation and the consumption-ledger check — rather than a
    parallel construction route that could drift from it.

    Args:
        issued_by: the minter identity to record.
        results_root: top-level results root whose
            `locked_evaluation/consumed.jsonl` the mint checks. Pass a
            `tmp_path` to exercise the duplicate-`evaluation_id` refusal.
        **overrides: any of `SYNTHETIC_BINDINGS`' keys, to vary or
            deliberately invalidate one binding (e.g.
            `evaluation_id="already-consumed"`, `split_hash=""`).

    Returns:
        The minted `LockedEvaluationToken`.

    Raises:
        cohorts.sealed_reference.TokenFieldError: if an override makes a
            binding absent, empty, or malformed, or if `evaluation_id` is
            already recorded as consumed under `results_root`. Tests assert
            this deliberately.
    """
    bindings = dict(SYNTHETIC_BINDINGS)
    unknown = set(overrides) - set(bindings)
    if unknown:
        raise TypeError(
            f"synthetic_token() got unexpected binding override(s): "
            f"{', '.join(sorted(unknown))}. Known bindings: "
            f"{', '.join(sorted(bindings))}."
        )
    bindings.update(overrides)

    return LockedEvaluationToken._mint(
        issued_by=issued_by,
        results_root=results_root,
        **bindings,
    )
