"""Suite-wide fixtures.

Currently one concern only: every test process must declare where
reference-access audit records go.

`cohorts.sealed_reference.SealedReference.unseal` and
`read_sealed_artifact` append `{evaluation_id, accessed_at}` to
`<results_root>/locked_evaluation/reference_access.jsonl` on every
authorized read (Requirement 14.5), and they *refuse the read* when no
results root is resolvable rather than handing over the sealed reference
with no audit record (see
`infra.consumption_ledger`'s docstring, "Where the reference-access
ledger's root comes from"). Neither call site is required to pass
`results_root=` --- the locked-evaluation runner does, but the many tests
that just need `unseal(token)` to work do not --- so the test process
declares its root once, here, through the same `CWPWM_RESULTS_ROOT`
variable any other process would use.

The root is a fresh directory per test, outside the test's own `tmp_path`.
Two reasons for both halves of that:

- *Per test*, so one test's access records cannot be seen by another. An
  access ledger that accumulated across the suite would make "exactly one
  record was appended" untestable.
- *Outside `tmp_path`*, because tests that assert over their `tmp_path`'s
  contents (directory listings, export scans, artifact trees) must not
  suddenly find a `locked_evaluation/` directory in them.

A test that wants to exercise the unconfigured case deletes the variable
itself (`monkeypatch.delenv(RESULTS_ROOT_ENV_VAR)`); a test that wants the
records in a specific place passes `results_root=` explicitly, which takes
precedence over this.
"""

from __future__ import annotations

import pytest

from infra.consumption_ledger import RESULTS_ROOT_ENV_VAR


@pytest.fixture(autouse=True)
def isolated_reference_access_root(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    """Point `CWPWM_RESULTS_ROOT` at a fresh, empty results root.

    Autouse: the point is that no test can read the sealed reference
    without an audit trail, so no test opts out of having one. `monkeypatch`
    restores (or removes) the previous value at teardown, so the variable
    never leaks between tests or out of the suite.

    Yields:
        The results root, for a test that wants to read the ledger back
        without spelling the path out again.
    """
    results_root = tmp_path_factory.mktemp("results_root")
    monkeypatch.setenv(RESULTS_ROOT_ENV_VAR, str(results_root))
    yield results_root
