"""Mechanical enforcement of torch isolation (task 2.2; Requirements 5.1, 5.2).

`design.md` Section 2.4 makes torch a *leaf* dependency: exactly five
modules under `src/models/` may import it, and every other code path --
the six causal estimators, all three witnesses, overlap diagnostics,
negative controls, export scanning, the sealed reference, the readiness
ledger, the ablation matrix, the release gate -- must import and run in an
environment where torch is not installed at all.

Why a subprocess rather than a plain import test: torch 2.4.1 *is*
installed in this project's environment (it is declared in the `neural`
extra, and `dev` pulls that in). An in-process `import causal.baselines`
therefore proves nothing -- a stray `import torch` would succeed silently.
The only mechanism that actually demonstrates isolation is to reconstruct
the torch-free environment, which this module does by spawning a fresh
interpreter with a `sys.meta_path` finder that refuses to resolve `torch`
and any `torch.*` submodule.

Three details make that refusal hard to fake:

1. The finder raises `_TorchImportBlocked`, which derives from
   `BaseException`, not from `ImportError`. A module wrapping its torch
   import in `try/except ImportError` -- the shape of an optional
   dependency -- cannot convert a violation into a silent pass.
2. After each successful import, the subprocess re-checks `sys.modules`
   for any `torch`-prefixed entry, so a violation via `importlib` or
   `__import__` that somehow bypassed the finder still shows up.
3. `test_the_blocking_finder_actually_blocks` and
   `test_a_module_importing_torch_is_detected` are the meta-tests: they
   assert the harness reports a failure for a bare `import torch` and for
   a synthetic module that imports torch. Without them, a broken finder
   would make every assertion below vacuously green.

Modules that later tasks create (`evaluation.release_gate` from task 16,
`models.variants` from task 12, `infra.readiness` from task 3) are named
in `TORCH_FREE_MODULES` today and skipped with a reason while absent, so
the guarantee extends over them automatically the moment they land rather
than depending on someone remembering to add them here.

**Validates: Requirements 5.1, 5.2**
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_INIT = SRC_DIR / "models" / "__init__.py"

#: Every module that must import with torch unavailable, per `design.md`
#: Section 2.4. Modules not yet written are skipped individually (see
#: `_MODULES_PENDING`), never dropped from this tuple.
TORCH_FREE_MODULES = (
    "causal.adjustment_set",
    "causal.baselines",
    "causal.overlap",
    "causal.propensity",
    "causal.negative_controls",
    # The three witnesses.
    "witnesses.temporal",
    "witnesses.causal",
    "witnesses.clinical",
    "witnesses.aggregation",
    "witnesses.audit_report",
    "witnesses.audit_io",
    "cohorts.pipeline",
    "cohorts.sealed_reference",
    "evaluation.metrics",
    "evaluation.statistics",
    "evaluation.ablation_matrix",
    "evaluation.robustness",
    "evaluation.release_gate",
    "models.variants",
    "infra.readiness",
    # The go/no-go gate's evidence layer: it runs at the checkpoint before
    # locked evaluation, in whatever environment the reviewer has, so it must
    # import with torch absent (task 2.5).
    "infra.checklist_verifiers",
    # The consumption ledger the token minter reads (task 7). Covered
    # transitively through `cohorts.sealed_reference` already, but named
    # here because task 7.1's locked-evaluation runner imports it directly.
    "infra.consumption_ledger",
    "scripts.scan_exports",
    "scripts.check_go_no_go",
    # The locked-evaluation runner itself (task 21): it must reach every
    # verification step -- option rejection, token loading, consumption
    # check, hash recomputation, occupied-path check -- before it ever
    # needs torch (which only the as-yet-unimplemented task 21.1
    # evaluation steps require).
    "scripts.run_locked_evaluation",
)

#: Which task creates each module that does not exist yet, so a skip
#: message says why rather than just "missing".
_MODULES_PENDING = {
    "evaluation.release_gate": "task 16 (cohort release gate)",
    "models.variants": "task 12 (experiment variant dataclass and model factory)",
    "infra.readiness": "task 3 (readiness ledger)",
    "models.dataset": "task 8 (tensor dataset builder)",
}

#: Emitted by the subprocess when its finder refuses a torch import.
BLOCKED_MARKER = "TORCH_IMPORT_BLOCKED"

#: Run in a fresh interpreter. Imports each module named in argv under a
#: meta-path finder that refuses torch, and reports one record per module
#: as JSON on stdout.
_ISOLATION_HARNESS = f'''
import importlib
import json
import sys

BLOCKED_MARKER = {BLOCKED_MARKER!r}


class _TorchImportBlocked(BaseException):
    """Derives from BaseException so `except ImportError` cannot swallow it."""


class _TorchBlockingFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise _TorchImportBlocked(BLOCKED_MARKER + ": " + fullname)
        return None


def _torch_modules():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


if _torch_modules():
    print(json.dumps({{"__harness_error__": "torch was already imported at startup"}}))
    raise SystemExit(2)

sys.meta_path.insert(0, _TorchBlockingFinder())

results = {{}}
for name in sys.argv[1:]:
    record = {{"ok": False, "kind": "", "error": "", "submodules": []}}
    try:
        importlib.import_module(name)
    except _TorchImportBlocked as exc:
        record["kind"] = "torch_import"
        record["error"] = str(exc)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported
        record["kind"] = "other_import_error"
        record["error"] = type(exc).__name__ + ": " + str(exc)
    else:
        leaked = _torch_modules()
        if leaked:
            record["kind"] = "torch_in_sys_modules"
            record["error"] = "torch modules loaded: " + ", ".join(leaked)
        else:
            record["ok"] = True
        record["submodules"] = sorted(
            m for m in sys.modules if m.startswith(name + ".")
        )
    results[name] = record

print(json.dumps(results))
'''


def _module_source_path(module: str) -> Path:
    """On-disk location of `module`, without importing anything.

    `scripts/` is not an installed package (it holds CLI entry points, and
    is importable as a namespace package from the repository root -- the
    convention `tests/unit/test_scan_exports.py` already relies on).
    Everything else lives under `src/`.
    """
    root = SCRIPTS_DIR if module.startswith("scripts.") else SRC_DIR
    parts = module.split(".")[1:] if module.startswith("scripts.") else module.split(".")
    as_module = root.joinpath(*parts).with_suffix(".py")
    if as_module.is_file():
        return as_module
    return root.joinpath(*parts) / "__init__.py"


def _module_exists(module: str) -> bool:
    return _module_source_path(module).is_file()


def _run_isolation_harness(
    modules: tuple[str, ...],
    extra_path: Path | None = None,
) -> dict[str, dict]:
    """Import `modules` in a fresh torch-free interpreter; return one record each."""
    path_entries = [str(SRC_DIR), str(REPO_ROOT)]
    if extra_path is not None:
        path_entries.insert(0, str(extra_path))

    completed = subprocess.run(
        [sys.executable, "-c", _ISOLATION_HARNESS, *modules],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ":".join(path_entries), "PATH": "/usr/bin:/bin"},
        timeout=300,
        check=False,
    )

    assert completed.stdout.strip(), (
        f"isolation harness produced no output (exit {completed.returncode});\n"
        f"stderr:\n{completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert "__harness_error__" not in payload, (
        f"isolation harness could not establish a torch-free interpreter: "
        f"{payload['__harness_error__']}"
    )
    return payload


@pytest.fixture(scope="module")
def isolation_results() -> dict[str, dict]:
    """One subprocess for every module that exists today."""
    present = tuple(m for m in TORCH_FREE_MODULES if _module_exists(m))
    return _run_isolation_harness(present)


# --------------------------------------------------------------------------
# The harness itself must be able to detect a violation
# --------------------------------------------------------------------------


def test_the_blocking_finder_actually_blocks():
    """A bare `import torch` inside the harness is reported as a violation.

    torch is installed here, so without the finder this import succeeds and
    every other assertion in this module becomes meaningless.
    """
    results = _run_isolation_harness(("torch",))

    record = results["torch"]
    assert not record["ok"], (
        "the harness imported torch successfully, so its meta-path finder is "
        "not blocking anything and this module proves nothing"
    )
    assert record["kind"] == "torch_import"
    assert BLOCKED_MARKER in record["error"]


def test_a_module_importing_torch_is_detected(tmp_path):
    """A module whose top-level body imports torch fails the harness.

    Also covers the optional-dependency shape: the second module wraps its
    import in `except ImportError`, which must not launder the violation.
    """
    (tmp_path / "_probe_plain_torch_import.py").write_text("import torch\n")
    (tmp_path / "_probe_guarded_torch_import.py").write_text(
        "try:\n    import torch\nexcept ImportError:\n    torch = None\n"
    )

    results = _run_isolation_harness(
        ("_probe_plain_torch_import", "_probe_guarded_torch_import"),
        extra_path=tmp_path,
    )

    for name, record in results.items():
        assert not record["ok"], f"{name} imported torch undetected"
        assert record["kind"] == "torch_import", (
            f"{name} failed for the wrong reason: {record['error']}"
        )


def test_the_harness_reports_a_clean_import_as_clean(tmp_path):
    """The complement: a torch-free module is not falsely flagged."""
    (tmp_path / "_probe_torch_free.py").write_text("VALUE = 1\n")

    results = _run_isolation_harness(("_probe_torch_free",), extra_path=tmp_path)

    assert results["_probe_torch_free"]["ok"], results["_probe_torch_free"]["error"]


# --------------------------------------------------------------------------
# Requirements 5.1-5.2: the non-neural layer imports without torch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module", TORCH_FREE_MODULES)
def test_module_imports_without_torch(module, isolation_results):
    if not _module_exists(module):
        pending = _MODULES_PENDING.get(module, "a later task")
        pytest.skip(
            f"{module} does not exist yet; it is created by {pending}. It stays "
            f"in TORCH_FREE_MODULES so this guarantee covers it as soon as it lands."
        )

    record = isolation_results[module]

    assert record["ok"], (
        f"{module} does not import in a torch-free interpreter "
        f"({record['kind']}): {record['error']}"
    )


def test_every_listed_module_is_accounted_for():
    """No module is silently absent from both the run and the pending list."""
    unexplained = [
        m
        for m in TORCH_FREE_MODULES
        if not _module_exists(m) and m not in _MODULES_PENDING
    ]

    assert unexplained == [], (
        f"these modules are missing and no creating task is recorded for them: "
        f"{unexplained}. Either the module was renamed or _MODULES_PENDING is stale."
    )


# --------------------------------------------------------------------------
# Second assertion: `src/models/__init__.py` pulls in no torch-bearing module
# --------------------------------------------------------------------------


def _imported_names(source: str) -> list[str]:
    """Every module name any import statement in `source` refers to.

    Walks the whole tree, so an import nested inside a function body is
    reported too. `models.variants` defers its torch imports into
    `build_model`'s body by design, but `models/__init__.py` is not allowed
    to reach a torch-bearing module by any route, deferred included.
    """
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.append(base)
            names.extend(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return names


def test_models_init_imports_no_torch_module_and_not_dataset():
    """`design.md` Section 2.4: `models/__init__.py` imports no `torch_*`, no `dataset`.

    Read statically, because the runtime check below can only observe what
    the current file happens to execute; the static check also catches an
    import placed inside a function or an `if TYPE_CHECKING` block.
    """
    imported = _imported_names(MODELS_INIT.read_text())

    offenders = [
        name
        for name in imported
        if any(part.startswith("torch") or part == "dataset" for part in name.split("."))
    ]

    assert offenders == [], (
        f"src/models/__init__.py imports {offenders}; importing a `torch_*` "
        f"module or `models.dataset` from the package __init__ makes torch a "
        f"trunk dependency of every `from models import ...` in the codebase "
        f"(design.md Section 2.4)"
    )


def test_importing_the_models_package_loads_no_torch_bearing_submodule():
    """The runtime complement: `import models` under the blocker stays clean.

    Asserted on its own subprocess so the loaded-submodule set is
    attributable to the package import alone.
    """
    record = _run_isolation_harness(("models",))["models"]

    assert record["ok"], (
        f"`import models` fails in a torch-free interpreter "
        f"({record['kind']}): {record['error']}"
    )

    forbidden = [
        name
        for name in record["submodules"]
        if name.split(".")[-1].startswith("torch_") or name == "models.dataset"
    ]

    assert forbidden == [], (
        f"importing the `models` package pulled in {forbidden}; those modules "
        f"import torch and must be reached only by an explicit import of the "
        f"module itself"
    )
