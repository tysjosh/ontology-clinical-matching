"""Collection root for torch-requiring tests (design.md Section 2.4).

Every test module in this package needs the optional `neural` extra
(`pip install -e '.[neural]'`). Each one must:

- guard its torch import with `pytest.importorskip("torch")` at module
  scope, so a base, torch-free environment skips rather than errors; and
- carry `pytestmark = pytest.mark.neural`, the marker registered in
  `pyproject.toml`, so `-m "not neural"` deselects the whole package.

Nothing under `tests/unit/`, `tests/property/`, `tests/integration/`,
`tests/e2e/`, `tests/frozen/`, or `tests/reporting/` may import torch --
`tests/unit/test_torch_isolation.py` enforces that mechanically.
"""
