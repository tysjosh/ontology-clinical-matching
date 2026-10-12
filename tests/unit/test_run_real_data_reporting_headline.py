"""Unit test for `scripts/run_real_data_reporting.py`'s
`_unsupported_headline_comparisons()` helper (task 22.1; Requirement
17.1-17.5).

Deliberately separate from `tests/integration/test_run_real_data_reporting.py`,
which is skipped when the real ACTG175 data is absent -- this helper takes
no data and no pipeline input at all, so it should be checkable
unconditionally.

**Validates: Requirements 17.1, 17.2, 17.3, 17.4, 17.5**
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_real_data_reporting import (  # noqa: E402
    _HEADLINE_COMPARISON_UNSUPPORTED_REASON,
    _unsupported_headline_comparisons,
)

from reporting.tables import HEADLINE_COMPARISON_NAMES  # noqa: E402


def test_names_exactly_the_five_headline_comparisons():
    result = _unsupported_headline_comparisons()
    assert set(result) == set(HEADLINE_COMPARISON_NAMES)


def test_every_entry_is_marked_unsupported_with_a_reason():
    result = _unsupported_headline_comparisons()
    for name, entry in result.items():
        assert entry["supported"] is False, name
        assert entry["reason"] == _HEADLINE_COMPARISON_UNSUPPORTED_REASON
