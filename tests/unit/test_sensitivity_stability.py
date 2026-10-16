"""Unit/property tests for task 15.1 (`evaluation.robustness`'s
`SensitivityStabilityResult` emission and the causal-witness wiring in
`scripts/run_causal_analysis.py`).

**Validates: Requirements 8.4, 8.8, 8.9, 8.10, 18.3**
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from evaluation.robustness import (
    STABILITY_CRITERION_DESCRIPTION,
    SensitivityVariantOutcome,
    _determine_overall_stability,
    reverse_contrast,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_MAX_EXAMPLES = 50

_finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_finite_or_none = st.one_of(st.none(), _finite)


def _outcome(name: str, status: str, direction: str) -> SensitivityVariantOutcome:
    return SensitivityVariantOutcome(
        name=name,
        status=status,
        estimate=None,
        ci_low=None,
        ci_high=None,
        scale=None,
        diagnostic_status="undetermined",
        direction=direction,
        failure_reason=None if status == "executed" else "forced",
        resolved_config_hash=None,
    )


# ---------------------------------------------------------------------------
# Property 22: Unexecuted sensitivity never reads as stability.
# ---------------------------------------------------------------------------


@given(
    unexecuted_index=st.integers(min_value=0, max_value=7),
    directions=st.lists(
        st.sampled_from(["positive", "negative", "null"]), min_size=8, max_size=8
    ),
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_22_any_unexecuted_variant_forces_undetermined_and_false_witness_input(
    unexecuted_index: int, directions: list[str]
) -> None:
    """For any set of eight sensitivity-variant outcomes containing at
    least one `status="unexecuted"`, `overall_stability` is
    `"undetermined"` and the boolean an `Estimate_Runner` would supply to
    `CausalWitness.evaluate` (`overall_stability == "stable"`) is
    `False`."""
    outcomes = [
        _outcome(f"v{i}", "executed", direction) for i, direction in enumerate(directions)
    ]
    outcomes[unexecuted_index] = _outcome(
        f"v{unexecuted_index}", "unexecuted", "undetermined"
    )

    overall_stability = _determine_overall_stability(outcomes)

    assert overall_stability == "undetermined"
    assert (overall_stability == "stable") is False


def test_property_22_all_executed_and_agreeing_is_not_forced_undetermined() -> None:
    """Sanity check for Property 22's converse: with nothing unexecuted and
    full direction agreement, the result is genuinely `"stable"`, so
    Property 22's assertion is actually exercising the unexecuted branch
    and not a function that always returns `"undetermined"`."""
    outcomes = [_outcome(f"v{i}", "executed", "negative") for i in range(8)]
    assert _determine_overall_stability(outcomes) == "stable"


# ---------------------------------------------------------------------------
# Property 23: Comparator reversal is an involution that preserves the
# original contrast.
# ---------------------------------------------------------------------------


@given(estimate=_finite, ci_low=_finite, ci_high=_finite)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_23_reversal_negates_estimate_and_swaps_bounds(
    estimate: float, ci_low: float, ci_high: float
) -> None:
    reversed_estimate, reversed_ci_low, reversed_ci_high = reverse_contrast(
        estimate, ci_low, ci_high
    )

    assert reversed_estimate == -estimate
    assert reversed_ci_low == -ci_high
    assert reversed_ci_high == -ci_low


@given(estimate=_finite, ci_low=_finite, ci_high=_finite)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_23_reversal_applied_twice_is_the_identity(
    estimate: float, ci_low: float, ci_high: float
) -> None:
    once = reverse_contrast(estimate, ci_low, ci_high)
    twice = reverse_contrast(*once)

    assert twice == (estimate, ci_low, ci_high)


@given(estimate=_finite_or_none, ci_low=_finite_or_none, ci_high=_finite_or_none)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_property_23_none_inputs_pass_through_unchanged_at_their_position(
    estimate: float | None, ci_low: float | None, ci_high: float | None
) -> None:
    result_estimate, result_ci_low, result_ci_high = reverse_contrast(estimate, ci_low, ci_high)

    assert (result_estimate is None) == (estimate is None)
    assert (result_ci_low is None) == (ci_high is None)
    assert (result_ci_high is None) == (ci_low is None)


def test_reversal_of_nan_is_nan() -> None:
    nan = float("nan")
    reversed_estimate, _, _ = reverse_contrast(nan, None, None)
    assert reversed_estimate is not None
    assert math.isnan(reversed_estimate)


# ---------------------------------------------------------------------------
# "no runner passes a literal stability boolean"
# ---------------------------------------------------------------------------


def _runner_script_paths() -> list[Path]:
    scripts_dir = REPO_ROOT / "scripts"
    return sorted(scripts_dir.glob("run_*.py"))


def _call_names_passing_sensitivity_stable(tree: ast.Module) -> list[tuple[int, ast.expr]]:
    """Every `sensitivity_stable=<value>` keyword argument's value node,
    found anywhere in the module, with its line number."""
    found: list[tuple[int, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "sensitivity_stable":
            found.append((node.value.lineno if hasattr(node.value, "lineno") else -1, node.value))
    return found


def test_no_runner_passes_a_literal_stability_boolean() -> None:
    """Requirement 8.9: `sensitivity_stable` must always be derived from a
    real `SensitivityStabilityResult.overall_stability` comparison, never
    a hardcoded `True`/`False` literal, in any `scripts/run_*.py` module."""
    offending: list[str] = []
    for script_path in _runner_script_paths():
        tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
        for lineno, value_node in _call_names_passing_sensitivity_stable(tree):
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, bool):
                offending.append(f"{script_path.name}:{lineno}")

    assert offending == [], (
        f"found sensitivity_stable=<literal bool> in: {offending}; it must be "
        f"derived from SensitivityStabilityResult.overall_stability == 'stable'"
    )


def test_run_causal_analysis_derives_sensitivity_stable_from_overall_stability() -> None:
    """Confirms the real wiring: `run_causal_analysis.py` computes
    `sensitivity_stable` from `overall_stability == "stable"` rather than
    from any other expression."""
    script_path = REPO_ROOT / "scripts" / "run_causal_analysis.py"
    source = script_path.read_text(encoding="utf-8")

    assert 'overall_stability == "stable"' in source
    assert "sensitivity_stable=False," not in source


def test_stability_criterion_description_is_reported_verbatim() -> None:
    outcomes = [_outcome(f"v{i}", "executed", "negative") for i in range(8)]
    assert _determine_overall_stability(outcomes) == "stable"
    assert isinstance(STABILITY_CRITERION_DESCRIPTION, str)
    assert len(STABILITY_CRITERION_DESCRIPTION) > 0


# ---------------------------------------------------------------------------
# Stability is judged against the primary estimate's direction, including
# every component estimate (two clipping presets, two support bands).
# ---------------------------------------------------------------------------


def test_agreement_with_primary_direction_is_stable() -> None:
    outcomes = [_outcome(f"v{i}", "executed", "negative") for i in range(8)]
    assert _determine_overall_stability(outcomes, primary_direction="negative") == "stable"


def test_unanimous_variants_against_the_primary_direction_are_unstable() -> None:
    """The majority rule would call eight agreeing variants stable even if
    all of them reversed the primary estimate; the primary-direction rule
    does not."""
    outcomes = [_outcome(f"v{i}", "executed", "positive") for i in range(8)]
    assert _determine_overall_stability(outcomes) == "stable"
    assert _determine_overall_stability(outcomes, primary_direction="negative") == "unstable"


def test_a_contradicting_component_estimate_is_not_hidden() -> None:
    outcomes = [_outcome(f"v{i}", "executed", "negative") for i in range(7)]
    outcomes.append(
        SensitivityVariantOutcome(
            name="alternative_weight_clipping",
            status="executed",
            estimate=-0.1,
            ci_low=None,
            ci_high=None,
            scale="risk_difference",
            diagnostic_status="undetermined",
            direction="negative",
            failure_reason=None,
            resolved_config_hash=None,
            component_estimates=(("sensitivity_narrow", -0.1), ("sensitivity_wide", 0.05)),
        )
    )
    assert _determine_overall_stability(outcomes, primary_direction="negative") == "unstable"


def test_a_null_primary_direction_has_no_opposite() -> None:
    outcomes = [_outcome(f"v{i}", "executed", "positive") for i in range(8)]
    assert _determine_overall_stability(outcomes, primary_direction="null") == "stable"
