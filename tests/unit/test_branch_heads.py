"""Unit tests for `src/models/branch_heads.py`.

Covers:

1. Factual/counterfactual tagging (Requirements 9.4, 9.5): for
   `observed_treatment_arm=1`, `branch_1`'s output is tagged `"factual"`
   and `branch_0`'s is tagged `"counterfactual"`; for
   `observed_treatment_arm=0`, the tagging is reversed.
2. Exactly one of the two returned `BranchOutput`s is tagged `"factual"`
   in both cases (never zero, never two) -- Requirement 9.5.
3. Basic shape/sanity: `event_hazard` is in `[0, 1]` (sigmoid-bounded);
   `next_cd4`/`next_cd8` are finite floats.

**Validates: Requirements 9.4, 9.5**
"""

import math

from models.branch_heads import BranchHead, BranchOutput, evaluate_both_branches

HIDDEN_SIZE = 6
HIDDEN_STATE = [0.1, -0.2, 0.3, 0.05, -0.4, 0.15]


def _make_heads() -> tuple[BranchHead, BranchHead]:
    branch_1_head = BranchHead.initialize(hidden_size=HIDDEN_SIZE, seed=1)
    branch_0_head = BranchHead.initialize(hidden_size=HIDDEN_SIZE, seed=2)
    return branch_1_head, branch_0_head


def _assert_sane(output: BranchOutput) -> None:
    assert 0.0 <= output.event_hazard <= 1.0
    assert isinstance(output.next_cd4, float)
    assert isinstance(output.next_cd8, float)
    assert math.isfinite(output.next_cd4)
    assert math.isfinite(output.next_cd8)


# ---------------------------------------------------------------------------
# Factual/counterfactual tagging (Requirements 9.4, 9.5)
# ---------------------------------------------------------------------------


def test_observed_treatment_arm_1_tags_branch_1_factual():
    branch_1_head, branch_0_head = _make_heads()

    branch_1_output, branch_0_output = evaluate_both_branches(
        HIDDEN_STATE, branch_1_head, branch_0_head, observed_treatment_arm=1
    )

    assert branch_1_output.tag == "factual"
    assert branch_0_output.tag == "counterfactual"


def test_observed_treatment_arm_0_tags_branch_0_factual():
    branch_1_head, branch_0_head = _make_heads()

    branch_1_output, branch_0_output = evaluate_both_branches(
        HIDDEN_STATE, branch_1_head, branch_0_head, observed_treatment_arm=0
    )

    assert branch_1_output.tag == "counterfactual"
    assert branch_0_output.tag == "factual"


def test_exactly_one_branch_tagged_factual_for_each_observed_arm():
    """Requirement 9.5: "Only one potential outcome per patient SHALL be
    described as factually observed" -- never zero, never two."""
    branch_1_head, branch_0_head = _make_heads()

    for observed_arm in (0, 1):
        branch_1_output, branch_0_output = evaluate_both_branches(
            HIDDEN_STATE, branch_1_head, branch_0_head, observed_treatment_arm=observed_arm
        )
        tags = [branch_1_output.tag, branch_0_output.tag]
        assert tags.count("factual") == 1
        assert tags.count("counterfactual") == 1


def test_both_branches_evaluated_on_the_same_hidden_state():
    """`decisions.md` Section 14.3: "For every eligible patient, both
    branches are evaluated: Ŷ_i(1), Ŷ_i(0)." -- both heads run on the
    identical hidden state (not, e.g., a hidden state that already
    encodes which arm was observed)."""
    branch_1_head, branch_0_head = _make_heads()

    branch_1_output_arm1, branch_0_output_arm1 = evaluate_both_branches(
        HIDDEN_STATE, branch_1_head, branch_0_head, observed_treatment_arm=1
    )
    branch_1_output_arm0, branch_0_output_arm0 = evaluate_both_branches(
        HIDDEN_STATE, branch_1_head, branch_0_head, observed_treatment_arm=0
    )

    # The underlying predictions are identical regardless of
    # observed_treatment_arm -- only the tags differ.
    assert branch_1_output_arm1.event_hazard == branch_1_output_arm0.event_hazard
    assert branch_1_output_arm1.next_cd4 == branch_1_output_arm0.next_cd4
    assert branch_1_output_arm1.next_cd8 == branch_1_output_arm0.next_cd8
    assert branch_0_output_arm1.event_hazard == branch_0_output_arm0.event_hazard


# ---------------------------------------------------------------------------
# Shape/sanity checks
# ---------------------------------------------------------------------------


def test_branch_outputs_are_sane():
    branch_1_head, branch_0_head = _make_heads()

    branch_1_output, branch_0_output = evaluate_both_branches(
        HIDDEN_STATE, branch_1_head, branch_0_head, observed_treatment_arm=1
    )

    _assert_sane(branch_1_output)
    _assert_sane(branch_0_output)


def test_invalid_observed_treatment_arm_raises():
    branch_1_head, branch_0_head = _make_heads()

    try:
        evaluate_both_branches(HIDDEN_STATE, branch_1_head, branch_0_head, observed_treatment_arm=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for observed_treatment_arm=2")


def test_branch_head_predict_rejects_wrong_length_hidden_state():
    branch_1_head, _ = _make_heads()

    try:
        branch_1_head.predict([0.1, 0.2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched hidden_state length")
