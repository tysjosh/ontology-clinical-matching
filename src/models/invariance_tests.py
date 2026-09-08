"""Reusable intervention-invariance check functions (Requirement 9.9;
`design.md`'s `models/invariance_tests.py` note: "property-based tests
(not unit fixtures) -- test_future_treatment_does_not_affect_earlier_state,
test_unchanged_treatment_preserves_factual_output,
test_changing_treatment_changes_counterfactual_output").

Requirement 9.9: "Intervention invariance tests SHALL verify that future
treatment cannot affect earlier states and that unchanged interventions
preserve the appropriate factual computation."

This module houses the invariance CHECKS themselves as small, reusable,
named functions operating directly on this codebase's existing model
components (`models.intervention_aware.GRUTransition`,
`models.branch_heads.evaluate_both_branches`,
`models.intervention_aware.intervention_features_from`) -- it does not
build any new model machinery, and it does not contain the `hypothesis`
test functions/strategies themselves. Following this project's
established test/source separation (see e.g. `models/losses.py` vs.
`tests/unit/test_losses.py`), the actual `hypothesis`-driven property
tests that call these functions with randomly generated inputs live in
`tests/property/test_invariance.py`.

Each `check_*` function either returns `True` or raises `AssertionError`
(via a plain `assert`) -- callers (property tests, or any future
diagnostic script/CLI) can call these directly without a pytest
dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

from models.branch_heads import BranchHead, evaluate_both_branches
from models.intervention_aware import GRUTransition, intervention_features_from


def check_future_treatment_does_not_affect_earlier_state(
    gru_transition: GRUTransition,
    z_sequence: Sequence[Sequence[float]],
    perturbed_last_z: Sequence[float],
    initial_hidden_state: Sequence[float] | None = None,
) -> bool:
    """Future treatment does not affect earlier state (Requirement 9.9,
    first half: "future treatment cannot affect earlier states").

    `z_sequence` is a landmark sequence (e.g. baseline, week 20, week 96).
    Rolling it out and then rolling out an otherwise-identical sequence
    whose LAST element is replaced by `perturbed_last_z` must produce
    IDENTICAL hidden states at every index *before* the last one -- the
    GRU recurrence's causal structure means `h_t` depends only on
    `h_{t-1}` and `z_t`, never on any future `z`.

    Raises `AssertionError` if any earlier hidden state changes.
    Returns `True` otherwise.
    """
    original_hidden_states = gru_transition.rollout(z_sequence, initial_hidden_state)

    perturbed_sequence = [*z_sequence[:-1], perturbed_last_z]
    perturbed_hidden_states = gru_transition.rollout(perturbed_sequence, initial_hidden_state)

    for index in range(len(z_sequence) - 1):
        assert original_hidden_states[index] == perturbed_hidden_states[index], (
            f"hidden state at index {index} changed when only the LAST "
            "timestep's intervention content was perturbed -- future "
            "treatment must not affect earlier states"
        )
    return True


def check_unchanged_treatment_preserves_factual_output(
    hidden_state: Sequence[float],
    branch_1_head: BranchHead,
    branch_0_head: BranchHead,
    observed_treatment_arm: int,
) -> bool:
    """Unchanged treatment preserves factual output (Requirement 9.9,
    second half: "unchanged interventions preserve the appropriate
    factual computation").

    Calling `evaluate_both_branches` twice with the SAME `hidden_state`
    and the SAME `observed_treatment_arm` must produce an IDENTICAL
    factual-tagged output both times.

    Raises `AssertionError` on any mismatch. Returns `True` otherwise.
    """
    first_branch_1, first_branch_0 = evaluate_both_branches(
        hidden_state, branch_1_head, branch_0_head, observed_treatment_arm
    )
    second_branch_1, second_branch_0 = evaluate_both_branches(
        hidden_state, branch_1_head, branch_0_head, observed_treatment_arm
    )

    first_factual = first_branch_1 if first_branch_1.tag == "factual" else first_branch_0
    second_factual = second_branch_1 if second_branch_1.tag == "factual" else second_branch_0

    assert first_factual == second_factual, (
        "repeated evaluation with an unchanged observed_treatment_arm "
        "must preserve the factual-tagged output"
    )
    return True


def check_changing_treatment_changes_counterfactual_output(
    hidden_state: Sequence[float],
    branch_1_head: BranchHead,
    branch_0_head: BranchHead,
) -> bool:
    """Changing treatment changes counterfactual output (Requirement
    9.9): for a FIXED hidden state, flipping `observed_treatment_arm`
    (0 -> 1 or 1 -> 0) swaps which branch carries the `"counterfactual"`
    tag (and which carries `"factual"`), while the underlying numeric
    predictions from each branch head do NOT depend on which arm is
    "observed" -- the branch heads only ever see the hidden state, never
    the observed-arm label.

    Raises `AssertionError` on any violation. Returns `True` otherwise.
    """
    branch_1_arm1, branch_0_arm1 = evaluate_both_branches(
        hidden_state, branch_1_head, branch_0_head, observed_treatment_arm=1
    )
    branch_1_arm0, branch_0_arm0 = evaluate_both_branches(
        hidden_state, branch_1_head, branch_0_head, observed_treatment_arm=0
    )

    # Which side is "counterfactual" swaps when the observed arm flips.
    assert branch_1_arm1.tag == "factual"
    assert branch_0_arm1.tag == "counterfactual"
    assert branch_1_arm0.tag == "counterfactual"
    assert branch_0_arm0.tag == "factual"

    # The underlying numeric predictions never depend on which arm was
    # "observed" -- only the tag changes.
    assert branch_1_arm1.event_hazard == branch_1_arm0.event_hazard
    assert branch_1_arm1.next_cd4 == branch_1_arm0.next_cd4
    assert branch_1_arm1.next_cd8 == branch_1_arm0.next_cd8
    assert branch_0_arm1.event_hazard == branch_0_arm0.event_hazard
    assert branch_0_arm1.next_cd4 == branch_0_arm0.next_cd4
    assert branch_0_arm1.next_cd8 == branch_0_arm0.next_cd8

    return True


def check_irrelevant_pathway_label_does_not_alter_predictions(
    treatment_arm: float,
    intervention_a: object,
    intervention_b: object,
) -> bool:
    """Irrelevant pathway-label changes do not alter predictions
    (Requirement 9.9's spirit, applied to this codebase's actual
    "pathway label" field: `models.intervention.Intervention
    .mechanism_ids`, which its own docstring documents as unused by any
    prediction path -- `decisions.md` Section 23 places mechanism/pathway
    encoders out of scope).

    `intervention_a` and `intervention_b` are expected to be two
    `models.intervention.Intervention` instances (or any object exposing
    `dose`/`adherence` attributes, matching `intervention_features_from`'s
    loose typing) that differ ONLY in `mechanism_ids`. Feeding both
    through `intervention_features_from` must produce an IDENTICAL
    feature vector, since that function only reads `dose`/`adherence` and
    never `mechanism_ids`.

    Raises `AssertionError` on any mismatch. Returns `True` otherwise.
    """
    features_a = intervention_features_from(treatment_arm, intervention_a)
    features_b = intervention_features_from(treatment_arm, intervention_b)

    assert features_a == features_b, (
        "varying mechanism_ids (an irrelevant pathway label, unused by "
        "intervention_features_from) must not change the resulting "
        "intervention feature vector"
    )
    return True
