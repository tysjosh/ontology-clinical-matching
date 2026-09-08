"""Property-based intervention-invariance tests (Requirement 9.9;
`design.md`'s `models/invariance_tests.py` note: "property-based tests
(not unit fixtures) -- test_future_treatment_does_not_affect_earlier_state,
test_unchanged_treatment_preserves_factual_output,
test_changing_treatment_changes_counterfactual_output").

These tests generate random synthetic patient states via `hypothesis` and
call the reusable check functions in `src/models/invariance_tests.py`,
which operate directly on this codebase's existing model components
(`models.intervention_aware.GRUTransition`, `models.branch_heads
.evaluate_both_branches`, `models.intervention_aware
.intervention_features_from`).

A fourth property (irrelevant pathway-label changes do not alter
predictions) is included using the actual field this codebase has for
that purpose: `models.intervention.Intervention.mechanism_ids` (see
`models/intervention.py`'s docstring and `decisions.md` Section 23 --
mechanism/pathway encoders are out of scope, and `mechanism_ids` is
documented as unused by any prediction path).

**Validates: Requirements 9.9**
"""

from __future__ import annotations

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from models.branch_heads import BranchHead
from models.intervention import Intervention
from models.intervention_aware import GRUTransition
from models.invariance_tests import (
    check_changing_treatment_changes_counterfactual_output,
    check_future_treatment_does_not_affect_earlier_state,
    check_irrelevant_pathway_label_does_not_alter_predictions,
    check_unchanged_treatment_preserves_factual_output,
)

_MAX_EXAMPLES = 30

# Fixed, small dimensions -- keep the pure-Python forward passes fast.
_INPUT_SIZE = 3
_HIDDEN_SIZE = 4

_finite_float = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)


def _z_vector():
    return st.lists(_finite_float, min_size=_INPUT_SIZE, max_size=_INPUT_SIZE)


def _hidden_vector():
    return st.lists(_finite_float, min_size=_HIDDEN_SIZE, max_size=_HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# Property 1: future treatment does not affect earlier state
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(
    z0=_z_vector(),
    z1=_z_vector(),
    z2=_z_vector(),
    perturbed_z2=_z_vector(),
)
def test_future_treatment_does_not_affect_earlier_state(z0, z1, z2, perturbed_z2):
    """For a 3-landmark `GRUTransition.rollout()` (baseline, week 20,
    week 96), changing ONLY the LAST timestep's (week 96's) z-vector --
    which carries the intervention channel's content -- must not change
    the hidden states produced at the earlier two timesteps. This follows
    directly from the GRU recurrence's causal structure: each `h_t`
    depends only on `h_{t-1}` and `z_t`, never on future `z`."""
    gru_transition = GRUTransition.initialize(
        input_size=_INPUT_SIZE, hidden_size=_HIDDEN_SIZE, seed=42
    )
    z_sequence = [z0, z1, z2]

    assert check_future_treatment_does_not_affect_earlier_state(
        gru_transition, z_sequence, perturbed_z2
    )


# ---------------------------------------------------------------------------
# Property 2: unchanged treatment preserves factual output
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(
    hidden_state=_hidden_vector(),
    observed_treatment_arm=st.sampled_from([0, 1]),
)
def test_unchanged_treatment_preserves_factual_output(hidden_state, observed_treatment_arm):
    """Repeated evaluation of `evaluate_both_branches` with an unchanged
    hidden state and an unchanged `observed_treatment_arm` must produce
    an identical factual-tagged output both times (determinism /
    preservation of the factual computation)."""
    branch_1_head = BranchHead.initialize(hidden_size=_HIDDEN_SIZE, seed=1)
    branch_0_head = BranchHead.initialize(hidden_size=_HIDDEN_SIZE, seed=2)

    assert check_unchanged_treatment_preserves_factual_output(
        hidden_state, branch_1_head, branch_0_head, observed_treatment_arm
    )


# ---------------------------------------------------------------------------
# Property 3: changing treatment changes counterfactual output
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(hidden_state=_hidden_vector())
def test_changing_treatment_changes_counterfactual_output(hidden_state):
    """For a FIXED hidden state, flipping `observed_treatment_arm` swaps
    which branch is tagged `"counterfactual"` vs. `"factual"`, while the
    underlying numeric predictions from each branch head never depend on
    which arm is "observed" -- only the tag does."""
    branch_1_head = BranchHead.initialize(hidden_size=_HIDDEN_SIZE, seed=1)
    branch_0_head = BranchHead.initialize(hidden_size=_HIDDEN_SIZE, seed=2)

    assert check_changing_treatment_changes_counterfactual_output(
        hidden_state, branch_1_head, branch_0_head
    )


# ---------------------------------------------------------------------------
# Property 4: irrelevant pathway-label changes do not alter predictions
# ---------------------------------------------------------------------------

_START_TIME = datetime(2020, 1, 1)

_mechanism_id_tuple = st.lists(
    st.text(alphabet=st.characters(whitelist_categories=("Ll", "Nd")), min_size=1, max_size=8),
    min_size=0,
    max_size=4,
).map(tuple)


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(
    treatment_arm=st.sampled_from([0.0, 1.0]),
    dose=st.one_of(st.none(), _finite_float),
    adherence=st.one_of(st.none(), _finite_float),
    mechanism_ids_a=_mechanism_id_tuple,
    mechanism_ids_b=_mechanism_id_tuple,
)
def test_irrelevant_pathway_label_does_not_alter_predictions(
    treatment_arm, dose, adherence, mechanism_ids_a, mechanism_ids_b
):
    """`models.intervention.Intervention.mechanism_ids` is this
    codebase's actual "pathway label" field, and its own docstring
    documents it as unused by any prediction path (mechanism/pathway
    encoders are out of scope per `decisions.md` Section 23). Varying
    `mechanism_ids` on an otherwise-identical `Intervention`, passed
    through `intervention_features_from`, must produce an identical
    feature vector regardless of `mechanism_ids`'s value."""
    intervention_a = Intervention(
        concept_id="zdv",
        dose=dose,
        unit=None,
        route=None,
        frequency=None,
        start_time=_START_TIME,
        stop_time=None,
        adherence=adherence,
        mechanism_ids=mechanism_ids_a,
    )
    intervention_b = Intervention(
        concept_id="zdv",
        dose=dose,
        unit=None,
        route=None,
        frequency=None,
        start_time=_START_TIME,
        stop_time=None,
        adherence=adherence,
        mechanism_ids=mechanism_ids_b,
    )

    assert check_irrelevant_pathway_label_does_not_alter_predictions(
        treatment_arm, intervention_a, intervention_b
    )
