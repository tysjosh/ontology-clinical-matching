"""Property-based intervention-invariance tests, parameterized over the
pure-Python prototype and the production torch model (Requirement 9.9 of
the parent spec; Requirements 3.4/3.6 and Correctness Property 12 of
`neural-experiment-pipeline`).

`design.md` Section "Reuse of the existing invariance properties against
the torch models (Property 12)": "`tests/property/test_invariance.py` is
refactored so its three properties are written once against a minimal
protocol (`rollout_both_arms`-shaped) and parameterized over
`["prototype", "torch"]`, with the torch case wrapped in
`pytest.importorskip("torch")`. The prototype case's behavior and
assertions are unchanged, so those tests cannot regress. This is the only
place where the prototypes and the production models are held to the same
statement, which is precisely why the prototypes are retained."

The protocol
-------------------------------------------------------------------------

`InterventionRollout` below is that minimal protocol. It is
`rollout_both_arms`-shaped in the literal sense: its central operation is
`rollout_both_arms(hidden_state, observed_treatment_arm) -> (branch_1,
branch_0)`, the arm pair of `models.branch_heads.evaluate_both_branches`
and the per-patient analogue of
`models.torch_intervention_aware.TorchInterventionAwareModel
.rollout_both_arms`. Around it sit the three members the properties need:

- `rollout(z_sequence, initial_hidden_state=None)` -- one hidden state per
  landmark, which is the *only* member
  `check_future_treatment_does_not_affect_earlier_state` touches on the
  object it is handed, so both subjects satisfy that check unchanged;
- `branch_1_head` / `branch_0_head` -- the two treatment-specific heads,
  each exposing `predict(hidden_state) -> (event_hazard, next_cd4,
  next_cd8)`. `evaluate_both_branches` reads nothing else off a head, so
  `rollout_both_arms` is one implementation shared by both subjects and
  the two remaining `check_*` functions run against either subject
  without modification.

Why the properties are expressed against per-patient hidden states rather
than against `TorchInterventionAwareModel.rollout_both_arms(batch)`
directly: the prototype parameterization's generated inputs and
assertions must not change (Property 12's "cannot regress" clause), and
those inputs are a `z` sequence and a hidden-state vector. The torch
subject therefore adapts the production model's *own* submodules -- its
`gru_cell` for the recurrence, its `hazard_head_arm1`/`hazard_head_arm0`
and shared `state_head` for the arm pair -- to that shape. Two
consequences worth stating plainly:

- Property 1 (future treatment does not affect earlier state) cannot be
  posed against the torch model's `rollout_both_arms`, because that
  method holds `z` constant across the three landmarks by design (see
  `torch_intervention_aware`'s module docstring). Driving the model's own
  `gru_cell` over a landmark-varying `z` sequence is the only way to pose
  the temporal statement at all, and it is the same recurrence
  `_rollout` runs.
- The batch-level counterparts -- object-identity of the shared inputs
  across the two arms, and channel-width invariance across the ablation
  modes -- are Correctness Properties 13 and 14, covered in
  `tests/neural/test_intervention_channels.py`. This module is
  deliberately not a second copy of those.

Torch-free collection (Property 12's second clause)
-------------------------------------------------------------------------

`tests/property/` is a torch-free collection root. This module imports no
torch at module scope: the `torch` parameterization's
`pytest.importorskip("torch")` and every torch import live inside
`_build_subject("torch")` and the torch adapter's method bodies, reached
only through a module-scoped fixture. In an environment without torch the
module still imports, the `prototype` parameterization still passes, and
only the `torch` parameterization skips. `tests/unit/test_torch_isolation
.py`'s subprocess harness is the model for verifying that.

A fourth property (irrelevant pathway-label changes do not alter
predictions) is prototype-only and unparameterized: it is a statement
about `models.intervention.Intervention.mechanism_ids` flowing through
`intervention_features_from`, not about a rollout, so it has no torch
counterpart to hold to the same statement. It is left exactly as it was.

**Validates: Requirements 9.9, 3.4, 3.6**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models.branch_heads import BranchHead, BranchOutput, evaluate_both_branches
from models.intervention import Intervention
from models.intervention_aware import GRUTransition
from models.invariance_tests import (
    check_changing_treatment_changes_counterfactual_output,
    check_future_treatment_does_not_affect_earlier_state,
    check_irrelevant_pathway_label_does_not_alter_predictions,
    check_unchanged_treatment_preserves_factual_output,
)
from models.sequence_baseline import N_LANDMARKS

_MAX_EXAMPLES = 30

# Fixed, small dimensions -- keep the pure-Python forward passes fast.
_INPUT_SIZE = 3
_HIDDEN_SIZE = 4

#: Seeds the prototype subject initializes with. Unchanged from before the
#: parameterization, so the prototype case draws the same components.
_GRU_SEED = 42
_BRANCH_1_SEED = 1
_BRANCH_0_SEED = 2

#: The two parameterizations of every property below.
_SUBJECT_NAMES = ("prototype", "torch")

_finite_float = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)


def _vector(width: int):
    """A `width`-long vector of finite floats.

    The prototype case draws `_INPUT_SIZE`/`_HIDDEN_SIZE`-wide vectors from
    this, i.e. exactly the strategy it drew before; the torch case draws
    its model's own representation and hidden widths.
    """
    return st.lists(_finite_float, min_size=width, max_size=width)


# ---------------------------------------------------------------------------
# The minimal `rollout_both_arms`-shaped protocol
# ---------------------------------------------------------------------------


class BranchHeadLike(Protocol):
    """One treatment-specific head, as `evaluate_both_branches` sees it.

    `evaluate_both_branches` reads exactly one member off each head it is
    given -- `predict` -- so this is the whole surface an implementation
    has to supply for `rollout_both_arms` to work against it.
    """

    hidden_size: int

    def predict(self, hidden_state: Any) -> tuple[float, float, float]:
        """`(event_hazard, next_cd4, next_cd8)` for one hidden state."""
        ...


@runtime_checkable
class InterventionRollout(Protocol):
    """What the three invariance properties need from an implementation.

    Deliberately minimal: a landmark rollout, the two branch heads, and
    the arm pair built from them. Nothing here mentions tensors, encoders,
    channels, or batches, which is what lets one statement bind both the
    pure-Python prototype and the production torch model.
    """

    name: str
    input_size: int
    hidden_size: int
    landmark_count: int
    branch_1_head: BranchHeadLike
    branch_0_head: BranchHeadLike

    def rollout(
        self, z_sequence: Any, initial_hidden_state: Any = None
    ) -> list[Any]:
        """One hidden state per element of `z_sequence`."""
        ...

    def rollout_both_arms(
        self, hidden_state: Any, observed_treatment_arm: int
    ) -> tuple[BranchOutput, BranchOutput]:
        """`(branch_1_output, branch_0_output)` for one hidden state."""
        ...


class _ArmPair:
    """`rollout_both_arms`, implemented once for every subject.

    Both potential outcomes come from the same hidden state and the same
    two heads (`decisions.md` Section 14.3: "For every eligible patient,
    both branches are evaluated"), and exactly one carries the `"factual"`
    tag. This is the identical call
    `check_unchanged_treatment_preserves_factual_output` and
    `check_changing_treatment_changes_counterfactual_output` make
    internally, which is why those two checks need no adaptation per
    subject.
    """

    branch_1_head: BranchHeadLike
    branch_0_head: BranchHeadLike

    def rollout_both_arms(
        self, hidden_state: Any, observed_treatment_arm: int
    ) -> tuple[BranchOutput, BranchOutput]:
        return evaluate_both_branches(
            hidden_state, self.branch_1_head, self.branch_0_head, observed_treatment_arm
        )


# ---------------------------------------------------------------------------
# Subject 1: the pure-Python prototype (torch-free)
# ---------------------------------------------------------------------------


@dataclass
class PrototypeRollout(_ArmPair):
    """`models.intervention_aware.GRUTransition` + `models.branch_heads`.

    The components, widths, and seeds are exactly the ones the three
    properties constructed inline before this refactor, so the prototype
    parameterization drives the same objects it always did.
    """

    gru_transition: GRUTransition
    branch_1_head: BranchHead
    branch_0_head: BranchHead

    name: str = "prototype"

    @classmethod
    def build(cls) -> PrototypeRollout:
        return cls(
            gru_transition=GRUTransition.initialize(
                input_size=_INPUT_SIZE, hidden_size=_HIDDEN_SIZE, seed=_GRU_SEED
            ),
            branch_1_head=BranchHead.initialize(
                hidden_size=_HIDDEN_SIZE, seed=_BRANCH_1_SEED
            ),
            branch_0_head=BranchHead.initialize(
                hidden_size=_HIDDEN_SIZE, seed=_BRANCH_0_SEED
            ),
        )

    @property
    def input_size(self) -> int:
        return self.gru_transition.input_size

    @property
    def hidden_size(self) -> int:
        return self.gru_transition.hidden_size

    @property
    def landmark_count(self) -> int:
        return N_LANDMARKS

    def rollout(
        self, z_sequence: Any, initial_hidden_state: Any = None
    ) -> list[list[float]]:
        return self.gru_transition.rollout(z_sequence, initial_hidden_state)


# ---------------------------------------------------------------------------
# Subject 2: the production torch model (built lazily, behind importorskip)
# ---------------------------------------------------------------------------


@dataclass
class TorchBranchHead:
    """One arm's head of `TorchInterventionAwareModel`, in prototype shape.

    `predict` returns the prototype's `(event_hazard, next_cd4, next_cd8)`
    triple from the production model's own submodules:

    - `event_hazard` is `outcome_from_hazards(sigmoid(hazard_head(h)))` --
      the hazard-derived 96-week event probability, which is the model's
      *only* route to a scalar outcome (Requirement 3.5, and the reason
      the module owns no outcome head). The prototype's single
      `event_hazard` scalar has no other faithful counterpart: the torch
      head emits one hazard per survival bin.
    - `next_cd4` / `next_cd8` come from the shared `state_head`, the
      analogue of the prototype's CD4/CD8 outputs.

    `torch` is imported inside `predict`, never at module scope, so this
    class's mere definition keeps `tests/property/` torch-free.
    """

    hidden_size: int
    hazard_head: Any
    state_head: Any
    outcome_from_hazards: Any

    def predict(self, hidden_state: Any) -> tuple[float, float, float]:
        """`(event_hazard, next_cd4, next_cd8)` for one hidden state.

        Raises `ValueError` on a hidden state of the wrong width, mirroring
        `BranchHead.predict`'s check so both subjects reject the same way.
        """
        import torch

        if len(hidden_state) != self.hidden_size:
            raise ValueError(
                f"expected hidden_state of length {self.hidden_size}, got {len(hidden_state)}"
            )
        with torch.no_grad():
            row = torch.tensor(
                [[float(value) for value in hidden_state]], dtype=torch.float32
            )
            hazards = torch.sigmoid(self.hazard_head(row))
            event_hazard = float(self.outcome_from_hazards(hazards)[0])
            next_cd4, next_cd8 = (float(value) for value in self.state_head(row)[0])
        return event_hazard, next_cd4, next_cd8


@dataclass
class TorchRollout(_ArmPair):
    """`models.torch_intervention_aware.TorchInterventionAwareModel`.

    `rollout` drives the model's own `gru_cell` over the supplied `z`
    sequence -- the same recurrence `TorchInterventionAwareModel._rollout`
    runs, one `gru_cell` call per landmark from a zero initial hidden
    state. The production `rollout_both_arms(batch)` cannot be used here
    because it holds `z` constant across landmarks by design, so a
    landmark-varying perturbation is not expressible through it; see this
    module's docstring.
    """

    model: Any
    branch_1_head: TorchBranchHead
    branch_0_head: TorchBranchHead

    name: str = "torch"

    @property
    def input_size(self) -> int:
        """`len(z) = 4 + 3 + 2`, the model's concatenated representation."""
        return self.model.representation_width

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

    @property
    def landmark_count(self) -> int:
        return self.model.landmark_count

    def rollout(
        self, z_sequence: Any, initial_hidden_state: Any = None
    ) -> list[list[float]]:
        import torch

        with torch.no_grad():
            z = torch.tensor(
                [[float(value) for value in z_t] for z_t in z_sequence],
                dtype=torch.float32,
            )
            if initial_hidden_state is None:
                hidden = torch.zeros(1, self.hidden_size, dtype=torch.float32)
            else:
                hidden = torch.tensor(
                    [[float(value) for value in initial_hidden_state]],
                    dtype=torch.float32,
                )
            hidden_states: list[list[float]] = []
            for index in range(z.shape[0]):
                hidden = self.model.gru_cell(z[index : index + 1], hidden)
                hidden_states.append([float(value) for value in hidden[0]])
        return hidden_states


def _build_torch_subject() -> TorchRollout:
    """Construct the torch subject, skipping the parameterization if torch
    is absent.

    Every torch-touching import lives here rather than at module scope,
    which is what keeps this module importable -- and its `prototype`
    parameterization passing -- in a torch-free environment (Property 12).
    """
    pytest.importorskip("torch")

    import torch

    from models.dataset import PatientDatasetBuilder
    from models.torch_intervention_aware import (
        TorchInterventionAwareModel,
        outcome_from_hazards,
    )

    # Parameter initialization is the only randomness in the module, and the
    # invariances must hold for whatever weights it produces; seeding makes a
    # failure reproducible rather than seed-dependent.
    torch.manual_seed(_GRU_SEED)
    model = TorchInterventionAwareModel(PatientDatasetBuilder().dimensions)
    model.eval()

    def head(hazard_head: Any) -> TorchBranchHead:
        return TorchBranchHead(
            hidden_size=model.hidden_size,
            hazard_head=hazard_head,
            state_head=model.state_head,
            outcome_from_hazards=outcome_from_hazards,
        )

    return TorchRollout(
        model=model,
        branch_1_head=head(model.hazard_head_arm1),
        branch_0_head=head(model.hazard_head_arm0),
    )


def _build_subject(name: str) -> InterventionRollout:
    if name == "prototype":
        return PrototypeRollout.build()
    if name == "torch":
        return _build_torch_subject()
    raise AssertionError(f"unknown subject {name!r}; expected one of {_SUBJECT_NAMES}")


@pytest.fixture(scope="module", params=_SUBJECT_NAMES)
def subject(request) -> InterventionRollout:
    """The subject under test, once per parameterization.

    Module-scoped for two reasons: `hypothesis` rejects function-scoped
    fixtures in `@given` tests (`HealthCheck.function_scoped_fixture`), and
    both subjects are deterministic and stateless, so one instance serves
    every example. `pytest.importorskip("torch")` runs here, before
    `hypothesis` starts, so the `torch` parameterization skips cleanly and
    the `prototype` parameterization is untouched.
    """
    return _build_subject(request.param)


# ---------------------------------------------------------------------------
# Property 1: future treatment does not affect earlier state
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_future_treatment_does_not_affect_earlier_state(subject, data):
    """For a 3-landmark rollout (baseline, week 20, week 96), changing ONLY
    the LAST timestep's (week 96's) z-vector -- which carries the
    intervention channel's content -- must not change the hidden states
    produced at the earlier two timesteps. This follows directly from the
    GRU recurrence's causal structure: each `h_t` depends only on
    `h_{t-1}` and `z_t`, never on future `z`.

    Written once and checked against both subjects: the prototype's
    `GRUTransition.rollout` and the production model's own `gru_cell`
    recurrence. `check_future_treatment_does_not_affect_earlier_state`
    reads only `.rollout` off the object it is handed, so the check itself
    is unchanged from before the parameterization."""
    z_sequence = [
        data.draw(_vector(subject.input_size), label=f"z{index}")
        for index in range(subject.landmark_count)
    ]
    perturbed_last_z = data.draw(_vector(subject.input_size), label="perturbed_last_z")

    assert check_future_treatment_does_not_affect_earlier_state(
        subject, z_sequence, perturbed_last_z
    )


# ---------------------------------------------------------------------------
# Property 2: unchanged treatment preserves factual output
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(data=st.data(), observed_treatment_arm=st.sampled_from([0, 1]))
def test_unchanged_treatment_preserves_factual_output(
    subject, data, observed_treatment_arm
):
    """Repeated evaluation of `rollout_both_arms` with an unchanged hidden
    state and an unchanged `observed_treatment_arm` must produce an
    identical factual-tagged output both times (determinism /
    preservation of the factual computation).

    Both subjects reach this through the same
    `evaluate_both_branches(hidden_state, branch_1_head, branch_0_head,
    observed_treatment_arm)` call; only what the two heads are differs."""
    hidden_state = data.draw(_vector(subject.hidden_size), label="hidden_state")

    assert check_unchanged_treatment_preserves_factual_output(
        hidden_state,
        subject.branch_1_head,
        subject.branch_0_head,
        observed_treatment_arm,
    )


# ---------------------------------------------------------------------------
# Property 3: changing treatment changes counterfactual output
# ---------------------------------------------------------------------------


@settings(max_examples=_MAX_EXAMPLES, deadline=None)
@given(data=st.data())
def test_changing_treatment_changes_counterfactual_output(subject, data):
    """For a FIXED hidden state, flipping `observed_treatment_arm` swaps
    which branch is tagged `"counterfactual"` vs. `"factual"`, while the
    underlying numeric predictions from each branch head never depend on
    which arm is "observed" -- only the tag does.

    Note what this does *not* claim: nothing here requires the two arms to
    agree with each other. Under the torch model's default
    `InterventionMode.LEARNED` the two arms' representations and
    predictions differ, which is the normal case; the statement is only
    that each arm's own prediction is independent of the observed-arm
    label."""
    hidden_state = data.draw(_vector(subject.hidden_size), label="hidden_state")

    assert check_changing_treatment_changes_counterfactual_output(
        hidden_state, subject.branch_1_head, subject.branch_0_head
    )


# ---------------------------------------------------------------------------
# Both parameterizations really do satisfy the protocol
# ---------------------------------------------------------------------------


def test_subject_satisfies_the_rollout_both_arms_protocol(subject):
    """Structural guard on the parameterization itself.

    Without this, a subject missing a member would surface as an
    `AttributeError` deep inside a `hypothesis` example, and a subject
    whose `rollout_both_arms` tagged both arms factual would go unnoticed
    because the three properties above reach the arm pair through the
    `check_*` functions. Asserted for both parameterizations, so the
    prototype and the production model are held to the same shape as well
    as the same statements."""
    assert isinstance(subject, InterventionRollout)
    assert subject.name in _SUBJECT_NAMES
    assert subject.input_size > 0
    assert subject.hidden_size > 0
    assert subject.landmark_count == N_LANDMARKS

    hidden_states = subject.rollout([[0.0] * subject.input_size] * subject.landmark_count)
    assert len(hidden_states) == subject.landmark_count
    assert all(len(state) == subject.hidden_size for state in hidden_states)

    for observed_treatment_arm in (0, 1):
        branch_1, branch_0 = subject.rollout_both_arms(
            hidden_states[-1], observed_treatment_arm
        )
        tags = [branch_1.tag, branch_0.tag]
        assert tags.count("factual") == 1, tags
        assert tags.count("counterfactual") == 1, tags


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
    feature vector regardless of `mechanism_ids`'s value.

    Prototype-only and unparameterized: this is a statement about the
    intervention *feature construction*, not about a rollout, so there is
    no torch counterpart to hold to the same statement (the torch model
    reads an already-built intervention channel off a `TensorBatch`)."""
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
