"""Temporal witness (Requirement 11.1, `design.md` Section 3.8, `decisions.md`
Section 16, "Temporal witness -- passes when").

`decisions.md` Section 16 freezes the temporal witness's *passing* condition
as the conjunction of five checks (copied verbatim, not re-derived):

    - all baseline features precede treatment;
    - outcomes occur after time zero;
    - state transitions preserve chronological ordering;
    - no future information enters any prediction;
    - held-out longitudinal (CD4/CD8) error is no worse than 5% above the
      factual GRU baseline (Section 14.2).

`decisions.md` does not spell out an explicit fail/uncertain distinction for
*this* witness the way it does for the causal witness (Section 16, "Causal
witness -- pass / fail / uncertain"). This module documents its own
interpretation, used nowhere else as a frozen decision:

    - **pass**: all five conditions hold.
    - **fail**: any *hard* violation is found -- a `time_zero.Violation`
      (baseline-precedes-treatment / no-future-leakage), an outcome recorded
      before time zero, or a non-chronological landmark-timestamp sequence.
      A hard violation is definite evidence of a temporal-consistency defect,
      so it takes priority over "uncertain" even if some other input is also
      missing.
    - **uncertain**: no hard violation was found, but a required input for a
      condition -- specifically the held-out/factual-baseline longitudinal
      MAE values -- was not supplied, so that condition could not be
      evaluated cleanly as pass or fail.

Design choices
--------------

- **Decoupled inputs, not raw cohort recomputation.** Mirroring
  `uncertainty.support_gate.PatientSupportGate` (`design.md` Section 3.7):
  this witness accepts already-derived per-patient signals (timelines,
  specs, event times, assignment times, landmark timestamps, MAE values)
  rather than reaching into `src/cohorts/cohort_construction.py` or
  `src/models/sequence_baseline.py` and recomputing everything itself. The
  caller is responsible for assembling these signals from whichever
  pipeline stage owns them.
- **Baseline-precedes-treatment / no-future-leakage reuse.**
  `src/protocols/time_zero.py::validate_time_zero` already checks that
  every baseline-covariate timestamp precedes or coincides with
  assignment/time-zero, which is exactly "all baseline features precede
  treatment" and is also *the* concrete, testable form of "no future
  information enters any prediction" available at this stage of the
  pipeline (rollout-level future-leakage checks belong to the
  intervention-invariance property tests in
  `src/models/invariance_tests.py`, not here). Rather than build a second,
  parallel leakage-detection mechanism, this module treats "no future
  information enters any prediction" as an alias of the
  `validate_time_zero` check and folds both into a single aggregate
  sub-result (`baseline_precedes_treatment_and_no_future_leakage`).
- **`WitnessResult` lives here.** `witnesses/` is a new package as of this
  task; `temporal.py` is its first module. `WitnessResult` is defined here
  (matching `design.md`'s Data Models section exactly: `witness`, `status`,
  `evidence`) and is meant to be imported from `witnesses.temporal` by the
  later `causal.py` / `clinical.py` / `aggregation.py` modules, rather than
  duplicated or moved to a new `witnesses/types.py`. This avoids an empty
  shared-types module for a three-field dataclass and keeps a single
  canonical definition; if a future task finds the cross-import from
  `witnesses.causal` back to `witnesses.temporal` awkward, that is a
  reasonable point to introduce `witnesses/types.py` and move it there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from protocols.eligibility_dsl import PatientTimeline
from protocols.schema import TargetTrialSpec
from protocols.time_zero import validate_time_zero

#: Frozen tolerance for the held-out longitudinal (CD4/CD8) error check
#: (`decisions.md` Section 16): held-out MAE may be no more than 5% worse
#: than the factual GRU baseline's MAE.
LONGITUDINAL_ERROR_TOLERANCE = 1.05


@dataclass(frozen=True)
class WitnessResult:
    """The outcome of one witness's evaluation (`design.md` Data Models).

    Attributes:
        witness: which witness produced this result.
        status: `"pass"`, `"fail"`, or `"uncertain"` (Requirement 11.8).
        evidence: rule-level evidence supporting `status`, always populated
            (mirrors `SupportDecision.reasons` always being populated) so an
            audit report can show *why*, not just *what*.
    """

    witness: Literal["temporal", "causal", "clinical"]
    status: Literal["pass", "fail", "uncertain"]
    evidence: dict


class TemporalWitness:
    """Evaluates the frozen five-part temporal-witness condition.

    Stateless -- `evaluate()` is a `@staticmethod` to mirror the
    `SomeThing.evaluate(...)` convention used elsewhere (e.g.
    `causal.overlap.OverlapDiagnostics.compute()`,
    `uncertainty.support_gate.PatientSupportGate.evaluate()`).
    """

    @staticmethod
    def evaluate(
        timelines_and_specs: Sequence[tuple[PatientTimeline, TargetTrialSpec]],
        event_times: Sequence[datetime],
        assignment_times: Sequence[datetime],
        landmark_timestamps: Sequence[Sequence[datetime]],
        held_out_cd4_cd8_mae: float | None = None,
        factual_baseline_cd4_cd8_mae: float | None = None,
    ) -> WitnessResult:
        """Evaluate the temporal witness for one cohort/evaluation batch.

        Args:
            timelines_and_specs: one `(timeline, spec)` pair per patient,
                passed straight through to
                `protocols.time_zero.validate_time_zero` to check
                baseline-precedes-treatment / no-future-leakage. Empty means
                nothing to check for this condition (vacuously satisfied),
                matching `validate_time_zero`'s own empty-input convention.
            event_times: per-patient outcome/endpoint event times, paired by
                index with `assignment_times`, for the
                outcome-after-time-zero check.
            assignment_times: per-patient assignment/time-zero timestamps,
                paired by index with `event_times`. Must be the same length
                as `event_times`.
            landmark_timestamps: one sequence of landmark timestamps per
                patient (e.g. `[baseline, week_20, week_96]`), for the
                chronological-ordering-preservation check. Each patient's
                sequence must be non-decreasing to pass.
            held_out_cd4_cd8_mae: the intervention-aware model's held-out
                CD4/CD8 mean absolute error. `None` if not yet available,
                in which case the longitudinal-error condition cannot be
                evaluated cleanly.
            factual_baseline_cd4_cd8_mae: the factual GRU baseline's CD4/CD8
                mean absolute error (`decisions.md` Section 14.2), used as
                the comparison point. `None` if not yet available.

        Returns:
            A `WitnessResult` with `witness="temporal"`, `status` per this
            module's documented pass/fail/uncertain rule, and `evidence`
            detailing every sub-check.

        Raises:
            ValueError: if `event_times` and `assignment_times` differ in
                length -- the caller has mismatched, unpaired sequences,
                which is a caller bug, not a domain outcome.
        """
        if len(event_times) != len(assignment_times):
            raise ValueError(
                "event_times and assignment_times must be the same length "
                f"(got {len(event_times)} and {len(assignment_times)})."
            )

        evidence: dict = {}

        # 1 & 4. Baseline-precedes-treatment, aliased with no-future-leakage.
        all_violations = []
        for timeline, spec in timelines_and_specs:
            all_violations.extend(validate_time_zero(timeline, spec))
        baseline_precedes_treatment_ok = len(all_violations) == 0
        evidence["baseline_precedes_treatment_and_no_future_leakage"] = {
            "pass": baseline_precedes_treatment_ok,
            "violation_count": len(all_violations),
            "violations": [
                {
                    "field_name": violation.field_name,
                    "observed_at": violation.observed_at.isoformat(),
                    "assignment_time": violation.assignment_time.isoformat(),
                }
                for violation in all_violations
            ],
            "note": (
                "'no future information enters any prediction' is treated "
                "as an alias of this same time_zero check at this stage of "
                "the pipeline; see module docstring."
            ),
        }

        # 2. Outcomes occur after time zero.
        outcome_violations = [
            index
            for index, (event_time, assignment_time) in enumerate(
                zip(event_times, assignment_times, strict=True)
            )
            if event_time < assignment_time
        ]
        outcomes_after_time_zero_ok = len(outcome_violations) == 0
        evidence["outcomes_after_time_zero"] = {
            "pass": outcomes_after_time_zero_ok,
            "violation_count": len(outcome_violations),
            "violating_indices": outcome_violations,
        }

        # 3. State transitions preserve chronological ordering.
        ordering_violations = [
            index
            for index, patient_landmarks in enumerate(landmark_timestamps)
            if any(
                earlier > later
                for earlier, later in zip(
                    patient_landmarks, patient_landmarks[1:], strict=False
                )
            )
        ]
        chronological_ordering_ok = len(ordering_violations) == 0
        evidence["chronological_ordering_preserved"] = {
            "pass": chronological_ordering_ok,
            "violation_count": len(ordering_violations),
            "violating_indices": ordering_violations,
        }

        # 5. Held-out longitudinal (CD4/CD8) error <= 5% worse than baseline.
        longitudinal_error_inputs_available = (
            held_out_cd4_cd8_mae is not None and factual_baseline_cd4_cd8_mae is not None
        )
        if longitudinal_error_inputs_available:
            longitudinal_error_ok = held_out_cd4_cd8_mae <= (
                factual_baseline_cd4_cd8_mae * LONGITUDINAL_ERROR_TOLERANCE
            )
        else:
            longitudinal_error_ok = None
        evidence["longitudinal_error_within_tolerance"] = {
            "pass": longitudinal_error_ok,
            "held_out_cd4_cd8_mae": held_out_cd4_cd8_mae,
            "factual_baseline_cd4_cd8_mae": factual_baseline_cd4_cd8_mae,
            "tolerance_multiplier": LONGITUDINAL_ERROR_TOLERANCE,
            "inputs_available": longitudinal_error_inputs_available,
        }

        hard_violation = not (
            baseline_precedes_treatment_ok
            and outcomes_after_time_zero_ok
            and chronological_ordering_ok
        )

        if hard_violation:
            status: Literal["pass", "fail", "uncertain"] = "fail"
        elif not longitudinal_error_inputs_available:
            status = "uncertain"
        elif longitudinal_error_ok:
            status = "pass"
        else:
            status = "fail"

        if hard_violation:
            status_rationale = "fail: hard violation found"
        elif not longitudinal_error_inputs_available:
            status_rationale = "uncertain: longitudinal-error inputs unavailable"
        elif longitudinal_error_ok:
            status_rationale = "pass: all five conditions hold"
        else:
            status_rationale = "fail: longitudinal error exceeds tolerance"
        evidence["status_rationale"] = status_rationale

        return WitnessResult(witness="temporal", status=status, evidence=evidence)
