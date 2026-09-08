"""Time-zero validation (Requirement 3.10-3.11, `design.md` Section 3.2).

`validate_time_zero(timeline, spec)` checks that every baseline-covariate/
eligibility-relevant field with a recorded observation timestamp was
observed at or before the patient's treatment-assignment ("time zero")
timestamp, per `spec.time_zero.definition`
(`configs/target_trials/main.yaml`: `"randomized_treatment_assignment"`).
Any baseline field whose timestamp is *strictly after* assignment is flagged
as a `Violation` — this is post-treatment information leaking into a
baseline feature (Requirement 3.11). Coincidence (timestamp exactly equal to
assignment time) is explicitly allowed (Requirement 3.10: "precede or
coincide with").

This function is written as a small, standalone, unit-testable function so
it can be reused unchanged both here and later inside
`src/protocols/consistency_checker.py` and `src/data/leakage_audit.py`
(`design.md` Section 3.2/3.3), and inside `src/witnesses/temporal.py`'s
no-future-leakage check (`design.md` Section 3.8).

`PatientTimeline` (from `src/protocols/eligibility_dsl.py`) is reused here
rather than waiting on the richer `src/data/timelines.py` type, per this
task's scope: it has been extended with `assignment_time` and
`baseline_covariate_timestamps` fields for exactly this purpose. When
`src/data/timelines.py` lands, its output must be adapted to (or made
compatible with) that same shape so this function does not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from protocols.eligibility_dsl import PatientTimeline
from protocols.schema import TargetTrialSpec


@dataclass(frozen=True)
class Violation:
    """A single time-zero violation: a baseline field observed after assignment.

    Attributes:
        field_name: the baseline-covariate/eligibility-relevant field whose
            observation timestamp violates time zero.
        observed_at: the timestamp at which the field was actually observed.
        assignment_time: the patient's treatment-assignment (time-zero)
            timestamp the field was compared against.
        description: a human-readable explanation of the violation, suitable
            for inclusion in an audit report or protocol-violation record.
    """

    field_name: str
    observed_at: datetime
    assignment_time: datetime
    description: str


def validate_time_zero(timeline: PatientTimeline, spec: TargetTrialSpec) -> list[Violation]:
    """Check that baseline-feature timestamps precede or coincide with assignment.

    For every entry in `timeline.baseline_covariate_timestamps`, compares the
    recorded observation timestamp against `timeline.assignment_time`. A
    timestamp strictly after assignment is a violation (post-treatment
    information leaking into a baseline feature); a timestamp at or before
    assignment is not. Every violation is reported — this function does not
    stop at the first one (a caller relying on the aggregate result needs to
    see every leaked field, not just one).

    Args:
        timeline: the patient's timeline, including `assignment_time` and
            `baseline_covariate_timestamps`.
        spec: the target-trial specification. `spec.time_zero.definition` is
            asserted to be `"randomized_treatment_assignment"` — the only
            time-zero definition this ACTG175 pipeline supports
            (`decisions.md` Section 5) — so a spec using a different
            definition fails loudly rather than being silently
            misinterpreted.

    Returns:
        A list of `Violation` records, one per baseline field observed after
        assignment. Empty when the timeline is clean, including when
        `assignment_time` is `None` (nothing to compare against) or when
        `baseline_covariate_timestamps` is empty (nothing to check).

    Raises:
        ValueError: if `spec.time_zero.definition` is not
            `"randomized_treatment_assignment"`.
    """
    if spec.time_zero.definition != "randomized_treatment_assignment":
        raise ValueError(
            "validate_time_zero only supports time_zero.definition == "
            f"'randomized_treatment_assignment'; got {spec.time_zero.definition!r}."
        )

    if timeline.assignment_time is None:
        return []

    assignment_time = timeline.assignment_time
    violations: list[Violation] = []
    for field_name, observed_at in timeline.baseline_covariate_timestamps.items():
        if observed_at > assignment_time:
            violations.append(
                Violation(
                    field_name=field_name,
                    observed_at=observed_at,
                    assignment_time=assignment_time,
                    description=(
                        f"Baseline field '{field_name}' was observed at "
                        f"{observed_at.isoformat()}, which is after the "
                        f"assignment/time-zero timestamp "
                        f"{assignment_time.isoformat()}. Post-treatment "
                        "information must not enter baseline features."
                    ),
                )
            )
    return violations
