"""Protocol-consistency checker (Requirement 3.13, `design.md` Section 3.2).

`ProtocolChecker.check(cohort, spec)` runs every deterministic protocol and
clinical-constraint check the pipeline currently owns against a list of
per-patient `PatientProtocolRecord`s, and returns a flat list of structured
`ProtocolViolation(patient_id, rule, timestamp, evidence)` records
(Requirement 3.13). This module is reused directly by the clinical witness
(`src/witnesses/clinical.py`, task 11.2, `design.md` Section 3.2/3.8) — the
checks below are not duplicated there.

Checks implemented (one `rule` name each):

- ``"ineligible"`` — the patient fails `ACTG175_ELIGIBILITY`
  (`src/protocols/eligibility_dsl.py`).
- ``"invalid_assignment"`` — `assigned_arm` is not `0` or `1`.
- ``"treatment_before_eligibility"`` — treatment started strictly before the
  point eligibility/assignment was established.
- ``"time_zero_violation"`` — a baseline field was observed after assignment
  (delegated to, and translated from, `time_zero.validate_time_zero`).
- ``"outcome_before_intervention"`` — the composite-endpoint event time
  precedes treatment start (or assignment, if no treatment-start time is
  recorded).
- ``"impossible_sequence"`` — `death_time` or `last_observed_time` precedes
  `assignment_time` on the endpoint record.
- ``"endpoint_window_violation"`` — the endpoint event occurs far beyond the
  trial's declared follow-up horizon.

Input shape
-----------
Per-patient inputs needed to run *every* check for one patient are bundled
into `PatientProtocolRecord`, a small, self-contained composition of the
existing per-module types (`PatientTimeline` from `eligibility_dsl.py`,
`PatientEndpointRecord` from `endpoints.py`) — deliberately not waiting on
`src/data/timelines.py` (`design.md` Section 3.2 / this task's scope note).
When `src/data/timelines.py` lands, its output should be adapted into a
`PatientProtocolRecord` at the call site; this module's checking logic does
not need to change.

Two assignment-time reference points are used, matching how the two
underlying types are already self-contained:

- eligibility/time-zero checks (``"ineligible"``, ``"treatment_before_eligibility"``,
  ``"time_zero_violation"``) use `eligibility_timeline.assignment_time`;
- endpoint-derived checks (``"outcome_before_intervention"``,
  ``"impossible_sequence"``, ``"endpoint_window_violation"``) use
  `endpoint_record.assignment_time`.

For a real patient these two values are expected to agree; keeping the
reference local to each bundled type avoids introducing a third,
independently-owned "canonical" timestamp that could drift from either.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from protocols.eligibility_dsl import ACTG175_ELIGIBILITY, PatientTimeline
from protocols.endpoints import CompositeEndpoint, PatientEndpointRecord
from protocols.schema import TargetTrialSpec
from protocols.time_zero import validate_time_zero

# "Far beyond the declared window" (endpoint-window violation): the trial's
# frozen secondary horizon is 96 weeks (`decisions.md` Section 5), but the
# *primary* follow-up window is "complete available trial follow-up" with no
# fixed cap, so a legitimate event can occur well past 96 weeks. This
# multiplier defines how many multiples of the secondary horizon are treated
# as clearly outside any plausible follow-up window for this trial, rather
# than flagging every event past the (uncapped) primary horizon. This is an
# engineering judgment call documented here, not a value frozen in
# `decisions.md` — it exists only to catch grossly invalid event timestamps
# (e.g. data-entry errors), not to enforce the analysis follow-up window
# itself.
ENDPOINT_WINDOW_MULTIPLIER = 2


@dataclass(frozen=True)
class ProtocolViolation:
    """A single structured protocol/clinical-constraint violation (Requirement 3.13).

    Attributes:
        patient_id: the patient (record) the violation pertains to.
        rule: the short, stable rule name that fired (e.g. `"ineligible"`,
            `"invalid_assignment"`, `"treatment_before_eligibility"`,
            `"time_zero_violation"`, `"outcome_before_intervention"`,
            `"impossible_sequence"`, `"endpoint_window_violation"`).
        timestamp: the timestamp most relevant to the violation, or `None`
            when no single timestamp applies (e.g. an invalid assignment
            value with no recorded treatment-start time).
        evidence: a rule-specific, JSON-serializable mapping of supporting
            detail (e.g. the eligibility failure reason, the two timestamps
            being compared, the offending value) — always populated so an
            audit report can explain *why* the rule fired, not just that it
            did.
    """

    patient_id: str
    rule: str
    timestamp: datetime | None
    evidence: dict


@dataclass(frozen=True)
class PatientProtocolRecord:
    """Bundles everything needed to run every protocol check for one patient.

    A simple composition of existing per-module types — no new patient-data
    representation is introduced.

    Attributes:
        patient_id: the patient identifier.
        eligibility_timeline: the patient's `PatientTimeline`
            (`eligibility_dsl.py`), used for the eligibility and time-zero
            checks. Its `assignment_time` is the reference point for
            eligibility/time-zero-related checks.
        assigned_arm: the patient's assigned treatment arm. Valid values are
            `0` (comparator) and `1` (intervention); any other value
            (including `None`) is an invalid assignment.
        endpoint_record: the patient's `PatientEndpointRecord`
            (`endpoints.py`), used to compute the composite-endpoint event
            time for the endpoint-derived checks. `None` if endpoint data is
            unavailable for this patient (endpoint-derived checks are then
            skipped for that patient — this is not itself flagged as a
            violation by this checker).
        treatment_start_time: the timestamp treatment actually started, or
            `None` if unavailable (in which case
            `"treatment_before_eligibility"` and
            `"outcome_before_intervention"` fall back to comparing against
            `eligibility_timeline.assignment_time` where noted, or are
            skipped where a treatment-start time is required and absent).
    """

    patient_id: str
    eligibility_timeline: PatientTimeline
    assigned_arm: int | None
    endpoint_record: PatientEndpointRecord | None
    treatment_start_time: datetime | None


class ProtocolChecker:
    """Runs every deterministic protocol/clinical-constraint check on a cohort."""

    @staticmethod
    def check(
        cohort: list[PatientProtocolRecord], spec: TargetTrialSpec
    ) -> list[ProtocolViolation]:
        """Check every patient in `cohort` against every protocol rule.

        Every applicable rule is checked for every patient (a patient may
        accumulate more than one violation); no check short-circuits the
        others.

        Args:
            cohort: the patients to check.
            spec: the target-trial specification, used for the time-zero
                check (`spec.time_zero.definition`) and the endpoint-window
                check (`spec.follow_up.secondary_horizon_weeks`).

        Returns:
            A flat list of `ProtocolViolation` records, in cohort order and,
            within a patient, in the rule order listed in the module
            docstring. Empty when no patient has any violation.
        """
        violations: list[ProtocolViolation] = []
        for record in cohort:
            violations.extend(ProtocolChecker._check_patient(record, spec))
        return violations

    @staticmethod
    def _check_patient(
        record: PatientProtocolRecord, spec: TargetTrialSpec
    ) -> list[ProtocolViolation]:
        violations: list[ProtocolViolation] = []

        violations.extend(ProtocolChecker._check_eligibility(record))
        violations.extend(ProtocolChecker._check_assignment(record))
        violations.extend(ProtocolChecker._check_treatment_before_eligibility(record))
        violations.extend(ProtocolChecker._check_time_zero(record, spec))
        violations.extend(ProtocolChecker._check_outcome_before_intervention(record))
        violations.extend(ProtocolChecker._check_impossible_sequence(record))
        violations.extend(ProtocolChecker._check_endpoint_window(record, spec))

        return violations

    # -- ineligible ---------------------------------------------------------

    @staticmethod
    def _check_eligibility(record: PatientProtocolRecord) -> list[ProtocolViolation]:
        result = ACTG175_ELIGIBILITY.evaluate(record.eligibility_timeline)
        if result.eligible:
            return []
        return [
            ProtocolViolation(
                patient_id=record.patient_id,
                rule="ineligible",
                timestamp=record.eligibility_timeline.assignment_time,
                evidence={
                    "reason": result.reason,
                    "missing_fields": list(result.missing_fields),
                },
            )
        ]

    # -- invalid_assignment ---------------------------------------------------

    @staticmethod
    def _check_assignment(record: PatientProtocolRecord) -> list[ProtocolViolation]:
        if record.assigned_arm in (0, 1):
            return []
        return [
            ProtocolViolation(
                patient_id=record.patient_id,
                rule="invalid_assignment",
                timestamp=record.treatment_start_time,
                evidence={"assigned_arm": record.assigned_arm},
            )
        ]

    # -- treatment_before_eligibility ---------------------------------------

    @staticmethod
    def _check_treatment_before_eligibility(
        record: PatientProtocolRecord,
    ) -> list[ProtocolViolation]:
        assignment_time = record.eligibility_timeline.assignment_time
        treatment_start_time = record.treatment_start_time
        if assignment_time is None or treatment_start_time is None:
            return []
        if treatment_start_time >= assignment_time:
            return []
        return [
            ProtocolViolation(
                patient_id=record.patient_id,
                rule="treatment_before_eligibility",
                timestamp=treatment_start_time,
                evidence={
                    "treatment_start_time": treatment_start_time,
                    "assignment_time": assignment_time,
                },
            )
        ]

    # -- time_zero_violation (delegates to time_zero.validate_time_zero) ----

    @staticmethod
    def _check_time_zero(
        record: PatientProtocolRecord, spec: TargetTrialSpec
    ) -> list[ProtocolViolation]:
        time_zero_violations = validate_time_zero(record.eligibility_timeline, spec)
        return [
            ProtocolViolation(
                patient_id=record.patient_id,
                rule="time_zero_violation",
                timestamp=violation.observed_at,
                evidence={
                    "field_name": violation.field_name,
                    "observed_at": violation.observed_at,
                    "assignment_time": violation.assignment_time,
                    "description": violation.description,
                },
            )
            for violation in time_zero_violations
        ]

    # -- outcome_before_intervention -----------------------------------------

    @staticmethod
    def _check_outcome_before_intervention(
        record: PatientProtocolRecord,
    ) -> list[ProtocolViolation]:
        if record.endpoint_record is None:
            return []

        reference_time = record.treatment_start_time
        reference_label = "treatment_start_time"
        if reference_time is None:
            reference_time = record.endpoint_record.assignment_time
            reference_label = "assignment_time"
        if reference_time is None:
            return []

        endpoint_result = CompositeEndpoint.compute(record.endpoint_record)
        if endpoint_result.event_time >= reference_time:
            return []

        return [
            ProtocolViolation(
                patient_id=record.patient_id,
                rule="outcome_before_intervention",
                timestamp=endpoint_result.event_time,
                evidence={
                    "event_time": endpoint_result.event_time,
                    "event_type": endpoint_result.event_type,
                    "reference_time": reference_time,
                    "reference_field": reference_label,
                },
            )
        ]

    # -- impossible_sequence -------------------------------------------------

    @staticmethod
    def _check_impossible_sequence(
        record: PatientProtocolRecord,
    ) -> list[ProtocolViolation]:
        endpoint_record = record.endpoint_record
        if endpoint_record is None:
            return []

        assignment_time = endpoint_record.assignment_time
        if assignment_time is None:
            return []

        violations: list[ProtocolViolation] = []

        if (
            endpoint_record.death_time is not None
            and endpoint_record.death_time < assignment_time
        ):
            violations.append(
                ProtocolViolation(
                    patient_id=record.patient_id,
                    rule="impossible_sequence",
                    timestamp=endpoint_record.death_time,
                    evidence={
                        "reason": "death_time_before_assignment_time",
                        "death_time": endpoint_record.death_time,
                        "assignment_time": assignment_time,
                    },
                )
            )

        if endpoint_record.last_observed_time < assignment_time:
            violations.append(
                ProtocolViolation(
                    patient_id=record.patient_id,
                    rule="impossible_sequence",
                    timestamp=endpoint_record.last_observed_time,
                    evidence={
                        "reason": "last_observed_time_before_assignment_time",
                        "last_observed_time": endpoint_record.last_observed_time,
                        "assignment_time": assignment_time,
                    },
                )
            )

        return violations

    # -- endpoint_window_violation --------------------------------------------

    @staticmethod
    def _check_endpoint_window(
        record: PatientProtocolRecord, spec: TargetTrialSpec
    ) -> list[ProtocolViolation]:
        endpoint_record = record.endpoint_record
        if endpoint_record is None:
            return []

        assignment_time = endpoint_record.assignment_time
        if assignment_time is None:
            return []

        endpoint_result = CompositeEndpoint.compute(endpoint_record)
        elapsed = endpoint_result.event_time - assignment_time
        if elapsed <= timedelta(0):
            # Already flagged (or would be flagged) by impossible_sequence /
            # outcome_before_intervention; the endpoint-window check only
            # concerns events too far in the future, not in the past.
            return []

        max_allowed = timedelta(
            weeks=ENDPOINT_WINDOW_MULTIPLIER * spec.follow_up.secondary_horizon_weeks
        )
        if elapsed <= max_allowed:
            return []

        return [
            ProtocolViolation(
                patient_id=record.patient_id,
                rule="endpoint_window_violation",
                timestamp=endpoint_result.event_time,
                evidence={
                    "event_time": endpoint_result.event_time,
                    "assignment_time": assignment_time,
                    "elapsed_weeks": elapsed.days / 7,
                    "max_allowed_weeks": (
                        ENDPOINT_WINDOW_MULTIPLIER * spec.follow_up.secondary_horizon_weeks
                    ),
                },
            )
        ]
