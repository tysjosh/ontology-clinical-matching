"""Executable eligibility DSL (Requirement 3.8-3.9, `design.md` Section 3.2).

A small expression tree — `And`, `Or`, `Not`, `Threshold`, `TemporalWindow`,
`DiagnosisHistory`, `MedicationExposure` — each implementing
`.evaluate(patient_timeline) -> EligibilityResult`. Only the operators
ACTG175 eligibility actually requires (HIV status, baseline CD4 range, age)
are exercised by the concrete `ACTG175_ELIGIBILITY` rule below; no
speculative operators beyond the seven node types the task/design call for
are added (`decisions.md` Section 4).

`src/protocols/schema.py` (the `TargetTrialSpec`) and `src/data/timelines.py`
(real patient-timeline construction) are built independently of this module.
To keep this module testable on its own and avoid a premature dependency on
either, `PatientTimeline` below is a minimal, self-contained input shape
owned by this module — not the eventual richly-typed timeline from
`src/data/timelines.py`. When that module lands, its `build_patient_timeline`
output should be adapted (or this module's evaluate() calls updated) to
consume the real type; the node types and `EligibilityResult` contract
defined here do not need to change.

Explicit missing-value policy
------------------------------
When a required field is absent (`None`, or the containing collection is
`None`) in the `patient_timeline`, the result is **ineligible** and the
field is listed in `missing_fields`. Missingness is never silently treated
as a pass or a fail of the underlying comparison — it is its own outcome,
surfaced in `missing_fields` so a caller can distinguish "failed the
criterion" from "could not evaluate the criterion." `Not` never launders a
missing-value ineligibility into an eligible result by negating it (see
`Not.evaluate` docstring).
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EligibilityResult:
    """The outcome of evaluating one node of the eligibility expression tree.

    `missing_fields` is always populated (never silently empty) whenever a
    required field was absent during evaluation, even if the overall
    expression still resolved to `eligible=True` via another clause (e.g. an
    `Or` where a different clause passed) — this keeps missingness visible
    to downstream reporting rather than hiding it behind a passing result.
    """

    eligible: bool
    reason: str
    missing_fields: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Minimal, self-contained patient-timeline input shape
# ---------------------------------------------------------------------------


@dataclass
class PatientTimeline:
    """Minimal input shape consumed by this module's expression-tree nodes.

    This is intentionally lightweight and independent of
    `src/data/timelines.py`'s eventual `PatientTimeline`/`build_patient_timeline`
    output, so `eligibility_dsl.py` can be developed and tested before that
    module exists, and wired to the real type later without changing any
    node's evaluation logic.

    Attributes:
        baseline_covariates: flat mapping of covariate name to value, used by
            `Threshold` and `TemporalWindow` (e.g. `"age"`, `"baseline_cd4"`,
            `"hiv_positive"`, `"enrollment_date"`). A key absent from this
            dict, or present with value `None`, is treated as missing.
        diagnosis_history: a list of records, each a mapping with at least a
            `"concept_id"` key and optionally a `"date"` key, used by
            `DiagnosisHistory`. `None` (as opposed to an empty list) means
            "diagnosis history is unavailable" and is treated as missing; an
            empty list means "no diagnoses recorded" and is a legitimate,
            non-missing state.
        medication_history: a list of records, each a mapping with at least a
            `"concept_id"` key and optionally a `"start_time"` key, used by
            `MedicationExposure`. Same missing-vs-empty distinction as
            `diagnosis_history`.
        assignment_time: the patient's treatment-assignment (randomization)
            timestamp — time zero per `decisions.md` Section 5 /
            `configs/target_trials/main.yaml`'s
            `time_zero.definition == "randomized_treatment_assignment"`.
            Extension added for `src/protocols/time_zero.py` (Requirement
            3.10-3.11); unused by the eligibility-DSL nodes above. `None`
            means the assignment time is unavailable/not yet known.
        baseline_covariate_timestamps: mapping of baseline-covariate/
            eligibility-relevant field name (matching a key in
            `baseline_covariates`) to the timestamp at which that value was
            observed. Extension added for `src/protocols/time_zero.py`
            (Requirement 3.10-3.11); unused by the eligibility-DSL nodes
            above. A field absent from this mapping is simply not checked by
            `validate_time_zero` (no timestamp is claimed for it), which is
            distinct from a field being present with a post-assignment
            timestamp (a violation).

    Note: `src/data/timelines.py` (not yet built) will eventually produce the
    richer, authoritative patient-timeline type used across the pipeline. It
    must produce a shape compatible with `assignment_time` and
    `baseline_covariate_timestamps` (or be adapted at the call site) so that
    `validate_time_zero` can be reused unchanged once that module lands.
    """

    baseline_covariates: dict[str, Any] = field(default_factory=dict)
    diagnosis_history: list[dict[str, Any]] | None = field(default_factory=list)
    medication_history: list[dict[str, Any]] | None = field(default_factory=list)
    assignment_time: datetime | None = None
    baseline_covariate_timestamps: dict[str, datetime] = field(default_factory=dict)


def _union_missing(results: list[EligibilityResult]) -> list[str]:
    """Union of missing fields across child results, order-preserving, deduped."""
    seen: dict[str, None] = {}
    for result in results:
        for name in result.missing_fields:
            seen.setdefault(name, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Expression tree
# ---------------------------------------------------------------------------


class EligibilityExpression:
    """Base class for every eligibility-DSL expression-tree node."""

    def evaluate(self, patient_timeline: PatientTimeline) -> EligibilityResult:
        raise NotImplementedError


class And(EligibilityExpression):
    """Logical AND over one or more clauses: eligible only if every clause is.

    Missing-value policy: `missing_fields` is the union of every clause's
    missing fields (not just the first one encountered), so a single `And`
    evaluation surfaces every absent field at once.
    """

    def __init__(self, *clauses: EligibilityExpression) -> None:
        if not clauses:
            raise ValueError("And requires at least one clause")
        self.clauses = clauses

    def evaluate(self, patient_timeline: PatientTimeline) -> EligibilityResult:
        results = [clause.evaluate(patient_timeline) for clause in self.clauses]
        eligible = all(result.eligible for result in results)
        missing_fields = _union_missing(results)
        if eligible:
            reason = "AND: all clauses satisfied."
        else:
            failing = [result.reason for result in results if not result.eligible]
            reason = "AND: failed — " + "; ".join(failing)
        return EligibilityResult(eligible=eligible, reason=reason, missing_fields=missing_fields)


class Or(EligibilityExpression):
    """Logical OR over one or more clauses: eligible if at least one clause is.

    Missing-value policy: missing fields from *every* clause are still
    reported in `missing_fields`, even clauses that were not needed because
    another clause already passed — a passing `Or` result never hides that
    part of the underlying data was unavailable.
    """

    def __init__(self, *clauses: EligibilityExpression) -> None:
        if not clauses:
            raise ValueError("Or requires at least one clause")
        self.clauses = clauses

    def evaluate(self, patient_timeline: PatientTimeline) -> EligibilityResult:
        results = [clause.evaluate(patient_timeline) for clause in self.clauses]
        eligible = any(result.eligible for result in results)
        missing_fields = _union_missing(results)
        if eligible:
            passing = [result.reason for result in results if result.eligible]
            reason = "OR: satisfied — " + "; ".join(passing)
        else:
            reason = "OR: all clauses ineligible — " + "; ".join(
                result.reason for result in results
            )
        return EligibilityResult(eligible=eligible, reason=reason, missing_fields=missing_fields)


class Not(EligibilityExpression):
    """Logical NOT over a single clause.

    Explicit missing-value policy: if the underlying clause's result carries
    any `missing_fields`, negation is *not* performed — the result stays
    ineligible with those same missing fields. This prevents a missing-value
    ineligibility from being silently laundered into an eligible result just
    because it was wrapped in `Not` (e.g. `Not(Threshold("age", ">=", 18))`
    on a patient with no recorded age must not become "eligible").
    """

    def __init__(self, clause: EligibilityExpression) -> None:
        self.clause = clause

    def evaluate(self, patient_timeline: PatientTimeline) -> EligibilityResult:
        result = self.clause.evaluate(patient_timeline)
        if result.missing_fields:
            return EligibilityResult(
                eligible=False,
                reason=(
                    "NOT: underlying clause has missing data, so negation is "
                    f"not performed — {result.reason}"
                ),
                missing_fields=list(result.missing_fields),
            )
        return EligibilityResult(
            eligible=not result.eligible,
            reason=f"NOT({result.reason})",
            missing_fields=[],
        )


_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


class Threshold(EligibilityExpression):
    """A single comparison against a baseline-covariate field.

    `op` must be one of `>=`, `<=`, `>`, `<`, `==`, `!=` — the comparators
    ACTG175 eligibility requires for baseline CD4 range and age (Requirement
    3.8). Boundary comparisons are inclusive/exclusive exactly as the chosen
    operator implies (e.g. `>=` is boundary-inclusive, `>` is not).
    """

    def __init__(self, field: str, op: str, value: Any) -> None:
        if op not in _COMPARATORS:
            raise ValueError(f"Unsupported operator {op!r}; must be one of {sorted(_COMPARATORS)}")
        self.field = field
        self.op = op
        self.value = value

    def evaluate(self, patient_timeline: PatientTimeline) -> EligibilityResult:
        observed = patient_timeline.baseline_covariates.get(self.field)
        if observed is None:
            return EligibilityResult(
                eligible=False,
                reason=(
                    f"Required field '{self.field}' is missing; cannot evaluate "
                    f"{self.field} {self.op} {self.value!r}."
                ),
                missing_fields=[self.field],
            )
        passed = _COMPARATORS[self.op](observed, self.value)
        reason = f"{self.field}={observed!r} {self.op} {self.value!r} -> {passed}"
        return EligibilityResult(eligible=passed, reason=reason, missing_fields=[])


class TemporalWindow(EligibilityExpression):
    """Checks that a timestamped baseline field falls within `[start, end]` (inclusive).

    Used for temporal eligibility windows (e.g. "enrolled within N days of
    diagnosis"). `field` is looked up in `baseline_covariates`.
    """

    def __init__(self, field: str, start: date | datetime, end: date | datetime) -> None:
        if start > end:
            raise ValueError(f"start ({start!r}) must be <= end ({end!r})")
        self.field = field
        self.start = start
        self.end = end

    def evaluate(self, patient_timeline: PatientTimeline) -> EligibilityResult:
        observed = patient_timeline.baseline_covariates.get(self.field)
        if observed is None:
            return EligibilityResult(
                eligible=False,
                reason=f"Required temporal field '{self.field}' is missing.",
                missing_fields=[self.field],
            )
        passed = self.start <= observed <= self.end
        reason = f"{self.field}={observed!r} within [{self.start!r}, {self.end!r}] -> {passed}"
        return EligibilityResult(eligible=passed, reason=reason, missing_fields=[])


class DiagnosisHistory(EligibilityExpression):
    """Checks presence/absence of a diagnosis concept in `diagnosis_history`.

    `present=True` (default) requires the concept to appear at least once;
    `present=False` requires it to never appear (an exclusion criterion). An
    optional `[window_start, window_end]` restricts matches to records whose
    `"date"` falls in that range (either bound may be omitted).
    """

    def __init__(
        self,
        concept_id: str,
        present: bool = True,
        window_start: date | datetime | None = None,
        window_end: date | datetime | None = None,
    ) -> None:
        self.concept_id = concept_id
        self.present = present
        self.window_start = window_start
        self.window_end = window_end

    def _within_window(self, event_date: Any) -> bool:
        if self.window_start is None and self.window_end is None:
            return True
        if event_date is None:
            return False
        if self.window_start is not None and event_date < self.window_start:
            return False
        if self.window_end is not None and event_date > self.window_end:
            return False
        return True

    def evaluate(self, patient_timeline: PatientTimeline) -> EligibilityResult:
        history = patient_timeline.diagnosis_history
        if history is None:
            return EligibilityResult(
                eligible=False,
                reason="diagnosis_history is missing; cannot evaluate diagnosis criterion.",
                missing_fields=["diagnosis_history"],
            )
        matches = [
            record
            for record in history
            if record.get("concept_id") == self.concept_id
            and self._within_window(record.get("date"))
        ]
        found = len(matches) > 0
        eligible = found if self.present else not found
        verb = "found" if found else "not found"
        reason = (
            f"diagnosis '{self.concept_id}' {verb} in history "
            f"-> eligible={eligible} (present={self.present})"
        )
        return EligibilityResult(eligible=eligible, reason=reason, missing_fields=[])


class MedicationExposure(EligibilityExpression):
    """Checks presence/absence of a medication concept in `medication_history`.

    `exposed=True` (default) requires the concept to have been administered
    at least once; `exposed=False` requires no such administration (e.g.
    "no prior antiretroviral exposure"). An optional `[window_start,
    window_end]` restricts matches to records whose `"start_time"` falls in
    that range.
    """

    def __init__(
        self,
        concept_id: str,
        exposed: bool = True,
        window_start: date | datetime | None = None,
        window_end: date | datetime | None = None,
    ) -> None:
        self.concept_id = concept_id
        self.exposed = exposed
        self.window_start = window_start
        self.window_end = window_end

    def _within_window(self, start_time: Any) -> bool:
        if self.window_start is None and self.window_end is None:
            return True
        if start_time is None:
            return False
        if self.window_start is not None and start_time < self.window_start:
            return False
        if self.window_end is not None and start_time > self.window_end:
            return False
        return True

    def evaluate(self, patient_timeline: PatientTimeline) -> EligibilityResult:
        history = patient_timeline.medication_history
        if history is None:
            return EligibilityResult(
                eligible=False,
                reason="medication_history is missing; cannot evaluate medication criterion.",
                missing_fields=["medication_history"],
            )
        matches = [
            record
            for record in history
            if record.get("concept_id") == self.concept_id
            and self._within_window(record.get("start_time"))
        ]
        found = len(matches) > 0
        eligible = found if self.exposed else not found
        verb = "found" if found else "not found"
        reason = (
            f"medication '{self.concept_id}' {verb} in history "
            f"-> eligible={eligible} (exposed={self.exposed})"
        )
        return EligibilityResult(eligible=eligible, reason=reason, missing_fields=[])


# ---------------------------------------------------------------------------
# ACTG175's concrete eligibility rule
# ---------------------------------------------------------------------------

# ACTG175's original entry criteria required documented HIV-1 infection,
# baseline CD4 T-cell count of 200-500 cells/mm^3, and a minimum age of 12
# years with no documented upper age bound (`decisions.md` Section 4 —
# "satisfy ACTG 175's original HIV and baseline-CD4 eligibility criteria";
# Section 2/24 for the trial identity). Age >= 12 and CD4 200-500 are
# corroborated by the trial's public registry record (ClinicalTrials.gov
# NCT00000625: minimumAge "12 Years", no maximumAge; eligibility text "HIV
# infection without AIDS with CD4 200-500 cells/mm3") and by Hammer SM, et
# al., NEJM 1996 (PMID 8813038) — the frozen source in `decisions.md`
# Section 24. This module expresses that rule concretely as an instance of
# the primitives above, rather than leaving the DSL only demonstrated in
# the abstract.
ACTG175_ELIGIBILITY = And(
    Threshold("hiv_positive", "==", True),
    Threshold("baseline_cd4", ">=", 200),
    Threshold("baseline_cd4", "<=", 500),
    Threshold("age", ">=", 12),
)
