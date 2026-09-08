"""Leakage auditing and the CD8 double-use safeguard (Requirement 5.10-5.11).

`design.md` Section 3.3 specifies this module as implementing
`LeakageAuditor.audit(timelines, adjustment_set) -> list[LeakageViolation]`,
checking: post-treatment variables in baseline features, outcome-derived
features, duplicate patient IDs across splits, and the CD8 double-use
safeguard (`decisions.md` Section 11.3) via a single reusable check,
`assert_no_self_adjustment(outcome_field, adjustment_set)`. That function is
implemented here as the canonical shared implementation — it is also called
directly by the negative-control-outcome regression builder
(`src/protocols/negative_controls.py`, a later task) rather than being
duplicated there.

Design note — why this module does not call `protocols/time_zero.py`
----------------------------------------------------------------------

`design.md` Section 3.3 describes this module as reusing
`protocols/time_zero.py`'s "post-treatment variables in baseline features"
concept. However, `validate_time_zero(timeline, spec)` operates on
`protocols.eligibility_dsl.PatientTimeline` — a small, separate type with
`assignment_time` and `baseline_covariate_timestamps` fields — while this
module's input is `data.timelines.PatientTimeline`, the richer canonical
timeline with `static_covariates` (a plain `dict[str, Any]`, no per-field
timestamp) and `treatment_events` (a list of `ClinicalEvent`, each with its
own `observed_at`). The two `PatientTimeline` types are intentionally
distinct (see `data/timelines.py`'s module docstring) and are not unified by
this task.

Rather than forcing a dependency on `time_zero.py`'s differently-shaped
input (which would require synthesizing an `assignment_time` and a
`baseline_covariate_timestamps` mapping that `data.timelines.PatientTimeline`
does not carry), this module implements the equivalent check directly
against `data.timelines.PatientTimeline`: it reconstructs each static
covariate's originating event from the timeline's own event lists (the
`ClinicalEvent` objects that produced `static_covariates`, `endpoint_events`,
etc. are threaded through to this module rather than re-derived from the
already-collapsed `static_covariates` dict alone — see
`_TimelineWithEvents`/the audited-timeline shape below) and compares its
`observed_at` against the earliest `treatment_events` timestamp. This keeps
the same semantic check (no post-treatment information in a baseline
feature) without depending on a type this module's input does not produce.

Concretely, since `data.timelines.PatientTimeline.static_covariates` is a
plain `{concept_id: value}` dict with no timestamp, this module recovers the
timestamp from `timeline.missingness_mask`'s keying convention is *not*
timestamped for static entries either (`data/timelines.py`: static entries
are keyed by `concept_id` alone). Baseline covariates in the canonical
timeline therefore carry no per-field observation timestamp at all by
design (they are, by construction, values captured at/around time zero from
`baseline`/`demographic`-modality events). To make the post-treatment check
meaningful anyway, `LeakageAuditor.audit` accepts the *originating events*
for each timeline's static covariates via an optional
`static_covariate_events` mapping (`{patient_id: {concept_id: observed_at}}`);
callers that have access to the original `ClinicalEvent`s that fed
`build_patient_timeline` (e.g. the leakage-audit pipeline stage, which runs
immediately after Stage 2's "build ClinicalEvents") can supply this. Only
values in this mapping are checked for the post-treatment rule — this is a
strictly additive, opt-in check that never runs (and never false-negatively
"passes") on a timeline whose caller didn't supply timestamps for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from data.timelines import PatientTimeline

#: Default denylist for the "indirect treatment/outcome encoding" heuristic
#: check. Any static-covariate concept_id containing one of these substrings
#: (case-insensitive) is flagged, on the theory that a field named this way
#: almost certainly encodes treatment status or outcome information
#: indirectly (e.g. a derived "treatment_switch_indicator" or
#: "outcome_flag" column) even though it is not itself a treatment or
#: endpoint event. This is a simple, documented heuristic, not an
#: exhaustive semantic analysis — real deployments should extend this list
#: as new indirect encodings are discovered.
DEFAULT_INDIRECT_ENCODING_DENYLIST: tuple[str, ...] = (
    "outcome",
    "endpoint",
    "treatment_switch",
    "arm_assignment",
    "event_indicator",
    "censor_reason",
)


class SelfAdjustmentError(Exception):
    """Raised when an outcome field also appears in its own adjustment set.

    This is the CD8 double-use safeguard (`decisions.md` Section 11.3):
    baseline CD8 must never be both the outcome (or pseudo-outcome) of a
    regression *and* a covariate in that same regression's adjustment set,
    since doing so would make the check tautological.
    """

    def __init__(self, outcome_field: str, adjustment_set: list[str]):
        self.outcome_field = outcome_field
        self.adjustment_set = adjustment_set
        super().__init__(
            f"Self-adjustment detected: outcome field '{outcome_field}' also "
            f"appears in its own adjustment set {adjustment_set!r}. A "
            "regression must not adjust for its own outcome (or "
            "pseudo-outcome) — this makes the check tautological. Remove "
            f"'{outcome_field}' from the adjustment set for this regression."
        )


def assert_no_self_adjustment(outcome_field: str, adjustment_set: list[str]) -> None:
    """Raise `SelfAdjustmentError` if `outcome_field` is in `adjustment_set`.

    The single shared implementation of the CD8 double-use safeguard
    (`decisions.md` Section 11.3): used both by `LeakageAuditor` (exercised
    directly in tests here) and by the negative-control-outcome regression
    builder (`src/protocols/negative_controls.py`), which must exclude
    baseline CD8 from its own adjustment set.

    Args:
        outcome_field: the name of the outcome (or pseudo-outcome) field of
            a regression, e.g. `"baseline_cd8"`.
        adjustment_set: the list of covariate names used as the right-hand
            side / adjustment set of that same regression.

    Raises:
        SelfAdjustmentError: if `outcome_field` appears anywhere in
            `adjustment_set`.
    """
    if outcome_field in adjustment_set:
        raise SelfAdjustmentError(outcome_field, adjustment_set)


@dataclass(frozen=True)
class LeakageViolation:
    """A single leakage-audit finding.

    Attributes:
        patient_id: the patient this violation pertains to, or `None` for a
            cohort-level violation that isn't attributable to a single
            patient (e.g. a duplicate-across-splits finding, which is
            reported once per offending patient_id but conceptually spans
            multiple split memberships rather than a single patient
            record — see `rule` docs below for how each rule populates this
            field).
        rule: a short machine-readable identifier for which check produced
            this violation, one of: `"post_treatment_baseline_variable"`,
            `"outcome_derived_feature"`, `"duplicate_patient_across_splits"`,
            `"indirect_treatment_or_outcome_encoding"`.
        evidence: a structured dict of rule-specific evidence (e.g. the
            offending `concept_id`, the observed/treatment timestamps
            compared, or the list of splits a duplicated patient_id was
            found in), sufficient to reconstruct why the violation was
            raised without re-running the audit.
    """

    patient_id: str | None
    rule: str
    evidence: dict[str, Any]


@dataclass
class LeakageAuditor:
    """Audits patient timelines and a proposed adjustment set for leakage.

    `design.md` Section 3.3. Each of the four checks below is independent:
    a caller may supply only the inputs relevant to the checks they want run
    (e.g. omitting `split_membership` skips the duplicate-across-splits
    check entirely, rather than raising).

    Attributes:
        indirect_encoding_denylist: substrings (case-insensitive) that flag
            a static-covariate `concept_id` as a likely indirect
            treatment/outcome encoding. Defaults to
            `DEFAULT_INDIRECT_ENCODING_DENYLIST`; callers may override with
            a project-specific list.
    """

    indirect_encoding_denylist: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_INDIRECT_ENCODING_DENYLIST
    )

    def audit(
        self,
        timelines: dict[str, PatientTimeline] | list[PatientTimeline],
        adjustment_set: list[str],
        split_membership: dict[str, list[str]] | None = None,
        static_covariate_events: dict[str, dict[str, datetime]] | None = None,
    ) -> list[LeakageViolation]:
        """Run every leakage check and return all violations found.

        Args:
            timelines: the cohort's timelines, either keyed by `patient_id`
                or as a plain list (each timeline's own `patient_id`
                attribute is used regardless of the input shape).
            adjustment_set: the proposed baseline adjustment set (covariate
                concept_ids). Used by the outcome-derived-feature and
                indirect-encoding checks to scope which static covariates
                are actually in play for the analysis being audited.
            split_membership: optional `{patient_id: [split_name, ...]}`
                mapping built by the caller from the actual per-split
                patient-id lists (e.g. `{"P001": ["dev"], "P002": ["dev",
                "test"]}`). A single `patient_id -> split` dict cannot by
                itself represent a duplicate, so this module requires the
                list-of-splits shape instead; any patient_id whose list has
                length > 1 is a duplicate-across-splits violation. `None`
                (the default) skips this check.
            static_covariate_events: optional
                `{patient_id: {concept_id: observed_at}}` mapping giving the
                originating event timestamp for each static covariate (see
                module docstring for why `data.timelines.PatientTimeline`
                does not carry this itself). `None` (the default) skips the
                post-treatment-baseline-variable check.

        Returns:
            All `LeakageViolation`s found, in the order: post-treatment
            baseline variables, outcome-derived features, indirect
            encodings, duplicate patients across splits.
        """
        timeline_list = list(timelines.values()) if isinstance(timelines, dict) else list(timelines)

        violations: list[LeakageViolation] = []
        violations.extend(
            self._check_post_treatment_baseline_variables(timeline_list, static_covariate_events)
        )
        violations.extend(self._check_outcome_derived_features(timeline_list, adjustment_set))
        violations.extend(self._check_indirect_encodings(timeline_list, adjustment_set))
        violations.extend(self._check_duplicate_patients_across_splits(split_membership))
        return violations

    def _check_post_treatment_baseline_variables(
        self,
        timeline_list: list[PatientTimeline],
        static_covariate_events: dict[str, dict[str, datetime]] | None,
    ) -> list[LeakageViolation]:
        """Flag any static covariate observed strictly after the earliest treatment event.

        A baseline (static) covariate is expected to be captured at or
        before time zero. If its originating event's `observed_at` is
        strictly after the timeline's earliest `treatment_events` timestamp,
        post-treatment information has leaked into a baseline feature
        (mirrors `protocols/time_zero.py`'s rule, adapted to this module's
        input shape — see module docstring).
        """
        if not static_covariate_events:
            return []

        violations: list[LeakageViolation] = []
        for timeline in timeline_list:
            covariate_timestamps = static_covariate_events.get(timeline.patient_id)
            if not covariate_timestamps or not timeline.treatment_events:
                continue

            earliest_treatment_time = min(
                event.observed_at for event in timeline.treatment_events
            )
            for concept_id, observed_at in covariate_timestamps.items():
                if concept_id not in timeline.static_covariates:
                    continue
                if observed_at > earliest_treatment_time:
                    violations.append(
                        LeakageViolation(
                            patient_id=timeline.patient_id,
                            rule="post_treatment_baseline_variable",
                            evidence={
                                "concept_id": concept_id,
                                "observed_at": observed_at.isoformat(),
                                "earliest_treatment_time": earliest_treatment_time.isoformat(),
                            },
                        )
                    )
        return violations

    def _check_outcome_derived_features(
        self, timeline_list: list[PatientTimeline], adjustment_set: list[str]
    ) -> list[LeakageViolation]:
        """Flag any static covariate whose concept_id overlaps an endpoint concept_id.

        Heuristic: a static covariate is "outcome-derived" if its
        `concept_id` is identical to (or, per the substring rule below,
        derived from) a `concept_id` present in that same timeline's
        `endpoint_events`. This is a simple concept_id-overlap check, not a
        semantic analysis of feature derivation — it catches the case where
        an endpoint concept was accidentally also materialized as a
        baseline feature (e.g. `aids_defining_event` appearing in both
        `static_covariates` and `endpoint_events`), which is the leakage
        pattern this check is meant to guard against.
        """
        violations: list[LeakageViolation] = []
        for timeline in timeline_list:
            endpoint_concept_ids = {event.concept_id for event in timeline.endpoint_events}
            if not endpoint_concept_ids:
                continue

            for concept_id in timeline.static_covariates:
                if concept_id not in adjustment_set:
                    continue
                if concept_id in endpoint_concept_ids:
                    violations.append(
                        LeakageViolation(
                            patient_id=timeline.patient_id,
                            rule="outcome_derived_feature",
                            evidence={
                                "concept_id": concept_id,
                                "matched_endpoint_concept_id": concept_id,
                            },
                        )
                    )
        return violations

    def _check_indirect_encodings(
        self, timeline_list: list[PatientTimeline], adjustment_set: list[str]
    ) -> list[LeakageViolation]:
        """Flag static covariates whose concept_id matches the indirect-encoding denylist.

        A simple, documented substring heuristic (see
        `indirect_encoding_denylist`): a static covariate whose concept_id
        contains a denylisted substring (case-insensitive) is flagged as a
        likely indirect treatment/outcome encoding, e.g. a derived
        `"treatment_switch_indicator"` field.
        """
        violations: list[LeakageViolation] = []
        for timeline in timeline_list:
            for concept_id in timeline.static_covariates:
                if concept_id not in adjustment_set:
                    continue
                lowered = concept_id.lower()
                for denylisted_substring in self.indirect_encoding_denylist:
                    if denylisted_substring in lowered:
                        violations.append(
                            LeakageViolation(
                                patient_id=timeline.patient_id,
                                rule="indirect_treatment_or_outcome_encoding",
                                evidence={
                                    "concept_id": concept_id,
                                    "matched_denylist_substring": denylisted_substring,
                                },
                            )
                        )
                        break
        return violations

    def _check_duplicate_patients_across_splits(
        self, split_membership: dict[str, list[str]] | None
    ) -> list[LeakageViolation]:
        """Flag any patient_id present in more than one split.

        `split_membership` maps `patient_id -> [split_name, ...]`, built by
        the caller from the actual per-split patient-id lists. A
        `patient_id -> single split` dict cannot itself represent a
        duplicate (each key maps to exactly one value), so this module
        requires the list-of-splits shape: any patient_id whose list has
        length > 1 was found in more than one split and is a
        duplicate-across-splits violation. This is a cohort-level leakage
        check (a patient appearing in both `dev` and `test` invalidates the
        train/test separation), so it is exercised via a dedicated
        `split_membership` parameter rather than by inferring split
        assignment from `timelines` itself.
        """
        if not split_membership:
            return []

        violations: list[LeakageViolation] = []
        for patient_id, splits in split_membership.items():
            if len(splits) > 1:
                violations.append(
                    LeakageViolation(
                        patient_id=patient_id,
                        rule="duplicate_patient_across_splits",
                        evidence={"splits": list(splits)},
                    )
                )
        return violations
