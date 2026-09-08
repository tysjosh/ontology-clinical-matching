"""Cohort-flow table: initial population -> analyzed population (Requirement 7.3, 15.3).

`build_cohort_flow_table` takes the same `(timelines, treatment_arms, spec)`
inputs as `cohort_construction.build_randomized_cohort` and produces a flat,
JSON-serializable summary of how the candidate population narrows down to
the final analyzed (randomized) cohort, plus a Mermaid `flowchart TD` source
string for Figure 3 (`design.md` Section 3.4, `decisions.md` Section 26).

Stages and the invariants that tie them together:

- `initial_population_count == eligible_count + ineligible_excluded_count`
- `eligible_count == arm_0_1_count + arm_invalid_or_excluded_count`
- `arm_0_1_count == censored_count + event_count == analyzed_population_count`

Note the last stage is a *sub-breakdown*, not a further exclusion:
`build_randomized_cohort` independently re-derives the same eligible,
arm-0/1-restricted set (see `cohort_construction.py`), and every patient in
that set has a composite-endpoint result which is either censored or an
event — so `analyzed_population_count` equals `arm_0_1_count` exactly, and
`censored_count + event_count` is simply that same population split by
outcome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cohorts.cohort_construction import build_randomized_cohort
from data.timelines import PatientTimeline as DataPatientTimeline
from protocols.eligibility_dsl import ACTG175_ELIGIBILITY
from protocols.eligibility_dsl import PatientTimeline as EligibilityPatientTimeline
from protocols.schema import TargetTrialSpec

#: The two treatment arms retained past the eligibility stage (mirrors
#: `cohort_construction.RETAINED_ARMS`).
_RETAINED_ARMS = frozenset({0, 1})


def _is_eligible(timeline: DataPatientTimeline) -> bool:
    """Evaluate ACTG175 eligibility for one candidate timeline.

    Local copy of the adapter in `cohort_construction._adapt_to_eligibility_timeline`:
    both `PatientTimeline` types use the same flat `concept_id -> value`
    shape for baseline covariates, so this is a direct copy, not worth
    importing/refactoring for.
    """
    eligibility_timeline = EligibilityPatientTimeline(
        baseline_covariates=dict(timeline.static_covariates),
    )
    return ACTG175_ELIGIBILITY.evaluate(eligibility_timeline).eligible


def build_cohort_flow_table(
    timelines: Sequence[DataPatientTimeline] | Mapping[str, DataPatientTimeline],
    treatment_arms: Mapping[str, int],
    spec: TargetTrialSpec,
) -> dict[str, Any]:
    """Build the cohort-flow table and its Mermaid diagram source.

    Args:
        timelines: the full candidate population's `PatientTimeline`s
            (before any eligibility/arm filtering), as a sequence or a
            `{patient_id: PatientTimeline}` mapping.
        treatment_arms: `{patient_id: arm}` raw treatment-arm assignments.
        spec: the target-trial specification, passed through unchanged to
            `build_randomized_cohort` (see that function's docstring for
            why it is accepted but not currently consulted).

    Returns:
        `{"flow_table": {...}, "mermaid": "flowchart TD\\n..."}`. The
        `flow_table` dict is flat and JSON-serializable; see the module
        docstring for the consistency invariants it satisfies.
    """
    timeline_list = list(timelines.values()) if isinstance(timelines, Mapping) else list(timelines)

    initial_population_count = len(timeline_list)

    eligible_count = 0
    ineligible_excluded_count = 0
    arm_0_count = 0
    arm_1_count = 0
    arm_invalid_or_excluded_count = 0

    for timeline in timeline_list:
        if not _is_eligible(timeline):
            ineligible_excluded_count += 1
            continue

        eligible_count += 1
        arm = treatment_arms.get(timeline.patient_id)
        if arm == 0:
            arm_0_count += 1
        elif arm == 1:
            arm_1_count += 1
        else:
            arm_invalid_or_excluded_count += 1

    arm_0_1_count = arm_0_count + arm_1_count

    cohort = build_randomized_cohort(timeline_list, treatment_arms, spec)
    analyzed_population_count = len(cohort)
    censored_count = sum(
        1 for patient in cohort if patient.endpoint_result.censoring_indicator
    )
    event_count = sum(1 for patient in cohort if patient.endpoint_result.event_indicator)

    flow_table: dict[str, Any] = {
        "initial_population_count": initial_population_count,
        "ineligible_excluded_count": ineligible_excluded_count,
        "eligible_count": eligible_count,
        "arm_0_count": arm_0_count,
        "arm_1_count": arm_1_count,
        "arm_0_1_count": arm_0_1_count,
        "arm_invalid_or_excluded_count": arm_invalid_or_excluded_count,
        "censored_count": censored_count,
        "event_count": event_count,
        "analyzed_population_count": analyzed_population_count,
        "stages": [
            {"name": "initial_population", "count": initial_population_count},
            {
                "name": "eligibility_exclusions",
                "count": ineligible_excluded_count,
                "eligible_count": eligible_count,
            },
            {
                "name": "arm_assignment",
                "count": arm_0_1_count,
                "arm_0_count": arm_0_count,
                "arm_1_count": arm_1_count,
                "excluded_count": arm_invalid_or_excluded_count,
            },
            {
                "name": "censoring",
                "count": analyzed_population_count,
                "censored_count": censored_count,
                "event_count": event_count,
            },
            {"name": "analyzed_population", "count": analyzed_population_count},
        ],
    }

    return {"flow_table": flow_table, "mermaid": _build_mermaid(flow_table)}


def _build_mermaid(flow_table: dict[str, Any]) -> str:
    """Render the flow table as a Mermaid `flowchart TD` source string."""
    lines = [
        "flowchart TD",
        f'    A["Initial population<br/>n={flow_table["initial_population_count"]}"]',
        f'    B["Eligible<br/>n={flow_table["eligible_count"]}"]',
        (
            '    X["Ineligible (excluded)<br/>n='
            f'{flow_table["ineligible_excluded_count"]}"]'
        ),
        (
            '    C["Arm 0/1 assigned<br/>n='
            f'{flow_table["arm_0_1_count"]} (arm0={flow_table["arm_0_count"]}, '
            f'arm1={flow_table["arm_1_count"]})"]'
        ),
        (
            '    Y["Invalid/other arm (excluded)<br/>n='
            f'{flow_table["arm_invalid_or_excluded_count"]}"]'
        ),
        (
            '    D["Analyzed population<br/>n='
            f'{flow_table["analyzed_population_count"]} '
            f'(censored={flow_table["censored_count"]}, '
            f'events={flow_table["event_count"]})"]'
        ),
        "    A --> B",
        "    A --> X",
        "    B --> C",
        "    B --> Y",
        "    C --> D",
    ]
    return "\n".join(lines)
