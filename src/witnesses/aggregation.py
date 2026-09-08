"""Witness aggregation (`decisions.md` Section 16, "Primary aggregation
policy"; Requirement 11.9).

`decisions.md` Section 16 freezes the following (copied verbatim, not
re-derived):

    Primary aggregation policy: all three witnesses required for release.
    Two-of-three is reported only as an exploratory sensitivity comparison,
    never as the primary gate.

Relationship to `infra.config.ExperimentConfig`
--------------------------------------------------

The config-level enforcement of "`witness_policy` other than
`ALL_REQUIRED` requires `exploratory=true`" already exists as
`ExperimentConfig._witness_policy_is_valid_and_gated` (task 1.1) -- see
`src/infra/config.py`. This module does **not** re-implement that
validation; `aggregate()` is a pure function over already-classified
`WitnessResult`s and a `WitnessPolicy`, with no knowledge of whether the
calling config is exploratory. Gating which policy a given run is
*allowed* to use is entirely `ExperimentConfig`'s job, already done.

`WitnessPolicy`'s string values are chosen to match
`ExperimentConfig.witness_policy`'s validated set exactly
(`{"ALL_REQUIRED", "TWO_OF_THREE", "RISK_WEIGHTED"}`) so a resolved
config's `witness_policy` string can be converted directly via
`WitnessPolicy(resolved_config.experiment.witness_policy)` without a
separate translation table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from witnesses.temporal import WitnessResult


class WitnessPolicy(str, Enum):
    """Aggregation policies for combining the three witnesses' results.

    - `ALL_REQUIRED`: the primary, frozen release-gate policy -- every
      witness must independently return `"pass"`.
    - `TWO_OF_THREE`: exploratory-only (`decisions.md` Section 16). Never
      valid as the primary release gate; `infra.config.ExperimentConfig`
      already rejects this policy on any non-exploratory config. This
      module implements only the pure aggregation arithmetic and does not
      re-check `exploratory` itself.
    - `RISK_WEIGHTED`: unused for ACTG175 (`decisions.md` Section 16 does
      not define a risk-weighting formula). `aggregate()` raises
      `NotImplementedError` for this policy rather than guessing at one.
    """

    ALL_REQUIRED = "ALL_REQUIRED"
    TWO_OF_THREE = "TWO_OF_THREE"
    RISK_WEIGHTED = "RISK_WEIGHTED"


@dataclass(frozen=True)
class AggregateDecision:
    """The result of `aggregate()`.

    Attributes:
        passed: whether the given `results` satisfy `policy`.
        policy: the policy that was applied.
        evidence: rule-level evidence supporting `passed`, always
            populated (mirrors `WitnessResult.evidence` / `SupportDecision.reasons`
            always being populated), including each witness's individual
            status and, for `TWO_OF_THREE`, the pass count.
    """

    passed: bool
    policy: WitnessPolicy
    evidence: dict


def aggregate(results: Sequence[WitnessResult], policy: WitnessPolicy) -> AggregateDecision:
    """Aggregate a set of witness results under `policy`.

    Args:
        results: the witness results to aggregate (typically the temporal,
            causal, and clinical witnesses' outputs -- three total, though
            this function does not enforce an exact count beyond what each
            policy's arithmetic implies).
        policy: which aggregation policy to apply.

    Returns:
        An `AggregateDecision` with `passed` set per `policy`'s rule and
        `evidence` detailing each witness's status.

    Raises:
        NotImplementedError: if `policy is WitnessPolicy.RISK_WEIGHTED` --
            this policy is documented as unused for ACTG175 in
            `decisions.md` Section 16, and no risk-weighting formula is
            frozen there.
    """
    statuses = {result.witness: result.status for result in results}
    pass_count = sum(1 for result in results if result.status == "pass")

    if policy is WitnessPolicy.ALL_REQUIRED:
        passed = len(results) > 0 and pass_count == len(results)
        evidence = {
            "policy": policy.value,
            "statuses": statuses,
            "pass_count": pass_count,
            "total": len(results),
            "rule": "all witnesses must have status=='pass'",
        }
        return AggregateDecision(passed=passed, policy=policy, evidence=evidence)

    if policy is WitnessPolicy.TWO_OF_THREE:
        # Exploratory-only per `decisions.md` Section 16 -- never the
        # primary release gate. `infra.config.ExperimentConfig` already
        # rejects this policy for non-exploratory configs; this function
        # implements only the pure "at least 2 of the (expected 3) pass"
        # arithmetic and does not re-check `exploratory` itself.
        passed = pass_count >= 2
        evidence = {
            "policy": policy.value,
            "statuses": statuses,
            "pass_count": pass_count,
            "total": len(results),
            "rule": "at least 2 of 3 witnesses must have status=='pass'",
            "note": (
                "exploratory-only sensitivity comparison; never the primary "
                "release gate (decisions.md Section 16)"
            ),
        }
        return AggregateDecision(passed=passed, policy=policy, evidence=evidence)

    if policy is WitnessPolicy.RISK_WEIGHTED:
        raise NotImplementedError(
            "WitnessPolicy.RISK_WEIGHTED is documented as unused in "
            "decisions.md Section 16 -- no risk-weighting formula is frozen "
            "for ACTG175, so this policy is not implemented."
        )

    raise ValueError(f"Unrecognized policy: {policy!r}")
