"""Cohort_Release_Gate: the single six-way conjunction that authorizes
releasing a cohort-level estimate (Requirement 11; `decisions.md` Section
16; `design.md` Section 3.8).

Why this module is new
-------------------------------------------------------------------------

The parent spec's design named `release_gate.py` under `src/uncertainty/`,
but the shipped code only ever grew `uncertainty/support_gate.py` (the
*patient*-level gate). No module computes the *cohort*-level formula --
`scripts/run_causal_analysis.py` reports the witness aggregation
(`witnesses.aggregation.aggregate`) alone, which is only three of the six
conditions `decisions.md` Section 16 actually requires. This module is
that missing cohort-level computation.

It lives under `evaluation/` rather than `uncertainty/` because it reads
`causal.overlap.OverlapReport` and witness results, both of which
`evaluation` already depends on elsewhere (`evaluation.metrics`,
`evaluation.statistics`), and because `uncertainty/` is kept free of any
import on `witnesses/` (the module docstring of `uncertainty.support_gate`
is explicit about this).

The frozen formula
-------------------------------------------------------------------------

`decisions.md` Section 16:

    release =
        overlap_valid
        AND balance_valid
        AND uncertainty_valid
        AND temporal_witness == pass
        AND causal_witness == pass
        AND clinical_witness == pass

realized here as a single boolean expression in `evaluate_release`, not
split across call sites that could drift out of sync with one another.

Where each input comes from
-------------------------------------------------------------------------

- `overlap_valid` <- `causal.overlap.OverlapReport.support_valid`.
- `balance_valid` <- `causal.overlap.OverlapReport.smd_sufficient`.

Both are read directly off the already-computed `OverlapReport` rather
than recomputed here (Requirement 11.1's "reading ... rather than
recomputing them") -- `OverlapReport.support_valid` already bundles
`smd_sufficient` as one of its four conjuncts, so `balance_valid` is also
implicitly part of `overlap_valid`; this module still reports them as two
named booleans because `decisions.md` Section 16 names them as two
separate top-level conditions, and the audit trail (Requirement 11.2,
11.5) should name both explicitly even though one subsumes the other.

- `temporal_pass` / `causal_pass` / `clinical_pass` <- read off the
  caller-supplied `witness_results` sequence (one
  `witnesses.temporal.WitnessResult` per witness, matched by
  `.witness` name). A missing witness in `witness_results` is treated as
  not passing (there is no witness there to report `"pass"`), which keeps
  `evaluate_release` total over any `witness_results` sequence rather than
  raising when a caller forgets one.
- `uncertainty_valid` is a plain caller-supplied boolean, matching
  `witnesses.causal.CausalWitness.evaluate`'s own
  `sensitivity_stable: bool` convention for a quantity this module does
  not compute itself (task 16.1 is the module that actually derives it,
  from `uncertainty.ensemble`/`uncertainty.support_gate`).

A causal-witness `"uncertain"` is explicitly not a pass (Requirement
11.3) -- `causal_pass` is `status == "pass"`, strictly, so `"uncertain"`
and `"fail"` both block release identically. This matches
`witnesses.causal.CausalWitness`'s own three-way classification, where
`"uncertain"` already blocks release under the primary `ALL_REQUIRED`
witness-aggregation policy (`witnesses.aggregation.aggregate`); this
module's conjunction makes the same rule explicit at the cohort-release
level rather than relying on it being true only incidentally.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from causal.overlap import OverlapReport
    from witnesses.temporal import WitnessResult


@dataclass(frozen=True)
class ReleaseDecision:
    """The result of `evaluate_release()`.

    Attributes:
        released: `True` only if all six conditions below hold.
        overlap_valid: `OverlapReport.support_valid`.
        balance_valid: `OverlapReport.smd_sufficient`.
        uncertainty_valid: the caller-supplied uncertainty-gate boolean.
        temporal_pass: `True` iff the temporal witness's status is
            exactly `"pass"`.
        causal_pass: `True` iff the causal witness's status is exactly
            `"pass"` (an `"uncertain"` causal witness is NOT a pass,
            Requirement 11.3).
        clinical_pass: `True` iff the clinical witness's status is
            exactly `"pass"`.
        reasons: every failing condition, named explicitly (Requirement
            11.5). Empty iff `released` is `True`.
    """

    released: bool
    overlap_valid: bool
    balance_valid: bool
    uncertainty_valid: bool
    temporal_pass: bool
    causal_pass: bool
    clinical_pass: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _status_for(witness_results: Sequence[WitnessResult], witness_name: str) -> str | None:
    for result in witness_results:
        if result.witness == witness_name:
            return result.status
    return None


def evaluate_release(
    overlap: OverlapReport,
    witness_results: Sequence[WitnessResult],
    uncertainty_valid: bool,
) -> ReleaseDecision:
    """Evaluate the frozen six-way cohort-release conjunction.

    Args:
        overlap: the already-computed overlap/balance report
            (`causal.overlap.OverlapDiagnostics.compute`'s output).
            `overlap_valid` and `balance_valid` are read from it, never
            recomputed.
        witness_results: the witnesses' results, in any order and any
            count -- `evaluate_release` looks up `"temporal"`,
            `"causal"`, and `"clinical"` by name. A missing witness is
            treated as not passing.
        uncertainty_valid: whether the uncertainty gate passes for this
            cohort/configuration (task 16.1 computes this value; this
            function accepts it as given, matching
            `witnesses.causal.CausalWitness.evaluate`'s
            `sensitivity_stable` convention for an input this module does
            not derive itself).

    Returns:
        A `ReleaseDecision` with all six condition statuses always
        populated (Requirement 11.2) and every failing condition named in
        `reasons` (Requirement 11.5).
    """
    overlap_valid = overlap.support_valid
    balance_valid = overlap.smd_sufficient

    temporal_status = _status_for(witness_results, "temporal")
    causal_status = _status_for(witness_results, "causal")
    clinical_status = _status_for(witness_results, "clinical")

    temporal_pass = temporal_status == "pass"
    causal_pass = causal_status == "pass"
    clinical_pass = clinical_status == "pass"

    reasons: list[str] = []
    if not overlap_valid:
        reasons.append("overlap_valid: fail (OverlapReport.support_valid is False)")
    if not balance_valid:
        reasons.append("balance_valid: fail (OverlapReport.smd_sufficient is False)")
    if not uncertainty_valid:
        reasons.append("uncertainty_valid: fail")
    if not temporal_pass:
        reasons.append(f"temporal_pass: fail (status={temporal_status!r})")
    if not causal_pass:
        reasons.append(f"causal_pass: fail (status={causal_status!r})")
    if not clinical_pass:
        reasons.append(f"clinical_pass: fail (status={clinical_status!r})")

    released = (
        overlap_valid
        and balance_valid
        and uncertainty_valid
        and temporal_pass
        and causal_pass
        and clinical_pass
    )

    return ReleaseDecision(
        released=released,
        overlap_valid=overlap_valid,
        balance_valid=balance_valid,
        uncertainty_valid=uncertainty_valid,
        temporal_pass=temporal_pass,
        causal_pass=causal_pass,
        clinical_pass=clinical_pass,
        reasons=tuple(reasons),
    )
