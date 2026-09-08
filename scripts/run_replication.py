#!/usr/bin/env python
"""Five-seed replication: the frozen cross-seed stability criterion
(neural-experiment-pipeline Requirements 9.6, 9.7, 18.1, 18.2; task 1.2).

Scope of this module *today*
---------------------------

This file is the eventual home of the Replication_Orchestrator and
Replication_Aggregator (`design.md` Section 3.11: `run_seed` and
`aggregate`), which task 18 / 18.1 implement. It is created early, and
deliberately empty of orchestration, because Requirement 18.2 forbids any
component from depending on a value `decisions.md` does not fix: the
cross-seed stability criterion had to be frozen in `decisions.md` and
expressed in code *before* the aggregator that applies it exists. So right
now this module contains exactly one thing -- the frozen criterion -- and
no pipeline code.

It is also deliberately torch-free and side-effect-free at import time
(`design.md` Section 2.4 keeps torch a leaf dependency), so
`tests/frozen/test_decisions_match.py` can import the constant in an
environment where torch is not installed, and so importing it neither
opens a split nor touches the sealed reference.

The frozen criterion
--------------------

`decisions.md` Section 26 amendment dated 2026-09-07 (amending Sections
19/20/25) freezes the cross-seed stability criterion as: effect-direction
agreement in at least 4 of the 5 frozen replication seeds
(`101, 202, 303, 404, 505`), together with no required sensitivity variant
contradicting the aggregate direction.

Per that amendment, mechanically:

- a seed *agrees* when its effect direction equals the aggregate direction,
  computed with the existing `evaluation.metrics.effect_direction_agreement`;
- the *required* sensitivity variants are exactly the eight frozen Section
  20 variants (`evaluation.robustness.SENSITIVITY_VARIANTS`);
- a required variant *contradicts* when it executed successfully and its
  effect direction is the opposite non-null direction of the aggregate; an
  unexecuted variant instead forces sensitivity stability to
  `undetermined` (Requirement 8.10) and is never counted as agreement;
- failing the criterion yields `directionally_unstable`, and every per-seed
  value is retained in that branch exactly as in the stable branch
  (Requirement 9.8).

The amendment also records what this criterion is *not*: it is a
prespecified consistency filter, not an inferential test. Every
inferential claim continues to come from `decisions.md` Section 19's
Holm-corrected patient-level bootstrap intervals.

`criterion_description` below is the human-readable sentence the
amendment records verbatim; Requirement 9.7 requires the aggregator to
report the applied criterion alongside its determination, and it reports
this string verbatim rather than paraphrasing it, so the reported criterion
and the frozen one cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CROSS_SEED_STABILITY_CRITERION",
    "CROSS_SEED_STABILITY_CRITERION_DESCRIPTION",
    "StabilityCriterion",
]


#: The frozen human-readable criterion, recorded verbatim in the
#: `decisions.md` Section 26 amendment dated 2026-09-07 and reported
#: verbatim by the Replication_Aggregator (Requirement 9.7).
#:
#: `tests/frozen/test_decisions_match.py` asserts this string appears
#: verbatim in that amendment entry, so editing it here without amending
#: `decisions.md` fails the suite.
CROSS_SEED_STABILITY_CRITERION_DESCRIPTION: str = (
    "Cross-seed stability holds if and only if the effect direction agrees in at "
    "least 4 of the 5 frozen replication seeds (101, 202, 303, 404, 505) and no "
    "required sensitivity variant contradicts the aggregate direction; otherwise "
    "the replication result is reported as directionally unstable with every "
    "per-seed value retained."
)


@dataclass(frozen=True)
class StabilityCriterion:
    """The cross-seed stability criterion the Replication_Aggregator applies.

    Frozen (immutable) because it is a `decisions.md` value, not a runtime
    knob: no caller may construct a different threshold for a run and have
    it read as the prespecified criterion. `aggregate()` (task 18.1) takes
    a `StabilityCriterion` argument so tests can exercise the aggregation
    logic at other thresholds, but the pipeline passes exactly
    `CROSS_SEED_STABILITY_CRITERION`.

    Attributes:
        min_agreeing_seeds: how many seeds must agree in effect direction
            with the aggregate direction. Frozen at 4.
        total_seeds: the frozen replication seed count (`decisions.md`
            Section 25's `[101, 202, 303, 404, 505]`). Frozen at 5. Stored
            alongside `min_agreeing_seeds` so the criterion reads as
            "4 of 5" on its own, and so an aggregator handed a different
            number of seed results can detect the mismatch instead of
            silently applying a 4-of-N rule.
        require_no_contradicting_sensitivity_variant: whether a required
            sensitivity variant contradicting the aggregate direction
            defeats stability on its own, regardless of seed agreement.
            Frozen at True -- the amendment's criterion is a conjunction.
        criterion_description: the human-readable criterion, reported
            verbatim alongside the determination (Requirement 9.7).
    """

    min_agreeing_seeds: int
    total_seeds: int
    require_no_contradicting_sensitivity_variant: bool
    criterion_description: str

    def __post_init__(self) -> None:
        if self.total_seeds < 1:
            raise ValueError(f"total_seeds must be positive, got {self.total_seeds}")
        if not 1 <= self.min_agreeing_seeds <= self.total_seeds:
            raise ValueError(
                "min_agreeing_seeds must lie in [1, total_seeds]; got "
                f"min_agreeing_seeds={self.min_agreeing_seeds}, "
                f"total_seeds={self.total_seeds}"
            )
        if not self.criterion_description.strip():
            raise ValueError("criterion_description must be a non-empty description")


#: The frozen cross-seed stability criterion (`decisions.md` Section 26
#: amendment dated 2026-09-07). The single value the Replication_Aggregator
#: applies; no other threshold is a permitted input to a reported run.
CROSS_SEED_STABILITY_CRITERION = StabilityCriterion(
    min_agreeing_seeds=4,
    total_seeds=5,
    require_no_contradicting_sensitivity_variant=True,
    criterion_description=CROSS_SEED_STABILITY_CRITERION_DESCRIPTION,
)
