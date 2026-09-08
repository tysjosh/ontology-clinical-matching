
# Known limitations

This document records limitations that are accepted by design rather than
defects to be fixed. Each entry states which requirement it resolves and why
the excluded approach is not viable for this project, so that a reviewer (or
a future contributor) does not mistake the absence of a check for an
oversight, and so that no downstream code or report is misread as claiming
to satisfy a check that was in fact ruled out. Later tasks may append further
entries below; existing entries must not be altered by later additions.

## Cross-site and calendar-time-shift robustness checks are out of scope

**Governs:** Requirement 13.2 ("chosen based on dataset support"), 13.8
("Cross-site evaluation SHALL be performed only when an external cohort is
legitimately available"), 13.9 ("Absence of cross-site validation SHALL be
reported as a limitation rather than approximated with a random split").
Frozen in `decisions.md` Section 20 ("Explicitly out of scope").

### Cross-site validation

Cross-site validation requires an external cohort collected at a different
site (or set of sites) from the training data, so that generalization across
sites can be assessed. ACTG175 is a single multi-center clinical trial
distributed as one pooled dataset: it does not ship as, and cannot be split
into, independently-recruited site-level cohorts suitable for a legitimate
cross-site evaluation, and no other clinically comparable external ACTG175
cohort is available to this project. Requirement 13.8's precondition — an
external cohort being *legitimately available* — is therefore not met, and
cross-site evaluation is not performed.

### Calendar-time (temporal) shift validation

Temporal-shift evaluation (Requirement 13.5) requires fitting on an earlier
enrollment period and evaluating on a later one, with a division that is
clinically meaningful and large enough on both sides to support estimation.
ACTG175 enrolled patients over a comparatively short, fixed window with no
documented protocol or standard-of-care change partway through, and the
dataset does not carry the kind of enrollment-date stratification that would
let an earlier/later split be justified as a meaningful stress test rather
than an arbitrary partition of one homogeneous enrollment period. ACTG175
therefore does not support calendar-time-shift validation credibly, and it is
excluded as a robustness check for this dataset (`decisions.md` Section 20).
This leaves clinically justified subgroup analysis (`prior_art_status`,
task 13) as the primary robustness check, consistent with Requirement 13.2.

### This is a reported limitation, not a substituted approximation

Per Requirement 13.9, the absence of cross-site and calendar-time-shift
validation is documented here as a limitation of the evaluation, not
compensated for by approximating either check with a random split of the
single available cohort. Concretely:

- No component of this pipeline (in particular the patient-level stratified
  split in `src/cohorts/splitting.py`, task 6.2) is to be interpreted, in
  code, in configuration, or in reporting, as standing in for cross-site
  validation. That split exists to produce development/validation/test
  partitions of one single-site-equivalent, single-trial cohort, and
  provides no evidence about generalization across sites.
- No component of this pipeline is to be interpreted as standing in for
  calendar-time-shift validation. In particular, no random or non-temporal
  split is presented as, or substituted for, an earlier-period /
  later-period evaluation.
- Any report or manuscript text generated from this project's results must
  state plainly that cross-site and calendar-time-shift robustness were not
  evaluated, for the dataset-support reasons above, rather than omitting the
  topic or implying it was addressed by another analysis.
