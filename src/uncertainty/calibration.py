"""Calibration fitting on validation data only (Requirements 10.3, 10.4;
`design.md` Section 3.7, `decisions.md` Section 15).

Requirement 10.3 states "Calibration SHALL be fitted on validation data
only." Requirement 10.4 states "Factual-outcome calibration SHALL include
Brier score, calibration slope, calibration intercept, and a reliability
diagram where applicable."

Why a `ValidationSet` wrapper, not a bare sequence pair
---------------------------------------------------------

`Calibrator.fit()` must reject anything that is not explicitly validation
data -- including a bare `list`/`tuple` of predictions, which is exactly
the shape (`Sequence[float]`) a caller might otherwise pass by mistake
from a development or test split. A bare `Sequence[float]` carries no
"which split am I" tag, so the enforcement cannot live in a runtime check
on the sequence's contents alone.

This module therefore follows the *same* pattern already established by
`cohorts.preprocessing.DevelopmentSet` (see that module's docstring) and
`data.imputation.DevelopmentFold`: a thin, frozen wrapper --
`ValidationSet` -- that is the explicit, visible marker that the
predictions/labels it carries are validation-split data.
`Calibrator.fit()` requires a `ValidationSet` instance and raises
`ValidationSetRequiredError` for anything else, including an unwrapped
`Sequence[float]` pair. A caller must explicitly write
`ValidationSet(predictions=val_predictions, labels=val_labels)` to fit --
passing bare lists is a visible, conspicuous mismatch at the call site,
not a silent bug.

Metrics computed
-----------------

- Brier score: the plain `mean((prediction_i - label_i)^2)`. This is
  deliberately *not* `evaluation.metrics.ipcw_brier_score`, which is an
  IPCW (censoring-weighted) estimator specific to the 96-week survival
  endpoint; calibration here is a general-purpose check over any binary
  factual outcome, so the simpler, non-survival-specific Brier score is
  the right tool.
- Calibration slope/intercept: delegates directly to
  `evaluation.metrics.calibration_slope_intercept`, which already
  implements the from-scratch logistic fit of
  `logit(Y) = intercept + slope * logit(predicted_risk)`. This module
  does not re-derive that fit.
- Reliability diagram: predictions are bucketed into `n_bins` fixed-width
  bins over `[0, 1]` (default 10, i.e. `[0, 0.1), [0.1, 0.2), ...,
  [0.9, 1.0]`, with the final bin closed on both ends so a prediction of
  exactly `1.0` is not dropped). Each non-empty bin contributes one
  `(mean_predicted, mean_observed, count)` tuple, in bin order -- the
  standard reliability-diagram data structure. Empty bins are omitted
  rather than reported as `(nan, nan, 0)`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evaluation.metrics import calibration_slope_intercept

#: Default number of equal-width bins over [0, 1] for the reliability diagram.
DEFAULT_N_BINS = 10


class ValidationSetRequiredError(TypeError):
    """Raised when `.fit()` is called with anything other than a `ValidationSet`.

    This is the enforcement mechanism for Requirement 10.3: calibration
    must be fitted on validation data only. A bare `Sequence[float]` pair
    of predictions/labels -- even one that happens to actually be
    validation data -- is rejected outright, because a plain sequence
    carries no "which split am I" tag. The caller must explicitly
    construct a `ValidationSet` to fit on it, which makes any accidental
    fit-on-development/test call visible at the call site rather than
    silently wrong. Mirrors `cohorts.preprocessing.DevelopmentSetRequiredError`.
    """

    def __init__(self, received: object) -> None:
        self.received = received
        super().__init__(
            "Calibrator.fit() requires a ValidationSet, got "
            f"{type(received).__name__!r}. Wrap the validation predictions "
            "and labels in ValidationSet(predictions=..., labels=...) to "
            "fit on them; bare lists/tuples/arrays -- and development/test "
            "predictions -- must never be passed to .fit() directly."
        )


@dataclass(frozen=True)
class ValidationSet:
    """An explicit, visible marker that `predictions`/`labels` are validation-split data.

    Analogous to `cohorts.preprocessing.DevelopmentSet`: `Calibrator.fit()`
    only accepts a `ValidationSet`, so a caller cannot pass a bare
    predictions/labels pair (development, test, or otherwise) to `.fit()`
    without explicitly (and visibly) constructing a `ValidationSet` around
    them first -- which is itself a conspicuous mislabeling, not an
    accident.

    Attributes:
        predictions: validation-split predicted probabilities in `[0, 1]`.
        labels: validation-split binary observed outcomes, paired
            index-for-index with `predictions`.
    """

    predictions: Sequence[float]
    labels: Sequence[bool]

    def __post_init__(self) -> None:
        if len(self.predictions) != len(self.labels):
            raise ValueError(
                "ValidationSet requires predictions and labels of equal "
                f"length, got {len(self.predictions)} and {len(self.labels)}."
            )


@dataclass(frozen=True)
class CalibrationResult:
    """The metrics produced by `Calibrator.fit()`.

    Attributes:
        brier_score: `mean((prediction_i - label_i)^2)`.
        calibration_intercept: from
            `evaluation.metrics.calibration_slope_intercept`.
        calibration_slope: from
            `evaluation.metrics.calibration_slope_intercept`.
        reliability_diagram: one `(mean_predicted, mean_observed, count)`
            tuple per non-empty bin, in bin order (see module docstring).
    """

    brier_score: float
    calibration_intercept: float
    calibration_slope: float
    reliability_diagram: list[tuple[float, float, int]]


def _brier_score(predictions: Sequence[float], labels: Sequence[bool]) -> float:
    observed = [1.0 if label else 0.0 for label in labels]
    return sum(
        (prediction - outcome) ** 2 for prediction, outcome in zip(predictions, observed, strict=True)
    ) / len(predictions)


def _reliability_diagram(
    predictions: Sequence[float], labels: Sequence[bool], n_bins: int
) -> list[tuple[float, float, int]]:
    bin_width = 1.0 / n_bins
    # bin_predictions[b] / bin_observed[b] accumulate raw values for bin b;
    # the last bin (index n_bins - 1) is closed on both ends so a
    # prediction of exactly 1.0 falls into it rather than an (n_bins)-th,
    # out-of-range bin.
    bin_predictions: list[list[float]] = [[] for _ in range(n_bins)]
    bin_observed: list[list[float]] = [[] for _ in range(n_bins)]

    for prediction, label in zip(predictions, labels, strict=True):
        bin_index = int(prediction / bin_width)
        if bin_index >= n_bins:
            bin_index = n_bins - 1
        elif bin_index < 0:
            bin_index = 0
        bin_predictions[bin_index].append(prediction)
        bin_observed[bin_index].append(1.0 if label else 0.0)

    diagram: list[tuple[float, float, int]] = []
    for preds, observed in zip(bin_predictions, bin_observed):
        if not preds:
            continue
        count = len(preds)
        mean_predicted = sum(preds) / count
        mean_observed = sum(observed) / count
        diagram.append((mean_predicted, mean_observed, count))
    return diagram


@dataclass
class Calibrator:
    """Fits calibration metrics on validation data only (Requirement 10.3).

    Attributes:
        n_bins: number of equal-width reliability-diagram bins over
            `[0, 1]`. Defaults to `DEFAULT_N_BINS` (10).
    """

    n_bins: int = DEFAULT_N_BINS
    _result: CalibrationResult | None = None

    def fit(self, validation_data: ValidationSet) -> CalibrationResult:
        """Compute Brier score, calibration slope/intercept, and a reliability diagram.

        Args:
            validation_data: a `ValidationSet` wrapping validation-split
                predictions and labels. Passing anything else (including
                a bare `Sequence[float]`/`Sequence[bool]` pair) raises
                `ValidationSetRequiredError` -- this is the enforced
                "fit only on validation data" contract.

        Returns:
            A `CalibrationResult` bundling all three metrics. Also stored
            on `self.result_` for later inspection.

        Raises:
            ValidationSetRequiredError: if `validation_data` is not a
                `ValidationSet`.
        """
        if not isinstance(validation_data, ValidationSet):
            raise ValidationSetRequiredError(validation_data)

        predictions = validation_data.predictions
        labels = validation_data.labels

        brier = _brier_score(predictions, labels)
        intercept, slope = calibration_slope_intercept(predictions, labels)
        diagram = _reliability_diagram(predictions, labels, self.n_bins)

        self._result = CalibrationResult(
            brier_score=brier,
            calibration_intercept=intercept,
            calibration_slope=slope,
            reliability_diagram=diagram,
        )
        return self._result

    @property
    def result_(self) -> CalibrationResult:
        """The `CalibrationResult` produced by the most recent `.fit()` call."""
        if self._result is None:
            raise RuntimeError("Calibrator.fit() has not been called yet.")
        return self._result
