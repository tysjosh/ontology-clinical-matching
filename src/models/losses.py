"""Training objective for the intervention-aware model (task 9.3;
`decisions.md` Section 14.3 "Frozen training objective"):

    L = L_survival + lambda_state * L_CD4/CD8 + lambda_balance * L_representation_balance

- `L_survival`: discrete-time survival negative log-likelihood. This is
  ALREADY implemented in `models.training` (`discrete_time_censored_nll`
  / `censoring_aware_loss`, task 8.2) and is reused directly, not
  reimplemented here -- see `total_loss` below, which accepts the
  already-computed survival NLL as a plain float rather than
  recomputing it.
- `L_CD4/CD8`: masked Huber loss, see `masked_huber`.
- `L_representation_balance`: an MMD penalty between treated and
  comparator representations, see `mmd`.

`lambda_state` and `lambda_balance` are each drawn from the frozen 3x3
grid in `configs/experiments/lambda_grid.yaml` (`load_lambda_grid`) and
selected by validation-split composite-endpoint concordance only, never
development- or test-split data (`select_lambda_by_validation_concordance`).

Dependency scope note: as elsewhere in `src/models` (see
`training.py`'s module docstring; `pyproject.toml` declares no
`numpy`/`torch`), this module is pure standard library plus `PyYAML`
(already an approved dependency, used the same way as
`infra.config`/`data.contract`/`data.normalization`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from evaluation.metrics import concordance_index

#: Default location of the frozen 3x3 lambda grid (`decisions.md`
#: Section 14.3).
DEFAULT_LAMBDA_GRID_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "experiments" / "lambda_grid.yaml"
)


# ---------------------------------------------------------------------------
# L_CD4/CD8: masked Huber loss
# ---------------------------------------------------------------------------


def _huber(residual: float, delta: float) -> float:
    """Standard Huber loss for one scalar residual: quadratic for
    `|residual| <= delta`, linear beyond it (so a few large outlying
    residuals cannot dominate the loss the way a pure squared-error term
    would).
    """
    abs_residual = abs(residual)
    if abs_residual <= delta:
        return 0.5 * residual * residual
    return delta * (abs_residual - 0.5 * delta)


def masked_huber(
    predicted_cd4_cd8: Sequence[tuple[float, float]],
    observed_cd4_cd8: Sequence[tuple[float, float] | None],
    delta: float = 1.0,
) -> float:
    """Masked Huber loss over paired CD4/CD8 predictions.

    Masking convention: `observed_cd4_cd8[i]` is `None` whenever that
    patient/timestep's CD4/CD8 observation is MISSING (not recorded) --
    e.g. a patient with no week-96 draw. A `None` entry is skipped
    entirely: it contributes nothing to the sum and is not counted in the
    averaging denominator, and in particular it is never treated as an
    observed value of `0` (which would silently penalize the model for
    every missing timestep as if CD4/CD8 had actually fallen to zero).
    This mirrors the fold-safe/missingness-aware handling already used
    elsewhere in this codebase (`data.timelines`'s missingness mask;
    `cohorts.preprocessing`'s development-only fitting) -- missing data is
    excluded, not imputed as an error.

    Each observed pair contributes the sum of the per-component
    (CD4, CD8) Huber losses (`_huber`, threshold `delta`), and the
    returned value is the MEAN over observed pairs (i.e. mean of the
    two-component sums, not mean per component-observation) -- matching
    the frozen formula's single scalar `L_CD4/CD8` term.

    Args:
        predicted_cd4_cd8: `(predicted_cd4, predicted_cd8)` per
            patient/timestep.
        observed_cd4_cd8: `(observed_cd4, observed_cd8)` per
            patient/timestep, or `None` where the observation is missing.
            Must be the same length as `predicted_cd4_cd8`.
        delta: the Huber threshold separating quadratic from linear
            behavior. Defaults to `1.0`.

    Returns:
        The mean masked Huber loss over all non-`None` entries, or `0.0`
        if every entry is masked (no observed pairs to average over).

    Raises:
        ValueError: if the two sequences have different lengths.
    """
    if len(predicted_cd4_cd8) != len(observed_cd4_cd8):
        raise ValueError(
            "predicted_cd4_cd8 and observed_cd4_cd8 must have the same length, "
            f"got {len(predicted_cd4_cd8)} and {len(observed_cd4_cd8)}"
        )

    total = 0.0
    n_observed = 0
    for predicted, observed in zip(predicted_cd4_cd8, observed_cd4_cd8, strict=True):
        if observed is None:
            continue
        predicted_cd4, predicted_cd8 = predicted
        observed_cd4, observed_cd8 = observed
        total += _huber(predicted_cd4 - observed_cd4, delta) + _huber(
            predicted_cd8 - observed_cd8, delta
        )
        n_observed += 1

    if n_observed == 0:
        return 0.0
    return total / n_observed


# ---------------------------------------------------------------------------
# L_representation_balance: linear-kernel MMD
# ---------------------------------------------------------------------------


def mmd(z_treated: Sequence[Sequence[float]], z_control: Sequence[Sequence[float]]) -> float:
    """Linear-kernel Maximum Mean Discrepancy penalty between two sets of
    representation vectors.

    Kernel-choice note: MMD is defined generically as
    `||E[phi(z_treated)] - E[phi(z_control)]||^2` for some feature map
    `phi`. This implements the simplest standard variant, the LINEAR
    kernel (`phi = identity`):

        MMD^2 = || mean(z_treated) - mean(z_control) ||^2

    rather than an RBF-kernel MMD with bandwidth selection. The linear
    kernel is a legitimate, well-known MMD variant (it is exactly the
    squared distance between the two group means in representation
    space); it is chosen here for its simplicity and because it needs no
    bandwidth hyperparameter or pairwise kernel-matrix computation, which
    keeps this pure-Python/no-numpy implementation cheap. An RBF-kernel
    MMD (which additionally penalizes higher-moment distributional
    differences beyond the mean) could be substituted later without
    changing `total_loss`'s signature.

    Args:
        z_treated: representation vectors for the treated group. Every
            vector must have the same length.
        z_control: representation vectors for the control/comparator
            group. Every vector must have the same length as the
            `z_treated` vectors.

    Returns:
        The squared linear-kernel MMD (`>= 0.0`; exactly `0.0` when the
        two groups' mean representations coincide). Returns `0.0` if
        either group is empty (no balance penalty can be computed without
        at least one representation per group).

    Raises:
        ValueError: if any vector's length disagrees with the others.
    """
    if not z_treated or not z_control:
        return 0.0

    dimension = len(z_treated[0])
    for group in (z_treated, z_control):
        for vector in group:
            if len(vector) != dimension:
                raise ValueError(
                    "all representation vectors must have the same length, "
                    f"expected {dimension}, got {len(vector)}"
                )

    mean_treated = [sum(values) / len(z_treated) for values in zip(*z_treated, strict=True)]
    mean_control = [sum(values) / len(z_control) for values in zip(*z_control, strict=True)]

    return sum((t - c) ** 2 for t, c in zip(mean_treated, mean_control, strict=True))


# ---------------------------------------------------------------------------
# total_loss: thin combination of the three frozen terms
# ---------------------------------------------------------------------------


def total_loss(
    survival_nll_value: float,
    masked_huber_value: float,
    mmd_value: float,
    lambda_state: float,
    lambda_balance: float,
) -> float:
    """The frozen training-objective formula (`decisions.md` Section 14.3):

        L = L_survival + lambda_state * L_CD4/CD8 + lambda_balance * L_representation_balance

    This function is a THIN, pure combination: it takes the three loss
    terms as already-computed scalars rather than recomputing any of
    them. In particular, `survival_nll_value` is expected to come from
    `models.training.censoring_aware_loss` /
    `discrete_time_censored_nll` (task 8.2's discrete-time survival NLL)
    computed by the CALLER -- this module does not reimplement it.
    `masked_huber_value` and `mmd_value` are expected to come from
    `masked_huber` and `mmd` above.

    Args:
        survival_nll_value: the discrete-time survival negative
            log-likelihood (`L_survival`), already computed by the caller.
        masked_huber_value: `L_CD4/CD8`, already computed via
            `masked_huber`.
        mmd_value: `L_representation_balance`, already computed via `mmd`.
        lambda_state: the CD4/CD8 loss weight, one grid value from
            `configs/experiments/lambda_grid.yaml`.
        lambda_balance: the representation-balance loss weight, one grid
            value from `configs/experiments/lambda_grid.yaml`.

    Returns:
        The combined scalar training loss.
    """
    return survival_nll_value + lambda_state * masked_huber_value + lambda_balance * mmd_value


# ---------------------------------------------------------------------------
# Frozen lambda grid loading
# ---------------------------------------------------------------------------


def load_lambda_grid(path: Path | str = DEFAULT_LAMBDA_GRID_PATH) -> list[tuple[float, float]]:
    """Load the frozen 3x3 `(lambda_state, lambda_balance)` grid from
    `configs/experiments/lambda_grid.yaml`.

    The YAML file stores `lambda_state` and `lambda_balance` as two
    separate three-value lists; this function crosses them (in
    `lambda_state`-major, `lambda_balance`-minor order) to produce the 9
    frozen combinations `decisions.md` Section 14.3 specifies.

    Args:
        path: location of the lambda-grid YAML file. Defaults to
            `configs/experiments/lambda_grid.yaml` at the repository root.

    Returns:
        A list of 9 `(lambda_state, lambda_balance)` tuples.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    lambda_state_values = data["lambda_state"]
    lambda_balance_values = data["lambda_balance"]

    return [
        (float(lambda_state), float(lambda_balance))
        for lambda_state in lambda_state_values
        for lambda_balance in lambda_balance_values
    ]


# ---------------------------------------------------------------------------
# Lambda selection by validation composite-endpoint concordance ONLY
# ---------------------------------------------------------------------------


def select_lambda_by_validation_concordance(
    candidate_results: dict[tuple[float, float], tuple[Sequence[float], Sequence[float], Sequence[bool]]],
) -> tuple[float, float]:
    """Select the `(lambda_state, lambda_balance)` combination with the
    highest VALIDATION-split composite-endpoint concordance.

    `decisions.md` Section 14.3: "lambda_state and lambda_balance are
    selected on validation data only, via a small prespecified grid ...
    selected by validation composite-endpoint concordance." This function
    structurally enforces the "validation only" part of that sentence:
    its ONLY input is `candidate_results`, a mapping from each of the 9
    frozen lambda combinations (see `load_lambda_grid`) to that
    combination's VALIDATION-split `(risk_scores, event_times,
    event_indicators)` triple -- i.e. the caller has already trained a
    model under each lambda combination on the development split and
    scored it on the validation split before calling this function. There
    is NO parameter anywhere in this signature through which
    development-split-only or test-split data could be passed in: unlike
    the typed `DevelopmentSet`-wrapper enforcement pattern used in
    `cohorts.preprocessing.FittedPreprocessing.fit` (which rejects a
    wrong-typed argument at call time), this function enforces the same
    "only validation data" guarantee even more simply -- by never
    declaring a parameter through which anything else could be passed at
    all. That is an acceptable, simpler enforcement mechanism given this
    function's narrow, single purpose.

    Args:
        candidate_results: for each of the 9 `(lambda_state,
            lambda_balance)` combinations, that combination's
            validation-split `(risk_scores, event_times,
            event_indicators)`, suitable for
            `evaluation.metrics.concordance_index`.

    Returns:
        The `(lambda_state, lambda_balance)` combination with the highest
        validation concordance index.

    Raises:
        ValueError: if `candidate_results` is empty.
    """
    if not candidate_results:
        raise ValueError("candidate_results must not be empty")

    best_combination: tuple[float, float] | None = None
    best_concordance = float("-inf")

    for combination, (risk_scores, event_times, event_indicators) in candidate_results.items():
        combination_concordance = concordance_index(risk_scores, event_times, event_indicators)
        if combination_concordance > best_concordance:
            best_concordance = combination_concordance
            best_combination = combination

    assert best_combination is not None  # candidate_results is non-empty (checked above)
    return best_combination
