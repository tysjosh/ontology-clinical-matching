"""Converged, deterministic Newton-type solvers for the in-house estimators.

Neural-experiment-pipeline task 7.7. Every conventional fit in this
repository used to be fixed-budget batch gradient descent with no stopping
tolerance (500 or 300 iterations, learning rate 0.1 or 0.05). On the
canonical development cohort that left the propensity model unconverged
(gradient norm ~1e-2) and shrank every Cox log-hazard ratio 40-50% toward
the null (`unadjusted_cox` -0.377 shipped vs -0.641 at the optimum); see
`.agents/tasks/propensity-convergence-investigation/findings.md`.

This module replaces the optimizer only. Each solver minimizes the same
objective the old loop descended:

- `fit_logistic_newton`: mean binary cross-entropy of
  `sigmoid(bias + w . x)`, by IRLS (Newton on the convex objective);
- `fit_linear_exact`: mean squared error of `bias + w . x`, by one exact
  normal-equation solve (Newton converges in a single step on a quadratic);
- `fit_cox_newton`: the negative weighted Breslow partial log-likelihood
  of `causal.baselines._fit_cox` (same tie handling, same weights, same
  linear-predictor clip `eta_clip`), by Newton-Raphson.

Frozen rule (`decisions.md` Section 26, task 7.9):

- zero initialization;
- Newton step from the exact gradient and Hessian; if the objective does
  not improve, the step is halved, at most `MAX_STEP_HALVINGS` times;
- converged when the accepted step has max absolute component below
  `SOLVER_TOLERANCE`;
- at most `SOLVER_MAX_ITERATIONS` Newton iterations.

A fit that reaches the cap, meets a singular Hessian, fails the line
search, or produces a non-finite coefficient reports `converged = False`
with the reason. Callers raise `SolverConvergenceError` on that rather than
use the coefficients.

Two conventions keep the solution well defined and identical to what the
old zero-initialized descent converged toward:

- a feature column that is exactly zero on every row carries no
  information (gradient descent from zero never moved its weight), so its
  coefficient is fixed at `0.0` and it is left out of the Newton system.
  This is the minimum-norm solution and happens, e.g., in a bootstrap
  resample where a rare binary covariate is absent;
- if the gradient at the zero start is already exactly below the
  tolerance (e.g. a Cox fit with no events), the start is the optimum;
- the Newton step is the minimum-norm step `H^+ g` (symmetric
  pseudo-inverse; eigenvalues below `_RANK_TOLERANCE` times the largest are
  treated as zero). With a full-rank design (the real cohort: Hessian
  condition number 106-208) this is the ordinary Newton step. With a
  rank-deficient design (a constant column collinear with the intercept, or
  duplicated columns, as in small test cohorts) the iterates stay in the
  row space of the design, so the solver returns the minimum-norm optimum,
  which is also what zero-initialized gradient descent converges to. The
  diagnostic records the numerical `rank`. Perfect separation has no finite
  optimum under either rule and still reports `converged = False`.

Pure Python, no third-party dependency, torch-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: Stop when the accepted Newton step's max |component| is below this.
SOLVER_TOLERANCE: float = 1e-8

#: Hard cap on Newton iterations.
SOLVER_MAX_ITERATIONS: int = 100

#: Maximum step halvings per iteration before the line search fails.
MAX_STEP_HALVINGS: int = 30

#: Relative pivot threshold below which a matrix is treated as singular.
_SINGULAR_PIVOT: float = 1e-13

#: Relative eigenvalue threshold for the Hessian pseudo-inverse.
_RANK_TOLERANCE: float = 1e-10

#: Slack for "objective did not increase" under float rounding.
_OBJECTIVE_SLACK: float = 1e-12


@dataclass(frozen=True)
class ConvergenceDiagnostic:
    """What one solver run did, recorded alongside its coefficients."""

    optimizer: str
    iterations: int
    final_max_abs_step: float
    final_gradient_norm: float
    final_objective: float
    converged: bool
    reason: str
    rank: int = -1
    n_params: int = -1

    def as_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer,
            "iterations": self.iterations,
            "final_max_abs_step": self.final_max_abs_step,
            "final_gradient_norm": self.final_gradient_norm,
            "final_objective": self.final_objective,
            "converged": self.converged,
            "reason": self.reason,
            "rank": self.rank,
            "n_params": self.n_params,
        }


class SolverConvergenceError(RuntimeError):
    """A fit did not meet the frozen convergence rule; its coefficients are not used."""

    def __init__(self, context: str, diagnostic: ConvergenceDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"{context}: {diagnostic.optimizer} did not converge "
            f"({diagnostic.reason}; iterations={diagnostic.iterations}, "
            f"max|step|={diagnostic.final_max_abs_step:.3e}, "
            f"|grad|={diagnostic.final_gradient_norm:.3e})"
        )


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def solve_linear_system(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> list[float] | None:
    """Solve `matrix @ x = rhs` by Gaussian elimination with partial pivoting.

    Returns `None` if the matrix is singular to working precision (a pivot
    below `_SINGULAR_PIVOT` times the largest entry) or not finite.
    """
    n = len(matrix)
    if n == 0:
        return []
    augmented = [[*map(float, row), float(value)] for row, value in zip(matrix, rhs, strict=True)]
    scale = max((abs(v) for row in augmented for v in row[:n]), default=0.0)
    if scale == 0.0 or not math.isfinite(scale):
        return None
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) <= _SINGULAR_PIVOT * scale:
            return None
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        for row in range(col + 1, n):
            factor = augmented[row][col] / pivot
            if factor != 0.0:
                augmented[row] = [
                    a - factor * b for a, b in zip(augmented[row], augmented[col], strict=True)
                ]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = augmented[row][n] - sum(augmented[row][k] * solution[k] for k in range(row + 1, n))
        solution[row] = total / augmented[row][row]
    if not all(math.isfinite(v) for v in solution):
        return None
    return solution


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _symmetric_eigen(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[list[float]]]:
    """Cyclic Jacobi eigendecomposition of a small symmetric matrix.

    Returns `(eigenvalues, eigenvectors)` with `eigenvectors[i]` the i-th
    eigenvector. Deterministic; intended for the <= ~20-dimensional
    Hessians in this module.
    """
    n = len(matrix)
    a = [list(map(float, row)) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    norm = math.sqrt(sum(x * x for row in a for x in row))
    if norm == 0.0:
        return [0.0] * n, v
    for _ in range(100):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off <= 1e-15 * norm:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if a[p][q] == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    eigenvalues = [a[i][i] for i in range(n)]
    eigenvectors = [[v[k][i] for k in range(n)] for i in range(n)]
    return eigenvalues, eigenvectors


def pseudo_inverse_solve(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> tuple[list[float], int] | None:
    """Minimum-norm solution of `matrix @ x = rhs` for a symmetric PSD matrix.

    Returns `(x, rank)`, or `None` if the matrix is not finite or has no
    eigenvalue above the rank threshold.
    """
    if not all(math.isfinite(x) for row in matrix for x in row):
        return None
    eigenvalues, eigenvectors = _symmetric_eigen(matrix)
    largest = max((abs(e) for e in eigenvalues), default=0.0)
    if largest == 0.0:
        return None
    cutoff = _RANK_TOLERANCE * largest
    solution = [0.0] * len(rhs)
    rank = 0
    for value, vector in zip(eigenvalues, eigenvectors, strict=True):
        if value <= cutoff:
            continue
        rank += 1
        coefficient = sum(u * r for u, r in zip(vector, rhs, strict=True)) / value
        for i, u in enumerate(vector):
            solution[i] += coefficient * u
    if not all(math.isfinite(x) for x in solution):
        return None
    return solution, rank


# ---------------------------------------------------------------------------
# Generic damped Newton
# ---------------------------------------------------------------------------

#: objective(theta) -> f ; derivatives(theta) -> (f, gradient, hessian)
Objective = Callable[[Sequence[float]], float]
Derivatives = Callable[[Sequence[float]], tuple[float, list[float], list[list[float]]]]


def _newton_minimize(
    n_params: int, objective: Objective, derivatives: Derivatives, optimizer: str
) -> tuple[list[float], ConvergenceDiagnostic]:
    theta = [0.0] * n_params
    f, gradient, hessian = derivatives(theta)
    last_step = float("inf")
    rank = -1

    def diagnostic(iterations: int, converged: bool, reason: str) -> ConvergenceDiagnostic:
        return ConvergenceDiagnostic(
            optimizer=optimizer,
            iterations=iterations,
            final_max_abs_step=last_step,
            final_gradient_norm=_norm(gradient),
            final_objective=f,
            converged=converged,
            reason=reason,
            rank=rank,
            n_params=n_params,
        )

    if n_params == 0 or max((abs(g) for g in gradient), default=0.0) < SOLVER_TOLERANCE:
        last_step = 0.0
        return theta, diagnostic(0, True, "zero_gradient_at_start")

    for iteration in range(1, SOLVER_MAX_ITERATIONS + 1):
        solved = pseudo_inverse_solve(hessian, gradient)
        if solved is None:
            return theta, diagnostic(iteration - 1, False, "singular_hessian")
        step, rank = solved
        full_max = max(abs(s) for s in step)
        if full_max < SOLVER_TOLERANCE:
            theta = [t - s for t, s in zip(theta, step, strict=True)]
            last_step = full_max
            f, gradient, hessian = derivatives(theta)
            return theta, diagnostic(iteration, True, "step_below_tolerance")

        scale = 1.0
        accepted = None
        for _ in range(MAX_STEP_HALVINGS + 1):
            candidate = [t - scale * s for t, s in zip(theta, step, strict=True)]
            if all(math.isfinite(c) for c in candidate):
                f_candidate = objective(candidate)
                limit = f + _OBJECTIVE_SLACK * max(1.0, abs(f))
                if math.isfinite(f_candidate) and f_candidate <= limit:
                    accepted = candidate
                    break
            scale *= 0.5
        if accepted is None:
            return theta, diagnostic(iteration - 1, False, "line_search_failed")

        theta = accepted
        last_step = scale * full_max
        f, gradient, hessian = derivatives(theta)
        if not all(math.isfinite(v) for v in (*theta, f, *gradient)):
            return theta, diagnostic(iteration, False, "non_finite")
        if last_step < SOLVER_TOLERANCE:
            return theta, diagnostic(iteration, True, "step_below_tolerance")

    return theta, diagnostic(SOLVER_MAX_ITERATIONS, False, "iteration_cap")


def _active_columns(features: Sequence[Sequence[float]], width: int) -> list[int]:
    """Columns not identically zero; the rest keep coefficient 0.0."""
    return [j for j in range(width) if any(row[j] != 0.0 for row in features)]


def _expand(active: Sequence[int], width: int, values: Sequence[float]) -> list[float]:
    full = [0.0] * width
    for j, value in zip(active, values, strict=True):
        full[j] = value
    return full


# ---------------------------------------------------------------------------
# Logistic regression (IRLS)
# ---------------------------------------------------------------------------


def _log1p_exp(z: float) -> float:
    """Numerically stable `log(1 + exp(z))`."""
    if z > 0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic_newton(
    features: Sequence[Sequence[float]], labels: Sequence[float]
) -> tuple[float, list[float], ConvergenceDiagnostic]:
    """Minimize mean BCE of `sigmoid(bias + w . x)`. Returns `(bias, weights, diagnostic)`."""
    n = len(features)
    width = len(features[0]) if n else 0
    if n == 0:
        raise ValueError("fit_logistic_newton needs at least one row")
    active = _active_columns(features, width)
    rows = [[1.0, *(row[j] for j in active)] for row in features]
    y = [float(v) for v in labels]
    k = len(rows[0])

    def objective(theta: Sequence[float]) -> float:
        total = 0.0
        for row, label in zip(rows, y, strict=True):
            z = sum(t * x for t, x in zip(theta, row, strict=True))
            total += _log1p_exp(z) - label * z
        return total / n

    def derivatives(theta: Sequence[float]):
        total = 0.0
        gradient = [0.0] * k
        hessian = [[0.0] * k for _ in range(k)]
        for row, label in zip(rows, y, strict=True):
            z = sum(t * x for t, x in zip(theta, row, strict=True))
            total += _log1p_exp(z) - label * z
            p = _sigmoid(z)
            error = p - label
            weight = p * (1.0 - p)
            for a in range(k):
                gradient[a] += error * row[a]
                wa = weight * row[a]
                hrow = hessian[a]
                for b in range(a, k):
                    hrow[b] += wa * row[b]
        for a in range(k):
            gradient[a] /= n
            for b in range(a, k):
                hessian[a][b] /= n
                hessian[b][a] = hessian[a][b]
        return total / n, gradient, hessian

    theta, diag = _newton_minimize(k, objective, derivatives, "logistic_irls")
    return theta[0], _expand(active, width, theta[1:]), diag


# ---------------------------------------------------------------------------
# Linear regression (exact)
# ---------------------------------------------------------------------------


def fit_linear_exact(
    features: Sequence[Sequence[float]], labels: Sequence[float]
) -> tuple[float, list[float], ConvergenceDiagnostic]:
    """Minimize mean squared error of `bias + w . x`. Returns `(bias, weights, diagnostic)`."""
    n = len(features)
    width = len(features[0]) if n else 0
    if n == 0:
        raise ValueError("fit_linear_exact needs at least one row")
    active = _active_columns(features, width)
    rows = [[1.0, *(row[j] for j in active)] for row in features]
    y = [float(v) for v in labels]
    k = len(rows[0])

    def objective(theta: Sequence[float]) -> float:
        total = 0.0
        for row, label in zip(rows, y, strict=True):
            total += (sum(t * x for t, x in zip(theta, row, strict=True)) - label) ** 2
        return total / (2.0 * n)

    def derivatives(theta: Sequence[float]):
        gradient = [0.0] * k
        hessian = [[0.0] * k for _ in range(k)]
        total = 0.0
        for row, label in zip(rows, y, strict=True):
            residual = sum(t * x for t, x in zip(theta, row, strict=True)) - label
            total += residual * residual
            for a in range(k):
                gradient[a] += residual * row[a]
                for b in range(a, k):
                    hessian[a][b] += row[a] * row[b]
        for a in range(k):
            gradient[a] /= n
            for b in range(a, k):
                hessian[a][b] /= n
                hessian[b][a] = hessian[a][b]
        return total / (2.0 * n), gradient, hessian

    theta, diag = _newton_minimize(k, objective, derivatives, "linear_least_squares")
    return theta[0], _expand(active, width, theta[1:]), diag


# ---------------------------------------------------------------------------
# Weighted Breslow Cox partial likelihood (Newton-Raphson)
# ---------------------------------------------------------------------------


def fit_cox_newton(
    features: Sequence[Sequence[float]],
    event_times: Sequence[object],
    event_flags: Sequence[bool],
    weights: Sequence[float],
    eta_clip: float,
) -> tuple[list[float], ConvergenceDiagnostic, list[list[float]]]:
    """Maximize the weighted Breslow partial log-likelihood of `causal.baselines._fit_cox`.

    Rows must already be sorted by ascending `event_times`. The objective is
    the negative log-likelihood divided by `n` (the old descent's scaling).

    Returns `(beta, diagnostic, information)`, where `information` is the
    unscaled observed-information matrix at `beta` (the same quantity
    `_fit_cox` inverts for its standard error).
    """
    n = len(features)
    width = len(features[0]) if n else 0
    active = _active_columns(features, width)
    rows = [[row[j] for j in active] for row in features]
    k = len(active)

    groups: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n and event_times[end + 1] == event_times[start]:
            end += 1
        groups.append((start, end - start + 1))
        start = end + 1

    def etas(beta: Sequence[float]) -> list[float]:
        return [
            max(-eta_clip, min(eta_clip, sum(b * x for b, x in zip(beta, row, strict=True))))
            for row in rows
        ]

    def objective(beta: Sequence[float]) -> float:
        eta = etas(beta)
        suffix = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + weights[i] * math.exp(eta[i])
        loglik = 0.0
        for g_start, size in groups:
            d_weight = 0.0
            linear = 0.0
            for i in range(g_start, g_start + size):
                if event_flags[i]:
                    d_weight += weights[i]
                    linear += weights[i] * eta[i]
            if d_weight == 0.0 or suffix[g_start] <= 0.0:
                continue
            loglik += linear - d_weight * math.log(suffix[g_start])
        return -loglik / n

    def full(beta: Sequence[float]):
        eta = etas(beta)
        exp_w = [weights[i] * math.exp(eta[i]) for i in range(n)]
        s0 = [0.0] * (n + 1)
        s1 = [[0.0] * k for _ in range(n + 1)]
        s2 = [[[0.0] * k for _ in range(k)] for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            row = rows[i]
            s0[i] = s0[i + 1] + exp_w[i]
            s1[i] = [s1[i + 1][a] + exp_w[i] * row[a] for a in range(k)]
            s2[i] = [
                [s2[i + 1][a][b] + exp_w[i] * row[a] * row[b] for b in range(k)] for a in range(k)
            ]
        loglik = 0.0
        score = [0.0] * k
        information = [[0.0] * k for _ in range(k)]
        for g_start, size in groups:
            d_weight = 0.0
            linear = 0.0
            sum_x = [0.0] * k
            for i in range(g_start, g_start + size):
                if event_flags[i]:
                    d_weight += weights[i]
                    linear += weights[i] * eta[i]
                    for a in range(k):
                        sum_x[a] += weights[i] * rows[i][a]
            if d_weight == 0.0 or s0[g_start] <= 0.0:
                continue
            total = s0[g_start]
            mean = [s1[g_start][a] / total for a in range(k)]
            loglik += linear - d_weight * math.log(total)
            for a in range(k):
                score[a] += sum_x[a] - d_weight * mean[a]
                for b in range(k):
                    information[a][b] += d_weight * (s2[g_start][a][b] / total - mean[a] * mean[b])
        return -loglik / n, score, information

    def derivatives(beta: Sequence[float]):
        f, score, information = full(beta)
        return f, [-s / n for s in score], [[v / n for v in row] for row in information]

    beta_active, diag = _newton_minimize(k, objective, derivatives, "cox_newton_raphson")
    _, _, information_active = full(beta_active)
    information = [[0.0] * width for _ in range(width)]
    for ai, a in enumerate(active):
        for bi, b in enumerate(active):
            information[a][b] = information_active[ai][bi]
    return _expand(active, width, beta_active), diag, information
