"""Deep ensemble for predictive uncertainty (Requirement 10.1, 10.2;
`design.md` Section 3.7, `decisions.md` Section 15).

`DeepEnsemble` is the primary predictive-uncertainty method for this
project (Requirement 10.1). It must work with *any* model class in this
codebase's `src/models` package -- `models.non_neural_baseline.CoxBaseline`,
`models.sequence_baseline.SequenceGRU`, and (once it exists)
`models.intervention_aware.InterventionAwareModel` -- without importing any
of them, so this module stays model-agnostic and dependency-free (pure
standard library, matching the rest of `src/models`/`src/causal`).

Design decision -- generic factory + generic predict function
---------------------------------------------------------------
The obvious "wrap any model class" design runs into a real obstacle in
this codebase: the models above do **not** share a common prediction
method name. `CoxBaseline` exposes `predict_risk(patient)`;
`SequenceGRU` exposes `forward(baseline, sequence)`; a future model may
expose something else again. A single hardcoded `model.predict(x)` call
would therefore only work for one model family and silently break for
the others.

`DeepEnsemble` instead takes two caller-supplied callables:

- `model_factory(seed) -> model`: constructs (and, if the caller wants,
  trains/fits) one replica given an integer seed. What "constructing a
  replica" means -- e.g. `lambda seed: CoxBaseline().fit(patients)` vs.
  `lambda seed: SequenceGRU.initialize(..., seed=seed)` -- is entirely up
  to the caller; `DeepEnsemble` only needs the result back.
- `predict_fn(model, x) -> float`: adapts whatever prediction method the
  caller's specific model class exposes (`model.predict_risk(x)`,
  `model.forward(*x)`, etc.) to a single scalar output, so `DeepEnsemble`
  itself never needs to know the method name.

This keeps `DeepEnsemble` fully generic while still letting every
existing (and future) model class in `src/models` use it unmodified.

API
---
Seeds are derived eagerly at construction time (not lazily on first
`.predict()` call): `DeepEnsemble.__init__` immediately derives
`n_replicas` distinct seeds via
`infra.seeding.derive_seed(master_seed, f"replica_{i}")` and calls
`model_factory(seed)` once per replica, so that by the time construction
returns, all `n_replicas` replicas exist and `.predict()` is a pure
read-only operation over them. Deriving seeds through `derive_seed`
(rather than e.g. `master_seed + i`) keeps replica seeding consistent
with every other seeded sub-stream in this codebase (train/val/test
splits, bootstrap replicates, negative-control sham labels) and
auditable from the manifest (Requirement 1.6).

`.predict(x) -> (mean, variance)` is a scalar-per-call API (one input
`x`, one `(mean, variance)` pair) rather than a batched one -- the
simplest interface that satisfies `design.md` Section 3.7's
`.predict(x) -> (mean, variance)` contract; callers needing per-patient
uncertainty over a batch call it once per patient. Variance is the
**population** variance (divide by `n_replicas`, not `n_replicas - 1`)
over the `n_replicas` replica outputs: the ensemble members are the
entire population of interest for this estimate (there is no larger
population of replicas being sampled from), and `decisions.md`/
`design.md` do not specify a Bessel-corrected sample variance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from infra.seeding import derive_seed

ModelT = TypeVar("ModelT")


class DeepEnsemble(Generic[ModelT]):
    """A deep ensemble of `n_replicas` independently seeded model replicas.

    Args:
        model_factory: `model_factory(seed) -> model` constructs one
            replica given an integer seed. Called once per replica, at
            construction time. May itself perform training/fitting (e.g.
            `lambda seed: CoxBaseline().fit(patients)`, ignoring `seed`
            if the wrapped model has no seeded internal randomness) or
            may use the seed directly (e.g.
            `lambda seed: SequenceGRU.initialize(dim, hidden, seed=seed)`).
        predict_fn: `predict_fn(model, x) -> float` adapts one replica's
            model-specific prediction method to a scalar output for
            input `x`. Not called until `.predict()` is invoked.
        n_replicas: number of ensemble members. Must be a positive
            integer. Defaults to `5` per `design.md` Section 3.7.
        master_seed: the top-level seed from which all `n_replicas`
            replica seeds are derived via `derive_seed`. Defaults to `0`.

    Raises:
        ValueError: if `n_replicas` is not a positive integer.
    """

    def __init__(
        self,
        model_factory: Callable[[int], ModelT],
        predict_fn: Callable[[ModelT, Any], float],
        n_replicas: int = 5,
        master_seed: int = 0,
    ) -> None:
        if n_replicas <= 0:
            raise ValueError(f"n_replicas must be positive, got {n_replicas}")

        self.n_replicas = n_replicas
        self.master_seed = master_seed
        self.predict_fn = predict_fn

        self.replica_seeds: list[int] = [
            derive_seed(master_seed, f"replica_{i}") for i in range(n_replicas)
        ]
        self.replicas: list[ModelT] = [
            model_factory(seed) for seed in self.replica_seeds
        ]

    def predict(self, x: Any) -> tuple[float, float]:
        """Return `(mean, variance)` of the `n_replicas` replica outputs on `x`.

        Calls `predict_fn(replica, x)` once per replica and combines the
        resulting `n_replicas` scalar outputs into their mean and
        population variance (see module docstring for the variance
        convention).
        """
        outputs = [self.predict_fn(replica, x) for replica in self.replicas]
        mean = sum(outputs) / self.n_replicas
        variance = sum((output - mean) ** 2 for output in outputs) / self.n_replicas
        return mean, variance
