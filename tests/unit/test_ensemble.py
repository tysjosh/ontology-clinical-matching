"""Unit tests for `src/uncertainty/ensemble.py` (Requirements 10.1, 10.2).

Uses a trivial synthetic "model" (a deterministic function of its
construction seed) rather than any real model class, so the ensemble's
mean/variance can be hand-verified against known replica outputs.

**Validates: Requirements 10.1, 10.2**
"""

from __future__ import annotations

from infra.seeding import derive_seed
from uncertainty.ensemble import DeepEnsemble


class _SeededConstant:
    """A trivial "model": remembers the seed it was constructed with and
    predicts a value derived deterministically from that seed and the
    query point `x`, so replica outputs are hand-verifiable."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def predict(self, x: float) -> float:
        return self.seed + x


def _predict_fn(model: _SeededConstant, x: float) -> float:
    return model.predict(x)


def test_five_distinct_seeded_replicas_are_constructed():
    """Confirms 5 distinct seeds are derived (via `derive_seed`) and used
    to construct 5 distinct replica models, each traceable to its seed."""
    ensemble = DeepEnsemble(
        model_factory=_SeededConstant, predict_fn=_predict_fn, n_replicas=5, master_seed=0
    )

    assert ensemble.n_replicas == 5
    assert len(ensemble.replicas) == 5

    expected_seeds = [derive_seed(0, f"replica_{i}") for i in range(5)]
    assert ensemble.replica_seeds == expected_seeds
    # All 5 derived seeds are distinct.
    assert len(set(expected_seeds)) == 5

    # Each replica's stored seed matches the seed it was constructed with,
    # confirming the ensemble actually used the derived seeds (not e.g.
    # five copies of the same seed).
    actual_seeds = [replica.seed for replica in ensemble.replicas]
    assert actual_seeds == expected_seeds


def test_predict_returns_hand_computed_mean_and_population_variance():
    """Confirms `.predict(x)` combines the 5 replicas' known outputs into
    the correct mean and population variance."""
    ensemble = DeepEnsemble(
        model_factory=_SeededConstant, predict_fn=_predict_fn, n_replicas=5, master_seed=42
    )

    x = 10.0
    expected_outputs = [seed + x for seed in ensemble.replica_seeds]
    expected_mean = sum(expected_outputs) / 5
    expected_variance = sum((o - expected_mean) ** 2 for o in expected_outputs) / 5

    mean, variance = ensemble.predict(x)

    assert mean == expected_mean
    assert variance == expected_variance


def test_different_master_seeds_produce_different_replica_seeds():
    """Two ensembles built from different master seeds should not share
    replica seeds (sanity check that derivation is seed-sensitive)."""
    ensemble_a = DeepEnsemble(
        model_factory=_SeededConstant, predict_fn=_predict_fn, n_replicas=5, master_seed=1
    )
    ensemble_b = DeepEnsemble(
        model_factory=_SeededConstant, predict_fn=_predict_fn, n_replicas=5, master_seed=2
    )

    assert ensemble_a.replica_seeds != ensemble_b.replica_seeds


def test_rejects_non_positive_n_replicas():
    import pytest

    with pytest.raises(ValueError):
        DeepEnsemble(
            model_factory=_SeededConstant, predict_fn=_predict_fn, n_replicas=0, master_seed=0
        )
