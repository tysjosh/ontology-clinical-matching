"""Unit tests for `src/infra/seeding.py`.

**Validates: Requirements 1.6, 9.1 (decisions.md Section 9)**
"""

import pytest

from infra.seeding import derive_seed


def test_same_input_produces_same_output():
    seed_a = derive_seed(12345, "dev")
    seed_b = derive_seed(12345, "dev")

    assert seed_a == seed_b


def test_different_label_produces_different_output():
    seed_dev = derive_seed(12345, "dev")
    seed_val = derive_seed(12345, "val")
    seed_test = derive_seed(12345, "test")

    assert len({seed_dev, seed_val, seed_test}) == 3


def test_different_master_seed_produces_different_output():
    seed_a = derive_seed(1, "dev")
    seed_b = derive_seed(2, "dev")

    assert seed_a != seed_b


def test_output_is_within_32_bit_unsigned_range():
    seed = derive_seed(12345, "ensemble_replica_0")

    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


def test_negative_master_seed_raises_value_error():
    with pytest.raises(ValueError):
        derive_seed(-1, "dev")


def test_master_seed_too_large_raises_value_error():
    with pytest.raises(ValueError):
        derive_seed(2**64, "dev")


def test_derivation_is_stable_across_repeated_process_style_calls():
    # Simulates the "no random-number stream reused across splits" guarantee
    # (decisions.md Section 9, step 7): repeatedly deriving the same label
    # from the same master seed must never drift.
    seeds = {derive_seed(999, "dev") for _ in range(5)}
    assert len(seeds) == 1
