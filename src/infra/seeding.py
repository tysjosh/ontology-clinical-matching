"""Deterministic seed derivation.

`derive_seed` is the single implementation of the HMAC-SHA-256 seed-derivation
rule frozen in `decisions.md` Section 9, step 7:

    seed_dev = HMAC(master_seed, "dev")
    seed_val = HMAC(master_seed, "val")
    seed_test = HMAC(master_seed, "test")

so that no random-number stream is ever reused across splits, ensemble
replicas, or bootstrap replicates (`design.md` Section 3.1 — "One function,
many callers, so the derivation logic is never duplicated.").

Every caller that needs a seed derived from one master seed and a label
(split name, ensemble-replica index, bootstrap-replicate index, negative-control
seed, etc.) must go through this function rather than hashing or deriving seeds
ad hoc, so that the derivation is auditable and reproducible from the manifest
(Requirement 1.6).
"""

from __future__ import annotations

import hashlib
import hmac

# Seeds are constrained to the range accepted by both `random.seed`,
# `numpy.random.default_rng`, and `torch.manual_seed` (32-bit unsigned range),
# so a derived seed can be handed directly to any of the RNGs used in this
# codebase without a caller-side range check.
_SEED_MODULUS = 2**32


def derive_seed(master_seed: int, label: str) -> int:
    """Deterministically derive a sub-seed from a master seed and a label.

    Uses HMAC-SHA-256 with the master seed (as an 8-byte big-endian key) and
    the UTF-8-encoded label as the message. The resulting digest is reduced
    to a non-negative integer in `[0, 2**32)` by taking the first 8 bytes of
    the digest as a big-endian integer and reducing modulo `2**32`.

    Deterministic: the same `(master_seed, label)` pair always produces the
    same output, on any machine, in any process (Requirement 1.6). Different
    labels (or different master seeds) produce different, independent-looking
    outputs, so distinct callers (e.g. `"dev"`, `"val"`, `"test"`, or
    `"ensemble_replica_0"`, `"ensemble_replica_1"`) never share a random-number
    stream (`decisions.md` Section 9, step 7).

    Args:
        master_seed: the top-level experiment seed. Must be a non-negative
            integer that fits in 8 bytes (i.e. `0 <= master_seed < 2**64`).
        label: a stable, human-readable identifier for the sub-stream being
            derived (e.g. a split name, replica index, or bootstrap-replicate
            index rendered as a string).

    Returns:
        A deterministic integer seed in `[0, 2**32)`.

    Raises:
        ValueError: if `master_seed` is negative or does not fit in 8 bytes.
    """
    if master_seed < 0:
        raise ValueError(f"master_seed must be non-negative, got {master_seed}")
    try:
        key = master_seed.to_bytes(8, byteorder="big", signed=False)
    except OverflowError as exc:
        raise ValueError(
            f"master_seed must fit in 8 bytes (0 <= master_seed < 2**64), got {master_seed}"
        ) from exc

    message = label.encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % _SEED_MODULUS
