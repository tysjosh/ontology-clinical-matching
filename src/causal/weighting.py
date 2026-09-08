"""Weight clipping presets (Requirement 7.10; `decisions.md` Section 11
"Propensity modeling").

`decisions.md` Section 11 freezes three named weight-clipping presets
(percentile bounds a weight is clamped into, not dropped at):

- `"primary"`: 1st/99th percentile;
- `"sensitivity_narrow"`: 0.5th/99.5th percentile;
- `"sensitivity_wide"`: 5th/95th percentile.

Percentile method
-------------------

Percentiles are computed by linear interpolation between the two closest
ranks in the sorted value list -- the same method NumPy's default
`numpy.percentile(..., method="linear")` uses. For a sorted sequence of `n`
values and target percentile `p` (in `[0, 100]`), the fractional rank is
`rank = (p / 100) * (n - 1)`; the result interpolates linearly between
`sorted_values[floor(rank)]` and `sorted_values[ceil(rank)]`. This is a
simple, standard, deterministic choice (as opposed to e.g. nearest-rank),
appropriate for a small, from-scratch implementation with no numpy
dependency (see `causal.propensity` module docstring for the same
dependency-scope note).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

#: `decisions.md` Section 11 -- the three frozen named clipping presets,
#: each a `(low_percentile, high_percentile)` pair.
CLIP_PRESETS: dict[str, tuple[float, float]] = {
    "primary": (1.0, 99.0),
    "sensitivity_narrow": (0.5, 99.5),
    "sensitivity_wide": (5.0, 95.0),
}


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence.

    See module docstring for the method. `sorted_values` must be
    non-empty and sorted ascending; `p` is a percentile in `[0, 100]`.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (n - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = rank - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def clip_weights(
    weights: Mapping[str, float] | Sequence[float], preset: str
) -> dict[str, float] | list[float]:
    """Clip `weights` into the percentile band named by `preset`.

    Values below the preset's low percentile are raised to it; values
    above the high percentile are lowered to it; values in between are
    unchanged. The percentile bounds are computed from `weights` itself
    (the same collection being clipped), matching standard weight-clipping
    practice for IPW-style weights.

    Args:
        weights: either a `{key: weight}` mapping (e.g.
            `{patient_id: weight}`) or a plain sequence of weights.
        preset: one of the frozen names in `CLIP_PRESETS`
            (`"primary"`, `"sensitivity_narrow"`, `"sensitivity_wide"`).

    Returns:
        The same container type as `weights` (a `dict` for a `Mapping`
        input, a `list` for a sequence input), with clipped values. An
        empty input returns an empty container of the same type without
        error (percentiles are undefined for zero values, so there is
        nothing to clip).

    Raises:
        ValueError: if `preset` is not one of `CLIP_PRESETS`.
    """
    if preset not in CLIP_PRESETS:
        raise ValueError(
            f"Unknown weight-clipping preset {preset!r}; expected one of " f"{sorted(CLIP_PRESETS)}"
        )
    low_percentile, high_percentile = CLIP_PRESETS[preset]

    is_mapping = isinstance(weights, Mapping)
    values = list(weights.values()) if is_mapping else list(weights)

    if not values:
        return {} if is_mapping else []

    sorted_values = sorted(values)
    low_bound = _percentile(sorted_values, low_percentile)
    high_bound = _percentile(sorted_values, high_percentile)

    def _clip(value: float) -> float:
        return min(max(value, low_bound), high_bound)

    if is_mapping:
        return {key: _clip(value) for key, value in weights.items()}
    return [_clip(value) for value in weights]
