"""Unit tests for `src/causal/weighting.py`.

Hand-checked weight lists confirm each of the three frozen presets
(`"primary"` 1st/99th, `"sensitivity_narrow"` 0.5th/99.5th,
`"sensitivity_wide"` 5th/95th percentile) clips extreme values correctly,
for both `dict` and plain-list inputs.

**Validates: Requirement 7.10**
"""

import pytest

from causal.weighting import CLIP_PRESETS, clip_weights


def test_all_three_presets_are_registered():
    assert set(CLIP_PRESETS) == {"primary", "sensitivity_narrow", "sensitivity_wide"}
    assert CLIP_PRESETS["primary"] == (1.0, 99.0)
    assert CLIP_PRESETS["sensitivity_narrow"] == (0.5, 99.5)
    assert CLIP_PRESETS["sensitivity_wide"] == (5.0, 95.0)


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        clip_weights([1.0, 2.0, 3.0], preset="not_a_preset")


def test_clip_weights_on_list_primary_preset_clips_extremes():
    # 100 evenly spaced weights, 1..100 -- an obvious extreme low (1) and
    # high (100) value, well outside the 1st/99th percentile band.
    weights = [float(i) for i in range(1, 101)]

    clipped = clip_weights(weights, preset="primary")

    assert isinstance(clipped, list)
    assert len(clipped) == len(weights)
    # The extreme low/high values must have moved (been clipped) upward/downward.
    assert clipped[0] > weights[0]
    assert clipped[-1] < weights[-1]
    # Middle-of-the-pack values are unaffected.
    assert clipped[49] == weights[49]


def test_clip_weights_sensitivity_narrow_clips_less_than_primary():
    weights = [float(i) for i in range(1, 101)]

    primary = clip_weights(weights, preset="primary")
    narrow = clip_weights(weights, preset="sensitivity_narrow")

    # Narrower percentile band (0.5/99.5) clips less aggressively than the
    # wider 1/99 band, so the narrow low bound is <= the primary low bound.
    assert narrow[0] <= primary[0]
    assert narrow[-1] >= primary[-1]


def test_clip_weights_sensitivity_wide_clips_more_than_primary():
    weights = [float(i) for i in range(1, 101)]

    primary = clip_weights(weights, preset="primary")
    wide = clip_weights(weights, preset="sensitivity_wide")

    # Wider clipping band (5/95) clips more aggressively than 1/99.
    assert wide[0] >= primary[0]
    assert wide[-1] <= primary[-1]


def test_clip_weights_on_dict_preserves_keys_and_clips_values():
    weights = {f"P{i:03d}": float(i) for i in range(1, 101)}

    clipped = clip_weights(weights, preset="primary")

    assert isinstance(clipped, dict)
    assert set(clipped.keys()) == set(weights.keys())
    assert clipped["P001"] > weights["P001"]
    assert clipped["P100"] < weights["P100"]
    assert clipped["P050"] == weights["P050"]


def test_clip_weights_hand_checked_small_list():
    # 10 values, 1..10. Sorted-rank linear interpolation for the primary
    # (1st/99th) band on n=10: rank = (p/100)*(n-1).
    # 1st percentile: rank = 0.01*9 = 0.09 -> interpolate between index 0
    # (value 1) and index 1 (value 2): 1 + 0.09*(2-1) = 1.09.
    # 99th percentile: rank = 0.99*9 = 8.91 -> interpolate between index 8
    # (value 9) and index 9 (value 10): 9 + 0.91*(10-9) = 9.91.
    weights = [float(i) for i in range(1, 11)]

    clipped = clip_weights(weights, preset="primary")

    assert clipped[0] == pytest.approx(1.09)
    assert clipped[-1] == pytest.approx(9.91)
    assert clipped[1:-1] == weights[1:-1]


def test_clip_weights_empty_input_returns_empty_same_type():
    assert clip_weights([], preset="primary") == []
    assert clip_weights({}, preset="primary") == {}


def test_clip_weights_does_not_mutate_input():
    weights = [float(i) for i in range(1, 101)]
    original = list(weights)

    clip_weights(weights, preset="primary")

    assert weights == original
