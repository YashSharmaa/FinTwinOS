"""Tests for the rolling z-score regime detector with hysteresis."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.models.time_series.regime import RegimeDetector


def shifted_series(
    n_calm: int = 250, n_stressed: int = 150, shift: float = 5.0, seed: int = 7
) -> np.ndarray:
    """N(0,1) noise that jumps to N(shift,1) at index ``n_calm``."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(0.0, 1.0, n_calm), rng.normal(shift, 1.0, n_stressed)]
    )


class TestRegimeDetector:
    def test_finds_planted_change_point(self):
        y = shifted_series()
        result = RegimeDetector(window=60, enter_z=3.0, exit_z=1.0).detect(y)
        entries = [t for t in result.change_points if result.labels[t] == 1]
        assert entries, "no stressed regime detected at all"
        assert 250 <= entries[0] <= 258

    def test_calm_period_mostly_unflagged_and_shift_held(self):
        y = shifted_series()
        result = RegimeDetector(window=60, enter_z=3.0, exit_z=1.0).detect(y)
        assert result.labels[:248].sum() <= 10  # rare noise excursions only
        assert result.labels[251:266].mean() > 0.9  # shift held while history adapts

    def test_labels_align_with_change_points(self):
        y = shifted_series(seed=3)
        result = RegimeDetector(window=50, enter_z=3.0, exit_z=1.0).detect(y)
        recomputed = [
            t for t in range(1, len(y)) if result.labels[t] != result.labels[t - 1]
        ]
        assert recomputed == result.change_points

    def test_hysteresis_keeps_regime_alive_longer(self):
        y = shifted_series()
        sticky = RegimeDetector(window=60, enter_z=3.0, exit_z=0.2).detect(y)
        loose = RegimeDetector(window=60, enter_z=3.0, exit_z=2.5).detect(y)
        assert sticky.labels.sum() > loose.labels.sum()

    def test_one_sided_mode_ignores_downward_shocks(self):
        y = shifted_series(shift=-5.0)
        result = RegimeDetector(
            window=60, enter_z=3.0, exit_z=1.0, two_sided=False
        ).detect(y)
        assert result.labels[250:260].sum() == 0

    def test_flat_series_with_jump_is_detected(self):
        y = np.concatenate([np.zeros(40), np.full(10, 7.0)])
        result = RegimeDetector(window=20, enter_z=3.0, exit_z=1.0).detect(y)
        assert result.labels[40] == 1

    def test_constant_series_stays_calm(self):
        result = RegimeDetector(window=10, enter_z=3.0, exit_z=1.0).detect(np.ones(60))
        assert result.labels.sum() == 0
        assert result.change_points == []

    def test_warmup_points_are_calm_with_zero_z(self):
        y = shifted_series()
        result = RegimeDetector(window=60, enter_z=3.0, exit_z=1.0).detect(y)
        assert np.all(result.labels[:60] == 0)
        assert np.all(result.zscores[:60] == 0.0)

    def test_segments_partition_the_series(self):
        y = shifted_series()
        result = RegimeDetector(window=60, enter_z=3.0, exit_z=1.0).detect(y)
        segments = result.segments()
        assert segments[0][0] == 0 and segments[-1][1] == len(y)
        for (_, end_a, label_a), (start_b, _, label_b) in zip(
            segments, segments[1:], strict=False
        ):
            assert end_a == start_b
            assert label_a != label_b

    def test_empty_series(self):
        result = RegimeDetector(window=10, enter_z=3.0, exit_z=1.0).detect([])
        assert result.labels.size == 0
        assert result.change_points == []
        assert result.segments() == []

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            RegimeDetector(window=10, enter_z=1.0, exit_z=1.0)
        with pytest.raises(ValueError):
            RegimeDetector(window=10, enter_z=2.0, exit_z=-0.5)
        with pytest.raises(ValueError):
            RegimeDetector(window=1)

    def test_non_finite_series_raises(self):
        with pytest.raises(ValueError):
            RegimeDetector(window=5, enter_z=2.0, exit_z=1.0).detect([1.0, np.inf, 2.0])
