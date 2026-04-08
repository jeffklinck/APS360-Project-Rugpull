"""Tests for build_timeseries_dataset candle-to-features logic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_timeseries_dataset import candles_to_features, T_BUCKETS


def _candle(unix_time: int, o: float, h: float, l: float, c: float, v: float) -> dict:
    return {"unixTime": unix_time, "o": o, "h": h, "l": l, "c": c, "v": v}


def test_candles_to_features_empty_returns_none():
    arr, n = candles_to_features([])
    assert arr is None
    assert n == 0


def test_candles_to_features_single_candle_returns_none():
    """Need at least 2 rows to compute log returns."""
    items = [_candle(1000, 1.0, 1.0, 1.0, 1.0, 0.0)]
    arr, n = candles_to_features(items, t_buckets=10)
    assert arr is None
    assert n == 0


def test_candles_to_features_two_candles_pads_to_t_buckets():
    base = 1000
    items = [
        _candle(base, 1.0, 1.0, 1.0, 1.0, 10.0),
        _candle(base + 60, 1.0, 1.1, 0.9, 1.05, 20.0),
    ]
    t_buckets = 5
    arr, n_actual = candles_to_features(items, t_buckets=t_buckets)
    assert arr is not None
    assert arr.shape == (t_buckets, 3)
    assert n_actual == 2
    # First row: log_return=0, volume=10, high_low_pct=0
    assert arr[0, 0] == 0.0
    assert arr[0, 1] == 10.0
    assert arr[0, 2] == 0.0
    # Second row: log_return=log(1.05/1.0), volume=20, (1.1-0.9)/1.05
    np.testing.assert_approx_equal(arr[1, 0], np.log(1.05))
    assert arr[1, 1] == 20.0
    np.testing.assert_approx_equal(arr[1, 2], (1.1 - 0.9) / 1.05)
    # Rest padded with zeros
    assert (arr[2:, :] == 0).all()


def test_candles_to_features_takes_first_t_buckets():
    base = 1000
    # 150 minutes of data; we want first 120
    items = [
        _candle(base + i * 60, 1.0 + i * 0.001, 1.0, 1.0, 1.0 + i * 0.001, 1.0)
        for i in range(150)
    ]
    arr, n_actual = candles_to_features(items, t_buckets=120)
    assert arr is not None
    assert arr.shape == (120, 3)
    assert n_actual == 120  # we cap at t_buckets in the loop


def test_candles_to_features_missing_minute_forward_fill():
    base = 1000
    # Minute 0 and minute 2 only; minute 1 missing
    items = [
        _candle(base, 1.0, 1.0, 1.0, 1.0, 10.0),
        _candle(base + 120, 1.0, 1.0, 1.0, 1.0, 5.0),  # same close -> log_return 0
    ]
    arr, n_actual = candles_to_features(items, t_buckets=5)
    assert arr is not None
    assert n_actual == 3  # we built 3 rows (0, 1 filled, 2)
    assert arr[0, 1] == 10.0
    assert arr[1, 1] == 0.0  # filled minute
    assert arr[2, 1] == 5.0
