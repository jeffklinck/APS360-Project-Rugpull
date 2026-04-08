"""Run tests without pytest (python3 tests/run_tests.py)."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

def run_data_io_tests():
    import tempfile
    import numpy as np
    from data_io import save_baseline_npz, load_baseline_npz, save_primary_npz, load_primary_npz, project_root, processed_dir

    assert project_root().is_dir()
    assert (project_root() / "scripts").is_dir()
    d = processed_dir()
    d.mkdir(parents=True, exist_ok=True)
    assert d.is_dir()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "baseline.npz"
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.array([0, 1] * 5, dtype=np.int64)
        save_baseline_npz(X, y, feature_names=["f1", "f2", "f3", "f4", "f5"], out_path=out)
        assert out.exists()
        X2, y2, names2 = load_baseline_npz(path=out)
        np.testing.assert_array_almost_equal(X, X2)
        np.testing.assert_array_equal(y, y2)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "primary.npz"
        X = np.random.randn(6, 120, 3).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        save_primary_npz(X, y, token_ids=[f"c{i}:a{i}" for i in range(6)], out_path=out)
        assert out.exists()
        X2, y2, ids2 = load_primary_npz(path=out)
        np.testing.assert_array_almost_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
    print("  data_io: OK")


def run_timeseries_tests():
    import numpy as np
    from build_timeseries_dataset import candles_to_features

    arr, n = candles_to_features([])
    assert arr is None and n == 0

    items = [{"unixTime": 1000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 10.0}]
    arr, n = candles_to_features(items, t_buckets=10)
    assert arr is None and n == 0

    items = [
        {"unixTime": 1000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 10.0},
        {"unixTime": 1060, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.05, "v": 20.0},
    ]
    arr, n_actual = candles_to_features(items, t_buckets=5)
    assert arr is not None and arr.shape == (5, 3) and n_actual == 2
    assert arr[0, 0] == 0.0 and arr[0, 1] == 10.0
    np.testing.assert_approx_equal(arr[1, 0], np.log(1.05))
    assert arr[1, 1] == 20.0
    print("  timeseries candles_to_features: OK")


def main():
    print("Running tests...")
    run_data_io_tests()
    run_timeseries_tests()
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
