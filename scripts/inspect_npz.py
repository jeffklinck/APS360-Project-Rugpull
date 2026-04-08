"""Print summary of baseline.npz / primary.npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))
from data_io import processed_dir, load_baseline_npz, load_primary_npz

# Expected schema (must match build_labeled_dataset / build_timeseries_dataset)
BASELINE_FEATURE_NAMES = ["price_usd", "liquidity_usd", "volume_h24", "tx_count_h24", "fdv_usd"]
PRIMARY_FEATURE_NAMES = ["log_return", "volume", "high_low_pct"]


def inspect_baseline(export: bool = False) -> bool:
    """Load baseline.npz, verify shape/keys, print summary. Optionally export sample CSV."""
    path = processed_dir() / "baseline.npz"
    if not path.exists():
        print(f"  baseline.npz not found at {path}")
        return False
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    expected = {"X", "y", "feature_names"}
    if not expected.issubset(set(keys)):
        print(f"  baseline.npz: missing keys. Expected {expected}, got {keys}")
        return False
    X = data["X"]
    y = data["y"]
    names = data["feature_names"].tolist() if "feature_names" in data else None
    N, F = X.shape
    print("baseline.npz")
    print(f"  X shape: {X.shape}  (N={N}, F={F})")
    print(f"  y shape: {y.shape}  dtype={y.dtype}")
    print(f"  feature_names: {names}")
    if names != BASELINE_FEATURE_NAMES:
        print(f"  WARNING: feature_names differ from expected {BASELINE_FEATURE_NAMES}")
    rug = int((y == 1).sum())
    nonrug = int((y == 0).sum())
    print(f"  class balance: rug={rug}  non_rug={nonrug}")
    print(f"  X dtype: {X.dtype}  min={X.min():.4f}  max={X.max():.4f}  mean={np.nanmean(X):.4f}")
    if export:
        out = processed_dir() / "inspect_baseline_sample.csv"
        header = (names or [f"f{i}" for i in range(F)]) + ["label"]
        sample = np.hstack([X[:20], y[:20].reshape(-1, 1)])
        np.savetxt(out, sample, delimiter=",", header=",".join(header), comments="")
        print(f"  Exported first 20 rows to {out}")
    return True


def inspect_primary(export: bool = False) -> bool:
    """Load primary.npz, verify shape/keys, print summary. Optionally export sample CSV."""
    path = processed_dir() / "primary.npz"
    if not path.exists():
        print(f"  primary.npz not found at {path}")
        return False
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    expected = {"X", "y", "token_ids"}
    if not expected.issubset(set(keys)):
        print(f"  primary.npz: missing keys. Expected {expected}, got {keys}")
        return False
    X = data["X"]
    y = data["y"]
    token_ids = data["token_ids"].tolist() if "token_ids" in data else None
    N, T, F = X.shape
    print("primary.npz")
    print(f"  X shape: {X.shape}  (N={N}, T={T}, F={F})")
    print(f"  y shape: {y.shape}  dtype={y.dtype}")
    print(f"  feature dims (per timestep): {PRIMARY_FEATURE_NAMES}")
    rug = int((y == 1).sum())
    nonrug = int((y == 0).sum())
    print(f"  class balance: rug={rug}  non_rug={nonrug}")
    print(f"  X dtype: {X.dtype}  min={X.min():.4f}  max={X.max():.4f}")
    if token_ids:
        print(f"  token_ids: {len(token_ids)} (e.g. {token_ids[0]})")
    if export:
        # Export one sample as CSV: rows = timesteps, cols = features + label
        out = processed_dir() / "inspect_primary_sample.csv"
        header = "t," + ",".join(PRIMARY_FEATURE_NAMES)
        sample = X[0]  # (T, F)
        t = np.arange(sample.shape[0])[:, None]
        block = np.hstack([t, sample])
        np.savetxt(out, block, delimiter=",", header=header, comments="")
        print(f"  Exported first sample (token 0) time series to {out}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify and view baseline.npz / primary.npz")
    ap.add_argument("--baseline", action="store_true", help="Only inspect baseline.npz")
    ap.add_argument("--primary", action="store_true", help="Only inspect primary.npz")
    ap.add_argument("--export", action="store_true", help="Write sample CSVs to data/processed/")
    args = ap.parse_args()
    do_baseline = args.baseline or not (args.baseline or args.primary)
    do_primary = args.primary or not (args.baseline or args.primary)
    ok = True
    if do_baseline:
        ok = inspect_baseline(export=args.export) and ok
    if do_primary:
        ok = inspect_primary(export=args.export) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
