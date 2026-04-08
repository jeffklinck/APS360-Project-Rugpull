#!/usr/bin/env python3
"""Dump rows from primary.npz (see -h)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))
from data_io import load_primary_npz, processed_dir

FEATURES = ["log_return", "volume", "high_low_pct"]


def print_sample(X, y, token_ids, i: int, num_steps: int = 20):
    """Print one sample to terminal."""
    label_str = "rug" if y[i] == 1 else "non_rug"
    tid = token_ids[i] if token_ids else "N/A"
    print(f"--- Sample {i} ---")
    print(f"  token_id: {tid}")
    print(f"  label: {int(y[i])} ({label_str})")
    print(f"  shape: (120, 3)  [120 timesteps × 3 features]")
    print()
    print("  t   log_return    volume   high_low_pct")
    for t in range(min(num_steps, 120)):
        row = X[i, t]
        print(f"  {t:3d}  {row[0]:+10.6f}  {row[1]:12.2f}  {row[2]:.6f}")
    if 120 > num_steps:
        print("  ...")
        for t in range(max(num_steps, 115), 120):
            row = X[i, t]
            print(f"  {t:3d}  {row[0]:+10.6f}  {row[1]:12.2f}  {row[2]:.6f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Print sample(s) from primary.npz")
    ap.add_argument("-n", "--num", type=int, default=1, help="Number of samples to print (default 1)")
    ap.add_argument("-i", "--index", type=int, default=None, help="Print only this sample index")
    ap.add_argument("-s", "--steps", type=int, default=20, help="Show first N timesteps (default 20)")
    args = ap.parse_args()

    path = processed_dir() / "primary.npz"
    if not path.exists():
        print(f"primary.npz not found at {path}", file=sys.stderr)
        return 1

    X, y, token_ids = load_primary_npz(path=path)
    N = len(y)
    print(f"primary.npz: {N} samples, X shape {X.shape}\n")

    if args.index is not None:
        indices = [args.index]
        if args.index < 0 or args.index >= N:
            print(f"Index {args.index} out of range [0, {N})", file=sys.stderr)
            return 1
    else:
        indices = range(min(args.num, N))

    for i in indices:
        print_sample(X, y, token_ids, i, num_steps=args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
