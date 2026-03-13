#!/usr/bin/env python3
"""Logistic regression baseline on primary.npz time-series data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))
from data_io import load_primary_npz, processed_dir

STAT_NAMES = [
    "log_return_mean", "log_return_std", "log_return_min", "log_return_max",
    "volume_mean", "volume_std", "volume_min", "volume_max", "volume_sum",
    "high_low_pct_mean", "high_low_pct_std", "high_low_pct_min", "high_low_pct_max",
]


def timeseries_to_aggregate(X: np.ndarray) -> np.ndarray:
    N, T, F = X.shape
    assert F == 3
    out = []
    for i in range(N):
        row = []
        for j in range(3):
            v = X[i, :, j].astype(np.float64)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            row.extend([np.mean(v), np.std(v), np.min(v), np.max(v)])
        vol_sum = np.sum(X[i, :, 1])
        row.append(np.log1p(np.clip(vol_sum, 0, 1e30)))
        out.append(row)
    agg = np.array(out, dtype=np.float64)
    agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation fraction (default 0.2)")
    args = ap.parse_args()

    path = processed_dir() / "primary.npz"
    if not path.exists():
        print(f"primary.npz not found at {path}. Run build_timeseries_dataset.py first.", file=sys.stderr)
        return 1

    X_ts, y, _ = load_primary_npz(path=path)
    y = y.astype(np.int32)
    N = len(y)
    print(f"Loaded primary.npz: {N} samples, X shape {X_ts.shape}")

    X_agg = timeseries_to_aggregate(X_ts)
    print(f"Aggregated to (N, {X_agg.shape[1]}) summary features: {STAT_NAMES}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_agg, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    print(f"Train: {len(y_train)}, Val: {len(y_val)} (stratified, seed={args.seed})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_val_s)

    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    print()
    print("--- Metrics (validation) ---")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced accuracy: {bal_acc:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  F1:               {f1:.4f}")
    print()
    print("Confusion matrix (val):  pred_0  pred_1")
    print(f"  true_0 (non_rug)     {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  true_1 (rug)         {cm[1,0]:5d}   {cm[1,1]:5d}")
    print()
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
