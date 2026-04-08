#!/usr/bin/env python3
"""Baseline (logistic on aggregates) vs saved CNN on the same 70/15/15 split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))

from data_io import load_primary_npz, processed_dir, project_root
from primary_split import split_train_val_test
from run_baseline_on_primary import timeseries_to_aggregate
from train_primary import (
    RugPullCNN1D,
    best_threshold_balanced_accuracy,
    metrics_from_probs_binary,
)


def load_cnn_eval_test(
    ckpt_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict | None:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    dropout = float(ckpt.get("dropout", 0.3))
    model = RugPullCNN1D(dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    mean = ckpt["mean"]
    std = ckpt["std"]
    thr = float(ckpt["decision_threshold"])

    Xn = (X_test.astype(np.float32) - mean) / std
    x_t = torch.from_numpy(Xn).permute(0, 2, 1).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x_t).cpu().numpy().flatten()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return metrics_from_probs_binary(y_test.astype(np.int64), probs, threshold=thr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline vs CNN on same primary.npz split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=str, default=None, help="Path to primary.npz")
    ap.add_argument(
        "--cnn-checkpoint",
        type=str,
        default=None,
        help="Path to primary_cnn.pt (default: results/primary_cnn.pt if present)",
    )
    ap.add_argument("--no-cnn", action="store_true", help="Only run baseline")
    args = ap.parse_args()

    path = Path(args.data) if args.data else processed_dir() / "primary.npz"
    if not path.exists():
        print(f"Missing {path}", file=sys.stderr)
        return 1

    X, y, _ = load_primary_npz(path=path)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, seed=args.seed)

    X_train_a = timeseries_to_aggregate(X_train)
    X_val_a = timeseries_to_aggregate(X_val)
    X_test_a = timeseries_to_aggregate(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_a)
    X_val_s = scaler.transform(X_val_a)
    X_test_s = scaler.transform(X_test_a)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    clf.fit(X_train_s, y_train.astype(np.int32))

    val_probs = clf.predict_proba(X_val_s)[:, 1]
    baseline_thr, baseline_val_ba = best_threshold_balanced_accuracy(
        y_val.astype(np.int64), val_probs
    )
    test_probs = clf.predict_proba(X_test_s)[:, 1]
    baseline_test = metrics_from_probs_binary(
        y_test.astype(np.int64), test_probs, threshold=baseline_thr
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_default = project_root() / "results" / "primary_cnn.pt"
    ckpt_path = Path(args.cnn_checkpoint) if args.cnn_checkpoint else ckpt_default

    cnn_test = None
    if not args.no_cnn and ckpt_path.is_file():
        cnn_test = load_cnn_eval_test(ckpt_path, X_test, y_test, device)
    elif not args.no_cnn:
        print(f"No CNN checkpoint at {ckpt_path}; baseline only.\n")

    print("Same split as train_primary.py: 70/15/15 stratified, seed=%d" % args.seed)
    print(f"Train {len(y_train)} | Val {len(y_val)} | Test {len(y_test)}")
    print()
    print("--- Logistic regression (aggregated features) ---")
    print(f"  Val threshold (balanced-acc search): {baseline_thr:.4f}  (val balanced acc: {baseline_val_ba:.4f})")
    print(
        f"  Test  accuracy={baseline_test['accuracy']:.4f}  balanced_acc={baseline_test['balanced_accuracy']:.4f}  "
        f"precision={baseline_test['precision']:.4f}  recall={baseline_test['recall']:.4f}  f1={baseline_test['f1']:.4f}"
    )
    print(
        f"  Test  confusion: TN={baseline_test['tn']} FP={baseline_test['fp']} "
        f"FN={baseline_test['fn']} TP={baseline_test['tp']}"
    )

    if cnn_test is not None:
        print()
        print(f"--- 1D CNN ({ckpt_path}) ---")
        print("  (CNN threshold from checkpoint)")
        print(
            f"  Test  accuracy={cnn_test['accuracy']:.4f}  balanced_acc={cnn_test['balanced_accuracy']:.4f}  "
            f"precision={cnn_test['precision']:.4f}  recall={cnn_test['recall']:.4f}  f1={cnn_test['f1']:.4f}"
        )
        print(
            f"  Test  confusion: TN={cnn_test['tn']} FP={cnn_test['fp']} "
            f"FN={cnn_test['fn']} TP={cnn_test['tp']}"
        )

    out_dir = project_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": "70/15/15 stratified",
        "seed": args.seed,
        "train_n": int(len(y_train)),
        "val_n": int(len(y_val)),
        "test_n": int(len(y_test)),
        "baseline": {
            "val_threshold_balanced_acc_search": float(baseline_thr),
            "test_metrics": {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in baseline_test.items()
            },
        },
    }
    if cnn_test is not None:
        payload["cnn"] = {
            "checkpoint": str(ckpt_path.resolve()),
            "test_metrics": {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in cnn_test.items()
            },
        }
    out_json = out_dir / "primary_baseline_cnn_comparison.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Saved {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
