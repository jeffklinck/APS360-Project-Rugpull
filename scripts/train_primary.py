#!/usr/bin/env python3
"""Train binary 1D CNN on primary.npz."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))
from data_io import load_primary_npz, processed_dir, project_root
from primary_split import split_train_val_test


class RugPullCNN1D(nn.Module):
    """Conv1d stack + MLP; input (B,3,120), output logits."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc1 = nn.Linear(128 * 15, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def metrics_from_probs_binary(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    balanced_acc = (tpr + tnr) / 2.0
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_threshold_balanced_accuracy(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Threshold on probs that maximizes balanced accuracy."""
    best_t, best_ba = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        m = metrics_from_probs_binary(y_true, probs, threshold=float(t))
        ba = m["balanced_accuracy"]
        if ba > best_ba:
            best_ba = ba
            best_t = float(t)
    return best_t, best_ba


def train_one_config(
    *,
    X_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_train_t: torch.Tensor,
    y_val_t: torch.Tensor,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    device: torch.device,
    seed: int,
    pos_weight: torch.Tensor,
    verbose_epochs: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
    )

    model = RugPullCNN1D(dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_balanced = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_threshold = 0.5

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_loader.dataset)

        model.eval()
        val_logits: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                val_logits.append(model(xb).cpu().numpy())
        val_logits_arr = np.vstack(val_logits).flatten()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits_arr))
        val_t, val_ba = best_threshold_balanced_accuracy(y_val, val_probs)
        vm = metrics_from_probs_binary(y_val, val_probs, threshold=val_t)
        if val_ba >= best_val_balanced:
            best_val_balanced = val_ba
            best_val_threshold = val_t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose_epochs and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"Epoch {epoch:3d}  train_loss={total_loss:.4f}  "
                f"val_bal_acc={vm['balanced_accuracy']:.4f} (thr={val_t:.2f})  val_f1={vm['f1']:.4f}"
            )

    assert best_state is not None
    return {
        "best_val_balanced": float(best_val_balanced),
        "best_val_threshold": float(best_val_threshold),
        "best_state": best_state,
        "model": model,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train 1D CNN. Multiple --lr / --dropout / --weight-decay / --batch-size values grid-search on val."
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument(
        "--batch-size",
        nargs="+",
        type=int,
        default=[32],
        metavar="N",
        help="Batch size(s); grid if >1 value",
    )
    ap.add_argument(
        "--lr",
        nargs="+",
        type=float,
        default=[1e-3],
        metavar="LR",
        help="Learning rate(s)",
    )
    ap.add_argument(
        "--weight-decay",
        nargs="+",
        type=float,
        default=[1e-4],
        metavar="WD",
        help="Adam weight decay",
    )
    ap.add_argument(
        "--dropout",
        nargs="+",
        type=float,
        default=[0.3],
        metavar="P",
        help="Dropout before last linear",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=str, default=None, help="Path to primary.npz (default: data/processed/primary.npz)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    path = Path(args.data) if args.data else processed_dir() / "primary.npz"
    if not path.exists():
        print(f"Missing {path}", file=sys.stderr)
        return 1

    X, y, _ = load_primary_npz(path=path)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, seed=args.seed
    )

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    X_train_t = torch.from_numpy(X_train).permute(0, 2, 1)
    X_val_t = torch.from_numpy(X_val).permute(0, 2, 1)
    X_test_t = torch.from_numpy(X_test).permute(0, 2, 1)
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([max(n_neg, 1) / max(n_pos, 1)], dtype=torch.float32, device=device)

    grid = list(
        itertools.product(args.lr, args.weight_decay, args.dropout, args.batch_size)
    )
    is_grid = len(grid) > 1

    print(f"Device: {device}")
    print(f"Train {len(y_train)} | Val {len(y_val)} | Test {len(y_test)} (rug {n_pos}, non-rug {n_neg} on train)")
    print(f"Model parameters: {count_parameters(RugPullCNN1D(dropout=0.3)):,}")
    print(f"BCE pos_weight={pos_weight.item():.4f}")
    if is_grid:
        print(f"Grid: {len(grid)} configs, {args.epochs} epochs; pick best val balanced acc")
    print()

    results: list[dict[str, Any]] = []
    best_run: dict[str, Any] | None = None

    for run_idx, (lr, weight_decay, dropout, batch_size) in enumerate(grid):
        label = f"lr={lr:g} wd={weight_decay:g} dropout={dropout} bs={batch_size}"
        if is_grid:
            print(f"[{run_idx + 1}/{len(grid)}] {label}")

        run = train_one_config(
            X_train_t=X_train_t,
            X_val_t=X_val_t,
            y_train_t=y_train_t,
            y_val_t=y_val_t,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            device=device,
            seed=args.seed,
            pos_weight=pos_weight,
            verbose_epochs=not is_grid,
        )
        summary = {
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "batch_size": batch_size,
            "best_val_balanced_accuracy": run["best_val_balanced"],
            "best_val_threshold": run["best_val_threshold"],
        }
        results.append(summary)
        if is_grid:
            print(f"         → best val balanced_acc={run['best_val_balanced']:.4f} (thr={run['best_val_threshold']:.2f})\n")

        if best_run is None or run["best_val_balanced"] > best_run["best_val_balanced"]:
            best_run = run

    assert best_run is not None
    model: RugPullCNN1D = best_run["model"]
    model.load_state_dict(best_run["best_state"])
    best_val_threshold = best_run["best_val_threshold"]

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=best_run["batch_size"],
    )
    model.eval()
    test_logits: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            test_logits.append(model(xb).cpu().numpy())
    test_logits_arr = np.vstack(test_logits).flatten()
    test_probs = 1.0 / (1.0 + np.exp(-test_logits_arr))
    test_m = metrics_from_probs_binary(y_test, test_probs, threshold=best_val_threshold)

    print()
    if is_grid:
        print(f"Best config (by val balanced accuracy): lr={best_run['lr']:g} weight_decay={best_run['weight_decay']:g} "
              f"dropout={best_run['dropout']} batch_size={best_run['batch_size']}")
    print("Test (held out):")
    print(f"  threshold={best_val_threshold:.4f} (from val)")
    print(
        f"  accuracy={test_m['accuracy']:.4f}  balanced_acc={test_m['balanced_accuracy']:.4f}  "
        f"precision={test_m['precision']:.4f}  recall={test_m['recall']:.4f}  f1={test_m['f1']:.4f}"
    )
    print(f"  confusion: TN={test_m['tn']} FP={test_m['fp']} FN={test_m['fn']} TP={test_m['tp']}")

    results_dir = project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "primary_cnn_metrics.json"
    payload: dict[str, Any] = {
        "split": "70/15/15 stratified",
        "seed": args.seed,
        "epochs": args.epochs,
        "train_n": int(len(y_train)),
        "val_n": int(len(y_val)),
        "test_n": int(len(y_test)),
        "parameters": count_parameters(model),
        "checkpoint_metric": "val_balanced_accuracy_at_tuned_threshold",
        "grid_search": is_grid,
        "best_hparams": {
            "lr": best_run["lr"],
            "weight_decay": best_run["weight_decay"],
            "dropout": best_run["dropout"],
            "batch_size": best_run["batch_size"],
        },
        "best_val_threshold": float(best_val_threshold),
        "pos_weight": float(pos_weight.item()),
        "test_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in test_m.items()},
    }
    if is_grid:
        payload["grid_runs"] = results
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_json}")

    ckpt_path = results_dir / "primary_cnn.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": mean,
            "std": std,
            "decision_threshold": best_val_threshold,
            "dropout": best_run["dropout"],
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
