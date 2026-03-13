"""Load/save baseline and primary npz and labeled table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def processed_dir(subdir: str | None = None) -> Path:
    d = project_root() / "data" / "processed"
    if subdir:
        d = d / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_primary_npz(
    X: np.ndarray,
    y: np.ndarray,
    token_ids: list[str] | None = None,
    out_path: Path | None = None,
) -> Path:
    out_path = out_path or processed_dir() / "primary.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {"X": X.astype(np.float32), "y": y.astype(np.int64)}
    if token_ids is not None:
        save["token_ids"] = np.array(token_ids, dtype=object)
    np.savez_compressed(out_path, **save)
    return out_path


def load_primary_npz(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    path = path or processed_dir() / "primary.npz"
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    token_ids = data["token_ids"].tolist() if "token_ids" in data else None
    return X, y, token_ids


def save_baseline_npz(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    out_path: Path | None = None,
) -> Path:
    out_path = out_path or processed_dir() / "baseline.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {"X": X.astype(np.float32), "y": y.astype(np.int64)}
    if feature_names is not None:
        save["feature_names"] = np.array(feature_names, dtype=object)
    np.savez_compressed(out_path, **save)
    return out_path


def load_baseline_npz(
    path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    path = path or processed_dir() / "baseline.npz"
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    names = data["feature_names"].tolist() if "feature_names" in data else None
    return X, y, names


def load_labeled_parquet(path: Path | None = None) -> pd.DataFrame:
    base = processed_dir()
    p_parquet = path or base / "labeled_pairs.parquet"
    p_csv = base / "labeled_pairs.csv"
    if path:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    return pd.read_parquet(p_parquet)  # raise
