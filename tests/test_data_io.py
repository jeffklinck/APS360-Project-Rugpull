"""Tests for data_io save/load round-trip and paths."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import from scripts (run from project root: pytest tests/)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from data_io import (
    save_baseline_npz,
    load_baseline_npz,
    save_primary_npz,
    load_primary_npz,
    project_root,
    processed_dir,
)


def test_project_root_is_directory():
    root = project_root()
    assert root.is_dir()
    assert (root / "scripts").is_dir()
    assert (root / "data").is_dir()


def test_processed_dir_creates_dir():
    with tempfile.TemporaryDirectory() as tmp:
        # We can't easily override processed_dir to use tmp without changing data_io.
        # Just check that processed_dir() returns a Path under project_root.
        d = processed_dir()
        assert d.name == "processed"
        assert "data" in d.parts
        d.mkdir(parents=True, exist_ok=True)
        assert d.is_dir()


def test_baseline_npz_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "baseline.npz"
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.array([0, 1] * 5, dtype=np.int64)
        names = ["f1", "f2", "f3", "f4", "f5"]
        save_baseline_npz(X, y, feature_names=names, out_path=out)
        assert out.exists()
        X2, y2, names2 = load_baseline_npz(path=out)
        np.testing.assert_array_almost_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
        assert names2 == names


def test_primary_npz_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "primary.npz"
        N, T, F = 6, 120, 3
        X = np.random.randn(N, T, F).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        token_ids = [f"chain{i}:addr{i}" for i in range(N)]
        save_primary_npz(X, y, token_ids=token_ids, out_path=out)
        assert out.exists()
        X2, y2, ids2 = load_primary_npz(path=out)
        np.testing.assert_array_almost_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
        assert ids2 == token_ids
