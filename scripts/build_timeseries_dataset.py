"""Build primary.npz time-series from Birdeye: first 2h after first trade per pair."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_scripts_dir = Path(__file__).resolve().parent
_project_root = _scripts_dir.parent
sys.path.insert(0, str(_scripts_dir))

_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

from data_io import load_labeled_parquet, load_primary_npz, processed_dir, save_primary_npz

BIRDEYE_BASE = "https://public-api.birdeye.so"
T_BUCKETS = 120  # 2 hours × 1 min
FEATURE_NAMES = ["log_return", "volume", "high_low_pct"]


BIRDEYE_MAX_RETRIES = 4
BIRDEYE_RETRY_BACKOFF = 2.0


def _birdeye_chain(chain_id: str) -> str:
    c = (chain_id or "").lower().strip()
    if c in ("solana", "ethereum", "arbitrum", "avalanche", "bsc", "optimism", "polygon", "base"):
        return c
    if c in ("eth", "matic"):
        return "ethereum" if c == "eth" else "polygon"
    return c or "solana"


def _birdeye_request(
    method: str,
    url: str,
    api_key: str,
    headers_extra: dict | None = None,
    **kwargs,
) -> requests.Response:
    headers = {"X-API-KEY": api_key, **(headers_extra or {})}
    last_err = None
    for attempt in range(BIRDEYE_MAX_RETRIES):
        try:
            r = requests.request(method, url, headers=headers, timeout=30, **kwargs)
            if r.status_code in (429, 503, 502, 504):
                last_err = requests.HTTPError(f"Birdeye {r.status_code}: {r.text[:200]}")
                time.sleep(BIRDEYE_RETRY_BACKOFF ** (attempt + 1))
                continue
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(BIRDEYE_RETRY_BACKOFF ** (attempt + 1))
    raise last_err


def fetch_first_trade_timestamp(pair_address: str, chain_id: str, api_key: str) -> int | None:
    url = f"{BIRDEYE_BASE}/defi/txs/pair"
    headers = {"X-API-KEY": api_key, "x-chain": _birdeye_chain(chain_id)}
    params = {"address": pair_address, "sort_type": "asc", "limit": 1, "offset": 0}
    r = _birdeye_request("get", url, api_key, headers_extra={"x-chain": _birdeye_chain(chain_id)}, params=params)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data.get("success"):
        return None
    items = (data.get("data") or {}).get("items") or []
    if not items:
        return None
    ts = items[0].get("blockUnixTime")
    if ts is None:
        return None
    try:
        return int(ts)
    except (TypeError, ValueError):
        return None


def fetch_ohlcv(pair_address: str, chain_id: str, time_from: int, time_to: int, api_key: str) -> list[dict]:
    url = f"{BIRDEYE_BASE}/defi/ohlcv/pair"
    params = {"address": pair_address, "type": "1m", "time_from": time_from, "time_to": time_to}
    r = _birdeye_request("get", url, api_key, headers_extra={"x-chain": _birdeye_chain(chain_id)}, params=params)
    if r.status_code != 200:
        raise requests.HTTPError(f"Birdeye {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not data.get("success") or "data" not in data:
        return []
    return data.get("data", {}).get("items") or []


def candles_to_features(items: list[dict], t_buckets: int = T_BUCKETS) -> tuple[np.ndarray | None, int]:
    if not items:
        return None, 0
    sorted_items = sorted(items, key=lambda x: x["unixTime"])
    t0 = sorted_items[0]["unixTime"]
    minute_data = {}
    for it in sorted_items:
        ts = it["unixTime"]
        idx = (ts - t0) // 60
        c = float(it.get("c") or 0)
        v = float(it.get("v") or 0)
        h, l_ = float(it.get("h") or c), float(it.get("l") or c)
        if c <= 0:
            c = 1e-9
        minute_data[idx] = {"c": c, "v": v, "hl": (h - l_) / c}

    if not minute_data:
        return None, 0
    max_idx = max(minute_data.keys())
    prev_c = None
    rows = []
    for i in range(min(max_idx + 1, t_buckets)):
        if i in minute_data:
            prev_c = minute_data[i]["c"]
            rows.append(minute_data[i])
        elif prev_c is not None:
            rows.append({"c": prev_c, "v": 0.0, "hl": 0.0})
        else:
            rows.append({"c": 1.0, "v": 0.0, "hl": 0.0})
    if len(rows) < 2:
        return None, 0
    arr = np.zeros((len(rows), 3), dtype=np.float32)
    arr[0, 0] = 0.0
    arr[0, 1] = rows[0]["v"]
    arr[0, 2] = rows[0]["hl"]
    for i in range(1, len(rows)):
        c_prev, c_curr = rows[i - 1]["c"], rows[i]["c"]
        if c_prev and c_prev > 0:
            arr[i, 0] = np.log(max(c_curr / c_prev, 1e-9))
        arr[i, 1] = rows[i]["v"]
        arr[i, 2] = rows[i]["hl"]
    n_actual = len(rows)
    if len(arr) >= t_buckets:
        arr = arr[:t_buckets]
    else:
        pad = np.zeros((t_buckets - len(arr), 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr, n_actual


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build (N,T,F) time-series from first 2h after pair launch (Birdeye first-trade + OHLCV)"
    )
    ap.add_argument("--hours", type=float, default=2.0, help="Window length in hours from launch (default 2)")
    ap.add_argument("--min-buckets", type=int, default=60, help="Skip pair if fewer candles (default 60)")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between Birdeye calls (default 1; Birdeye ~60 rpm)")
    ap.add_argument("--resume", action="store_true", help="Skip pairs already in primary.npz; append new ones and save at end (and every --checkpoint-every pairs)")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="Save primary.npz every N pairs when using --resume (0=only at end)")
    args = ap.parse_args()

    api_key = os.environ.get("BIRDEYE_API_KEY", "").strip()
    if not api_key:
        print("Set BIRDEYE_API_KEY in the environment. Get a key at https://bds.birdeye.so", file=sys.stderr)
        return 1

    try:
        df = load_labeled_parquet()
    except Exception as e:
        print(f"Could not load labeled_pairs: {e}. Run build_labeled_dataset.py first.", file=sys.stderr)
        return 1
    if df is None or df.empty:
        print("No labeled_pairs found. Run build_labeled_dataset.py first.", file=sys.stderr)
        return 1

    n_pairs = len(df)
    est_min = (n_pairs * 3 * args.delay) / 60.0
    print(f"Processing {n_pairs} pairs (delay={args.delay}s). Estimated time: ~{est_min:.0f}–{est_min * 2:.0f} min.")

    t_buckets = max(1, int(args.hours * 60))
    window_seconds = int(args.hours * 3600)
    out_path = processed_dir() / "primary.npz"

    X_list = []
    y_list = []
    token_ids = []
    done_ids = set()
    if args.resume and out_path.exists():
        try:
            X_old, y_old, ids_old = load_primary_npz(path=out_path)
            if ids_old and len(ids_old) > 0:
                done_ids = set(ids_old)
                X_list = [X_old[i] for i in range(len(ids_old))]
                y_list = y_old.tolist()
                token_ids = list(ids_old)
                print(f"Resume: loaded {len(ids_old)} pairs from {out_path}; will skip those and add new ones.")
        except Exception as e:
            print(f"Resume: could not load {out_path}: {e}. Starting from scratch.", file=sys.stderr)

    skipped = 0
    no_launch = 0
    for i, row in df.iterrows():
        pair_address = str(row.get("pair_address", "")).strip()
        chain_id = str(row.get("chain_id", "")).strip()
        label = int(row.get("label", 0))
        tid = f"{chain_id}:{pair_address}"
        if tid in done_ids:
            continue
        if not pair_address:
            skipped += 1
            continue
        try:
            launch_ts = fetch_first_trade_timestamp(pair_address, chain_id, api_key)
            time.sleep(args.delay)
            if launch_ts is None:
                no_launch += 1
                continue
            time_from = launch_ts
            time_to = launch_ts + window_seconds
            items = fetch_ohlcv(pair_address, chain_id, time_from, time_to, api_key)
            time.sleep(args.delay)
            arr, n_actual = candles_to_features(items, t_buckets=t_buckets)
            if arr is None or n_actual < args.min_buckets:
                skipped += 1
                continue
            X_list.append(arr)
            y_list.append(label)
            token_ids.append(tid)
            done_ids.add(tid)
            if args.checkpoint_every and args.resume and len(X_list) % args.checkpoint_every == 0:
                X_cp = np.stack(X_list, axis=0).astype(np.float32)
                y_cp = np.array(y_list, dtype=np.int64)
                save_primary_npz(X_cp, y_cp, token_ids=token_ids, out_path=out_path)
                print(f"  Checkpoint: saved {len(X_list)} pairs to {out_path}")
        except Exception as e:
            print(f"  Skip {pair_address[:20]}...: {e}", file=sys.stderr)
            skipped += 1
        time.sleep(args.delay)

    if not X_list:
        print("No time-series collected.", file=sys.stderr)
        return 1

    if no_launch:
        print(f"Skipped {no_launch} pairs with no first-trade (launch) timestamp from Birdeye.", file=sys.stderr)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"Collected {len(X_list)} tokens, skipped {skipped}. X shape: {X.shape}")

    save_primary_npz(X, y, token_ids=token_ids, out_path=out_path)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
