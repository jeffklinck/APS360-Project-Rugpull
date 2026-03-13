"""Build labeled dataset: rug (Comparitech) + non-rug (CoinGecko), pair data from DexScreener."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_scripts_dir = Path(__file__).resolve().parent
_project_root = _scripts_dir.parent
sys.path.insert(0, str(_scripts_dir))

from coingecko_client import fetch_top_by_market_cap
from dexscreener_client import get_pair, search

MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
CHECKPOINT_EVERY = 25

FEATURE_COLUMNS = [
    "price_usd",
    "liquidity_usd",
    "volume_h24",
    "tx_count_h24",
    "fdv_usd",
]
METADATA_COLUMNS = ["pair_address", "chain_id", "base_symbol", "quote_symbol", "label", "source"]


def _num(obj, default: float = 0.0) -> float:
    if obj is None:
        return default
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default


def _liquidity_usd(pair: dict) -> float:
    """Extract liquidity in USD from a pair object (search or get_pair response)."""
    liq = pair.get("liquidity")
    if liq is None:
        return 0.0
    if isinstance(liq, (int, float)):
        return _num(liq)
    return _num(liq.get("usd"))


def pick_best_pair(pairs: list[dict]) -> dict | None:
    """
    When a token has many pairs, pick one: the pair with the highest liquidity (USD).
    That gives the most representative / main market. Falls back to first if no liquidity in response.
    """
    if not pairs:
        return None
    best = max(pairs, key=_liquidity_usd)
    return best if _liquidity_usd(best) >= 0 else pairs[0]


def _retry_request(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with retries on 429, 503, timeout, connection errors."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 503, 502, 504):
                last_err = e
            else:
                raise
        except Exception as e:
            # e.g. requests.RequestException without response
            if getattr(e, "response", None) is not None and getattr(e.response, "status_code", None) in (429, 503, 502, 504):
                last_err = e
            else:
                raise
        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF ** (attempt + 1)
            print(f"  Retry in {wait:.0f}s ({attempt + 1}/{MAX_RETRIES})...", file=sys.stderr)
            time.sleep(w)
    raise last_err


def pair_to_row(pair: dict, label: int, source: str, query: str = "") -> dict:
    """Flatten pair dict to row. query used for resume."""
    liq = pair.get("liquidity") or {}
    vol = pair.get("volume") or {}
    tx = pair.get("txns") or pair.get("transactions") or {}
    h24 = tx.get("h24") if isinstance(tx, dict) else {}
    if isinstance(vol, dict):
        v24 = vol.get("h24") or 0
    else:
        v24 = vol
    buys = h24.get("buys", 0) if isinstance(h24, dict) else 0
    sells = h24.get("sells", 0) if isinstance(h24, dict) else 0
    base = pair.get("baseToken") or {}
    quote = pair.get("quoteToken") or {}
    if isinstance(liq, dict):
        liq_usd = liq.get("usd") or liq.get("usd") or 0
    else:
        liq_usd = liq
    return {
        "price_usd": _num(pair.get("priceUsd") or pair.get("price_usd")),
        "liquidity_usd": _num(liq_usd),
        "volume_h24": _num(v24),
        "tx_count_h24": _num(buys) + _num(sells),
        "fdv_usd": _num(pair.get("fdv") or pair.get("fdv_usd")),
        "pair_address": pair.get("pairAddress") or pair.get("pair_address") or "",
        "chain_id": pair.get("chainId") or pair.get("chain_id") or "",
        "base_symbol": (base.get("symbol") or "") if isinstance(base, dict) else "",
        "quote_symbol": (quote.get("symbol") or "") if isinstance(quote, dict) else "",
        "label": label,
        "source": source,
        "query": query,
    }


def get_rug_names(csv_path: Path, max_names: int | None = None) -> list[str]:
    """Get unique names from Comparitech CSV."""
    if not csv_path.exists():
        return []
    seen: set[str] = set()
    out: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for key in ("Company Name", "Type of Crypto/Token Affected"):
                val = (row.get(key) or "").strip()
                if not val:
                    continue
                for part in re.split(r"[\n,]+", val):
                    part = part.strip()
                    if len(part) >= 2 and part.lower() not in seen:
                        seen.add(part.lower())
                        out.append(part)
                        if max_names and len(out) >= max_names:
                            return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build labeled dataset (rug + non-rug) from Comparitech + CoinGecko + DexScreener")
    ap.add_argument("--rug", type=int, default=700, help="Max rug-pull pairs to collect (default 700; use 678 to match typical Comparitech+DexScreener overlap)")
    ap.add_argument("--nonrug", type=int, default=700, help="Max non-rug pairs from CoinGecko top (default 700)")
    ap.add_argument("--delay", type=float, default=0.25, help="Seconds between DexScreener calls")
    ap.add_argument("--out-dir", type=str, default=None, help="Output dir (default data/processed)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing labeled_pairs.csv (skip already-collected names/pairs)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or _project_root / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "labeled_pairs.parquet"
    csv_path_out = out_dir / "labeled_pairs.csv"
    npz_path = out_dir / "baseline.npz"

    rows: list[dict] = []
    rug_queries_done: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()
    rug_collected = 0
    nonrug_collected = 0
    if args.resume and csv_path_out.exists():
        try:
            df_existing = pd.read_csv(csv_path_out)
            rows = df_existing.to_dict("records")
            if "query" in df_existing.columns:
                rug_queries_done = set(df_existing.loc[df_existing["label"] == 1, "query"].dropna().astype(str).str.strip())
            else:
                rug_queries_done = set()
            seen_pair = set(zip(df_existing["chain_id"].astype(str), df_existing["pair_address"].astype(str)))
            rug_collected = int((df_existing["label"] == 1).sum())
            nonrug_collected = int((df_existing["label"] == 0).sum())
            print(f"Resume: loaded {len(rows)} rows ({rug_collected} rug, {nonrug_collected} non-rug). Skipping {len(rug_queries_done)} rug names, {len(seen_pair)} pairs.")
        except Exception as e:
            print(f"Resume failed ({e}), starting fresh.", file=sys.stderr)
            rows = []

    def _checkpoint():
        if not rows:
            return
        df_check = pd.DataFrame(rows)
        df_check.to_csv(csv_path_out, index=False)
        if (len(rows)) % 100 == 0:
            print(f"  Checkpoint: {len(rows)} rows written to {csv_path_out}")

    csv_path = _project_root / "data" / "raw" / "comparitech_rugpulls.csv"
    rug_names = get_rug_names(csv_path, max_names=None)
    print(f"Rug pulls: checking up to {len(rug_names)} names (target {args.rug}, have {rug_collected})...")
    for name in rug_names:
        if rug_collected >= args.rug:
            break
        if name.strip() in rug_queries_done:
            continue
        try:
            pairs = _retry_request(search, name)
            if not pairs:
                time.sleep(args.delay)
                continue
            p = pick_best_pair(pairs)
            if not p:
                time.sleep(args.delay)
                continue
            chain = p.get("chainId") or p.get("chain_id") or ""
            addr = p.get("pairAddress") or p.get("pair_address") or ""
            if not addr or (chain, addr) in seen_pair:
                time.sleep(args.delay)
                continue
            detail = _retry_request(get_pair, chain, addr)
            time.sleep(args.delay)
            if detail:
                row = pair_to_row(detail, label=1, source="comparitech", query=name.strip())
                rows.append(row)
                seen_pair.add((chain, addr))
                rug_queries_done.add(name.strip())
                rug_collected += 1
                if rug_collected % CHECKPOINT_EVERY == 0:
                    _checkpoint()
                if rug_collected % 50 == 0:
                    print(f"  Rug: {rug_collected}/{args.rug}")
        except Exception as e:
            print(f"  Error {name!r}: {e}", file=sys.stderr)
        time.sleep(args.delay)

    print(f"Non-rug: fetching top {args.nonrug} from CoinGecko (have {nonrug_collected})...")
    coins = fetch_top_by_market_cap(args.nonrug)
    for c in coins:
        if nonrug_collected >= args.nonrug:
            break
        name = c.get("name") or c.get("symbol") or ""
        symbol = (c.get("symbol") or "").upper()
        if not name:
            continue
        try:
            pairs = _retry_request(search, name)
            if not pairs:
                time.sleep(args.delay)
                continue
            p = pick_best_pair(pairs)
            if not p:
                time.sleep(args.delay)
                continue
            chain = p.get("chainId") or p.get("chain_id") or ""
            addr = p.get("pairAddress") or p.get("pair_address") or ""
            key = (chain, addr)
            if key in seen_pair:
                time.sleep(args.delay)
                continue
            detail = _retry_request(get_pair, chain, addr)
            time.sleep(args.delay)
            if detail:
                base = (detail.get("baseToken") or {}).get("symbol") or ""
                quote = (detail.get("quoteToken") or {}).get("symbol") or ""
                if base and quote:
                    seen_pair.add(key)
                    row = pair_to_row(detail, label=0, source="coingecko", query=name.strip())
                    rows.append(row)
                    nonrug_collected += 1
                    if nonrug_collected % CHECKPOINT_EVERY == 0:
                        _checkpoint()
                    if nonrug_collected % 50 == 0:
                        print(f"  Non-rug: {nonrug_collected}/{args.nonrug}")
        except Exception as e:
            print(f"  Error {name!r}: {e}", file=sys.stderr)
        time.sleep(args.delay)

    if not rows:
        print("No rows collected.", file=sys.stderr)
        return 1

    _checkpoint()
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved {len(df)} rows to {parquet_path}")
    except ImportError:
        print(f"Saved {len(df)} rows to {csv_path_out} (install pyarrow for Parquet)")

    X = df[FEATURE_COLUMNS].astype(np.float32).values
    y = df["label"].astype(np.int64).values
    np.savez_compressed(npz_path, X=X, y=y, feature_names=np.array(FEATURE_COLUMNS, dtype=object))
    print(f"Saved baseline.npz: X {X.shape}, y {y.shape} to {npz_path}")

    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    print(f"Class balance: rug={n1} non-rug={n0}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
