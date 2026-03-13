"""
CoinGecko API client for fetching top coins by market cap (non–rug-pull set).
Free API: https://api.coingecko.com/api/v3 (no key). ~30 calls/min on free tier.

Use /coins/markets to get biggest/oldest (by market_cap) tokens for negative labels.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
# Free tier ~30/min; space out calls
RATE_LIMIT_DELAY = 2.5


def coins_markets(
    vs_currency: str = "usd",
    order: str = "market_cap_desc",
    per_page: int = 250,
    page: int = 1,
) -> list[dict]:
    """
    GET /coins/markets — list coins sorted by market cap (desc = biggest first).
    Returns list of dicts: id, symbol, name, market_cap, market_cap_rank, current_price, etc.
    """
    r = requests.get(
        f"{COINGECKO_BASE}/coins/markets",
        params={
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_top_by_market_cap(n: int = 500) -> list[dict]:
    """
    Fetch top N coins by market cap (multiple pages if needed; 250 per page).
    Returns list of coin dicts (id, symbol, name, market_cap, ...).
    """
    out: list[dict] = []
    page = 1
    while len(out) < n:
        batch = coins_markets(per_page=min(250, n - len(out)), page=page)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 250:
            break
        page += 1
        time.sleep(RATE_LIMIT_DELAY)
    return out[:n]


if __name__ == "__main__":
    coins = fetch_top_by_market_cap(30)
    print(f"Fetched {len(coins)} coins. Top 5:")
    for c in coins[:5]:
        print(f"  {c.get('market_cap_rank')} {c.get('symbol')} {c.get('name')} cap={c.get('market_cap')}")
