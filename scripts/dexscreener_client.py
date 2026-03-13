"""
DexScreener API client for rug-pull predictor data.
Base URL: https://api.dexscreener.com
No API key required. Rate limits: 300 req/min (pairs/search), 60/min (token-profiles, orders).
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

# Default base URL; can override via config
DEXSCREENER_BASE = "https://api.dexscreener.com"


def _load_base_url_from_config() -> str:
    """Load dexscreener_base_url from configs/config.yaml if present (optional PyYAML)."""
    try:
        import yaml
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return (cfg.get("data") or {}).get("dexscreener_base_url") or DEXSCREENER_BASE
    except Exception:  # no yaml or missing key
        pass
    return DEXSCREENER_BASE


def search(query: str, base_url: str | None = None) -> list[dict]:
    """
    Search for pairs/tokens by name, symbol, or address.
    GET /latest/dex/search?q=QUERY
    Returns list of pair objects (chainId, pairAddress, baseToken, quoteToken, priceUsd, etc.).
    """
    base = base_url or _load_base_url_from_config()
    r = requests.get(f"{base}/latest/dex/search", params={"q": query}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("pairs", data) if isinstance(data, dict) else (data or [])


def get_pair(chain_id: str, pair_address: str, base_url: str | None = None) -> dict | None:
    """
    Get detailed data for one pair.
    GET /latest/dex/pairs/{chainId}/{pairAddress}
    Returns single pair object or None if not found.
    """
    base = base_url or _load_base_url_from_config()
    r = requests.get(f"{base}/latest/dex/pairs/{chain_id}/{pair_address}", timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    return data.get("pair") if isinstance(data, dict) else (data[0] if data else None)


def get_token_pairs(chain_id: str, token_address: str, base_url: str | None = None) -> list[dict]:
    """
    Get all pairs for a token.
    GET /token-pairs/v1/{chainId}/{tokenAddress}
    Returns list of pair objects.
    """
    base = base_url or _load_base_url_from_config()
    r = requests.get(f"{base}/token-pairs/v1/{chain_id}/{token_address}", timeout=30)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def get_tokens(chain_id: str, token_addresses: list[str], base_url: str | None = None) -> list[dict]:
    """
    Get token data for multiple addresses (comma-separated in API).
    GET /tokens/v1/{chainId}/{tokenAddresses}
    """
    base = base_url or _load_base_url_from_config()
    addrs = ",".join(token_addresses)
    r = requests.get(f"{base}/tokens/v1/{chain_id}/{addrs}", timeout=30)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def rate_limit_delay(requests_per_minute: int = 300) -> None:
    """Sleep to avoid hitting rate limit (e.g. 300/min -> 0.2s between calls)."""
    time.sleep(60.0 / requests_per_minute)


if __name__ == "__main__":
    # Quick test: search and fetch one pair
    print("DexScreener API test")
    pairs = search("WETH")
    print(f"Search 'WETH': {len(pairs)} pair(s)")
    if pairs:
        p = pairs[0]
        chain = p.get("chainId", "ethereum")
        addr = p.get("pairAddress", "")
        print(f"  First: {p.get('baseToken', {}).get('symbol')}/{p.get('quoteToken', {}).get('symbol')} on {chain}")
        if addr:
            rate_limit_delay()
            detail = get_pair(chain, addr)
            if detail:
                print(f"  Price: {detail.get('priceUsd')}, Liquidity: {detail.get('liquidity', {}).get('usd')}")
