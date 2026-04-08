"""Rug name set for labels; optional full CSV at data/raw/comparitech_rugpulls.csv."""

from __future__ import annotations

import re
from pathlib import Path

# Seed list: named scams from the Comparitech "biggest" list (article text).
# Normalized to lowercase for matching; add variants/symbols as needed.
COMPARITECH_KNOWN_RUGPULLS: set[str] = {
    "onecoin",
    "africrypt",
    "gainbitcoin",
    "bitconnect",
    "plustoken",
    "wirecard",
    "thodex",
    "wotoken",
    "arbistar",
    "bitclub",
    "bitclub network",
}

# Optional: token symbols that often appear in scam contexts (extend as needed)
COMPARITECH_KNOWN_SYMBOLS: set[str] = set()


def _normalize(s: str) -> str:
    """Lowercase and collapse non-alphanumeric for fuzzy match."""
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def is_known_rugpull(name: str | None, symbol: str | None) -> bool:
    """
    Return True if the token name or symbol matches a known Comparitech scam.
    name/symbol can be from DexScreener (e.g. baseToken.name, baseToken.symbol).
    """
    if not name and not symbol:
        return False
    norm_name = _normalize(name or "")
    norm_symbol = _normalize(symbol or "")
    for known in COMPARITECH_KNOWN_RUGPULLS | COMPARITECH_KNOWN_SYMBOLS:
        norm_known = _normalize(known)
        if not norm_known:
            continue
        if norm_known in norm_name or norm_known in norm_symbol:
            return True
        # Exact match for short symbols
        if norm_symbol and norm_known == norm_symbol:
            return True
    return False


def load_extra_list(csv_path: Path | None = None) -> set[str]:
    """
    Load additional names/symbols from a CSV.
    Expected format: one column 'name' or 'symbol' (or first column used).
    Returns a set of normalized strings to add to the known list.
    """
    path = csv_path or Path(__file__).resolve().parent.parent / "data" / "raw" / "comparitech_rugpulls.csv"
    if not path.exists():
        return set()
    out: set[str] = set()
    try:
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                for key in (
                    "name",
                    "symbol",
                    "token",
                    "project",
                    "Company Name",  # main column in Comparitech Datawrapper CSV
                    "City",
                    "Details",
                    "Type of Crypto/Token Affected",
                ):
                    if key in row and row[key].strip():
                        out.add(row[key].strip().lower())
                if not any(row.get(k) for k in ("name", "symbol", "token", "project", "Company Name", "City", "Details")) and row:
                    first = list(row.values())[0]
                    if first and first.strip():
                        out.add(first.strip().lower())
    except Exception:
        pass
    return out


def get_known_rugpulls(include_extra: bool = True) -> set[str]:
    """All known rug-pull names/symbols (seed + optional CSV)."""
    s = set(COMPARITECH_KNOWN_RUGPULLS | COMPARITECH_KNOWN_SYMBOLS)
    if include_extra:
        s |= load_extra_list()
    return s


if __name__ == "__main__":
    print("Comparitech seed list:", sorted(COMPARITECH_KNOWN_RUGPULLS))
    print("BitConnect name match:", is_known_rugpull("BitConnect", "BCC"))
    print("Random token match:", is_known_rugpull("Random Token", "RND"))
