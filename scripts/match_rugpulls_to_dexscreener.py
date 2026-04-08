"""DexScreener search hit rate for names in data/raw/comparitech_rugpulls.csv."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

# Add scripts dir for imports when run from project root
_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))

from dexscreener_client import search

_project_root = Path(__file__).resolve().parent.parent
CSV_PATH = _project_root / "data" / "raw" / "comparitech_rugpulls.csv"
RATE_LIMIT_DELAY = 60 / 280  # 280 requests per minute to stay under 300


def get_searchable_names(csv_path: Path) -> list[str]:
    """Unique, non-empty names from Company Name and Type of Crypto/Token Affected."""
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
                # Split on newline/comma for multiple tokens
                for part in re.split(r"[\n,]+", val):
                    part = part.strip()
                    if len(part) >= 2 and part.lower() not in seen:
                        seen.add(part.lower())
                        out.append(part)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Match Comparitech rug pulls to DexScreener search")
    ap.add_argument("--sample", type=int, default=None, help="Only check first N names (default: all)")
    ap.add_argument("--delay", type=float, default=RATE_LIMIT_DELAY, help="Seconds between API calls")
    args = ap.parse_args()

    names = get_searchable_names(CSV_PATH)
    if not names:
        print("No names found in CSV. Run scrape_datawrapper_rugpulls.py first.", file=sys.stderr)
        return 1

    if args.sample:
        names = names[: args.sample]
    print(f"Checking {len(names)} unique names against DexScreener search...")
    print(f"Delay between calls: {args.delay:.2f}s")
    matched = []
    no_match = []

    for i, name in enumerate(names):
        try:
            pairs = search(name)
            if pairs and len(pairs) > 0:
                matched.append((name, len(pairs)))
            else:
                no_match.append(name)
        except Exception as e:
            print(f"  Error searching '{name}': {e}", file=sys.stderr)
            no_match.append(name)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(names)} (matched so far: {len(matched)})")
        time.sleep(args.delay)

    n = len(names)
    m = len(matched)
    print()
    print("Results")
    print("-------")
    print(f"Total names checked:    {n}")
    print(f"Matched (≥1 pair):     {m} ({100 * m / n:.1f}%)")
    print(f"No / zero results:     {n - m}")
    if matched:
        print()
        print("Sample matches (name → number of pairs):")
        for name, count in sorted(matched, key=lambda x: -x[1])[:15]:
            print(f"  {name!r} → {count} pairs")
    if no_match and len(no_match) <= 30:
        print()
        print("No match:", no_match[:30])
    elif no_match:
        print()
        print("Sample no-match:", no_match[:15])

    return 0


if __name__ == "__main__":
    sys.exit(main())
