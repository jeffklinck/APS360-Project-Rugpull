"""Try to download Comparitech CSV via Datawrapper embed → data/raw/comparitech_rugpulls.csv."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

EMBED_JS_URL = "https://datawrapper.dwcdn.net/9nRA9/embed.js"
# Fallback: CSV export URL extracted from embed (sheet may require "publish to web" or link sharing)
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/16RSWQLhqtUQeRTSEB_MsEDM_vicpTVTHsk2JwzlUeug/"
    "export?format=csv&gid=1362312412"
)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "comparitech_rugpulls.csv"
REQUEST_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def get_csv_url_from_embed() -> str | None:
    """Fetch embed.js and extract google-spreadsheet-src URL."""
    try:
        r = requests.get(EMBED_JS_URL, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        text = r.text
    except Exception as e:
        print(f"Failed to fetch embed.js: {e}", file=sys.stderr)
        return None
    # "google-spreadsheet-src":"https://docs.google.com/..."
    m = re.search(r'"google-spreadsheet-src"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).replace("\\u0026", "&")
    return None


def download_csv(url: str) -> str | None:
    """Download CSV from URL; return content or None."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"Failed to download CSV: {e}", file=sys.stderr)
        return None


def main() -> int:
    url = get_csv_url_from_embed() or GOOGLE_SHEET_CSV_URL
    print(f"Using URL: {url[:80]}...")
    content = download_csv(url)
    if not content or len(content.strip()) < 100:
        print(
            "Could not get CSV (sheet may require login or publish-to-web). "
            "Download manually from the chart page or request access to Comparitech's sheet.",
            file=sys.stderr,
        )
        return 1
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    lines = content.strip().splitlines()
    print(f"Saved {len(lines)} rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
