# Crypto Rug Pull Predictor

Deep learning model to predict rug pull risk from the first few hours of trading activity for a crypto token (APS360 project).

## Setup

1. Clone the repo (if you haven’t already):
   ```bash
   git clone https://github.com/jeffklinck/APS360-Project-Rugpull.git
   cd APS360-Project-Rugpull
   ```

2. (Optional) Set environment variables for the time-series pipeline: see **docs/ENV.md**. Copy `.env.example` to `.env` and add your `BIRDEYE_API_KEY` if you will run `build_timeseries_dataset.py`.

3. Install dependencies (use `python3` and `pip3` if your system doesn’t alias `python`/`pip` to Python 3):
   ```bash
   pip3 install -r requirements.txt
   ```

4. (Optional) Use a virtual environment to isolate packages:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Data source

We use the **DexScreener API** (`https://api.dexscreener.com`) for pair/token discovery and current stats (no API key; 300 req/min for search/pairs). See `scripts/dexscreener_client.py` for the client. Historical time-series for model input may use an additional source (e.g. Birdeye) later.

**Rug-pull labels** use the [Comparitech/Moody crypto scams tracker](https://www.comparitech.com/crypto/cryptocurrency-scams/). The full table (45+ pages) is on [Datawrapper](https://www.datawrapper.de/_/9nRA9/). Run `python3 scripts/scrape_datawrapper_rugpulls.py` to try downloading it as CSV into `data/raw/comparitech_rugpulls.csv`; see `scripts/comparitech_rugpulls.py` and **docs/COMPARITECH_DATA.md** for details.

## Project layout

| Path        | Purpose                                              |
|------------|------------------------------------------------------|
| `data/`    | Raw and processed token/time-series data             |
| `scripts/` | Data download, preprocessing, baseline, primary model |
| `notebooks/` | Exploration and figures for the report             |
| `configs/` | Config (e.g. `config.yaml`) for paths and DexScreener base URL |
| `CHECKIN_PLAN.md` | Plan for check-in experiments and deliverables   |

## How to run

1. **Labeled dataset (rug + non-rug):**  
   `python3 scripts/build_labeled_dataset.py`  
   Uses Comparitech (rug) and CoinGecko top-by-market-cap (non-rug), fetches pair stats from DexScreener. Writes `data/processed/labeled_pairs.csv` (and `.parquet` if pyarrow installed) and `data/processed/baseline.npz`.

2. **Time-series per token (for primary model):**  
   `export BIRDEYE_API_KEY=your_key && python3 scripts/build_timeseries_dataset.py`  
   Reads labeled pairs, gets each pair's first-trade (launch) time from Birdeye, then fetches 1‑min OHLCV for the first 2 hours from launch per pair. Builds **X (N, T, F)** with T=120, F=3 (log_return, volume, high_low_pct). Writes `data/processed/primary.npz`. See **docs/TIME_SERIES_DATA.md**.

3. **Baseline / primary training:** `python3 scripts/run_baseline.py` and `python3 scripts/train_primary.py` (TBD).

## References

See the project proposal and `CHECKIN_PLAN.md` for goals, data sources, and experiment plan.
