# Rug-pull predictor (APS360)

```bash
pip3 install -r requirements.txt
```

Set `BIRDEYE_API_KEY` in `.env` for OHLCV (copy from `.env.example` if present).

Pipeline:

1. `python3 scripts/build_labeled_dataset.py`
2. `python3 scripts/build_timeseries_dataset.py`
3. `python3 scripts/train_primary.py`
4. `python3 scripts/compare_baseline_cnn_primary.py` (after step 3 checkpoint exists)

Tests: `python3 tests/run_tests.py`
