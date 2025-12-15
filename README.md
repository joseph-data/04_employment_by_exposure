---
title: DAIOE Explorer - Employment
emoji: 🌍
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
license: mit
---
# DAIOE Exposure Explorer (Shiny for Python)

Interactive Shiny app for exploring AI exposure across SSYK2012 occupations. It fetches Statistics Sweden (SCB) employment data, blends it with DAIOE exposure scores, and visualizes employment by exposure level, age group, and year (with raw counts or indexed to a chosen base year). DAIOE scores are pulled from the pre-translated SSYK2012 CSV in `joseph-data/07_translate_ssyk`; no translation is performed inside this app.

## Quick start
- Python: 3.11+ recommended.
- Install deps: `pip install -r requirements.txt`
- Run the app: `shiny run --reload app.py` (or `shiny run --port 8000 app.py`)
- In the browser: choose Level, Weighting, Sub-index, Year range, and whether to display raw counts or index to a base year; the base-year select is tied to the slider range. Hit **Refresh data** to force a recompute/cache refresh.

## Project layout
- `app.py` — Shiny UI/server: sidebar inputs, plot tab (`exposure_plot`), data grid + download tab.
- `src/config.py` — UI defaults and data source constants (DAIOE CSV, SCB tables, year bounds).
- `src/scb_fetch.py` — SCB API client; pulls employment by occupation/age/year and applies exclusions.
- `src/pipeline.py` — Core ETL: cleanse DAIOE, fetch/prepare employment, aggregate to all levels, compute percentiles/exposure buckets, returns `{weighted, simple}` DataFrames.
- `src/data_manager.py` — Caching wrapper: runs the pipeline, writes CSVs to `data/` (or `DATA_CACHE_DIR`) for reuse.
- `src/plotting.py` — Plotly figure builder used by the app.

## Data flow
1. `load_payload()` (`src/data_manager.py`) loads cached CSVs if present; otherwise runs the pipeline and writes cache files.
2. `run_pipeline()` (`src/pipeline.py`) reads DAIOE scores, fetches SCB employment, aligns year bounds, aggregates to levels 1–4 with weighted or simple averages, adds percentile and exposure level columns, and returns two DataFrames.
3. `app.py` keeps the pipeline output in a reactive store, filters by inputs, renders the Plotly subplot across age groups (with an optional base-year index line/label), and provides a filterable grid plus CSV/JSON downloads using the current display mode.

## Caching and downloads
- Cache location defaults to `data/` in the repo; override with `DATA_CACHE_DIR=/path/to/cache`.
- Use the **Refresh data** button to force re-fetch and recompute (clears the in-memory cache and rewrites CSVs).
- Downloads use the currently filtered view and the selected format.

## Notes and troubleshooting
- First run may take time while SCB data is fetched and the DAIOE file is loaded; cached runs are fast.
- If plots show blank, check that the pipeline caches exist and the output widget id (`exposure_plot`) matches the render function (already aligned).
- Network access is required for SCB and the pre-translated DAIOE CSV unless you point those constants to local files in `src/config.py`.
- In index mode the dashed vertical line and label mark the selected base year used for normalization.
