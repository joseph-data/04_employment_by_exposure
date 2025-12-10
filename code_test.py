import sys
from pathlib import Path
from functools import lru_cache

import pandas as pd
import plotly.express as px

# ======================================================
# Make local src importable
# ======================================================
ROOT = Path.cwd().resolve().parent
sys.path.insert(0, str(ROOT))


@lru_cache(maxsize=1)
def load_pipeline():
    """
    Load and cache the pipeline payload.

    Runs src.main.run_pipeline() exactly once.
    Subsequent calls reuse the cached payload.
    """
    from src import main as pipeline_main

    return pipeline_main.run_pipeline()


# ======================================================
# 1. Select Exposure Weighting
# ======================================================
# Options of weighting include: "simple" or "weighted"
payload = load_pipeline()
df = payload["weighted"]


# ======================================================
# 2. Row Filters
# ======================================================

# Filter for SSYK level 1, 2, 3, 4 occupations
idx1 = df["level"] == 3

# Restrict to year 2022
idx2 = df["year"].between(left=2014, right=2023, inclusive="both")


# ======================================================
# 3. Column Filters
# ======================================================
# Keep all non-DAIOE columns, and from DAIOE columns
# keep only those related to "videogames".
cols = [c for c in df.columns if not c.startswith("daioe_") or "allapps" in c]


# ======================================================
# 4. Final Filtered Output
# ======================================================
df_filtered = df[idx1 & idx2][cols]


import plotly.express as px

# Choose age group (e.g. "16-65", "16-24", "25-29", ...)
AGE_GROUP = "16-24"

# 1. Filter to the chosen age group
df_age = df_filtered[df_filtered["age"] == AGE_GROUP].copy()

# Optional: guard against empty result
if df_age.empty:
    raise ValueError(f"No rows found for age group {AGE_GROUP!r}.")

# 2. Aggregate employment by year and exposure level
df_plot = df_age.groupby(["year", "daioe_allapps_exposure_level"], as_index=False)[
    "employment"
].sum()

# 3. Line plot: employment over time, colored by exposure level
fig = px.line(
    df_plot,
    x="year",
    y="employment",
    color="daioe_allapps_exposure_level",
    markers=True,
    labels={
        "year": "Year",
        "employment": "Number of employed persons",
        "daioe_allapps_exposure_level": "Exposure level",
    },
    title=f"Employment over time for age group {AGE_GROUP} by exposure level",
)

fig.show()
