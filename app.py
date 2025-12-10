from typing import List, Tuple, Dict
from pathlib import Path
import os

from shiny import reactive
from shiny.express import input, ui, render, module
from shinywidgets import output_widget, render_plotly

from functools import lru_cache

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.callbacks import Points
from plotly.subplots import make_subplots

from src.config import (
    DEFAULT_LEVEL,
    DEFAULT_WEIGHTING,
    DEFAULT_YEAR_RANGE,
    GLOBAL_YEAR_MAX,
    GLOBAL_YEAR_MIN,
    LEVEL_OPTIONS,
    METRIC_OPTIONS,
    WEIGHTING_OPTIONS,
)

# ======================================================
#  PLEMINARIES
# ======================================================


DATA_DIR = Path(__file__).resolve().parent / "data"
WEIGHTED_PATH = DATA_DIR / "daioe_weighted.csv"
SIMPLE_PATH = DATA_DIR / "daioe_simple.csv"
META_PATH = DATA_DIR / "daioe_meta.csv"


def _load_cached_payload() -> Dict[str, object] | None:
    """
    Attempt to load precomputed pipeline outputs from CSV cache.
    """
    if not (WEIGHTED_PATH.exists() and SIMPLE_PATH.exists() and META_PATH.exists()):
        return None
    try:
        weighted = pd.read_csv(WEIGHTED_PATH)
        simple = pd.read_csv(SIMPLE_PATH)
        meta_df = pd.read_csv(META_PATH)
        meta_row = meta_df.iloc[0] if not meta_df.empty else {}
        daioe_cols = [
            c.strip()
            for c in str(meta_row.get("daioe_cols", "")).split(",")
            if c.strip()
        ]
        return {
            "taxonomy": meta_row.get("taxonomy", "ssyk2012"),
            "weighted": weighted,
            "simple": simple,
            "daioe_cols": daioe_cols,
            "year_min": int(meta_row["year_min"]),
            "year_max": int(meta_row["year_max"]),
        }
    except Exception as exc:
        print(f"Cache read failed, refreshing instead: {exc}")
        return None


@lru_cache(maxsize=1)
def load_pipeline(force_refresh: bool = False):
    """
    Prefer cached CSVs; fall back to running the pipeline.
    Set env FORCE_REFRESH=1 or pass force_refresh=True to force a refresh.
    """
    from src import main as pipeline_main

    if not force_refresh and os.getenv("FORCE_REFRESH", "0") != "1":
        cached = _load_cached_payload()
        if cached is not None:
            return cached

    print("Running pipeline (cache miss or forced refresh)...")
    payload = pipeline_main.run_pipeline()  # also writes fresh CSVs
    return payload


# Hold Data so UI can refresh it on demand
payload_state = reactive.Value(load_pipeline())

# Shared UI options.

LEVEL_CHOICES = {value: label for label, value in LEVEL_OPTIONS}


def metric_mapping() -> Dict[str, str]:
    return {value: label for label, value in METRIC_OPTIONS}


def weighting_mapping() -> Dict[str, str]:
    return {value: label for label, value in WEIGHTING_OPTIONS}


#### APP BEGINS HERE

# with ui.div(class_="col-md-10 col-lg-8 py-5 mx-auto text-lg-center text-left"):
#     ui.h1("Number of Employed Persons by Level of AI Exposure")


with ui.sidebar(open="desktop", bg="#f8f8f8"):
    ui.input_select(
        "level",
        "Level",
        LEVEL_CHOICES,
        selected=DEFAULT_LEVEL,
    )

    ui.input_select(
        "weighting",
        "Weighting",
        weighting_mapping(),
        selected=DEFAULT_WEIGHTING,
    )

    ui.input_select(
        "metric",
        "Sub-index",
        metric_mapping(),
        selected=METRIC_OPTIONS[0][1],
    )

    ui.input_slider(
        "year_range",
        "Year range",
        min=GLOBAL_YEAR_MIN,
        max=GLOBAL_YEAR_MAX,
        value=DEFAULT_YEAR_RANGE,
        step=1,
        sep="",
    )
    ui.input_action_button(
        "refresh_data",
        "Refresh data",
        class_="btn btn-primary mt-3",
    )


@reactive.calc
def filtered_data():
    weighting = input.weighting()
    level = int(input.level())
    df = payload_state()[weighting]
    idx1 = df["level"] == level
    idx2 = df["year"].between(
        left=input.year_range()[0], right=input.year_range()[1], inclusive="both"
    )
    metric = input.metric()

    cols = [c for c in df.columns if not c.startswith("daioe_") or metric in c]
    df_filtered = df[idx1 & idx2][cols]
    df_sorted = df_filtered.sort_values(
        [f"daioe_{metric}_exposure_level", "year"], ascending=[False, True]
    )
    return df_sorted


@reactive.effect
@reactive.event(input.refresh_data)
def _refresh_payload():
    """
    Refresh the pipeline outputs on demand, showing a progress bar.
    """
    load_pipeline.cache_clear()
    with ui.Progress(min=0, max=3) as p:
        p.set(1, message="Refreshing data…")
        new_payload = load_pipeline(force_refresh=True)
        p.set(3, message="Done")
    payload_state.set(new_payload)


with ui.nav_panel("Visuals"):
    with ui.div(style="display:flex; justify-content:center;"):
        output_widget("plot")

        @render_plotly
        def exposure_plot():
            df = filtered_data()
            metric = input.metric()
            exposure_col = f"daioe_{metric}_exposure_level"

            df = df.dropna(subset=["age", exposure_col])
            age_groups = sorted(df["age"].unique())

            fig = make_subplots(
                rows=len(age_groups),
                cols=1,
                shared_xaxes=False,
                subplot_titles=[
                    f"Employed Persons Aged {age} Years by AI Exposure ({metric_mapping()[metric]}, {weighting_mapping()[input.weighting()]})"
                    for age in age_groups
                ],
                vertical_spacing=0.03,
            )

            for i, age in enumerate(age_groups, start=1):
                df_age = df[df["age"] == age]
                df_plot = df_age.groupby(["year", exposure_col], as_index=False)[
                    "employment"
                ].sum()

                for exposure_level, sub in df_plot.groupby(exposure_col):
                    fig.add_trace(
                        go.Scatter(
                            x=sub["year"],
                            y=sub["employment"],
                            mode="lines+markers",
                            line=dict(width=3),
                            marker=dict(size=9),
                            name=f"Level {exposure_level}",
                            showlegend=(
                                i == 1
                            ),  # Only show legend items for the first plot
                            hovertemplate=(
                                "Age: %{customdata[0]}<br>"
                                "Exposure Level: Level %{customdata[1]}<br>"
                                "Year: %{x}<br>"
                                "Employed Persons: %{y:,}<extra></extra>"
                            ),
                            customdata=list(
                                zip([age] * len(sub), [exposure_level] * len(sub))
                            ),
                        ),
                        row=i,
                        col=1,
                    )

                # X label on each subplot
                fig.update_xaxes(
                    title_text="Year",
                    tickmode="linear",
                    dtick=1,
                    row=i,
                    col=1,
                )

                # Y label on each subplot
                fig.update_yaxes(
                    title_text="Employed Persons",
                    tickformat=",",
                    rangemode="tozero",
                    row=i,
                    col=1,
                )

            # Shift Subplot Titles Up
            fig.update_annotations(yshift=30)

            fig.update_layout(
                height=700 * len(age_groups),
                width=1300,
                legend=dict(
                    title="Exposure Level",
                    orientation="h",
                    x=0.5,
                    y=1.02,
                    xanchor="center",
                    yanchor="bottom",
                ),
                margin=dict(t=100, l=50, r=80, b=40),
            )

            return fig


with ui.nav_panel("Data"):

    @render.data_frame
    def display_df():
        df = filtered_data()
        return render.DataGrid(
            df,
            height=800,
            selection_mode="rows",
            filters=True,
        )

    ui.input_radio_buttons(
        "download_format",
        "Download format",
        {"csv": "CSV", "json": "JSON"},
        selected="csv",
    )

    # ui.download_button("download_data", "Download Filtered Data")

    @render.download(
        filename=lambda: f"employment_data_{input.metric()}_{input.level()}.{input.download_format()}"
    )
    def download_data():
        df = filtered_data()
        format_type = input.download_format()

        if format_type == "csv":
            # Yield CSV text (Shiny will handle encoding)
            yield df.to_csv(index=False)

        elif format_type == "json":
            # Yield JSON text
            yield df.to_json(orient="records", indent=2)


# with ui.layout_columns(col_widths=[12]):
#     for i in range(1, 9):
#         with ui.card():
#             f"Card {i}"
