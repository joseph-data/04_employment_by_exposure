import pandas as pd
from shiny import reactive
from shiny.express import input, ui
from shinywidgets import output_widget, render_plotly
from pathlib import Path

# Import organized modules
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
from src.data_manager import load_payload
from src.plotting import create_exposure_plot

# Helpers for UI mapping
LEVEL_CHOICES = {value: label for label, value in LEVEL_OPTIONS}
METRIC_MAPPING = {value: label for label, value in METRIC_OPTIONS}
WEIGHTING_MAPPING = {value: label for label, value in WEIGHTING_OPTIONS}
YEAR_RANGE_DEFAULT = list(range(DEFAULT_YEAR_RANGE[0], DEFAULT_YEAR_RANGE[1] + 1))

# ======================================================
#  REACTIVE STATE
# ======================================================
# Load once on startup; values stay in-memory until app restart.
payload_store = reactive.Value(load_payload())


# Defaults for resetting filters
DEFAULT_METRIC = METRIC_OPTIONS[10][1]
DEFAULT_BASE_YEAR = 2022
DEFAULT_COUNT_MODE = "raw"


@reactive.calc
def filtered_data():
    payload = payload_store.get()
    if payload is None:
        return pd.DataFrame()

    df = payload[input.weighting()]

    # Filter by Level and Year
    mask = (df["level"] == int(input.level())) & (
        df["year"].between(
            input.year_range()[0], input.year_range()[1], inclusive="both"
        )
    )

    # Select columns
    metric = input.metric()
    cols = [c for c in df.columns if not c.startswith("daioe_") or metric in c]

    df_filtered = df[mask][cols].sort_values(
        [f"daioe_{metric}_exposure_level", "year"], ascending=[False, True]
    )

    return df_filtered


@reactive.calc
def base_year_choices():
    year_start, year_end = input.year_range()
    return list(range(year_start, year_end + 1))


@reactive.effect
@reactive.event(input.year_range)
def _sync_base_year_select():
    # Keep base-year choices aligned with the year slider span.
    years = base_year_choices()
    preferred_default = DEFAULT_BASE_YEAR if DEFAULT_BASE_YEAR in years else years[-1]
    try:
        selected_raw = input.base_year()
        selected = int(selected_raw) if selected_raw is not None else preferred_default
    except Exception:
        selected = preferred_default
    if selected not in years:
        selected = preferred_default
    ui.update_select("base_year", choices=years, selected=selected)


@reactive.calc
def display_series():
    df = filtered_data()
    if df.empty:
        return pd.DataFrame(), {
            "value_col": "value_for_plot",
            "y_label": "Employed persons",
            "is_index": False,
            "base_year": None,
        }

    metric = input.metric()
    exposure_col = f"daioe_{metric}_exposure_level"
    grouped = (
        df.dropna(subset=["age"])
        .groupby(["age", "year", exposure_col], as_index=False)["employment"]
        .sum()
    )

    if input.count_mode() == "raw":
        grouped["value_for_plot"] = grouped["employment"]
        return grouped, {
            "value_col": "value_for_plot",
            "y_label": "Employed persons",
            "is_index": False,
            "base_year": None,
        }

    years = base_year_choices()
    try:
        base_year_raw = input.base_year()
        base_year = (
            int(base_year_raw)
            if base_year_raw is not None
            else (DEFAULT_BASE_YEAR if DEFAULT_BASE_YEAR in years else years[-1])
        )
    except Exception:
        base_year = DEFAULT_BASE_YEAR if DEFAULT_BASE_YEAR in years else years[-1]
    if base_year not in years:
        base_year = DEFAULT_BASE_YEAR if DEFAULT_BASE_YEAR in years else years[-1]
    base = grouped[grouped["year"] == base_year][
        ["age", exposure_col, "employment"]
    ].rename(columns={"employment": "base_employment"})
    series = grouped.merge(base, on=["age", exposure_col], how="left")
    denom = series["base_employment"].replace(0, pd.NA)
    series["value_for_plot"] = series["employment"] / denom
    series["employment_index_base"] = base_year

    return series, {
        "value_col": "value_for_plot",
        "y_label": f"Index (base {base_year}=1.0)",
        "is_index": True,
        "base_year": base_year,
    }


# ======================================================
#  UI LAYOUT
# ======================================================
css_file = Path(__file__).parent / "css" / "theme.css"

ui.include_css(css_file)

ui.tags.head(
    ui.tags.link(
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    )
)

ui.page_opts(
    fillable=False,
    fillable_mobile=True,
    full_width=True,
    id="page",
    lang="en",
)

with ui.sidebar(open="always", position="right"):
    ui.input_select(
        "level",
        "Occupation Classification Level",
        LEVEL_CHOICES,
        selected=DEFAULT_LEVEL,
    )
    ui.input_select(
        "weighting", "Weighting", WEIGHTING_MAPPING, selected=DEFAULT_WEIGHTING
    )
    ui.input_select("metric", "Sub-index", METRIC_MAPPING, selected=DEFAULT_METRIC)

    ui.input_radio_buttons(
        "count_mode",
        "Employed persons display",
        {"raw": "Raw counts", "index": "Index to base year"},
        selected=DEFAULT_COUNT_MODE,
    )
    with ui.panel_conditional("input.count_mode == 'index'"):
        ui.input_select(
            "base_year",
            "Base year",
            YEAR_RANGE_DEFAULT,
            selected=DEFAULT_BASE_YEAR,
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
        "reset_filters",
        "Reset filters",
        icon=ui.tags.i(class_="fas fa-rotate-left"),
        class_="btn-primary mt-3",
    )


@reactive.effect
@reactive.event(input.reset_filters)
def _reset_filters():
    ui.update_select("level", selected=DEFAULT_LEVEL)
    ui.update_select("weighting", selected=DEFAULT_WEIGHTING)
    ui.update_select("metric", selected=DEFAULT_METRIC)
    ui.update_radio_buttons("count_mode", selected=DEFAULT_COUNT_MODE)
    ui.update_slider("year_range", value=DEFAULT_YEAR_RANGE)
    ui.update_select("base_year", selected=DEFAULT_BASE_YEAR)


# Fixed navigation structure
# with ui.navset_tab(id="main_tabs"):
# with ui.nav_panel("Visuals"):
with ui.div(style="display:flex; justify-content:center;"):
    # Keep output id aligned with render function name
    output_widget("exposure_plot")

    @render_plotly
    def exposure_plot2():  # Renamed from exposure_plot2 to match output_widget id
        df, meta = display_series()
        if df.empty:
            return None

        return create_exposure_plot(
            df,
            metric=input.metric(),
            metric_label=METRIC_MAPPING[input.metric()],
            weighting_label=WEIGHTING_MAPPING[input.weighting()],
            value_col=meta["value_col"],
            y_axis_label=meta["y_label"],
            is_index=meta["is_index"],
            base_year=meta["base_year"],
        )

# with ui.nav_panel("Data"):

#     @render.data_frame
#     def display_df():
#         df, meta = display_series()
#         if df.empty:
#             return render.DataGrid(
#                 pd.DataFrame(), height=800, selection_mode="rows", filters=True
#             )

#         table = df.copy()
#         table = table.rename(columns={"value_for_plot": "display_value"})
#         return render.DataGrid(
#             table, height=800, selection_mode="rows", filters=True
#         )

#     ui.input_radio_buttons(
#         "download_format",
#         "Download format",
#         {"csv": "CSV", "json": "JSON"},
#         selected="csv",
#     )

#     @render.download(
#         filename=lambda: f"data_{input.metric()}.{input.download_format()}"
#     )
#     def download_data():
#         df, _meta = display_series()
#         if input.download_format() == "csv":
#             yield df.to_csv(index=False)
#         else:
#             yield df.to_json(orient="records", indent=2)
