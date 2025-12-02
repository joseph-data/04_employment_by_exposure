"""
Shiny app: Employment by AI exposure (5 levels), faceted by age group.
Uses simple-average exposure scores from
data/03_daioe_aggregated/daioe_ssyk2012_simple_avg.csv and SCB AKU
employment pulled via scripts/04_occ.py.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_line,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_continuous,
    theme,
    theme_bw,
)
from shiny import App, render, ui

ROOT = Path(__file__).resolve().parent
EXPOSURE_PATH = ROOT / "data" / "03_daioe_aggregated" / "daioe_ssyk2012_simple_avg.csv"
OCC_PATH = ROOT / "scripts" / "04_occ.py"

# Keep labels consistent with app.py (emoji + text)
METRIC_OPTIONS: List[Tuple[str, str]] = [
    ("📚 All Applications", "allapps"),
    ("♟️ Abstract strategy games", "stratgames"),
    ("🎮 Real-time video games", "videogames"),
    ("🖼️🔎 Image recognition", "imgrec"),
    ("🧩🖼️ Image comprehension", "imgcompr"),
    ("🖌️🖼️ Image generation", "imggen"),
    ("📖 Reading comprehension", "readcompr"),
    ("✍️🤖 Language modelling", "lngmod"),
    ("🌐🔤 Translation", "translat"),
    ("🗣️🎙️ Speech recognition", "speechrec"),
    ("🧠✨ Generative AI", "genai"),
]
METRIC_LABELS: Dict[str, str] = {key: label for label, key in METRIC_OPTIONS}
METRIC_REVERSE: Dict[str, str] = {label: key for label, key in METRIC_OPTIONS}

AGE_ORDER: List[str] = [
    "16-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
]
AGE_LABELS: Dict[str, str] = {age: f"{age} Years" for age in AGE_ORDER}
AGE_LABEL_ORDER: List[str] = [AGE_LABELS[age] for age in AGE_ORDER]


def _load_occ_module():
    """Load the employment fetcher from scripts/04_occ.py."""
    spec = importlib.util.spec_from_file_location("scripts.occ", OCC_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_employment() -> pd.DataFrame:
    """Fetch SCB AKU employment by occupation, age, and year."""
    occ_mod = _load_occ_module()
    df = occ_mod.fetch_scb_aku_occupations()
    df = df.rename(columns={"code_3": "code"})
    df["code"] = df["code"].astype(str).str.zfill(3)
    df["year"] = df["year"].astype(int)
    df["value"] = df["value"].astype(int)
    df = df[df["age"].isin(AGE_ORDER)].copy()
    return df


@lru_cache(maxsize=len(METRIC_LABELS))
def load_exposure(metric: str) -> pd.DataFrame:
    """
    Load exposure data (simple average) and assign 5 exposure levels
    using percentile rankings for the chosen metric.
    """
    pct_col = f"pct_rank_{metric}"
    df = pd.read_csv(EXPOSURE_PATH)
    df = df[df["level"] == 3].copy()
    df = df[df[pct_col].notna()]
    df["code"] = df["code"].astype(int).astype(str).str.zfill(3)
    df["year"] = df["year"].astype(int)
    df["exposure_level"] = pd.cut(
        df[pct_col],
        bins=np.linspace(0, 1, 6),
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype(int)
    return df[["code", "year", "exposure_level"]]


@lru_cache(maxsize=1)
def available_years() -> List[int]:
    """
    Determine years available in both employment and exposure data to
    keep base-year choices valid across metrics.
    """
    emp_years = set(load_employment()["year"].unique())
    exp_df = pd.read_csv(EXPOSURE_PATH)
    exp_years = set(exp_df[exp_df["level"] == 3]["year"].astype(int).unique())
    years = sorted(emp_years & exp_years)
    return years


BASE_YEAR_CHOICES: Dict[str, str] = {
    "none": "No index (raw employment)",
    **{str(y): f"Base {y}=1" for y in available_years()},
}


def build_employment_series(metric: str, base_year: int | None) -> pd.DataFrame:
    """
    Merge employment with exposure levels and aggregate employment by
    age/year/exposure level. Optionally index to the chosen base year.
    """
    metric_key = METRIC_REVERSE.get(metric, metric)  # accept either label or key

    emp = load_employment()
    exp = load_exposure(metric_key)

    merged = emp.merge(exp, on=["code", "year"], how="inner")
    grouped = merged.groupby(["age", "year", "exposure_level"], as_index=False)[
        "value"
    ].sum()
    grouped = grouped.rename(columns={"value": "employment"})

    if base_year is not None:
        base = grouped[grouped["year"] == base_year][
            ["age", "exposure_level", "employment"]
        ].rename(columns={"employment": "base_employment"})

        grouped = grouped.merge(base, on=["age", "exposure_level"], how="left")
        grouped = grouped[grouped["base_employment"].notna()].copy()
        grouped["series_value"] = grouped["employment"] / grouped["base_employment"]
    else:
        grouped["series_value"] = grouped["employment"]

    grouped["age"] = pd.Categorical(grouped["age"], categories=AGE_ORDER, ordered=True)
    grouped["age_label"] = grouped["age"].map(AGE_LABELS)
    grouped["age_label"] = pd.Categorical(
        grouped["age_label"], categories=AGE_LABEL_ORDER, ordered=True
    )
    grouped = grouped.sort_values(["age", "exposure_level", "year"])
    return grouped


def make_plot(df: pd.DataFrame, metric: str, base_year: int | None):
    """Create a faceted line plot similar to the provided reference."""
    # Color-blind–friendly 5-color palette (Okabe-Ito inspired)
    palette = {
        1: "#0072B2",  # blue
        2: "#56B4E9",  # light blue
        3: "#009E73",  # green
        4: "#E69F00",  # orange
        5: "#D55E00",  # vermillion
    }

    title = "Employment by AI exposure (5 levels) by age group"
    subtitle = "Aggregated DAIOE, SCB AKU employment"
    y_label = f"Index (base={base_year}=1)" if base_year is not None else "Employment"

    p = (
        ggplot(df, aes(x="year", y="series_value", color="factor(exposure_level)"))
        + geom_line(size=1)
        + facet_wrap("~age_label", ncol=3)
        + scale_color_manual(
            values=palette,
            name="AI exposure",
            labels=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
        )
        + scale_x_continuous(breaks=sorted(df["year"].unique()))
        + labs(title=title, subtitle=subtitle, x="Year", y=y_label)
        + theme_bw()
        + theme(
            figure_size=(14, 10),
            axis_text_x=element_text(rotation=45, hjust=1),
            plot_title=element_text(size=12, weight="bold"),
            plot_subtitle=element_text(size=10),
            legend_position="top",
        )
    )
    if base_year is not None:
        p = p + geom_vline(
            xintercept=base_year,
            linetype="dashed",
            color="#555555",
            alpha=0.6,
            size=0.8,
        )
    return p


app_ui = ui.page_fluid(
    ui.input_select(
        "metric",
        "Exposure metric",
        # mapping consistent with app.py; keys are metric ids, values are emoji labels
        choices={key: label for label, key in METRIC_OPTIONS},
        selected="genai",
    ),
    ui.input_select(
        "base_year",
        "Base year (optional index)",
        choices=BASE_YEAR_CHOICES,
        selected="none",
    ),
    ui.output_plot("exposure_plot", width="100%", height="1800px"),
)


def server(input, output, session):
    @render.plot
    def exposure_plot():
        metric = input.metric()
        base_year_raw = input.base_year()
        base_year = None if base_year_raw == "none" else int(base_year_raw)
        df = build_employment_series(metric, base_year)
        return make_plot(df, metric, base_year)


app = App(app_ui, server)


if __name__ == "__main__":
    # Run with: shiny run --reload app_ai_exposure.py
    app.run()
