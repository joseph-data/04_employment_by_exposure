import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Configuration / constants
# ============================================================

DEFAULT_LINE_COLORS: dict[int, str] = {
    1: "#1f77b4",
    2: "#d62728",
    3: "#2ca02c",
    4: "#9467bd",
    5: "#ff7f0e",
}

HOVER_TEMPLATE_INDEX = (
    "Age: %{customdata[0]}<br>"
    "Exposure Level: Level %{customdata[1]} (1 = least exposed, 5 = most exposed)<br>"
    "Year: %{x}<br>"
    "Index: %{y:.1f}<extra></extra>"
)

HOVER_TEMPLATE_RAW = (
    "Age: %{customdata[0]}<br>"
    "Exposure Level: Level %{customdata[1]} (1 = least exposed, 5 = most exposed)<br>"
    "Year: %{x}<br>"
    "Number of Employed Persons: %{y:,}<extra></extra>"
)


# ============================================================
# Helper functions
# ============================================================


def _build_palette(line_colors: dict[int, str] | None) -> dict[int, str]:
    """
    Merge user-supplied colors with defaults (user overrides default).
    """
    return {**DEFAULT_LINE_COLORS, **(line_colors or {})}


def _resolve_color(
    exposure_level: int | str,
    palette: dict[int, str],
) -> str | None:
    """
    Get color for an exposure level, trying both raw and int-casted keys.
    """
    # Try using the level directly (if it's already an int key).
    color = palette.get(exposure_level)  # type: ignore[arg-type]
    if color is not None:
        return color

    # Fall back to int(exposure_level) if possible.
    try:
        level_int = int(exposure_level)
        return palette.get(level_int)
    except (TypeError, ValueError):
        return None


# ============================================================
# Main plotting function
# ============================================================


def create_exposure_plot(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    weighting_label: str,
    *,
    value_col: str = "employment",
    y_axis_label: str = "Employed Persons",
    is_index: bool = False,
    base_year: int | None = None,
    line_colors: dict[int, str] | None = None,
) -> go.Figure:
    """
    Generate a multi-row subplot figure for AI exposure by age group.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns 'age', 'year', value_col and
        'daioe_{metric}_exposure_level'.
    metric : str
        Metric name used in the exposure column suffix.
    metric_label : str
        Human-readable label for the metric (for titles).
    weighting_label : str
        Label describing the weighting approach (for titles).
    value_col : str, default "employment"
        Column used for the Y-axis (e.g., counts or indices).
    y_axis_label : str, default "Employed Persons"
        Y-axis title.
    is_index : bool, default False
        If True, use index-style hover text; otherwise value-style.
    base_year : int | None, default None
        Optional vertical reference line and annotation for a base year.
    line_colors : dict[int, str] | None, default None
        Optional mapping of exposure level -> hex color. Overrides defaults.

    Returns
    -------
    go.Figure
        A Plotly Figure with one subplot per age group.
    """
    exposure_col = f"daioe_{metric}_exposure_level"

    # ------------------------------------------------------------------
    # 1. Clean and prepare data
    # ------------------------------------------------------------------
    df_clean = df.dropna(subset=["age", exposure_col, value_col]).copy()
    age_groups = sorted(df_clean["age"].unique())

    if not age_groups:
        # No valid data to plot
        return go.Figure()

    hover_template = HOVER_TEMPLATE_INDEX if is_index else HOVER_TEMPLATE_RAW
    palette = _build_palette(line_colors)

    # ------------------------------------------------------------------
    # 2. Create multi-row subplot scaffolding
    # ------------------------------------------------------------------
    subplot_titles = [
        (
            f"<b>Employed Persons Aged {age} Years by AI Exposure Level "
            f"({metric_label}, {weighting_label})</b>"
        )
        for age in age_groups
    ]

    fig = make_subplots(
        rows=len(age_groups),
        cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.03,
    )

    # ------------------------------------------------------------------
    # 3. Add traces per age group and exposure level
    # ------------------------------------------------------------------
    for i, age in enumerate(age_groups, start=1):
        df_age = df_clean[df_clean["age"] == age]

        # Aggregate by year and exposure level
        df_plot = df_age.groupby(["year", exposure_col], as_index=False)[
            value_col
        ].sum()

        for exposure_level, sub in df_plot.groupby(exposure_col):
            color = _resolve_color(exposure_level, palette)

            fig.add_trace(
                go.Scatter(
                    x=sub["year"],
                    y=sub[value_col],
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                    marker=dict(size=9, color=color),
                    name=f"Level {exposure_level}",
                    showlegend=(i == 1),  # legend only in first row
                    hovertemplate=hover_template,
                    customdata=list(
                        zip(
                            [age] * len(sub),
                            [exposure_level] * len(sub),
                        )
                    ),
                ),
                row=i,
                col=1,
            )

        # Axes for this row
        fig.update_xaxes(
            title_text="Year",
            tickmode="linear",
            dtick=1,
            row=i,
            col=1,
        )
        fig.update_yaxes(
            title_text=y_axis_label,
            tickformat=",",
            rangemode="tozero",
            row=i,
            col=1,
        )

    # ------------------------------------------------------------------
    # 4. Global layout tweaks
    # ------------------------------------------------------------------
    fig.update_annotations(yshift=30)
    fig.update_layout(
        height=700 * len(age_groups),
        width=1000,
        legend=dict(
            title="Exposure Level (1 = least exposed, 5 = most exposed)",
            orientation="h",
            x=0.5,
            y=1.02,
            xanchor="center",
            yanchor="bottom",
            bordercolor="#c7c7c7",
            borderwidth=2,
            bgcolor="#f9f9f9",
            font=dict(size=12),
        ),
        margin=dict(t=100, l=50, r=80, b=40),
        plot_bgcolor="#f5f7fb",
        xaxis_showgrid=True,
        # yaxis_showgrid=True,
        # xaxis_gridcolor="#e6ecf5",
        # yaxis_gridcolor="#e6ecf5",
    )

    # ------------------------------------------------------------------
    # 5. Optional base-year line and annotation
    # ------------------------------------------------------------------
    if base_year is not None:
        fig.add_vline(
            x=base_year,
            line_width=2,
            line_dash="dash",
            line_color="black",
            opacity=0.8,
            row="all",
            col=1,
        )

        annotation_text = (
            "Base year 2022 — ChatGPT launch and generative AI takeoff"
            if base_year == 2022
            else f"Base year {base_year} (normalization anchor)"
        )

        n_rows = len(age_groups)

        for i in range(1, n_rows + 1):
            # Plotly validation rules:
            # - First subplot (i=1) must use 'x' and 'y domain' (with space).
            # - Subsequent subplots (i>1) must use 'x{i}' and 'y{i} domain'.
            if i == 1:
                xref_val = "x"
                yref_val = "y domain"
            else:
                xref_val = f"x{i}"
                yref_val = f"y{i} domain"

            fig.add_annotation(
                x=base_year,
                xref=xref_val,
                y=0.955,  # Position just above the plot area (1.0)
                yref=yref_val,
                text=annotation_text,
                showarrow=False,
                font=dict(color="black", size=11),
                bgcolor="rgba(255,255,255,0.7)",
                yshift=10,  # Slight upward shift to clear titles/ticks
            )

    return fig
