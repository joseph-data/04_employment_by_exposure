import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


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
):
    """
    Generates the multi-row subplot figure for AI exposure.
    """
    exposure_col = f"daioe_{metric}_exposure_level"

    # Clean data for plotting
    df_clean = df.dropna(subset=["age", exposure_col, value_col]).copy()
    age_groups = sorted(df_clean["age"].unique())

    if not age_groups:
        return go.Figure()

    hover_template_index = (
        "Age: %{customdata[0]}<br>"
        "Exposure Level: Level %{customdata[1]}<br>"
        "Year: %{x}<br>"
        "Index: %{y:.1f}<extra></extra>"
    )
    hover_template_raw = (
        "Age: %{customdata[0]}<br>"
        "Exposure Level: Level %{customdata[1]}<br>"
        "Year: %{x}<br>"
        "Number of Employed Persons: %{y:,}<extra></extra>"
    )
    hover_template = hover_template_index if is_index else hover_template_raw

    fig = make_subplots(
        rows=len(age_groups),
        cols=1,
        shared_xaxes=False,
        subplot_titles=[
            f"Employed Persons Aged {age} Years by AI Exposure ({metric_label}, {weighting_label})"
            for age in age_groups
        ],
        vertical_spacing=0.03,
    )

    for i, age in enumerate(age_groups, start=1):
        df_age = df_clean[df_clean["age"] == age]
        df_plot = df_age.groupby(["year", exposure_col], as_index=False)[value_col].sum()

        for exposure_level, sub in df_plot.groupby(exposure_col):
            fig.add_trace(
                go.Scatter(
                    x=sub["year"],
                    y=sub[value_col],
                    mode="lines+markers",
                    line=dict(width=3),
                    marker=dict(size=9),
                    name=f"Level {exposure_level}",
                    showlegend=(i == 1),
                    hovertemplate=hover_template,
                    customdata=list(zip([age] * len(sub), [exposure_level] * len(sub))),
                ),
                row=i,
                col=1,
            )

        fig.update_xaxes(title_text="Year", tickmode="linear", dtick=1, row=i, col=1)
        fig.update_yaxes(
            title_text=y_axis_label,
            tickformat=",",
            rangemode="tozero",
            row=i,
            col=1,
        )

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
    if base_year is not None:
        fig.add_vline(
            x=base_year,
            line_width=2,
            line_dash="dash",
            line_color="black",
            opacity=0.8,
        )
        fig.add_annotation(
            x=base_year,
            y=1.03,
            xref="x",
            yref="paper",
            text=f"Base year {base_year} (normalization anchor)",
            showarrow=False,
            font=dict(color="black", size=12),
            bgcolor="rgba(255,255,255,0.7)",
        )
    return fig
