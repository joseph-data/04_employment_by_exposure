"""
Core pipeline logic: merging DAIOE scores with SCB employment weights.

All data loading and aggregation steps are contained here, making this the
central data transformation module.
"""

from __future__ import annotations

# Import constants and the new SCB fetch function
from .config import DAIOE_SOURCE, DEFAULT_SEP, TAXONOMY
from .scb_fetch import fetch_all_employment_data


from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers (Copied from main.py)
# ---------------------------------------------------------------------------


def split_code_label(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    parts = series.astype(str).str.split(" ", n=1, expand=True)
    parts = parts.fillna({0: "", 1: ""})
    return parts[0], parts[1]


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")


def filter_years(
    df: pd.DataFrame, year_min: Optional[int], year_max: Optional[int], *, year_col: str
) -> pd.DataFrame:
    """Filter a frame to [year_min, year_max] bounds when provided."""
    if year_min is None and year_max is None:
        return df.copy()
    mask = pd.Series(True, index=df.index, dtype=bool)
    if year_min is not None:
        mask &= df[year_col] >= year_min
    if year_max is not None:
        mask &= df[year_col] <= year_max
    mask = mask.fillna(False)
    return df[mask].copy()


def resolve_year_bounds(
    daioe_df: pd.DataFrame,
    emp_df: pd.DataFrame,
    override_min: Optional[int],
    override_max: Optional[int],
) -> Tuple[int, int]:
    """
    Determine year bounds from data; overrides win if provided.
    """
    daioe_years = daioe_df["year"].dropna().astype(int)
    emp_years = emp_df["year"].dropna().astype(int)
    if daioe_years.empty or emp_years.empty:
        raise ValueError("Cannot infer year range: missing 'year' values in data.")

    auto_min = max(daioe_years.min(), emp_years.min())
    auto_max = min(daioe_years.max(), emp_years.max())

    year_min = override_min if override_min is not None else auto_min
    year_max = override_max if override_max is not None else auto_max
    year_min = max(year_min, auto_min)
    year_max = min(year_max, auto_max)

    if year_min > year_max:
        raise ValueError(
            "No overlapping years between DAIOE and employment data "
            f"(DAIOE {daioe_years.min()}–{daioe_years.max()}, "
            f"employment {emp_years.min()}–{emp_years.max()})."
        )
    return year_min, year_max


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_daioe_raw(
    source: str | Path = DAIOE_SOURCE, sep: str = DEFAULT_SEP
) -> pd.DataFrame:
    """Load the translated DAIOE CSV for SSYK2012."""
    return pd.read_csv(source, sep=sep)


def prepare_daioe(
    raw: pd.DataFrame,
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean DAIOE frame, parse code/label columns, filter years, and coerce metrics.
    """
    df = raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    ensure_columns(df, ["year"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = filter_years(df, year_min, year_max, year_col="year")

    daioe_cols = [col for col in df.columns if col.startswith("daioe_")]
    if not daioe_cols:
        raise KeyError("Expected at least one 'daioe_*' column in DAIOE file.")

    level_cols = {
        4: "ssyk2012_4",
        3: "ssyk2012_3",
        2: "ssyk2012_2",
        1: "ssyk2012_1",
    }
    ensure_columns(df, list(level_cols.values()))

    for level, col in level_cols.items():
        codes, labels = split_code_label(df[col])
        df[f"code{level}"] = codes.str.strip().str.zfill(level)
        df[f"label{level}"] = labels.fillna("").str.strip()

    # Coerce DAIOE metrics to numeric
    for metric in daioe_cols:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df, daioe_cols


def load_employment(
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    Wrapper to fetch and group employment data.
    (MODIFIED to call fetch_all_employment_data from .scb_fetch)
    """
    base_df = fetch_all_employment_data()
    if base_df.empty:
        raise ValueError("SCB Fetch returned an empty DataFrame.")

    emp = base_df.copy()
    emp["code4"] = emp["code_4"].astype(str).str.zfill(4)
    emp["age"] = emp["age"].astype(str).str.strip()
    emp["year"] = pd.to_numeric(emp["year"], errors="coerce").astype("Int64")
    emp["value"] = pd.to_numeric(emp["value"], errors="coerce").fillna(0)
    emp = filter_years(emp, year_min, year_max, year_col="year")

    grouped = (
        emp.groupby(["code4", "age", "year"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "employment"})
    )
    return grouped


# ---------------------------------------------------------------------------
# Aggregation Helpers
# ---------------------------------------------------------------------------


def compute_employment_views(
    daioe_df: pd.DataFrame, employment: pd.DataFrame
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Produce per-level employment tables:
    - age: employment by year/age/code{level}
    - total: employment summed over ages by year/code{level}
    """
    base_cols = [
        "year",
        "code4",
        "label4",
        "code3",
        "label3",
        "code2",
        "label2",
        "code1",
        "label1",
    ]
    base_map = daioe_df[base_cols].drop_duplicates()
    emp_labeled = employment.merge(
        base_map, on=["year", "code4"], how="inner", validate="many_to_many"
    )

    views: Dict[int, Dict[str, pd.DataFrame]] = {}
    for level in (4, 3, 2, 1):
        code_col, label_col = f"code{level}", f"label{level}"
        age_view = emp_labeled.groupby(
            ["year", "age", code_col, label_col], as_index=False
        )["employment"].sum()
        total_view = (
            age_view.groupby(["year", code_col, label_col], as_index=False)[
                "employment"
            ]
            .sum()
            .rename(columns={"employment": "employment_total"})
        )
        views[level] = {"age": age_view, "total": total_view}
    return views


def compute_children_maps(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Count how many descendants each code has in the next-lower level (per year).
    """
    base = df[["year", "code4", "code3", "code2", "code1"]].drop_duplicates()
    counts: Dict[int, pd.DataFrame] = {}
    counts[3] = (
        base.groupby(["year", "code3"])["code4"]
        .nunique()
        .reset_index(name="n_children")
    )
    counts[2] = (
        base.groupby(["year", "code2"])["code3"]
        .nunique()
        .reset_index(name="n_children")
    )
    counts[1] = (
        base.groupby(["year", "code1"])["code2"]
        .nunique()
        .reset_index(name="n_children")
    )
    lvl4 = base.groupby(["year", "code4"]).size().reset_index(name="n_children")
    lvl4["n_children"] = 1
    counts[4] = lvl4
    return counts


def add_percentiles(
    df: pd.DataFrame, metrics: List[str], *, group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute percentile rank within the provided grouping for each DAIOE metric.
    """
    grouping = group_cols or ["level", "year"]
    for metric in metrics:
        df[f"{metric}_pctile"] = df.groupby(grouping)[metric].rank(
            pct=True, ascending=True
        )
    return df


def add_exposure_levels(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Bucket percentile ranks into five exposure levels (1=lowest, 5=highest).
    """
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [1, 2, 3, 4, 5]
    for metric in metrics:
        pct_col = f"{metric}_pctile"
        exposure_col = f"{metric}_exposure_level"
        df[exposure_col] = pd.cut(
            df[pct_col],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        ).astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Aggregation Functions
# ---------------------------------------------------------------------------


def level_four(
    df: pd.DataFrame,
    daioe_cols: List[str],
    n_children: pd.DataFrame,
    emp_totals: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = ["year", "code4", "label4", *daioe_cols]
    base = df[base_cols].copy()
    grouped = (
        base.groupby(["year", "code4", "label4"], as_index=False)
        .mean()
        .merge(emp_totals, on=["year", "code4", "label4"], how="left")
        .merge(n_children, on=["year", "code4"], how="left")
    )
    grouped["level"] = 4
    grouped["taxonomy"] = TAXONOMY
    grouped = grouped.rename(columns={"code4": "code", "label4": "label"})
    ordered = [
        "taxonomy",
        "level",
        "code",
        "label",
        "year",
        "n_children",
        "employment_total",
        *daioe_cols,
    ]
    return grouped[ordered]


def aggregate_level(
    df: pd.DataFrame,
    *,
    daioe_cols: List[str],
    n_children: pd.DataFrame,
    emp_totals: pd.DataFrame,
    level: int,
    method: Literal["weighted", "simple"],
) -> pd.DataFrame:
    if level not in (1, 2, 3):
        raise ValueError("Aggregation from level 4 only supports levels 1–3.")

    code_col, label_col = f"code{level}", f"label{level}"
    group_cols = ["year", code_col, label_col]

    if method == "weighted":
        tmp = df[group_cols + ["employment_total"] + daioe_cols].copy()

        agg_cols = {}
        for metric in daioe_cols:
            mask = tmp[metric].notna() & tmp["employment_total"].notna()
            tmp[f"{metric}_wx"] = tmp[metric].where(mask, 0) * tmp[
                "employment_total"
            ].where(mask, 0)
            tmp[f"{metric}_w"] = tmp["employment_total"].where(mask, 0)
            agg_cols[f"{metric}_wx"] = "sum"
            agg_cols[f"{metric}_w"] = "sum"

        grouped = tmp.groupby(group_cols, as_index=False).agg(agg_cols)

        # --- MISSING LOGIC WAS HERE ---
        for metric in daioe_cols:
            denom = grouped[f"{metric}_w"].replace(0, pd.NA)
            grouped[metric] = grouped[f"{metric}_wx"] / denom
            grouped.drop(columns=[f"{metric}_wx", f"{metric}_w"], inplace=True)
        # --- END OF MISSING LOGIC ---

    else:
        grouped = df[group_cols + daioe_cols].groupby(group_cols, as_index=False).mean()

    grouped = grouped.merge(
        n_children,
        on=["year", code_col],
        how="left",
    ).merge(emp_totals, on=["year", code_col, label_col], how="left")

    grouped["taxonomy"] = TAXONOMY
    grouped["level"] = level
    grouped = grouped.rename(columns={code_col: "code", label_col: "label"})

    ordered = [
        "taxonomy",
        "level",
        "code",
        "label",
        "year",
        "n_children",
        "employment_total",
        *daioe_cols,
    ]
    return grouped[ordered]


def build_pipeline(
    df: pd.DataFrame,
    daioe_cols: List[str],
    children: Dict[int, pd.DataFrame],
    emp_views: Dict[int, Dict[str, pd.DataFrame]],
    method: Literal["weighted", "simple"],
) -> pd.DataFrame:
    base = df.merge(emp_views[4]["total"], on=["year", "code4", "label4"], how="left")

    lvl4 = level_four(base, daioe_cols, children[4], emp_views[4]["total"])
    lvl3 = aggregate_level(
        base,
        daioe_cols=daioe_cols,
        n_children=children[3],
        emp_totals=emp_views[3]["total"],
        level=3,
        method=method,
    )
    lvl2 = aggregate_level(
        base,
        daioe_cols=daioe_cols,
        n_children=children[2],
        emp_totals=emp_views[2]["total"],
        level=2,
        method=method,
    )
    lvl1 = aggregate_level(
        base,
        daioe_cols=daioe_cols,
        n_children=children[1],
        emp_totals=emp_views[1]["total"],
        level=1,
        method=method,
    )

    combined_no_age = pd.concat([lvl1, lvl2, lvl3, lvl4], ignore_index=True)
    combined_no_age = combined_no_age.sort_values(
        ["level", "code", "year"], ignore_index=True
    )
    combined_no_age = add_percentiles(
        combined_no_age, daioe_cols, group_cols=["level", "year"]
    )
    combined_no_age = add_exposure_levels(combined_no_age, daioe_cols)

    expanded_levels = []
    for level, level_df in ((1, lvl1), (2, lvl2), (3, lvl3), (4, lvl4)):
        scored = combined_no_age[combined_no_age["level"] == level]
        age_view = emp_views[level]["age"].rename(
            columns={f"code{level}": "code", f"label{level}": "label"}
        )
        with_age = scored.merge(age_view, on=["year", "code", "label"], how="left")
        expanded_levels.append(with_age)

    combined = pd.concat(expanded_levels, ignore_index=True)
    combined = combined.sort_values(["level", "code", "year", "age"], ignore_index=True)

    ordered = [
        "taxonomy",
        "level",
        "code",
        "label",
        "year",
        "n_children",
        "employment_total",
        "age",
        "employment",
        *daioe_cols,
        *[f"{metric}_pctile" for metric in daioe_cols],
        *[f"{metric}_exposure_level" for metric in daioe_cols],
    ]
    return combined[ordered]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    source: str | Path = DAIOE_SOURCE,
    sep: str = DEFAULT_SEP,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Dict[str, object]:
    """Orchestrates the data loading and aggregation pipeline."""

    # 1. Load raw data (DAIOE and Employment)
    raw = load_daioe_raw(source, sep=sep)
    daioe_df, daioe_cols = prepare_daioe(raw, year_min=None, year_max=None)

    # load_employment calls fetch_all_employment_data from .scb_fetch
    employment = load_employment(year_min=None, year_max=None)

    # 2. Resolve Year Bounds
    resolved_min, resolved_max = resolve_year_bounds(
        daioe_df, employment, year_min, year_max
    )

    # 3. Filter data to resolved bounds
    daioe_df = filter_years(daioe_df, resolved_min, resolved_max, year_col="year")
    employment = filter_years(employment, resolved_min, resolved_max, year_col="year")

    if daioe_df.empty or employment.empty:
        raise ValueError(
            f"No rows remain after filtering to years {resolved_min}–{resolved_max}."
        )

    # 4. Compute aggregation helpers
    children = compute_children_maps(daioe_df)
    emp_views = compute_employment_views(daioe_df, employment)

    # 5. Build Final DataFrames
    weighted = build_pipeline(
        daioe_df,
        daioe_cols=daioe_cols,
        children=children,
        emp_views=emp_views,
        method="weighted",
    )
    simple = build_pipeline(
        daioe_df,
        daioe_cols=daioe_cols,
        children=children,
        emp_views=emp_views,
        method="simple",
    )

    # Package and return payload for app consumption and caching
    return {
        "weighted": weighted,
        "simple": simple,
    }
