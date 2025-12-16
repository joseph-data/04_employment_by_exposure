"""Core pipeline logic: merge DAIOE scores with SCB employment counts.

This module orchestrates the loading, cleaning and aggregation of two
datasets:

* The DAIOE dataset, which provides AI exposure scores by SSYK2012
  occupational codes and year.
* The employment dataset retrieved from Statistics Sweden (SCB),
  containing counts of employed persons by occupation, age and year.

The primary entry point is :func:`run_pipeline`, which produces two
DataFrames: a weighted aggregation and a simple mean aggregation of
exposure scores at each hierarchy level of SSYK2012.  Additional helper
functions support year filtering, percentile ranking and exposure
level assignment.
"""

from __future__ import annotations

from .config import DAIOE_SOURCE, DEFAULT_SEP, TAXONOMY
from .scb_fetch import fetch_all_employment_data

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import logging
import pandas as pd

# Module‑level logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (Copied from main.py)
# ---------------------------------------------------------------------------


def split_code_label(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Split a SSYK code+label string into separate code and label parts.

    Parameters
    ----------
    series : pd.Series
        A Series of strings of the form ``"<code> <label>"`` or ``None``.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Two Series: codes (zero‑padded to retain leading zeros) and labels.
    """
    parts = series.astype(str).str.split(" ", n=1, expand=True)
    parts = parts.fillna({0: "", 1: ""})
    return parts[0], parts[1]


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Raise an error if the DataFrame lacks any of the required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")


def filter_years(
    df: pd.DataFrame,
    year_min: Optional[int],
    year_max: Optional[int],
    *,
    year_col: str,
) -> pd.DataFrame:
    """Return a DataFrame filtered to the inclusive year range.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing a column with year values.
    year_min : Optional[int]
        Lower bound (inclusive) on the year filter; ``None`` leaves the lower
        bound unbounded.
    year_max : Optional[int]
        Upper bound (inclusive) on the year filter; ``None`` leaves the upper
        bound unbounded.
    year_col : str
        Name of the column in ``df`` holding year values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only rows where ``year_col`` lies
        between ``year_min`` and ``year_max``.  Missing year values are
        excluded.
    """
    if year_min is None and year_max is None:
        return df.copy()
    mask = pd.Series(True, index=df.index, dtype=bool)
    if year_min is not None:
        mask &= df[year_col] >= year_min
    if year_max is not None:
        mask &= df[year_col] <= year_max
    mask = mask.fillna(False)
    return df.loc[mask].copy()


def resolve_year_bounds(
    daioe_df: pd.DataFrame,
    emp_df: pd.DataFrame,
    override_min: Optional[int],
    override_max: Optional[int],
) -> Tuple[int, int]:
    """Compute the intersection of year ranges from two datasets.

    Parameters
    ----------
    daioe_df : pd.DataFrame
        DataFrame containing a ``year`` column for the DAIOE dataset.
    emp_df : pd.DataFrame
        DataFrame containing a ``year`` column for the employment dataset.
    override_min : Optional[int]
        If provided, override the inferred minimum year.  Otherwise the
        maximum of the two dataset minima is used.
    override_max : Optional[int]
        If provided, override the inferred maximum year.  Otherwise the
        minimum of the two dataset maxima is used.

    Returns
    -------
    Tuple[int, int]
        The inclusive year range ``(year_min, year_max)`` to apply to both
        datasets.  Raises ``ValueError`` if no overlap exists.
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
    """Load the pre‑translated DAIOE CSV for SSYK2012.

    Parameters
    ----------
    source : str or Path
        Path or URL to the translated DAIOE CSV.
    sep : str, optional
        Column delimiter; defaults to `","`.

    Returns
    -------
    pd.DataFrame
        The raw DAIOE data as read from the CSV.
    """
    return pd.read_csv(source, sep=sep)


def prepare_daioe(
    raw: pd.DataFrame,
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Clean and enrich the DAIOE DataFrame.

    This function performs several steps:

    * Drop the unhelpful ``Unnamed: 0`` column if present.
    * Ensure the presence of a ``year`` column and convert it to pandas
      nullable integer type.
    * Optionally filter to a year range.
    * Extract separate code and label columns for each level of the
      SSYK2012 hierarchy.
    * Identify and coerce all ``daioe_*`` metric columns to numeric.

    Parameters
    ----------
    raw : pd.DataFrame
        The raw DAIOE data frame.
    year_min, year_max : Optional[int], optional
        Lower and upper bounds for year filtering.  Pass ``None`` to
        disable filtering on that bound.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        A tuple of (cleaned DataFrame, list of DAIOE metric column names).
    """
    df = raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    ensure_columns(df, ["year"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = filter_years(df, year_min, year_max, year_col="year")

    # Identify DAIOE metric columns
    daioe_cols = [col for col in df.columns if col.startswith("daioe_")]
    if not daioe_cols:
        raise KeyError("Expected at least one 'daioe_*' column in DAIOE file.")

    # Map from hierarchy level to column name; ensure all expected hierarchy
    # columns exist.
    level_cols = {
        4: "ssyk2012_4",
        3: "ssyk2012_3",
        2: "ssyk2012_2",
        1: "ssyk2012_1",
    }
    ensure_columns(df, list(level_cols.values()))

    # Parse combined code/label strings into separate columns for each level
    for level, col in level_cols.items():
        codes, labels = split_code_label(df[col])
        df[f"code{level}"] = codes.str.strip().str.zfill(level)
        df[f"label{level}"] = labels.fillna("").str.strip()

    # Coerce all DAIOE metrics to numeric
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

    Parameters
    ----------
    year_min, year_max : Optional[int], optional
        Bounds on the year range to filter after retrieval.

    Returns
    -------
    pd.DataFrame
        Employment counts grouped by 4‑digit occupation code, age and year.
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
    """Build employment views by SSYK level.

    For each SSYK level (4 down to 1), this function produces two views:

    * ``age``: employment counts grouped by year, age and the given code and label.
    * ``total``: employment counts summed across ages grouped by year, code and label.

    Parameters
    ----------
    daioe_df : pd.DataFrame
        The DAIOE DataFrame with hierarchical code/label columns.
    employment : pd.DataFrame
        The employment DataFrame with columns ``year``, ``age``, ``code4`` and
        ``employment``.

    Returns
    -------
    Dict[int, Dict[str, pd.DataFrame]]
        A nested dictionary keyed by level (4–1) and view name (``"age"`` or
        ``"total"``).  ``views[level]["age"]`` contains a DataFrame
        indexed by year/age/code with a column ``employment``.  ``views[level]["total"]``
        contains a DataFrame indexed by year/code with a column
        ``employment_total``.
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
    # Attach SSYK labels (by year+code4) to the raw employment rows; inner join keeps the shared universe.
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
    """Count the number of descendants for each code at each hierarchy level.

    For levels 3, 2 and 1, the number of children is the count of unique
    codes at the next lower level (e.g., the number of 4‑digit codes under
    each 3‑digit code).  For level 4, there are no further subdivisions, so
    ``n_children`` is set to 1 by convention.

    Parameters
    ----------
    df : pd.DataFrame
        DAIOE DataFrame containing code columns ``code4``–``code1`` and
        ``year``.

    Returns
    -------
    Dict[int, pd.DataFrame]
        A dictionary keyed by level (4–1) where each value is a DataFrame
        with columns ``year``, ``code<level>`` and ``n_children``.
    """
    # Compute child counts per year so aggregation reflects the set of codes present that year.
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
    # For level 4 there are no further subdivisions; assign 1 by definition
    lvl4 = base.groupby(["year", "code4"]).size().reset_index(name="n_children")
    lvl4["n_children"] = 1
    counts[4] = lvl4
    return counts


def add_percentiles(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add percentile ranks to the DataFrame for each metric.

    Percentile ranks are computed within the provided grouping.  Lower
    numeric metric values correspond to lower percentile ranks (i.e.,
    higher exposure gets a higher percentile).  Ties use the default
    ``average`` ranking method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing metric columns.
    metrics : List[str]
        Names of the metric columns for which to compute percentile ranks.
    group_cols : Optional[List[str]]
        Columns to group by when computing percentiles.  Defaults to
        ``["level", "year"]``.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional ``<metric>_pctile`` columns.
    """
    grouping = group_cols or ["level", "year"]
    for metric in metrics:
        df[f"{metric}_pctile"] = df.groupby(grouping)[metric].rank(
            pct=True, ascending=True
        )
    return df


def add_exposure_levels(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Assign discrete exposure levels from percentile ranks.

    Percentiles are binned into quintiles: (0–0.2] → 1 (least exposed),
    (0.2–0.4] → 2, …, (0.8–1.0] → 5 (most exposed).  The resulting
    exposure level columns are nullable integers of type ``Int64``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing percentile columns ``<metric>_pctile``.
    metrics : List[str]
        Metric names corresponding to the percentile columns.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional ``<metric>_exposure_level`` columns.
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
    """Aggregate exposure scores at the 4‑digit SSYK level.

    For level 4, a simple arithmetic mean of the raw exposure scores is used
    (per year/code combination).  Employment totals and the number of
    descendants (always one at level 4) are merged onto the result.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing one row per year/code4/label4/metric.
    daioe_cols : List[str]
        Names of the DAIOE metric columns.
    n_children : pd.DataFrame
        DataFrame with columns ``year``, ``code4`` and ``n_children``;
        for level 4 this is always 1.
    emp_totals : pd.DataFrame
        DataFrame with columns ``year``, ``code4``, ``label4`` and
        ``employment_total``.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns ``taxonomy``, ``level``, ``code``,
        ``label``, ``year``, ``n_children``, ``employment_total``, and the
        DAIOE metric columns.
    """
    base_cols = ["year", "code4", "label4", *daioe_cols]
    base = df[base_cols].copy()
    # Simple arithmetic mean for all metrics at the 4‑digit level
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
    """Aggregate exposure metrics from 4‑digit codes to a higher level.

    Parameters
    ----------
    df : pd.DataFrame
        The base DataFrame containing columns ``year``, ``code<level>``,
        ``label<level>``, ``employment_total`` and the DAIOE metric columns.
    daioe_cols : List[str]
        Names of the DAIOE metric columns.
    n_children : pd.DataFrame
        DataFrame with the number of descendants for the given level.
    emp_totals : pd.DataFrame
        Employment totals grouped to the target level.
    level : int
        Target SSYK level (1, 2 or 3).
    method : {"weighted", "simple"}
        Aggregation method.  ``"weighted"`` computes a weighted mean using
        ``employment_total`` as the weight.  ``"simple"`` computes a
        simple arithmetic mean.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with the same schema as ``level_four``, but
        with ``level`` set appropriately.
    """
    if level not in (1, 2, 3):
        raise ValueError("Aggregation from level 4 only supports levels 1–3.")

    code_col, label_col = f"code{level}", f"label{level}"
    group_cols = ["year", code_col, label_col]

    if method == "weighted":
        # Compute weighted means per metric (masking missing metrics/weights without dropping whole rows).
        tmp = df[group_cols + ["employment_total"] + daioe_cols].copy()
        # Dictionary to collect aggregation instructions for groupby
        agg_map: Dict[str, str] = {}
        for metric in daioe_cols:
            # Only weight rows where both metric and employment_total are non‑null
            mask = tmp[metric].notna() & tmp["employment_total"].notna()
            wx_col = f"{metric}_wx"
            w_col = f"{metric}_w"
            tmp[wx_col] = tmp[metric].where(mask, 0) * tmp["employment_total"].where(mask, 0)
            tmp[w_col] = tmp["employment_total"].where(mask, 0)
            agg_map[wx_col] = "sum"
            agg_map[w_col] = "sum"
        grouped = tmp.groupby(group_cols, as_index=False).agg(agg_map)
        # Compute weighted means and drop intermediate columns
        for metric in daioe_cols:
            wx_col = f"{metric}_wx"
            w_col = f"{metric}_w"
            denom = grouped[w_col].replace(0, pd.NA)
            grouped[metric] = grouped[wx_col] / denom
            grouped.drop(columns=[wx_col, w_col], inplace=True)
    else:
        # Simple arithmetic mean of the metrics
        grouped = df[group_cols + daioe_cols].groupby(group_cols, as_index=False).mean()

    # Merge number of descendants and employment totals
    grouped = (
        grouped.merge(n_children, on=["year", code_col], how="left")
        .merge(emp_totals, on=["year", code_col, label_col], how="left")
    )

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
    """Construct the full exposure dataset across all hierarchy levels.

    This function orchestrates the aggregation of 4‑digit codes up to
    levels 3, 2 and 1, attaches employment counts by age, computes
    percentile ranks and exposure levels, and returns a single
    DataFrame suitable for consumption by the front‑end Shiny app.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DAIOE DataFrame with ``code4`` and ``label4`` columns.
    daioe_cols : List[str]
        Names of the DAIOE metric columns.
    children : Dict[int, pd.DataFrame]
        Mapping of level → children count DataFrame (output of
        :func:`compute_children_maps`).
    emp_views : Dict[int, Dict[str, pd.DataFrame]]
        Mapping of level → employment views (output of
        :func:`compute_employment_views`).
    method : {"weighted", "simple"}
        Aggregation method to use when rolling up metrics.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per (level, code, year, age) combination,
        containing employment counts, exposure metrics, percentile ranks and
        exposure levels.
    """
    # Attach 4‑digit employment totals to the base DataFrame
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

    # Combine all levels and compute percentiles/exposure levels on the no-age table
    # so exposure bins are consistent across age groups.
    combined_no_age = pd.concat([lvl1, lvl2, lvl3, lvl4], ignore_index=True)
    combined_no_age = combined_no_age.sort_values(
        ["level", "code", "year"], ignore_index=True
    )
    combined_no_age = add_percentiles(
        combined_no_age, daioe_cols, group_cols=["level", "year"]
    )
    combined_no_age = add_exposure_levels(combined_no_age, daioe_cols)

    # Merge age‑specific employment counts for each level
    expanded_levels: list[pd.DataFrame] = []
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
    """Run the full data pipeline and return weighted and simple results.

    Parameters
    ----------
    source : str or Path, optional
        Location of the translated DAIOE CSV.  Defaults to
        ``config.DAIOE_SOURCE``.
    sep : str, optional
        Column delimiter for the DAIOE CSV.  Defaults to ",".
    year_min, year_max : Optional[int], optional
        Bounds on the year range.  If provided, the intersection of
        ``[year_min, year_max]`` and the available data range is used.

    Returns
    -------
    Dict[str, object]
        A dictionary with two keys, ``"weighted"`` and ``"simple"``,
        containing DataFrames as returned by :func:`build_pipeline` using
        employment‐weighted means and simple means respectively.
    """

    # 1. Load raw inputs (DAIOE and Employment)
    raw = load_daioe_raw(source, sep=sep)
    daioe_df, daioe_cols = prepare_daioe(raw, year_min=None, year_max=None)

    # load_employment calls fetch_all_employment_data from .scb_fetch
    employment = load_employment(year_min=None, year_max=None)

    # 2. Resolve Year Bounds (restrict to the intersection so we never mix unmatched years)
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
