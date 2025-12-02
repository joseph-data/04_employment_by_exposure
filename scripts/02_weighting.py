from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

Taxonomy = Literal["ssyk2012", "ssyk96"]

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # pragma: no cover - interactive fallback
    ROOT = Path.cwd()

DATA_DIR = ROOT / "data"


def data_path(*parts: str | Path) -> Path:
    return DATA_DIR.joinpath(*parts)


def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")
    return files[-1]


def load_daioe_raw(taxonomy: Taxonomy, sep: str = "\t") -> pd.DataFrame:
    return pd.read_csv(data_path("01_daioe_raw", f"daioe_{taxonomy}.csv"), sep=sep)


def load_scb_employment(taxonomy: Taxonomy) -> pd.DataFrame:
    scb_path = latest_file(data_path("02_scb_data"), f"{taxonomy}*.csv")
    return pd.read_csv(scb_path).drop(columns=["year"], errors="ignore")


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")


def split_code_label(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.astype(str).str.split(" ", n=1, expand=True)
    parts = parts.fillna({0: "", 1: ""})
    return parts[0], parts[1]


def prepare_raw_dataframe(raw: pd.DataFrame, taxonomy: Taxonomy) -> tuple[pd.DataFrame, list[str]]:
    df = raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    ensure_columns(df, ["year"])

    daioe_cols = [col for col in df.columns if col.startswith("daioe_")]
    if not daioe_cols:
        raise KeyError("Expected at least one 'daioe_*' column in DAIOE raw file.")

    code_cols = {
        4: f"{taxonomy}_4",
        3: f"{taxonomy}_3",
        2: f"{taxonomy}_2",
        1: f"{taxonomy}_1",
    }
    ensure_columns(df, list(code_cols.values()))

    for level, col in code_cols.items():
        codes, labels = split_code_label(df[col])
        df[f"code{level}"] = codes
        df[f"label{level}"] = labels

    df["code4"] = df["code4"].str.zfill(4)
    for level in (1, 2, 3):
        df[f"code{level}"] = df[f"code{level}"].str.lstrip("0")

    return df, daioe_cols


def attach_employment(df: pd.DataFrame, scb: pd.DataFrame) -> pd.DataFrame:
    scb_lvl4 = scb[scb["level"] == 4].copy()
    if scb_lvl4.empty:
        raise ValueError("SCB data must contain level-4 rows for weighting.")

    scb_lvl4["code4"] = scb_lvl4["code"].astype(str).str.zfill(4)
    merged = df.merge(
        scb_lvl4[["code4", "value"]],
        on="code4",
        how="left",
        validate="many_to_one",
    )
    return merged.rename(columns={"value": "emp"})


def compute_children_maps(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    counts = {
        1: df.groupby(["year", "code1"])["code2"].nunique().reset_index(name="n_children"),
        2: df.groupby(["year", "code2"])["code3"].nunique().reset_index(name="n_children"),
        3: df.groupby(["year", "code3"])["code4"].nunique().reset_index(name="n_children"),
    }
    lvl4 = df.groupby(["year", "code4"]).size().reset_index(name="n_children")
    lvl4["n_children"] = 1
    counts[4] = lvl4
    return counts


def aggregate_level(
    df: pd.DataFrame,
    *,
    daioe_cols: list[str],
    n_children: dict[int, pd.DataFrame],
    taxonomy: Taxonomy,
    level: int,
    method: Literal["weighted", "simple"],
) -> pd.DataFrame:
    if level not in (1, 2, 3):
        raise ValueError("Only levels 1–3 can be aggregated from level 4.")

    code_col, label_col = f"code{level}", f"label{level}"
    group_cols = ["year", code_col, label_col]

    if method == "weighted":
        tmp = df[group_cols + ["emp"] + daioe_cols].copy()
        for metric in daioe_cols:
            mask = tmp[metric].notna()
            tmp[f"{metric}_wx"] = tmp[metric].where(mask, 0) * tmp["emp"].where(mask, 0)
            tmp[f"{metric}_w"] = tmp["emp"].where(mask, 0)
        agg_cols = {f"{metric}_wx": "sum" for metric in daioe_cols}
        agg_cols.update({f"{metric}_w": "sum" for metric in daioe_cols})
        grouped = tmp.groupby(group_cols, as_index=False).agg(agg_cols)
        for metric in daioe_cols:
            denom = grouped[f"{metric}_w"].replace(0, pd.NA)
            grouped[metric] = grouped[f"{metric}_wx"] / denom
            grouped.drop(columns=[f"{metric}_wx", f"{metric}_w"], inplace=True)
    else:
        grouped = df[group_cols + daioe_cols].groupby(group_cols, as_index=False).mean()

    grouped = grouped.merge(
        n_children[level],
        left_on=["year", code_col],
        right_on=["year", code_col],
        how="left",
    )

    out = grouped[["year", code_col, label_col, "n_children"] + daioe_cols].copy()
    out["taxonomy"] = taxonomy
    out["level"] = level
    out = out.rename(columns={code_col: "code", label_col: "label"})
    out["code"] = out["code"].astype(str)
    return out


def base_level_four(df: pd.DataFrame, daioe_cols: list[str], taxonomy: Taxonomy, n_children: pd.DataFrame) -> pd.DataFrame:
    base = df[["year", "code4", "label4"] + daioe_cols].copy()
    base = base.merge(n_children, on=["year", "code4"], how="left")
    base["taxonomy"] = taxonomy
    base["level"] = 4
    base = base.rename(columns={"code4": "code", "label4": "label"})
    base["code"] = base["code"].astype(str)
    return base


def add_percentiles(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    pct_cols: list[str] = []
    for metric in metrics:
        suffix = metric.removeprefix("daioe_")
        rank_col = f"pct_rank_{suffix}"
        df[rank_col] = df.groupby(["year", "level"])[metric].rank(pct=True)
        pct_cols.append(rank_col)
    return pct_cols


def build_pipeline(
    df: pd.DataFrame,
    *,
    daioe_cols: list[str],
    taxonomy: Taxonomy,
    n_children: dict[int, pd.DataFrame],
    method: Literal["weighted", "simple"],
) -> pd.DataFrame:
    lvl4 = base_level_four(df, daioe_cols, taxonomy, n_children[4])
    lvl1 = aggregate_level(df, daioe_cols=daioe_cols, n_children=n_children, taxonomy=taxonomy, level=1, method=method)
    lvl2 = aggregate_level(df, daioe_cols=daioe_cols, n_children=n_children, taxonomy=taxonomy, level=2, method=method)
    lvl3 = aggregate_level(df, daioe_cols=daioe_cols, n_children=n_children, taxonomy=taxonomy, level=3, method=method)

    combined = pd.concat([lvl1, lvl2, lvl3, lvl4], ignore_index=True)
    pct_cols = add_percentiles(combined, daioe_cols)
    ordered = [
        "taxonomy",
        "level",
        "code",
        "label",
        "year",
        "n_children",
        *daioe_cols,
        *pct_cols,
    ]
    return combined[ordered].sort_values(["level", "code", "year"], ignore_index=True)


def write_outputs(taxonomy: Taxonomy, weighted: pd.DataFrame, simple: pd.DataFrame) -> tuple[Path, Path]:
    out_dir = data_path("03_daioe_aggregated")
    out_dir.mkdir(parents=True, exist_ok=True)
    weighted_path = out_dir / f"daioe_{taxonomy}_emp_weighted.csv"
    simple_path = out_dir / f"daioe_{taxonomy}_simple_avg.csv"
    weighted.to_csv(weighted_path, index=False)
    simple.to_csv(simple_path, index=False)
    return weighted_path, simple_path


def run_weighting(taxonomy: Taxonomy, sep: str = "\t") -> tuple[Path, Path]:
    raw = load_daioe_raw(taxonomy, sep=sep)
    scb = load_scb_employment(taxonomy)
    prepared, daioe_cols = prepare_raw_dataframe(raw, taxonomy)
    prepared = attach_employment(prepared, scb)
    n_children = compute_children_maps(prepared)

    weighted = build_pipeline(
        prepared,
        daioe_cols=daioe_cols,
        taxonomy=taxonomy,
        n_children=n_children,
        method="weighted",
    )
    simple = build_pipeline(
        prepared,
        daioe_cols=daioe_cols,
        taxonomy=taxonomy,
        n_children=n_children,
        method="simple",
    )
    return write_outputs(taxonomy, weighted, simple)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAIOE weighting pipeline")
    parser.add_argument(
        "--taxonomy",
        default="ssyk2012",
        choices=["ssyk2012", "ssyk96"],
        help="Taxonomy to process (default: ssyk2012)",
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Delimiter used in DAIOE raw files (default: tab)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weighted_path, simple_path = run_weighting(args.taxonomy, sep=args.sep)
    print("Written employment-weighted file:", weighted_path)
    print("Written simple-average file:    ", simple_path)


if __name__ == "__main__":
    main()
