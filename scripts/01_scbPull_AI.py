from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
from pyscbwrapper import SCB

Taxonomy = Literal["ssyk2012", "ssyk96"]


try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # pragma: no cover - interactive fallback
    ROOT = Path.cwd().resolve()

DATA_DIR = ROOT / "data"
SCB_DIR = DATA_DIR / "02_scb_data"

TABLES = {
    "ssyk2012": ("en", "AM", "AM0208", "AM0208E", "YREG51BAS"),
    "ssyk96": ("en", "AM", "AM0208", "AM0208E", "YREG33"),
}


def coerce_year(value: str | int | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def latest_year(var_block: dict) -> str:
    years = [coerce_year(year) for year in var_block.get("year", [])]
    valid = [year for year in years if year is not None]
    if not valid:
        raise ValueError("SCB variable metadata did not provide any valid years")
    return str(max(valid))


def fetch_taxonomy_dataframe(taxonomy: Taxonomy) -> tuple[pd.DataFrame, str]:
    if taxonomy not in TABLES:
        raise KeyError(f"Unknown taxonomy '{taxonomy}'")

    scb = SCB(*TABLES[taxonomy])
    var_block = scb.get_variables()
    occupations_key, occupations = next(iter(var_block.items()))
    clean_key = occupations_key.replace(" ", "")

    year = latest_year(var_block)
    scb.set_query(**{clean_key: occupations, "year": [year]})
    scb_fetch = scb.get_data()["data"]

    codes = scb.get_query()["query"][0]["selection"]["values"]
    occ_dict = dict(zip(codes, occupations))

    records = []
    for record in scb_fetch:
        code, obs_year = record["key"][:2]
        if code == "0002":
            continue  # drop unspecified bucket
        value = int(record["values"][0])
        records.append(
            {
                "code_4": str(code).zfill(4),
                "code_3": str(code).zfill(4)[:3],
                "code_2": str(code).zfill(4)[:2],
                "code_1": str(code).zfill(4)[:1],
                "year": obs_year,
                "value": value,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"SCB returned no data for taxonomy '{taxonomy}'")

    level_map = {4: "code_4", 3: "code_3", 2: "code_2", 1: "code_1"}
    frames = []
    for level, column in level_map.items():
        level_df = (
            df.groupby(["year", column], as_index=False)["value"]
            .sum()
            .rename(columns={column: "code"})
        )
        level_df["level"] = level
        frames.append(level_df)

    stacked = (
        pd.concat(frames, ignore_index=True)
        .assign(taxonomy=taxonomy)[["taxonomy", "year", "level", "code", "value"]]
        .sort_values(["year", "level", "code"], ignore_index=True)
    )

    return stacked, year


def write_taxonomy_csv(df: pd.DataFrame, taxonomy: Taxonomy, year: str) -> Path:
    SCB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCB_DIR / f"{taxonomy}_en_{year}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def pull_taxonomy(taxonomy: Taxonomy) -> Path:
    df, year = fetch_taxonomy_dataframe(taxonomy)
    return write_taxonomy_csv(df, taxonomy, year)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull SCB weights for a taxonomy")
    parser.add_argument(
        "--taxonomy",
        default="ssyk2012",
        choices=["ssyk2012", "ssyk96"],
        help="Taxonomy to download (default: ssyk2012)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = pull_taxonomy(args.taxonomy)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
