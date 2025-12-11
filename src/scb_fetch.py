"""
Handles interactions with the SCB (Statistics Sweden) API.
"""

from typing import Tuple
import pandas as pd
from pyscbwrapper import SCB
from .config import AGE_EXCLUSIONS, EXCLUDED_CODES, TABLES


def fetch_scb_table(
    table_id: str, config: Tuple[str, str, str, str, str]
) -> pd.DataFrame:
    """Fetches a specific table configuration from SCB."""
    print(f"\n--- Starting Fetch: {table_id} ---")
    try:
        scb = SCB(*config)
        var_ = scb.get_variables()

        def get_key_raw(term: str) -> str:
            return next(k for k in var_ if term in k.lower())

        occ_key_raw = get_key_raw("occupation")
        year_key_raw = get_key_raw("year")
        age_key_raw = get_key_raw("age")

        all_ages = var_[age_key_raw]
        filtered_ages = [age for age in all_ages if age not in AGE_EXCLUSIONS]

        query_args = {
            occ_key_raw.replace(" ", ""): var_[occ_key_raw],
            year_key_raw: var_[year_key_raw],
            age_key_raw: filtered_ages,
        }
        scb.set_query(**query_args)

        raw_data = scb.get_data()
        scb_fetch = raw_data["data"]

        # Mapping Logic
        query_meta = scb.get_query()["query"]
        occ_meta_vals = next(
            q["selection"]["values"]
            for q in query_meta
            if "occupation" in q["code"].lower() or q["code"] == "Yrke2012"
        )
        occ_dict = dict(zip(occ_meta_vals, var_[occ_key_raw]))

        records = []
        for r in scb_fetch:
            code, age, year = r["key"][:3]
            records.append(
                {
                    "code_4": code,
                    "occupation": occ_dict.get(code, code),
                    "age": age,
                    "year": year,
                    "value": r["values"][0],
                    "source_table": table_id,
                }
            )
        return pd.DataFrame(records)

    except Exception as e:
        print(f"🚨 Error processing table {table_id}: {e}")
        return pd.DataFrame()


def fetch_all_employment_data() -> pd.DataFrame:
    """Main entry point to fetch and combine all configured SCB tables."""
    print("\n--- BEGINNING DATA COLLECTION ---")
    dfs = []
    for tab_id, config in TABLES.items():
        df_part = fetch_scb_table(tab_id, config)
        if not df_part.empty:
            dfs.append(df_part)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Priority handling for overlapping years/tables
    table_priority = {key: i for i, key in enumerate(TABLES.keys())}
    df["table_priority"] = df["source_table"].map(table_priority)
    df = (
        df.sort_values(["code_4", "age", "year", "table_priority"])
        .drop_duplicates(subset=["code_4", "age", "year"], keep="last")
        .drop(columns=["table_priority"])
    )

    df = df[~df["code_4"].isin(EXCLUDED_CODES)].reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df
