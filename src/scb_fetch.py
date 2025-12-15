"""Helpers for fetching employment data from the SCB API.

This module wraps the ``pyscbwrapper`` library to download
occupation/employment tables from Statistics Sweden.  Error handling
and logging are centralised here so that callers of ``fetch_all_employment_data``
can remain agnostic of the details.
"""

from typing import Tuple
import logging

import pandas as pd
from pyscbwrapper import SCB

from .config import AGE_EXCLUSIONS, EXCLUDED_CODES, TABLES

logger = logging.getLogger(__name__)


def fetch_scb_table(
    table_id: str, config: Tuple[str, str, str, str, str]
) -> pd.DataFrame:
    """Fetch and transform a single SCB table.

    Parameters
    ----------
    table_id : str
        A key identifying which table definition in ``TABLES`` to use.
    config : Tuple[str, str, str, str, str]
        The tuple of (language, subject, table, variable_code, filter) used
        by ``pyscbwrapper.SCB`` to form the query.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing one row per (4‑digit occupation code, age,
        year) combination.  Returns an empty frame on error.
    """
    logger.info("Starting SCB fetch for table %s", table_id)
    try:
        scb = SCB(*config)
        var_ = scb.get_variables()

        def get_key_raw(term: str) -> str:
            return next(k for k in var_ if term in k.lower())

        # Identify variable keys from the SCB metadata
        occ_key_raw = get_key_raw("occupation")
        year_key_raw = get_key_raw("year")
        age_key_raw = get_key_raw("age")

        # Filter out excluded ages
        all_ages = var_[age_key_raw]
        filtered_ages = [age for age in all_ages if age not in AGE_EXCLUSIONS]

        # Build the query: remove spaces from the occupation key because SCB
        # uses inconsistent spacing conventions
        query_args = {
            occ_key_raw.replace(" ", ""): var_[occ_key_raw],
            year_key_raw: var_[year_key_raw],
            age_key_raw: filtered_ages,
        }
        scb.set_query(**query_args)

        raw_data = scb.get_data()
        scb_fetch = raw_data.get("data", [])

        # Build a mapping from code to human‑readable occupation name using the
        # query metadata.  We fall back to the code itself if no mapping
        # exists.
        query_meta = scb.get_query().get("query", [])
        occ_meta_vals = next(
            q["selection"]["values"]
            for q in query_meta
            if "occupation" in q["code"].lower() or q["code"] == "Yrke2012"
        )
        occ_dict = dict(zip(occ_meta_vals, var_[occ_key_raw]))

        records = []
        for r in scb_fetch:
            code, age, year = r.get("key", [])[:3]
            records.append(
                {
                    "code_4": code,
                    "occupation": occ_dict.get(code, code),
                    "age": age,
                    "year": year,
                    "value": r.get("values", [None])[0],
                    "source_table": table_id,
                }
            )
        return pd.DataFrame.from_records(records)

    except Exception as exc:
        logger.error("Error processing SCB table %s: %s", table_id, exc)
        return pd.DataFrame()


def fetch_all_employment_data() -> pd.DataFrame:
    """Fetch and consolidate employment data across all configured SCB tables.

    The configured tables in ``TABLES`` may overlap in years.  When
    overlaps occur, later tables in the dictionary take precedence over
    earlier ones.  Rows whose occupation codes are listed in
    ``EXCLUDED_CODES`` are removed.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by (code_4, age, year) with a single
        numeric ``value`` column containing the employment counts.
        Returns an empty frame if no data could be retrieved.
    """
    logger.info("Beginning employment data collection from SCB")
    dfs: list[pd.DataFrame] = []
    for tab_id, config in TABLES.items():
        df_part = fetch_scb_table(tab_id, config)
        if not df_part.empty:
            dfs.append(df_part)
        else:
            logger.warning("No data retrieved for table %s", tab_id)

    # If nothing fetched, return an empty DataFrame
    if not dfs:
        logger.warning("All SCB table fetches returned empty DataFrames")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Resolve overlaps between tables by assigning a priority to each table.
    table_priority = {key: i for i, key in enumerate(TABLES.keys())}
    df["table_priority"] = df["source_table"].map(table_priority)
    df = (
        df.sort_values(["code_4", "age", "year", "table_priority"])
        .drop_duplicates(subset=["code_4", "age", "year"], keep="last")
        .drop(columns=["table_priority"])
    )

    # Exclude specified codes and coerce the value column to numeric
    df = df[~df["code_4"].isin(EXCLUDED_CODES)].reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df