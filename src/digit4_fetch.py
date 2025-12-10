from typing import Tuple, Dict, Any, List
import pandas as pd
from pyscbwrapper import SCB

try:
    from .config import AGE_EXCLUSIONS, EXCLUDED_CODES, TABLES
except ImportError:
    from config import AGE_EXCLUSIONS, EXCLUDED_CODES, TABLES


def fetch_scb_table(
    table_id: str, config: Tuple[str, str, str, str, str]
) -> pd.DataFrame:
    """
    1. Initializes the SCB connection.
    2. Dynamically finds variable keys (Occupation, Year, Age).
    3. Filters the Age group (excluding '65-69 years') and sets the API query.
    4. Fetches the data, maps occupation codes to names, and returns a clean DataFrame.
    """
    print(f"\n--- Starting Fetch: {table_id} ---")

    try:
        # Initialize the SCB wrapper for the current table configuration
        scb = SCB(*config)

        # ---------------------------------------------
        # 1. VARIABLE DISCOVERY AND AGE FILTERING
        # ---------------------------------------------
        # Fetch the current table's variable metadata
        var_: Dict[str, List[str]] = scb.get_variables()

        # Helper function to find the raw variable name (e.g., 'Occupation (SSYK 2012)')
        def get_key_raw(term: str) -> str:
            return next(k for k in var_ if term in k.lower())

        # Identify the full, raw variable keys
        occ_key_raw = get_key_raw("occupation")
        year_key_raw = get_key_raw("year")
        age_key_raw = get_key_raw("age")

        # Prepare Age Filter: Exclude the specified group before querying the API
        all_ages = var_[age_key_raw]
        filtered_ages = [age for age in all_ages if age not in AGE_EXCLUSIONS]

        # ---------------------------------------------
        # 2. QUERY PREPARATION AND EXECUTION
        # ---------------------------------------------
        # Prepare the query dictionary for scb.set_query().
        # The occupation key is cleaned to handle spaces, which is necessary for some SCB endpoints.
        query_args: Dict[str, Any] = {
            occ_key_raw.replace(" ", ""): var_[
                occ_key_raw
            ],  # Cleaned key used; fetch all occupation values
            year_key_raw: var_[year_key_raw],  # Fetch all years in this range
            age_key_raw: filtered_ages,  # Fetch only the pre-filtered age groups
        }

        scb.set_query(**query_args)

        # Fetch the data from the SCB API
        raw_data = scb.get_data()
        scb_fetch = raw_data["data"]

        # Print the row count directly for internal tracking
        row_count = len(scb_fetch)
        print(f"-> Successfully fetched **{row_count}** raw rows for {table_id}.")

        # ---------------------------------------------
        # 3. CODE-TO-NAME MAPPING
        # ---------------------------------------------
        # Retrieve the query metadata to find the official list of codes (e.g., "0010")
        query_meta = scb.get_query()["query"]

        # Locate the occupation codes list (values) within the query metadata
        occ_meta_vals = next(
            q["selection"]["values"]
            for q in query_meta
            if q["code"] == "Yrke2012" or "occupation" in q["code"].lower()
        )

        # Create the mapping dictionary: Code (e.g., "0010") -> Text Name (e.g., "Army officers")
        # Uses the codes from metadata and the corresponding names from the variable list (var_)
        occ_dict = dict(zip(occ_meta_vals, var_[occ_key_raw]))

        # ---------------------------------------------
        # 4. PARSING AND DATAFRAME CONSTRUCTION
        # ---------------------------------------------
        records = []
        for r in scb_fetch:
            # SCB results return keys in the order requested: [Code, Age, Year] (for this query structure)
            code, age, year = r["key"][:3]

            records.append(
                {
                    "code_4": code,
                    "occupation": occ_dict.get(
                        code, code
                    ),  # Map code to human-readable name
                    "age": age,
                    "year": year,
                    "value": r["values"][
                        0
                    ],  # The actual data point value (as a string)
                    "source_table": table_id,  # Reference the original table ID
                }
            )

        return pd.DataFrame(records)

    except Exception as e:
        print(f"🚨 Error processing table {table_id}: {e}")
        return pd.DataFrame()  # Return empty DF on failure


# ==============================================================================
#                      MAIN EXECUTION BLOCK
# ==============================================================================

def main() -> pd.DataFrame:
    # 1. Loop through tables and collect DataFrames
    print("\n--- BEGINNING DATA COLLECTION ---")
    dfs: List[pd.DataFrame] = []
    for tab_id, config in TABLES.items():
        df_part = fetch_scb_table(tab_id, config)
        if not df_part.empty:
            dfs.append(df_part)

    # 2. Combine and Finalize the dataset
    if dfs:
        # Concatenate all fetched DataFrames into one master table
        df = pd.concat(dfs, ignore_index=True)

        # Resolve overlaps: if multiple source tables cover the same code/age/year,
        # keep the value from the latest table (based on TABLES definition order).
        table_priority = {key: i for i, key in enumerate(TABLES.keys())}
        df["table_priority"] = df["source_table"].map(table_priority)
        df = (
            df.sort_values(["code_4", "age", "year", "table_priority"])
            .drop_duplicates(subset=["code_4", "age", "year"], keep="last")
            .drop(columns=["table_priority"])
        )

        # ---------------------------------------------
        # 3. FINAL DATA CLEANING AND CASTING
        # ---------------------------------------------

        # Filter: Remove unidentified groups (usually "0000" and "0002")
        df = df[~df["code_4"].isin(EXCLUDED_CODES)].reset_index(drop=True)

        # Type Casting: Convert the 'value' column to numeric, setting SCB's ".." to NaN
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # ---------------------------------------------
        # 4. SUMMARY OUTPUT
        # ---------------------------------------------
        print(f"\n--- FINAL DATASET SUMMARY ---")
        print(f"Total final rows after filtering (removing 0000/0002): **{len(df)}**")
        print("\nDataFrame Head:")
        print(df.head(10))
        print(
            "\nAge groups included in the final dataset (Should not include '65-69 years'):"
        )
        print(df["age"].unique())
        return df

    print("\n--- FINAL DATASET SUMMARY ---")
    print("No data fetched successfully.")
    return pd.DataFrame()


if __name__ == "__main__":
    main()
