import pandas as pd
from pyscbwrapper import SCB
from pathlib import Path


# Optional: project root if you need it elsewhere
ROOT = Path.cwd().resolve().parents[1]

TAX_ID = "ssyk2012"

TABLES = {
    "ssyk2012_tab": ("en", "AM", "AM0208", "AM0208B", "YREG61BAS"),
    # "ssyk96_tab": ("en", "AM", "AM0208", "AM0208E", "YREG33"),
}


def fetch_scb_aku_occupations(tax_id: str = TAX_ID) -> pd.DataFrame:
    """
    Fetch SCB AKU employment by occupation (SSYK 2012), age and year,
    and return a cleaned DataFrame at the SSYK3 level (string codes).

    Columns:
      - code_3      (SSYK code as returned by SCB; can be 2–4 digits)
      - occupation  (text label from SCB)
      - age
      - year
      - value       (string as provided by SCB)
    """

    # ---- 1) Init SCB table ----
    scb = SCB(*TABLES[f"{tax_id}_tab"])
    var_ = scb.get_variables()

    # First variable is the occupation variable (as in your original code)
    occupations_key, occupations = next(iter(var_.items()))
    clean_key = occupations_key.replace(" ", "")

    # ---- 2) Years: coerce to int, use all valid years ----
    def coerce_year(y):
        try:
            return int(y)
        except Exception:
            return None

    years = [coerce_year(y) for y in var_["year"]]
    years = [y for y in years if y is not None]
    if not years:
        raise ValueError("No valid years found in SCB variables")

    years_sorted = sorted(set(years))
    year_values = [str(y) for y in years_sorted]

    # ---- 3) All ages as provided by SCB ----
    age_values = var_["age"]

    # ---- 4) Build and send query ----
    scb.set_query(
        **{
            clean_key: occupations,
            "year": year_values,  # all years
            "age": age_values,  # all ages
        }
    )

    scb_data = scb.get_data()
    scb_fetch = scb_data["data"]

    # Map occupation codes to their labels
    codes = scb.get_query()["query"][0]["selection"]["values"]
    occ_dict = dict(zip(codes, occupations))

    # ---- 5) Build DataFrame ----
    records = []
    for r in scb_fetch:
        # The order follows the SCB query; your original code assumed:
        # occupation code, age, year
        code, age, year = r["key"]
        name = occ_dict.get(code, code)
        value = r["values"][0]  # raw string
        records.append(
            {
                "code_3": code,
                "occupation": name,
                "age": age,
                "year": year,
                "value": value,
            }
        )

    df = pd.DataFrame(records)

    # Remove unidentified group 002 (as in your original code)
    df = df[df["code_3"] != "002"].reset_index(drop=True)

    return df


def main() -> pd.DataFrame:
    """Entry point when run as a script; returns the DataFrame."""
    df = fetch_scb_aku_occupations()
    # Optional: quick check
    print(df.head())
    print(f"\nRows: {len(df)}, columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    main()
