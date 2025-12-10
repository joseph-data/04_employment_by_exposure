"""
Configuration constants for SSYK DAIOE data pipeline.

This module defines:
- DAIOE_SOURCE: URL to translated DAIOE scores (GitHub repository)
- CODEBOOK_SOURCE: URL for the translated SSYK2012 codebook (Excel)
- DEFAULT_SEP: CSV delimiter for DAIOE files
- TABLES: SCB API table definitions for different year ranges
  * 14_to_18: Years 2014-2018 (YREG51)
  * 19_to_21: Years 2019-2021 (YREG51N)  
  * 20_to_23: Years 2020-2023 (YREG51BAS)
- AGE_EXCLUSIONS: Age groups to filter out during data fetch
- EXCLUDED_CODES: Occupation codes to exclude (unclassified groups)

Each SCB table configuration is a tuple: (language, subject, area, table, content)
"""

from typing import Dict, List, Tuple

# Default DAIOE source (translated SSYK2012 scores)
DAIOE_SOURCE: str = (
    "https://raw.githubusercontent.com/joseph-data/07_translate_ssyk/main/"
    "03_translated_files/daioe_ssyk2012_translated.csv"
)

# Source for the translated SSYK2012 Excel codebook (remote to avoid bundling local copy)
CODEBOOK_SOURCE: str = (
    "https://raw.githubusercontent.com/joseph-data/07_translate_ssyk/main/"
    "02_translation_files/ssyk2012_en.xlsx"
)

# CSV delimiter for DAIOE source files
DEFAULT_SEP: str = ","

# SCB table definitions: each tuple is (language, subject, area, table, content)
TABLES: Dict[str, Tuple[str, str, str, str, str]] = {
    "14_to_18": ("en", "AM", "AM0208", "AM0208E", "YREG51"),
    "19_to_21": ("en", "AM", "AM0208", "AM0208E", "YREG51N"),
    "20_to_23": ("en", "AM", "AM0208", "AM0208E", "YREG51BAS"),
}

# Age groups to drop when querying the SCB API
AGE_EXCLUSIONS: List[str] = ["65-69 years"]

# Occupation codes to drop after the fetch
EXCLUDED_CODES: List[str] = ["0002", "0000"]
