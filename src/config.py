"""
Configuration constants for SSYK DAIOE data pipeline.
"""

from typing import Dict, List, Literal, Tuple

# ======================================================
#  DATA SOURCES / CONSTANTS
# ======================================================
TAXONOMY: Literal["ssyk2012"] = "ssyk2012"

# Pre-translated DAIOE file;
DAIOE_SOURCE: str = (
    "https://raw.githubusercontent.com/joseph-data/07_translate_ssyk/main/"
    "03_translated_files/daioe_ssyk2012_translated.csv"
)

DEFAULT_SEP: str = ","

# SCB table definitions (order matters: later entries override overlaps in `scb_fetch.py`)
TABLES: Dict[str, Tuple[str, str, str, str, str]] = {
    "14_to_18": ("en", "AM", "AM0208", "AM0208E", "YREG51"),
    "19_to_21": ("en", "AM", "AM0208", "AM0208E", "YREG51N"),
    "20_to_23": ("en", "AM", "AM0208", "AM0208E", "YREG51BAS"),
}

AGE_EXCLUSIONS: List[str] = ["65-69 years"]
EXCLUDED_CODES: List[str] = ["0002", "0000"]

# ======================================================
#  UI DEFAULTS
# ======================================================
LEVEL_OPTIONS: List[Tuple[str, str]] = [
    ("Level 4 (4-digit)", "4"),
    ("Level 3 (3-digit)", "3"),
    ("Level 2 (2-digit)", "2"),
    ("Level 1 (1-digit)", "1"),
]

DEFAULT_LEVEL: str = "3"
DEFAULT_WEIGHTING: str = "weighted"

METRIC_OPTIONS: List[Tuple[str, str]] = [
    ("📚 All Applications", "allapps"),
    ("♟️ Abstract strategy games", "stratgames"),
    ("🎮 Real-time video games", "videogames"),
    ("🖼️🔎 Image recognition", "imgrec"),
    ("🧩🖼️ Image comprehension", "imgcompr"),
    ("🖌️🖼️ Image generation", "imggen"),
    ("📖 Reading comprehension", "readcompr"),
    ("✍️🤖 Language modelling", "lngmod"),
    ("🌐🔤 Translation", "translat"),
    ("🗣️🎙️ Speech recognition", "speechrec"),
    ("🧠✨ Generative AI", "genai"),
]

WEIGHTING_OPTIONS: List[Tuple[str, str]] = [
    ("Employment weighted", "weighted"),
    ("Simple average", "simple"),
]

GLOBAL_YEAR_MIN: int = 2014
GLOBAL_YEAR_MAX: int = 2023
DEFAULT_YEAR_RANGE: Tuple[int, int] = (GLOBAL_YEAR_MIN, GLOBAL_YEAR_MAX)
