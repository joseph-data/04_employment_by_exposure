from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Works both when run as a script or imported from a notebook with CWD at
# project root. Does not change the working directory.
# ---------------------------------------------------------------------------


def _module_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    # Notebook/interactive fallback: infer src/ location relative to CWD
    cwd = Path.cwd().resolve()
    if (cwd / "digit4_aggregate.py").exists():
        return cwd
    if (cwd / "src" / "digit4_aggregate.py").exists():
        return cwd / "src"
    return cwd


MODULE_DIR = _module_dir()
FETCH_SCRIPT_PATH = MODULE_DIR / "digit4_fetch.py"

try:
    from .config import CODEBOOK_SOURCE as DEFAULT_CODEBOOK_SOURCE
except ImportError:
    from config import CODEBOOK_SOURCE as DEFAULT_CODEBOOK_SOURCE

# Sheet names mapped to their SSYK digit depth in the codebook
SHEET_LEVELS: Dict[str, int] = {
    "4-digit": 4,
    "3-digit": 3,
    "2-digit": 2,
    "1-digit": 1,
}

# ---------------------------------------------------------------------------
# Excel parsing helpers (no external engine needed)
# ---------------------------------------------------------------------------

 

def _resolve_codebook_stream(source: str | Path) -> BytesIO | Path:
    """
    Return a file-like object (for URLs) or Path (for local files) for the codebook.

    The default source points to the GitHub-hosted translation workbook so we no
    longer need a bundled copy in data/.
    """
    source_str = str(source)
    if source_str.lower().startswith(("http://", "https://")):
        response = requests.get(source_str, timeout=30)
        response.raise_for_status()
        return BytesIO(response.content)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Codebook not found at {path}")
    return path


def _load_shared_strings(zf: ZipFile) -> List[str]:
    """Return list of shared strings used in the workbook."""
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        # Workbook with no shared strings
        return []

    ns = {"n": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: List[str] = []
    for si in root.findall("n:si", ns):
        parts = [t.text or "" for t in si.findall(".//n:t", ns)]
        strings.append("".join(parts))
    return strings



def _sheet_paths(zf: ZipFile) -> Dict[str, str]:
    """Map sheet name -> path inside the zip (e.g., xl/worksheets/sheet1.xml)."""
    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rel_root.findall("r:Relationship", rel_ns)
    }

    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    ns = {"n": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    result: Dict[str, str] = {}
    for sheet in wb_root.find("n:sheets", ns):
        name = sheet.attrib["name"]
        rid = sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rel_map[rid]
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        result[name] = target
    return result



def _col_idx(cell_ref: str) -> int:
    """Convert Excel cell reference (A1) into zero-based column index."""
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _read_sheet_rows(
    zf: ZipFile, sheet_path: str, shared_strings: List[str]
) -> List[List[str]]:
    """Return sheet rows as lists, aligning columns by cell reference."""
    root = ET.fromstring(zf.read(sheet_path))
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows: List[List[str]] = []

    for row in root.findall(f".//{ns}row"):
        cells = {}
        max_idx = -1
        for c in row.findall(f"{ns}c"):
            ref = c.attrib.get("r", "")
            col_idx = _col_idx(ref)
            v_elem = c.find(f"{ns}v")
            value = ""
            if v_elem is not None and v_elem.text is not None:
                value = v_elem.text
                # Shared string lookup
                if c.attrib.get("t") == "s":
                    try:
                        value = shared_strings[int(value)]
                    except (IndexError, ValueError):
                        pass
            cells[col_idx] = value
            max_idx = max(max_idx, col_idx)

        if max_idx >= 0:
            row_vals = [""] * (max_idx + 1)
            for idx, val in cells.items():
                row_vals[idx] = val
            rows.append(row_vals)

    return rows


def _rows_to_pairs(rows: List[List[str]], level: int) -> List[Tuple[str, str]]:
    """
    Extract (code, name) pairs from the parsed rows.

    Assumes there is a header row containing "SSYK" and "Name".
    """
    header_idx = None
    for i, row in enumerate(rows):
        if any(str(cell).strip().lower() == "name" for cell in row):
            header_idx = i
            break

    if header_idx is None:
        return []

    headers = rows[header_idx]
    code_col = next(i for i, h in enumerate(headers) if "ssyk" in str(h).lower())
    name_col = next(i for i, h in enumerate(headers) if "name" in str(h).lower())

    pairs: List[Tuple[str, str]] = []
    for row in rows[header_idx + 1 :]:
        if len(row) <= max(code_col, name_col):
            continue
        code = str(row[code_col]).strip()
        name = str(row[name_col]).strip()
        if code and code.lower() != "nan":
            code_norm = code.zfill(level)
            pairs.append((code_norm, name))
    return pairs


def load_codebook(source: str | Path = DEFAULT_CODEBOOK_SOURCE) -> Dict[str, str]:
    """
    Load code -> occupation name across levels from the provided workbook.

    The source can be a local path or an HTTP(S) URL (default: GitHub).

    For each sheet in SHEET_LEVELS, we read all (code, name) pairs and
    map the appropriate code prefix (4/3/2/1 digits) to the occupation name.
    """
    stream = _resolve_codebook_stream(source)
    if hasattr(stream, "seek"):
        stream.seek(0)

    codebook: Dict[str, str] = {}
    with ZipFile(stream) as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_path_map = _sheet_paths(zf)

        for sheet_name, level in SHEET_LEVELS.items():
            sheet_path = sheet_path_map.get(sheet_name)
            if not sheet_path:
                continue
            rows = _read_sheet_rows(zf, sheet_path, shared_strings)
            for code, name in _rows_to_pairs(rows, level):
                # Align each code with the specified digit level (keep leading zeros)
                codebook[str(code).strip()[:level].zfill(level)] = name.strip()
    return codebook


# ---------------------------------------------------------------------------
# Fetch loader
# ---------------------------------------------------------------------------


def load_fetch_module(fetch_path: Path = FETCH_SCRIPT_PATH):
    """
    Load the fetch script (digit4_fetch.py) as a module without requiring
    it to have a valid Python identifier as filename.
    """
    spec = importlib.util.spec_from_file_location("digit4_fetch", fetch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {fetch_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(fetch_path.parent))
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    finally:
        sys.path.pop(0)
    return module


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------


def expand_levels(df: pd.DataFrame, codebook: Dict[str, str]) -> pd.DataFrame:
    """
    Expand 4-digit codes to 3, 2, and 1 digit levels and aggregate values.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame from digit4_fetch.main() with columns:
        - code_4
        - occupation (4-digit name)
        - age
        - year
        - value

    codebook : Dict[str, str]
        Mapping from SSYK code prefixes (1–4 digits) to occupation names.

    Returns
    -------
    DataFrame
        Columns: code, level, occupation, age, year, value
        Where:
        - code: 4/3/2/1 digit SSYK code
        - level: 4, 3, 2, or 1
        - occupation: occupation name for that code level
        - age: age group
        - year: year
        - value: aggregated employed persons
    """
    if df.empty:
        return pd.DataFrame(
            columns=["code", "level", "occupation", "age", "year", "value"]
        )

    df = df.copy()
    df["code_4"] = df["code_4"].astype(str).str.strip().str.zfill(4)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    expanded_rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        base_code = row["code_4"]
        for level in (4, 3, 2, 1):
            code = base_code[:level]
            expanded_rows.append(
                {
                    "code": code,
                    "level": level,
                    "occupation": codebook.get(code, code),
                    "age": row["age"],
                    "year": row["year"],
                    "value": row["value"],
                }
            )

    expanded = pd.DataFrame(expanded_rows)

    # Aggregate: sum value over code/level/occupation/age/year
    aggregated = (
        expanded.groupby(
            ["code", "level", "occupation", "age", "year"], as_index=False
        )["value"]
        .sum()
        .sort_values(["level", "code", "year", "age"])
        .reset_index(drop=True)
    )

    return aggregated


def main(
    codebook_path: str | Path = DEFAULT_CODEBOOK_SOURCE,
    base_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Orchestrator:

    1. If base_df is None: call digit4_fetch.main() to fetch the raw 4-digit data.
    2. Load the SSYK2012 codebook (1–4 digit levels) from the configured source.
    3. Expand and aggregate to levels 4, 3, 2, and 1.
    4. Print a small summary and return the aggregated DataFrame.
    """
    # 1. Get base 4-digit data
    if base_df is None:
        fetch_module = load_fetch_module()  # dynamic import of fetch script
        base_df = (
            fetch_module.main()
        )  # must return DataFrame with code_4/age/year/value

    # 2. Load codebook mapping
    codebook = load_codebook(codebook_path)

    # 3. Expand + aggregate
    aggregated = expand_levels(base_df, codebook)

    print("\n--- AGGREGATION COMPLETE ---")
    print(f"Aggregated rows: {len(aggregated)}")
    unmatched = aggregated.loc[
        aggregated["occupation"] == aggregated["code"], "code"
    ].unique()
    if len(unmatched):
        print(f"Unmapped codes (occupation fell back to code): {list(unmatched)[:10]}")
        if len(unmatched) > 10:
            print(f"... and {len(unmatched) - 10} more")
    else:
        print("All codes mapped to occupation names.")
    print("Sample:")
    print(aggregated.head(10))

    return aggregated


if __name__ == "__main__":
    main()
