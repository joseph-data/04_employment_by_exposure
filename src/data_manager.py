"""Data manager for loading and caching pipeline results.

This module encapsulates the logic for computing the heavy data
transformations in ``pipeline.py`` and persisting the results to disk.
It adds a small amount of resilience around caching and uses
``logging`` instead of printing directly to stdout.  The cache files
include a version tag to make it easy to invalidate caches when
fundamental changes are made to the pipeline logic.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Dict
from functools import lru_cache

import pandas as pd

from . import pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache setup
# ---------------------------------------------------------------------------
# A version tag to embed into the cache filenames.  Bump this value
# whenever the underlying ``pipeline`` logic changes in a way that
# invalidates existing caches.
CACHE_VERSION: str = "v1"


def _resolve_cache_dir() -> Path:
    """Select a writable directory for caching.

    The lookup order is:

    1. The ``DATA_CACHE_DIR`` environment variable, if set.
    2. A ``data`` folder at the repository root.
    3. A temporary directory in ``/tmp``.

    Each candidate path is tested for writability by attempting to
    create and delete a sentinel file.  The first path that succeeds
    is returned.  If none succeed, a final fallback directory in ``/tmp``
    is created and returned.
    """
    candidates: list[Path] = []
    env = os.getenv("DATA_CACHE_DIR")
    if env:
        # Expand relative or user paths to absolute
        candidates.append(Path(env).expanduser().resolve())

    # Repo root /data (two levels up from this file)
    candidates.append(Path(__file__).resolve().parent.parent / "data")
    # Temp fallback
    candidates.append(Path(tempfile.gettempdir()) / "employment_ai_cache")

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            return path
        except Exception:
            continue

    # Final fallback: ensure the last candidate exists
    fallback = Path(tempfile.gettempdir()) / "employment_ai_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# Resolve the directory once at import time
DATA_DIR: Path = _resolve_cache_dir()

# Build cache file paths with version tags.  This allows caches from
# different versions of the pipeline to coexist without overwriting
# each other.  For example, ``daioe_weighted_v1.csv``.
WEIGHTED_CACHE: Path = DATA_DIR / f"daioe_weighted_{CACHE_VERSION}.csv"
SIMPLE_CACHE: Path = DATA_DIR / f"daioe_simple_{CACHE_VERSION}.csv"


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV atomically.

    The CSV is first written to a temporary file in the same directory
    and then renamed to the final location.  This avoids leaving a
    partially written file if the process is interrupted mid‑write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


@lru_cache(maxsize=1)
def _compute_pipeline_payload() -> Dict[str, pd.DataFrame]:
    """Runs the heavy pipeline calculation."""
    return pipeline.run_pipeline()


def load_payload(force_recompute: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load data from disk cache if available, otherwise compute and save.

    Parameters
    ----------
    force_recompute : bool, optional
        If ``True``, recompute the pipeline even if cache files exist.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with keys ``"weighted"`` and ``"simple"``
        containing the respective DataFrames.
    """
    # If a cached payload exists and recomputation is not forced, return it
    if not force_recompute and WEIGHTED_CACHE.exists() and SIMPLE_CACHE.exists():
        logger.info("Loading pipeline output from cache directory %s", DATA_DIR)
        try:
            weighted_df = pd.read_csv(WEIGHTED_CACHE)
            simple_df = pd.read_csv(SIMPLE_CACHE)
            return {"weighted": weighted_df, "simple": simple_df}
        except Exception as exc:
            # If reading the cache fails, fall back to recomputing
            logger.warning(
                "Error reading cache files %s and %s: %s; falling back to recompute",
                WEIGHTED_CACHE,
                SIMPLE_CACHE,
                exc,
            )

    if force_recompute:
        # Clear the LRU cache before recomputing
        _compute_pipeline_payload.cache_clear()

    logger.info("Computing pipeline data – this may take a while…")
    payload = _compute_pipeline_payload()

    # Persist to disk atomically
    try:
        _atomic_to_csv(payload["weighted"], WEIGHTED_CACHE)
        _atomic_to_csv(payload["simple"], SIMPLE_CACHE)
        logger.info(
            "Cache updated: weighted=%s, simple=%s", WEIGHTED_CACHE.name, SIMPLE_CACHE.name
        )
    except Exception as exc:
        logger.warning("Could not write cache files: %s", exc)

    return payload