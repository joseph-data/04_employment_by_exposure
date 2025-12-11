import os
import tempfile
from pathlib import Path
from typing import Dict
from functools import lru_cache
import pandas as pd
from . import pipeline


def _resolve_cache_dir() -> Path:
    """Finds a writable directory for caching."""
    candidates = []
    if os.getenv("DATA_CACHE_DIR"):
        candidates.append(Path(os.getenv("DATA_CACHE_DIR")))

    # Repo root/data
    candidates.append(Path(__file__).parent.parent / "data")
    # Temp
    candidates.append(Path(tempfile.gettempdir()) / "employment_ai_cache")

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            (path / ".write_test").write_text("ok")
            (path / ".write_test").unlink()
            return path
        except Exception:
            continue
    return Path(tempfile.gettempdir()) / "employment_ai_cache"


DATA_DIR = _resolve_cache_dir()
WEIGHTED_CACHE = DATA_DIR / "daioe_weighted.csv"
SIMPLE_CACHE = DATA_DIR / "daioe_simple.csv"


@lru_cache(maxsize=1)
def _compute_pipeline_payload() -> Dict[str, pd.DataFrame]:
    """Runs the heavy pipeline calculation."""
    return pipeline.run_pipeline()


def load_payload(force_recompute: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load data from disk cache if available, otherwise compute and save.
    """
    if not force_recompute and WEIGHTED_CACHE.exists() and SIMPLE_CACHE.exists():
        print(f"Loading from cache: {DATA_DIR}")
        return {
            "weighted": pd.read_csv(WEIGHTED_CACHE),
            "simple": pd.read_csv(SIMPLE_CACHE),
        }

    if force_recompute:
        _compute_pipeline_payload.cache_clear()

    print("Computing pipeline...")
    payload = _compute_pipeline_payload()

    try:
        payload["weighted"].to_csv(WEIGHTED_CACHE, index=False)
        payload["simple"].to_csv(SIMPLE_CACHE, index=False)
    except Exception as exc:
        print(f"Warning: could not write cache: {exc}")

    return payload
