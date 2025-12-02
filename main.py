from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def load_module(name: str, filename: str):
    """Import a script with a numeric prefix via importlib."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not load module '{name}' from {filename}")
    spec.loader.exec_module(module)
    return module


SCB_PULL = load_module("scb_pull", "01_scbPull.py")
WEIGHTING = load_module("weighting", "02_weighting.py")


def run_pipeline(taxonomies: Iterable[WEIGHTING.Taxonomy]):
    """Run SCB pull + weighting for each taxonomy and collect output paths."""
    summary = []
    for taxonomy in taxonomies:
        scb_path = SCB_PULL.pull_taxonomy(taxonomy)
        weighted_path, simple_path = WEIGHTING.run_weighting(taxonomy)
        summary.append(
            {
                "taxonomy": taxonomy,
                "scb": scb_path,
                "weighted": weighted_path,
                "simple": simple_path,
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull SCB data and build employment-weighted DAIOE aggregates",
    )
    parser.add_argument(
        "--taxonomy",
        action="append",
        choices=["ssyk2012", "ssyk96"],
        help="Taxonomy to refresh (can be provided multiple times). Defaults to both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomies = args.taxonomy or ["ssyk2012", "ssyk96"]
    results = run_pipeline(taxonomies)

    print("\nDAIOE datasets refreshed:\n" + "-" * 40)
    for item in results:
        print(f"Taxonomy: {item['taxonomy']}")
        print(f"  SCB weights:         {item['scb']}")
        print(f"  Employment-weighted: {item['weighted']}")
        print(f"  Simple-average:      {item['simple']}\n")
    print("Outputs are ready under data/03_daioe_aggregated for app.py")


if __name__ == "__main__":
    main()
