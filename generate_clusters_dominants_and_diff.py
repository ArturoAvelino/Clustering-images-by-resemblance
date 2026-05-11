from __future__ import annotations

"""
Generate clusters_dominant_classes_and_diff.csv from clusters_summary_classes.csv.

Usage:
  python generate_clusters_dominants_and_diff.py /path/to/clusters_summary_classes.csv
  python generate_clusters_dominants_and_diff.py /path/to/clusters_summary_classes.csv \
    --output /path/to/clusters_dominant_classes_and_diff.csv
"""

import argparse
from pathlib import Path
from typing import List, Optional

from pipeline.summary import summarize_cluster_dominant_classes_and_diff_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read clusters_summary_classes.csv and write "
            "clusters_dominant_classes_and_diff.csv with the dominant class, "
            "second-dominant class, object counts, and percentage difference "
            "for each cluster."
        )
    )
    parser.add_argument(
        "summary_classes_csv",
        type=Path,
        help="Path to clusters_summary_classes.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination CSV path. Defaults to "
            "clusters_dominant_classes_and_diff.csv next to "
            "clusters_summary_classes.csv."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = summarize_cluster_dominant_classes_and_diff_csv(
        args.summary_classes_csv,
        args.output,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
