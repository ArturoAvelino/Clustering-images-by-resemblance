"""Regenerate richness.csv from an existing clusters.csv file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pipeline.richness import write_richness_csv


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line paths and write the location richness table."""
    parser = argparse.ArgumentParser(
        description="Create richness.csv from the image_id and cluster columns of clusters.csv."
    )
    parser.add_argument("clusters_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    write_richness_csv(args.clusters_csv, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
