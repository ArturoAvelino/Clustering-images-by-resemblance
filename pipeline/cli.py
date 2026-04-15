from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from .config import build_config, config_to_yaml, validate_config
from .pipeline import run_pipeline
from .summary import summarize_clusters_csv


def _parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = [p for p in value.replace(" ", ",").split(",") if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("background color must be R,G,B")
    try:
        rgb = tuple(int(float(p)) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("background color must be numeric R,G,B") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("background color values must be between 0 and 255")
    return rgb


def build_parser(*, prog: Optional[str] = None, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        add_help=add_help,
        description="DINOv2 -> UMAP -> HDBSCAN clustering pipeline",
    )
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--print-config", action="store_true", help="Print merged config (YAML) and exit")
    parser.add_argument("--input-dir", help="Folder with input JPG images")
    parser.add_argument("--output-dir", help="Folder to store embeddings and CSV output")
    parser.add_argument("--model-name", help="DINOv2 model name")
    parser.add_argument("--model-repo", help="Local clone path for the DINOv2 repo")
    parser.add_argument("--ssl-ca-bundle", help="Path to a CA bundle PEM file for HTTPS verification")
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dtype", choices=["float16", "float32"])
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--umap-dim", type=int)
    parser.add_argument("--umap-neighbors", type=int)
    parser.add_argument("--umap-min-dist", type=float)
    parser.add_argument("--umap-metric")
    parser.add_argument("--hdbscan-min-cluster-size", type=int)
    parser.add_argument("--hdb-min-cluster-size", dest="hdbscan_min_cluster_size", type=int)
    parser.add_argument("--hdb-min-samples", type=int)
    parser.add_argument("--hdb-metric")
    autocrop = parser.add_mutually_exclusive_group()
    autocrop.add_argument("--autocrop", dest="autocrop", action="store_true", default=None)
    autocrop.add_argument("--no-autocrop", dest="autocrop", action="store_false")
    parser.add_argument(
        "--autocrop-threshold",
        type=int,
        help="Color distance threshold used to separate background from foreground",
    )
    parser.add_argument("--autocrop-padding", type=int)
    parser.add_argument(
        "--background-color",
        type=_parse_rgb,
        help="Background RGB color as R,G,B (default tuned for blue backgrounds)",
    )
    parser.add_argument("--size-feature-weight", type=float)
    two_pass = parser.add_mutually_exclusive_group()
    two_pass.add_argument("--two-pass", dest="two_pass", action="store_true", default=None)
    two_pass.add_argument("--no-two-pass", dest="two_pass", action="store_false")
    fast_tune = parser.add_mutually_exclusive_group()
    fast_tune.add_argument("--fast-tune", dest="fast_tune", action="store_true", default=None)
    fast_tune.add_argument("--no-fast-tune", dest="fast_tune", action="store_false")
    parser.add_argument("--fast-model-name")
    parser.add_argument("--fast-img-size", type=int)
    parser.add_argument("--fast-umap-dim", type=int)
    parser.add_argument("--fast-umap-neighbors", type=int)
    parser.add_argument("--fast-batch-size", type=int)
    parser.add_argument("--fast-num-workers", type=int)
    parser.add_argument("--refine-prob-threshold", type=float)
    refine = parser.add_mutually_exclusive_group()
    refine.add_argument(
        "--refine-include-noise", dest="refine_include_noise", action="store_true", default=None
    )
    refine.add_argument("--no-refine-noise", dest="refine_include_noise", action="store_false")
    write_dim = parser.add_mutually_exclusive_group()
    write_dim.add_argument(
        "--write-dimreduction-vector", dest="write_dimreduction_vector", action="store_true", default=None
    )
    write_dim.add_argument(
        "--no-write-dimreduction-vector",
        dest="write_dimreduction_vector",
        action="store_false",
    )
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument(
        "--summarize-clusters",
        type=Path,
        help="Generate summary_clusters.csv from an existing clusters.csv and exit.",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None, *, prog: Optional[str] = None) -> argparse.Namespace:
    parser = build_parser(prog=prog)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None, *, prog: Optional[str] = None) -> int:
    args = parse_args(argv, prog=prog)
    if args.summarize_clusters is not None:
        summarize_clusters_csv(args.summarize_clusters)
        return 0
    cfg = build_config(args)
    validate_config(cfg)
    if args.print_config:
        print(config_to_yaml(cfg))
        return 0
    run_pipeline(cfg)
    return 0
