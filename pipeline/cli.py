from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from .config import build_config, config_to_yaml, validate_config
from .pipeline import run_pipeline
from .summary import (
    summarize_classes_in_clusters_csv,
    summarize_cluster_dominant_classes_and_diff_csv,
    summarize_clustering_score_report_csv,
    summarize_clusters_csv,
)


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
    parser.add_argument(
        "--compute",
        choices=["full", "only-dimreduction-and-clustering", "only-clustering"],
        help="Which stages to run (default: full pipeline).",
    )
    parser.add_argument(
        "--dino-files",
        help=(
            "Directory containing embeddings.dat, embeddings.json, sizes.npy, and images.txt "
            "from a previous run (required for only-dimreduction-and-clustering)."
        ),
    )
    parser.add_argument(
        "--umap-files",
        help=(
            "Directory containing umap.npy and images.txt from a previous run "
            "(required for only-clustering)."
        ),
    )
    parser.add_argument(
        "--subset-images",
        help="Text file of exact images.txt entries to select from dino_files; UMAP + HDBSCAN only.",
    )
    parser.add_argument("--input-dir", help="Folder with input JPG images")
    parser.add_argument("--output-dir", help="Folder to store embeddings and CSV output")
    parser.add_argument("--model-name", help="DINOv2 model name")
    parser.add_argument("--dino-model", dest="dino_model", help="Local clone path for the DINOv2 repo")
    parser.add_argument(
        "--model-repo",
        dest="dino_model",
        help="Deprecated: use --dino-model for the local DINOv2 repo path",
    )
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
    parser.add_argument(
        "--hdb-cluster-selection-method",
        choices=["eom", "leaf"],
        help="HDBSCAN cluster selection method. Default: eom.",
    )
    parser.add_argument(
        "--hdb-cluster-selection-epsilon",
        type=float,
        help="HDBSCAN cluster_selection_epsilon; larger values merge nearby clusters.",
    )
    hdb_single = parser.add_mutually_exclusive_group()
    hdb_single.add_argument(
        "--hdb-allow-single-cluster",
        dest="hdb_allow_single_cluster",
        action="store_true",
        default=None,
        help="Allow HDBSCAN to return a single non-noise cluster.",
    )
    hdb_single.add_argument(
        "--no-hdb-allow-single-cluster",
        dest="hdb_allow_single_cluster",
        action="store_false",
        help="Disallow single-cluster HDBSCAN results. Default.",
    )
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
    parser.add_argument(
        "--refine-prob-threshold",
        type=float,
        help=(
            "Two-pass mode only: refine pass-1 samples whose HDBSCAN membership "
            "probability is below this 0-1 cutoff. Default: 0.7."
        ),
    )
    refine = parser.add_mutually_exclusive_group()
    refine.add_argument(
        "--refine-include-noise",
        dest="refine_include_noise",
        action="store_true",
        default=None,
        help=(
            "Two-pass mode only: also send pass-1 noise points (cluster -1) to the "
            "refinement pass. Default: enabled."
        ),
    )
    refine.add_argument(
        "--no-refine-noise",
        dest="refine_include_noise",
        action="store_false",
        help="Two-pass mode only: do not automatically refine pass-1 noise points.",
    )
    write_dim = parser.add_mutually_exclusive_group()
    write_dim.add_argument(
        "--write-dimreduction-vector", dest="write_dimreduction_vector", action="store_true", default=None
    )
    write_dim.add_argument(
        "--no-write-dimreduction-vector",
        dest="write_dimreduction_vector",
        action="store_false",
    )
    subclustering = parser.add_mutually_exclusive_group()
    subclustering.add_argument(
        "--subclustering",
        dest="subclustering",
        action="store_true",
        default=None,
        help=(
            "Run automatic post-pipeline subclustering for clusters larger than "
            "--min-for-subclustering. Default: enabled."
        ),
    )
    subclustering.add_argument(
        "--no-subclustering",
        dest="subclustering",
        action="store_false",
        help="Skip automatic post-pipeline subclustering.",
    )
    parser.add_argument(
        "--min-for-subclustering",
        type=int,
        help="Minimum cluster size that triggers automatic subclustering. Default: 1000.",
    )
    parser.add_argument(
        "--subclustering-umap-dim",
        type=int,
        help="UMAP dimensionality used for automatic subclustering. Default: 60.",
    )
    parser.add_argument(
        "--subclustering-umap-neighbors",
        type=int,
        help="UMAP n_neighbors used for automatic subclustering. Default: 30.",
    )
    parser.add_argument(
        "--subclustering-hdbscan-min-cluster-size",
        type=int,
        help="HDBSCAN min_cluster_size used for automatic subclustering. Default: 7.",
    )
    parser.add_argument(
        "--subclustering-hdb-min-samples",
        type=int,
        help="HDBSCAN min_samples used for automatic subclustering. Default: 6.",
    )
    parser.add_argument(
        "--subclustering-hdb-cluster-selection-method",
        choices=["eom", "leaf"],
        help=(
            "HDBSCAN cluster selection method used for subclustering. "
            "Defaults to the main HDBSCAN method."
        ),
    )
    parser.add_argument(
        "--subclustering-hdb-cluster-selection-epsilon",
        type=float,
        help=(
            "HDBSCAN cluster_selection_epsilon used for subclustering. "
            "Defaults to the main HDBSCAN epsilon."
        ),
    )
    sub_hdb_single = parser.add_mutually_exclusive_group()
    sub_hdb_single.add_argument(
        "--subclustering-hdb-allow-single-cluster",
        dest="subclustering_hdb_allow_single_cluster",
        action="store_true",
        default=None,
        help=(
            "Allow HDBSCAN to return one non-noise cluster during automatic "
            "subclustering. Defaults to the main HDBSCAN setting."
        ),
    )
    sub_hdb_single.add_argument(
        "--no-subclustering-hdb-allow-single-cluster",
        dest="subclustering_hdb_allow_single_cluster",
        action="store_false",
        help="Disallow single-cluster HDBSCAN results during automatic subclustering.",
    )
    merge_noise = parser.add_mutually_exclusive_group()
    merge_noise.add_argument(
        "--merge-noise-subclusters",
        dest="merge_noise_subclusters",
        action="store_true",
        default=None,
        help=(
            "After automatic subclustering, remap non-noise subclusters from "
            "parent cluster -1 into new top-level cluster IDs."
        ),
    )
    merge_noise.add_argument(
        "--no-merge-noise-subclusters",
        dest="merge_noise_subclusters",
        action="store_false",
        help="Keep noise subclusters only in subclusters/cluster_-1/. Default.",
    )
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument(
        "--summarize-clusters",
        type=Path,
        help="Generate clusters_summary.csv from an existing clusters.csv and exit.",
    )
    parser.add_argument(
        "--summarize-classes-in-clusters",
        type=Path,
        dest="summarize_classes_in_clusters",
        help=(
            "Generate clusters_summary_classes.csv from an existing clusters.csv and exit. "
            "Only basenames ending with _class_1234.jpg contribute to class-derived columns. "
            "Use together with --classes-benchmark-file to include %%_of_total_class columns."
        ),
    )
    parser.add_argument(
        "--classes-benchmark-file",
        type=Path,
        dest="classes_benchmark_file",
        help=(
            "Path to a CSV with columns 'label_id' and 'count' giving the total number of "
            "images per class in the dataset. Used to compute class_X_%%_of_total_class "
            "columns in clusters_summary_classes.csv."
        ),
    )
    parser.add_argument(
        "--summarize-dominants-and-diff",
        type=Path,
        dest="summarize_dominants_and_diff",
        help=(
            "Generate clusters_dominant_classes_and_diff.csv from an existing "
            "clusters_summary_classes.csv, including diff_1st-2nd_norm, and exit."
        ),
    )
    parser.add_argument(
        "--summary-scores",
        type=Path,
        dest="summary_scores",
        help=(
            "Generate clustering_score_report.csv from an existing "
            "clusters_dominant_classes_and_diff.csv and the sibling "
            "clusters_summary_classes.csv, then exit."
        ),
    )
    return parser


def parse_args(argv: Optional[List[str]] = None, *, prog: Optional[str] = None) -> argparse.Namespace:
    parser = build_parser(prog=prog)
    return parser.parse_args(argv)


def _write_config_record(cfg) -> None:
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "parameters_used.txt"
    config_path.write_text(config_to_yaml(cfg), encoding="utf-8")


def main(argv: Optional[List[str]] = None, *, prog: Optional[str] = None) -> int:
    args = parse_args(argv, prog=prog)
    if args.summarize_clusters is not None:
        summarize_clusters_csv(args.summarize_clusters)
        return 0
    if args.summarize_classes_in_clusters is not None:
        summary_classes_path = summarize_classes_in_clusters_csv(
            args.summarize_classes_in_clusters,
            benchmark_path=getattr(args, "classes_benchmark_file", None),
        )
        summarize_cluster_dominant_classes_and_diff_csv(summary_classes_path)
        return 0
    if args.summarize_dominants_and_diff is not None:
        summarize_cluster_dominant_classes_and_diff_csv(args.summarize_dominants_and_diff)
        return 0
    if args.summary_scores is not None:
        summarize_clustering_score_report_csv(args.summary_scores)
        return 0
    cfg = build_config(args)
    validate_config(cfg)
    _write_config_record(cfg)
    if args.print_config:
        print(config_to_yaml(cfg))
        return 0
    run_pipeline(cfg)
    return 0
