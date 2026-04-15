from __future__ import annotations

"""
DINOv2 → UMAP → HDBSCAN clustering pipeline for unlabeled insect images.

What it does
------------
1) Extracts DINOv2 embeddings for each image (default: 768‑dim with dinov2_vitb14).
2) Reduces dimensionality with UMAP to preserve local neighborhoods.
3) Clusters with HDBSCAN and writes a CSV of image_id → cluster label.

Inputs
------
- A folder containing JPG images (any size/aspect ratio is fine).
- Background is expected to be a solid color (default: blue); an optional auto-crop removes
  background-colored margins.
- Optional file-size filtering can restrict images by size (in KB) via config.

Outputs
-------
- clusters.csv: columns [image_id, cluster, probabilities, outlier_scores, dim_reduction]
  (noise labeled as -1; dim_reduction is a JSON array unless write_dimreduction_vector is false).
- embeddings.dat / embeddings.json: saved embedding matrix + metadata.
- umap.npy: reduced vectors used for clustering.
- images.txt: stable list of image paths used for the run.

How to use (CLI)
---------------
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output
python clustering compute-clusters --config /path/to/config.yaml --batch-size 8  # CLI overrides config
python clustering compute-clusters --config /path/to/config.yaml --print-config

Example YAML config
-------------------
cropped_images_dir: "/path/to/images"
output_dir: "/path/to/output"
batch_size: 16
num_workers: 2
umap_dim: 30
hdbscan_min_cluster_size: 25
two_pass: true
autocrop: true
image_size_in_kbytes_min: 10
image_size_in_kbytes_max: 99.99
write_dimreduction_vector: true

An annotated YAML template is available at config_files/config_example_run_full_pipeline.yaml.

Optional modes
--------------
- --two-pass: fast first pass + refine uncertain samples with full model.
- --fast-tune: quick pass only (for parameter tuning).
- --no-autocrop: disable auto-crop to non-white pixels.

How to use (Python)
-------------------
from main import clustering
clustering("/path/to/images", "/path/to/output", two_pass=True)
"""

import sys
from multiprocessing import freeze_support
from typing import List, Optional

from pipeline.cli import main as _cli_main, parse_args
from pipeline.config import PipelineConfig
from pipeline.pipeline import clustering, run_pipeline
from pipeline.summary import summarize_clusters_csv


def main(argv: Optional[List[str]] = None) -> int:
    return _cli_main(argv)


__all__ = [
    "PipelineConfig",
    "clustering",
    "run_pipeline",
    "summarize_clusters_csv",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main(sys.argv[1:]))
