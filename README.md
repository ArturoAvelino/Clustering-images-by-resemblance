# DINOv2 -> UMAP -> HDBSCAN clustering pipeline

This project clusters unlabeled arthropod images using a three-stage pipeline:

1. **Embedding**: Extracts DINOv2 image embeddings from a pretrained ViT model.
2. **Dimensionality reduction**: Uses UMAP to reduce embeddings while preserving local neighborhoods.
3. **Clustering**: Uses HDBSCAN to group similar images and flag noise.
4. **Size-aware weighting**: Adds a size feature (non-background pixel area) so arthropod size influences clustering.

Artifacts are written to the output directory, including embeddings, reduced vectors, and a CSV that maps each image to a cluster label.

## Requirements

The code expects these Python packages to be available:

- torch
- torchvision
- umap-learn
- hdbscan
- pillow
- numpy
- certifi

Install them with:

```bash
pip install -r requirements.txt
```

## DINOv2 setup (local clone)

This repo expects a local clone of `facebookresearch/dinov2` when `--model-repo`
or `DINOv2_REPO` is used. The `dinov2/` directory is not synced to this repo, so
clone it separately:

```bash
git clone https://github.com/facebookresearch/dinov2 ./dinov2
```

Then point the pipeline at the local clone:

```bash
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output --model-repo ./dinov2
```

## Inputs

- A folder containing JPG/JPEG images (any size/aspect ratio).
- Background is expected to be a solid color (default: blue). The pipeline can auto-crop
  non-background pixels (autocrop is off by default).
- Optional file-size filtering can include only images within a size range.

## Outputs

The output directory contains:

- `clusters.csv` with columns `[image_id, cluster]` (noise is `-1`)
- `summary_clusters.csv` with columns `[cluster, num_obj_in_cluster]`
- `embeddings.dat` and `embeddings.json` (embedding matrix + metadata)
- `umap.npy` (UMAP-reduced vectors)
- `images.txt` (stable list of image paths used)

When `--two-pass` or `--fast-tune` is used, outputs are grouped under `output_dir/stages/`.

## Usage (CLI)

Basic run:

```bash
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output
```

Show help for the clustering pipeline options:

```bash
python clustering compute-clusters --help
```

Estimate a good background distance threshold from sample images:

```bash
python clustering calibrate-threshold --input-dir /path/to/images --background-color 45,71,159
```

Show help for the threshold calibration options:

```bash
python clustering calibrate-threshold --help
```

What `calibrate_threshold.py` does:

- Samples pixels from a subset of images, computes RGB distance to the background color, and applies Otsu's method to each image to find a foreground/background split.
- Prints a suggested median threshold and summary stats (p25/p75/min/max) so you can pick a conservative or aggressive cutoff.
- Use the suggested value as `--autocrop-threshold` (or in config) and validate on a few images; increase it if too much background remains, decrease it if you are losing object pixels.

Using a YAML config:

```bash
python clustering compute-clusters --config /path/to/config.yaml
```

Print the merged config:

```bash
python clustering compute-clusters --config /path/to/config.yaml --print-config
```

Generate a summary file from an existing clusters.csv:

```bash
python clustering compute-clusters --summarize-clusters /path/to/output/clusters.csv
```

## CLI commands

| Command | Purpose |
| --- | --- |
| `compute-clusters` | Run the DINOv2 → UMAP → HDBSCAN clustering pipeline. |
| `calibrate-threshold` | Estimate a background color distance threshold for auto-cropping. |
| `copy-crops-to-cluster-dirs` | Copy clustered images/JSON into cluster-labeled folders. |

Organize clustered outputs into folders (copies images and matching `.JSON` metadata, even if either the image or the JSON file is missing):

```bash
python clustering copy-crops-to-cluster-dirs --clusters /path/to/output/clusters.csv \
  --input-dir /path/to/images --dest-dir /path/to/clustered
```

Copy only `.JSON` metadata (leave images in place):

```bash
python clustering copy-crops-to-cluster-dirs --clusters /path/to/output/clusters.csv \
  --input-dir /path/to/images --dest-dir /path/to/clustered --json-only
```

Show help for the cluster directory copy options:

```bash
python clustering copy-crops-to-cluster-dirs --help
```

What `copy-crops-to-cluster-dirs` does:

- Reads `clusters.csv` and groups images into subfolders named after their cluster ID (for example `0/`, `1/`, `-1/` for noise).
- Uses the `image_id` column as a path relative to `--input-dir` and mirrors the original subfolder structure under each cluster unless `--flat` is provided.
- Copies matching `.JSON` metadata files alongside the images, or uses `--json-only` to copy just metadata while leaving images in place.
- Handles destination conflicts with `--on-conflict` (`rename`, `overwrite`, `skip`, or `error`) and supports `--dry-run` for previews.

## Model download and SSL errors

The default loader downloads the DINOv2 repo via `torch.hub`. If your environment has SSL
verification issues, you can avoid HTTPS by pointing to a local clone of the repo:

```bash
git clone https://github.com/facebookresearch/dinov2 /path/to/dinov2
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output --model-repo /path/to/dinov2
```

You can also set an environment variable:

```bash
DINOv2_REPO=/path/to/dinov2 python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output
```

If you need to supply a custom certificate bundle (corporate proxies, older Python installs),
pass a PEM bundle path:

```bash
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output --ssl-ca-bundle /path/to/ca-bundle.pem
```

## Configuration

The annotated YAML template lives at `config_files/config_example.yaml` and is
the recommended starting point. The pipeline reads YAML configs directly. The
most important fields are:

- `cropped_images_dir` (the older `input_image_dir` key is still accepted)
- `output_dir`
- `model_repo` (optional local clone path)
- `batch_size`
- `num_workers`
- `umap_dim`
- `hdb_min_cluster_size`
- `two_pass` or `fast_tune`
- `autocrop`
- `background_color` (RGB background color as `[R, G, B]`; default is tuned for blue)
- `autocrop_threshold` (color-distance threshold used to separate background from foreground)
- `size_feature_weight` (higher values emphasize size)
- `image_size_in_kbytes_min` / `image_size_in_kbytes_max` (optional file-size filter; KB = 1024 bytes)

For white backgrounds, set `background_color` to `[255, 255, 255]` and tune
`autocrop_threshold` if needed.

Size filtering is applied when `images.txt` is generated. If you change the size
range after a run, delete `images.txt` or rerun with `--force` to rebuild it.

## Python API

You can run the pipeline in Python:

```python
from main import clustering

output_csv = clustering(
    "/path/to/images",
    "/path/to/output",
    batch_size=16,
    num_workers=2,
    umap_dim=30,
    hdb_min_cluster_size=25,
    two_pass=True,
    model_repo="/path/to/dinov2",
)
```

## About the clusters ID values in `clusters.csv`

The HDBSCAN cluster IDs are just arbitrary labels; lower IDs are
not “denser” or more “precise” than higher ones.

What the code does:

- pipeline/algorithms.py creates an hdbscan.HDBSCAN(...) instance,
  runs fit_predict, and returns labels plus probabilities_ if
  available. There’s no post‑processing that ranks or reorders
  cluster IDs by density or similarity. pipeline/algorithms.py.

- Noise points are labeled `-1` by HDBSCAN; that’s the only ID with a
  defined meaning beyond “this is cluster k.”
  pipeline/algorithms.py. It is, cluster `-1` is the noise/outlier label from HDBSCAN — items the
algorithm did not assign to any cluster. You can see it treated
as “uncertain” and the organizer will place those images into a `-1/`
folder.

In summary:

- Cluster IDs like 0, 1, 50 carry no intrinsic significance or
  ordering; a cluster labeled 50 is not inherently less similar or
  less dense than cluster 1.

- Within a cluster, similarity/density is not encoded by the ID. If
  you want a per‑point “confidence,” HDBSCAN exposes `probabilities_`,
  which this code already returns and uses to flag uncertain points
  in the two‑pass flow. `pipeline/pipeline.py`.

- Lower cluster IDs are not more similar/dense than higher IDs.

## Notes

- If you see SSL errors, prefer using a local DINOv2 repo as described above.
- For large datasets, consider lowering `batch_size` or `num_workers`.
