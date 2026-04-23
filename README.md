# DINOv2 -> UMAP -> HDBSCAN clustering pipeline

This project clusters unlabeled arthropod images using a three-stage pipeline:

1. **Embedding**: Extracts DINOv2 image embeddings from a pretrained ViT model.
2. **Dimensionality reduction**: Uses UMAP to reduce embeddings while preserving local neighborhoods.
3. **Clustering**: Uses HDBSCAN to group similar images and flag noise.

(Optional) **Size-aware weighting**: Adds a size feature (non-background pixel area) so arthropod size influences clustering.

Artifacts are written to the output directory, including embeddings, reduced vectors, and a CSV that maps each image to a cluster label plus HDBSCAN metadata.

## Requirements

- Python 3.11 or higher.

The code expects these Python packages to be available:

- torch
- torchvision
- umap-learn
- hdbscan
- pillow
- numpy
- certifi

## Recommended installation procedure

Clone the repository. In the terminal, go to the folder directory where you want 
to install this code and type:

```bash
git clone https://github.com/ArturoAvelino/Clustering-images-by-resemblance.git

cd Clustering-images-by-resemblance
```

Create a Python virtual environment inside the `Clustering-images-by-resemblance` directory:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Update `pip` (the package installer):

```bash
pip install --upgrade pip
```

And install the required packages:

```bash
pip install -r requirements.txt
```


## DINOv2 setup (local clone)

This repo expects a local clone of `facebookresearch/dinov2` when `--dino-model`
or `DINOv2_REPO` is used. The `dinov2/` directory is not synced to this repo, so
clone it separately:

```bash
git clone https://github.com/facebookresearch/dinov2 ./dinov2
```

Then point the pipeline at the local clone:

```bash
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output --dino-model ./dinov2
```

## Model download and SSL errors

The default loader downloads the DINOv2 repo via `torch.hub`. If your environment has SSL
verification issues, you can avoid HTTPS by pointing to a local clone of the repo:

```bash
git clone https://github.com/facebookresearch/dinov2 /path/to/dinov2
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output --dino-model /path/to/dinov2
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

## Usage (CLI)

CLI commands:

| Command | Purpose |
| --- | --- |
| `compute-clusters` | Run the DINOv2 → UMAP → HDBSCAN clustering pipeline. |
| `copy-crops-to-cluster-dirs` | Copy clustered images/JSON into cluster-labeled folders. |
| `calibrate-threshold` | Estimate a background color distance threshold for auto-cropping. |

Basic run:

```bash
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output
```

Show help for the clustering pipeline options:

```bash
python clustering compute-clusters --help
```

Using a YAML config:

```bash
python clustering compute-clusters --config /path/to/config.yaml
```

Print the values of all the config variables used, including default interval variables:

```bash
python clustering compute-clusters --config /path/to/config.yaml --print-config
```

### Rerun dimensionality-reduction (UMAP) + clustering (HDBSCAN) without embeddings (DINOv2)

Run only UMAP + HDBSCAN using embeddings from a previous run:

```bash
python clustering compute-clusters \
  --compute only-dimreduction-and-clustering \
  --config /path/to/config_example_run_only_dimreduction_and_clustering.yaml
```

Use the `only-dimreduction-and-clustering` compute mode to skip DINOv2 and reuse
cached outputs from a previous run. The directory passed in `dino_files` must
contain:

- `embeddings.dat`
- `embeddings.json`
- `sizes.npy`
- `images.txt`

The new outputs (`umap.npy`, `clusters.csv`, `images.txt`, `summary_clusters.csv`)
are written to `output_dir`.

### Rerun clustering (HDBSCAN) without embeddings (DINOv2) + dimensionality-reduction (UMAP)

Run only HDBSCAN using UMAP outputs from a previous run:

```bash
python clustering compute-clusters \
  --compute only-clustering \
  --config /path/to/config_example_run_only_clustering.yaml
```

Use the `only-clustering` compute mode to skip DINOv2 and UMAP, reusing cached
UMAP outputs from a previous run. The directory passed in `umap_files` must
contain:

- `umap.npy`
- `images.txt`

The new outputs (`clusters.csv`, `images.txt`, `summary_clusters.csv`) are written
to `output_dir`.

### Generate a summary file from an existing clusters.csv:

```bash
python clustering compute-clusters --summarize-clusters /path/to/output/clusters.csv
```

### Organize clustered outputs into folders

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

Copy high-confidence representatives and outliers into per-cluster subdirectories:

```bash
python clustering copy-crops-to-cluster-dirs --clusters /path/to/output/clusters.csv \
  --input-dir /path/to/images --dest-dir /path/to/clustered \
  --subdir-confidence 0.9 --subdir-outliers 0.1
```

Show help for the cluster directory copy options:

```bash
python clustering copy-crops-to-cluster-dirs --help
```

What `copy-crops-to-cluster-dirs` does:

- Reads `clusters.csv` and groups images into subfolders named after their cluster ID (for example `0/`, `1/`, `-1/` for noise).
- Uses the `image_id` column as a path relative to `--input-dir` and mirrors the original subfolder structure under each cluster unless `--flat` is provided.
- Copies matching `.JSON` metadata files alongside the images, or uses `--json-only` to copy just metadata while leaving images in place.
- Optionally creates per-cluster subdirectories:
  `representatives_confid_XX` for images with `probabilities >= XX` and `outlier_scores <= 0.1`,
  and `outliers_YY` for images with `outlier_scores >= YY`. These are skipped when `--json-only` is used.
- Handles destination conflicts with `--on-conflict` (`rename`, `overwrite`, `skip`, or `error`) and supports `--dry-run` for previews.

### Background color threshold calibration

The pipeline can auto-crop images to remove background pixels.
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


## Input Configuration File

The annotated YAML template lives at `config_files/config_example_run_full_pipeline.yaml`
and is the recommended starting point for full runs. A second template,
`config_files/config_example_only_dimreduction_and_clustering.yaml`, shows the
minimal inputs to rerun UMAP + HDBSCAN from cached DINOv2 outputs. A third template,
`config_files/config_example_only_clustering.yaml`, shows the minimal inputs to rerun
HDBSCAN using cached UMAP outputs. The pipeline reads YAML configs directly. The
most important fields are:

- `cropped_images_dir` (the older `input_image_dir` key is still accepted)
- `output_dir`
- `dino_model` (optional local clone path)
- `batch_size`
- `num_workers`
- `umap_dim`
- `hdbscan_min_cluster_size`
- `write_dimreduction_vector` (default `true`, writes the UMAP vector to `clusters.csv`)
- `two_pass` or `fast_tune` (recommended: `false`)
- `autocrop` (default: `false`)
- `background_color` (RGB background color as `[R, G, B]`; default is tuned for blue)
- `autocrop_threshold` (color-distance threshold used to separate background from foreground)
- `size_feature_weight` (higher values emphasize size)
- `image_size_in_kbytes_min` / `image_size_in_kbytes_max` (optional file-size filter; KB = 1024 bytes)
- `compute` (use `only-dimreduction-and-clustering` to skip embedding, or `only-clustering` to skip embedding + UMAP)
- `dino_files` (directory containing embeddings.dat, embeddings.json, sizes.npy, and images.txt)
- `umap_files` (directory containing umap.npy and images.txt)

For white backgrounds, set `background_color` to `[255, 255, 255]` and tune
`autocrop_threshold` if needed.

`model_repo` is still accepted for backward compatibility, but `dino_model` is the
preferred config key going forward.

### UMAP and HDBSCAN configuration details

Use these definitions to tune clustering behavior. All UMAP settings operate on
the DINOv2 embedding vectors; HDBSCAN operates on the UMAP-reduced vectors.

- `umap_dim`: Target dimensionality of the UMAP projection used for clustering.
  Higher values preserve more structure from the original embeddings but
  increase runtime and can make density-based clustering less distinct. Lower
  values speed up HDBSCAN and can simplify structure but may discard relevant
  variation. A practical starting range is 15-60; increase if clusters look
  over-merged, decrease if clustering is noisy or unstable.
- `umap_neighbors`: Number of nearest neighbors used to build the UMAP graph.
  Smaller values emphasize local structure and can split fine-grained clusters.
  Larger values emphasize global structure, smoothing the manifold and often
  reducing the number of clusters. Typical values are 10-50; push lower for
  fine-grained grouping, higher for broader grouping.
- `umap_min_dist`: Minimum allowed distance between points in the UMAP space.
  Lower values (close to 0.0) allow tight packing and compact clusters; higher
  values spread points apart and can reduce very dense clumps. Start with 0.0-0.2
  for cluster discovery, raise it if you see overly tight blobs.
- `umap_metric`: Distance metric used by UMAP on the original embeddings.
  `cosine` is a common choice for high-dimensional embeddings (including DINOv2)
  because it focuses on angular similarity. `euclidean` can work but may be more
  sensitive to embedding norm; only switch if you know your embeddings are
  normalized or you have a clear reason.
- `hdbscan_min_cluster_size`: Minimum cluster size HDBSCAN will consider. Smaller
  values yield more (and smaller) clusters; larger values merge smaller groups
  into noise or larger clusters. Set this to roughly the smallest cluster size
  you care about.
- `hdb_min_samples`: Minimum samples in a neighborhood for a point to be
  considered a core point. Higher values make clustering more conservative and
  increase the number of points labeled as noise; lower values are more liberal
  but can create spurious clusters. A good starting point is 5-20 or the same
  as `hdbscan_min_cluster_size` for stricter clustering.
- `hdb_metric`: Distance metric used by HDBSCAN on the UMAP output. `euclidean`
  is standard in low-dimensional UMAP spaces. Only change this if you have a
  specific reason and can explain how distances should behave in the reduced
  space.

Size filtering is applied when `images.txt` is generated. If you change the size
range after a run, delete `images.txt` or rerun with `--force` to rebuild it.

When using `compute: only-dimreduction-and-clustering`, the pipeline skips the
embedding step entirely and reads cached files from `dino_files`. You can point
`dino_files` at the output directory of a previous run (for example
`/path/to/output` or `/path/to/output/stages/pass1`) as long as it contains the
required files.

When using `compute: only-clustering`, the pipeline reads cached UMAP outputs
from `umap_files`. You can point `umap_files` at the output directory of a
previous run (for example `/path/to/output` or `/path/to/output/umap_hdbscan_only`)
as long as it contains `umap.npy` and `images.txt`.

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
    hdbscan_min_cluster_size=25,
    two_pass=False,
    dino_model="/path/to/dinov2",
)
```

## Inputs

- A folder containing JPG/JPEG images (any size/aspect ratio).
- Background is expected to be a solid color (default: blue). The pipeline can auto-crop
  non-background pixels (autocrop is off by default).
- Optional file-size filtering can include only images within a size range.

## Outputs

The output directory contains:

- `clusters.csv` with columns `[image_id, cluster, probabilities, outlier_scores, dim_reduction]`
  (noise is `-1`; `dim_reduction` is a JSON array of
  UMAP values, length = `umap_dim` unless `write_dimreduction_vector: false`,
  in which case the column is empty)
- `summary_clusters.csv` with columns `[cluster, num_obj_in_cluster]`
- `embeddings.dat` and `embeddings.json` (embedding matrix + metadata)
- `umap.npy` (UMAP-reduced vectors)
- `images.txt` (stable list of image paths used)

When `--two-pass` or `--fast-tune` is used, outputs are grouped under `output_dir/stages/`.
When running `--compute only-dimreduction-and-clustering`, embeddings are read from
`dino_files` while `umap.npy`, `clusters.csv`, and `images.txt` are written to `output_dir`.
When running `--compute only-clustering`, `umap.npy` and `images.txt` are read from
`umap_files`, while `clusters.csv` and `images.txt` are written to `output_dir`.

## Two-pass mode (pass 1 / pass 2)

When `two_pass: true` is enabled in the configuration input file, the pipeline runs HDBSCAN in two stages:

1. **Pass 1 (fast stage)**: Uses the *fast* UMAP settings to reduce the full
   dataset, then clusters all images. By default, `fast_umap_dim=15`, so the
   UMAP vectors given to HDBSCAN in pass 1 have 15 elements each.
2. **Pass 2 (refinement stage)**: Re-runs UMAP + HDBSCAN only on the uncertain
   subset, using the *full* settings. The UMAP vectors given to HDBSCAN in pass 2
   have `umap_dim` elements (for example 30).

Recommendation: prefer `two_pass: false` so all objects are clustered using
`umap_dim` consistently.

## DINOv2 embedding dimension

The embedding vector length is the model's `embed_dim`. This pipeline takes the
CLS token (when the model returns a token sequence) and stores a vector of size
`embed_dim` per image. The dimension is read from `model.embed_dim` and, if that
is missing, inferred from a forward pass.

Common backbone sizes used here:

| Model | Embedding dimension |
| --- | --- |
| `dinov2_vits14` | 384 |
| `dinov2_vitb14` (default) | 768 |
| `dinov2_vitl14` | 1024 |
| `dinov2_vitg14` | 1536 |

If you supply a custom `model_name`, the embedding size will match that model's
`embed_dim`.

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
  you want per‑point diagnostics, HDBSCAN exposes `probabilities_` and
  `outlier_scores_`, which this code writes to `clusters.csv` and uses
  to flag uncertain points in the two‑pass flow. `pipeline/pipeline.py`.

- The `dim_reduction` column stores the per-image UMAP output as a JSON array.
  The length matches the UMAP dimensionality for the stage that produced the row.
  In `--two-pass` or `--fast-tune` runs, this can be `fast_umap_dim` unless you
  set it to match `umap_dim`. If `write_dimreduction_vector: false`, this
  column is left empty.

- Lower cluster IDs are not more similar/dense than higher IDs.

## Notes

- If you see SSL errors, prefer using a local DINOv2 repo as described above.
- For large datasets, consider lowering `batch_size` or `num_workers`.
