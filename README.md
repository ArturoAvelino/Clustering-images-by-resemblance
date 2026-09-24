# DINOv2 -> UMAP -> HDBSCAN clustering pipeline

This project clusters unlabeled arthropod images using a three-stage pipeline:

1. **Embedding**: Extracts DINOv2 image embeddings from a pretrained ViT model.
2. **Dimensionality reduction**: Uses UMAP to reduce embeddings while preserving local neighborhoods.
3. **Clustering**: Uses HDBSCAN to group similar images and flag noise.

(Optional) **Size-aware weighting**: Adds a size feature (non-background pixel area) so arthropod size influences clustering.

Artifacts are written to the output directory, including embeddings, reduced vectors, and a CSV that maps each image to a cluster label plus HDBSCAN metadata. The pipeline now also annotates each `clusters.csv` row with whether the filename carries a valid strict class label.

After the main run finishes, automatic subclustering is enabled by default. Any
final cluster with more than 1000 objects is processed again as its own subset,
using the cached DINOv2 artifacts from the main run and configurable
subclustering settings. By default, subclustering uses `umap_dim=60`,
`umap_neighbors=30`, `hdbscan_min_cluster_size=7`, and `hdb_min_samples=6`.
When `merge_noise_subclusters: true`, non-noise subclusters found inside parent
cluster `-1` are remapped into fresh top-level cluster IDs in `clusters.csv`.

## Cluster a subset using existing embeddings and sizes

Keep the **original full `images.txt` in its original order** beside
`embeddings.dat`, `embeddings.json`, and `sizes.npy`. Row numbers are the link
between filenames and features; replacing this index with 2,000 names cannot
identify the corresponding rows in a 15-million-row cache.

Create a separate `subset_images.txt` containing one requested image per line.
Entries must match the original index exactly, including directory prefixes
and case. Blank lines are ignored; duplicate or missing entries are errors.
The subset may be listed in any order, which is preserved in the outputs.
No image files or DINO model are needed for this mode.

Use [the subset configuration template](config_files/config_example_run_subset.yaml),
or add these fields to your UMAP + HDBSCAN configuration:

```yaml
compute: only-dimreduction-and-clustering
dino_files: /Users/aavelino/run_all_1
subset_images: /Users/aavelino/run_all_1/subset_images.txt
output_dir: /Users/aavelino/run_all_1/subset_2000
subclustering: false
```

Then run from the repository directory:

```bash
python clustering compute-clusters \
  --compute only-dimreduction-and-clustering \
  --config /Users/aavelino/run_all_1/config_run_only_dimreduction_and_clustering.yaml
```

The CLI equivalent of the selection key is `--subset-images /path/to/list.txt`.
`output_dir` must differ from `dino_files`. The full index is streamed once,
and selected embeddings and sizes are read through memory maps and copied into
the output directory. Memory use scales with the subset, not the full index.
The output contains a reusable subset cache (`images.txt`, `embeddings.dat`,
`embeddings.json`, `sizes.npy`), fresh `umap.npy`, `clusters.csv`, and summaries.
Neither DINO embeddings nor image sizes are recomputed.

For later parameter experiments, point `dino_files` at `subset_2000` and omit
`subset_images` to avoid scanning the full index again. For HDBSCAN-only
experiments, use `compute: only-clustering` and `umap_files: .../subset_2000`.

UMAP and HDBSCAN are fitted on the selected images, so these clusters can differ
from the same images' clusters in a full-dataset run. Size standardization also
uses the subset's mean and standard deviation. Set `size_feature_weight` to your
desired weight (the pipeline default `0.0` ignores size). Adjust UMAP neighbors
and HDBSCAN minimum cluster size for the smaller population. Set
`subclustering: true` if you want the usual automatic subdivision of large
clusters; it also reuses the subset cache.

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
| `copy-crops-to-subdirs-representative` | Copy high-probability, low-outlier-score crop images into threshold-labeled per-cluster representative subdirectories. |
| `copy-crops-to-subdir-outliers` | Copy low-probability, high-outlier-score crop images into threshold-labeled per-cluster outlier subdirectories. |
| `calibrate-threshold` | Estimate a background color distance threshold for auto-cropping. |

Standalone helper script:

| Script | Purpose |
| --- | --- |
| `count-classes-on-labeled-filenames` | Scan a directory tree and count only basenames that end with `_class_1234.jpg`, writing `classes_in_dataset.csv` with one row per extracted class ID, optionally enriched with class names from a BIGLE labels CSV. |

Basic run:

```bash
python clustering compute-clusters --input-dir /path/to/images --output-dir /path/to/output
```

Disable automatic subclustering, or change the trigger threshold:

```bash
python clustering compute-clusters \
  --input-dir /path/to/images \
  --output-dir /path/to/output \
  --no-subclustering

python clustering compute-clusters \
  --input-dir /path/to/images \
  --output-dir /path/to/output \
  --min-for-subclustering 2000
```

Merge non-noise subclusters found inside the parent noise cluster (`-1`) back
into the top-level `clusters.csv`:

```bash
python clustering compute-clusters \
  --input-dir /path/to/images \
  --output-dir /path/to/output \
  --merge-noise-subclusters
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

Generate `classes_in_dataset.csv` from image filenames that follow the strict
`_class_1234.jpg` rule:

```bash
python count-classes-on-labeled-filenames \
  --files-dir /path/to/folder/ \
  --output-dir /path/to/output/directory/
```

This helper walks `--files-dir` recursively and counts only files whose
basename ends exactly with `_class_1234.jpg`. That means `_class_` must appear
immediately before the class value, the class value must be exactly 4 digits,
and those digits must be immediately followed by the `.jpg` extension. Files
that do not match that exact basename pattern are ignored. The output
`classes_in_dataset.csv` uses the headers `class_ID` and `num_objs`.

To also write class names, pass a BIGLE labels CSV such as `labels.csv` with at
least `id` and `name` columns:

```bash
python count-classes-on-labeled-filenames \
  --files-dir /path/to/folder/ \
  --biigleID-to-names-file /path/to/labels.csv \
  --output-dir /path/to/output/directory/
```

With `--biigleID-to-names-file`, the helper matches each extracted `class_ID`
against the BIIGLE CSV `id` column and writes `classes_in_dataset.csv` with the
headers `class_ID`, `class_name`, and `num_objs`. If the option is omitted, the
command keeps the current behavior and writes only `class_ID` and `num_objs`.

The clustering pipeline reuses the same strict filename rule for the
`clusters.csv` `labeled` column and for class-based summary files:

- `True`: basename ends with `_class_1234.jpg`
- `False`: any other basename, including missing `_class_`, non-4-digit class
  values, or extensions other than `.jpg`

`count-classes-on-labeled-filenames` now uses that same strict rule. The old
`--num-characters-to-read-class` option is kept only for CLI compatibility and
is ignored.

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

The new outputs (`umap.npy`, `clusters.csv`, `images.txt`, `clusters_summary.csv`)
are written to `output_dir`.

Automatic subclustering also works in this mode because the command has access
to `dino_files`. It uses the cached DINOv2 artifacts from `dino_files`, the
newly computed parent `umap.npy` from `output_dir`, and writes subcluster outputs
under `output_dir/subclusters/`.

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

The new outputs (`clusters.csv`, `images.txt`, `clusters_summary.csv`) are written
to `output_dir`.

Automatic subclustering in `only-clustering` mode requires DINOv2 artifacts. If
you also pass `--dino-files /path/to/previous/dino/output`, the subclustering
step slices those cached embeddings and sizes. If `--dino-files` is omitted,
the main clustering run still completes, but post-pipeline subclustering is
skipped because there are no embeddings to reuse for subset UMAP.

### Generate a summary file from an existing clusters.csv:

```bash
python clustering compute-clusters --summarize-clusters /path/to/file/clusters.csv
```

This writes `clusters_summary.csv` next to `clusters.csv` with one row per
cluster. Its columns are `cluster_id`, `num_objs_in_cluster`, and
`num_classes_in_cluster`. The class count uses the same class extraction rule as
`clusters_summary_classes.csv`.

### Generate richness.csv from an existing clusters.csv:

The pipeline writes `richness.csv` only when `write_richness_file: true` is set
in the config, or when `--write-richness-file` is passed to
`compute-clusters`. In that automatic mode, richness generation happens after
the corresponding `clusters.csv` has been written. Automatic subcluster outputs
follow the same flag, so each subcluster `clusters.csv` gets a sibling
`richness.csv` only when the flag is enabled.

You can also generate `richness.csv` on demand from an existing clusters file:

```bash
python clustering write_richness_file \
  --clusters-file /path/to/clusters.csv \
  --output-dir /path/to/output/directory
```

This writes `/path/to/output/directory/richness.csv`. The input file must have
`image_id` and `cluster` columns.

Location IDs are parsed only from the filename prefix before the first
`_r<digits>c<digits>` marker. For example,
`A09-G_r4c4_obj_328626_class_4200.jpg` maps to `A09-G`. Filenames without this
marker, or with no prefix before the marker, are skipped and written to
`location_IDs_not_found.csv` beside `richness.csv`.

### Generate a per-class breakdown of clusters:

`clusters_summary_classes.csv` is written automatically every time the pipeline
runs. It contains one row per cluster with counts and percentages for each image
class found in the dataset.

For cluster summary outputs, a file contributes to class-derived columns only
when its basename ends with `_class_1234.jpg` (for example,
`A01-A_r5c4_obj_280286_class_4218.jpg` contributes class `4218`). Files that do
not match that exact pattern are still counted in `num_objs_in_cluster`, but
they are ignored for `num_classes_in_cluster`, per-class counts, dominant-class
selection, and the score report.

If you need the dataset-wide counts for those filename labels, generate
`classes_in_dataset.csv` directly from the image directory:

```bash
python count-classes-on-labeled-filenames \
  --files-dir /path/to/folder/ \
  --output-dir /path/to/output/directory/
```

To regenerate the file from an existing `clusters.csv` without re-running the
pipeline:

```bash
python clustering compute-clusters \
  --summarize-classes-in-clusters /path/to/file/clusters.csv
```

This also writes the derived `clusters_dominant_classes_and_diff.csv` and
`clustering_score_report.csv` next to `clusters_summary_classes.csv`.

To also include `class_X_%_of_total_class` columns (what fraction of each
class's total dataset images fall in each cluster), supply a benchmark CSV with
columns `label_id` and `count`:

```bash
python clustering compute-clusters \
  --summarize-classes-in-clusters /path/to/file/clusters.csv \
  --classes-benchmark-file /path/to/classes_benchmark.csv
```

The `classes_benchmark.csv` file looks like:

```
label_id,count
4196,29879
4197,421
4198,11
4200,2378
...
```

To have the pipeline use a benchmark file on every run, either pass
`--classes-benchmark-file` alongside `--input-dir` / `--output-dir`, or add
`classes_benchmark_file: /path/to/classes_benchmark.csv` to your YAML config.

### Generate dominant-class counts and differences per cluster

`clusters_dominant_classes_and_diff.csv` is written automatically every time the
pipeline writes `clusters_summary_classes.csv`. The derived
`clustering_score_report.csv` is written at the same time. The dominant-classes
file contains:

- `cluster_id`
- `num_objs_in_cluster`
- `num_classes_in_cluster`
- `1st_dom_class`
- `1st_dom_%`
- `1st_dom_num_objs`
- `2nd_dom_class`
- `2nd_dom_%`
- `2nd_dom_num_objs`
- `diff_1st-2nd_%`
- `diff_1st-2nd_norm`

The dominant classes are selected by comparing all
`class_X_%_of_the_cluster` columns within the same cluster row. The
`num_objs_in_cluster` and `num_classes_in_cluster` are copied from
`clusters_summary_classes.csv`. The `1st_dom_num_objs` and `2nd_dom_num_objs`
values come from the matching `class_X` count columns in
`clusters_summary_classes.csv`. If one class is 100% of a cluster and no other
class has a positive percentage, the second dominant class is written as `0000`,
the second percentage is `0`, the second object count is `0`, and the
difference is `100.00`. `diff_1st-2nd_norm` is `diff_1st-2nd_% / 100`.

To regenerate this file from an existing `clusters_summary_classes.csv`:

```bash
python clustering compute-clusters \
  --summarize-dominants-and-diff /path/to/file/clusters_summary_classes.csv
```

You can also use the standalone script:

```bash
python generate_clusters_dominants_and_diff.py /path/to/file/clusters_summary_classes.csv
```

Use `--output` with the standalone script to choose a custom output path.

### Generate clustering score report

`clustering_score_report.csv` contains one row with:

- `average_diff_1st-2nd_norm`
- `norm_num_dom_classes`
- `inv_average_num_classes_in_clusters`
- `proportion_objs_in_noise_cluster`
- `average_score`

where,
- `average_diff_1st-2nd_norm` is the average of `diff_1st-2nd_norm` from
`clusters_dominant_classes_and_diff.csv`.
- `norm_num_dom_classes` is the number
of distinct non-`0000` values in `1st_dom_class`, divided by the total number of
different classes present in `clusters_summary_classes.csv`.
- `inv_average_num_classes_in_clusters` is `1 / average_num_classes_in_clusters`,
where `average_num_classes_in_clusters` is the average of
`num_classes_in_cluster` across `clusters_dominant_classes_and_diff.csv`.
- `proportion_objs_in_noise_cluster` is computed as
`(total input images - objects in cluster -1) / total input images`.
- `average_score` is the arithmetic mean of those four normalized values.

The scores have been defined to have values between `0` and `1` only, where `0`
is the worst possible score and `1` is the best possible score.

To regenerate only the score report from an existing
`clusters_dominant_classes_and_diff.csv`:

```bash
python clustering compute-clusters \
  --summary-scores /path/to/file/clusters_dominant_classes_and_diff.csv
```

### Organize clustered outputs into folders

Organize clustered outputs into folders (copies images and matching `.JSON` metadata, 
even if either the image or the JSON file is missing):

```bash
python clustering copy-crops-to-cluster-dirs --clusters-file /path/to/file/clusters.csv \
  --input-dir /path/to/images --dest-dir /path/to/clustered
```

Copy only `.JSON` metadata (leave images in place):

```bash
python clustering copy-crops-to-cluster-dirs --clusters-file /path/to/file/clusters.csv \
  --input-dir /path/to/images --dest-dir /path/to/clustered --json-only
```

Copy high-confidence representatives and outliers into per-cluster subdirectories:

```bash
python clustering copy-crops-to-cluster-dirs --clusters-file /path/to/file/clusters.csv \
  --input-dir /path/to/images --dest-dir /path/to/clustered \
  --subdir-representative 0.99 --subdir-outliers 0.8
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
  `representative_prob_X_outlierscore_0.001` for images with `probabilities >= X` and `outlier_scores <= 0.001`,
  and `outliers_score_Y` for images with `outlier_scores >= Y`. These are skipped when `--json-only` is used.
- `copy-crops-to-subdirs-representative` and `copy-crops-to-subdir-outliers` copy selected crop images only; they do not copy `.JSON` metadata. Their destination subdirectory names include the thresholds used in the command, for example `representative_prob_0.99_outlierscore_0.001` and `outliers_prob_0.3_outlierscore_0.7`.
- Handles destination conflicts with `--on-conflict` (`rename`, `overwrite`, `skip`, or `error`) and supports `--dry-run` for previews.

### Copy representative and outlier crop images into subdirectories, based on probability and outlier score thresholds provided by the user

Create only the representative-image subdirectories:

```bash
python clustering copy-crops-to-subdirs-representative \
  --clusters-file /path/to/file/clusters.csv \
  --input-dir /path/to/crop_image_files/directory \
  --dest-dir /path/to/clustered/directory \
  --probability 0.99 \
  --outlier-score 0.001
```

This command reads `image_id`, `cluster`, `probabilities`, and `outlier_scores`
from `clusters.csv`. For each row where `probabilities >= 0.99` and
`outlier_scores <= 0.001`, it copies the crop image from `--input-dir` into
`<dest-dir>/<cluster>/representative_prob_0.99_outlierscore_0.001/`. For
example, an image in cluster `16` is copied under
`/path/to/clustered/directory/16/representative_prob_0.99_outlierscore_0.001/`.

Create only the outlier-image subdirectories:

```bash
python clustering copy-crops-to-subdir-outliers \
  --clusters-file /path/to/file/clusters.csv \
  --input-dir /path/to/crop_image_files/directory \
  --dest-dir /path/to/clustered/directory \
  --probability 0.3 \
  --outlier-score 0.7
```

This command applies the inverted selection rule. It copies rows where
`probabilities <= 0.3` and `outlier_scores >= 0.7` into
`<dest-dir>/<cluster>/outliers_prob_0.3_outlierscore_0.7/`.

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
`config_files/config_example_run_only_dimreduction_and_clustering.yaml`, shows the
minimal inputs to rerun UMAP + HDBSCAN from cached DINOv2 outputs. A third template,
`config_files/config_example_run_only_clustering.yaml`, shows the minimal inputs to rerun
HDBSCAN using cached UMAP outputs. The pipeline reads YAML configs directly. The
most important fields are:

- `cropped_images_dir` (the older `input_image_dir` key is still accepted)
- `output_dir`
- `dino_model` (optional local clone path)
- `batch_size`
- `num_workers`
- `umap_dim`
- `umap_neighbors` (default `30`)
- `umap_min_dist` (default `0.0`)
- `umap_metric` (default `cosine`)
- `umap_fit_sample_size` (optional low-memory mode for very large datasets)
- `umap_transform_batch_size` (default `100000`; used with `umap_fit_sample_size`)
- `hdbscan_min_cluster_size`
- `hdb_min_samples` (default `10`)
- `hdb_cluster_selection_method` (default `eom`; allowed values: `eom`, `leaf`)
- `hdb_cluster_selection_epsilon` (default `0.0`; larger values merge nearby HDBSCAN clusters)
- `hdb_allow_single_cluster` (default `false`)
- `hdbscan_fit_sample_size` (optional low-memory mode for very large datasets)
- `hdbscan_predict_batch_size` (default `100000`; used with `hdbscan_fit_sample_size`)
- `hdbscan_core_dist_n_jobs` (default `1`; lower values reduce RAM spikes)
- `write_dimreduction_vector` (default `true`, writes the UMAP vector to `clusters.csv`)
- `write_richness_file` (default `false`, writes `richness.csv` after `clusters.csv`)
- `two_pass` or `fast_tune` (recommended: `false`)
- `refine_prob_threshold` (default `0.7`; used only when `two_pass: true`)
- `refine_include_noise` (default `true`; used only when `two_pass: true`)
- `subclustering` (default `true`; runs automatic post-pipeline subclustering)
- `min_for_subclustering` (default `1000`; clusters must be larger than this value)
- `subclustering_umap_dim` / `subclustering_umap_neighbors`
- `subclustering_hdbscan_min_cluster_size` / `subclustering_hdb_min_samples`
- `subclustering_hdb_cluster_selection_method`, `subclustering_hdb_cluster_selection_epsilon`, and `subclustering_hdb_allow_single_cluster` (optional overrides; when omitted, subclustering uses the main HDBSCAN selection settings)
- `merge_noise_subclusters` (default `false`; remaps non-noise subclusters found inside parent cluster `-1` into top-level cluster IDs)
- `autocrop` (default: `false`)
- `background_color` (RGB background color as `[R, G, B]`; default is tuned for blue)
- `autocrop_threshold` (color-distance threshold used to separate background from foreground)
- `size_feature_weight` (default `0.0`; higher values emphasize size)
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
- `umap_fit_sample_size`: Optional low-memory mode for very large datasets. When
  unset, UMAP is fitted on every image embedding, which gives the most direct
  result but can require a large amount of RAM because UMAP must build a
  nearest-neighbor graph over the full dataset. When this value is set and the
  dataset has more rows than the value, the pipeline randomly samples that many
  embeddings, fits UMAP only on the sample, and then transforms all embeddings
  through the fitted reducer in batches. This changes the result: it is an
  approximate projection anchored by the sampled images, not an exact full-data
  UMAP fit. Use it when a full UMAP run is killed by memory pressure, swap usage,
  or the operating system after the cached DINOv2 files have already been
  produced. Leave it unset for small and medium datasets that fit comfortably in
  RAM, for final benchmark runs where exact full-data UMAP is required, or when
  you need maximum sensitivity to rare small groups.

  Choose the value as the largest sample that fits comfortably in memory and
  still represents the dataset's diversity. Practical starting points are
  `100000-250000` on constrained machines, `500000` on a 64 GB machine, and
  `1000000` or more only when there is enough RAM and runtime budget. For a
  15 million image dataset on 64 GB RAM, use the dedicated subsection below,
  which starts more conservatively. If the run succeeds but visually important
  rare clusters are missing or over-merged, increase the sample size if memory
  allows. The sample is deterministic (`random_state=42`), so repeated runs with
  the same inputs and settings should select the same rows.
- `umap_transform_batch_size`: Number of full-dataset rows transformed at a time
  after UMAP has been fitted on `umap_fit_sample_size`. This setting does not
  control how much data UMAP uses to learn the manifold; it controls the size of
  each transform batch written to `umap.npy`. Larger batches can be faster
  because there is less Python overhead, but they hold more temporary `float32`
  data in RAM. Smaller batches reduce peak memory at the cost of more batches and
  longer runtime. The default is `100000`, which is a reasonable starting point
  for large runs. If memory is still tight during the transform phase, reduce it
  to `50000`, `25000`, or `10000`. If memory is stable and the transform phase is
  slow, increase it to `200000` or `500000`.
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
- `hdb_cluster_selection_method`: HDBSCAN cluster selection method. `eom` is
  the default and usually gives broader, more stable clusters. `leaf` can expose
  finer-grained clusters but often increases the number of points labeled as
  noise.
- `hdb_cluster_selection_epsilon`: Distance threshold for merging nearby
  clusters in HDBSCAN's condensed tree. Keep this at `0.0` by default. Small
  values such as `0.05-0.1` can reduce noise by merging nearby clusters, but
  larger values can over-merge visually distinct groups.
- `hdb_allow_single_cluster`: Allows HDBSCAN to return a single non-noise
  cluster when the density tree supports it. Leave this `false` for normal
  discovery runs unless you expect one dominant group.
- `hdbscan_fit_sample_size`: Optional low-memory mode for the HDBSCAN fitting
  stage. When unset, HDBSCAN is fitted on every row in `umap.npy`. That is the
  standard behavior: every reduced vector contributes to the density tree that
  defines the final clusters. It is also the most memory-hungry behavior,
  especially when `umap.npy` contains millions of rows.

  When `hdbscan_fit_sample_size` is set and the UMAP output has more rows than
  that value, the pipeline uses a different two-step strategy. First, it selects
  a deterministic random sample of `hdbscan_fit_sample_size` rows from
  `umap.npy` and fits HDBSCAN only on that sample. Second, it uses HDBSCAN's
  approximate prediction logic to assign every row in the full dataset to the
  clusters learned from the sample. The final `clusters.csv` still contains one
  row per image, but the cluster structure is learned from the sample rather
  than from all images at once.

  This setting reduces peak RAM because the expensive HDBSCAN fit sees only the
  sampled rows. It is useful when UMAP has completed successfully but HDBSCAN is
  killed by the operating system, grows into swap, or cannot finish on the
  available RAM. It is especially useful in resume modes such as
  `compute: only-clustering` or `compute: only-dimreduction-and-clustering`,
  where the cached `umap.npy` or DINOv2 artifacts already exist and the goal is
  to finish clustering without recomputing earlier stages.

  The tradeoff is that this is approximate clustering. Clusters that are absent
  from the sample, weakly represented in the sample, or much smaller than the
  sample's resolution may be missed, merged into a nearby sampled cluster, or
  assigned as noise. Larger, common, well-separated groups are more likely to be
  preserved. Use a larger sample when rare clusters matter, when clusters have
  subtle boundaries, or when the first sampled run produces too much noise or
  overly broad clusters. Use a smaller sample when the fit still exceeds memory
  or when you mainly need a coarse first pass over a massive dataset.

  Start with the largest value that fits comfortably in memory. Practical
  starting points are `100000-250000` on constrained machines and `500000` on a
  machine with more memory headroom. For a 15 million image dataset on 64 GB RAM,
  use the dedicated subsection below, which starts more conservatively. If the
  result is too coarse, contains too much noise, or misses expected smaller
  clusters, increase the sample size if memory allows. The value must be larger
  than `hdbscan_min_cluster_size`; in practice, it should be much larger than
  the smallest cluster you want HDBSCAN to learn.

  This setting works together with `hdbscan_predict_batch_size`. The fit sample
  size controls how many rows are used to learn clusters and therefore affects
  clustering quality, memory, and runtime. The prediction batch size controls how
  many full-dataset rows are assigned at a time after fitting and mostly affects
  memory and speed during the assignment phase.
- `hdbscan_predict_batch_size`: Number of UMAP rows assigned at a time when
  `hdbscan_fit_sample_size` is enabled. It is used only for the approximate
  assignment phase after HDBSCAN has been fitted on the sample. It does not
  change cluster structure directly; it changes peak memory and runtime while
  applying `approximate_predict` to the full dataset. The default is `100000`.
  Lower it to `50000`, `25000`, or `10000` if the prediction phase still causes
  RAM spikes. Raise it to `200000` or more if memory is stable and prediction is
  dominated by batch overhead.
- `hdbscan_core_dist_n_jobs`: Number of worker threads HDBSCAN may use while
  computing core distances during fitting. Higher values can speed up HDBSCAN,
  but each worker can increase peak memory. The default is `1` because large
  datasets are usually memory-bound rather than CPU-bound. Keep it at `1` on
  64 GB machines or whenever the process is killed unexpectedly. Increase it
  gradually, for example to `2` or `4`, only after confirming that the fit has
  plenty of unused RAM. Avoid setting it to the full CPU count on very large
  datasets unless the machine has correspondingly large memory headroom.

For very large cached reruns, a conservative starting configuration is:

```yaml
compute: "only-dimreduction-and-clustering"
umap_fit_sample_size: 500000
umap_transform_batch_size: 100000
hdbscan_fit_sample_size: 500000
hdbscan_predict_batch_size: 100000
hdbscan_core_dist_n_jobs: 1
write_dimreduction_vector: false
subclustering: false
```

If this still exceeds available memory, reduce the two sample sizes first, then
reduce the batch sizes. If it succeeds and the results are too coarse, increase
the sample sizes before increasing batch sizes. Batch sizes mostly affect memory
and speed; sample sizes affect both memory and clustering quality.

#### Suggested settings for 15 million images on a 64 GB RAM computer

This subsection is specifically for the scenario where the dataset has about
15 million images, the machine has 64 GB of RAM, and the DINOv2 artifacts have
already been computed. In that case, use
`compute: "only-dimreduction-and-clustering"` so the pipeline reuses the existing
`sizes.npy`, `images.txt`, `embeddings.json`, and `embeddings.dat` files and runs
only UMAP plus HDBSCAN.

Start with this conservative configuration:

```yaml
compute: "only-dimreduction-and-clustering"

dino_files: "/path/to/existing/dino_artifacts"
output_dir: "/path/to/output/umap_hdbscan_15m_lowmem"

size_feature_weight: 0.0

umap_dim: 30
umap_neighbors: 30
umap_min_dist: 0.0
umap_metric: "cosine"

umap_fit_sample_size: 250000
umap_transform_batch_size: 50000

hdbscan_min_cluster_size: 500
hdb_min_samples: 50
hdb_metric: "euclidean"
hdb_cluster_selection_method: "eom"
hdb_cluster_selection_epsilon: 0.0
hdb_allow_single_cluster: false

hdbscan_fit_sample_size: 250000
hdbscan_predict_batch_size: 50000
hdbscan_core_dist_n_jobs: 1

write_dimreduction_vector: false
write_richness_file: false

subclustering: false
merge_noise_subclusters: false
```

The first run should favor finishing successfully over preserving every possible
fine-grained cluster. Starting both `umap_fit_sample_size` and
`hdbscan_fit_sample_size` at `250000` keeps the expensive fitting steps smaller
than a `500000` sample, which is safer on a 64 GB machine because UMAP and
HDBSCAN allocate internal structures beyond the raw arrays. If the run succeeds
and the result is too coarse, rerun with:

```yaml
umap_fit_sample_size: 500000
hdbscan_fit_sample_size: 500000
```

Keep the batch sizes at `50000` for the first attempt:

```yaml
umap_transform_batch_size: 50000
hdbscan_predict_batch_size: 50000
```

These batch sizes mostly affect memory spikes and speed. They do not usually
improve clustering quality. If RAM usage is stable and well below the machine's
limit, increase them to `100000` to reduce batch overhead. If the process is
still killed during transform or prediction, reduce them to `25000` or `10000`.

Keep `hdbscan_core_dist_n_jobs: 1` for this machine. HDBSCAN may run faster with
more workers, but more workers can also create RAM spikes. On a 64 GB computer
with 15 million rows, HDBSCAN is usually memory-bound, so increasing workers is
risky until a single-worker run has completed and memory usage has been observed.

Set `write_dimreduction_vector: false` for this scale. Writing the UMAP vector
into every `clusters.csv` row creates a very large CSV and adds substantial I/O.
The numeric UMAP vectors are already stored in `umap.npy`, so the CSV column is
usually unnecessary for a 15 million image run.

Set `subclustering: false` for the first top-level run. Automatic subclustering
can launch additional UMAP and HDBSCAN jobs on large clusters, which may exceed
64 GB even if the top-level clustering succeeds. After the top-level run
finishes, inspect `clusters.csv` and run targeted subclustering later on smaller
subsets if needed.

The suggested `hdbscan_min_cluster_size: 500` and `hdb_min_samples: 50` are more
conservative than the template defaults. With 15 million images,
`hdbscan_min_cluster_size: 25` is likely to produce many tiny clusters and more
unstable density structure. If you need finer clusters after the first
successful run, try `hdbscan_min_cluster_size: 250` with `hdb_min_samples: 25`,
or `hdbscan_min_cluster_size: 100` with `hdb_min_samples: 10`.

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

### Automatic subclustering

Automatic subclustering runs after the final `clusters.csv` has been written and
summarized unless `--no-subclustering` or `subclustering: false` is set. It
checks every final cluster label, including noise label `-1`, and selects labels
whose object count is greater than `min_for_subclustering`.

Subclustering has its own configurable UMAP and HDBSCAN parameters:

- `subclustering_umap_dim` (default `60`)
- `subclustering_umap_neighbors` (default `30`)
- `subclustering_hdbscan_min_cluster_size` (default `7`)
- `subclustering_hdb_min_samples` (default `6`)
- `subclustering_hdb_cluster_selection_method` (optional; defaults to the main `hdb_cluster_selection_method`)
- `subclustering_hdb_cluster_selection_epsilon` (optional; defaults to the main `hdb_cluster_selection_epsilon`)
- `subclustering_hdb_allow_single_cluster` (optional; defaults to the main `hdb_allow_single_cluster`)

For each selected parent cluster, the pipeline creates
`output_dir/subclusters/cluster_<label>/` and writes:

- subset `images.txt`
- subset `embeddings.dat`, `embeddings.json`, and `sizes.npy` copied from the cached parent DINOv2 outputs
- `parent_umap.npy`, containing the selected rows from the parent UMAP output for traceability
- a fresh subset `umap.npy` computed with the subclustering UMAP settings
- subset `clusters.csv` computed with the subclustering HDBSCAN settings
- optional `richness.csv` computed from that subset's location and cluster assignments
- the usual summary CSV files for that subset

The step does not re-run DINOv2 embedding. It slices the cached embedding matrix
and size array, then runs only the subset UMAP and HDBSCAN stages needed for the
large parent cluster. A top-level `output_dir/subclusters/subclusters_summary.csv`
lists each parent cluster that was subclustered, the number of non-noise
subclusters, the number of subset images still labeled as noise, and the path to
its subset `clusters.csv`.

When `merge_noise_subclusters: true`, the pipeline uses
`subclusters/cluster_-1/clusters.csv` to rewrite the top-level `clusters.csv`.
Every non-noise subcluster found inside parent cluster `-1` is assigned a fresh
top-level cluster ID after the current maximum cluster ID. Rows merged this way
receive `parent_cluster=-1` and `subcluster=<subset label>` traceability columns,
and their HDBSCAN probability/outlier metadata is copied from the subclustering
run. Any images that remain `-1` in `subclusters/cluster_-1/clusters.csv` stay
`-1` in the top-level output. After merging, all summary CSV files are regenerated
from the rewritten top-level `clusters.csv`.

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

- `clusters.csv` with columns `[image_id, cluster, labeled, probabilities, outlier_scores, dim_reduction]`
  (noise is `-1`; `dim_reduction` is a JSON array of
  UMAP values, length = `umap_dim` unless `write_dimreduction_vector: false`,
  in which case the column is empty). `labeled` is `True` only when the basename in
  `image_id` ends exactly with `_class_1234.jpg`, meaning `_class_` appears immediately
  before a 4-digit class value and that value is immediately followed by the `.jpg`
  extension. Otherwise `labeled` is `False`. When `merge_noise_subclusters: true`,
  the file also includes `parent_cluster` and `subcluster` columns for rows that
  were recovered from parent cluster `-1`.
- `clusters_summary.csv` with columns `[cluster_id, num_objs_in_cluster, num_classes_in_cluster]`
- Optional `richness.csv` with one row per location ID and columns `[location, cluster_<ID>, ..., richness]`.
  A location ID is the filename portion before the first `_r<digits>c<digits>` marker,
  so both `A09-G_r4c4_obj_328626_class_4200.jpg` and
  `A09-G_r11c3_obj_96_class_run4.jpg` belong to location `A09-G`. Filenames
  without that marker, or with no prefix before it, are skipped and written to
  `location_IDs_not_found.csv`. Each cluster column counts images from that
  location assigned to that cluster; `richness` counts how many of those cluster
  columns are nonzero. Cluster columns are ordered numerically and location rows
  retain their order of first appearance in `clusters.csv`. Set
  `write_richness_file: true` or pass `--write-richness-file` to write this file
  automatically after `clusters.csv`, including for automatic subcluster outputs.
  It can also be generated independently with
  `python clustering write_richness_file --clusters-file /path/to/clusters.csv --output-dir /path/to/output`.
- `crop_filename_not_found_in_embedding.csv` when cached reruns or subset runs
  skip image names that are listed in `images.txt` or `subset_images.txt` but do
  not have a matching cached embedding row. This can happen when names are
  misspelled, contain different Unicode normalization, or include invalid
  characters compared with the cached feature index.
- `classes_in_dataset.csv` with columns `[class_ID, num_objs]` by default, or `[class_ID, class_name, num_objs]` when `python count-classes-on-labeled-filenames ... --biigleID-to-names-file /path/to/labels.csv` is used after recursively scanning the labeled image directory
- `clusters_summary_classes.csv` with columns `[cluster_id, num_objs_in_cluster, num_classes_in_cluster, class_X, class_X_%_of_the_cluster, ...]` — one row per cluster, one set of columns per valid class found across the dataset. A class is recognized only when the basename ends with `_class_1234.jpg`; non-matching filenames still contribute to `num_objs_in_cluster` but are excluded from class-derived columns. Add `class_X_%_of_total_class` columns by supplying `--classes-benchmark-file`.
- `clusters_dominant_classes_and_diff.csv` with columns `[cluster_id, num_objs_in_cluster, num_classes_in_cluster, 1st_dom_class, 1st_dom_%, 1st_dom_num_objs, 2nd_dom_class, 2nd_dom_%, 2nd_dom_num_objs, diff_1st-2nd_%, diff_1st-2nd_norm]`, derived from the `num_objs_in_cluster`, `num_classes_in_cluster`, `class_X`, and `class_X_%_of_the_cluster` columns in `clusters_summary_classes.csv`
- `clustering_score_report.csv` with columns `[average_diff_1st-2nd_norm, norm_num_dom_classes, inv_average_num_classes_in_clusters, proportion_objs_in_noise_cluster, average_score]`, derived from `clusters_dominant_classes_and_diff.csv` together with the sibling `clusters_summary_classes.csv`
- `embeddings.dat` and `embeddings.json` (embedding matrix + metadata)
- `umap.npy` (UMAP-reduced vectors)
- `images.txt` (stable list of image paths used)
- `subclusters/` when automatic subclustering finds oversized final clusters.
  Each `subclusters/cluster_<label>/` directory contains subset cached artifacts,
  `parent_umap.npy`, a fresh subset `umap.npy`, subset `clusters.csv`, and the
  usual summary CSVs.

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

`refine_prob_threshold` controls which pass-1 samples are treated as uncertain.
HDBSCAN reports a membership probability in the range 0-1 for each sample; lower
values mean weaker confidence that the sample belongs to its assigned cluster.
During two-pass mode, any sample with probability below `refine_prob_threshold`
is sent to pass 2 for refinement.

`refine_include_noise` controls whether pass-1 noise assignments are always
refined. When `refine_include_noise: true`, samples assigned to noise
(`cluster == -1`) are also sent to pass 2 regardless of probability. When it is
`false`, only the probability threshold decides which samples are refined.

Both refinement settings are valid config keys in all three example YAML files so
you can keep a consistent config schema across run modes. They are only used by
full runs with `two_pass: true`; they are ignored when `two_pass: false`,
`fast_tune: true`, or when using the `only-dimreduction-and-clustering` /
`only-clustering` compute modes.

Use the default `refine_prob_threshold: 0.7` as a balanced starting point. Lower
values such as `0.4-0.6` refine fewer samples and run faster, but can leave
borderline assignments from pass 1 unchanged. Higher values such as `0.8-0.9`
refine more samples and can improve conservative clustering, but pass 2 takes
longer and may include many already-reasonable assignments. Avoid values near
`0.0` unless you only want to refine noise points, and avoid values near `1.0`
unless you intentionally want almost every non-perfect pass-1 assignment to be
rerun.

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
