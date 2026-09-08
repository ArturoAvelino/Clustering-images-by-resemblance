from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import replace
import json
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

from .algorithms import HDBSCANClusterer, UMAPReducer
from .config import ClusterResult, PipelineConfig, StagePaths, make_fast_config
from .data import ImageDataset, ImageIndex, compute_size_features
from .embedding import DINOv2Embedder, ensure_deps, resolve_device
from .model_repo import auto_model_repo
from .richness import write_richness_csv
from .ssl_utils import configure_ssl
from .subset import prepare_subset_cache
from .summary import (
    summarize_classes_in_clusters_csv,
    summarize_cluster_dominant_classes_and_diff_csv,
    summarize_clusters_csv,
)


def summarize_cluster_outputs(clusters_csv_path: Path, benchmark_path: Path | None = None) -> None:
    """Write all summary CSVs, including richness.csv, beside clusters.csv.

    This is used for both the final top-level cluster assignments and every
    automatically generated ``subclusters/cluster_<label>/clusters.csv`` file.
    """
    summarize_clusters_csv(clusters_csv_path)
    summary_classes_path = summarize_classes_in_clusters_csv(clusters_csv_path, benchmark_path)
    summarize_cluster_dominant_classes_and_diff_csv(summary_classes_path)
    write_richness_csv(clusters_csv_path)


def stage_dir(cfg: PipelineConfig, stage: str) -> Path:
    if cfg.two_pass or cfg.fast_tune:
        return cfg.output_dir / "stages" / stage
    return cfg.output_dir


def stage_paths(base_dir: Path) -> StagePaths:
    return StagePaths(
        index_path=base_dir / "images.txt",
        emb_path=base_dir / "embeddings.dat",
        meta_path=base_dir / "embeddings.json",
        size_path=base_dir / "sizes.npy",
        umap_path=base_dir / "umap.npy",
        csv_path=base_dir / "clusters.csv",
    )


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, rem = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(rem)}m"


def _log_timing(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def _read_image_index(index_path: Path) -> list[str]:
    with index_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_image_index(index_path: Path, rel_paths: Iterable[str]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for rel in rel_paths:
            f.write(rel + "\n")


def _required_subclustering_artifacts(paths: StagePaths) -> dict[str, Path]:
    return {
        "images.txt": paths.index_path,
        "embeddings.dat": paths.emb_path,
        "embeddings.json": paths.meta_path,
        "sizes.npy": paths.size_path,
        "umap.npy": paths.umap_path,
    }


def _load_cluster_members(clusters_csv_path: Path) -> dict[int, list[str]]:
    """Read final cluster assignments as cluster label -> image paths."""
    members: dict[int, list[str]] = defaultdict(list)
    with clusters_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{clusters_csv_path} has no header row")
        missing = {"image_id", "cluster"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{clusters_csv_path} must have columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            rel = row.get("image_id")
            label = row.get("cluster")
            if not rel or label is None:
                continue
            members[int(float(label))].append(rel)
    return dict(members)


def _copy_subset_artifacts(
    source_paths: StagePaths,
    output_paths: StagePaths,
    subset_indices: list[int],
    subset_rel_paths: list[str],
) -> bool:
    """Write cached DINOv2 artifacts for a cluster subset without re-embedding images."""
    previous_rel_paths = (
        _read_image_index(output_paths.index_path)
        if output_paths.index_path.exists()
        else None
    )
    subset_changed = previous_rel_paths != subset_rel_paths
    output_paths.index_path.parent.mkdir(parents=True, exist_ok=True)
    _write_image_index(output_paths.index_path, subset_rel_paths)

    with source_paths.meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["num_images"])
    dim = int(meta["embed_dim"])
    dtype = np.float16 if meta["dtype"] == "float16" else np.float32
    if max(subset_indices, default=-1) >= n:
        raise ValueError("Subclustering subset index exceeds cached embedding count.")

    src_emb = np.memmap(source_paths.emb_path, mode="r", dtype=dtype, shape=(n, dim))
    dst_emb = np.memmap(
        output_paths.emb_path,
        mode="w+",
        dtype=dtype,
        shape=(len(subset_indices), dim),
    )
    chunk_size = 5000
    for start in range(0, len(subset_indices), chunk_size):
        stop = start + chunk_size
        dst_emb[start:stop] = src_emb[subset_indices[start:stop]]
    dst_emb.flush()

    sub_meta = dict(meta)
    sub_meta["num_images"] = len(subset_indices)
    with output_paths.meta_path.open("w", encoding="utf-8") as f:
        json.dump(sub_meta, f, indent=2)

    sizes = np.load(source_paths.size_path, mmap_mode="r")
    if sizes.shape[0] != n:
        raise ValueError(
            f"Size feature count ({sizes.shape[0]}) does not match embeddings ({n})."
        )
    np.save(output_paths.size_path, np.asarray(sizes[subset_indices]))

    parent_umap = np.load(source_paths.umap_path, mmap_mode="r")
    if parent_umap.ndim != 2 or parent_umap.shape[0] != n:
        raise ValueError(
            "Parent umap.npy must be 2D and match the cached embedding count."
        )
    np.save(
        output_paths.index_path.parent / "parent_umap.npy",
        np.asarray(parent_umap[subset_indices]),
    )
    return subset_changed


def _subclustering_config(cfg: PipelineConfig, output_dir: Path) -> PipelineConfig:
    """Return a config using the automatic subclustering parameters."""
    return replace(
        cfg,
        output_dir=output_dir,
        compute="full",
        subset_images=None,
        two_pass=False,
        fast_tune=False,
        max_images=None,
        subclustering=False,
        merge_noise_subclusters=False,
        umap_dim=cfg.subclustering_umap_dim,
        umap_neighbors=cfg.subclustering_umap_neighbors,
        hdbscan_min_cluster_size=cfg.subclustering_hdbscan_min_cluster_size,
        hdb_min_samples=cfg.subclustering_hdb_min_samples,
        hdb_cluster_selection_method=(
            cfg.subclustering_hdb_cluster_selection_method
            or cfg.hdb_cluster_selection_method
        ),
        hdb_cluster_selection_epsilon=(
            cfg.subclustering_hdb_cluster_selection_epsilon
            if cfg.subclustering_hdb_cluster_selection_epsilon is not None
            else cfg.hdb_cluster_selection_epsilon
        ),
        hdb_allow_single_cluster=(
            cfg.subclustering_hdb_allow_single_cluster
            if cfg.subclustering_hdb_allow_single_cluster is not None
            else cfg.hdb_allow_single_cluster
        ),
    )


def _umap_matches(path: Path, expected_rows: int, expected_cols: int) -> bool:
    if not path.exists():
        return False
    data = np.load(path, mmap_mode="r")
    return data.ndim == 2 and data.shape == (expected_rows, expected_cols)


def run_auto_subclustering(
    cfg: PipelineConfig,
    clusters_csv_path: Path,
    source_paths: StagePaths | None,
    log_path: Path | None = None,
) -> bool:
    """Subcluster final clusters larger than ``cfg.min_for_subclustering``.

    The step reuses cached DINOv2 artifacts by slicing `images.txt`,
    `embeddings.dat`, `embeddings.json`, and `sizes.npy` for each large cluster.
    It also saves the corresponding rows from the parent `umap.npy` as
    `parent_umap.npy` for traceability, then computes a fresh subset UMAP and
    HDBSCAN labels using the configurable subclustering parameters.

    When ``cfg.merge_noise_subclusters`` is true, non-noise subclusters created
    from parent cluster ``-1`` are remapped into fresh top-level cluster IDs in
    ``clusters_csv_path``. The rewritten CSV keeps row order and adds
    ``parent_cluster`` and ``subcluster`` traceability columns.
    """
    if not cfg.subclustering:
        return False
    if source_paths is None:
        msg = "[subclustering] Skipped: cached DINOv2 artifacts are unavailable."
        print(msg)
        if log_path is not None:
            _log_timing(log_path, msg)
        return False

    missing = [
        name
        for name, path in _required_subclustering_artifacts(source_paths).items()
        if not path.exists()
    ]
    if missing:
        msg = (
            "[subclustering] Skipped: missing cached artifacts "
            + ", ".join(missing)
            + f" in {source_paths.index_path.parent}"
        )
        print(msg)
        if log_path is not None:
            _log_timing(log_path, msg)
        return False

    members = _load_cluster_members(clusters_csv_path)
    oversized = {
        label: rels
        for label, rels in members.items()
        if len(rels) > cfg.min_for_subclustering
    }
    if not oversized:
        msg = (
            "[subclustering] No clusters exceed "
            f"{cfg.min_for_subclustering} objects."
        )
        print(msg)
        if log_path is not None:
            _log_timing(log_path, msg)
        return False

    source_rel_paths = _read_image_index(source_paths.index_path)
    source_index = {rel: idx for idx, rel in enumerate(source_rel_paths)}
    summary_rows: list[list[str | int]] = []
    noise_subcluster_csv: Path | None = None
    sub_root = cfg.output_dir / "subclusters"
    sub_root.mkdir(parents=True, exist_ok=True)
    print(
        "[subclustering] Running automatic subclustering for "
        f"{len(oversized)} cluster(s) larger than {cfg.min_for_subclustering}."
    )
    for label in sorted(oversized):
        subset_rel_paths = oversized[label]
        missing_rels = [rel for rel in subset_rel_paths if rel not in source_index]
        if missing_rels:
            raise ValueError(
                "Final clusters.csv contains image paths not present in cached "
                f"{source_paths.index_path}: {missing_rels[:3]}"
            )
        subset_indices = [source_index[rel] for rel in subset_rel_paths]
        sub_dir = sub_root / f"cluster_{label}"
        sub_paths = stage_paths(sub_dir)
        sub_cfg = _subclustering_config(cfg, sub_dir)
        subset_changed = _copy_subset_artifacts(
            source_paths, sub_paths, subset_indices, subset_rel_paths
        )

        t0 = time.perf_counter()
        if cfg.force or subset_changed or not _umap_matches(
            sub_paths.umap_path, len(subset_rel_paths), sub_cfg.umap_dim
        ):
            UMAPReducer(sub_cfg).reduce(
                sub_paths.emb_path,
                sub_paths.meta_path,
                sub_paths.size_path,
                sub_paths.umap_path,
            )
            umap_skipped = False
        else:
            umap_skipped = True
        umap_msg = (
            f"[subclustering cluster={label}] UMAP reduction: "
            f"{_format_duration(time.perf_counter() - t0)}"
            + (" (skipped)" if umap_skipped else "")
        )
        print(umap_msg)
        if log_path is not None:
            _log_timing(log_path, umap_msg)

        t0 = time.perf_counter()
        clusterer = HDBSCANClusterer(sub_cfg)
        result = clusterer.fit(sub_paths.umap_path)
        hdb_msg = (
            f"[subclustering cluster={label}] HDBSCAN: "
            f"{_format_duration(time.perf_counter() - t0)}"
        )
        print(hdb_msg)
        if log_path is not None:
            _log_timing(log_path, hdb_msg)

        dim_reduction = (
            np.load(sub_paths.umap_path) if cfg.write_dimreduction_vector else None
        )
        clusterer.write_csv(
            sub_paths.csv_path,
            subset_rel_paths,
            result.labels,
            result.probabilities,
            result.outlier_scores,
            result.exemplars,
            dim_reduction,
        )
        summarize_cluster_outputs(sub_paths.csv_path, cfg.classes_benchmark_file)
        sub_counts = Counter(int(x) for x in result.labels)
        sub_noise = int(sub_counts.get(-1, 0))
        if label == -1:
            noise_subcluster_csv = sub_paths.csv_path
        summary_rows.append(
            [
                label,
                len(subset_rel_paths),
                len([x for x in sub_counts if x != -1]),
                sub_noise,
                str(sub_paths.csv_path),
            ]
        )

    summary_path = sub_root / "subclusters_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "parent_cluster",
                "num_parent_objects",
                "num_subclusters",
                "num_noise_in_subclusters",
                "clusters_csv",
            ]
        )
        writer.writerows(summary_rows)
    if cfg.merge_noise_subclusters and noise_subcluster_csv is not None:
        return _merge_noise_subclusters_into_csv(clusters_csv_path, noise_subcluster_csv)
    return False


def _merge_noise_subclusters_into_csv(
    clusters_csv_path: Path, noise_subcluster_csv: Path
) -> bool:
    """Merge non-noise labels from parent noise subclustering into final CSV."""
    with clusters_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{clusters_csv_path} has no header row")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        return False

    max_label = max(int(float(row["cluster"])) for row in rows)
    next_label = max_label + 1 if max_label >= 0 else 0

    with noise_subcluster_csv.open("r", encoding="utf-8", newline="") as f:
        sub_reader = csv.DictReader(f)
        if sub_reader.fieldnames is None:
            raise ValueError(f"{noise_subcluster_csv} has no header row")
        sub_rows = {row["image_id"]: row for row in sub_reader}

    sub_label_map: dict[int, int] = {}
    for sub_row in sub_rows.values():
        sub_label = int(float(sub_row["cluster"]))
        if sub_label == -1 or sub_label in sub_label_map:
            continue
        sub_label_map[sub_label] = next_label
        next_label += 1
    if not sub_label_map:
        return False

    for extra in ["parent_cluster", "subcluster"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    changed = False
    for row in rows:
        row.setdefault("parent_cluster", "")
        row.setdefault("subcluster", "")
        if int(float(row["cluster"])) != -1:
            continue
        sub_row = sub_rows.get(row["image_id"])
        if sub_row is None:
            continue
        sub_label = int(float(sub_row["cluster"]))
        row["parent_cluster"] = "-1"
        row["subcluster"] = str(sub_label)
        if sub_label == -1:
            continue
        row["cluster"] = str(sub_label_map[sub_label])
        for key in ["probabilities", "outlier_scores", "dim_reduction"]:
            if key in row and key in sub_row:
                row[key] = sub_row[key]
        changed = True
    if not changed:
        return False

    tmp_path = clusters_csv_path.with_suffix(clusters_csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(clusters_csv_path)
    return True


def _finish_pipeline_run(
    cfg: PipelineConfig,
    final_csv: Path,
    source_paths: StagePaths | None,
    log_path: Path,
    total_start: float,
) -> Path:
    """Finalize top-level and nested summaries from the final cluster labels."""
    summarize_cluster_outputs(final_csv, cfg.classes_benchmark_file)
    merged = run_auto_subclustering(cfg, final_csv, source_paths, log_path)
    if merged:
        msg = "[subclustering] Merged non-noise subclusters from parent cluster -1."
        print(msg)
        _log_timing(log_path, msg)
        summarize_cluster_outputs(final_csv, cfg.classes_benchmark_file)
    total_dt = time.perf_counter() - total_start
    total_msg = f"[total] Pipeline runtime: {_format_duration(total_dt)}"
    print(total_msg)
    _log_timing(log_path, total_msg)
    return final_csv


def run_stage(
    cfg: PipelineConfig,
    device: str,
    rel_paths: List[str],
    base_dir: Path,
    write_csv: bool = True,
    stage_label: str = "pipeline",
    log_path: Path | None = None,
) -> tuple[StagePaths, ClusterResult]:
    paths = stage_paths(base_dir)
    index = ImageIndex(cfg.input_dir, paths.index_path)
    index.write(rel_paths)

    total_steps = 4 + (1 if write_csv else 0)
    with tqdm(
        total=total_steps, desc=f"Stages[{stage_label}]", unit="step", position=0
    ) as stage_bar:
        t0 = time.perf_counter()
        if cfg.force or not paths.size_path.exists():
            sizes = compute_size_features(
                cfg.input_dir, rel_paths, cfg.background_color, cfg.autocrop_threshold
            )
            paths.size_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(paths.size_path, sizes)
            size_skipped = False
        else:
            print(f"Size features exist in {paths.size_path}; skipping size step.")
            size_skipped = True
        size_dt = time.perf_counter() - t0
        size_msg = (
            f"[{stage_label}] Size features: {_format_duration(size_dt)}"
            + (" (skipped)" if size_skipped else "")
        )
        print(size_msg)
        if log_path is not None:
            _log_timing(log_path, size_msg)
        stage_bar.update(1)

        t0 = time.perf_counter()
        if cfg.force or not paths.emb_path.exists() or not paths.meta_path.exists():
            embedder = DINOv2Embedder(cfg, device)
            dataset = ImageDataset(cfg.input_dir, rel_paths, embedder.transform)
            embedder.embed(dataset, paths.emb_path, paths.meta_path)
            emb_skipped = False
        else:
            print(f"Embeddings exist in {paths.emb_path}; skipping embedding step.")
            emb_skipped = True
        emb_dt = time.perf_counter() - t0
        emb_msg = (
            f"[{stage_label}] DINO embedding: {_format_duration(emb_dt)}"
            + (" (skipped)" if emb_skipped else "")
        )
        print(emb_msg)
        if log_path is not None:
            _log_timing(log_path, emb_msg)
        stage_bar.update(1)

        t0 = time.perf_counter()
        if cfg.force or not paths.umap_path.exists():
            reducer = UMAPReducer(cfg)
            reducer.reduce(paths.emb_path, paths.meta_path, paths.size_path, paths.umap_path)
            umap_skipped = False
        else:
            print(f"UMAP output exists in {paths.umap_path}; skipping reduction step.")
            umap_skipped = True
        umap_dt = time.perf_counter() - t0
        umap_msg = (
            f"[{stage_label}] UMAP reduction: {_format_duration(umap_dt)}"
            + (" (skipped)" if umap_skipped else "")
        )
        print(umap_msg)
        if log_path is not None:
            _log_timing(log_path, umap_msg)
        stage_bar.update(1)

        t0 = time.perf_counter()
        clusterer = HDBSCANClusterer(cfg)
        result = clusterer.fit(paths.umap_path)
        hdb_dt = time.perf_counter() - t0
        hdb_msg = f"[{stage_label}] HDBSCAN: {_format_duration(hdb_dt)}"
        print(hdb_msg)
        if log_path is not None:
            _log_timing(log_path, hdb_msg)
        stage_bar.update(1)

        if write_csv:
            t0 = time.perf_counter()
            dim_reduction = (
                np.load(paths.umap_path) if cfg.write_dimreduction_vector else None
            )
            clusterer.write_csv(
                paths.csv_path,
                rel_paths,
                result.labels,
                result.probabilities,
                result.outlier_scores,
                result.exemplars,
                dim_reduction,
            )
            csv_dt = time.perf_counter() - t0
            csv_msg = f"[{stage_label}] CSV write: {_format_duration(csv_dt)}"
            print(csv_msg)
            if log_path is not None:
                _log_timing(log_path, csv_msg)
            stage_bar.update(1)
    return paths, result


def select_uncertain(
    result: "ClusterResult", threshold: float, include_noise: bool = True
) -> np.ndarray:
    """Return pass-1 sample indices that should be reclustered in pass 2.

    Samples are selected when their HDBSCAN membership probability is below
    ``threshold``. If ``include_noise`` is true, samples labeled as noise
    (cluster ``-1``) are also selected regardless of probability.
    """
    labels = result.labels
    mask = np.zeros(labels.shape, dtype=bool)
    if include_noise:
        mask |= labels == -1
    if result.probabilities is not None:
        mask |= result.probabilities < threshold
    return np.flatnonzero(mask)


def merge_labels(
    base_labels: np.ndarray,
    subset_indices: np.ndarray,
    subset_labels: np.ndarray,
) -> np.ndarray:
    merged = base_labels.copy()
    positive = base_labels[base_labels >= 0]
    offset = int(positive.max()) + 1 if positive.size else 0
    remapped = np.where(subset_labels >= 0, subset_labels + offset, -1)
    merged[subset_indices] = remapped
    return merged


def merge_optional_array(
    base: np.ndarray | None,
    subset_indices: np.ndarray,
    subset: np.ndarray | None,
    length: int,
    fill_value,
) -> np.ndarray | None:
    if base is None and subset is None:
        return None
    if base is None:
        dtype = subset.dtype if subset is not None else type(fill_value)
        merged = np.full(length, fill_value, dtype=dtype)
    else:
        merged = np.array(base, copy=True)
        if merged.shape[0] != length:
            raise ValueError("Base array length does not match labels length.")
    if subset is not None:
        merged[subset_indices] = subset
    return merged


def merge_dim_reduction(
    base: np.ndarray,
    subset_indices: np.ndarray,
    subset: np.ndarray,
    length: int,
) -> List[List[float]]:
    if base.ndim != 2 or subset.ndim != 2:
        raise ValueError("dim_reduction arrays must be 2D.")
    if base.shape[0] != length:
        raise ValueError("Base dim_reduction length does not match labels length.")
    if subset.shape[0] != subset_indices.shape[0]:
        raise ValueError("Subset dim_reduction length does not match subset indices.")
    merged = [row.tolist() for row in base]
    for out_idx, base_idx in enumerate(subset_indices.tolist()):
        merged[base_idx] = subset[out_idx].tolist()
    return merged


def merge_results(
    base: ClusterResult,
    subset_indices: np.ndarray,
    subset: ClusterResult,
) -> ClusterResult:
    length = base.labels.shape[0]
    merged_labels = merge_labels(base.labels, subset_indices, subset.labels)
    merged_probabilities = merge_optional_array(
        base.probabilities, subset_indices, subset.probabilities, length, np.nan
    )
    merged_outlier_scores = merge_optional_array(
        base.outlier_scores, subset_indices, subset.outlier_scores, length, np.nan
    )
    merged_exemplars = merge_optional_array(
        base.exemplars, subset_indices, subset.exemplars, length, False
    )
    return ClusterResult(
        labels=merged_labels,
        probabilities=merged_probabilities,
        outlier_scores=merged_outlier_scores,
        exemplars=merged_exemplars,
    )


def run_pipeline(cfg: PipelineConfig) -> Path:
    total_start = time.perf_counter()
    log_path = cfg.output_dir / "timings.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_timing(log_path, "=== Pipeline timing log ===")
    if cfg.compute == "only-dimreduction-and-clustering":
        final_csv = run_dimreduction_and_clustering(cfg, log_path, total_start)
        return final_csv
    if cfg.compute == "only-clustering":
        final_csv = run_clustering_only(cfg, log_path, total_start)
        return final_csv

    ensure_deps(require_torch=True)
    auto_repo = auto_model_repo(cfg)
    if auto_repo is not None:
        cfg.dino_model = auto_repo
    configure_ssl(cfg)
    if cfg.two_pass and cfg.fast_tune:
        raise ValueError("Choose either two_pass or fast_tune, not both.")
    if cfg.torch_threads is not None:
        from .torch_utils import torch

        torch.set_num_threads(cfg.torch_threads)
    device = resolve_device()

    index_path = cfg.output_dir / "images.txt"
    index = ImageIndex(
        cfg.input_dir,
        index_path,
        cfg.image_size_in_kbytes_min,
        cfg.image_size_in_kbytes_max,
    )
    if cfg.force or not index_path.exists():
        rel_paths = index.build(cfg.max_images)
    else:
        rel_paths = index.load()
    if not rel_paths:
        raise ValueError(
            "No input images found. The input directory must contain at least one .jpg/.jpeg file. "
            f"input_dir={cfg.input_dir}"
        )

    if cfg.fast_tune:
        fast_cfg = make_fast_config(cfg)
        fast_dir = stage_dir(cfg, "fast")
        paths, _ = run_stage(
            fast_cfg,
            device,
            rel_paths,
            fast_dir,
            write_csv=True,
            stage_label="fast",
            log_path=log_path,
        )
        return _finish_pipeline_run(cfg, paths.csv_path, paths, log_path, total_start)

    if cfg.two_pass:
        fast_cfg = make_fast_config(cfg)
        pass1_dir = stage_dir(cfg, "pass1")
        pass1_paths, pass1_result = run_stage(
            fast_cfg,
            device,
            rel_paths,
            pass1_dir,
            write_csv=True,
            stage_label="pass1",
            log_path=log_path,
        )
        uncertain_idx = select_uncertain(
            pass1_result, cfg.refine_prob_threshold, cfg.refine_include_noise
        )
        if uncertain_idx.size == 0 or uncertain_idx.size < cfg.hdbscan_min_cluster_size:
            final_csv = cfg.output_dir / "clusters.csv"
            pass1_umap = (
                np.load(pass1_paths.umap_path) if cfg.write_dimreduction_vector else None
            )
            HDBSCANClusterer.write_csv(
                final_csv,
                rel_paths,
                pass1_result.labels,
                pass1_result.probabilities,
                pass1_result.outlier_scores,
                pass1_result.exemplars,
                pass1_umap,
            )
            return _finish_pipeline_run(
                cfg, final_csv, pass1_paths, log_path, total_start
            )

        subset_paths = [rel_paths[i] for i in uncertain_idx.tolist()]
        pass2_dir = stage_dir(cfg, "pass2")
        pass2_paths, pass2_result = run_stage(
            cfg,
            device,
            subset_paths,
            pass2_dir,
            write_csv=True,
            stage_label="pass2",
            log_path=log_path,
        )
        merged_result = merge_results(pass1_result, uncertain_idx, pass2_result)
        merged_umap = None
        if cfg.write_dimreduction_vector:
            pass1_umap = np.load(pass1_paths.umap_path)
            pass2_umap = np.load(pass2_paths.umap_path)
            merged_umap = merge_dim_reduction(
                pass1_umap, uncertain_idx, pass2_umap, len(rel_paths)
            )
        final_csv = cfg.output_dir / "clusters.csv"
        HDBSCANClusterer.write_csv(
            final_csv,
            rel_paths,
            merged_result.labels,
            merged_result.probabilities,
            merged_result.outlier_scores,
            merged_result.exemplars,
            merged_umap,
        )
        return _finish_pipeline_run(cfg, final_csv, pass1_paths, log_path, total_start)

    full_dir = stage_dir(cfg, "full")
    paths, _ = run_stage(
        cfg,
        device,
        rel_paths,
        full_dir,
        write_csv=True,
        stage_label="full",
        log_path=log_path,
    )
    return _finish_pipeline_run(cfg, paths.csv_path, paths, log_path, total_start)


def run_dimreduction_and_clustering(
    cfg: PipelineConfig, log_path: Path, total_start: float
) -> Path:
    if cfg.dino_files is None:
        raise ValueError("dino_files must be set for only-dimreduction-and-clustering.")
    base_dir = cfg.dino_files
    if not base_dir.exists():
        raise ValueError(f"dino_files does not exist: {base_dir}")
    input_paths = stage_paths(base_dir)
    output_paths = stage_paths(cfg.output_dir)
    if cfg.subset_images is not None:
        prepare_subset_cache(input_paths, output_paths, cfg.subset_images)
        input_paths = output_paths
    required = {
        "images.txt": input_paths.index_path,
        "embeddings.dat": input_paths.emb_path,
        "embeddings.json": input_paths.meta_path,
        "sizes.npy": input_paths.size_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            "Missing required DINOv2 artifacts for only-dimreduction-and-clustering: "
            f"{missing_str} in {base_dir}"
        )
    with input_paths.index_path.open("r", encoding="utf-8") as f:
        rel_paths = [line.strip() for line in f if line.strip()]
    if not rel_paths:
        raise ValueError(f"No image paths found in {input_paths.index_path}")
    with input_paths.meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    expected = int(meta.get("num_images", -1))
    if expected > 0 and expected != len(rel_paths):
        raise ValueError(
            "Mismatch between embeddings.json num_images and images.txt length. "
            f"num_images={expected} images.txt={len(rel_paths)}"
        )
    output_paths.index_path.parent.mkdir(parents=True, exist_ok=True)
    with output_paths.index_path.open("w", encoding="utf-8") as f:
        for rel in rel_paths:
            f.write(rel + "\n")

    t0 = time.perf_counter()
    reducer = UMAPReducer(cfg)
    reducer.reduce(
        input_paths.emb_path,
        input_paths.meta_path,
        input_paths.size_path,
        output_paths.umap_path,
    )
    umap_dt = time.perf_counter() - t0
    umap_msg = f"[dim-reduction-only] UMAP reduction: {_format_duration(umap_dt)}"
    print(umap_msg)
    _log_timing(log_path, umap_msg)

    t0 = time.perf_counter()
    clusterer = HDBSCANClusterer(cfg)
    result = clusterer.fit(output_paths.umap_path)
    hdb_dt = time.perf_counter() - t0
    hdb_msg = f"[dim-reduction-only] HDBSCAN: {_format_duration(hdb_dt)}"
    print(hdb_msg)
    _log_timing(log_path, hdb_msg)

    t0 = time.perf_counter()
    dim_reduction = np.load(output_paths.umap_path) if cfg.write_dimreduction_vector else None
    clusterer.write_csv(
        output_paths.csv_path,
        rel_paths,
        result.labels,
        result.probabilities,
        result.outlier_scores,
        result.exemplars,
        dim_reduction,
    )
    csv_dt = time.perf_counter() - t0
    csv_msg = f"[dim-reduction-only] CSV write: {_format_duration(csv_dt)}"
    print(csv_msg)
    _log_timing(log_path, csv_msg)

    subclustering_source_paths = StagePaths(
        index_path=input_paths.index_path,
        emb_path=input_paths.emb_path,
        meta_path=input_paths.meta_path,
        size_path=input_paths.size_path,
        umap_path=output_paths.umap_path,
        csv_path=output_paths.csv_path,
    )
    return _finish_pipeline_run(
        cfg,
        output_paths.csv_path,
        subclustering_source_paths,
        log_path,
        total_start,
    )


def run_clustering_only(cfg: PipelineConfig, log_path: Path, total_start: float) -> Path:
    """Run HDBSCAN only using cached UMAP outputs."""
    if cfg.umap_files is None:
        raise ValueError("umap_files must be set for only-clustering.")
    base_dir = cfg.umap_files
    if not base_dir.exists():
        raise ValueError(f"umap_files does not exist: {base_dir}")
    input_paths = stage_paths(base_dir)
    output_paths = stage_paths(cfg.output_dir)
    required = {
        "images.txt": input_paths.index_path,
        "umap.npy": input_paths.umap_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            "Missing required UMAP artifacts for only-clustering: "
            f"{missing_str} in {base_dir}"
        )
    with input_paths.index_path.open("r", encoding="utf-8") as f:
        rel_paths = [line.strip() for line in f if line.strip()]
    if not rel_paths:
        raise ValueError(f"No image paths found in {input_paths.index_path}")
    umap_data = np.load(input_paths.umap_path)
    if umap_data.ndim != 2:
        raise ValueError(f"umap.npy must be 2D, got shape={umap_data.shape}")
    if umap_data.shape[0] != len(rel_paths):
        raise ValueError(
            "Mismatch between umap.npy rows and images.txt length. "
            f"umap.npy={umap_data.shape[0]} images.txt={len(rel_paths)}"
        )
    output_paths.index_path.parent.mkdir(parents=True, exist_ok=True)
    with output_paths.index_path.open("w", encoding="utf-8") as f:
        for rel in rel_paths:
            f.write(rel + "\n")

    t0 = time.perf_counter()
    clusterer = HDBSCANClusterer(cfg)
    result = clusterer.fit(input_paths.umap_path)
    hdb_dt = time.perf_counter() - t0
    hdb_msg = f"[clustering-only] HDBSCAN: {_format_duration(hdb_dt)}"
    print(hdb_msg)
    _log_timing(log_path, hdb_msg)

    t0 = time.perf_counter()
    dim_reduction = umap_data if cfg.write_dimreduction_vector else None
    clusterer.write_csv(
        output_paths.csv_path,
        rel_paths,
        result.labels,
        result.probabilities,
        result.outlier_scores,
        result.exemplars,
        dim_reduction,
    )
    csv_dt = time.perf_counter() - t0
    csv_msg = f"[clustering-only] CSV write: {_format_duration(csv_dt)}"
    print(csv_msg)
    _log_timing(log_path, csv_msg)

    subclustering_source_paths = None
    if cfg.dino_files is not None:
        dino_paths = stage_paths(cfg.dino_files)
        subclustering_source_paths = StagePaths(
            index_path=dino_paths.index_path,
            emb_path=dino_paths.emb_path,
            meta_path=dino_paths.meta_path,
            size_path=dino_paths.size_path,
            umap_path=input_paths.umap_path,
            csv_path=output_paths.csv_path,
        )
    return _finish_pipeline_run(
        cfg,
        output_paths.csv_path,
        subclustering_source_paths,
        log_path,
        total_start,
    )


def clustering(
    input_image_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 2,
    umap_dim: int = 30,
    hdbscan_min_cluster_size: int = 25,
    **overrides,
) -> Path:
    """
    Run the full clustering pipeline and return the path to ``clusters.csv``.

    The CSV includes columns: image_id, cluster, labeled, probabilities,
    outlier_scores, dim_reduction. ``labeled`` is ``True`` only when the
    basename ends with ``_class_1234.jpg``.
    A sibling ``richness.csv`` is also written with location-by-cluster counts.
    When automatic subclustering runs, each subcluster ``clusters.csv`` receives
    its own sibling ``richness.csv`` as well.

    Parameters
    ----------
    input_image_dir : str | Path
        Folder containing JPG images to cluster.
    output_dir : str | Path
        Folder where artifacts and clusters.csv are written.
    batch_size : int
        Embedding batch size (lower is safer for RAM/IO).
    num_workers : int
        DataLoader workers (keep low for external drives).
    umap_dim : int
        Target dimensionality for UMAP.
    hdbscan_min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    **overrides
        Any other PipelineConfig fields to override, e.g. two_pass=True,
        autocrop=False, fast_tune=True, model_name="dinov2_vitb14",
        dino_model="/path/to/dinov2".
    """
    cfg = PipelineConfig(
        input_dir=Path(input_image_dir),
        output_dir=Path(output_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        umap_dim=umap_dim,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        **overrides,
    )
    from .config import validate_config

    validate_config(cfg)
    return run_pipeline(cfg)
