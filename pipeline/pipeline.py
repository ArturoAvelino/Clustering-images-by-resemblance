from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .algorithms import HDBSCANClusterer, UMAPReducer
from .config import ClusterResult, PipelineConfig, StagePaths, make_fast_config
from .data import ImageDataset, ImageIndex, compute_size_features
from .embedding import DINOv2Embedder, ensure_deps, resolve_device
from .model_repo import auto_model_repo
from .ssl_utils import configure_ssl
from .summary import summarize_clusters_csv


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
    ensure_deps()
    auto_repo = auto_model_repo(cfg)
    if auto_repo is not None:
        cfg.model_repo = auto_repo
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
        summarize_clusters_csv(paths.csv_path)
        total_dt = time.perf_counter() - total_start
        total_msg = f"[total] Pipeline runtime: {_format_duration(total_dt)}"
        print(total_msg)
        _log_timing(log_path, total_msg)
        return paths.csv_path

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
            summarize_clusters_csv(final_csv)
            total_dt = time.perf_counter() - total_start
            total_msg = f"[total] Pipeline runtime: {_format_duration(total_dt)}"
            print(total_msg)
            _log_timing(log_path, total_msg)
            return final_csv

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
        summarize_clusters_csv(final_csv)
        total_dt = time.perf_counter() - total_start
        total_msg = f"[total] Pipeline runtime: {_format_duration(total_dt)}"
        print(total_msg)
        _log_timing(log_path, total_msg)
        return final_csv

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
    summarize_clusters_csv(paths.csv_path)
    total_dt = time.perf_counter() - total_start
    total_msg = f"[total] Pipeline runtime: {_format_duration(total_dt)}"
    print(total_msg)
    _log_timing(log_path, total_msg)
    return paths.csv_path


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
    Run the full clustering pipeline and return the path to the CSV output.

    The CSV includes columns: image_id, cluster, probabilities, outlier_scores,
    dim_reduction.

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
        autocrop=False, fast_tune=True, model_name="dinov2_vitb14".
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
