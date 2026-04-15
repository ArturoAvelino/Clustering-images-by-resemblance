from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List

import numpy as np

from .config import ClusterResult, PipelineConfig


class UMAPReducer:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def reduce(
        self, emb_path: Path, meta_path: Path, size_path: Path, out_path: Path
    ) -> None:
        """Reduces embeddings dimensionality using UMAP and persists result"""
        import umap

        meta = self._load_meta(meta_path)
        n = meta["num_images"]
        dim = meta["embed_dim"]
        if n <= 0:
            raise ValueError(
                f"Embeddings metadata reports 0 samples in {meta_path}. "
                "This usually means no input images were found or the index was empty."
            )
        dtype = np.float16 if meta["dtype"] == "float16" else np.float32
        emb = np.memmap(emb_path, mode="r", dtype=dtype, shape=(n, dim))
        emb = np.asarray(emb, dtype=np.float32)
        sizes = np.load(size_path).astype(np.float32)
        if sizes.shape[0] != n:
            raise ValueError(
                f"Size feature count ({sizes.shape[0]}) does not match embeddings ({n})."
            )
        size_feature = sizes.reshape(-1, 1)
        mean = float(size_feature.mean())
        std = float(size_feature.std())
        if std < 1e-6:
            std = 1.0
        size_feature = (size_feature - mean) / std
        size_feature *= float(self.cfg.size_feature_weight)
        emb = np.concatenate([emb, size_feature], axis=1)
        reducer = umap.UMAP(
            n_components=self.cfg.umap_dim,
            n_neighbors=self.cfg.umap_neighbors,
            min_dist=self.cfg.umap_min_dist,
            metric=self.cfg.umap_metric,
            low_memory=True,
            random_state=42,
        )
        low = reducer.fit_transform(emb)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, low.astype(np.float32))

    @staticmethod
    def _load_meta(meta_path: Path) -> dict:
        import json

        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)


class HDBSCANClusterer:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def fit(self, umap_path: Path) -> ClusterResult:
        import hdbscan

        data = np.load(umap_path)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.hdbscan_min_cluster_size,
            min_samples=self.cfg.hdb_min_samples,
            metric=self.cfg.hdb_metric,
            cluster_selection_method="eom",
            approx_min_span_tree=True,
            prediction_data=True,
            core_dist_n_jobs=os.cpu_count() or 1,
        )
        labels = clusterer.fit_predict(data)
        probabilities = getattr(clusterer, "probabilities_", None)
        outlier_scores = getattr(clusterer, "outlier_scores_", None)
        exemplars = None
        try:
            exemplar_points = clusterer.exemplars_
        except Exception:
            exemplar_points = None
        if exemplar_points is not None:
            exemplars = self._build_exemplar_mask(data, exemplar_points)
        return ClusterResult(
            labels=labels,
            probabilities=probabilities,
            outlier_scores=outlier_scores,
            exemplars=exemplars,
        )

    @staticmethod
    def write_csv(
        out_csv: Path,
        rel_paths: List[str],
        labels: np.ndarray,
        probabilities: np.ndarray | None = None,
        outlier_scores: np.ndarray | None = None,
        exemplars: np.ndarray | None = None,
        dim_reduction: np.ndarray | List[List[float]] | None = None,
    ) -> None:
        """Write clustering results to CSV with optional HDBSCAN metadata and UMAP output."""
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if dim_reduction is not None:
            if isinstance(dim_reduction, np.ndarray):
                if dim_reduction.ndim != 2:
                    raise ValueError("dim_reduction must be a 2D array.")
                if dim_reduction.shape[0] != len(rel_paths):
                    raise ValueError(
                        "dim_reduction length does not match the number of images."
                    )
            elif len(dim_reduction) != len(rel_paths):
                raise ValueError(
                    "dim_reduction length does not match the number of images."
                )
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            headers = ["image_id", "cluster", "probabilities", "outlier_scores"]
            if dim_reduction is not None:
                headers.append("dim_reduction")
            writer.writerow(headers)
            for idx, (rel, label) in enumerate(zip(rel_paths, labels)):
                prob = (
                    ""
                    if probabilities is None
                    else round(float(probabilities[idx]), 4)
                )
                outlier = (
                    ""
                    if outlier_scores is None
                    else round(float(outlier_scores[idx]), 4)
                )
                row_values = [rel, int(label), prob, outlier]
                if dim_reduction is not None:
                    row = (
                        dim_reduction[idx].tolist()
                        if isinstance(dim_reduction, np.ndarray)
                        else dim_reduction[idx]
                    )
                    dim_row = [round(float(x), 4) for x in row]
                    row_values.append(json.dumps(dim_row))
                writer.writerow(row_values)

    @staticmethod
    def _build_exemplar_mask(
        data: np.ndarray, exemplar_points: List[np.ndarray]
    ) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError("HDBSCAN exemplar mapping expects 2D input data.")
        if not exemplar_points:
            return np.zeros(data.shape[0], dtype=bool)
        exemplars = [ex for ex in exemplar_points if ex is not None and ex.size > 0]
        if not exemplars:
            return np.zeros(data.shape[0], dtype=bool)
        exemplar_array = np.concatenate(exemplars, axis=0)
        data_c = np.ascontiguousarray(data)
        exemplar_c = np.ascontiguousarray(exemplar_array)
        row_dtype = np.dtype((np.void, data_c.dtype.itemsize * data_c.shape[1]))
        data_view = data_c.view(row_dtype).ravel()
        exemplar_view = exemplar_c.view(row_dtype).ravel()
        return np.isin(data_view, exemplar_view)
