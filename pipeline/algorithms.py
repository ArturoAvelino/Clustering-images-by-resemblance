from __future__ import annotations

import csv
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
            min_cluster_size=self.cfg.hdb_min_cluster_size,
            min_samples=self.cfg.hdb_min_samples,
            metric=self.cfg.hdb_metric,
            cluster_selection_method="eom",
            approx_min_span_tree=True,
            prediction_data=True,
            core_dist_n_jobs=os.cpu_count() or 1,
        )
        labels = clusterer.fit_predict(data)
        probabilities = getattr(clusterer, "probabilities_", None)
        return ClusterResult(labels=labels, probabilities=probabilities)

    @staticmethod
    def write_csv(out_csv: Path, rel_paths: List[str], labels: np.ndarray) -> None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "cluster"])
            for rel, label in zip(rel_paths, labels):
                writer.writerow([rel, int(label)])
