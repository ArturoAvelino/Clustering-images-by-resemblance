from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from .class_labels import is_strictly_labeled_jpg
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
        sizes = np.load(size_path, mmap_mode="r")
        if sizes.shape[0] != n:
            raise ValueError(
                f"Size feature count ({sizes.shape[0]}) does not match embeddings ({n})."
            )
        size_stats = self._size_stats(sizes) if self.cfg.size_feature_weight > 0 else None
        reducer = umap.UMAP(
            n_components=self.cfg.umap_dim,
            n_neighbors=self.cfg.umap_neighbors,
            min_dist=self.cfg.umap_min_dist,
            metric=self.cfg.umap_metric,
            low_memory=True,
            random_state=42,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fit_sample_size = self.cfg.umap_fit_sample_size
        if fit_sample_size is not None and n > fit_sample_size:
            self._fit_sample_transform_all(reducer, emb, sizes, size_stats, out_path)
            return

        with self._umap_input_memmap(emb, sizes, size_stats, out_path.parent) as data:
            low = reducer.fit_transform(data)
        out = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=low.shape,
        )
        out[:] = low.astype(np.float32, copy=False)
        out.flush()
        del out, low

    def _fit_sample_transform_all(
        self,
        reducer,
        emb: np.memmap,
        sizes: np.ndarray,
        size_stats: tuple[float, float] | None,
        out_path: Path,
    ) -> None:
        n = emb.shape[0]
        sample_size = int(self.cfg.umap_fit_sample_size or n)
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        sample = self._build_input_rows(emb, sizes, size_stats, sample_idx)
        reducer.fit(sample)
        del sample

        out = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=(n, self.cfg.umap_dim),
        )
        batch_size = int(self.cfg.umap_transform_batch_size)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = self._build_input_slice(emb, sizes, size_stats, start, stop)
            out[start:stop] = reducer.transform(batch).astype(np.float32, copy=False)
            out.flush()
        del out

    def _umap_input_memmap(
        self,
        emb: np.memmap,
        sizes: np.ndarray,
        size_stats: tuple[float, float] | None,
        temp_dir: Path,
    ):
        if emb.dtype == np.float32 and size_stats is None:
            return _ArrayContext(emb)

        n, dim = emb.shape
        input_dim = dim + (1 if size_stats is not None else 0)
        return _TemporaryInputMemmap(
            shape=(n, input_dim),
            temp_dir=temp_dir,
            fill=lambda out: self._fill_input_memmap(out, emb, sizes, size_stats),
        )

    def _fill_input_memmap(
        self,
        out: np.memmap,
        emb: np.memmap,
        sizes: np.ndarray,
        size_stats: tuple[float, float] | None,
    ) -> None:
        batch_size = int(self.cfg.umap_transform_batch_size)
        for start in range(0, emb.shape[0], batch_size):
            stop = min(start + batch_size, emb.shape[0])
            out[start:stop] = self._build_input_slice(emb, sizes, size_stats, start, stop)
        out.flush()

    def _build_input_rows(
        self,
        emb: np.memmap,
        sizes: np.ndarray,
        size_stats: tuple[float, float] | None,
        rows: np.ndarray,
    ) -> np.ndarray:
        data = np.asarray(emb[rows], dtype=np.float32)
        if size_stats is None:
            return data
        size_feature = self._normalized_size_feature(sizes[rows], size_stats)
        return np.concatenate([data, size_feature], axis=1)

    def _build_input_slice(
        self,
        emb: np.memmap,
        sizes: np.ndarray,
        size_stats: tuple[float, float] | None,
        start: int,
        stop: int,
    ) -> np.ndarray:
        data = np.asarray(emb[start:stop], dtype=np.float32)
        if size_stats is None:
            return data
        size_feature = self._normalized_size_feature(sizes[start:stop], size_stats)
        return np.concatenate([data, size_feature], axis=1)

    def _normalized_size_feature(
        self, values: np.ndarray, size_stats: tuple[float, float]
    ) -> np.ndarray:
        mean, std = size_stats
        feature = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        feature = (feature - mean) / std
        feature *= float(self.cfg.size_feature_weight)
        return feature

    @staticmethod
    def _size_stats(sizes: np.ndarray) -> tuple[float, float]:
        mean = float(np.mean(sizes, dtype=np.float64))
        std = float(np.std(sizes, dtype=np.float64))
        if std < 1e-6:
            std = 1.0
        return mean, std

    @staticmethod
    def _load_meta(meta_path: Path) -> dict:
        import json

        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)


class _ArrayContext:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __enter__(self) -> np.ndarray:
        return self.array

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _TemporaryInputMemmap:
    def __init__(self, shape: tuple[int, int], temp_dir: Path, fill) -> None:
        self.shape = shape
        self.temp_dir = temp_dir
        self.fill = fill
        self._tmp = None
        self._array = None

    def __enter__(self) -> np.memmap:
        self._tmp = tempfile.NamedTemporaryFile(
            prefix="umap_input_",
            suffix=".dat",
            dir=self.temp_dir,
        )
        self._array = np.memmap(
            self._tmp.name,
            mode="w+",
            dtype=np.float32,
            shape=self.shape,
        )
        self.fill(self._array)
        return self._array

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._array is not None:
            self._array.flush()
        self._array = None
        if self._tmp is not None:
            self._tmp.close()


class HDBSCANClusterer:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def fit(self, umap_path: Path) -> ClusterResult:
        import hdbscan

        data = np.load(umap_path, mmap_mode="r")
        if data.ndim != 2:
            raise ValueError(f"umap.npy must be 2D, got shape={data.shape}")
        fit_sample_size = self.cfg.hdbscan_fit_sample_size
        if fit_sample_size is not None and data.shape[0] > fit_sample_size:
            return self._fit_sample_predict_all(hdbscan, data, fit_sample_size)
        data = np.asarray(data, dtype=np.float32)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.hdbscan_min_cluster_size,
            min_samples=self.cfg.hdb_min_samples,
            metric=self.cfg.hdb_metric,
            cluster_selection_method=self.cfg.hdb_cluster_selection_method,
            cluster_selection_epsilon=self.cfg.hdb_cluster_selection_epsilon,
            allow_single_cluster=self.cfg.hdb_allow_single_cluster,
            approx_min_span_tree=True,
            prediction_data=False,
            core_dist_n_jobs=self.cfg.hdbscan_core_dist_n_jobs or 1,
        )
        labels = clusterer.fit_predict(data)
        probabilities = getattr(clusterer, "probabilities_", None)
        outlier_scores = getattr(clusterer, "outlier_scores_", None)
        return ClusterResult(
            labels=labels,
            probabilities=probabilities,
            outlier_scores=outlier_scores,
            exemplars=None,
        )

    def _fit_sample_predict_all(
        self, hdbscan, data: np.ndarray, sample_size: int
    ) -> ClusterResult:
        n = data.shape[0]
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        sample = np.asarray(data[sample_idx], dtype=np.float32)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.hdbscan_min_cluster_size,
            min_samples=self.cfg.hdb_min_samples,
            metric=self.cfg.hdb_metric,
            cluster_selection_method=self.cfg.hdb_cluster_selection_method,
            cluster_selection_epsilon=self.cfg.hdb_cluster_selection_epsilon,
            allow_single_cluster=self.cfg.hdb_allow_single_cluster,
            approx_min_span_tree=True,
            prediction_data=True,
            core_dist_n_jobs=self.cfg.hdbscan_core_dist_n_jobs or 1,
        )
        clusterer.fit(sample)
        labels = np.empty(n, dtype=np.int32)
        probabilities = np.empty(n, dtype=np.float32)
        batch_size = int(self.cfg.hdbscan_predict_batch_size)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = np.asarray(data[start:stop], dtype=np.float32)
            batch_labels, batch_probabilities = hdbscan.approximate_predict(
                clusterer, batch
            )
            labels[start:stop] = batch_labels.astype(np.int32, copy=False)
            probabilities[start:stop] = batch_probabilities
        return ClusterResult(
            labels=labels,
            probabilities=probabilities,
            outlier_scores=None,
            exemplars=None,
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
        """
        Write clustering results to CSV with label validation metadata.

        The output always includes a ``labeled`` column. A row is marked
        ``True`` only when the basename from ``image_id`` ends with the exact
        pattern ``_class_1234.jpg``; otherwise it is marked ``False``.
        Optional HDBSCAN metadata and the serialized UMAP vector are appended
        after that column.
        """
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
            headers = ["image_id", "cluster", "labeled", "probabilities", "outlier_scores"]
            if dim_reduction is not None:
                headers.append("dim_reduction")
            writer.writerow(headers)
            for idx, (rel, label) in enumerate(zip(rel_paths, labels)):
                labeled = is_strictly_labeled_jpg(rel)
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
                row_values = [rel, int(label), labeled, prob, outlier]
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
