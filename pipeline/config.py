from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


@dataclass
class PipelineConfig:
    """Pipeline runtime configuration loaded from CLI flags and/or YAML files.

    The same schema is accepted for full runs, `only-dimreduction-and-clustering`,
    and `only-clustering`. Some fields are mode-specific at runtime; for example,
    `refine_prob_threshold` and `refine_include_noise` are only used when
    `two_pass` is enabled during a full run.
    """

    input_dir: Optional[Path] = None
    output_dir: Path = Path("")
    compute: str = "full"
    dino_files: Optional[Path] = None
    subset_images: Optional[Path] = None
    umap_files: Optional[Path] = None
    model_name: str = "dinov2_vitb14"
    dino_model: Optional[str] = None
    ssl_ca_bundle: Optional[Path] = None
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    dtype: str = "float16"
    max_images: Optional[int] = None
    umap_dim: int = 30
    umap_neighbors: int = 30
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 25
    hdb_min_samples: int = 10
    hdb_metric: str = "euclidean"
    hdb_cluster_selection_method: str = "eom"
    hdb_cluster_selection_epsilon: float = 0.0
    hdb_allow_single_cluster: bool = False
    autocrop: bool = False
    autocrop_threshold: int = 35
    autocrop_padding: int = 2
    background_color: tuple[int, int, int] = (45, 71, 159)
    size_feature_weight: float = 0.0
    image_size_in_kbytes_min: Optional[float] = None
    image_size_in_kbytes_max: Optional[float] = None
    two_pass: bool = False
    fast_tune: bool = False
    fast_model_name: str = "dinov2_vits14"
    fast_img_size: int = 196
    fast_umap_dim: int = 15
    fast_umap_neighbors: int = 20
    fast_batch_size: Optional[int] = None
    fast_num_workers: Optional[int] = None
    refine_prob_threshold: float = 0.7
    refine_include_noise: bool = True
    write_dimreduction_vector: bool = True
    force: bool = False
    torch_threads: Optional[int] = None
    classes_benchmark_file: Optional[Path] = None
    subclustering: bool = True
    min_for_subclustering: int = 1000
    subclustering_umap_dim: int = 60
    subclustering_umap_neighbors: int = 30
    subclustering_hdbscan_min_cluster_size: int = 7
    subclustering_hdb_min_samples: int = 6
    subclustering_hdb_cluster_selection_method: Optional[str] = None
    subclustering_hdb_cluster_selection_epsilon: Optional[float] = None
    subclustering_hdb_allow_single_cluster: Optional[bool] = None
    merge_noise_subclusters: bool = False


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: Optional[np.ndarray]
    outlier_scores: Optional[np.ndarray]
    exemplars: Optional[np.ndarray]


@dataclass
class StagePaths:
    index_path: Path
    emb_path: Path
    meta_path: Path
    size_path: Path
    umap_path: Path
    csv_path: Path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be an object/dict at the top level.")

    if "input_image_dir" in data:
        if "cropped_images_dir" not in data:
            data["cropped_images_dir"] = data["input_image_dir"]
        data.pop("input_image_dir")

    if "input_dir" in data:
        raise ValueError("input_dir is no longer supported. Use cropped_images_dir instead.")

    if "cropped_images_dir" in data:
        data["input_dir"] = data.pop("cropped_images_dir")

    if "hdb_min_cluster_size" in data:
        if "hdbscan_min_cluster_size" in data:
            raise ValueError(
                "Use only one of hdb_min_cluster_size or hdbscan_min_cluster_size."
            )
        data["hdbscan_min_cluster_size"] = data.pop("hdb_min_cluster_size")

    if "model_repo" in data:
        if "dino_model" in data:
            raise ValueError("Use only one of model_repo or dino_model.")
        data["dino_model"] = data.pop("model_repo")

    valid = {field.name for field in fields(PipelineConfig)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    return data


def default_config_dict() -> dict:
    placeholder = PipelineConfig()
    return dict(placeholder.__dict__)


def build_config(args) -> PipelineConfig:
    cfg_data = default_config_dict()
    if args.config is not None:
        cfg_data.update(load_config(Path(args.config)))

    overrides = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "compute": args.compute,
        "dino_files": args.dino_files,
        "subset_images": getattr(args, "subset_images", None),
        "umap_files": args.umap_files,
        "model_name": args.model_name,
        "dino_model": args.dino_model,
        "ssl_ca_bundle": args.ssl_ca_bundle,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "dtype": args.dtype,
        "max_images": args.max_images,
        "umap_dim": args.umap_dim,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_metric": args.umap_metric,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdb_min_samples": args.hdb_min_samples,
        "hdb_metric": args.hdb_metric,
        "hdb_cluster_selection_method": args.hdb_cluster_selection_method,
        "hdb_cluster_selection_epsilon": args.hdb_cluster_selection_epsilon,
        "hdb_allow_single_cluster": args.hdb_allow_single_cluster,
        "autocrop": args.autocrop,
        "autocrop_threshold": args.autocrop_threshold,
        "autocrop_padding": args.autocrop_padding,
        "background_color": args.background_color,
        "size_feature_weight": args.size_feature_weight,
        "two_pass": args.two_pass,
        "fast_tune": args.fast_tune,
        "fast_model_name": args.fast_model_name,
        "fast_img_size": args.fast_img_size,
        "fast_umap_dim": args.fast_umap_dim,
        "fast_umap_neighbors": args.fast_umap_neighbors,
        "fast_batch_size": args.fast_batch_size,
        "fast_num_workers": args.fast_num_workers,
        "refine_prob_threshold": args.refine_prob_threshold,
        "refine_include_noise": args.refine_include_noise,
        "write_dimreduction_vector": args.write_dimreduction_vector,
        "force": args.force,
        "torch_threads": args.torch_threads,
        "classes_benchmark_file": getattr(args, "classes_benchmark_file", None),
        "subclustering": getattr(args, "subclustering", None),
        "min_for_subclustering": getattr(args, "min_for_subclustering", None),
        "subclustering_umap_dim": getattr(args, "subclustering_umap_dim", None),
        "subclustering_umap_neighbors": getattr(args, "subclustering_umap_neighbors", None),
        "subclustering_hdbscan_min_cluster_size": getattr(
            args, "subclustering_hdbscan_min_cluster_size", None
        ),
        "subclustering_hdb_min_samples": getattr(args, "subclustering_hdb_min_samples", None),
        "subclustering_hdb_cluster_selection_method": getattr(
            args, "subclustering_hdb_cluster_selection_method", None
        ),
        "subclustering_hdb_cluster_selection_epsilon": getattr(
            args, "subclustering_hdb_cluster_selection_epsilon", None
        ),
        "subclustering_hdb_allow_single_cluster": getattr(
            args, "subclustering_hdb_allow_single_cluster", None
        ),
        "merge_noise_subclusters": getattr(args, "merge_noise_subclusters", None),
    }
    for key, value in overrides.items():
        if value is not None:
            cfg_data[key] = value

    compute_mode = cfg_data.get("compute") or "full"
    if compute_mode not in {"full", "only-dimreduction-and-clustering", "only-clustering"}:
        raise ValueError(
            "compute must be 'full', 'only-dimreduction-and-clustering', or 'only-clustering'."
        )
    if not cfg_data.get("output_dir"):
        raise ValueError("output_dir is required (CLI --output-dir or config file).")
    if compute_mode == "only-dimreduction-and-clustering":
        if not cfg_data.get("dino_files"):
            raise ValueError(
                "dino_files is required when compute=only-dimreduction-and-clustering."
            )
    elif compute_mode == "only-clustering":
        if not cfg_data.get("umap_files"):
            raise ValueError("umap_files is required when compute=only-clustering.")
    else:
        if not cfg_data.get("input_dir"):
            raise ValueError("input_dir is required (CLI --input-dir or config file).")

    if cfg_data.get("input_dir") is not None:
        cfg_data["input_dir"] = Path(cfg_data["input_dir"])
    cfg_data["output_dir"] = Path(cfg_data["output_dir"])
    if cfg_data.get("background_color") is not None:
        cfg_data["background_color"] = tuple(int(x) for x in cfg_data["background_color"])
    if cfg_data.get("ssl_ca_bundle"):
        cfg_data["ssl_ca_bundle"] = Path(cfg_data["ssl_ca_bundle"])
    if cfg_data.get("dino_files") is not None:
        cfg_data["dino_files"] = Path(cfg_data["dino_files"])
    if cfg_data.get("subset_images") is not None:
        cfg_data["subset_images"] = Path(cfg_data["subset_images"])
    if cfg_data.get("umap_files") is not None:
        cfg_data["umap_files"] = Path(cfg_data["umap_files"])
    if cfg_data.get("classes_benchmark_file") is not None:
        cfg_data["classes_benchmark_file"] = Path(cfg_data["classes_benchmark_file"])
    return PipelineConfig(**cfg_data)


def validate_config(cfg: PipelineConfig) -> None:
    errors: List[str] = []
    if cfg.subset_images is not None:
        if cfg.compute != "only-dimreduction-and-clustering":
            errors.append("subset_images requires compute=only-dimreduction-and-clustering")
        if cfg.dino_files is not None and cfg.output_dir.resolve() == cfg.dino_files.resolve():
            errors.append("subset_images requires output_dir different from dino_files")
    if cfg.compute not in {"full", "only-dimreduction-and-clustering", "only-clustering"}:
        errors.append(
            "compute must be 'full', 'only-dimreduction-and-clustering', or 'only-clustering'"
        )
    if cfg.compute == "only-dimreduction-and-clustering":
        if cfg.dino_files is None:
            errors.append("dino_files is required when compute=only-dimreduction-and-clustering")
        if cfg.two_pass or cfg.fast_tune:
            errors.append("two_pass and fast_tune are not supported when compute=only-dimreduction-and-clustering")
    if cfg.compute == "only-clustering":
        if cfg.umap_files is None:
            errors.append("umap_files is required when compute=only-clustering")
        if cfg.two_pass or cfg.fast_tune:
            errors.append("two_pass and fast_tune are not supported when compute=only-clustering")
    if cfg.compute == "full" and cfg.input_dir is None:
        errors.append("input_dir is required when compute=full")
    if cfg.img_size <= 0:
        errors.append("img_size must be > 0")
    if cfg.batch_size <= 0:
        errors.append("batch_size must be > 0")
    if cfg.num_workers < 0:
        errors.append("num_workers must be >= 0")
    if cfg.umap_dim <= 1:
        errors.append("umap_dim must be > 1")
    if cfg.umap_neighbors <= 2:
        errors.append("umap_neighbors must be > 2")
    if not (0.0 <= cfg.umap_min_dist <= 1.0):
        errors.append("umap_min_dist must be between 0 and 1")
    if cfg.hdbscan_min_cluster_size < 2:
        errors.append("hdbscan_min_cluster_size must be >= 2")
    if cfg.hdb_min_samples is not None and cfg.hdb_min_samples < 1:
        errors.append("hdb_min_samples must be >= 1")
    if cfg.hdb_cluster_selection_method not in {"eom", "leaf"}:
        errors.append("hdb_cluster_selection_method must be 'eom' or 'leaf'")
    if cfg.hdb_cluster_selection_epsilon < 0:
        errors.append("hdb_cluster_selection_epsilon must be >= 0")
    if cfg.autocrop_threshold < 0:
        errors.append("autocrop_threshold must be >= 0")
    if cfg.autocrop_padding < 0:
        errors.append("autocrop_padding must be >= 0")
    if cfg.background_color is None:
        errors.append("background_color is required")
    elif len(cfg.background_color) != 3:
        errors.append("background_color must have 3 values (R,G,B)")
    else:
        for value in cfg.background_color:
            if value < 0 or value > 255:
                errors.append("background_color values must be between 0 and 255")
    if cfg.size_feature_weight < 0:
        errors.append("size_feature_weight must be >= 0")
    if cfg.image_size_in_kbytes_min is not None and cfg.image_size_in_kbytes_min < 0:
        errors.append("image_size_in_kbytes_min must be >= 0")
    if cfg.image_size_in_kbytes_max is not None and cfg.image_size_in_kbytes_max < 0:
        errors.append("image_size_in_kbytes_max must be >= 0")
    if (
        cfg.image_size_in_kbytes_min is not None
        and cfg.image_size_in_kbytes_max is not None
        and cfg.image_size_in_kbytes_min > cfg.image_size_in_kbytes_max
    ):
        errors.append("image_size_in_kbytes_min must be <= image_size_in_kbytes_max")
    if cfg.fast_img_size <= 0:
        errors.append("fast_img_size must be > 0")
    if cfg.fast_umap_dim <= 1:
        errors.append("fast_umap_dim must be > 1")
    if cfg.fast_umap_neighbors <= 2:
        errors.append("fast_umap_neighbors must be > 2")
    if cfg.fast_batch_size is not None and cfg.fast_batch_size <= 0:
        errors.append("fast_batch_size must be > 0")
    if cfg.fast_num_workers is not None and cfg.fast_num_workers < 0:
        errors.append("fast_num_workers must be >= 0")
    if not (0.0 <= cfg.refine_prob_threshold <= 1.0):
        errors.append("refine_prob_threshold must be between 0 and 1")
    if cfg.two_pass and cfg.fast_tune:
        errors.append("two_pass and fast_tune cannot both be true")
    if cfg.min_for_subclustering < 1:
        errors.append("min_for_subclustering must be >= 1")
    if cfg.subclustering_umap_dim <= 1:
        errors.append("subclustering_umap_dim must be > 1")
    if cfg.subclustering_umap_neighbors <= 2:
        errors.append("subclustering_umap_neighbors must be > 2")
    if cfg.subclustering_hdbscan_min_cluster_size < 2:
        errors.append("subclustering_hdbscan_min_cluster_size must be >= 2")
    if cfg.subclustering_hdb_min_samples is not None and cfg.subclustering_hdb_min_samples < 1:
        errors.append("subclustering_hdb_min_samples must be >= 1")
    if (
        cfg.subclustering_hdb_cluster_selection_method is not None
        and cfg.subclustering_hdb_cluster_selection_method not in {"eom", "leaf"}
    ):
        errors.append("subclustering_hdb_cluster_selection_method must be 'eom' or 'leaf'")
    if (
        cfg.subclustering_hdb_cluster_selection_epsilon is not None
        and cfg.subclustering_hdb_cluster_selection_epsilon < 0
    ):
        errors.append("subclustering_hdb_cluster_selection_epsilon must be >= 0")
    if errors:
        msg = "Invalid configuration:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)


def config_to_yaml(cfg: PipelineConfig) -> str:
    data = cfg.__dict__.copy()
    input_dir = data.pop("input_dir", None)
    if input_dir is not None:
        data["cropped_images_dir"] = str(input_dir)
    data["output_dir"] = str(data["output_dir"])
    if data.get("ssl_ca_bundle") is not None:
        data["ssl_ca_bundle"] = str(data["ssl_ca_bundle"])
    if data.get("dino_files") is not None:
        data["dino_files"] = str(data["dino_files"])
    if data.get("subset_images") is not None:
        data["subset_images"] = str(data["subset_images"])
    if data.get("umap_files") is not None:
        data["umap_files"] = str(data["umap_files"])
    if data.get("classes_benchmark_file") is not None:
        data["classes_benchmark_file"] = str(data["classes_benchmark_file"])
    return yaml.safe_dump(data, sort_keys=True)


def make_fast_config(cfg: PipelineConfig) -> PipelineConfig:
    return replace(
        cfg,
        model_name=cfg.fast_model_name,
        img_size=cfg.fast_img_size,
        umap_dim=cfg.fast_umap_dim,
        umap_neighbors=cfg.fast_umap_neighbors,
        batch_size=cfg.fast_batch_size or cfg.batch_size,
        num_workers=cfg.fast_num_workers or cfg.num_workers,
    )
