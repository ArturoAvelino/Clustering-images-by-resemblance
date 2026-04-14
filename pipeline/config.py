from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    model_name: str = "dinov2_vitb14"
    model_repo: Optional[str] = None
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
    hdb_min_cluster_size: int = 25
    hdb_min_samples: int = 10
    hdb_metric: str = "euclidean"
    autocrop: bool = False
    autocrop_threshold: int = 35
    autocrop_padding: int = 2
    background_color: tuple[int, int, int] = (45, 71, 159)
    size_feature_weight: float = 4.0
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
    force: bool = False
    torch_threads: Optional[int] = None


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

    valid = {field.name for field in fields(PipelineConfig)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    return data


def default_config_dict() -> dict:
    placeholder = PipelineConfig(input_dir=Path(""), output_dir=Path(""))
    return dict(placeholder.__dict__)


def build_config(args) -> PipelineConfig:
    cfg_data = default_config_dict()
    if args.config is not None:
        cfg_data.update(load_config(Path(args.config)))

    overrides = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "model_repo": args.model_repo,
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
        "hdb_min_cluster_size": args.hdb_min_cluster_size,
        "hdb_min_samples": args.hdb_min_samples,
        "hdb_metric": args.hdb_metric,
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
        "force": args.force,
        "torch_threads": args.torch_threads,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg_data[key] = value

    if not cfg_data.get("input_dir"):
        raise ValueError("input_dir is required (CLI --input-dir or config file).")
    if not cfg_data.get("output_dir"):
        raise ValueError("output_dir is required (CLI --output-dir or config file).")

    cfg_data["input_dir"] = Path(cfg_data["input_dir"])
    cfg_data["output_dir"] = Path(cfg_data["output_dir"])
    if cfg_data.get("background_color") is not None:
        cfg_data["background_color"] = tuple(int(x) for x in cfg_data["background_color"])
    if cfg_data.get("ssl_ca_bundle"):
        cfg_data["ssl_ca_bundle"] = Path(cfg_data["ssl_ca_bundle"])
    return PipelineConfig(**cfg_data)


def validate_config(cfg: PipelineConfig) -> None:
    errors: List[str] = []
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
    if cfg.hdb_min_cluster_size < 2:
        errors.append("hdb_min_cluster_size must be >= 2")
    if cfg.hdb_min_samples is not None and cfg.hdb_min_samples < 1:
        errors.append("hdb_min_samples must be >= 1")
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
    if errors:
        msg = "Invalid configuration:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)


def config_to_yaml(cfg: PipelineConfig) -> str:
    data = cfg.__dict__.copy()
    data["cropped_images_dir"] = str(data.pop("input_dir"))
    data["output_dir"] = str(data["output_dir"])
    if data.get("ssl_ca_bundle") is not None:
        data["ssl_ca_bundle"] = str(data["ssl_ca_bundle"])
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
