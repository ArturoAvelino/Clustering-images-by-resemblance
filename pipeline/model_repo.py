from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .config import PipelineConfig


def _is_dinov2_repo(path: Path) -> bool:
    return (path / "hubconf.py").exists()


def _local_repo_hint() -> Optional[str]:
    for key in ("DINOv2_REPO", "DINOV2_REPO"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def auto_model_repo(cfg: PipelineConfig) -> Optional[str]:
    if cfg.dino_model:
        return None

    for key in ("DINOv2_REPO", "DINOV2_REPO"):
        value = os.environ.get(key)
        if value:
            path = Path(value).expanduser()
            if path.exists() and _is_dinov2_repo(path):
                print(f"Using local DINOv2 repo from {key}: {path}")
                return str(path)

    candidates = [
        Path.cwd() / "dinov2",
        Path.home() / "dinov2",
        Path.home() / "PycharmProjects" / "dinov2",
        Path.home() / "Projects" / "dinov2",
    ]
    for path in candidates:
        if path.exists() and _is_dinov2_repo(path):
            print(f"Using local DINOv2 repo at {path}")
            return str(path)
    return None


__all__ = ["auto_model_repo", "_local_repo_hint"]
