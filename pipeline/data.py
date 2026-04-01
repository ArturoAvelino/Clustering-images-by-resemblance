from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image, ImageFile

from .config import PipelineConfig
from .torch_utils import Dataset, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageIndex:
    def __init__(
        self,
        root: Path,
        index_path: Path,
        min_kbytes: Optional[float] = None,
        max_kbytes: Optional[float] = None,
    ) -> None:
        self.root = root
        self.index_path = index_path
        self.min_kbytes = min_kbytes
        self.max_kbytes = max_kbytes

    def build(self, max_images: Optional[int] = None) -> List[str]:
        paths: List[str] = []
        for path in sorted(self._iter_images(self.root, self.min_kbytes, self.max_kbytes)):
            rel = str(path.relative_to(self.root))
            paths.append(rel)
            if max_images is not None and len(paths) >= max_images:
                break
        self.write(paths)
        return paths

    def load(self) -> List[str]:
        with self.index_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def write(self, rel_paths: List[str]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as f:
            for rel in rel_paths:
                f.write(rel + "\n")

    @staticmethod
    def _iter_images(
        root: Path, min_kbytes: Optional[float], max_kbytes: Optional[float]
    ) -> Iterable[Path]:
        min_bytes = None if min_kbytes is None else float(min_kbytes) * 1024.0
        max_bytes = None if max_kbytes is None else float(max_kbytes) * 1024.0
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                lower = name.lower()
                if lower.endswith(".jpg") or lower.endswith(".jpeg"):
                    path = Path(dirpath) / name
                    if min_bytes is not None or max_bytes is not None:
                        try:
                            size_bytes = path.stat().st_size
                        except OSError:
                            continue
                        if min_bytes is not None and size_bytes < min_bytes:
                            continue
                        if max_bytes is not None and size_bytes > max_bytes:
                            continue
                    yield path


def _foreground_mask(
    arr: np.ndarray, background_color: tuple[int, int, int], distance: float
) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3:
        return np.zeros(arr.shape[:2], dtype=bool)
    ref = np.array(background_color, dtype=np.int16)
    diff = arr.astype(np.int16) - ref
    dist_sq = np.sum(diff * diff, axis=2)
    return dist_sq > float(distance) * float(distance)


class AutoCropNonBackground:
    def __init__(
        self,
        background_color: tuple[int, int, int],
        distance: float = 35,
        padding: int = 2,
    ) -> None:
        self.background_color = background_color
        self.distance = distance
        self.padding = padding

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img)
        mask = _foreground_mask(arr, self.background_color, self.distance)
        if not mask.any():
            return img
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        if self.padding > 0:
            y0 = max(0, y0 - self.padding)
            x0 = max(0, x0 - self.padding)
            y1 = min(arr.shape[0], y1 + self.padding)
            x1 = min(arr.shape[1], x1 + self.padding)
        return img.crop((int(x0), int(y0), int(x1), int(y1)))


class PadToSquare:
    def __init__(self, fill: tuple[int, int, int]) -> None:
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        canvas = Image.new("RGB", (size, size), self.fill)
        canvas.paste(img, ((size - w) // 2, (size - h) // 2))
        return canvas


def build_transform(cfg: PipelineConfig):
    ops = []
    if cfg.autocrop:
        ops.append(
            AutoCropNonBackground(
                cfg.background_color, cfg.autocrop_threshold, cfg.autocrop_padding
            )
        )
    ops.append(PadToSquare(cfg.background_color))
    ops.append(transforms.Resize((cfg.img_size, cfg.img_size)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return transforms.Compose(ops)


def compute_size_features(
    root: Path, rel_paths: List[str], background_color: tuple[int, int, int], distance: float
) -> np.ndarray:
    sizes = np.zeros(len(rel_paths), dtype=np.float32)
    for idx, rel in enumerate(rel_paths):
        path = root / rel
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                arr = np.asarray(img)
        except Exception:
            continue
        mask = _foreground_mask(arr, background_color, distance)
        if mask.any():
            sizes[idx] = float(mask.mean())
    return sizes


class ImageDataset(Dataset):
    def __init__(self, root: Path, rel_paths: List[str], transform) -> None:
        self.root = root
        self.rel_paths = rel_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        rel = self.rel_paths[idx]
        path = self.root / rel
        img = Image.open(path).convert("RGB")
        if self.transform is None:
            return img
        return self.transform(img)
