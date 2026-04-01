from __future__ import annotations

"""Estimate a good background color distance threshold for auto-cropping.

The pipeline's auto-crop uses a simple rule: pixels close to the background
color are treated as background, and the remaining pixels are considered the
foreground object. This script samples pixels from a subset of images, measures
their Euclidean distance in RGB space from the provided background color, and
then applies Otsu's method to suggest a threshold that separates background
and foreground distances. It prints a recommended threshold plus summary stats
so you can choose a conservative or aggressive cutoff.

Typical usage:
- Provide a representative input folder.
- Pass the same --background-color you intend to use for clustering.
- Start with the suggested median threshold, then adjust based on visual checks.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


def _parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = [p for p in value.replace(" ", ",").split(",") if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("background color must be R,G,B")
    try:
        rgb = tuple(int(float(p)) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("background color must be numeric R,G,B") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("background color values must be between 0 and 255")
    return rgb


def _iter_images(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith((".jpg", ".jpeg", ".png")):
                yield Path(dirpath) / name


def _sample_distances(
    img: Image.Image,
    background_color: tuple[int, int, int],
    max_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    if arr.ndim != 3 or arr.shape[2] != 3:
        return np.empty(0, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    if max_pixels > 0 and flat.shape[0] > max_pixels:
        idx = rng.choice(flat.shape[0], size=max_pixels, replace=False)
        flat = flat[idx]
    ref = np.array(background_color, dtype=np.int16)
    diff = flat.astype(np.int16) - ref
    dist = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.int32)).astype(np.float32)
    return dist


def _otsu_threshold(distances: np.ndarray, bins: int = 256) -> float:
    if distances.size == 0:
        return 0.0
    max_val = float(distances.max())
    if max_val <= 0.0:
        return 0.0
    hist, bin_edges = np.histogram(distances, bins=bins, range=(0.0, max_val))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    prob = hist / total
    omega = np.cumsum(prob)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mu = np.cumsum(prob * bin_centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b_sq = (mu_total * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b_sq))
    return float(bin_centers[idx])


def estimate_threshold(
    root: Path,
    background_color: tuple[int, int, int],
    sample_images: int,
    max_pixels: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    paths = list(_iter_images(root))
    if not paths:
        raise ValueError(f"No images found in {root}")
    if sample_images > 0 and len(paths) > sample_images:
        paths = rng.choice(paths, size=sample_images, replace=False).tolist()

    thresholds = []
    for path in paths:
        try:
            with Image.open(path) as img:
                distances = _sample_distances(img, background_color, max_pixels, rng)
        except Exception:
            continue
        if distances.size == 0:
            continue
        thresholds.append(_otsu_threshold(distances))

    if not thresholds:
        raise ValueError("Could not compute thresholds from sampled images.")
    values = np.array(thresholds, dtype=np.float32)
    return {
        "count": int(values.size),
        "suggested_threshold": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate a background color distance threshold.")
    parser.add_argument("--input-dir", required=True, help="Folder with sample images")
    parser.add_argument(
        "--background-color",
        type=_parse_rgb,
        default=(45, 71, 159),
        help="Background RGB color as R,G,B",
    )
    parser.add_argument("--sample-images", type=int, default=20)
    parser.add_argument("--max-pixels", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = estimate_threshold(
        Path(args.input_dir),
        args.background_color,
        args.sample_images,
        args.max_pixels,
        args.seed,
    )
    print("Suggested --autocrop-threshold:", stats["suggested_threshold"])
    print(
        "Stats:",
        f"count={stats['count']} p25={stats['p25']:.2f} "
        f"p75={stats['p75']:.2f} min={stats['min']:.2f} max={stats['max']:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
