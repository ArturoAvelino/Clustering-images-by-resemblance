#!/usr/bin/env python3
"""Copy clustered images into cluster-named folders.

This script is the implementation behind the CLI command
``python clustering copy-crops-to-cluster-dirs`` (or you can run it directly
via ``python copy_crops_to_cluster_dirs.py``).

What this script does
---------------------
Reads a ``clusters.csv`` file produced by the pipeline and copies image files
into subfolders named after their cluster label (the value in the ``cluster``
column). The CSV can contain extra metadata columns (for example probabilities
or outlier scores); only ``image_id`` and ``cluster`` are required unless you
enable confidence/outlier subdirectories. The script uses the ``image_id``
column to locate each image relative to ``--input-dir`` and then copies the
file into a cluster folder. If a matching ``.JSON`` file exists with the same
basename as the image, it is copied alongside the image into the same cluster
folder. This makes it easy to review clustered outputs in a filesystem browser
while keeping the original folder structure intact (unless ``--flat`` is
specified).

Inputs
------
- ``--clusters``: Path to the ``clusters.csv`` file. The CSV must have columns
  ``image_id`` and ``cluster`` (extra columns are ignored unless you enable
  confidence/outlier subdirectories).
- ``--input-dir``: Root directory for the images listed in ``image_id``. The
  pipeline writes ``image_id`` values as paths relative to the input directory.
- ``--subdir-confidence``: Requires ``probabilities`` and ``outlier_scores``
  columns in ``clusters.csv``.
- ``--subdir-outliers``: Requires ``outlier_scores`` in ``clusters.csv``.

Outputs
-------
- Creates a folder per cluster label (e.g., ``0/``, ``1/``, ``-1/``) and copies
  images into those folders.
- By default, the folder structure under each cluster mirrors the original
  relative paths; use ``--flat`` to avoid nested subfolders.
- Optionally creates per-cluster subdirectories for representative images and
  outliers using ``--subdir-confidence`` and ``--subdir-outliers``. These copy
  images only (not JSON) and are ignored when ``--json-only`` is set.
- Use ``--json-only`` to copy only the matching ``.JSON`` files while leaving
  images in place.
- Prints a summary of copied, skipped, and missing files, including JSON copied
  without images.

Usage
-----
Copy images in place (cluster folders created inside the input directory)::

  python clustering copy-crops-to-cluster-dirs --clusters /path/to/output/clusters.csv \\
    --input-dir /path/to/images

Copy images into a separate destination root::

  python clustering copy-crops-to-cluster-dirs --clusters /path/to/output/clusters.csv \\
    --input-dir /path/to/images --dest-dir /path/to/clustered

Preview changes without copying files::

  python clustering copy-crops-to-cluster-dirs --clusters /path/to/output/clusters.csv \\
    --input-dir /path/to/images --dry-run

Copy high-confidence representatives and outliers into subdirectories::

  python clustering copy-crops-to-cluster-dirs --clusters /path/to/output/clusters.csv \\
    --input-dir /path/to/images --dest-dir /path/to/clustered \\
    --subdir-confidence 0.9 --subdir-outliers 0.8

Conflict handling
-----------------
If a destination file already exists, the default behavior is to rename the
incoming file by appending ``_1``, ``_2``, etc. Use ``--on-conflict`` to change
this to ``overwrite``, ``skip``, or ``error``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy images into subfolders named after their cluster label "
            "from clusters.csv."
        )
    )
    parser.add_argument(
        "--clusters",
        required=True,
        type=Path,
        help="Path to clusters.csv produced by the pipeline.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Root directory that image_id values are relative to.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help=(
            "Directory to create cluster folders in. Defaults to --input-dir."
        ),
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Place all images directly inside each cluster folder (no subfolders).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without changing any files.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help=(
            "Copy only matching .JSON files and leave images in place. "
            "Missing images are still counted when JSON is copied."
        ),
    )
    parser.add_argument(
        "--subdir-confidence",
        type=_parse_threshold,
        default=None,
        help=(
            "Create a per-cluster subdirectory named "
            "representatives_confid_XX and copy images with probabilities >= XX "
            "and outlier_scores <= 0.01. Ignored with --json-only."
        ),
    )
    parser.add_argument(
        "--subdir-outliers",
        type=_parse_threshold,
        default=None,
        help=(
            "Create a per-cluster subdirectory named outliers_YY and copy "
            "images with outlier_scores >= YY. Ignored with --json-only."
        ),
    )
    parser.add_argument(
        "--on-conflict",
        choices=("rename", "overwrite", "skip", "error"),
        default="rename",
        help="What to do if the destination file already exists.",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class ThresholdSpec:
    value: float
    label: str


def _parse_threshold(raw: str) -> ThresholdSpec:
    raw = raw.strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid float value: {raw!r}"
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            "Value must be between 0.0 and 1.0."
        )
    return ThresholdSpec(value=value, label=raw)


def _normalize_cluster(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw


def _parse_float_field(row: dict[str, str], field: str) -> float | None:
    raw = (row.get(field) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _resolve_conflict(dest: Path, mode: str) -> Path | None:
    if not dest.exists():
        return dest
    if mode == "overwrite":
        if dest.is_file():
            dest.unlink()
        return dest
    if mode == "skip":
        return None
    if mode == "error":
        raise FileExistsError(dest)
    # mode == "rename"
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    for idx in range(1, 10_000):
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not resolve name conflict for {dest}")


def _resolve_conflict_pair(
    dest_image: Path,
    dest_json: Path,
    mode: str,
) -> tuple[Path, Path] | None:
    if mode == "overwrite":
        if dest_image.exists() and dest_image.is_file():
            dest_image.unlink()
        if dest_json.exists() and dest_json.is_file():
            dest_json.unlink()
        return dest_image, dest_json

    if mode == "skip":
        if dest_image.exists() or dest_json.exists():
            return None
        return dest_image, dest_json

    if mode == "error":
        if dest_image.exists():
            raise FileExistsError(dest_image)
        if dest_json.exists():
            raise FileExistsError(dest_json)
        return dest_image, dest_json

    # mode == "rename"
    if not dest_image.exists() and not dest_json.exists():
        return dest_image, dest_json

    stem = dest_image.stem
    suffix = dest_image.suffix
    parent = dest_image.parent
    for idx in range(1, 10_000):
        candidate_image = parent / f"{stem}_{idx}{suffix}"
        candidate_json = candidate_image.with_suffix(dest_json.suffix)
        if not candidate_image.exists() and not candidate_json.exists():
            return candidate_image, candidate_json
    raise RuntimeError(f"Could not resolve name conflict for {dest_image}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    clusters_path = args.clusters
    input_dir = args.input_dir
    dest_dir = args.dest_dir or input_dir

    if not clusters_path.exists():
        print(f"clusters.csv not found: {clusters_path}", file=sys.stderr)
        return 2
    if not input_dir.exists():
        print(f"input directory not found: {input_dir}", file=sys.stderr)
        return 2

    copied_images = 0
    copied_json = 0
    copied_json_without_image = 0
    copied_confidence = 0
    copied_outliers = 0
    skipped = 0
    missing_images = 0
    missing_json = 0

    with clusters_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("clusters.csv has no header row", file=sys.stderr)
            return 2
        if "image_id" not in reader.fieldnames or "cluster" not in reader.fieldnames:
            print(
                "clusters.csv must have columns: image_id, cluster",
                file=sys.stderr,
            )
            return 2
        if args.subdir_confidence is not None:
            if "probabilities" not in reader.fieldnames:
                print(
                    "clusters.csv must include probabilities when "
                    "--subdir-confidence is used",
                    file=sys.stderr,
                )
                return 2
            if "outlier_scores" not in reader.fieldnames:
                print(
                    "clusters.csv must include outlier_scores when "
                    "--subdir-confidence is used",
                    file=sys.stderr,
                )
                return 2
        if args.subdir_outliers is not None and "outlier_scores" not in reader.fieldnames:
            print(
                "clusters.csv must include outlier_scores when "
                "--subdir-outliers is used",
                file=sys.stderr,
            )
            return 2

        for row in reader:
            rel = (row.get("image_id") or "").strip()
            cluster_raw = row.get("cluster")
            if not rel or cluster_raw is None:
                skipped += 1
                continue

            cluster = _normalize_cluster(cluster_raw)
            if not cluster:
                skipped += 1
                continue

            rel_path = Path(rel)
            src = input_dir / rel_path
            json_rel = rel_path.with_suffix(".JSON")
            json_src = input_dir / json_rel

            image_exists = src.exists()
            json_exists = json_src.exists()

            if args.flat:
                dest = dest_dir / cluster / rel_path.name
                dest_json = dest_dir / cluster / json_rel.name
            else:
                dest = dest_dir / cluster / rel_path
                dest_json = dest_dir / cluster / json_rel

            if args.json_only:
                if not json_exists:
                    missing_json += 1
                    continue
                dest_json = _resolve_conflict(dest_json, args.on_conflict)
                if dest_json is None:
                    skipped += 1
                    continue
            else:
                if not image_exists and not json_exists:
                    missing_images += 1
                    missing_json += 1
                    continue
                if image_exists and json_exists:
                    resolved = _resolve_conflict_pair(dest, dest_json, args.on_conflict)
                    if resolved is None:
                        skipped += 1
                        continue
                    dest, dest_json = resolved
                elif image_exists:
                    dest = _resolve_conflict(dest, args.on_conflict)
                    if dest is None:
                        skipped += 1
                        continue
                elif json_exists:
                    dest_json = _resolve_conflict(dest_json, args.on_conflict)
                    if dest_json is None:
                        skipped += 1
                        continue

            if args.dry_run:
                if args.json_only:
                    print(f"{json_src} -> {dest_json}")
                    copied_json += 1
                    if not image_exists:
                        copied_json_without_image += 1
                else:
                    if image_exists:
                        print(f"{src} -> {dest}")
                        copied_images += 1
                    else:
                        missing_images += 1
                    if json_exists:
                        print(f"{json_src} -> {dest_json}")
                        copied_json += 1
                        if not image_exists:
                            copied_json_without_image += 1
                    else:
                        missing_json += 1
                    if image_exists:
                        if args.subdir_confidence is not None:
                            prob = _parse_float_field(row, "probabilities")
                            outlier = _parse_float_field(row, "outlier_scores")
                            if (
                                prob is not None
                                and outlier is not None
                                and prob >= args.subdir_confidence.value
                                and outlier <= 0.01
                            ):
                                subdir = (
                                    dest_dir
                                    / cluster
                                    / f"representatives_confid_{args.subdir_confidence.label}"
                                )
                                sub_dest = (
                                    subdir / rel_path.name
                                    if args.flat
                                    else subdir / rel_path
                                )
                                print(f"{src} -> {sub_dest}")
                                copied_confidence += 1
                        if args.subdir_outliers is not None:
                            outlier = _parse_float_field(row, "outlier_scores")
                            if outlier is not None and outlier >= args.subdir_outliers.value:
                                subdir = (
                                    dest_dir
                                    / cluster
                                    / f"outliers_{args.subdir_outliers.label}"
                                )
                                sub_dest = (
                                    subdir / rel_path.name
                                    if args.flat
                                    else subdir / rel_path
                                )
                                print(f"{src} -> {sub_dest}")
                                copied_outliers += 1
                continue

            if args.json_only:
                dest_json.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(json_src), str(dest_json))
                copied_json += 1
                if not image_exists:
                    copied_json_without_image += 1
            else:
                if image_exists:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dest))
                    copied_images += 1
                else:
                    missing_images += 1

                if json_exists:
                    dest_json.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(json_src), str(dest_json))
                    copied_json += 1
                    if not image_exists:
                        copied_json_without_image += 1
                else:
                    missing_json += 1

                if image_exists:
                    if args.subdir_confidence is not None:
                        prob = _parse_float_field(row, "probabilities")
                        outlier = _parse_float_field(row, "outlier_scores")
                        if (
                            prob is not None
                            and outlier is not None
                            and prob >= args.subdir_confidence.value
                            and outlier <= 0.01
                        ):
                            subdir = (
                                dest_dir
                                / cluster
                                / f"representatives_confid_{args.subdir_confidence.label}"
                            )
                            sub_dest = (
                                subdir / rel_path.name
                                if args.flat
                                else subdir / rel_path
                            )
                            resolved = _resolve_conflict(sub_dest, args.on_conflict)
                            if resolved is None:
                                skipped += 1
                            else:
                                resolved.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(str(src), str(resolved))
                                copied_confidence += 1
                    if args.subdir_outliers is not None:
                        outlier = _parse_float_field(row, "outlier_scores")
                        if outlier is not None and outlier >= args.subdir_outliers.value:
                            subdir = (
                                dest_dir
                                / cluster
                                / f"outliers_{args.subdir_outliers.label}"
                            )
                            sub_dest = (
                                subdir / rel_path.name
                                if args.flat
                                else subdir / rel_path
                            )
                            resolved = _resolve_conflict(sub_dest, args.on_conflict)
                            if resolved is None:
                                skipped += 1
                            else:
                                resolved.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(str(src), str(resolved))
                                copied_outliers += 1

    print(f"Copied images: {copied_images}")
    if copied_json:
        print(f"Copied JSON: {copied_json}")
    if copied_json_without_image:
        print(f"Copied JSON without image: {copied_json_without_image}")
    if copied_confidence:
        print(f"Copied confidence representatives: {copied_confidence}")
    if copied_outliers:
        print(f"Copied outlier images: {copied_outliers}")
    if skipped:
        print(f"Skipped: {skipped}")
    if missing_images:
        print(f"Missing images: {missing_images}")
    if missing_json:
        print(f"Missing JSON: {missing_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
