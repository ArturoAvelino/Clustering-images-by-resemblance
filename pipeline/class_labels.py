from __future__ import annotations

"""Utilities for extracting class labels from image filenames and counting them."""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional

DEFAULT_CLASS_ID_NUM_CHARACTERS = 4
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def _sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def extract_class_id_from_filename(
    filename: str,
    num_characters_to_read_class: int = DEFAULT_CLASS_ID_NUM_CHARACTERS,
) -> str:
    """
    Extract the class ID from the last characters before a filename extension.

    If the filename stem is shorter than ``num_characters_to_read_class``, the
    full stem is returned.
    """
    if num_characters_to_read_class <= 0:
        raise ValueError("num_characters_to_read_class must be greater than 0")
    stem, dot, _ = filename.rpartition(".")
    if not dot:
        stem = filename
    return stem[-num_characters_to_read_class:]


def _is_supported_image_filename(filename: str) -> bool:
    _, dot, extension = filename.rpartition(".")
    if not dot:
        return False
    return f".{extension.lower()}" in SUPPORTED_IMAGE_EXTENSIONS


def iter_image_filenames(root_dir: Path) -> Iterable[str]:
    """Yield image filenames from ``root_dir`` recursively with low overhead."""
    stack = [os.fspath(root_dir)]
    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                            continue
                        if _is_supported_image_filename(entry.name) and entry.is_file(
                            follow_symlinks=False
                        ):
                            yield entry.name
                    except OSError:
                        continue
        except OSError:
            continue


def count_classes_in_labeled_filenames(
    files_dir: Path,
    num_characters_to_read_class: int,
) -> Counter[str]:
    """
    Count image files by the class ID encoded in the end of each filename stem.

    The directory traversal is recursive and uses ``os.scandir`` to minimize
    per-file overhead when scanning very large datasets.
    """
    counts: Counter[str] = Counter()
    for filename in iter_image_filenames(files_dir):
        class_id = extract_class_id_from_filename(filename, num_characters_to_read_class)
        counts[class_id] += 1
    return counts


def write_classes_in_dataset_csv(
    counts: Mapping[str, int],
    output_path: Path,
) -> Path:
    """Write ``classes_in_dataset.csv`` with ``class_ID`` and ``num_objs`` columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_ID", "num_objs"])
        for class_id in sorted(counts.keys(), key=_sort_key):
            writer.writerow([class_id, counts[class_id]])
    return output_path


def generate_classes_in_dataset_csv(
    files_dir: Path,
    output_dir: Path,
    num_characters_to_read_class: int,
) -> Path:
    """Scan ``files_dir`` and write ``classes_in_dataset.csv`` into ``output_dir``."""
    counts = count_classes_in_labeled_filenames(
        files_dir=files_dir,
        num_characters_to_read_class=num_characters_to_read_class,
    )
    return write_classes_in_dataset_csv(
        counts=counts,
        output_path=output_dir / "classes_in_dataset.csv",
    )


def build_count_classes_parser(
    *,
    prog: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Build the CLI parser for ``count-classes-on-labeled-filenames``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Count image files by the class ID stored in the last characters of "
            "each filename stem and write classes_in_dataset.csv."
        ),
    )
    parser.add_argument(
        "--files-dir",
        type=Path,
        required=True,
        help="Directory containing the labeled image files to scan recursively.",
    )
    parser.add_argument(
        "--num-characters-to-read-class",
        type=int,
        required=True,
        help="Number of trailing filename-stem characters to use as the class ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where classes_in_dataset.csv will be written.",
    )
    return parser


def parse_count_classes_args(
    argv: Optional[list[str]] = None,
    *,
    prog: Optional[str] = None,
) -> argparse.Namespace:
    """Parse CLI arguments for the filename-based class counting command."""
    return build_count_classes_parser(prog=prog).parse_args(argv)


def main(argv: Optional[list[str]] = None, *, prog: Optional[str] = None) -> int:
    """CLI entry point for ``count-classes-on-labeled-filenames``."""
    args = parse_count_classes_args(argv, prog=prog)
    if args.num_characters_to_read_class <= 0:
        raise SystemExit("--num-characters-to-read-class must be greater than 0")
    if not args.files_dir.exists():
        raise SystemExit(f"--files-dir does not exist: {args.files_dir}")
    if not args.files_dir.is_dir():
        raise SystemExit(f"--files-dir is not a directory: {args.files_dir}")
    generate_classes_in_dataset_csv(
        files_dir=args.files_dir,
        output_dir=args.output_dir,
        num_characters_to_read_class=args.num_characters_to_read_class,
    )
    return 0
