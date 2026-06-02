from __future__ import annotations

"""Utilities for counting class IDs in filenames and enriching them with names."""

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional

DEFAULT_CLASS_ID_NUM_CHARACTERS = 4
STRICT_JPG_CLASS_ID_PATTERN = re.compile(r"_class_(\d{4})\.jpg$")
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


def extract_strict_jpg_class_id(filename: str) -> str | None:
    """
    Return the class ID only when ``filename`` ends with ``_class_1234.jpg``.

    The match is evaluated against the basename only. Filenames that do not
    contain ``_class_`` immediately before a 4-digit class ID and the ``.jpg``
    extension are treated as unlabeled and return ``None``.
    """
    match = STRICT_JPG_CLASS_ID_PATTERN.search(Path(filename).name)
    if match is None:
        return None
    return match.group(1)


def is_strictly_labeled_jpg(filename: str) -> bool:
    """
    Return ``True`` when ``filename`` encodes a 4-digit JPG class label.

    A filename is treated as labeled only when its basename ends with the
    exact pattern ``_class_1234.jpg``. This matches the class-validation rule
    used by the cluster summary outputs.
    """
    return extract_strict_jpg_class_id(filename) is not None


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


def load_bigle_id_to_name_map(csv_path: Path) -> dict[str, str]:
    """
    Load a BIGLE label CSV into a mapping from class ID to class name.

    The file must include ``id`` and ``name`` columns. Rows with an empty
    ``id`` value are ignored. If the same ``id`` appears multiple times, the
    last non-empty ``name`` value wins.
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"BIGLE IDs CSV is empty: {csv_path}")
        missing_columns = {"id", "name"} - set(reader.fieldnames)
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"BIGLE IDs CSV must contain columns {missing_list}: {csv_path}"
            )

        id_to_name: dict[str, str] = {}
        for row in reader:
            class_id = str(row.get("id", "")).strip()
            if not class_id:
                continue
            class_name = str(row.get("name", "")).strip()
            if class_name or class_id not in id_to_name:
                id_to_name[class_id] = class_name
    return id_to_name


def write_classes_in_dataset_csv(
    counts: Mapping[str, int],
    output_path: Path,
    *,
    class_names: Optional[Mapping[str, str]] = None,
) -> Path:
    """
    Write ``classes_in_dataset.csv``.

    When ``class_names`` is provided, the CSV contains ``class_ID``,
    ``class_name``, and ``num_objs`` columns. Otherwise it keeps the legacy
    ``class_ID`` and ``num_objs`` columns.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if class_names is None:
            writer.writerow(["class_ID", "num_objs"])
        else:
            writer.writerow(["class_ID", "class_name", "num_objs"])
        for class_id in sorted(counts.keys(), key=_sort_key):
            if class_names is None:
                writer.writerow([class_id, counts[class_id]])
            else:
                writer.writerow([class_id, class_names.get(class_id, ""), counts[class_id]])
    return output_path


def generate_classes_in_dataset_csv(
    files_dir: Path,
    output_dir: Path,
    num_characters_to_read_class: int,
    *,
    biigle_id_to_names_file: Optional[Path] = None,
) -> Path:
    """
    Scan ``files_dir`` and write ``classes_in_dataset.csv`` into ``output_dir``.

    When ``biigle_id_to_names_file`` is provided, class names are loaded from
    its ``id`` and ``name`` columns and written alongside each ``class_ID``.
    """
    counts = count_classes_in_labeled_filenames(
        files_dir=files_dir,
        num_characters_to_read_class=num_characters_to_read_class,
    )
    class_names = None
    if biigle_id_to_names_file is not None:
        class_names = load_bigle_id_to_name_map(biigle_id_to_names_file)
    return write_classes_in_dataset_csv(
        counts=counts,
        output_path=output_dir / "classes_in_dataset.csv",
        class_names=class_names,
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
            "each filename stem and write classes_in_dataset.csv. Optionally "
            "enrich the output with class names from a BIGLE labels CSV."
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
        "--biigleID-to-names-file",
        type=Path,
        help=(
            "Optional BIGLE labels CSV with 'id' and 'name' columns. When "
            "provided, classes_in_dataset.csv also includes a class_name column."
        ),
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
    if args.biigleID_to_names_file is not None:
        if not args.biigleID_to_names_file.exists():
            raise SystemExit(
                "--biigleID-to-names-file does not exist: "
                f"{args.biigleID_to_names_file}"
            )
        if not args.biigleID_to_names_file.is_file():
            raise SystemExit(
                "--biigleID-to-names-file is not a file: "
                f"{args.biigleID_to_names_file}"
            )
        try:
            args.biigleID_to_names_file = args.biigleID_to_names_file.resolve(
                strict=True
            )
        except OSError as exc:
            raise SystemExit(
                "--biigleID-to-names-file could not be resolved: "
                f"{args.biigleID_to_names_file} ({exc})"
            ) from exc
    try:
        generate_classes_in_dataset_csv(
            files_dir=args.files_dir,
            output_dir=args.output_dir,
            num_characters_to_read_class=args.num_characters_to_read_class,
            biigle_id_to_names_file=args.biigleID_to_names_file,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0
