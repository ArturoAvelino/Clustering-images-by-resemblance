from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path, PurePosixPath
from typing import List, Optional


DEFAULT_IMAGE_ROOT = "/home/ubuntu/crops/cropped"
DEFAULT_SAMPLE_SIZE = 100


def generate_image_sample_files(
    clusters_path: Path,
    output_dir: Path,
    *,
    image_root: str = DEFAULT_IMAGE_ROOT,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
) -> tuple[Path, Path]:
    """Sample up to ``sample_size`` rows per cluster from a clusters CSV.

    Rows are selected uniformly without replacement using reservoir sampling,
    so the input is read once and memory use is proportional to the number of
    clusters times the requested sample size, rather than the CSV size.
    Selected rows are written in their original file order.
    """
    clusters_path = Path(clusters_path)
    output_dir = Path(output_dir)
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if not clusters_path.is_file():
        raise FileNotFoundError(f"clusters.csv not found: {clusters_path}")

    sampling_path = output_dir / "clusters_sampling.csv"
    file_list_path = output_dir / "file_list.txt"
    if clusters_path.resolve() == sampling_path.resolve():
        raise ValueError("The output clusters_sampling.csv must not overwrite the input CSV")

    rng = random.Random(seed)
    seen_by_cluster: dict[str, int] = {}
    samples_by_cluster: dict[str, list[tuple[int, list[str]]]] = {}

    with clusters_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"clusters.csv is empty: {clusters_path}") from exc
        if "cluster" not in header or "image_id" not in header:
            raise ValueError("clusters.csv must contain 'cluster' and 'image_id' columns")
        cluster_column = header.index("cluster")
        image_column = header.index("image_id")

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Expected {len(header)} columns but found {len(row)} "
                    f"in {clusters_path} at line {line_number}"
                )
            cluster = row[cluster_column].strip()
            image_id = row[image_column].strip()
            if not cluster or not image_id:
                raise ValueError(
                    f"Missing cluster or image_id in {clusters_path} at line {line_number}"
                )

            seen = seen_by_cluster.get(cluster, 0) + 1
            seen_by_cluster[cluster] = seen
            cluster_samples = samples_by_cluster.get(cluster)
            if cluster_samples is None:
                cluster_samples = []
                samples_by_cluster[cluster] = cluster_samples
            item = (line_number, row)
            if len(cluster_samples) < sample_size:
                cluster_samples.append(item)
            else:
                replacement_index = rng.randrange(seen)
                if replacement_index < sample_size:
                    cluster_samples[replacement_index] = item

    selected_rows = sorted(
        (item for cluster_samples in samples_by_cluster.values() for item in cluster_samples),
        key=lambda item: item[0],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root_path = PurePosixPath(image_root.rstrip("/"))

    with sampling_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(header)
        writer.writerows(row for _, row in selected_rows)

    with file_list_path.open("w", encoding="utf-8", newline="") as destination:
        for _, row in selected_rows:
            image_id = row[image_column].strip().replace("\\", "/")
            filename = PurePosixPath(image_id).name
            destination.write(f"{image_root_path / filename}\n")

    return file_list_path, sampling_path


def build_image_sample_parser(
    *, prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        add_help=add_help,
        description="Randomly sample images per cluster for download.",
    )
    parser.add_argument("--clusters-file", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where file_list.txt and clusters_sampling.csv are written.",
    )
    parser.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
        help=f"Image directory used to build download paths (default: {DEFAULT_IMAGE_ROOT}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Maximum number of rows sampled per cluster (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument("--seed", type=int, help="Optional seed for repeatable sampling.")
    return parser


def generate_image_sample_files_main(
    argv: Optional[List[str]] = None, *, prog: Optional[str] = None
) -> int:
    args = build_image_sample_parser(prog=prog).parse_args(argv)
    file_list_path, sampling_path = generate_image_sample_files(
        args.clusters_file,
        args.output_dir,
        image_root=args.image_root,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Wrote {file_list_path}")
    print(f"Wrote {sampling_path}")
    return 0
