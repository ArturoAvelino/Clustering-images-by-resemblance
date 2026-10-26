"""Create a filtered clusters CSV with row-aligned UMAP vectors from a ``.npy`` file."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
import tempfile
from typing import List, Optional

import numpy as np


def _cluster_key(value: str) -> str:
    """Normalize numeric cluster labels so ``1`` and ``1.0`` are equivalent."""
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return value


def _read_summary_counts(summary_file: Path) -> dict[str, int] | None:
    """Read cluster counts from a sibling summary when it has a supported schema."""
    if not summary_file.is_file():
        return None

    try:
        with summary_file.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                return None
            cluster_column = (
                "cluster_id"
                if "cluster_id" in reader.fieldnames
                else "acluster_id"
                if "acluster_id" in reader.fieldnames
                else None
            )
            if cluster_column is None or "num_objs_in_cluster" not in reader.fieldnames:
                return None

            counts: dict[str, int] = {}
            for row in reader:
                cluster = _cluster_key(row.get(cluster_column) or "")
                raw_count = (row.get("num_objs_in_cluster") or "").strip()
                if not cluster or not raw_count:
                    return None
                count = int(raw_count)
                if count < 0 or cluster in counts:
                    return None
                counts[cluster] = count
            return counts
    except (OSError, ValueError):
        return None


def _count_clusters(clusters_file: Path) -> Counter[str]:
    """Count rows per cluster by streaming the assignments CSV."""
    counts: Counter[str] = Counter()
    with clusters_file.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"clusters CSV is empty: {clusters_file}") from exc
        if header.count("cluster") != 1:
            raise ValueError("clusters CSV must contain exactly one 'cluster' column.")
        cluster_index = header.index("cluster")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Expected {len(header)} columns but found {len(row)} "
                    f"in {clusters_file} at line {line_number}."
                )
            cluster = _cluster_key(row[cluster_index])
            if cluster:
                counts[cluster] += 1
    return counts


def generate_clusters_and_vectors_file(
    clusters_file: Path,
    umap_file: Path,
    output_dir: Path,
    *,
    min_cluster_size: int = 100,
    max_cluster_size: int = 1_000_000,
    output_name: str = "clusters_and_vectors.csv",
) -> Path:
    """Write rows from qualifying clusters and add each row's UMAP vector.

    The UMAP row at index *i* is attached to CSV data row *i*. This matches the
    pipeline artifacts, where ``clusters.csv`` and ``umap.npy`` use the same
    image order. Only clusters with strictly more than ``min_cluster_size``
    rows and strictly fewer than ``max_cluster_size`` rows are written. A
    sibling ``clusters_summary.csv`` is used to obtain cluster counts when it
    is available and has a supported schema; its counts are checked against
    the assignments during generation. Otherwise, counts are collected in a
    streaming pass over ``clusters.csv``.

    The CSV is streamed and the UMAP array is memory-mapped, keeping working
    memory bounded by cluster count maps, one CSV row, and one UMAP vector.

    The output column is placed immediately after ``outlier_scores``. Existing
    ``dim_reduction`` columns are replaced in that position. UMAP values are
    rounded to four decimal places to match the pipeline's CSV representation.

    Args:
        clusters_file: Existing cluster assignments CSV.
        umap_file: Two-dimensional NumPy ``.npy`` array with one vector per CSV
            data row, in the same order.
        output_dir: Directory where the output CSV is written.
        min_cluster_size: Exclude clusters with this many rows or fewer.
            Defaults to 100.
        max_cluster_size: Exclude clusters with this many rows or more.
            Defaults to 1,000,000.
        output_name: Output CSV filename. Defaults to
            ``clusters_and_vectors.csv``.

    Returns:
        The path to the generated CSV file.

    Raises:
        FileNotFoundError: If either input file does not exist.
        ValueError: If the UMAP array shape or CSV header is invalid, or the
            row counts do not match, the size bounds are invalid, or a sibling
            summary disagrees with the cluster assignments.
    """
    clusters_file = Path(clusters_file)
    umap_file = Path(umap_file)
    output_dir = Path(output_dir)
    output_path = output_dir / output_name

    if min_cluster_size < 0 or max_cluster_size <= min_cluster_size:
        raise ValueError(
            "Cluster size bounds must satisfy 0 <= min_cluster_size < max_cluster_size."
        )

    if not clusters_file.is_file():
        raise FileNotFoundError(f"clusters CSV not found: {clusters_file}")
    if not umap_file.is_file():
        raise FileNotFoundError(f"UMAP array not found: {umap_file}")
    if output_path.resolve() in {clusters_file.resolve(), umap_file.resolve()}:
        raise ValueError("The output file must not overwrite either input file.")

    umap_vectors = np.load(umap_file, mmap_mode="r", allow_pickle=False)
    if umap_vectors.ndim != 2 or umap_vectors.shape[1] == 0:
        raise ValueError(
            f"UMAP array must be a non-empty-width 2D array; got shape {umap_vectors.shape}."
        )

    summary_counts = _read_summary_counts(
        clusters_file.with_name("clusters_summary.csv")
    )
    cluster_counts = (
        summary_counts if summary_counts is not None else _count_clusters(clusters_file)
    )
    included_clusters = {
        cluster
        for cluster, count in cluster_counts.items()
        if min_cluster_size < count < max_cluster_size
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    rows_seen = 0
    observed_cluster_counts: Counter[str] = Counter()
    try:
        with clusters_file.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"clusters CSV is empty: {clusters_file}") from exc
            if header.count("outlier_scores") != 1:
                raise ValueError(
                    "clusters CSV must contain exactly one 'outlier_scores' column."
                )
            if header.count("cluster") != 1:
                raise ValueError("clusters CSV must contain exactly one 'cluster' column.")
            if header.count("dim_reduction") > 1:
                raise ValueError(
                    "clusters CSV must not contain duplicate 'dim_reduction' columns."
                )

            output_header = [name for name in header if name != "dim_reduction"]
            outlier_index = output_header.index("outlier_scores")
            output_header.insert(outlier_index + 1, "dim_reduction")

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=output_dir,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.writer(destination)
                writer.writerow(output_header)

                input_dim_index = (
                    header.index("dim_reduction")
                    if "dim_reduction" in header
                    else None
                )
                cluster_index = header.index("cluster")
                for line_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"Expected {len(header)} columns but found {len(row)} "
                            f"in {clusters_file} at line {line_number}."
                        )
                    if rows_seen >= umap_vectors.shape[0]:
                        raise ValueError(
                            "clusters CSV has more data rows than umap.npy "
                            f"({rows_seen + 1} > {umap_vectors.shape[0]})."
                        )

                    cluster = _cluster_key(row[cluster_index])
                    if cluster:
                        observed_cluster_counts[cluster] += 1

                    if cluster not in included_clusters:
                        rows_seen += 1
                        continue

                    output_row = [
                        value
                        for index, value in enumerate(row)
                        if index != input_dim_index
                    ]
                    vector = [
                        round(float(value), 4)
                        for value in umap_vectors[rows_seen]
                    ]
                    output_row.insert(
                        outlier_index + 1,
                        json.dumps(vector, separators=(",", ":")),
                    )
                    writer.writerow(output_row)
                    rows_seen += 1

        if rows_seen != umap_vectors.shape[0]:
            raise ValueError(
                "clusters CSV and umap.npy row counts do not match "
                f"({rows_seen} != {umap_vectors.shape[0]})."
            )
        if summary_counts is not None and dict(observed_cluster_counts) != summary_counts:
            raise ValueError(
                "clusters_summary.csv counts do not match the cluster assignments "
                f"in {clusters_file}. Regenerate or remove the summary file and retry."
            )
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return output_path


def build_clusters_and_vectors_parser(
    *, prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    """Build the parser for writing vectors from clusters within size bounds."""
    parser = argparse.ArgumentParser(
        prog=prog,
        add_help=add_help,
        description=(
            "Create clusters_and_vectors.csv with row-aligned UMAP vectors, "
            "including only clusters within the requested size bounds."
        ),
    )
    parser.add_argument("--clusters-file", type=Path, required=True)
    parser.add_argument("--umap-file", type=Path, required=True)
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=100,
        help="Keep clusters with more than this many images. Default: 100.",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=1_000_000,
        help="Keep clusters with fewer than this many images. Default: 1000000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where clusters_and_vectors.csv is written.",
    )
    return parser


def generate_clusters_and_vectors_file_main(
    argv: Optional[List[str]] = None, *, prog: Optional[str] = None
) -> int:
    """CLI entry point for generating a size-filtered clusters CSV with vectors."""
    args = build_clusters_and_vectors_parser(prog=prog).parse_args(argv)
    output_path = generate_clusters_and_vectors_file(
        args.clusters_file,
        args.umap_file,
        args.output_dir,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
    )
    print(f"Wrote {output_path}")
    return 0
