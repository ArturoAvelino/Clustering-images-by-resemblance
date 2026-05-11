from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Optional


def _normalize_cluster(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw


def _sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def summarize_clusters_csv(
    clusters_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Generate clusters_summary.csv from clusters.csv.

    The output contains one row per cluster with the normalized cluster ID, the
    number of objects assigned to the cluster, and the number of distinct image
    classes in the cluster. Class IDs are extracted with the same rule used by
    clusters_summary_classes.csv: the last 4 characters of each image filename
    stem.
    """
    if output_path is None:
        output_path = clusters_path.with_name("clusters_summary.csv")

    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters.csv not found: {clusters_path}")

    counts: Counter[str] = Counter()
    classes_by_cluster: dict[str, set[str]] = {}
    with clusters_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("clusters.csv has no header row")
        if "cluster" not in reader.fieldnames:
            raise ValueError("clusters.csv must have a 'cluster' column")
        if "image_id" not in reader.fieldnames:
            raise ValueError("clusters.csv must have an 'image_id' column")
        for row in reader:
            cluster_raw = row.get("cluster")
            image_id = row.get("image_id", "")
            if cluster_raw is None or not image_id:
                continue
            cluster = _normalize_cluster(cluster_raw)
            if not cluster:
                continue
            counts[cluster] += 1
            classes_by_cluster.setdefault(cluster, set()).add(_extract_class_id(image_id))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "num_objs_in_cluster", "num_classes_in_cluster"])
        for cluster in sorted(counts.keys(), key=_sort_key):
            writer.writerow([cluster, counts[cluster], len(classes_by_cluster[cluster])])

    return output_path


def _extract_class_id(image_id: str) -> str:
    """Extract class ID from the last 4 characters of the image filename stem."""
    return Path(image_id).stem[-4:]


def summarize_classes_in_clusters_csv(
    clusters_path: Path,
    benchmark_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate clusters_summary_classes.csv from clusters.csv.

    For each cluster, counts images per class (extracted from the last 4
    characters of each image filename stem, e.g. 'class_4218.jpg' → '4218').
    Produces one row per cluster with per-class counts and percentages.

    If benchmark_path is provided (a CSV with columns 'label_id' and 'count'),
    also writes class_X_%_of_total_class columns showing what fraction of each
    class's total dataset images fall in the cluster.

    Parameters
    ----------
    clusters_path : Path
        Path to clusters.csv (must have image_id and cluster columns).
    benchmark_path : Path, optional
        Path to classes_benchmark.csv with columns label_id, count.
    output_path : Path, optional
        Destination for the output CSV. Defaults to clusters_summary_classes.csv
        next to clusters_path.
    """
    if output_path is None:
        output_path = clusters_path.with_name("clusters_summary_classes.csv")

    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters.csv not found: {clusters_path}")

    # cluster_id -> Counter(class_id -> count)
    cluster_class_counts: dict[str, Counter[str]] = {}
    cluster_totals: Counter[str] = Counter()

    with clusters_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("clusters.csv has no header row")
        if "cluster" not in reader.fieldnames:
            raise ValueError("clusters.csv must have a 'cluster' column")
        if "image_id" not in reader.fieldnames:
            raise ValueError("clusters.csv must have an 'image_id' column")
        for row in reader:
            cluster_raw = row.get("cluster")
            image_id = row.get("image_id", "")
            if cluster_raw is None or not image_id:
                continue
            cluster = _normalize_cluster(cluster_raw)
            if not cluster:
                continue
            class_id = _extract_class_id(image_id)
            if cluster not in cluster_class_counts:
                cluster_class_counts[cluster] = Counter()
            cluster_class_counts[cluster][class_id] += 1
            cluster_totals[cluster] += 1

    # Load benchmark totals if provided
    benchmark: dict[str, int] = {}
    if benchmark_path is not None:
        if not benchmark_path.exists():
            raise FileNotFoundError(f"classes_benchmark.csv not found: {benchmark_path}")
        with benchmark_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_id = str(row.get("label_id", "")).strip()
                count_str = str(row.get("count", "")).strip()
                if label_id and count_str:
                    try:
                        benchmark[label_id] = int(count_str)
                    except ValueError:
                        pass

    # Union of all class IDs across all clusters, sorted numerically
    all_classes: set[str] = set()
    for counts in cluster_class_counts.values():
        all_classes.update(counts.keys())
    sorted_classes = sorted(all_classes, key=_sort_key)

    # Build header: for each class, emit count + optional %_of_total + %_of_cluster
    header = ["cluster_id", "num_objs_in_cluster", "num_classes_in_cluster"]
    for cls in sorted_classes:
        header.append(f"class_{cls}")
        if benchmark:
            header.append(f"class_{cls}_%_of_total_class")
        header.append(f"class_{cls}_%_of_the_cluster")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cluster in sorted(cluster_class_counts.keys(), key=_sort_key):
            counts = cluster_class_counts[cluster]
            total = cluster_totals[cluster]
            row: list = [cluster, total, len(counts)]
            for cls in sorted_classes:
                cls_count = counts.get(cls, 0)
                row.append(cls_count)
                if benchmark:
                    total_in_dataset = benchmark.get(cls, 0)
                    pct_of_total = (
                        round(cls_count / total_in_dataset * 100, 2)
                        if total_in_dataset > 0
                        else 0.0
                    )
                    row.append(pct_of_total)
                pct_of_cluster = round(cls_count / total * 100, 2) if total > 0 else 0.0
                row.append(pct_of_cluster)
            writer.writerow(row)

    return output_path
