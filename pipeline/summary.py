from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


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
    if output_path is None:
        output_path = clusters_path.with_name("summary_clusters.csv")

    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters.csv not found: {clusters_path}")

    counts: Counter[str] = Counter()
    with clusters_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("clusters.csv has no header row")
        if "cluster" not in reader.fieldnames:
            raise ValueError("clusters.csv must have a 'cluster' column")
        for row in reader:
            cluster_raw = row.get("cluster")
            if cluster_raw is None:
                continue
            cluster = _normalize_cluster(cluster_raw)
            if not cluster:
                continue
            counts[cluster] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "num_obj_in_cluster"])
        for cluster in sorted(counts.keys(), key=_sort_key):
            writer.writerow([cluster, counts[cluster]])

    return output_path
