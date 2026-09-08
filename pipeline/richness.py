from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from .summary import _normalize_cluster, _sort_key


_LOCATION_BOUNDARY = re.compile(r"_r\d+c\d+")


def extract_location_id(image_id: str) -> str:
    """Return the filename prefix before its first ``_r<digits>c<digits>`` marker."""
    filename = re.split(r"[/\\]", image_id)[-1]
    match = _LOCATION_BOUNDARY.search(filename)
    if match is None or match.start() == 0:
        raise ValueError(
            f"Cannot extract a location ID from image_id {image_id!r}; expected a "
            "filename containing '_r<digits>c<digits>'."
        )
    return filename[: match.start()]


def write_richness_csv(
    clusters_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Create a location-by-cluster count table from ``clusters.csv``.

    Location IDs are extracted from the basename of each ``image_id`` by taking
    everything before the first ``_r<digits>c<digits>`` marker. The output has
    one row per location, one ``cluster_<ID>`` count column per cluster, and a
    final ``richness`` column counting clusters with at least one image at that
    location. Locations retain their first-seen order; cluster columns are
    sorted numerically when possible.
    """
    if output_path is None:
        output_path = clusters_path.with_name("richness.csv")
    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters.csv not found: {clusters_path}")

    counts_by_location: dict[str, Counter[str]] = {}
    clusters: set[str] = set()
    with clusters_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("clusters.csv has no header row")
        missing = {"image_id", "cluster"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "clusters.csv must have columns: " + ", ".join(sorted(missing))
            )
        for line_number, row in enumerate(reader, start=2):
            image_id = str(row.get("image_id", "")).strip()
            cluster = _normalize_cluster(str(row.get("cluster", "")))
            if not image_id or not cluster:
                raise ValueError(
                    f"Missing image_id or cluster in {clusters_path} at line {line_number}."
                )
            try:
                location = extract_location_id(image_id)
            except ValueError as exc:
                raise ValueError(f"{clusters_path}, line {line_number}: {exc}") from exc
            counts_by_location.setdefault(location, Counter())[cluster] += 1
            clusters.add(cluster)

    sorted_clusters = sorted(clusters, key=_sort_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            ["location", *(f"cluster_{cluster}" for cluster in sorted_clusters), "richness"]
        )
        for location, location_counts in counts_by_location.items():
            cluster_counts = [location_counts[cluster] for cluster in sorted_clusters]
            writer.writerow(
                [location, *cluster_counts, sum(count > 0 for count in cluster_counts)]
            )
    return output_path
