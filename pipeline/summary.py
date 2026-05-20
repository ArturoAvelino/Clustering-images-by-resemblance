from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Optional

from .class_labels import (
    DEFAULT_CLASS_ID_NUM_CHARACTERS,
    extract_class_id_from_filename,
)


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


def _format_percent(value: float) -> str:
    return f"{value:.2f}"


def _class_from_cluster_percent_column(column_name: str) -> str | None:
    prefix = "class_"
    suffix = "_%_of_the_cluster"
    if column_name.startswith(prefix) and column_name.endswith(suffix):
        return column_name[len(prefix) : -len(suffix)]
    return None


def _class_count_column(class_id: str) -> str:
    return f"class_{class_id}"


def _read_count(row: dict[str, str], column: str, cluster_id: str) -> str:
    raw_value = str(row.get(column, "")).strip()
    if not raw_value:
        return "0"
    try:
        return str(int(float(raw_value)))
    except ValueError as exc:
        raise ValueError(
            f"Invalid object count for cluster {cluster_id}, column {column}: "
            f"{raw_value!r}"
        ) from exc


def _read_float(row: dict[str, str], column: str, cluster_id: str) -> float:
    raw_value = str(row.get(column, "")).strip()
    if not raw_value:
        return 0.0
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value for cluster {cluster_id}, column {column}: "
            f"{raw_value!r}"
        ) from exc


def _read_int(row: dict[str, str], column: str, cluster_id: str) -> int:
    raw_value = str(row.get(column, "")).strip()
    if not raw_value:
        return 0
    try:
        return int(float(raw_value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer value for cluster {cluster_id}, column {column}: "
            f"{raw_value!r}"
        ) from exc


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


def summarize_cluster_dominant_classes_and_diff_csv(
    summary_classes_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Generate clusters_dominant_classes_and_diff.csv from clusters_summary_classes.csv.

    The output keeps one row per cluster and reports the class with the largest
    ``class_X_%_of_the_cluster`` value, the matching ``class_X`` object count,
    the class with the second-largest percentage and its count, and the
    percentage difference between them. It also copies ``num_objs_in_cluster``
    and ``num_classes_in_cluster`` from the source file. If the dominant class
    accounts for 100% of the cluster and no other class has a positive
    percentage, the second dominant class is written as ``0000`` with percentage
    ``0`` and object count ``0``.
    """
    if output_path is None:
        output_path = summary_classes_path.with_name(
            "clusters_dominant_classes_and_diff.csv"
        )

    if not summary_classes_path.exists():
        raise FileNotFoundError(
            f"clusters_summary_classes.csv not found: {summary_classes_path}"
        )

    with summary_classes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("clusters_summary_classes.csv has no header row")
        if "cluster_id" not in reader.fieldnames:
            raise ValueError("clusters_summary_classes.csv must have a 'cluster_id' column")
        if "num_objs_in_cluster" not in reader.fieldnames:
            raise ValueError(
                "clusters_summary_classes.csv must have a 'num_objs_in_cluster' column"
            )
        if "num_classes_in_cluster" not in reader.fieldnames:
            raise ValueError(
                "clusters_summary_classes.csv must have a 'num_classes_in_cluster' column"
            )

        class_columns: list[tuple[str, str]] = []
        for field in reader.fieldnames:
            class_id = _class_from_cluster_percent_column(field)
            if class_id is not None:
                class_columns.append((class_id, field))
                count_field = _class_count_column(class_id)
                if count_field not in reader.fieldnames:
                    raise ValueError(
                        "clusters_summary_classes.csv must contain matching "
                        f"{count_field} count column for {field}"
                    )
        if not class_columns:
            raise ValueError(
                "clusters_summary_classes.csv must contain class_X_%_of_the_cluster columns"
            )

        rows: list[list[str]] = []
        for row in reader:
            cluster_id = str(row.get("cluster_id", "")).strip()
            if not cluster_id:
                continue
            num_objs = str(row.get("num_objs_in_cluster", "")).strip()
            num_classes = str(row.get("num_classes_in_cluster", "")).strip()
            values: list[tuple[str, float]] = []
            for class_id, field in class_columns:
                raw_value = str(row.get(field, "")).strip()
                try:
                    percent = float(raw_value) if raw_value else 0.0
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid percentage value for cluster {cluster_id}, column {field}: "
                        f"{raw_value!r}"
                    ) from exc
                values.append((class_id, percent))

            values.sort(key=lambda item: (-item[1], _sort_key(item[0])))
            first_class, first_percent = values[0]
            positive_second = next(
                ((class_id, percent) for class_id, percent in values[1:] if percent > 0.0),
                None,
            )
            if round(first_percent, 2) == 100.0 and positive_second is None:
                second_class = "0000"
                second_percent = 0.0
                second_count = "0"
            elif len(values) > 1:
                second_class, second_percent = values[1]
                second_count = _read_count(
                    row, _class_count_column(second_class), cluster_id
                )
            else:
                second_class = "0000"
                second_percent = 0.0
                second_count = "0"
            first_count = _read_count(row, _class_count_column(first_class), cluster_id)
            diff = first_percent - second_percent
            rows.append(
                [
                    cluster_id,
                    num_objs,
                    num_classes,
                    first_class,
                    _format_percent(first_percent),
                    first_count,
                    second_class,
                    (
                        "0"
                        if second_class == "0000" and second_percent == 0.0
                        else _format_percent(second_percent)
                    ),
                    second_count,
                    _format_percent(diff),
                ]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cluster_id",
                "num_objs_in_cluster",
                "num_classes_in_cluster",
                "1st_dom_class",
                "1st_dom_%",
                "1st_dom_num_objs",
                "2nd_dom_class",
                "2nd_dom_%",
                "2nd_dom_num_objs",
                "diff_1st-2nd_%",
            ]
        )
        writer.writerows(rows)

    summarize_clustering_score_report_csv(output_path)
    return output_path


def summarize_clustering_score_report_csv(
    dominant_classes_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Generate clustering_score_report.csv from clusters_dominant_classes_and_diff.csv.

    The report contains one row with aggregate scores computed from every
    cluster except cluster ``-1``: the sum of ``diff_1st-2nd_%`` and the number
    of distinct classes appearing in ``1st_dom_class``. It also reports negative
    penalty values for the summed ``num_classes_in_cluster`` values and for the
    ``num_objs_in_cluster`` value in the cluster ``-1`` row. ``clustering_score``
    is the arithmetic sum of the four preceding report values.
    """
    if output_path is None:
        output_path = dominant_classes_path.with_name("clustering_score_report.csv")

    if not dominant_classes_path.exists():
        raise FileNotFoundError(
            f"clusters_dominant_classes_and_diff.csv not found: {dominant_classes_path}"
        )

    with dominant_classes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("clusters_dominant_classes_and_diff.csv has no header row")
        required_columns = [
            "cluster_id",
            "num_objs_in_cluster",
            "num_classes_in_cluster",
            "1st_dom_class",
            "diff_1st-2nd_%",
        ]
        for column in required_columns:
            if column not in reader.fieldnames:
                raise ValueError(
                    "clusters_dominant_classes_and_diff.csv must have a "
                    f"{column!r} column"
                )

        sum_diff = 0.0
        sum_num_classes = 0
        num_objs_in_noise_cluster = 0
        dominant_classes: set[str] = set()
        for row in reader:
            cluster_id = str(row.get("cluster_id", "")).strip()
            if not cluster_id:
                continue
            if cluster_id == "-1":
                num_objs_in_noise_cluster = _read_int(
                    row, "num_objs_in_cluster", cluster_id
                )
                continue
            sum_diff += _read_float(row, "diff_1st-2nd_%", cluster_id)
            sum_num_classes += _read_int(row, "num_classes_in_cluster", cluster_id)
            dominant_class = str(row.get("1st_dom_class", "")).strip()
            if dominant_class:
                dominant_classes.add(dominant_class)

    num_dom_classes = len(dominant_classes)
    neg_sum_num_classes = -sum_num_classes
    neg_num_objs_in_noise_cluster = -num_objs_in_noise_cluster
    clustering_score = (
        sum_diff
        + num_dom_classes
        + neg_sum_num_classes
        + neg_num_objs_in_noise_cluster
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sum_diff_1st-2nd_%",
                "num_dom_classes",
                "neg_sum_num_classes_in_clusters",
                "neg_num_objs_in_noise_cluster",
                "clustering_score",
            ]
        )
        writer.writerow(
            [
                _format_percent(sum_diff),
                num_dom_classes,
                neg_sum_num_classes,
                neg_num_objs_in_noise_cluster,
                _format_percent(clustering_score),
            ]
        )

    return output_path


def summarize_cluster_dominants_and_diff_csv(
    summary_classes_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Backward-compatible alias for summarize_cluster_dominant_classes_and_diff_csv.

    Defaults now write clusters_dominant_classes_and_diff.csv.
    """
    return summarize_cluster_dominant_classes_and_diff_csv(
        summary_classes_path,
        output_path,
    )


def _extract_class_id(image_id: str) -> str:
    """Extract the default 4-character class ID from an image filename."""
    return extract_class_id_from_filename(
        Path(image_id).name,
        DEFAULT_CLASS_ID_NUM_CHARACTERS,
    )


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
