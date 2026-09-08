from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.pipeline import summarize_cluster_outputs
from pipeline.richness import extract_location_id, write_richness_csv


class RichnessCsvTests(unittest.TestCase):
    def test_extract_location_id_accepts_single_and_multiple_digits(self) -> None:
        self.assertEqual(
            extract_location_id("nested/A09-G_r4c4_obj_328626_class_4200.jpg"),
            "A09-G",
        )
        self.assertEqual(
            extract_location_id("ZA55-F_r11c3_obj_96_class_run4.jpg"),
            "ZA55-F",
        )
        self.assertEqual(
            extract_location_id(r"nested\BM24-C_r10c3_obj_26_class_run4.jpg"),
            "BM24-C",
        )

    def test_write_richness_csv_counts_clusters_and_richness(self) -> None:
        rows = [
            ("A06-C_r2c7_obj_1_class_run4.jpg", 0),
            ("A15-E_r2c7_obj_2_class_run4.jpg", 0),
            ("HM05-E_r0c0_obj_3_class_run4.jpg", 2),
            ("HM05-E_r1c7_obj_1_class_run4.jpg", 2),
            ("BM21D_r3c7_obj_2_class_run4.jpg", 3),
            ("BM21D_r3c7_obj_3_class_run4.jpg", 3),
            ("BM21D_r3c8_obj_3_class_run4.jpg", 3),
            ("A06-C_r2c0_obj_1_class_run4.jpg", 10),
            ("A06-C_r9c4_obj_108_class_run4.jpg", 32),
            ("HM05-E_r7c5_obj_7_class_run4.jpg", 32),
            ("BM21D_r3c9_obj_3_class_run4.jpg", 32),
        ]
        with tempfile.TemporaryDirectory() as directory:
            clusters_path = Path(directory) / "clusters.csv"
            with clusters_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image_id", "cluster"])
                writer.writerows(rows)

            output_path = write_richness_csv(clusters_path)
            with output_path.open("r", encoding="utf-8", newline="") as stream:
                result = list(csv.reader(stream))

        self.assertEqual(
            result,
            [
                ["location", "cluster_0", "cluster_2", "cluster_3", "cluster_10", "cluster_32", "richness"],
                ["A06-C", "1", "0", "0", "1", "1", "3"],
                ["A15-E", "1", "0", "0", "0", "0", "1"],
                ["HM05-E", "0", "2", "0", "0", "1", "2"],
                ["BM21D", "0", "0", "3", "0", "1", "2"],
            ],
        )

    def test_write_richness_csv_rejects_unparseable_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clusters_path = Path(directory) / "clusters.csv"
            clusters_path.write_text("image_id,cluster\nno_location.jpg,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                write_richness_csv(clusters_path)

    def test_cluster_output_summaries_always_include_richness(self) -> None:
        clusters_path = Path("subclusters/cluster_7/clusters.csv")
        classes_path = clusters_path.with_name("clusters_summary_classes.csv")
        with (
            patch("pipeline.pipeline.summarize_clusters_csv"),
            patch(
                "pipeline.pipeline.summarize_classes_in_clusters_csv",
                return_value=classes_path,
            ),
            patch("pipeline.pipeline.summarize_cluster_dominant_classes_and_diff_csv"),
            patch("pipeline.pipeline.write_richness_csv") as write_richness,
        ):
            summarize_cluster_outputs(clusters_path)

        write_richness.assert_called_once_with(clusters_path)


if __name__ == "__main__":
    unittest.main()
