from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.cli import parse_args
from pipeline.config import build_config, config_to_yaml, validate_config
from pipeline.pipeline import run_pipeline, stage_paths
from pipeline.subset import prepare_subset_cache


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = stage_paths(self.root / "full")
        self.output = stage_paths(self.root / "subset")
        self.source.index_path.parent.mkdir()
        self.names = [f"nested/A01_r1c1_obj_{i}.jpg" for i in range(80)]
        self.source.index_path.write_text("\n".join(self.names) + "\n")
        self.emb = np.random.default_rng(42).normal(size=(80, 8)).astype("float16")
        self.emb.tofile(self.source.emb_path)
        self.source.meta_path.write_text(json.dumps(dict(num_images=80, embed_dim=8, dtype="float16")))
        np.save(self.source.size_path, np.arange(80, dtype="float32"))
        self.selection = self.root / "requested.txt"
        self.rows = list(range(79, 0, -2))
        self.selection.write_text("\n".join(self.names[i] for i in self.rows))

    def prepare(self):
        prepare_subset_cache(self.source, self.output, self.selection)

    def test_exact_rows_order_and_unchanged_source(self):
        before = {p: p.read_bytes() for p in self.source.index_path.parent.iterdir()}
        self.prepare()
        actual = np.fromfile(self.output.emb_path, dtype="float16").reshape(-1, 8)
        np.testing.assert_array_equal(actual, self.emb[self.rows])
        np.testing.assert_array_equal(np.load(self.output.size_path), self.rows)
        self.assertEqual(self.output.index_path.read_text().splitlines(), [self.names[i] for i in self.rows])
        self.assertEqual(json.loads(self.output.meta_path.read_text())["num_images"], len(self.rows))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_invalid_selection_does_not_write_outputs(self):
        for contents, error in [("", "empty"), ("absent.jpg", "not found"),
                                (self.names[0] + "\n" + self.names[0], "duplicate")]:
            with self.subTest(error=error):
                self.selection.write_text(contents)
                with self.assertRaisesRegex(ValueError, error):
                    self.prepare()
                self.assertFalse(self.output.index_path.parent.exists())

    def test_replaced_source_index_rejected(self):
        self.source.index_path.write_text(self.selection.read_text())
        with self.assertRaisesRegex(ValueError, "Restore the original"):
            self.prepare()

    def test_ambiguous_source_name_rejected(self):
        self.source.index_path.write_text("\n".join([self.names[79]] * 80))
        with self.assertRaisesRegex(ValueError, "more than once"):
            self.prepare()

    def test_source_directory_and_link_protected(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            prepare_subset_cache(self.source, self.source, self.selection)
        self.output.index_path.parent.mkdir()
        self.output.emb_path.symlink_to(self.source.emb_path)
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            self.prepare()

    def test_corrupt_artifacts_rejected(self):
        np.save(self.source.size_path, np.zeros(3))
        with self.assertRaisesRegex(ValueError, "one size per embedding"):
            self.prepare()
        np.save(self.source.size_path, np.zeros(80))
        self.source.emb_path.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "byte size"):
            self.prepare()

    def test_cli_yaml_and_real_pipeline_without_feature_computation(self):
        config_path = self.root / "config.yaml"
        config_path.write_text(
            f'dino_files: "{self.source.index_path.parent}"\n'
            f'output_dir: "{self.output.index_path.parent}"\n'
            f'subset_images: "{self.selection}"\n'
            'umap_dim: 3\numap_neighbors: 5\nhdbscan_min_cluster_size: 3\n'
            'hdb_min_samples: 2\nsize_feature_weight: 0.5\nsubclustering: false\n'
        )
        cfg = build_config(parse_args(["--config", str(config_path), "--compute", "only-dimreduction-and-clustering"]))
        validate_config(cfg)
        self.assertIn(str(self.selection), config_to_yaml(cfg))
        with (
            patch("pipeline.pipeline.compute_size_features", side_effect=AssertionError("recomputed sizes")),
            patch("pipeline.pipeline.DINOv2Embedder", side_effect=AssertionError("recomputed DINO")),
        ):
            result = run_pipeline(cfg)
        with result.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["image_id"] for r in rows], [self.names[i] for i in self.rows])
        self.assertEqual(np.load(self.output.umap_path).shape, (len(self.rows), 3))
        override = build_config(parse_args([
            "--config", str(config_path), "--subset-images", "override.txt",
            "--compute", "only-dimreduction-and-clustering",
        ]))
        self.assertEqual(override.subset_images, Path("override.txt"))
        override.compute = "full"
        with self.assertRaisesRegex(ValueError, "subset_images requires compute"):
            validate_config(override)


if __name__ == "__main__":
    unittest.main()
