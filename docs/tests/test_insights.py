"""Protect the boundary between research preparation and published plot data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "home"))
import insight_data


class InsightData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(insight_data, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.sample = {
            "metadata": {"status": "complete", "default_feature": "gamma", "default_model": "ESM-1b"},
            "features": [{"key": "gamma", "label": "Gamma"}],
            "models": [{"key": "esm_1b", "label": "ESM-1b"}],
            "records": [{"id": "assay", "features": {"gamma": 0.5}, "models": {"esm_1b": 0.6},
                         "publication": {"url": "https://doi.org/10.example/source"},
                         "worker_log": "/private/preparation/cache"}],
        }

    def write(self, name="proteingym"):
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        (folder / "data.json").write_text(json.dumps(self.sample))

    def test_partial_preparation_is_not_published(self):
        self.sample["metadata"]["status"] = "partial"
        self.write()
        self.assertEqual(insight_data.load_insights(), [])
        self.assertEqual(len(insight_data.load_insights(include_partial=True)), 1)

    def test_official_scores_and_model_defaults_survive_normalization(self):
        self.sample["records"][0]["models"] = {"ESM-1b": 0.6}
        self.write()
        panel, = insight_data.load_insights()
        self.assertEqual(panel["default_outcome"], "esm_1b")
        self.assertEqual(panel["records"][0]["outcomes"]["esm_1b"], 0.6)
        self.assertNotIn("worker_log", panel["records"][0])
        self.assertEqual(panel["records"][0]["publication"]["url"], "https://doi.org/10.example/source")

    def test_invalid_scores_and_duplicate_dataset_points_are_rejected(self):
        self.sample["records"][0]["models"]["esm_1b"] = 1.6
        self.write()
        with self.assertRaisesRegex(ValueError, "declared range"):
            insight_data.load_insights()
        self.sample["records"][0]["models"]["esm_1b"] = float("nan")
        self.write()
        with self.assertRaisesRegex(ValueError, "Invalid"):
            insight_data.load_insights()
        self.sample["records"][0]["models"]["esm_1b"] = 0.6
        self.sample["records"].append(dict(self.sample["records"][0]))
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate dataset"):
            insight_data.load_insights()

    def test_evolution_percentiles_must_be_fractions(self):
        self.sample["records"][0]["models"]["esm_1b"] = 99.9
        self.write("evolution")
        with self.assertRaisesRegex(ValueError, "declared range"):
            insight_data.load_insights()
