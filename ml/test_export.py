"""Actual ONNX numerical smoke test with explicitly synthetic inputs and weights."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_common import example_dataset
from train import TemporalClassifier, export_and_check, seed_everything


class ExportTests(unittest.TestCase):
    def test_fixed_shape_export_includes_normalization_and_preserves_logits(self):
        seed_everything(100)
        rng = np.random.default_rng(100)
        data = example_dataset()
        data.x = rng.normal(size=(9, 48, 224)).astype(np.float32)
        mean = rng.normal(size=224).astype(np.float32)
        std = rng.uniform(0.5, 2.0, size=224).astype(np.float32)
        model = TemporalClassifier(mean, std, outputs=3, channels=8)
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix=".synthetic-export-test-", dir=test_root) as directory:
            self.assertEqual(Path(directory).resolve().parent, test_root)
            report = export_and_check(model, data, Path(directory))
            self.assertTrue(report["class_predictions_identical"])
            self.assertEqual(report["checked_examples"], 9)
            self.assertLess(report["max_abs_error"], 1e-5)
            self.assertEqual(report["browser_parity"], "not_yet_tested")


if __name__ == "__main__":
    unittest.main()
