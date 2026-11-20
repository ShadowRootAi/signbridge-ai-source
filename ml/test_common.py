"""Tests of leakage prevention and failure accounting, using synthetic data only."""
import unittest

import numpy as np

from common import (FeatureDataset, classification_metrics, decisions,
                    tune_rejection, validate_dataset)


def example_dataset():
    return FeatureDataset(
        x=np.zeros((9, 4, 2), dtype=np.float32),
        y=np.array([0, 1, -1] * 3),
        signer=np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3),
        source_id=np.array([f"source-{i}" for i in range(9)]),
        split=np.array(["train"] * 3 + ["val"] * 3 + ["test"] * 3),
        classes=["known-a", "known-b"],
        raw_label=np.array(["known-a", "known-b", "unknown-a", "known-a", "known-b", "unknown-b", "known-a", "known-b", "unknown-c"]),
        quality=np.ones(9, dtype=np.float32), metadata={},
    )


class LeakageTests(unittest.TestCase):
    def test_valid_disjoint_dataset(self):
        validate_dataset(example_dataset())

    def test_reject_signer_leakage(self):
        data = example_dataset()
        data.signer[3] = "a"
        with self.assertRaisesRegex(ValueError, "signer leakage"):
            validate_dataset(data)

    def test_reject_source_leakage(self):
        data = example_dataset()
        data.source_id[3] = data.source_id[0]
        with self.assertRaisesRegex(ValueError, "source_id leakage"):
            validate_dataset(data)

    def test_reject_unknown_category_reuse(self):
        data = example_dataset()
        data.raw_label[8] = "unknown-b"
        with self.assertRaisesRegex(ValueError, "Unknown category leakage"):
            validate_dataset(data)


class SelectiveMetricsTests(unittest.TestCase):
    def test_reject_all_is_not_perfect_accuracy(self):
        metric = classification_metrics(np.array([0, 1, -1]), np.array([0, 1, 0]), np.zeros(3, dtype=bool), 2)
        self.assertIsNone(metric["accepted_known_accuracy"])
        self.assertEqual(metric["known_coverage"], 0)
        self.assertEqual(metric["known_rejection_rate"], 1)

    def test_unknown_false_acceptance_does_not_hide_in_accuracy(self):
        metric = classification_metrics(np.array([0, 1, -1, -1]), np.array([0, 1, 0, 1]), np.array([True, True, True, False]), 2)
        self.assertEqual(metric["closed_set_accuracy"], 1)
        self.assertEqual(metric["unknown_false_acceptance_rate"], 0.5)
        self.assertAlmostEqual(metric["all_accepted_accuracy"], 2 / 3)

    def test_background_winner_is_rejected(self):
        _, accepted, _ = decisions(np.array([[0.1, 0.2, 0.7]]), np.ones(1), 2, 0, 0)
        self.assertFalse(accepted[0])

    def test_blank_and_insufficient_tracking_never_pass_the_quality_floor(self):
        probs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        _, accepted, _ = decisions(probs, np.array([0.0, 0.24, 0.25]), 2, 0, 0)
        np.testing.assert_array_equal(accepted, np.array([False, False, True]))

    def test_validation_cannot_remove_engineering_quality_floor(self):
        tuning = tune_rejection(np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([0, 1]), np.ones(2), 2)
        self.assertGreaterEqual(tuning["min_quality"], 0.25)

    def test_validation_threshold_sacrifices_coverage_honestly(self):
        probs = np.array([[0.95, 0.05], [0.25, 0.75], [0.85, 0.15]])
        labels = np.array([0, 1, -1])
        tuning = tune_rejection(probs, labels, np.ones(3), 2, target_far=0, target_accepted_accuracy=1)
        self.assertEqual(tuning["validation_metrics"]["unknown_false_acceptance_rate"], 0)
        self.assertEqual(tuning["validation_metrics"]["known_coverage"], 0.5)
        self.assertEqual(tuning["validation_metrics"]["accepted_known_accuracy"], 1)

    def test_missing_unknowns_are_explicit(self):
        tuning = tune_rejection(np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([0, 1]), np.ones(2), 2)
        self.assertFalse(tuning["unknown_constraint_evaluated"])
        self.assertIsNone(tuning["validation_metrics"]["unknown_false_acceptance_rate"])


if __name__ == "__main__":
    unittest.main()
