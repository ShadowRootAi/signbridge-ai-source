"""Guard against amplifying previously constant features on unseen people."""
import unittest

import numpy as np

from preprocess import FEATURE_SCHEMA
from train import fit_normalization, normalize_numpy


class NormalizationTests(unittest.TestCase):
    def test_constant_coordinates_are_ignored_and_constant_masks_stay_bounded(self):
        train = np.zeros((2, 3, 224), dtype=np.float32)
        train[:, :, -4:] = 1
        mean, std, keep, info = fit_normalization(train, FEATURE_SCHEMA)
        unseen = np.ones((1, 3, 224), dtype=np.float32) * 100
        unseen[:, :, -4:] = 0
        normalized = normalize_numpy(unseen, mean, std, keep)
        np.testing.assert_array_equal(normalized, np.zeros_like(normalized))
        np.testing.assert_array_equal(std[-4:], np.ones(4))
        self.assertEqual(info["ignored_training_invariant_indices"], list(range(220)))

    def test_near_constant_features_have_floor_and_bounded_novel_values(self):
        train = np.array([[[0.0, 0.0], [0.0001, 2.0]]], dtype=np.float32)
        mean, std, keep, _ = fit_normalization(train)
        self.assertAlmostEqual(float(std[0]), 0.01)
        unseen = np.array([[[1000.0, -1000.0]]], dtype=np.float32)
        normalized = normalize_numpy(unseen, mean, std, keep)
        np.testing.assert_array_equal(normalized, np.array([[[8.0, -8.0]]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
