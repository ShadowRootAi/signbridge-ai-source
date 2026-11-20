"""Mathematical invariants for the multiple-negative prototype comparison."""
import unittest
import numpy as np
from common import softmax
from v3_centroid_experiment import logits_for


class NegativePrototypeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray([[[-2.0]], [[0.5]], [[4.0]]], dtype=np.float32)
        self.model = {"mean": np.zeros(1, dtype=np.float32), "std": np.ones(1, dtype=np.float32),
                      "keep": np.ones(1, dtype=np.float32),
                      "centroids": np.asarray([[-3], [-1], [1], [3], [5]], dtype=np.float32)}

    def test_manual_distances_and_four_outputs(self):
        expected = np.asarray([[-1, -1, -9, -25], [-12.25, -2.25, -.25, -6.25], [-49, -25, -9, -1]])
        np.testing.assert_array_equal(logits_for(self.x, self.model), expected)

    def test_duplicate_negative_does_not_inflate_probability_mass(self):
        duplicate = {**self.model, "centroids": np.concatenate([self.model["centroids"], self.model["centroids"][-1:]])}
        np.testing.assert_array_equal(softmax(logits_for(self.x, self.model)), softmax(logits_for(self.x, duplicate)))

    def test_negative_order_and_input_preservation(self):
        before = self.x.copy()
        reordered = {**self.model, "centroids": self.model["centroids"][[0, 1, 2, 4, 3]]}
        np.testing.assert_array_equal(logits_for(self.x, self.model), logits_for(self.x, reordered))
        np.testing.assert_array_equal(self.x, before)

    def test_single_negative_reproduces_original_centroid_logits(self):
        model = {**self.model, "centroids": self.model["centroids"][:4]}
        direct = -np.square(self.x.reshape(-1, 1)-model["centroids"].ravel()).astype(np.float32)
        np.testing.assert_array_equal(logits_for(self.x, model), direct)


if __name__ == "__main__":
    unittest.main()
