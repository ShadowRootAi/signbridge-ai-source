"""Meaningful coordinate invariance and temporal/missing-data contract tests."""
import copy
import unittest

import numpy as np

from preprocess import FEATURE_COUNT, TIMESTEPS, frame_features, preprocess


def frame(t=0.0):
    hand = [{"x": 0.3 + i * 0.002, "y": 0.3 + i * 0.003} for i in range(21)]
    pose = [{"x": 0.5, "y": 0.5, "visibility": 1.0} for _ in range(33)]
    pose[11]["x"], pose[12]["x"] = 0.3, 0.7
    return {"t": t, "width": 640, "height": 480, "left": hand, "right": [], "pose": pose, "face": []}


class PreprocessTests(unittest.TestCase):
    def test_translation_preserves_features(self):
        original = frame()
        translated = copy.deepcopy(original)
        for key in ("left", "right", "pose", "face"):
            for point in translated[key]:
                point["x"] += 0.1
                point["y"] += 0.07
        np.testing.assert_allclose(frame_features(original)[0], frame_features(translated)[0], rtol=1e-6, atol=1e-6)

    def test_hand_body_position_is_preserved_separately(self):
        original = frame()
        moved = copy.deepcopy(original)
        for point in moved["left"]:
            point["y"] += 0.1
        first = np.asarray(frame_features(original)[0])
        second = np.asarray(frame_features(moved)[0])
        self.assertGreater(np.max(np.abs(first[:42] - second[:42])), 0.1)
        np.testing.assert_allclose(first[42:84], second[42:84], atol=1e-6)

    def test_missing_pose_retains_local_hand_but_quality_fails(self):
        raw = frame()
        raw["pose"] = []
        values, quality = frame_features(raw)
        self.assertEqual(quality, 0)
        self.assertEqual(values[:42], [0] * 42)
        self.assertTrue(any(value != 0 for value in values[42:84]))
        self.assertEqual(values[-4:], [1, 0, 0, 0])

    def test_single_frame_repeats_and_stays_float32(self):
        x, quality = preprocess([frame()])
        self.assertEqual(x.shape, (TIMESTEPS, FEATURE_COUNT))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_array_equal(x[0], x[-1])
        self.assertEqual(quality, 1)

    def test_timestamps_must_increase(self):
        with self.assertRaisesRegex(ValueError, "Non-monotonic"):
            preprocess([frame(1), frame(1)])

    def test_resampling_preserves_endpoints(self):
        first, last = frame(10), frame(20)
        for point in last["left"]:
            point["y"] += 0.1
        x, _ = preprocess([first, last])
        np.testing.assert_array_equal(x[0], np.asarray(frame_features(first)[0], dtype=np.float32))
        np.testing.assert_array_equal(x[-1], np.asarray(frame_features(last)[0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
