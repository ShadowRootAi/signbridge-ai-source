"""Synthetic contract tests, never evidence of sign-recognition accuracy."""
import unittest
import numpy as np

from common import classification_metrics
from openset_core import eligibility_score, fit_prototypes, motion_metrics, prototype_distances, score_gate


def config():
    return {"labels": ["a", "b", "c"], "rejection": {"temperature": 1.0, "confidenceThreshold": .5,
                                                           "minQuality": .25, "minMotionBodyRms": 0.0}}


class MotionTests(unittest.TestCase):
    def test_body_rms_uses_joint_pairs_not_coordinate_count(self):
        x = np.zeros((1, 48, 224), dtype=np.float32)
        x[:, :, 220] = 1; x[:, :, 222] = 1
        for t in range(48):
            x[:, t, 0:42:2] = t
        result = motion_metrics(x)
        self.assertEqual(result["motionBodyRms"][0], 1.0)
        self.assertEqual(result["motionLocalRms"][0], 0.0)
        self.assertEqual(result["validTrackedPairFraction"][0], .5)

    def test_missingness_jump_does_not_create_fake_motion(self):
        x = np.zeros((1, 48, 224), dtype=np.float32)
        x[:, :, 220:224] = 1
        x[:, -1, :168] = 100
        x[:, -1, 220:222] = 0
        result = motion_metrics(x)
        self.assertEqual(result["motionBodyRms"][0], 0)
        self.assertEqual(result["motionLocalRms"][0], 0)

    def test_static_valid_sequence_is_not_rejected_with_motion_disabled(self):
        result = score_gate(np.array([[10., 0., 0., -1.]]), np.ones(1), np.zeros(1), config())
        self.assertTrue(result["accepted"][0])


class PrototypeTests(unittest.TestCase):
    def test_distance_uses_active_dimensions_and_first_index_tie(self):
        prototype = {"embeddingSize": 2, "mean": [0, 0], "scale": [1, 1], "active": [1, 0],
                     "centroids": [[0, 0], [2, 0], [4, 0]]}
        distances = prototype_distances(np.array([[3, 999]]), prototype)
        np.testing.assert_array_equal(distances, [[3, 1, 1]])
        self.assertEqual(int(distances.argmin(axis=1)[0]), 1)

    def test_train_only_prototypes_ignore_constant_coordinates(self):
        e = np.array([[0, 9], [1, 9], [3, 9], [4, 9], [6, 9], [7, 9]], dtype=float)
        p = fit_prototypes(e, np.array([0, 0, 1, 1, 2, 2]), ["a", "b", "c"])
        self.assertEqual(p["active"], [1, 0])
        self.assertEqual(p["activeCount"], 1)
        self.assertEqual([len(v) for v in p["trainLeaveOneOutDistances"]], [2, 2, 2])

    def test_nearest_disagreement_rejects_high_confidence(self):
        c = config(); c["prototypeGate"] = {"embeddingSize": 1, "mean": [0], "scale": [1], "active": [1],
                                            "centroids": [[0], [2], [4]], "classDistanceCutoffs": None}
        result = score_gate(np.array([[10., 0., 0., -1.]]), np.ones(1), np.zeros(1), c, np.array([[2.]]))
        self.assertFalse(result["accepted"][0]); self.assertTrue(result["reasons"]["prototypeDisagreement"][0])


class SelectionTests(unittest.TestCase):
    def test_reject_all_cannot_win_selection(self):
        report = classification_metrics(np.array([0, 1, 2, -1]), np.array([0, 1, 2, 0]), np.zeros(4, dtype=bool), 3)
        self.assertIsNone(eligibility_score(report))

    def test_unknown_far_is_primary_after_coverage_and_accuracy_constraints(self):
        y = np.array([0, 1, 2, -1])
        pred = np.array([0, 1, 2, 0])
        safe = classification_metrics(y, pred, np.array([True, True, False, False]), 3)
        broad = classification_metrics(y, pred, np.ones(4, dtype=bool), 3)
        self.assertGreater(eligibility_score(safe), eligibility_score(broad))

    def test_invalid_logits_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Invalid classifier"):
            score_gate(np.array([[np.nan, 0, 0, 0]]), np.ones(1), np.zeros(1), config())

    def test_zero_quality_never_passes(self):
        c = config(); c["rejection"]["minQuality"] = 0
        result = score_gate(np.array([[10., 0., 0., -1.]]), np.zeros(1), np.zeros(1), c)
        self.assertFalse(result["accepted"][0])


if __name__ == "__main__":
    unittest.main()
