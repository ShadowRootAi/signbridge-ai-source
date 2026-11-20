"""Synthetic structural fixtures test the guard only; never training or claimed signing examples."""
from copy import deepcopy
import unittest
from v4_tracking import measure_raw_tracking


def frames(n=10):
    points=lambda n:[{"x":.5,"y":.5,"visibility":1.0} for _ in range(n)]
    return [{"t":i*1000/30,"width":640,"height":480,"left":points(21),"right":points(21),"pose":points(25),"face":points(387)} for i in range(n)]


class TrackingTests(unittest.TestCase):
    def test_complete_static_input_allowed_without_motion_assumption(self):
        r=measure_raw_tracking(frames(4));self.assertTrue(r["usable"]);self.assertEqual(r["requiredDataCompleteness"],1)
        self.assertEqual(r["validMotionPairs"],3)

    def test_resting_hand_cannot_hide_active_hand_loss(self):
        f=frames(15)
        for i in (0,1,2,3,7,10):f[i]["right"]=[]
        r=measure_raw_tracking(f);self.assertEqual(r["eitherHandDetection"],1);self.assertEqual(r["rightHandCompleteness"],.6)
        self.assertFalse(r["usable"]);self.assertEqual(r["longestMissingRuns"]["rightHand"],4)

    def test_threshold_boundary_not_reported_complete(self):
        f=frames(10);f[5]["right"]=[];r=measure_raw_tracking(f)
        self.assertTrue(r["usable"]);self.assertFalse(r["allRequiredFramesPresent"]);self.assertEqual(r["requiredDataCompleteness"],.9)
        f[6]["right"]=[];self.assertFalse(measure_raw_tracking(f)["usable"])

    def test_pair_validity_excludes_both_missing_endpoints(self):
        f=frames(5);f[2]["left"]=[];r=measure_raw_tracking(f)
        self.assertEqual(r["validMotionPairs"],2);self.assertEqual(r["validMotionPairFraction"],.5)

    def test_empty_and_too_short_fail_closed(self):
        for n in (0,1,3):self.assertFalse(measure_raw_tracking(frames(n))["usable"])

    def test_nonfinite_and_missing_points_are_not_valid_presence(self):
        f=frames();f[0]["right"][0]["x"]=float("nan");f[1]["right"]=f[1]["right"][:20]
        r=measure_raw_tracking(f);self.assertFalse(r["usable"]);self.assertEqual(r["rightHandCompleteness"],.8)

    def test_invalid_timestamps_dimensions_and_shoulders(self):
        for change in (lambda f:f[1].update(t=f[0]["t"]), lambda f:f[0].update(width=0),
                       lambda f:f[0].update(t=float("nan"))):
            f=frames();change(f);self.assertFalse(measure_raw_tracking(f)["usable"])
        f=frames();f[0]["pose"][11]["visibility"]=.49;f[1]["pose"][12]["visibility"]=float("nan")
        self.assertEqual(measure_raw_tracking(f)["poseCompleteness"],.8)

    def test_hand_visibility_not_misused_as_confidence(self):
        f=frames()
        for frame in f:
            for hand in ("left","right"):
                for p in frame[hand]:p["visibility"]=0
        self.assertTrue(measure_raw_tracking(f)["usable"])


if __name__=="__main__":unittest.main()
