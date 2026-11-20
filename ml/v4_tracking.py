"""Independent Python counterpart of the raw-frame engineering guard; never feature fitting."""
from __future__ import annotations
import math

POLICY = {
    "version": "signbridge-raw-both-hands-v4-20261108",
    "minimumFrames": 4,
    "minimumRequiredCoverage": .90,
    "shoulderVisibilityMinimum": .5,
    "motionThreshold": None,
    "interpretation": "Engineering input-completeness rule, not a fitted or validated accuracy guarantee.",
    "limitation": "Requires both hands; may reject a valid one-handed sign when the other hand is outside the frame.",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_points(points, count):
    return isinstance(points, list) and len(points) >= count and all(
        isinstance(p, dict) and finite(p.get("x")) and finite(p.get("y")) for p in points[:count])


def measure_raw_tracking(frames):
    n = len(frames)
    pose, left, right, face, required, either = [], [], [], [], [], []
    invalid_metadata, ordered = 0, True
    for i, frame in enumerate(frames):
        metadata = finite(frame.get("t")) and finite(frame.get("width")) and frame["width"] > 0 and finite(frame.get("height")) and frame["height"] > 0
        invalid_metadata += not metadata
        if not finite(frame.get("t")) or (i and (not finite(frames[i-1].get("t")) or frame["t"] <= frames[i-1]["t"])):
            ordered = False
        def visibility(j):
            v = frame["pose"][j].get("visibility")
            return 1 if v is None else v
        shoulders = valid_points(frame.get("pose"),25) and all(finite(visibility(j)) and visibility(j)>=POLICY["shoulderVisibilityMinimum"] for j in (11,12))
        p = metadata and shoulders; l = metadata and valid_points(frame.get("left"),21); r = metadata and valid_points(frame.get("right"),21)
        pose.append(p); left.append(l); right.append(r); face.append(metadata and valid_points(frame.get("face"),387))
        required.append(p and l and r); either.append(p and (l or r))
    def fraction(values): return sum(values)/n if n else 0
    def longest(values):
        best, current = 0, 0
        for v in values:
            current = 0 if v else current+1; best = max(best,current)
        return best
    total_pairs = max(0,n-1)
    valid_pairs = sum(required[i-1] and required[i] and finite(frames[i-1].get("t")) and finite(frames[i].get("t")) and frames[i]["t"]>frames[i-1]["t"] for i in range(1,n))
    reasons = []
    if n<POLICY["minimumFrames"]: reasons.append("too_few_frames")
    if invalid_metadata or not ordered: reasons.append("invalid_frame_metadata")
    if not n or fraction(required)<POLICY["minimumRequiredCoverage"]: reasons.append("incomplete_required_landmarks")
    return {"policyVersion":POLICY["version"], "source":"raw_landmark_frames", "frameCount":n,
            "poseCompleteness":fraction(pose), "leftHandCompleteness":fraction(left), "rightHandCompleteness":fraction(right),
            "facePresence":fraction(face), "eitherHandDetection":fraction(either), "requiredDataCompleteness":fraction(required),
            "allRequiredFramesPresent":n>0 and all(required), "usable":not reasons, "reasons":reasons,
            "minRequiredCoverage":POLICY["minimumRequiredCoverage"], "minimumFrames":POLICY["minimumFrames"],
            "invalidMetadataFrames":int(invalid_metadata), "timestampsStrictlyIncreasing":ordered,
            "missingFrameCounts":{k:n-sum(v) for k,v in (("pose",pose),("leftHand",left),("rightHand",right),("required",required))},
            "longestMissingRuns":{k:longest(v) for k,v in (("pose",pose),("leftHand",left),("rightHand",right),("required",required))},
            "validMotionPairFraction":valid_pairs/total_pairs if total_pairs else 0,
            "validMotionPairs":int(valid_pairs), "totalMotionPairs":total_pairs}
