"""Independent Python implementation of web/src/features.ts, schema v1.

Keep this explicit implementation independent from the browser implementation so
real raw-landmark fixtures can detect train/browser feature drift.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

FEATURE_SCHEMA = "signbridge-landmarks-xy-v1"
TIMESTEPS = 48
FEATURE_COUNT = 224
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]
FACE_INDICES = [1, 4, 33, 133, 362, 263, 61, 291, 13, 14, 70, 300, 105, 334, 159, 386, 152]


def _valid(points: list[dict], n: int) -> bool:
    return len(points) >= n and all(
        isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float))
        and math.isfinite(p["x"]) and math.isfinite(p["y"]) for p in points[:n])


def frame_features(frame: dict[str, Any]) -> tuple[list[float], float]:
    aspect = frame["width"] / frame["height"]
    left, right, pose, face = (frame[key] for key in ("left", "right", "pose", "face"))

    def xy(point):
        return point["x"] * aspect, point["y"]

    def distance(a, b):
        return math.hypot((a["x"] - b["x"]) * aspect, a["y"] - b["y"])

    def visibility(point):
        result = point.get("visibility")
        return 1 if result is None else result

    pose_valid = _valid(pose, 25) and visibility(pose[11]) >= 0.5 and visibility(pose[12]) >= 0.5
    left_valid = _valid(left, 21)
    right_valid = _valid(right, 21)
    face_valid = _valid(face, 387)
    origin = ((pose[11]["x"] + pose[12]["x"]) * aspect / 2,
              (pose[11]["y"] + pose[12]["y"]) / 2) if pose_valid else (0, 0)
    body_scale = max(0.05, distance(pose[11], pose[12])) if pose_valid else 1
    values: list[float] = []

    def append(point, center, scale):
        if point is None:
            values.extend((0.0, 0.0))
        else:
            p = xy(point)
            for coordinate, center_coordinate in zip(p, center):
                value = (coordinate - center_coordinate) / scale
                values.append(max(-8.0, min(8.0, value if math.isfinite(value) else 0.0)))

    for hand, present in ((left, left_valid), (right, right_valid)):
        for i in range(21):
            append(hand[i] if pose_valid and present else None, origin, body_scale)
        wrist = xy(hand[0]) if present else (0, 0)
        palm_scale = max(0.01, distance(hand[0], hand[9])) if present else 1
        for i in range(21):
            append(hand[i] if present else None, wrist, palm_scale)
    for i in POSE_INDICES:
        append(pose[i] if pose_valid else None, origin, body_scale)
    face_center = xy(face[1]) if face_valid else (0, 0)
    face_scale = max(0.01, distance(face[33], face[263])) if face_valid else 1
    for i in FACE_INDICES:
        append(face[i] if face_valid else None, face_center, face_scale)
    values.extend(map(float, (left_valid, right_valid, pose_valid, face_valid)))
    if len(values) != FEATURE_COUNT:
        raise AssertionError("Feature schema length mismatch")
    return values, float(pose_valid and (left_valid or right_valid))


def preprocess(frames: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    if not frames:
        raise ValueError("No frames to analyze")
    if any(frame["t"] <= frames[i - 1]["t"] for i, frame in enumerate(frames) if i):
        raise ValueError("Non-monotonic frame timestamps")
    features = [frame_features(frame) for frame in frames]
    start = frames[0]["t"]
    span = frames[-1]["t"] - start
    sequence = []
    lo = 0
    for i in range(TIMESTEPS):
        timestamp = start + span * i / (TIMESTEPS - 1)
        while lo + 1 < len(frames) and frames[lo + 1]["t"] < timestamp:
            lo += 1
        hi = min(lo + 1, len(frames) - 1)
        weight = 0 if hi == lo else (timestamp - frames[lo]["t"]) / (frames[hi]["t"] - frames[lo]["t"])
        sequence.append([value + weight * (features[hi][0][j] - value) for j, value in enumerate(features[lo][0])])
    return np.asarray(sequence, dtype=np.float32), sum(feature[1] for feature in features) / len(features)
