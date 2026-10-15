"""A small input contract independent of Hailo and camera libraries.

Coordinates are normalized to [0, 1], with (0, 0) at the upper-left.
Timestamps are seconds from a monotonic clock supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence


COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.confidence)):
            raise ValueError("landmark values must be finite")
        if not (0 <= self.x <= 1 and 0 <= self.y <= 1 and 0 <= self.confidence <= 1):
            raise ValueError("landmark coordinates and confidence must be in [0, 1]")


@dataclass(frozen=True)
class PoseFrame:
    timestamp: float
    landmarks: Mapping[str, Landmark]

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be a nonnegative finite number")


def from_coco_keypoints(
    timestamp: float,
    keypoints: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int] | None = None,
) -> PoseFrame:
    """Convert one person's 17 COCO keypoints into a PoseFrame.

    Each input item is (x, y, confidence). Omit image_size for normalized
    coordinates; provide (width, height) for pixel coordinates. Selecting and
    tracking the person belongs to the camera adapter, not this function.
    """
    if len(keypoints) != len(COCO_NAMES):
        raise ValueError("COCO pose requires exactly 17 keypoints")
    width, height = image_size if image_size is not None else (1, 1)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    landmarks = {}
    for name, point in zip(COCO_NAMES, keypoints, strict=True):
        if len(point) != 3:
            raise ValueError("each keypoint must be (x, y, confidence)")
        landmarks[name] = Landmark(point[0] / width, point[1] / height, point[2])
    return PoseFrame(timestamp, landmarks)
