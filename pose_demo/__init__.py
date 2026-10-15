"""Hardware-independent pose analysis for the FITPT showcase."""

from .pipeline import SquatSession
from .pose import Landmark, PoseFrame, from_coco_keypoints
from .squat import SquatAnalyzer, SquatResult

__all__ = [
    "Landmark",
    "PoseFrame",
    "SquatAnalyzer",
    "SquatResult",
    "SquatSession",
    "from_coco_keypoints",
]
