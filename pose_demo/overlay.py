"""Presentation data a Pi display callback can draw over the camera frame."""

from __future__ import annotations

from dataclasses import dataclass

from .pose import Landmark, PoseFrame
from .squat import SquatResult


SKELETON_EDGES = (
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)


@dataclass(frozen=True)
class Overlay:
    reps_text: str
    phase_text: str
    feedback_text: str
    visible_joints: dict[str, Landmark]
    visible_edges: tuple[tuple[str, str], ...]


def make_overlay(frame: PoseFrame, result: SquatResult, min_confidence: float = 0.5) -> Overlay:
    joints = {name: joint for name, joint in frame.landmarks.items() if joint.confidence >= min_confidence}
    edges = tuple((a, b) for a, b in SKELETON_EDGES if a in joints and b in joints)
    return Overlay(f"Squats: {result.reps}", result.phase.value.title(), result.feedback, joints, edges)
