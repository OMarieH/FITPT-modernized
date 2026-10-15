"""Side-view squat state machine driven by hip, knee, and ankle angles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import acos, degrees, hypot

from .pose import Landmark, PoseFrame


class SquatPhase(str, Enum):
    SEEKING = "seeking"
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


def joint_angle(a: Landmark, joint: Landmark, b: Landmark) -> float | None:
    ax, ay = a.x - joint.x, a.y - joint.y
    bx, by = b.x - joint.x, b.y - joint.y
    denominator = hypot(ax, ay) * hypot(bx, by)
    if denominator < 1e-8:
        return None
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / denominator))
    return degrees(acos(cosine))


@dataclass(frozen=True)
class SquatResult:
    timestamp: float
    phase: SquatPhase
    reps: int
    knee_angle: float | None
    feedback: str
    rep_completed: bool = False


class SquatAnalyzer:
    """Counts a rep after standing -> depth -> full standing extension.

    Thresholds are demonstration defaults, not a clinical assessment. The
    camera should show the full body from the side. At least two consecutive
    valid frames must support each state transition.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.5,
        standing_angle: float = 160,
        descent_angle: float = 145,
        bottom_angle: float = 105,
        ascent_angle: float = 125,
        stable_frames: int = 2,
        max_missing_frames: int = 5,
    ) -> None:
        if not (0 < bottom_angle < ascent_angle < descent_angle < standing_angle < 180):
            raise ValueError("squat angle thresholds must increase from bottom to standing")
        if not 0 <= min_confidence <= 1 or stable_frames < 1 or max_missing_frames < 1:
            raise ValueError("invalid confidence or frame threshold")
        self.min_confidence = min_confidence
        self.standing_angle = standing_angle
        self.descent_angle = descent_angle
        self.bottom_angle = bottom_angle
        self.ascent_angle = ascent_angle
        self.stable_frames = stable_frames
        self.max_missing_frames = max_missing_frames
        self.phase = SquatPhase.SEEKING
        self.reps = 0
        self._candidate: SquatPhase | None = None
        self._candidate_count = 0
        self._last_timestamp: float | None = None
        self._missing_frames = 0
        self._notice_until = 0.0

    def _angle(self, frame: PoseFrame) -> float | None:
        angles = []
        for side in ("left", "right"):
            joints = [frame.landmarks.get(f"{side}_{part}") for part in ("hip", "knee", "ankle")]
            if any(joint is None or joint.confidence < self.min_confidence for joint in joints):
                continue
            angle = joint_angle(*joints)  # type: ignore[arg-type]
            if angle is not None:
                angles.append(angle)
        return sum(angles) / len(angles) if angles else None

    def _confirm(self, candidate: SquatPhase | None) -> bool:
        if candidate is None:
            self._candidate = None
            self._candidate_count = 0
            return False
        if candidate == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = candidate
            self._candidate_count = 1
        if self._candidate_count >= self.stable_frames:
            self._candidate = None
            self._candidate_count = 0
            return True
        return False

    def update(self, frame: PoseFrame) -> SquatResult:
        if self._last_timestamp is not None and frame.timestamp <= self._last_timestamp:
            raise ValueError("pose timestamps must increase")
        self._last_timestamp = frame.timestamp
        angle = self._angle(frame)
        if angle is None:
            self._confirm(None)
            self._missing_frames += 1
            if self._missing_frames >= self.max_missing_frames:
                # A broken track cannot be joined to a previous partial rep.
                self.phase = SquatPhase.SEEKING
            return SquatResult(frame.timestamp, self.phase, self.reps, None, "Keep your hips, knees, and ankles in view")
        self._missing_frames = 0

        completed = False
        candidate: SquatPhase | None = None
        feedback = "Go lower on the next squat" if frame.timestamp < self._notice_until else ""
        if self.phase == SquatPhase.SEEKING:
            if angle >= self.standing_angle:
                candidate = SquatPhase.STANDING
            else:
                feedback = "Stand upright to begin"
        elif self.phase == SquatPhase.STANDING:
            if angle <= self.descent_angle:
                candidate = SquatPhase.DESCENDING
        elif self.phase == SquatPhase.DESCENDING:
            if angle <= self.bottom_angle:
                candidate = SquatPhase.BOTTOM
            elif angle >= self.standing_angle:
                candidate = SquatPhase.STANDING
        elif self.phase == SquatPhase.BOTTOM:
            if angle >= self.ascent_angle:
                candidate = SquatPhase.ASCENDING
        elif self.phase == SquatPhase.ASCENDING:
            if angle <= self.bottom_angle:
                candidate = SquatPhase.BOTTOM
            elif angle >= self.standing_angle:
                candidate = SquatPhase.STANDING
                completed = True
            else:
                feedback = "Stand fully to finish the rep"

        if self._confirm(candidate):
            self.phase = candidate  # type: ignore[assignment]
            if completed:
                self.reps += 1
                self._notice_until = 0.0
                feedback = "Rep counted"
            elif candidate == SquatPhase.STANDING and not completed:
                self._notice_until = frame.timestamp + 2.0
                feedback = "Go lower on the next squat"
            else:
                feedback = ""
        return SquatResult(
            frame.timestamp,
            self.phase,
            self.reps,
            angle,
            feedback,
            completed and self.phase == SquatPhase.STANDING,
        )
