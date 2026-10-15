"""Connect pose input, analysis, display data, and recording."""

from __future__ import annotations

from .overlay import Overlay, make_overlay
from .pose import PoseFrame
from .session import SessionRecorder
from .smoothing import LandmarkSmoother
from .squat import SquatAnalyzer, SquatResult


class SquatSession:
    def __init__(self, *, smoothing_alpha: float = 0.45, stable_frames: int = 2) -> None:
        self.smoother = LandmarkSmoother(alpha=smoothing_alpha)
        self.analyzer = SquatAnalyzer(stable_frames=stable_frames)
        self.recorder = SessionRecorder()

    def process(self, frame: PoseFrame) -> tuple[SquatResult, Overlay]:
        smooth_frame = self.smoother.update(frame)
        result = self.analyzer.update(smooth_frame)
        self.recorder.record(result)
        return result, make_overlay(smooth_frame, result)
