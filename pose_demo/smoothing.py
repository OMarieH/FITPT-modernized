"""Simple exponential smoothing for visible landmarks."""

from __future__ import annotations

from .pose import Landmark, PoseFrame


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.45, min_confidence: float = 0.5) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        self.alpha = alpha
        self.min_confidence = min_confidence
        self._previous: dict[str, Landmark] = {}

    def update(self, frame: PoseFrame) -> PoseFrame:
        smoothed: dict[str, Landmark] = {}
        for name, current in frame.landmarks.items():
            if current.confidence < self.min_confidence:
                continue
            previous = self._previous.get(name)
            if previous is None:
                smoothed[name] = current
            else:
                a = self.alpha
                smoothed[name] = Landmark(
                    a * current.x + (1 - a) * previous.x,
                    a * current.y + (1 - a) * previous.y,
                    current.confidence,
                )
        # A lost joint is removed so it cannot become stale or affect an angle.
        self._previous = smoothed
        return PoseFrame(frame.timestamp, smoothed)

    def reset(self) -> None:
        self._previous.clear()
