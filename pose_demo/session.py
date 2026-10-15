"""Versioned JSON session summary for a future FITPT RAG integration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .squat import SquatResult


class SessionRecorder:
    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_timestamp: float | None = None
        self.last_timestamp: float | None = None
        self.reps = 0
        self.events: list[dict[str, object]] = []
        self._last_feedback = ""

    def record(self, result: SquatResult) -> None:
        if self.started_timestamp is None:
            self.started_timestamp = result.timestamp
        self.last_timestamp = result.timestamp
        self.reps = result.reps
        if result.rep_completed:
            self.events.append({"time_seconds": round(result.timestamp - self.started_timestamp, 2), "type": "rep_completed", "rep": result.reps})
        if result.feedback and result.feedback != self._last_feedback:
            self.events.append({"time_seconds": round(result.timestamp - self.started_timestamp, 2), "type": "feedback", "message": result.feedback})
        self._last_feedback = result.feedback

    def summary(self) -> dict[str, object]:
        duration = 0.0 if self.started_timestamp is None or self.last_timestamp is None else self.last_timestamp - self.started_timestamp
        return {
            "schema_version": 1,
            "exercise": "squat",
            "started_at": self.started_at,
            "duration_seconds": round(duration, 2),
            "reps": self.reps,
            "events": list(self.events),
        }

    def save(self, path: str | Path) -> Path:
        """Write a new file; refuse to overwrite an existing session."""
        destination = Path(path)
        with destination.open("x", encoding="utf-8") as output:
            json.dump(self.summary(), output, indent=2)
            output.write("\n")
        return destination
