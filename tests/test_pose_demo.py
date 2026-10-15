import json
import math
import tempfile
import unittest
from pathlib import Path

from pose_demo import Landmark, PoseFrame, SquatSession, from_coco_keypoints
from pose_demo.smoothing import LandmarkSmoother


def pose_at_angle(timestamp, angle, *, confidence=1.0):
    radians = math.radians(angle)
    landmarks = {}
    for side, offset in (("left", 0.0), ("right", 0.4)):
        landmarks[f"{side}_knee"] = Landmark(0.3 + offset, 0.5, confidence)
        landmarks[f"{side}_ankle"] = Landmark(0.3 + offset, 0.8, confidence)
        landmarks[f"{side}_hip"] = Landmark(0.3 + offset + math.sin(radians) * 0.2, 0.5 + math.cos(radians) * 0.2, confidence)
    return PoseFrame(timestamp, landmarks)


class PoseDemoTests(unittest.TestCase):
    def test_complete_squat_counts_once_and_exports_summary(self):
        session = SquatSession(smoothing_alpha=1, stable_frames=2)
        angles = [170] * 3 + [140] * 3 + [95] * 3 + [135] * 3 + [170] * 5
        results = [session.process(pose_at_angle(i, angle))[0] for i, angle in enumerate(angles)]
        self.assertEqual(1, results[-1].reps)
        self.assertEqual(1, sum(result.rep_completed for result in results))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            session.recorder.save(path)
            summary = json.loads(path.read_text())
            self.assertEqual(1, summary["reps"])
            self.assertEqual(1, sum(event["type"] == "rep_completed" for event in summary["events"]))
            with self.assertRaises(FileExistsError):
                session.recorder.save(path)

    def test_shallow_squat_does_not_count_and_gives_feedback(self):
        session = SquatSession(smoothing_alpha=1, stable_frames=2)
        angles = [170] * 3 + [135] * 3 + [170] * 3
        results = [session.process(pose_at_angle(i, angle))[0] for i, angle in enumerate(angles)]
        self.assertEqual(0, results[-1].reps)
        self.assertIn("Go lower on the next squat", [result.feedback for result in results])

    def test_missing_joints_do_not_count_and_smoothing_discards_them(self):
        session = SquatSession(smoothing_alpha=1, stable_frames=2)
        session.process(pose_at_angle(0, 170))
        session.process(pose_at_angle(1, 170))
        missing = PoseFrame(2, {})
        result, overlay = session.process(missing)
        self.assertIsNone(result.knee_angle)
        self.assertEqual(0, result.reps)
        self.assertEqual({}, overlay.visible_joints)

    def test_lost_track_discards_partial_rep(self):
        session = SquatSession(smoothing_alpha=1, stable_frames=2)
        angles = [170] * 3 + [135] * 3 + [95] * 3
        for i, angle in enumerate(angles):
            session.process(pose_at_angle(i, angle))
        for i in range(9, 15):
            session.process(PoseFrame(i, {}))
        results = [session.process(pose_at_angle(i, 170))[0] for i in range(15, 19)]
        self.assertEqual(0, results[-1].reps)

    def test_coco_adapter_and_smoothing(self):
        keypoints = [(0.5, 0.5, 0.9)] * 17
        frame = from_coco_keypoints(1, keypoints)
        self.assertEqual(0.5, frame.landmarks["left_knee"].x)
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update(frame)
        moved = from_coco_keypoints(2, [(0.7, 0.5, 0.9)] * 17)
        self.assertAlmostEqual(0.6, smoother.update(moved).landmarks["left_knee"].x)


if __name__ == "__main__":
    unittest.main()
