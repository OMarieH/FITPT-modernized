# Squat pose core

This package is the hardware-independent part of the Raspberry Pi 5 showcase.
It accepts one person's COCO 17-keypoint pose, counts complete squats, prepares
overlay data, and writes a JSON session summary. It uses only Python's standard
library. The existing FITPT RAG backend is separate.

Run the tests from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

## Input and output contract

The future Hailo callback should select and track one person, extract its
17 COCO keypoints in standard order, and call `from_coco_keypoints`. Pass
normalized `(x, y, confidence)` triples, or provide `image_size=(width, height)`
for pixel coordinates. Use `time.monotonic()` for the frame timestamp. Hailo's
actual callback object layout must be inspected on the Pi before writing this
adapter. The installed `rpicam-hello` pose plugin already verifies the camera,
accelerator, and skeleton display, but does not expose its landmarks here.

```python
from pathlib import Path
from time import monotonic
from pose_demo import SquatSession, from_coco_keypoints

session = SquatSession()

# In the future Hailo callback, after extracting one person's keypoints:
frame = from_coco_keypoints(monotonic(), keypoints)
result, overlay = session.process(frame)
# Draw overlay.visible_edges / overlay.visible_joints and the three text fields.
# At the end of the session, choose a new destination filename:
session.recorder.save(Path("squat-session.json"))
```

`save` refuses to overwrite an existing file. The summary contains a schema
version, exercise name, UTC start time, duration, rep count, and timestamped
rep/feedback events. It is ready for later FITPT integration without calling
OpenAI or Pinecone during the live demo.

The analyzer expects a full-body side view and uses hip-knee-ankle angles. Its
angle thresholds and feedback are starting values for a showcase, not medical
or coaching judgments. It requires two confirming frames per transition and
discards an unfinished rep after five frames without usable joints. Rep count
is retained. Bicep curls can be added as another analyzer after squat behavior
is measured on the Pi.
