from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import yaml

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_roi import LaneROI, RoiConfig
from src.io_video import VideoReader


# ============================================================
# SETTINGS
# ============================================================

INPUT = "data/VBOX0011_Trim.mp4"

# Change this to inspect another point in the video
TIME = 756.0

OUTPUT = "outputs/roi_horizon_check.png"

CONFIG = "config/default.yaml"


# ============================================================
# LOAD CONFIG
# ============================================================

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}


# ============================================================
# INITIALIZE
# ============================================================

horizon = HorizonDetector(
    HorizonConfig.from_dict(
        cfg.get("horizon", {})
    )
)

roi = LaneROI(
    RoiConfig.from_dict(
        cfg.get("roi", {})
    )
)


# ============================================================
# READ FRAME
# ============================================================

vr = VideoReader(INPUT)

fps = vr.info.fps

frame_number = int(
    round(TIME * fps)
)

frame = vr.read_frame(frame_number)

vr.release()

if frame is None:
    raise RuntimeError(
        f"Could not read frame {frame_number}"
    )


# ============================================================
# FRAME INFO
# ============================================================

h, w = frame.shape[:2]


# ============================================================
# HORIZON
# ============================================================

horizon_y = horizon.detect(frame)


# ============================================================
# DRAW
# ============================================================

vis = frame.copy()

# ------------------------------------------------------------
# Draw ROI polygon
# ------------------------------------------------------------

vertices = cfg.get("roi", {}).get("vertices")

if not vertices:
    vertices = [
        [0.10, 0.90],
        [0.30, 0.55],
        [0.70, 0.55],
        [0.90, 0.90],
    ]


# IMPORTANT:
# Match the actual LaneROI behaviour:
# the horizon detector replaces the Y position of
# the two upper ROI points.

roi_points = []

for i, (x, y) in enumerate(vertices):

    px = int(round(x * w))
    py = int(round(y * h))

    if horizon_y is not None and i in (1, 2):
        py = horizon_y

    roi_points.append([px, py])


roi_points = np.array(
    roi_points,
    dtype=np.int32
)


# ------------------------------------------------------------
# Fill ROI lightly
# ------------------------------------------------------------

overlay = vis.copy()

cv2.fillPoly(
    overlay,
    [roi_points],
    (255, 0, 255)
)

vis = cv2.addWeighted(
    overlay,
    0.15,
    vis,
    0.85,
    0
)


# ------------------------------------------------------------
# ROI outline
# ------------------------------------------------------------

cv2.polylines(
    vis,
    [roi_points],
    True,
    (255, 0, 255),
    3
)


# ------------------------------------------------------------
# Horizon line
# ------------------------------------------------------------

if horizon_y is not None:

    cv2.line(
        vis,
        (0, horizon_y),
        (w - 1, horizon_y),
        (0, 255, 255),
        3
    )


# ------------------------------------------------------------
# Mark upper ROI points
# ------------------------------------------------------------

for i, (x, y) in enumerate(roi_points):

    cv2.circle(
        vis,
        (int(x), int(y)),
        6,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        vis,
        f"P{i + 1}",
        (int(x) + 8, int(y) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# INFORMATION
# ============================================================

cv2.putText(
    vis,
    f"Time: {TIME:.1f}s",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    vis,
    f"Frame: {frame_number}",
    (10, 58),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    vis,
    f"Horizon Y: {horizon_y}",
    (10, 86),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    vis,
    f"ROI top Y: {roi_points[1][1]}",
    (10, 114),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 0, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    vis,
    f"ROI bottom Y: {roi_points[0][1]}",
    (10, 142),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 0, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

success = cv2.imwrite(
    OUTPUT,
    vis
)

if not success:
    raise RuntimeError(
        f"Could not save {OUTPUT}"
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("ROI + HORIZON INSPECTION")
print("=" * 60)

print(f"Input       : {INPUT}")
print(f"Time        : {TIME:.2f} s")
print(f"Frame       : {frame_number}")
print(f"Resolution  : {w} x {h}")
print(f"Horizon     : {horizon_y}")

print()
print("ROI POINTS")
print("-" * 60)

for i, (x, y) in enumerate(roi_points):
    print(
        f"P{i + 1}: "
        f"x={x:3d}, y={y:3d}"
    )

print()
print(f"Output      : {OUTPUT}")
print("=" * 60)