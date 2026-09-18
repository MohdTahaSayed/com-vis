from __future__ import annotations

import sys
import os

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


import cv2
import yaml
import numpy as np


from src.horizon import (
    HorizonDetector,
    HorizonConfig,
)

from src.lane_roi import (
    LaneROI,
    RoiConfig,
)

from src.lane_color import (
    LaneColor,
    LaneColorConfig,
)

from src.lane_edges import (
    LaneEdges,
    CannyConfig,
)


# ============================================================
# SETTINGS
# ============================================================

INPUT = "data/VBOX0011_Trim.mp4"

# Time in seconds
TIME = 360.0

OUTPUT = "outputs/sliding_windows_check.png"


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    "config/default.yaml",
    "r",
    encoding="utf-8"
) as f:

    cfg = yaml.safe_load(f) or {}


# ============================================================
# INITIALIZE PIPELINE COMPONENTS
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


lane_color = LaneColor(
    LaneColorConfig.from_dict(
        cfg.get("lane_color", {})
    )
)


edges_module = LaneEdges(
    CannyConfig.from_dict(
        cfg.get("canny", {})
    ),
    roi,
    lane_color,
    reinforce_with_hsv=True,
)


# ============================================================
# READ VIDEO FRAME
# ============================================================

cap = cv2.VideoCapture(INPUT)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {INPUT}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 25.0


frame_number = int(
    TIME * fps
)


cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    frame_number
)


ok, frame = cap.read()

cap.release()


if not ok:
    raise RuntimeError(
        f"Could not read frame {frame_number}"
    )


h, w = frame.shape[:2]


# ============================================================
# HORIZON
# ============================================================

horizon_y = horizon.detect(frame)


# ============================================================
# LANE EDGE PIPELINE
# ============================================================

edges_roi, edges_raw, hsv_hits = (
    edges_module.compute(
        frame,
        top_y_override=horizon_y
    )
)


# ============================================================
# HISTOGRAM
# ============================================================

# Same histogram logic used by LaneFitter.

bottom = edges_roi[
    int(h * 0.70):,
    :
]


histogram = np.sum(
    bottom > 0,
    axis=0
).astype(np.float32)


# Smooth histogram.

kernel = np.ones(
    15,
    dtype=np.float32
) / 15.0


histogram = np.convolve(
    histogram,
    kernel,
    mode="same"
)


# ------------------------------------------------------------
# LEFT BASE
# ------------------------------------------------------------

left_lo = 0
left_hi = int(
    0.45 * w
)


left_histogram = histogram[
    left_lo:left_hi
]


if left_histogram.max() > 0:

    left_base = (
        left_lo
        + int(
            np.argmax(
                left_histogram
            )
        )
    )

else:

    left_base = int(
        0.25 * w
    )


# ------------------------------------------------------------
# RIGHT BASE
# ------------------------------------------------------------

right_lo = int(
    0.55 * w
)

right_hi = w


right_histogram = histogram[
    right_lo:right_hi
]


if right_histogram.max() > 0:

    right_base = (
        right_lo
        + int(
            np.argmax(
                right_histogram
            )
        )
    )

else:

    right_base = int(
        0.75 * w
    )


# ============================================================
# VISUALIZATION IMAGE
# ============================================================

vis = frame.copy()

# Slightly darken the original image so
# the windows and lane pixels are easier to see.

vis = (
    vis.astype(np.float32) * 0.75
).astype(np.uint8)


# ============================================================
# DRAW ROI TRAPEZOID
# ============================================================

vertices = cfg.get(
    "roi",
    {}
).get(
    "vertices"
)


if not vertices:

    vertices = [
        [0.10, 0.90],
        [0.30, 0.55],
        [0.70, 0.55],
        [0.90, 0.90],
    ]


roi_points = np.array(
    [
        [
            int(x * w),
            int(y * h)
        ]

        for x, y in vertices
    ],
    dtype=np.int32
)


cv2.polylines(
    vis,
    [roi_points],
    True,
    (255, 255, 0),
    2
)


# ============================================================
# DRAW HORIZON
# ============================================================

if horizon_y is not None:

    cv2.line(
        vis,
        (0, horizon_y),
        (w - 1, horizon_y),
        (0, 255, 255),
        2
    )


# ============================================================
# DRAW BASE POINTS
# ============================================================

cv2.circle(
    vis,
    (left_base, h - 10),
    7,
    (0, 255, 255),
    -1
)


cv2.circle(
    vis,
    (right_base, h - 10),
    7,
    (0, 255, 0),
    -1
)


# ============================================================
# SLIDING WINDOW SETTINGS
# ============================================================

sliding_cfg = cfg.get(
    "sliding_window",
    {}
)


n_windows = int(
    sliding_cfg.get(
        "n_windows",
        9
    )
)


window_width_frac = float(
    sliding_cfg.get(
        "window_width_frac",
        0.12
    )
)


min_pixels_per_window = int(
    sliding_cfg.get(
        "min_pixels_per_window",
        6
    )
)


min_pixels_to_recenter = int(
    sliding_cfg.get(
        "min_pixels_to_recenter",
        12
    )
)


max_recenter_jump_frac = float(
    sliding_cfg.get(
        "max_recenter_jump_frac",
        0.12
    )
)


left_max_x_frac = float(
    sliding_cfg.get(
        "left_max_x_frac",
        0.58
    )
)


right_min_x_frac = float(
    sliding_cfg.get(
        "right_min_x_frac",
        0.42
    )
)


# ============================================================
# WINDOW DIMENSIONS
# ============================================================

window_width = max(
    24,
    int(
        window_width_frac * w
    )
)


window_height = h // n_windows


max_recenter_jump = int(
    max_recenter_jump_frac * w
)


# ============================================================
# GET EDGE PIXELS
# ============================================================

nonzero_y, nonzero_x = np.nonzero(
    edges_roi
)


# ============================================================
# DRAW SLIDING WINDOWS
# ============================================================

def draw_windows(
    base_x,
    side,
    color
):

    current_x = int(base_x)

    print()
    print(
        f"{side.upper()} WINDOWS"
    )

    print(
        "-" * 50
    )

    for i in range(
        n_windows
    ):

        # ----------------------------------------------------
        # Vertical position
        # ----------------------------------------------------

        y_low = (
            h
            - (i + 1) * window_height
        )

        y_high = (
            h
            - i * window_height
        )


        # ----------------------------------------------------
        # Horizontal position
        # ----------------------------------------------------

        x_low = (
            current_x
            - window_width // 2
        )

        x_high = (
            current_x
            + window_width // 2
        )


        # ----------------------------------------------------
        # Keep window on correct side
        # ----------------------------------------------------

        if side == "left":

            x_high = min(
                x_high,
                int(
                    left_max_x_frac
                    * w
                )
            )

        else:

            x_low = max(
                x_low,
                int(
                    right_min_x_frac
                    * w
                )
            )


        # ----------------------------------------------------
        # Find edge pixels inside window
        # ----------------------------------------------------

        selection = (
            (nonzero_y >= y_low)
            &
            (nonzero_y < y_high)
            &
            (nonzero_x >= x_low)
            &
            (nonzero_x < x_high)
        )


        ys = nonzero_y[
            selection
        ]

        xs = nonzero_x[
            selection
        ]


        pixel_count = len(
            xs
        )


        # ----------------------------------------------------
        # Draw window
        # ----------------------------------------------------

        cv2.rectangle(
            vis,
            (x_low, y_low),
            (x_high, y_high),
            color,
            2
        )


        # ----------------------------------------------------
        # Draw selected pixels
        # ----------------------------------------------------

        if pixel_count >= min_pixels_per_window:

            for x, y in zip(
                xs,
                ys
            ):

                cv2.circle(
                    vis,
                    (int(x), int(y)),
                    1,
                    color,
                    -1
                )


        # ----------------------------------------------------
        # Calculate new center
        # ----------------------------------------------------

        old_x = current_x

        recentered = False

        if (
            pixel_count
            >= min_pixels_to_recenter
        ):

            proposed_x = int(
                np.median(xs)
            )

            jump = (
                proposed_x
                - current_x
            )


            if abs(jump) <= max_recenter_jump:

                current_x = proposed_x

                recentered = True


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{side[0].upper()}{i + 1}: "
            f"{pixel_count}px"
        )


        cv2.putText(
            vis,
            label,
            (
                max(
                    2,
                    x_low + 3
                ),
                min(
                    h - 5,
                    y_low + 18
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Terminal diagnostics
        # ----------------------------------------------------

        print(
            f"W{i + 1}: "
            f"x={old_x:3d} -> "
            f"{current_x:3d} | "
            f"pixels={pixel_count:4d} | "
            f"recenter={recentered}"
        )


# ============================================================
# LEFT WINDOWS
# ============================================================

draw_windows(
    left_base,
    "left",
    (0, 255, 255)
)


# ============================================================
# RIGHT WINDOWS
# ============================================================

draw_windows(
    right_base,
    "right",
    (0, 255, 0)
)


# ============================================================
# INFORMATION TEXT
# ============================================================

cv2.putText(
    vis,
    f"Horizon: {horizon_y}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


cv2.putText(
    vis,
    f"Left base: {left_base}",
    (10, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)


cv2.putText(
    vis,
    f"Right base: {right_base}",
    (10, 75),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)


cv2.putText(
    vis,
    f"Window: {window_width} x {window_height}",
    (10, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (255, 255, 255),
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
        f"Could not save: {OUTPUT}"
    )


# ============================================================
# FINAL TERMINAL OUTPUT
# ============================================================

print()
print(
    "=" * 60
)

print(
    "SLIDING WINDOW INSPECTION"
)

print(
    "=" * 60
)

print(
    f"Input       : {INPUT}"
)

print(
    f"Time        : {TIME:.2f} s"
)

print(
    f"Frame       : {frame_number}"
)

print(
    f"Resolution  : {w} x {h}"
)

print(
    f"Horizon     : {horizon_y}"
)

print(
    f"Left base   : {left_base}"
)

print(
    f"Right base  : {right_base}"
)

print(
    f"Window size : "
    f"{window_width} x "
    f"{window_height}"
)

print(
    f"Output      : {OUTPUT}"
)

print(
    "=" * 60
)