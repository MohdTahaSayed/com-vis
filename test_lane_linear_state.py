"""
TEMPORARY LINEAR LANE + TEMPORAL STATE TEST
--------------------------------------------
This is an EXPERIMENT only. It does NOT modify the permanent project.

Instead of the current quadratic sliding-window fit, this test uses:

Frame
 -> Horizon
 -> ROI + Canny + HSV reinforcement
 -> Hough line segments
 -> Linear fit for left/right lane:
        x = m*y + b
 -> Simple lane-pair geometric validation
 -> Existing LaneState temporal smoothing / HOLD / MISS

Why x = m*y + b?
The image lane boundary is easier to compare vertically this way:
for a given image row y, the equation gives the lane x-position.

The existing src/lane_state.py is reused unchanged.

Usage:
    python .\test_lane_linear_state.py --input ".\data\VBOX0011_Trim.mp4"

Recommended:
    python .\test_lane_linear_state.py --input ".\data\VBOX0011_Trim.mp4" --start-frame 3456 --num-frames 40

If the temporary HSV config exists:
    python .\test_lane_linear_state.py --input ".\data\VBOX0011_Trim.mp4" --config ".\config\hsv_test.yaml"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.io_video import VideoReader
from src.horizon import HorizonDetector, HorizonConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_hough import LaneHough, HoughConfig
from src.lane_state import LaneState, StateConfig


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def unpack_segment(seg):
    """
    Convert common Hough segment representations into:
        x1, y1, x2, y2

    Supports:
        [x1, y1, x2, y2]
        [[x1, y1, x2, y2]]
        numpy equivalents
    """
    arr = np.asarray(seg).reshape(-1)

    if arr.size < 4:
        return None

    x1, y1, x2, y2 = arr[:4]

    return (
        float(x1),
        float(y1),
        float(x2),
        float(y2),
    )


def fit_linear_from_segments(
    segments,
    y_min,
    y_max,
    frame_w,
    side,
    min_segments=1,
):
    """
    Fit x = m*y + b from Hough segment endpoints.

    Only endpoints inside the useful vertical interval are used.
    Segments are weighted by their geometric length.

    Returns:
        coeffs = [m, b]
        confidence
        number of usable segments
        rms error
    """

    if segments is None:
        return None, 0.0, 0, 0.0

    xs = []
    ys = []
    weights = []

    for seg in segments:

        parsed = unpack_segment(seg)

        if parsed is None:
            continue

        x1, y1, x2, y2 = parsed

        # Ignore nearly horizontal segments.
        if abs(y2 - y1) < 5:
            continue

        length = float(
            np.hypot(
                x2 - x1,
                y2 - y1
            )
        )

        if length < 5:
            continue

        # Clip segment endpoints into validation interval.
        points = [
            (x1, y1),
            (x2, y2),
        ]

        for x, y in points:

            if y < y_min or y > y_max:
                continue

            # Broad side sanity check.
            if side == "left" and x > frame_w * 0.60:
                continue

            if side == "right" and x < frame_w * 0.40:
                continue

            xs.append(x)
            ys.append(y)
            weights.append(length)

    if len(xs) < max(2, min_segments):
        return None, 0.0, len(xs), 0.0

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # x = m*y + b
    try:
        m, b = np.polyfit(
            ys,
            xs,
            1,
            w=np.sqrt(weights),
        )
    except Exception:
        return None, 0.0, len(xs), 0.0

    coeffs = np.array(
        [m, b],
        dtype=np.float64
    )

    predicted = m * ys + b

    rms = float(
        np.sqrt(
            np.mean(
                (xs - predicted) ** 2
            )
        )
    )

    # Confidence based on usable endpoint count.
    confidence = min(
        1.0,
        len(xs) / 30.0
    )

    return (
        coeffs,
        confidence,
        len(xs),
        rms,
    )


def linear_x(coeffs, y):
    if coeffs is None:
        return None

    m, b = coeffs
    return float(m * y + b)


def validate_linear_lane_pair(
    left,
    right,
    y_min,
    y_max,
    frame_w,
    min_width=30,
    max_width=800,
):
    """
    Simple validation for x = m*y+b lane equations.

    Checks:
      1. Both fits exist.
      2. Left stays left of right.
      3. Lane width is reasonable.
      4. Curves remain inside a broad image boundary.
      5. Left/right orientation is sensible.
    """

    if left is None:
        return False, False, False, "no_left", "no_right", "need_both"

    if right is None:
        return False, False, False, "", "no_right", "need_both"

    ys = np.linspace(
        y_min,
        y_max,
        40
    )

    xs_l = left[0] * ys + left[1]
    xs_r = right[0] * ys + right[1]

    left_ok = True
    right_ok = True

    left_reason = "ok"
    right_reason = "ok"

    # Orientation:
    # left boundary normally moves right as y increases.
    # right boundary normally moves left as y increases.
    if left[0] <= 0:
        left_ok = False
        left_reason = f"left_slope={left[0]:.3f}"

    if right[0] >= 0:
        right_ok = False
        right_reason = f"right_slope={right[0]:.3f}"

    # Broad image limits.
    if xs_l.min() < -0.15 * frame_w or xs_l.max() > 1.15 * frame_w:
        left_ok = False
        left_reason = "left_off_canvas"

    if xs_r.min() < -0.15 * frame_w or xs_r.max() > 1.15 * frame_w:
        right_ok = False
        right_reason = "right_off_canvas"

    pair_ok = False
    pair_reason = "need_both"

    if left_ok and right_ok:

        widths = xs_r - xs_l

        if np.any(widths <= 0):
            pair_reason = "crossing"

        else:
            w_min = float(widths.min())
            w_max = float(widths.max())

            if w_min < min_width:
                pair_reason = f"width_min={w_min:.0f}"

            elif w_max > max_width:
                pair_reason = f"width_max={w_max:.0f}"

            else:
                pair_ok = True
                pair_reason = "ok"

    return (
        left_ok,
        right_ok,
        pair_ok,
        left_reason,
        right_reason,
        pair_reason,
    )


def draw_linear_line(
    img,
    coeffs,
    y_min,
    y_max,
    color,
    thickness,
):
    if coeffs is None:
        return

    h, w = img.shape[:2]

    ys = np.linspace(
        y_min,
        y_max,
        100
    )

    xs = coeffs[0] * ys + coeffs[1]

    pts = np.column_stack(
        (xs, ys)
    ).astype(np.int32)

    valid = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    )

    pts = pts[valid]

    if len(pts) >= 2:
        cv2.polylines(
            img,
            [pts],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )


def put_text(img, text, y, color=(255, 255, 255)):
    cv2.putText(
        img,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


def main():

    ap = argparse.ArgumentParser(
        description="Temporary linear lane + temporal state test"
    )

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--start-frame",
        type=int,
        default=3456,
    )

    ap.add_argument(
        "--num-frames",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--config",
        default="config/default.yaml",
    )

    ap.add_argument(
        "--outdir",
        default="outputs/linear_lane_state_check",
    )

    args = ap.parse_args()

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    cfg = load_config(
        args.config
    )

    # ============================================================
    # EXISTING PIPELINE COMPONENTS
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

    hough = LaneHough(
        HoughConfig.from_dict(
            cfg.get("hough", {})
        )
    )

    state_cfg = StateConfig.from_dict(
        cfg.get("lane_state", {})
    )

    lane_state = LaneState(
        state_cfg
    )

    video = VideoReader(
        args.input
    )

    fps = video.info.fps

    records = []

    print("=" * 86)
    print("LINEAR LANE + TEMPORAL STATE TEST")
    print("=" * 86)
    print(f"Input              : {args.input}")
    print(f"Start frame        : {args.start_frame}")
    print(f"Frames tested      : {args.num_frames}")
    print(f"FPS                : {fps:.2f}")
    print()
    print("LINE EQUATION")
    print("-" * 86)
    print("x = m*y + b")
    print()
    print("LANE STATE")
    print("-" * 86)
    print(f"smoothing_alpha    : {state_cfg.smoothing_alpha}")
    print(f"hold_frames        : {state_cfg.hold_frames}")
    print(f"miss_frames        : {state_cfg.miss_frames}")
    print(f"min_confidence     : {state_cfg.min_confidence}")
    print("=" * 86)

    for i in range(args.num_frames):

        frame_idx = (
            args.start_frame + i
        )

        frame = video.read_frame(
            frame_idx
        )

        if frame is None:
            print(
                f"[STOP] Could not read frame {frame_idx}"
            )
            break

        h, w = frame.shape[:2]

        timestamp = (
            frame_idx / fps
        )

        # --------------------------------------------------------
        # HORIZON
        # --------------------------------------------------------

        horizon_y = horizon.detect(
            frame
        )

        # --------------------------------------------------------
        # EDGES
        # --------------------------------------------------------

        edges_roi, edges_raw, hsv_hits = (
            edges_module.compute(
                frame,
                top_y_override=horizon_y,
            )
        )

        # --------------------------------------------------------
        # HOUGH
        # --------------------------------------------------------

        (
            left_segments,
            right_segments,
            discard_segments,
        ) = hough.classify(
            edges_roi
        )

        # --------------------------------------------------------
        # VALIDATION RANGE
        # --------------------------------------------------------

        if horizon_y is not None:
            y_min = max(
                int(h * 0.62),
                int(horizon_y + 10)
            )
        else:
            y_min = int(h * 0.62)

        y_max = int(h * 0.95)

        # --------------------------------------------------------
        # LINEAR FIT
        # --------------------------------------------------------

        left_raw, left_conf, left_n, left_rms = (
            fit_linear_from_segments(
                left_segments,
                y_min,
                y_max,
                w,
                "left",
            )
        )

        right_raw, right_conf, right_n, right_rms = (
            fit_linear_from_segments(
                right_segments,
                y_min,
                y_max,
                w,
                "right",
            )
        )

        # --------------------------------------------------------
        # GEOMETRIC VALIDATION
        # --------------------------------------------------------

        (
            left_ok,
            right_ok,
            pair_ok,
            left_reason,
            right_reason,
            pair_reason,
        ) = validate_linear_lane_pair(
            left_raw,
            right_raw,
            y_min,
            y_max,
            w,
        )

        # --------------------------------------------------------
        # SEND ONLY VALID INDIVIDUAL LINES TO LaneState
        # --------------------------------------------------------

        state_left = (
            left_raw
            if left_ok
            else None
        )

        state_right = (
            right_raw
            if right_ok
            else None
        )

        state_left_conf = (
            left_conf
            if left_ok
            else 0.0
        )

        state_right_conf = (
            right_conf
            if right_ok
            else 0.0
        )

        # --------------------------------------------------------
        # TEMPORAL SMOOTHING
        # --------------------------------------------------------

        (
            (smooth_left, left_status, left_state_conf),
            (smooth_right, right_status, right_state_conf),
        ) = lane_state.update(
            state_left,
            state_left_conf,
            state_right,
            state_right_conf,
        )

        # --------------------------------------------------------
        # POSITION AT y=85%
        # --------------------------------------------------------

        eval_y = int(
            h * 0.85
        )

        raw_left_x = linear_x(
            left_raw,
            eval_y
        )

        raw_right_x = linear_x(
            right_raw,
            eval_y
        )

        smooth_left_x = linear_x(
            smooth_left,
            eval_y
        )

        smooth_right_x = linear_x(
            smooth_right,
            eval_y
        )

        # --------------------------------------------------------
        # RECORD
        # --------------------------------------------------------

        records.append({
            "frame": frame_idx,
            "time_sec": timestamp,
            "horizon_y": horizon_y,

            "left_segments": len(left_segments),
            "right_segments": len(right_segments),

            "left_points": left_n,
            "right_points": right_n,

            "left_rms": left_rms,
            "right_rms": right_rms,

            "left_valid": left_ok,
            "right_valid": right_ok,
            "pair_valid": pair_ok,

            "left_status": left_status,
            "right_status": right_status,

            "left_fail_count": lane_state.left.fail_count,
            "right_fail_count": lane_state.right.fail_count,

            "left_confidence": left_state_conf,
            "right_confidence": right_state_conf,

            "raw_left_x_y85": raw_left_x,
            "raw_right_x_y85": raw_right_x,

            "smooth_left_x_y85": smooth_left_x,
            "smooth_right_x_y85": smooth_right_x,

            "left_m": (
                float(left_raw[0])
                if left_raw is not None
                else None
            ),

            "left_b": (
                float(left_raw[1])
                if left_raw is not None
                else None
            ),

            "right_m": (
                float(right_raw[0])
                if right_raw is not None
                else None
            ),

            "right_b": (
                float(right_raw[1])
                if right_raw is not None
                else None
            ),

            "left_reason": left_reason,
            "right_reason": right_reason,
            "pair_reason": pair_reason,
        })

        # --------------------------------------------------------
        # TERMINAL
        # --------------------------------------------------------

        print(
            f"{frame_idx:5d} | "
            f"L={left_status:5s} "
            f"c={left_state_conf:.2f} "
            f"m={left_raw[0]: .4f} "
            f"b={left_raw[1]:7.1f} "
            if left_raw is not None
            else
            f"{frame_idx:5d} | "
            f"L={left_status:5s} "
            f"c={left_state_conf:.2f} "
            f"m= N/A     b= N/A   "
        , end="")

        print(
            f"| R={right_status:5s} "
            f"c={right_state_conf:.2f} "
            f"m={right_raw[0]: .4f} "
            f"b={right_raw[1]:7.1f} "
            if right_raw is not None
            else
            f"| R={right_status:5s} "
            f"c={right_state_conf:.2f} "
            f"m= N/A     b= N/A   ",
            end=""
        )

        print(
            f"| PAIR={str(pair_ok):5s} "
            f"| Lfail={lane_state.left.fail_count:<2d} "
            f"Rfail={lane_state.right.fail_count:<2d}"
        )

        # --------------------------------------------------------
        # SAVE VISUALS
        # --------------------------------------------------------

        if (
            i == 0
            or i == args.num_frames - 1
            or i % 10 == 0
        ):

            vis = frame.copy()

            if horizon_y is not None:
                cv2.line(
                    vis,
                    (0, int(horizon_y)),
                    (w - 1, int(horizon_y)),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Raw Hough-derived linear fits.
            draw_linear_line(
                vis,
                left_raw,
                y_min,
                y_max,
                (255, 255, 0),
                2,
            )

            draw_linear_line(
                vis,
                right_raw,
                y_min,
                y_max,
                (255, 255, 0),
                2,
            )

            # Smoothed lines.
            draw_linear_line(
                vis,
                smooth_left,
                y_min,
                y_max,
                (0, 255, 0),
                5,
            )

            draw_linear_line(
                vis,
                smooth_right,
                y_min,
                y_max,
                (0, 0, 255),
                5,
            )

            put_text(
                vis,
                f"FRAME {frame_idx} | t={timestamp:.2f}s",
                25,
            )

            put_text(
                vis,
                "YELLOW=RAW LINE | GREEN/RED=SMOOTHED",
                50,
            )

            put_text(
                vis,
                f"STATE L={left_status} R={right_status}",
                75,
            )

            put_text(
                vis,
                f"PAIR={pair_ok} | "
                f"Lconf={left_state_conf:.2f} "
                f"Rconf={right_state_conf:.2f}",
                100,
            )

            put_text(
                vis,
                f"Hough L={len(left_segments)} "
                f"R={len(right_segments)}",
                125,
            )

            cv2.imwrite(
                os.path.join(
                    args.outdir,
                    f"frame_{i:04d}.png"
                ),
                vis,
            )

    video.release()

    # ============================================================
    # CSV
    # ============================================================

    csv_path = os.path.join(
        args.outdir,
        "linear_lane_state_results.csv"
    )

    if records:

        fieldnames = list(
            records[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(records)

    # ============================================================
    # SUMMARY
    # ============================================================

    if records:

        left_ok_count = sum(
            r["left_status"] == "OK"
            for r in records
        )

        left_hold_count = sum(
            r["left_status"] == "HOLD"
            for r in records
        )

        left_miss_count = sum(
            r["left_status"] == "MISS"
            for r in records
        )

        right_ok_count = sum(
            r["right_status"] == "OK"
            for r in records
        )

        right_hold_count = sum(
            r["right_status"] == "HOLD"
            for r in records
        )

        right_miss_count = sum(
            r["right_status"] == "MISS"
            for r in records
        )

        pair_ok_count = sum(
            r["pair_valid"]
            for r in records
        )

        print()
        print("=" * 86)
        print("LINEAR LANE TEST COMPLETE")
        print("=" * 86)
        print(f"Frames processed     : {len(records)}")
        print()
        print("LEFT")
        print(f"OK                   : {left_ok_count}")
        print(f"HOLD                 : {left_hold_count}")
        print(f"MISS                 : {left_miss_count}")
        print()
        print("RIGHT")
        print(f"OK                   : {right_ok_count}")
        print(f"HOLD                 : {right_hold_count}")
        print(f"MISS                 : {right_miss_count}")
        print()
        print(
            f"Pair validation TRUE : "
            f"{pair_ok_count}/{len(records)}"
        )
        print()
        print(f"CSV                  : {csv_path}")
        print(f"Images               : {args.outdir}")
        print("=" * 86)


if __name__ == "__main__":
    main()
