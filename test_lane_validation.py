
from __future__ import annotations

import argparse
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
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def label(img, text, color=(255, 255, 255)):
    out = img.copy()

    cv2.putText(
        out, text, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, text, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        color, 1, cv2.LINE_AA
    )
    return out


def draw_polyline(img, coeffs, y_range, color, thickness=4):
    if coeffs is None:
        return img

    h, w = img.shape[:2]

    ys = np.linspace(
        y_range[0],
        y_range[1],
        150
    )

    xs = np.polyval(coeffs, ys)

    pts = np.column_stack(
        (xs, ys)
    ).astype(np.int32)

    pts = pts[
        (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    ]

    if len(pts) >= 2:
        cv2.polylines(
            img,
            [pts],
            False,
            color,
            thickness,
            cv2.LINE_AA
        )

    return img


def get_frame(path, frame_idx=None, time_sec=None):
    video = VideoReader(path)
    fps = video.info.fps

    if frame_idx is not None:
        idx = frame_idx
    elif time_sec is not None:
        idx = int(round(time_sec * fps))
    else:
        idx = 0

    frame = video.read_frame(idx)
    video.release()

    if frame is None:
        raise RuntimeError(f"Could not read frame {idx}")

    return frame, idx, fps


def main():
    ap = argparse.ArgumentParser(
        description="Temporary lane geometric validation inspection"
    )

    ap.add_argument("--input", required=True)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument(
        "--config",
        default="config/default.yaml"
    )
    ap.add_argument(
        "--outdir",
        default="outputs/lane_validation_check"
    )

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_config(args.config)

    # ============================================================
    # FRAME
    # ============================================================

    frame, frame_idx, fps = get_frame(
        args.input,
        args.frame,
        args.time
    )

    h, w = frame.shape[:2]
    timestamp = frame_idx / fps

    # ============================================================
    # HORIZON
    # ============================================================

    horizon = HorizonDetector(
        HorizonConfig.from_dict(
            cfg.get("horizon", {})
        )
    )

    horizon_y = horizon.detect(frame)

    # ============================================================
    # ROI + HSV + CANNY
    # ============================================================

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

    edges_roi, edges_raw, hsv_hits = edges_module.compute(
        frame,
        top_y_override=horizon_y
    )

    # ============================================================
    # QUADRATIC SLIDING-WINDOW FIT
    # ============================================================

    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(
            cfg.get("sliding_window", {})
        )
    )

    left, right = fitter.fit(
        edges_roi
    )

    # ============================================================
    # VALIDATION RANGE
    # ============================================================

    if horizon_y is not None:
        y_min = max(
            int(h * 0.62),
            int(horizon_y + 10)
        )
    else:
        y_min = int(h * 0.62)

    y_max = int(h * 0.95)

    y_range = (
        y_min,
        y_max
    )

    # ============================================================
    # GEOMETRIC VALIDATION
    # ============================================================

    validator = LaneValidation(
        ValidationConfig.from_dict(
            cfg.get("validation", {})
        )
    )

    validation = validator.validate(
        left.coeffs,
        right.coeffs,
        y_range,
        w
    )

    # ============================================================
    # CALCULATE GEOMETRIC DIAGNOSTICS FOR INSPECTION
    # ============================================================

    ys = np.linspace(
        y_range[0],
        y_range[1],
        50
    )

    left_x = None
    right_x = None
    widths = None

    if left.coeffs is not None:
        left_x = np.polyval(
            left.coeffs,
            ys
        )

    if right.coeffs is not None:
        right_x = np.polyval(
            right.coeffs,
            ys
        )

    if left_x is not None and right_x is not None:
        widths = right_x - left_x

    # ============================================================
    # IMAGE 1: ORIGINAL + HORIZON
    # ============================================================

    p1 = frame.copy()

    if horizon_y is not None:
        cv2.line(
            p1,
            (0, int(horizon_y)),
            (w - 1, int(horizon_y)),
            (0, 255, 255),
            3,
            cv2.LINE_AA
        )

    p1 = label(
        p1,
        f"ORIGINAL | frame={frame_idx} | horizon={horizon_y}"
    )

    # ============================================================
    # IMAGE 2: FIT + VALIDATION RANGE
    # ============================================================

    p2 = frame.copy()

    if horizon_y is not None:
        cv2.line(
            p2,
            (0, int(horizon_y)),
            (w - 1, int(horizon_y)),
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Validation start/end markers.
    cv2.line(
        p2,
        (0, y_range[0]),
        (w - 1, y_range[0]),
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.line(
        p2,
        (0, y_range[1]),
        (w - 1, y_range[1]),
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    p2 = draw_polyline(
        p2,
        left.coeffs,
        y_range,
        (0, 255, 0),
        5
    )

    p2 = draw_polyline(
        p2,
        right.coeffs,
        y_range,
        (0, 0, 255),
        5
    )

    p2 = label(
        p2,
        "QUADRATIC FIT | GREEN=LEFT RED=RIGHT | CYAN=VALIDATION LIMITS"
    )

    # ============================================================
    # IMAGE 3: WIDTH CHECK
    # ============================================================

    p3 = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    if widths is not None:
        min_width = float(widths.min())
        max_width = float(widths.max())

        # Plot width over y.
        width_min_cfg = validator.cfg.min_lane_width_px
        width_max_cfg = validator.cfg.max_lane_width_px

        # Scale width plot into image width.
        plot_left = 80
        plot_right = w - 25
        plot_top = 55
        plot_bottom = h - 45

        width_range = max(
            float(width_max_cfg),
            max_width,
            1.0
        )

        pts = []

        for y, lane_width in zip(
            ys,
            widths
        ):
            x_plot = int(
                plot_left
                + (lane_width / width_range)
                * (plot_right - plot_left)
            )

            y_plot = int(
                plot_bottom
                - (
                    (y - y_range[0])
                    / max(1, y_range[1] - y_range[0])
                )
                * (plot_bottom - plot_top)
            )

            pts.append(
                (x_plot, y_plot)
            )

        if len(pts) >= 2:
            cv2.polylines(
                p3,
                [np.array(pts, dtype=np.int32)],
                False,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

        # Min/max threshold lines.
        for threshold, color, text in [
            (
                width_min_cfg,
                (0, 255, 255),
                f"MIN allowed = {width_min_cfg}px"
            ),
            (
                width_max_cfg,
                (0, 0, 255),
                f"MAX allowed = {width_max_cfg}px"
            ),
        ]:
            x_threshold = int(
                plot_left
                + (threshold / width_range)
                * (plot_right - plot_left)
            )

            cv2.line(
                p3,
                (x_threshold, plot_top),
                (x_threshold, plot_bottom),
                color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                p3,
                text,
                (max(5, x_threshold - 85), 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

        cv2.putText(
            p3,
            f"WIDTH RANGE: min={min_width:.1f}px  max={max_width:.1f}px",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    else:
        cv2.putText(
            p3,
            "WIDTH CHECK: requires both fits",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    p3 = label(
        p3,
        "LANE WIDTH ACROSS VALIDATION RANGE"
    )

    # ============================================================
    # IMAGE 4: VALIDATION RESULT ON ORIGINAL
    # ============================================================

    p4 = frame.copy()

    p4 = draw_polyline(
        p4,
        left.coeffs,
        y_range,
        (0, 255, 0),
        5
    )

    p4 = draw_polyline(
        p4,
        right.coeffs,
        y_range,
        (0, 0, 255),
        5
    )

    if widths is not None:
        # Draw several width samples.
        for idx in [
            0,
            12,
            25,
            37,
            49
        ]:
            y = int(ys[idx])
            xl = int(left_x[idx])
            xr = int(right_x[idx])

            if 0 <= xl < w and 0 <= xr < w:
                cv2.line(
                    p4,
                    (xl, y),
                    (xr, y),
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    p4,
                    f"{widths[idx]:.0f}px",
                    (
                        max(5, min(w - 80, (xl + xr) // 2)),
                        max(15, y - 5)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA
                )

    status = (
        f"L={validation.left_ok} "
        f"R={validation.right_ok} "
        f"PAIR={validation.pair_ok}"
    )

    cv2.putText(
        p4,
        status,
        (10, h - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        p4,
        status,
        (10, h - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (
            (0, 255, 0)
            if validation.pair_ok
            else (0, 0, 255)
        ),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        p4,
        (
            f"L: {validation.reason_left} | "
            f"R: {validation.reason_right} | "
            f"PAIR: {validation.reason_pair}"
        ),
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    p4 = label(
        p4,
        "GEOMETRIC VALIDATION | QUADRATIC"
    )

    # ============================================================
    # SAVE
    # ============================================================

    outputs = {
        "01_original_horizon.png": p1,
        "02_quadratic_fit_validation_range.png": p2,
        "03_lane_width_check.png": p3,
        "04_validation_result.png": p4,
    }

    for name, image in outputs.items():
        cv2.imwrite(
            os.path.join(args.outdir, name),
            image
        )

    # ============================================================
    # PRINT
    # ============================================================

    print("=" * 78)
    print("LANE GEOMETRIC VALIDATION TEST")
    print("=" * 78)

    print(f"Input              : {args.input}")
    print(f"Frame              : {frame_idx}")
    print(f"Time               : {timestamp:.2f} s")
    print(f"Resolution         : {w} x {h}")
    print(f"Horizon            : {horizon_y}")
    print(
        f"Validation y-range : "
        f"{y_range[0]} -> {y_range[1]}"
    )

    print()
    print("QUADRATIC FITS")
    print("-" * 78)

    print(
        f"Left  valid={left.valid} "
        f"pixels={left.n_pixels} "
        f"RMS={left.rms_error:.2f}"
    )

    print(
        f"Right valid={right.valid} "
        f"pixels={right.n_pixels} "
        f"RMS={right.rms_error:.2f}"
    )

    print()
    print("VALIDATION CONFIG")
    print("-" * 78)

    print(
        f"max_sweep_frac     : "
        f"{validator.cfg.max_sweep_frac}"
    )
    print(
        f"min_lane_width_px  : "
        f"{validator.cfg.min_lane_width_px}"
    )
    print(
        f"max_lane_width_px  : "
        f"{validator.cfg.max_lane_width_px}"
    )

    print()
    print("VALIDATION RESULT")
    print("-" * 78)

    print(f"Left OK             : {validation.left_ok}")
    print(f"Left reason         : {validation.reason_left}")
    print(f"Right OK            : {validation.right_ok}")
    print(f"Right reason        : {validation.reason_right}")
    print(f"Pair OK             : {validation.pair_ok}")
    print(f"Pair reason         : {validation.reason_pair}")

    if widths is not None:
        print()
        print("LANE WIDTH")
        print("-" * 78)
        print(f"Minimum width       : {widths.min():.2f} px")
        print(f"Maximum width       : {widths.max():.2f} px")
        print(f"Width at top        : {widths[0]:.2f} px")
        print(f"Width at bottom     : {widths[-1]:.2f} px")

        # Lower 70% is what current validator uses for minimum width.
        width_start = int(len(widths) * 0.30)
        usable = widths[width_start:]

        print(
            f"Usable min width    : "
            f"{usable.min():.2f} px"
        )

    print()
    print(
        f"Outputs saved to    : {args.outdir}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
