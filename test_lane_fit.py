from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.io_video import VideoReader


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.putText(
        out, text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        color, 1, cv2.LINE_AA
    )
    return out


def get_frame(args):
    video = VideoReader(args.input)
    fps = video.info.fps

    if args.frame is not None:
        idx = args.frame
    elif args.time is not None:
        idx = int(round(args.time * fps))
    else:
        idx = 0

    frame = video.read_frame(idx)
    video.release()

    if frame is None:
        raise RuntimeError(f"Could not read frame {idx}")

    return frame, idx, fps


def make_lane_pixels(frame, cfg):
    horizon = HorizonDetector(
        HorizonConfig.from_dict(cfg.get("horizon", {}))
    )
    roi = LaneROI(
        RoiConfig.from_dict(cfg.get("roi", {}))
    )
    lane_color = LaneColor(
        LaneColorConfig.from_dict(cfg.get("lane_color", {}))
    )

    edges = LaneEdges(
        CannyConfig.from_dict(cfg.get("canny", {})),
        roi,
        lane_color,
        reinforce_with_hsv=True,
    )

    horizon_y = horizon.detect(frame)
    edges_roi, _, _ = edges.compute(
        frame,
        top_y_override=horizon_y
    )

    return edges_roi, horizon_y


def histogram_and_bases(binary, fitter):
    h, w = binary.shape[:2]

    bottom_start = int(h * 0.70)
    histogram = np.sum(
        binary[bottom_start:, :] > 0,
        axis=0
    ).astype(np.float32)

    kernel_size = 15
    if w >= kernel_size:
        kernel = np.ones(kernel_size, dtype=np.float32)
        kernel /= kernel.sum()
        smooth = np.convolve(
            histogram, kernel, mode="same"
        )
    else:
        smooth = histogram

    left_end = int(w * 0.45)
    left_region = smooth[:left_end]
    right_start = int(w * 0.55)
    right_region = smooth[right_start:]

    if np.max(left_region) > 0:
        left_base = int(np.argmax(left_region))
    else:
        left_base = int(w * 0.25)

    left_base += fitter.cfg.left_base_shift_px
    left_base = max(0, min(w - 1, left_base))

    if np.max(right_region) > 0:
        right_base = int(
            right_start + np.argmax(right_region)
        )
    else:
        right_base = int(w * 0.75)

    right_base += fitter.cfg.right_base_shift_px
    right_base = max(0, min(w - 1, right_base))

    return histogram, smooth, left_base, right_base


def draw_histogram(binary, histogram, smooth, left_base, right_base):
    h, w = binary.shape[:2]
    canvas_h = 220
    canvas = np.zeros((canvas_h, w, 3), dtype=np.uint8)

    max_val = max(float(np.max(smooth)), 1.0)

    for x in range(w):
        y = int(
            canvas_h - 1
            - (smooth[x] / max_val) * (canvas_h - 20)
        )
        cv2.line(
            canvas,
            (x, canvas_h - 1),
            (x, max(0, y)),
            (180, 180, 180),
            1
        )

    cv2.line(
        canvas,
        (left_base, 0),
        (left_base, canvas_h),
        (0, 255, 255),
        3
    )
    cv2.line(
        canvas,
        (right_base, 0),
        (right_base, canvas_h),
        (0, 255, 0),
        3
    )

    cv2.putText(
        canvas,
        f"Histogram | L base={left_base} | R base={right_base}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return canvas


def draw_windows(binary, x_base, side, cfg):
    h, w = binary.shape[:2]

    n = max(1, cfg.n_windows)
    win_w = max(
        20,
        int(cfg.window_width_frac * w)
    )
    win_h = max(1, h // n)
    max_jump = cfg.max_recenter_jump_frac * w

    nonzero_y, nonzero_x = np.nonzero(binary)

    current_x = float(x_base)

    out = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR
    )

    window_color = (
        (0, 255, 255)
        if side == "LEFT"
        else (0, 255, 0)
    )

    counts = []

    for i in range(n):
        y_low = max(
            0,
            h - (i + 1) * win_h
        )
        y_high = min(
            h,
            h - i * win_h
        )

        x_low = int(
            current_x - win_w / 2
        )
        x_high = int(
            current_x + win_w / 2
        )

        cv2.rectangle(
            out,
            (x_low, y_low),
            (x_high, y_high),
            window_color,
            2
        )

        selection = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= x_low)
            & (nonzero_x < x_high)
        )

        xs = nonzero_x[selection]
        ys = nonzero_y[selection]

        counts.append(len(xs))

        # Draw selected lane pixels
        for x, y in zip(xs, ys):
            cv2.circle(
                out,
                (int(x), int(y)),
                1,
                (0, 0, 255),
                -1
            )

        if len(xs) >= cfg.min_pixels_per_window:
            if len(xs) >= cfg.min_pixels_to_recenter:
                new_x = float(np.median(xs))
                delta = new_x - current_x

                if abs(delta) > max_jump:
                    if delta > 0:
                        new_x = current_x + max_jump
                    else:
                        new_x = current_x - max_jump

                current_x = new_x

        cv2.putText(
            out,
            f"W{i+1}: {len(xs)} px",
            (max(5, x_low), max(15, y_low + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    out = label(
        out,
        f"SLIDING WINDOWS - {side} | base={x_base}"
    )

    return out, counts


def draw_fit(frame, left, right, y_range):
    out = frame.copy()
    h, w = frame.shape[:2]

    # Draw selected pixels
    if left.x_pixels is not None and left.y_pixels is not None:
        for x, y in zip(left.x_pixels, left.y_pixels):
            cv2.circle(
                out, (int(x), int(y)), 1,
                (255, 255, 0), -1
            )

    if right.x_pixels is not None and right.y_pixels is not None:
        for x, y in zip(right.x_pixels, right.y_pixels):
            cv2.circle(
                out, (int(x), int(y)), 1,
                (0, 255, 255), -1
            )

    ys = np.linspace(
        y_range[0],
        y_range[1],
        100
    )

    if left.coeffs is not None:
        xs = np.polyval(left.coeffs, ys)
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
                out,
                [pts],
                False,
                (0, 255, 0),
                4,
                cv2.LINE_AA
            )

    if right.coeffs is not None:
        xs = np.polyval(right.coeffs, ys)
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
                out,
                [pts],
                False,
                (0, 0, 255),
                4,
                cv2.LINE_AA
            )

    out = label(
        out,
        "POLYNOMIAL FIT | LEFT=GREEN RIGHT=RED"
    )

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Temporary sliding-window lane-fit inspection"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--outdir",
        default="outputs/lane_fit_check"
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_config(args.config)
    frame, frame_idx, fps = get_frame(args)

    h, w = frame.shape[:2]
    timestamp = frame_idx / fps

    edges_roi, horizon_y = make_lane_pixels(
        frame, cfg
    )

    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(
            cfg.get("sliding_window", {})
        )
    )

    # Single-frame inspection: no previous coefficients.
    left, right = fitter.fit(
        edges_roi,
        previous_left=None,
        previous_right=None
    )

    histogram, smooth, left_base, right_base = (
        histogram_and_bases(
            edges_roi,
            fitter
        )
    )

    left_windows, left_counts = draw_windows(
        edges_roi,
        left_base,
        "LEFT",
        fitter.cfg
    )

    right_windows, right_counts = draw_windows(
        edges_roi,
        right_base,
        "RIGHT",
        fitter.cfg
    )

    both_windows = np.hstack(
        [left_windows, right_windows]
    )

    # Validation-range-like display for the fit itself.
    if horizon_y is not None:
        y_top = max(
            int(h * 0.62),
            horizon_y + 10
        )
    else:
        y_top = int(h * 0.62)

    y_bottom = int(h * 0.95)
    y_range = (y_top, y_bottom)

    fit_display = draw_fit(
        frame,
        left,
        right,
        y_range
    )

    original = label(
        frame.copy(),
        f"ORIGINAL | frame={frame_idx} | t={timestamp:.2f}s"
    )

    lane_pixels_display = label(
        cv2.cvtColor(
            edges_roi,
            cv2.COLOR_GRAY2BGR
        ),
        f"LANE PIXELS | horizon={horizon_y}"
    )

    histogram_display = draw_histogram(
        edges_roi,
        histogram,
        smooth,
        left_base,
        right_base
    )

    outputs = {
        "01_original.png": original,
        "02_lane_pixels.png": lane_pixels_display,
        "03_histogram.png": histogram_display,
        "04_left_windows.png": left_windows,
        "05_right_windows.png": right_windows,
        "06_windows_both.png": both_windows,
        "07_polynomial_fit.png": fit_display,
    }

    for name, img in outputs.items():
        cv2.imwrite(
            os.path.join(args.outdir, name),
            img
        )

    print("=" * 75)
    print("SLIDING-WINDOW POLYNOMIAL FIT INSPECTION")
    print("=" * 75)
    print(f"Input             : {args.input}")
    print(f"Frame             : {frame_idx}")
    print(f"Time              : {timestamp:.2f} s")
    print(f"Resolution        : {w} x {h}")
    print(f"FPS               : {fps:.2f}")
    print(f"Horizon Y         : {horizon_y}")
    print()
    print("ACTIVE CONFIG")
    print(f"n_windows         : {fitter.cfg.n_windows}")
    print(f"window_width_frac : {fitter.cfg.window_width_frac}")
    print(f"min_px_recenter   : {fitter.cfg.min_pixels_to_recenter}")
    print(f"min_px/window     : {fitter.cfg.min_pixels_per_window}")
    print(f"min_px_total      : {fitter.cfg.min_pixels_total}")
    print(f"max_fit_rms_px    : {fitter.cfg.max_fit_rms_px}")
    print(f"max_jump_frac     : {fitter.cfg.max_recenter_jump_frac}")
    print(f"tracking_enabled  : {fitter.cfg.tracking_enabled}")
    print(f"tracking_margin   : {fitter.cfg.tracking_margin_px}")
    print()
    print("HISTOGRAM BASES")
    print(f"Left base         : {left_base}")
    print(f"Right base        : {right_base}")
    print()
    print("WINDOW PIXEL COUNTS (bottom -> top)")
    print(f"Left              : {left_counts}")
    print(f"Right             : {right_counts}")
    print()
    print("FIT RESULTS")
    print(
        f"Left  valid={left.valid} "
        f"pixels={left.n_pixels} "
        f"RMS={left.rms_error:.2f} "
        f"confidence={left.confidence:.3f} "
        f"base={left.x_base}"
    )
    print(
        f"Right valid={right.valid} "
        f"pixels={right.n_pixels} "
        f"RMS={right.rms_error:.2f} "
        f"confidence={right.confidence:.3f} "
        f"base={right.x_base}"
    )

    if left.coeffs is not None:
        print(f"Left  coeffs       : {left.coeffs}")
    else:
        print("Left  coeffs       : None")

    if right.coeffs is not None:
        print(f"Right coeffs       : {right.coeffs}")
    else:
        print("Right coeffs       : None")

    print()
    print(f"Outputs saved to   : {args.outdir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
