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
from src.lane_hough import LaneHough, HoughConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.putText(
        out, text, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, text, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        color, 1, cv2.LINE_AA
    )
    return out


def overlay_mask(frame, mask, color):
    out = frame.copy()
    out[mask > 0] = color
    return out


def draw_segments(frame, segments, color, thickness=2):
    out = frame.copy()
    for s in segments:
        cv2.line(
            out,
            (int(s.x1), int(s.y1)),
            (int(s.x2), int(s.y2)),
            color,
            thickness,
            cv2.LINE_AA
        )
    return out


def draw_roi(frame, roi, horizon_y):
    out = frame.copy()

    # Recreate normalized ROI polygon for visualization.
    vertices = roi.cfg.vertices
    pts = np.array(
        [[int(x * frame.shape[1]), int(y * frame.shape[0])]
         for x, y in vertices],
        dtype=np.int32
    )

    cv2.polylines(
        out, [pts], True, (255, 0, 255), 3, cv2.LINE_AA
    )

    cv2.line(
        out,
        (0, int(horizon_y)),
        (frame.shape[1] - 1, int(horizon_y)),
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    return out


def collect_pixels_for_side(fitter, binary, x_base, side):
    """
    Uses the project's existing LaneFitter pixel-collection logic
    with the same sliding-window parameters.
    No temporal previous coefficients are supplied for this
    single-frame inspection.
    """
    return fitter._collect_side_pixels(
        binary,
        x_base,
        side,
        previous_coeffs=None
    )


def linear_fit(x_pixels, y_pixels, min_pixels_total, max_rms):
    if len(x_pixels) < min_pixels_total:
        return None, float("inf"), False

    try:
        # LINEAR: x = m*y + b
        coeffs = np.polyfit(y_pixels, x_pixels, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None, float("inf"), False

    predicted = np.polyval(coeffs, y_pixels)
    residuals = x_pixels - predicted
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return coeffs, rms, rms <= max_rms


def curve_direction_ok(coeffs, height, side):
    y_bottom = int(height * 0.90)
    y_top = int(height * 0.55)

    x_bottom = float(np.polyval(coeffs, y_bottom))
    x_top = float(np.polyval(coeffs, y_top))

    if side == "left":
        return x_bottom < x_top
    if side == "right":
        return x_bottom > x_top
    return True


def draw_fit(frame, left_coeffs, right_coeffs, y_top, y_bottom):
    out = frame.copy()
    h, w = frame.shape[:2]

    ys = np.linspace(y_top, y_bottom, 150)

    if left_coeffs is not None:
        xs = np.polyval(left_coeffs, ys)
        pts = np.column_stack((xs, ys)).astype(np.int32)
        pts = pts[
            (pts[:, 0] >= 0) & (pts[:, 0] < w) &
            (pts[:, 1] >= 0) & (pts[:, 1] < h)
        ]
        if len(pts) >= 2:
            cv2.polylines(
                out, [pts], False, (0, 255, 0), 4, cv2.LINE_AA
            )

    if right_coeffs is not None:
        xs = np.polyval(right_coeffs, ys)
        pts = np.column_stack((xs, ys)).astype(np.int32)
        pts = pts[
            (pts[:, 0] >= 0) & (pts[:, 0] < w) &
            (pts[:, 1] >= 0) & (pts[:, 1] < h)
        ]
        if len(pts) >= 2:
            cv2.polylines(
                out, [pts], False, (0, 0, 255), 4, cv2.LINE_AA
            )

    return label(
        out,
        "LINEAR FIT | LEFT=GREEN | RIGHT=RED"
    )


def draw_windows(binary, fitter, left_base, right_base):
    """
    Visualization of the exact same sliding-window geometry used
    by LaneFitter._collect_side_pixels().
    """
    h, w = binary.shape[:2]
    n = fitter.cfg.n_windows
    window_height = max(1, h // n)
    window_width = max(
        10,
        int(w * fitter.cfg.window_width_frac)
    )

    base_canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    canvases = []
    counts_by_side = {}

    for side, x_base, color in [
        ("LEFT", left_base, (0, 255, 255)),
        ("RIGHT", right_base, (0, 255, 0)),
    ]:
        canvas = base_canvas.copy()
        current_x = int(x_base)

        nonzero_y, nonzero_x = np.nonzero(binary > 0)
        counts = []

        for window in range(n):
            y_high = h - window * window_height
            y_low = max(
                0, h - (window + 1) * window_height
            )

            half = window_width // 2
            x_low = max(0, current_x - half)
            x_high = min(w, current_x + half)

            cv2.rectangle(
                canvas,
                (x_low, y_low),
                (x_high, y_high),
                color,
                2
            )

            region = (
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= x_low)
                & (nonzero_x < x_high)
            )

            xs = nonzero_x[region]
            ys = nonzero_y[region]
            counts.append(len(xs))

            for x, y in zip(xs, ys):
                cv2.circle(
                    canvas, (int(x), int(y)), 1,
                    (0, 0, 255), -1
                )

            cv2.putText(
                canvas,
                f"W{window + 1}: {len(xs)} px",
                (max(3, x_low), max(15, y_low + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            if len(xs) >= fitter.cfg.min_pixels_to_recenter:
                new_x = int(np.median(xs))
                max_jump = int(
                    w * fitter.cfg.max_recenter_jump_frac
                )
                if abs(new_x - current_x) <= max_jump:
                    current_x = new_x

        canvas = label(
            canvas,
            f"SLIDING WINDOWS - {side} | base={x_base}"
        )
        canvases.append(canvas)
        counts_by_side[side] = counts

    both = np.hstack(canvases)
    return canvases[0], canvases[1], both, counts_by_side


def make_histogram(binary, fitter):
    h, w = binary.shape[:2]
    bottom_start = int(h * 0.70)

    histogram = np.sum(
        binary[bottom_start:, :] > 0,
        axis=0
    ).astype(np.float32)

    kernel = np.ones(15, dtype=np.float32)
    kernel /= kernel.sum()
    smooth = np.convolve(
        histogram, kernel, mode="same"
    )

    left_end = int(w * 0.45)
    left_region = smooth[:left_end]

    if np.max(left_region) > 0:
        left_base = int(np.argmax(left_region))
    else:
        left_base = int(w * 0.25)

    left_base += fitter.cfg.left_base_shift_px
    left_base = int(np.clip(left_base, 0, w - 1))

    right_start = int(w * 0.55)
    right_region = smooth[right_start:]

    if np.max(right_region) > 0:
        right_base = (
            int(np.argmax(right_region)) + right_start
        )
    else:
        right_base = int(w * 0.75)

    right_base += fitter.cfg.right_base_shift_px
    right_base = int(np.clip(right_base, 0, w - 1))

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
        canvas, (left_base, 0),
        (left_base, canvas_h),
        (0, 255, 255), 3
    )
    cv2.line(
        canvas, (right_base, 0),
        (right_base, canvas_h),
        (0, 255, 0), 3
    )

    canvas = label(
        canvas,
        f"HISTOGRAM | L base={left_base} | R base={right_base}"
    )

    return canvas, left_base, right_base


def main():
    ap = argparse.ArgumentParser(
        description="Temporary linear lane pipeline inspection"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--horizon-y", type=int, default=340)
    ap.add_argument(
        "--outdir",
        default="outputs/linear_pipeline_check"
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_config(args.config)

    # ------------------------------------------------------------
    # FRAME
    # ------------------------------------------------------------
    video = VideoReader(args.input)
    fps = video.info.fps

    if args.frame is not None:
        frame_idx = args.frame
    elif args.time is not None:
        frame_idx = int(round(args.time * fps))
    else:
        frame_idx = 0

    frame = video.read_frame(frame_idx)
    video.release()

    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx}")

    h, w = frame.shape[:2]
    timestamp = frame_idx / fps

    # ------------------------------------------------------------
    # HORIZON
    # ------------------------------------------------------------
    horizon_detector = HorizonDetector(
        HorizonConfig.from_dict(cfg.get("horizon", {}))
    )
    detected_horizon = horizon_detector.detect(frame)

    # User-requested temporary lower horizon cutoff.
    horizon_y = int(np.clip(args.horizon_y, 0, h - 1))

    # ------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------
    roi = LaneROI(
        RoiConfig.from_dict(cfg.get("roi", {}))
    )

    roi_context = draw_roi(
        frame, roi, horizon_y
    )

    # ------------------------------------------------------------
    # HSV
    # ------------------------------------------------------------
    lane_color = LaneColor(
        LaneColorConfig.from_dict(cfg.get("lane_color", {}))
    )

    white, yellow, union = lane_color.masks(frame)

    white_roi = roi.apply(
        white,
        top_y_override=horizon_y
    )
    union_roi = roi.apply(
        union,
        top_y_override=horizon_y
    )

    # ------------------------------------------------------------
    # CANNY + HSV REINFORCEMENT
    # ------------------------------------------------------------
    edges = LaneEdges(
        CannyConfig.from_dict(cfg.get("canny", {})),
        roi,
        lane_color,
        reinforce_with_hsv=True,
    )

    edges_roi, _, _ = edges.compute(
        frame,
        top_y_override=horizon_y
    )

    # Same reinforcement logic explicitly for visualization.
    union_dil = cv2.dilate(
        union_roi,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    reinforced = cv2.bitwise_and(
        edges_roi,
        union_dil
    )

    if int((reinforced > 0).sum()) < 200:
        hough_input = edges_roi
        hsv_fallback = True
    else:
        hough_input = reinforced
        hsv_fallback = False

    # ------------------------------------------------------------
    # HOUGH
    # ------------------------------------------------------------
    hough = LaneHough(
        HoughConfig.from_dict(
            cfg.get("hough", {})
        )
    )

    left_segments, right_segments, discarded = (
        hough.classify(hough_input)
    )

    hough_overlay = frame.copy()
    hough_overlay = draw_segments(
        hough_overlay,
        left_segments,
        (255, 0, 0),
        3
    )
    hough_overlay = draw_segments(
        hough_overlay,
        right_segments,
        (0, 0, 255),
        3
    )
    hough_overlay = label(
        hough_overlay,
        f"HOUGH | LEFT={len(left_segments)} RIGHT={len(right_segments)}"
    )

    # ------------------------------------------------------------
    # SLIDING WINDOWS + LINEAR FIT
    # ------------------------------------------------------------
    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(
            cfg.get("sliding_window", {})
        )
    )

    # We deliberately use the SAME histogram/base calculation.
    histogram_img, left_base, right_base = make_histogram(
        hough_input,
        fitter
    )

    left_x, left_y = collect_pixels_for_side(
        fitter,
        hough_input,
        left_base,
        "left"
    )

    right_x, right_y = collect_pixels_for_side(
        fitter,
        hough_input,
        right_base,
        "right"
    )

    # Linear instead of the project's normal quadratic fit.
    left_coeffs, left_rms, left_rms_ok = linear_fit(
        left_x,
        left_y,
        fitter.cfg.min_pixels_total,
        fitter.cfg.max_fit_rms_px
    )

    right_coeffs, right_rms, right_rms_ok = linear_fit(
        right_x,
        right_y,
        fitter.cfg.min_pixels_total,
        fitter.cfg.max_fit_rms_px
    )

    left_direction_ok = (
        left_coeffs is not None
        and curve_direction_ok(
            left_coeffs, h, "left"
        )
    )
    right_direction_ok = (
        right_coeffs is not None
        and curve_direction_ok(
            right_coeffs, h, "right"
        )
    )

    # ------------------------------------------------------------
    # VISUAL OUTPUTS
    # ------------------------------------------------------------
    outputs = {}

    outputs["01_original.png"] = label(
        frame.copy(),
        f"ORIGINAL | frame={frame_idx} | t={timestamp:.2f}s"
    )

    outputs["02_horizon_roi.png"] = label(
        roi_context,
        f"HORIZON | detected={detected_horizon} | TEST={horizon_y}"
    )

    hsv_overlay = overlay_mask(
        frame,
        white_roi,
        (0, 255, 0)
    )
    outputs["03_hsv_roi.png"] = label(
        hsv_overlay,
        "HSV WHITE HITS | ROI"
    )

    outputs["04_canny_roi.png"] = label(
        cv2.cvtColor(
            edges_roi,
            cv2.COLOR_GRAY2BGR
        ),
        "CANNY + ROI"
    )

    outputs["05_reinforced_edges.png"] = label(
        cv2.cvtColor(
            hough_input,
            cv2.COLOR_GRAY2BGR
        ),
        "HOUGH INPUT | CANNY + ROI + HSV"
    )

    outputs["06_hough.png"] = hough_overlay

    outputs["07_histogram.png"] = histogram_img

    left_win, right_win, both_win, counts = draw_windows(
        hough_input,
        fitter,
        left_base,
        right_base
    )

    outputs["08_left_windows.png"] = left_win
    outputs["09_right_windows.png"] = right_win
    outputs["10_windows_both.png"] = both_win

    lane_pixels_overlay = frame.copy()

    for x, y in zip(left_x, left_y):
        cv2.circle(
            lane_pixels_overlay,
            (int(x), int(y)),
            1,
            (0, 255, 255),
            -1
        )

    for x, y in zip(right_x, right_y):
        cv2.circle(
            lane_pixels_overlay,
            (int(x), int(y)),
            1,
            (255, 255, 0),
            -1
        )

    outputs["11_selected_lane_pixels.png"] = label(
        lane_pixels_overlay,
        f"SELECTED PIXELS | L={len(left_x)} R={len(right_x)}"
    )

    fit_overlay = draw_fit(
        frame,
        left_coeffs,
        right_coeffs,
        max(0, horizon_y),
        int(h * 0.95)
    )

    # Mark temporary horizon.
    cv2.line(
        fit_overlay,
        (0, horizon_y),
        (w - 1, horizon_y),
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    outputs["12_linear_fit.png"] = fit_overlay

    # ------------------------------------------------------------
    # Also save a direct comparison: original + linear fit
    # ------------------------------------------------------------
    comparison = frame.copy()

    if left_coeffs is not None:
        ys = np.linspace(
            horizon_y,
            int(h * 0.95),
            150
        )
        xs = np.polyval(left_coeffs, ys)
        pts = np.column_stack(
            (xs, ys)
        ).astype(np.int32)
        pts = pts[
            (pts[:, 0] >= 0) & (pts[:, 0] < w) &
            (pts[:, 1] >= 0) & (pts[:, 1] < h)
        ]
        if len(pts) >= 2:
            cv2.polylines(
                comparison,
                [pts],
                False,
                (0, 255, 0),
                5,
                cv2.LINE_AA
            )

    if right_coeffs is not None:
        ys = np.linspace(
            horizon_y,
            int(h * 0.95),
            150
        )
        xs = np.polyval(right_coeffs, ys)
        pts = np.column_stack(
            (xs, ys)
        ).astype(np.int32)
        pts = pts[
            (pts[:, 0] >= 0) & (pts[:, 0] < w) &
            (pts[:, 1] >= 0) & (pts[:, 1] < h)
        ]
        if len(pts) >= 2:
            cv2.polylines(
                comparison,
                [pts],
                False,
                (0, 0, 255),
                5,
                cv2.LINE_AA
            )

    outputs["13_final_linear_pipeline.png"] = label(
        comparison,
        "FINAL TEMP TEST | LINEAR FIT | HORIZON LOWERED"
    )

    for name, img in outputs.items():
        cv2.imwrite(
            os.path.join(args.outdir, name),
            img
        )

    # ------------------------------------------------------------
    # NUMERIC SUMMARY
    # ------------------------------------------------------------
    print("=" * 78)
    print("TEMPORARY LINEAR END-TO-END LANE PIPELINE")
    print("=" * 78)
    print(f"Input                 : {args.input}")
    print(f"Frame                 : {frame_idx}")
    print(f"Time                  : {timestamp:.2f} s")
    print(f"Resolution            : {w} x {h}")
    print(f"FPS                   : {fps:.2f}")
    print()
    print("HORIZON")
    print(f"Detected horizon      : {detected_horizon}")
    print(f"Temporary test horizon: {horizon_y}")
    print()
    print("HSV")
    print(f"White ROI coverage    : {100 * (white_roi > 0).sum() / (h * w):.2f}%")
    print(f"HSV fallback          : {hsv_fallback}")
    print()
    print("HOUGH")
    print(f"Left segments          : {len(left_segments)}")
    print(f"Right segments         : {len(right_segments)}")
    print(f"Discarded              : {len(discarded)}")
    print()
    print("SLIDING WINDOW CONFIG")
    print(f"n_windows              : {fitter.cfg.n_windows}")
    print(f"window_width_frac      : {fitter.cfg.window_width_frac}")
    print(f"min_pixels_to_recenter : {fitter.cfg.min_pixels_to_recenter}")
    print(f"min_pixels_total       : {fitter.cfg.min_pixels_total}")
    print(f"max_fit_rms_px         : {fitter.cfg.max_fit_rms_px}")
    print(f"max_recenter_jump_frac : {fitter.cfg.max_recenter_jump_frac}")
    print()
    print("BASES")
    print(f"Left base              : {left_base}")
    print(f"Right base             : {right_base}")
    print()
    print("WINDOW PIXEL COUNTS (bottom -> top)")
    print(f"Left                   : {counts['LEFT']}")
    print(f"Right                  : {counts['RIGHT']}")
    print()
    print("LINEAR FIT")
    print(f"Left pixels            : {len(left_x)}")
    print(f"Left RMS               : {left_rms:.2f}")
    print(f"Left RMS OK            : {left_rms_ok}")
    print(f"Left direction OK      : {left_direction_ok}")
    print(f"Left coeffs            : {left_coeffs}")
    print()
    print(f"Right pixels           : {len(right_x)}")
    print(f"Right RMS              : {right_rms:.2f}")
    print(f"Right RMS OK           : {right_rms_ok}")
    print(f"Right direction OK     : {right_direction_ok}")
    print(f"Right coeffs           : {right_coeffs}")
    print()
    print("OUTPUTS")
    print(f"Saved to               : {args.outdir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
