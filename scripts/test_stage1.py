from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

# ------------------------------------------------------------
# Allow imports from project root
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_hough import LaneHough, HoughConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.io_video import VideoReader


# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = "config/default.yaml"


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================
# GET FRAME
# ============================================================

def get_frame(args):

    vr = VideoReader(args.input)

    fps = vr.info.fps

    if args.time is not None:
        idx = int(round(args.time * fps))

    elif args.frame is not None:
        idx = args.frame

    else:
        idx = 0

    frame = vr.read_frame(idx)

    vr.release()

    if frame is None:
        raise RuntimeError(
            f"Could not read frame {idx} from {args.input}"
        )

    timestamp = idx / fps

    return frame, idx, timestamp


# ============================================================
# TEXT LABEL
# ============================================================

def label(
    img,
    text,
    position=(8, 24),
    color=(0, 255, 255),
):

    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )

    return img


# ============================================================
# DRAW POLYNOMIAL
# ============================================================

def draw_polynomial(
    frame,
    coeffs,
    y_range,
    color,
    thickness=3,
):

    if coeffs is None:
        return frame

    y1, y2 = y_range

    ys = np.linspace(
        y1,
        y2,
        100,
    )

    a, b, c = coeffs

    xs = (
        a * ys * ys
        + b * ys
        + c
    )

    pts = np.stack(
        [xs, ys],
        axis=1,
    ).astype(np.int32)

    h, w = frame.shape[:2]

    valid = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    )

    pts = pts[valid]

    if len(pts) >= 2:

        cv2.polylines(
            frame,
            [pts],
            False,
            color,
            thickness,
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Stage 1 lane detection verification"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame number to inspect",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Timestamp in seconds to inspect",
    )

    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="YAML configuration path",
    )

    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    cfg = load_config(
        args.config
    )

    # --------------------------------------------------------
    # Initialize modules
    # --------------------------------------------------------

    horizon = HorizonDetector(
        HorizonConfig.from_dict(
            cfg.get(
                "horizon",
                {}
            )
        )
    )

    roi = LaneROI(
        RoiConfig.from_dict(
            cfg.get(
                "roi",
                {}
            )
        )
    )

    lane_color = LaneColor(
        LaneColorConfig.from_dict(
            cfg.get(
                "lane_color",
                {}
            )
        )
    )

    edges_module = LaneEdges(
        CannyConfig.from_dict(
            cfg.get(
                "canny",
                {}
            )
        ),
        roi,
        lane_color,
        reinforce_with_hsv=True,
    )

    hough = LaneHough(
        HoughConfig.from_dict(
            cfg.get(
                "hough",
                {}
            )
        )
    )

    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(
            cfg.get(
                "sliding_window",
                {}
            )
        )
    )

    validator = LaneValidation(
        ValidationConfig.from_dict(
            cfg.get(
                "validation",
                {}
            )
        )
    )

    # --------------------------------------------------------
    # Read selected frame
    # --------------------------------------------------------

    frame, frame_idx, timestamp = get_frame(
        args
    )

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Stage 1A: Horizon detection
    # --------------------------------------------------------

    horizon_y = horizon.detect(
        frame
    )

    # --------------------------------------------------------
    # Stage 1B: Canny + ROI + HSV reinforcement
    # --------------------------------------------------------

    edges_roi, edges_raw, hsv_hits = (
        edges_module.compute(
            frame,
            top_y_override=horizon_y,
        )
    )

    # --------------------------------------------------------
    # Stage 1C: Hough
    # --------------------------------------------------------

    left_segments, right_segments, discard_segments = (
        hough.classify(
            edges_roi
        )
    )

    # --------------------------------------------------------
    # Stage 1D: Sliding-window polynomial fitting
    #
    # The fitter accepts optional previous-frame polynomials.
    # For a single-frame test there is no previous frame.
    # --------------------------------------------------------

    left, right = fitter.fit(
        edges_roi,
        previous_left=None,
        previous_right=None,
    )

    # --------------------------------------------------------
    # Stage 1E: Validation
    #
    # Validate lane only below the detected horizon.
    # --------------------------------------------------------

    if horizon_y is not None:

        y_min = max(
            int(h * 0.62),
            int(horizon_y + 10),
        )

    else:

        y_min = int(h * 0.62)

    y_max = int(
        h * 0.95
    )

    y_range = (
        y_min,
        y_max,
    )

    validation = validator.validate(
        left.coeffs,
        right.coeffs,
        y_range,
        w,
    )

    # ========================================================
    # PRINT DIAGNOSTICS
    # ========================================================

    print()
    print("=" * 70)
    print("STAGE 1 LANE DETECTION TEST")
    print("=" * 70)

    print(
        f"Input       : {args.input}"
    )

    print(
        f"Frame       : {frame_idx}"
    )

    print(
        f"Timestamp   : {timestamp:.2f} s"
    )

    print(
        f"Resolution  : {w} x {h}"
    )

    print(
        f"Horizon     : {horizon_y}"
    )

    print(
        f"Validate y  : "
        f"{y_range[0]} -> {y_range[1]}"
    )

    print()
    print("CANNY / ROI")
    print("-" * 70)

    print(
        f"Raw Canny edges : "
        f"{int((edges_raw > 0).sum())} px"
    )

    print(
        f"ROI edges       : "
        f"{int((edges_roi > 0).sum())} px"
    )

    if hsv_hits is not None:

        print(
            f"HSV hits        : "
            f"{int((hsv_hits > 0).sum())} px"
        )

    print()
    print("HOUGH")
    print("-" * 70)

    print(
        f"Left segments   : "
        f"{len(left_segments)}"
    )

    print(
        f"Right segments  : "
        f"{len(right_segments)}"
    )

    print(
        f"Discarded       : "
        f"{len(discard_segments)}"
    )

    print()
    print("POLYNOMIAL FIT")
    print("-" * 70)

    print(
        f"LEFT  : "
        f"pixels={left.n_pixels} "
        f"confidence={left.confidence:.2f} "
        f"fit={'YES' if left.coeffs is not None else 'NO'}"
    )

    print(
        f"RIGHT : "
        f"pixels={right.n_pixels} "
        f"confidence={right.confidence:.2f} "
        f"fit={'YES' if right.coeffs is not None else 'NO'}"
    )

    if left.coeffs is not None:

        print(
            f"LEFT coeffs  : "
            f"{left.coeffs}"
        )

    if right.coeffs is not None:

        print(
            f"RIGHT coeffs : "
            f"{right.coeffs}"
        )

    print()
    print("VALIDATION")
    print("-" * 70)

    print(
        f"Left valid    : "
        f"{validation.left_ok}"
    )

    print(
        f"Right valid   : "
        f"{validation.right_ok}"
    )

    print(
        f"Pair valid    : "
        f"{validation.pair_ok}"
    )

    print(
        f"Left reason   : "
        f"{validation.reason_left}"
    )

    print(
        f"Right reason  : "
        f"{validation.reason_right}"
    )

    print(
        f"Pair reason   : "
        f"{validation.reason_pair}"
    )

    # ========================================================
    # VISUALIZATION 1
    # ROI + Hough
    # ========================================================

    p1 = frame.copy()

    p1 = roi.debug_render(
        p1,
        top_y_override=horizon_y,
    )

    p1 = hough.debug_render(
        p1,
        left_segments,
        right_segments,
        discard_segments,
    )

    if horizon_y is not None:

        cv2.line(
            p1,
            (0, horizon_y),
            (w - 1, horizon_y),
            (255, 255, 0),
            2,
        )

    label(
        p1,
        (
            f"ROI + Hough | "
            f"t={timestamp:.1f}s"
        ),
    )

    # ========================================================
    # VISUALIZATION 2
    # RAW CANNY
    # ========================================================

    p2 = cv2.cvtColor(
        edges_raw,
        cv2.COLOR_GRAY2BGR,
    )

    label(
        p2,
        "Canny raw - whole frame",
    )

    # ========================================================
    # VISUALIZATION 3
    # ROI EDGES
    # ========================================================

    p3 = cv2.cvtColor(
        edges_roi,
        cv2.COLOR_GRAY2BGR,
    )

    label(
        p3,
        (
            f"Final ROI edges - "
            f"{int((edges_roi > 0).sum())} px"
        ),
    )

    # ========================================================
    # VISUALIZATION 4
    # POLYNOMIAL FIT
    # ========================================================

    p4 = frame.copy()

    p4 = draw_polynomial(
        p4,
        left.coeffs,
        y_range,
        (0, 255, 255),
        3,
    )

    p4 = draw_polynomial(
        p4,
        right.coeffs,
        y_range,
        (0, 255, 0),
        3,
    )

    label(
        p4,
        (
            f"Polynomial fit | "
            f"L={left.n_pixels}px "
            f"R={right.n_pixels}px"
        ),
    )

    # Validation text

    cv2.putText(
        p4,
        (
            f"L={validation.left_ok} "
            f"R={validation.right_ok} "
            f"PAIR={validation.pair_ok}"
        ),
        (8, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        p4,
        (
            f"L={validation.left_ok} "
            f"R={validation.right_ok} "
            f"PAIR={validation.pair_ok}"
        ),
        (8, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # ========================================================
    # COMBINE
    # ========================================================

    combo = np.hstack(
        [p1, p2, p3, p4]
    )

    output_path = os.path.join(
        args.outdir,
        "stage1_verify.png",
    )

    cv2.imwrite(
        output_path,
        combo,
    )

    # Save individual images too

    cv2.imwrite(
        os.path.join(
            args.outdir,
            "stage1_edges_raw.png",
        ),
        edges_raw,
    )

    cv2.imwrite(
        os.path.join(
            args.outdir,
            "stage1_edges_roi.png",
        ),
        edges_roi,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"[ok] wrote: {output_path}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()