from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.artifact_mask import ArtifactMask, ArtifactMaskConfig
from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_hough import LaneHough, HoughConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.io_video import VideoReader


# ============================================================
# CONFIG LOADING
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
    else:
        idx = args.frame

    frame = vr.read_frame(idx)

    # Calculate timestamp using actual FPS
    ts = idx / fps if fps > 0 else 0.0

    vr.release()

    if frame is None:
        raise RuntimeError(
            f"Could not read frame {idx} "
            f"(time={ts:.2f}s)"
        )

    return frame, idx, ts


# ============================================================
# LABEL IMAGE
# ============================================================

def label(
    img,
    text,
    color=(0, 255, 255),
):

    cv2.putText(
        img,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )

    return img


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description="Stage 1 lane detection debugging"
    )

    ap.add_argument(
        "--input",
        required=True,
        help="Input video path",
    )

    ap.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame number",
    )

    ap.add_argument(
        "--time",
        type=float,
        default=None,
        help="Timestamp in seconds",
    )

    ap.add_argument(
        "--config",
        default="config/default.yaml",
        help="YAML configuration",
    )

    ap.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory",
    )

    args = ap.parse_args()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    cfg = load_config(args.config)

    # --------------------------------------------------------
    # Initialize modules
    # --------------------------------------------------------

    artifact_mask = ArtifactMask(
        ArtifactMaskConfig.from_dict(
            cfg.get("artifact_mask", {})
        )
    )

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

    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(
            cfg.get("sliding_window", {})
        )
    )

    validator = LaneValidation(
        ValidationConfig.from_dict(
            cfg.get("validation", {})
        )
    )

    # --------------------------------------------------------
    # Read selected frame
    # --------------------------------------------------------

    frame, idx, ts = get_frame(args)

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # STAGE 0
    # Artifact masking
    # --------------------------------------------------------

    f_masked = artifact_mask.apply(frame)

    # --------------------------------------------------------
    # HORIZON
    # --------------------------------------------------------

    horizon_y = horizon.detect(
        f_masked
    )

    # --------------------------------------------------------
    # CANNY + ROI + HSV
    # --------------------------------------------------------

    edges_roi, edges_raw, hsv_hits = (
        edges_module.compute(
            f_masked,
            top_y_override=horizon_y,
        )
    )

    # --------------------------------------------------------
    # HOUGH
    # --------------------------------------------------------

    left_segs, right_segs, discard_segs = (
        hough.classify(edges_roi)
    )

    # --------------------------------------------------------
    # LANE FIT
    #
    # IMPORTANT:
    # New fitter receives frame=f_masked.
    # --------------------------------------------------------

    left, right = fitter.fit(
        edges_roi,
        frame=f_masked,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    y_range = (
        int(h * 0.55),
        int(h * 0.95),
    )

    validation = validator.validate(
        left.coeffs,
        right.coeffs,
        y_range,
        w,
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
        left_segs,
        right_segs,
        discard_segs,
    )

    status = (
        f"L:{validation.left_ok} "
        f"R:{validation.right_ok} "
        f"pair:{validation.pair_ok}"
    )

    cv2.putText(
        p1,
        status,
        (8, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        p1,
        status,
        (8, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    label(
        p1,
        f"t={ts:.2f}s "
        f"L={len(left_segs)} "
        f"R={len(right_segs)} Hough",
    )

    # ========================================================
    # VISUALIZATION 2
    # Raw Canny
    # ========================================================

    p2 = cv2.cvtColor(
        edges_raw,
        cv2.COLOR_GRAY2BGR,
    )

    label(
        p2,
        "Canny raw",
    )

    # ========================================================
    # VISUALIZATION 3
    # Canny + ROI + HSV reinforcement
    # ========================================================

    p3 = cv2.cvtColor(
        edges_roi,
        cv2.COLOR_GRAY2BGR,
    )

    label(
        p3,
        f"Lane edges ROI "
        f"({int((edges_roi > 0).sum())} px)",
    )

    # ========================================================
    # VISUALIZATION 4
    # Polynomial fit
    # ========================================================

    p4 = frame.copy()

    p4 = fitter.debug_render(
        p4,
        left,
        right,
    )

    label(
        p4,
        "Lane-shaped sliding windows + polynomial fit",
    )

    # ========================================================
    # VISUALIZATION 5
    # Artifact mask
    # ========================================================

    p5 = artifact_mask.debug_render(
        frame,
        fill=True,
    )

    label(
        p5,
        "Artifact mask",
    )

    # ========================================================
    # VISUALIZATION 6
    # HSV mask
    # ========================================================

    if hsv_hits is not None:

        p6 = cv2.cvtColor(
            hsv_hits,
            cv2.COLOR_GRAY2BGR,
        )

        label(
            p6,
            "HSV white/yellow mask",
        )

    else:

        p6 = np.zeros_like(frame)

        label(
            p6,
            "HSV reinforcement unavailable",
        )

    # ========================================================
    # VISUALIZATION 7
    # Horizon
    # ========================================================

    p7 = frame.copy()

    p7 = horizon.debug_render(
        p7,
        horizon_y,
    )

    label(
        p7,
        "Detected horizon",
    )

    # ========================================================
    # VISUALIZATION 8
    # Selected lane pixels
    # ========================================================

    p8 = frame.copy()

    for x, y in left.pixels:

        cv2.circle(
            p8,
            (int(x), int(y)),
            1,
            (0, 255, 255),
            -1,
        )

    for x, y in right.pixels:

        cv2.circle(
            p8,
            (int(x), int(y)),
            1,
            (0, 255, 0),
            -1,
        )

    label(
        p8,
        f"Selected pixels "
        f"L={left.n_pixels} "
        f"R={right.n_pixels}",
    )

    # ========================================================
    # VISUALIZATION 9
    # Final lane result
    # ========================================================

    p9 = frame.copy()

    p9 = fitter.debug_render(
        p9,
        left,
        right,
        draw_pixels=False,
    )

    final_status = (
        f"L={'OK' if validation.left_ok else 'FAIL'} "
        f"R={'OK' if validation.right_ok else 'FAIL'} "
        f"PAIR={'OK' if validation.pair_ok else 'FAIL'}"
    )

    cv2.putText(
        p9,
        final_status,
        (8, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        p9,
        final_status,
        (8, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    label(
        p9,
        "FINAL LANE CURVES",
    )

    # ========================================================
    # COMBINE INTO 3 x 3 GRID
    # ========================================================

    panels = [
        p1, p2, p3,
        p4, p5, p6,
        p7, p8, p9,
    ]

    target_h = 300

    resized = []

    for panel in panels:

        ph, pw = panel.shape[:2]

        scale = target_h / float(ph)

        target_w = int(pw * scale)

        resized_panel = cv2.resize(
            panel,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

        resized.append(
            resized_panel
        )

    # Make all panels same size.
    target_w = max(
        p.shape[1]
        for p in resized
    )

    normalized = []

    for p in resized:

        if p.shape[1] < target_w:

            pad = target_w - p.shape[1]

            p = cv2.copyMakeBorder(
                p,
                0,
                0,
                0,
                pad,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        normalized.append(p)

    row1 = np.hstack(
        normalized[0:3]
    )

    row2 = np.hstack(
        normalized[3:6]
    )

    row3 = np.hstack(
        normalized[6:9]
    )

    combo = np.vstack(
        [row1, row2, row3]
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    out = os.path.join(
        args.outdir,
        "stage1_verify.png",
    )

    cv2.imwrite(
        out,
        combo,
    )

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

    cv2.imwrite(
        os.path.join(
            args.outdir,
            "stage1_hsv.png",
        ),
        hsv_hits
        if hsv_hits is not None
        else np.zeros_like(edges_raw),
    )

    # ========================================================
    # PRINT DEBUG INFORMATION
    # ========================================================

    print()
    print("=" * 60)
    print("STAGE 1 DEBUG")
    print("=" * 60)

    print(
        f"frame       = {idx}"
    )

    print(
        f"time        = {ts:.2f}s"
    )

    print(
        f"resolution  = {w}x{h}"
    )

    print(
        f"horizon_y   = {horizon_y}"
    )

    print()

    print(
        f"Hough:"
    )

    print(
        f"  left      = {len(left_segs)}"
    )

    print(
        f"  right     = {len(right_segs)}"
    )

    print(
        f"  discarded = {len(discard_segs)}"
    )

    print()

    print(
        f"Edges:"
    )

    print(
        f"  raw       = {int((edges_raw > 0).sum())}"
    )

    print(
        f"  ROI       = {int((edges_roi > 0).sum())}"
    )

    print()

    print(
        f"Histogram bases:"
    )

    print(
        f"  left base  = {left.x_base}"
    )

    print(
        f"  right base = {right.x_base}"
    )

    print()

    print(
        f"Selected lane pixels:"
    )

    print(
        f"  left      = {left.n_pixels}"
    )

    print(
        f"  right     = {right.n_pixels}"
    )

    print()

    print(
        f"Confidence:"
    )

    print(
        f"  left      = {left.confidence:.3f}"
    )

    print(
        f"  right     = {right.confidence:.3f}"
    )

    print()

    print(
        f"Validation:"
    )

    print(
        f"  left      = {validation.left_ok}"
    )

    print(
        f"  right     = {validation.right_ok}"
    )

    print(
        f"  pair      = {validation.pair_ok}"
    )

    print()

    print(
        f"[OK] wrote:"
    )

    print(
        f"     {out}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()