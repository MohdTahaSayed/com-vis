"""
Lane Analytics - Visualize individual lane-detection stages.

Stages shown:

1. Original frame
2. Artifact mask
3. Horizon detection
4. Raw Canny edges
5. ROI applied to Canny
6. HSV white/yellow mask
7. Final reinforced edges
8. Hough line classification
9. Polynomial lane fit

Usage:

    python scripts\inspect_lane_stages.py ^
        --input data\VBOX0011_Trim.mp4 ^
        --time 902

Outputs:

    outputs/lane_inspection_original.png
    outputs/lane_inspection_artifact.png
    outputs/lane_inspection_horizon.png
    outputs/lane_inspection_canny_raw.png
    outputs/lane_inspection_roi.png
    outputs/lane_inspection_hsv.png
    outputs/lane_inspection_final_edges.png
    outputs/lane_inspection_hough.png
    outputs/lane_inspection_fit.png
    outputs/lane_inspection_all.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml


# =====================================================================
# Project root
# =====================================================================

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


# =====================================================================
# Project modules
# =====================================================================

from src.artifact_mask import (
    ArtifactMask,
    ArtifactMaskConfig,
)

from src.horizon import (
    HorizonDetector,
    HorizonConfig,
)

from src.lane_color import (
    LaneColor,
    LaneColorConfig,
)

from src.lane_roi import (
    LaneROI,
    RoiConfig,
)

from src.lane_edges import (
    LaneEdges,
    CannyConfig,
)

from src.lane_hough import (
    LaneHough,
    HoughConfig,
)

from src.lane_fit import (
    LaneFitter,
    SlidingWindowConfig,
)

from src.lane_validation import (
    LaneValidation,
    ValidationConfig,
)

from src.io_video import VideoReader


# =====================================================================
# Configuration
# =====================================================================

DEFAULT_CONFIG = "config/default.yaml"


# =====================================================================
# Load YAML
# =====================================================================

def load_config(path: str):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f) or {}


# =====================================================================
# Label helper
# =====================================================================

def label(
    image: np.ndarray,
    text: str,
    color=(0, 255, 255),
):
    """
    Add readable text to an image.
    """

    cv2.putText(
        image,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    return image


# =====================================================================
# Convert grayscale to BGR
# =====================================================================

def gray_to_bgr(image):

    if len(image.shape) == 2:

        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    return image.copy()


# =====================================================================
# Resize for combined inspection
# =====================================================================

def resize_for_display(
    image,
    width=360,
):

    h, w = image.shape[:2]

    scale = width / float(w)

    height = int(
        h * scale
    )

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


# =====================================================================
# Create colored edge visualization
# =====================================================================

def edge_overlay(
    frame,
    edges,
):

    out = (
        frame.astype(np.float32)
        * 0.45
    ).astype(np.uint8)

    out[
        edges > 0
    ] = (
        0,
        255,
        0
    )

    return out


# =====================================================================
# Draw Hough lines
# =====================================================================

def draw_hough(
    frame,
    left_segments,
    right_segments,
    discard_segments,
):

    out = frame.copy()

    # ---------------------------------------------------------------
    # Left lane segments
    # ---------------------------------------------------------------

    for seg in left_segments:

        x1 = seg.x1
        y1 = seg.y1
        x2 = seg.x2
        y2 = seg.y2

        cv2.line(
            out,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

    # ---------------------------------------------------------------
    # Right lane segments
    # ---------------------------------------------------------------

    for seg in right_segments:

        x1 = seg.x1
        y1 = seg.y1
        x2 = seg.x2
        y2 = seg.y2

        cv2.line(
            out,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    # ---------------------------------------------------------------
    # Discarded segments
    # ---------------------------------------------------------------

    for seg in discard_segments:

        x1 = seg.x1
        y1 = seg.y1
        x2 = seg.x2
        y2 = seg.y2

        cv2.line(
            out,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            1,
        )

    return out
    out = frame.copy()

    # ---------------------------------------------------------------
    # Left lane segments
    # ---------------------------------------------------------------

    for seg in left_segments:

        x1, y1, x2, y2 = seg

        cv2.line(
            out,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

    # ---------------------------------------------------------------
    # Right lane segments
    # ---------------------------------------------------------------

    for seg in right_segments:

        x1, y1, x2, y2 = seg

        cv2.line(
            out,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    # ---------------------------------------------------------------
    # Discarded segments
    # ---------------------------------------------------------------

    for seg in discard_segments:

        x1, y1, x2, y2 = seg

        cv2.line(
            out,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            1,
        )

    return out


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect individual lane detection stages."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=902.0,
        help="Frame time in seconds",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Exact frame number",
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML configuration",
    )

    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------------

    cfg = load_config(
        args.config
    )

    # ---------------------------------------------------------------
    # Build project modules
    # ---------------------------------------------------------------

    artifact = ArtifactMask(
        ArtifactMaskConfig.from_dict(
            cfg.get(
                "artifact_mask",
                {}
            )
        )
    )

    horizon = HorizonDetector(
        HorizonConfig.from_dict(
            cfg.get(
                "horizon",
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

    roi = LaneROI(
        RoiConfig.from_dict(
            cfg.get(
                "roi",
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

    # ---------------------------------------------------------------
    # Read video frame
    # ---------------------------------------------------------------

    video = VideoReader(
        args.input
    )

    fps = video.info.fps

    if args.frame is not None:

        frame_idx = int(
            args.frame
        )

    else:

        frame_idx = int(
            round(
                args.time * fps
            )
        )

    frame = video.read_frame(
        frame_idx
    )

    video.release()

    if frame is None:

        raise RuntimeError(
            f"Could not read frame {frame_idx}"
        )

    timestamp = (
        frame_idx / fps
    )

    # =================================================================
    # STAGE 0 - ORIGINAL
    # =================================================================

    original = frame.copy()

    original_display = label(
        original.copy(),
        (
            f"ORIGINAL | "
            f"frame={frame_idx} | "
            f"t={timestamp:.2f}s"
        ),
        (255, 255, 255),
    )

    # =================================================================
    # STAGE 0 - ARTIFACT MASK
    # =================================================================

    masked = artifact.apply(
        frame
    )

    artifact_display = artifact.debug_render(
        frame.copy(),
        fill=True
    )

    artifact_display = label(
        artifact_display,
        "ARTIFACT MASK",
        (0, 0, 255),
    )

    # =================================================================
    # STAGE 1 - HORIZON
    # =================================================================

    horizon_y = horizon.detect(
        masked
    )

    horizon_display = horizon.debug_render(
        masked.copy(),
        horizon_y
    )

    horizon_display = label(
        horizon_display,
        f"HORIZON | y={horizon_y}",
        (0, 255, 255),
    )

    # =================================================================
    # STAGE 2 - RAW CANNY
    # =================================================================

    # Use the actual LaneEdges implementation.
    edges_roi, edges_raw, hsv_hits = (
        edges_module.compute(
            masked,
            top_y_override=horizon_y
        )
    )

    canny_raw_display = gray_to_bgr(
        edges_raw
    )

    canny_raw_display = label(
        canny_raw_display,
        (
            f"CANNY RAW | "
            f"thresholds="
            f"{cfg['canny']['low_threshold']}/"
            f"{cfg['canny']['high_threshold']}"
        ),
        (255, 255, 255),
    )

    # =================================================================
    # STAGE 3 - ROI APPLIED
    # =================================================================

    roi_edges_display = gray_to_bgr(
        edges_module.roi.apply(
            edges_raw,
            top_y_override=horizon_y
        )
    )

    roi_edges_display = label(
        roi_edges_display,
        (
            f"ROI EDGES | "
            f"pixels="
            f"{int((roi_edges_display[:, :, 0] > 0).sum())}"
        ),
        (255, 255, 255),
    )

    # =================================================================
    # STAGE 4 - HSV MASK
    # =================================================================

    white_mask, yellow_mask, union_mask = (
        lane_color.masks(
            masked
        )
    )

    # Yellow is currently disabled in your YAML,
    # so the union is effectively the white mask.
    hsv_display = np.zeros_like(
        frame
    )

    hsv_display[
        white_mask > 0
    ] = (
        255,
        255,
        255
    )

    hsv_display[
        yellow_mask > 0
    ] = (
        0,
        0,
        255
    )

    hsv_display = label(
        hsv_display,
        (
            f"HSV WHITE/YELLOW | "
            f"pixels="
            f"{int((union_mask > 0).sum())}"
        ),
        (255, 255, 255),
    )

    # =================================================================
    # STAGE 5 - FINAL REINFORCED EDGES
    # =================================================================

    final_edges_display = edge_overlay(
        masked,
        edges_roi
    )

    final_edges_display = label(
        final_edges_display,
        (
            f"FINAL EDGES | "
            f"pixels="
            f"{int((edges_roi > 0).sum())}"
        ),
        (0, 255, 0),
    )

    # =================================================================
    # STAGE 6 - HOUGH
    # =================================================================

    left_segments, right_segments, discard_segments = (
        hough.classify(
            edges_roi
        )
    )

    hough_display = draw_hough(
        masked,
        left_segments,
        right_segments,
        discard_segments,
    )

    hough_display = label(
        hough_display,
        (
            f"HOUGH | "
            f"L={len(left_segments)} "
            f"R={len(right_segments)} "
            f"discard={len(discard_segments)}"
        ),
        (255, 255, 255),
    )

    # =================================================================
    # STAGE 7 - POLYNOMIAL FIT
    # =================================================================

    left, right = fitter.fit(
        edges_roi
    )

    fit_display = masked.copy()

    # ---------------------------------------------------------------
    # Draw polynomial curves
    # ---------------------------------------------------------------

    def draw_fit(
        result,
        color,
    ):

        if (
            result is None
            or result.coeffs is None
        ):

            return

        a, b, c = result.coeffs

        h, w = fit_display.shape[:2]

        ys = np.linspace(
            int(h * 0.45),
            h - 1,
            200
        )

        xs = (
            a * ys * ys
            + b * ys
            + c
        )

        points = []

        for x, y in zip(
            xs,
            ys
        ):

            if (
                0 <= x < w
                and 0 <= y < h
            ):

                points.append(
                    [
                        int(x),
                        int(y)
                    ]
                )

        if len(points) >= 2:

            pts = np.asarray(
                points,
                dtype=np.int32
            )

            cv2.polylines(
                fit_display,
                [pts],
                False,
                color,
                4,
            )

    draw_fit(
        left,
        (0, 255, 255)
    )

    draw_fit(
        right,
        (0, 255, 0)
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    h, w = frame.shape[:2]

    y_range = (
        int(h * 0.55),
        int(h * 0.95)
    )

    validation = validator.validate(
        left.coeffs,
        right.coeffs,
        y_range,
        w,
    )

    fit_display = label(
        fit_display,
        (
            f"POLY FIT | "
            f"Lpx={left.n_pixels} "
            f"Rpx={right.n_pixels} | "
            f"L={validation.left_ok} "
            f"R={validation.right_ok} "
            f"PAIR={validation.pair_ok}"
        ),
        (255, 255, 255),
    )

    # =================================================================
    # SAVE INDIVIDUAL IMAGES
    # =================================================================

    outputs = {

        "lane_inspection_original.png":
            original_display,

        "lane_inspection_artifact.png":
            artifact_display,

        "lane_inspection_horizon.png":
            horizon_display,

        "lane_inspection_canny_raw.png":
            canny_raw_display,

        "lane_inspection_roi.png":
            roi_edges_display,

        "lane_inspection_hsv.png":
            hsv_display,

        "lane_inspection_final_edges.png":
            final_edges_display,

        "lane_inspection_hough.png":
            hough_display,

        "lane_inspection_fit.png":
            fit_display,
    }

    for filename, image in outputs.items():

        path = os.path.join(
            args.outdir,
            filename
        )

        cv2.imwrite(
            path,
            image
        )

    # =================================================================
    # COMBINED IMAGE
    # =================================================================

    # Resize every image to 360 px wide.
    display_images = []

    for title, image in [

        ("1 ORIGINAL", original_display),
        ("2 ARTIFACT", artifact_display),
        ("3 HORIZON", horizon_display),
        ("4 CANNY RAW", canny_raw_display),
        ("5 ROI EDGES", roi_edges_display),
        ("6 HSV", hsv_display),
        ("7 FINAL EDGES", final_edges_display),
        ("8 HOUGH", hough_display),
        ("9 POLY FIT", fit_display),

    ]:

        img = resize_for_display(
            image,
            360
        )

        # Add title bar.
        cv2.rectangle(
            img,
            (0, 0),
            (img.shape[1], 38),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            img,
            title,
            (8, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        display_images.append(
            img
        )

    # ---------------------------------------------------------------
    # Arrange as 3 x 3 grid
    # ---------------------------------------------------------------

    rows = []

    for i in range(
        0,
        len(display_images),
        3
    ):

        row = display_images[
            i:i + 3
        ]

        while len(row) < 3:

            row.append(
                np.zeros_like(
                    display_images[0]
                )
            )

        rows.append(
            np.hstack(row)
        )

    combined = np.vstack(
        rows
    )

    combined_path = os.path.join(
        args.outdir,
        "lane_inspection_all.png"
    )

    cv2.imwrite(
        combined_path,
        combined
    )

    # =================================================================
    # TERMINAL SUMMARY
    # =================================================================

    print()
    print("=" * 70)
    print("LANE PIPELINE STAGE INSPECTION")
    print("=" * 70)

    print(
        f"[info] frame        : {frame_idx}"
    )

    print(
        f"[info] timestamp    : {timestamp:.2f}s"
    )

    print(
        f"[info] resolution   : "
        f"{w}x{h}"
    )

    print(
        f"[info] horizon      : "
        f"{horizon_y}"
    )

    print()

    print(
        f"[info] Canny raw    : "
        f"{int((edges_raw > 0).sum())} px"
    )

    print(
        f"[info] ROI edges    : "
        f"{int((edges_roi > 0).sum())} px"
    )

    print(
        f"[info] HSV hits     : "
        f"{int((union_mask > 0).sum())} px"
    )

    print(
        f"[info] Hough left   : "
        f"{len(left_segments)}"
    )

    print(
        f"[info] Hough right  : "
        f"{len(right_segments)}"
    )

    print(
        f"[info] Hough discard: "
        f"{len(discard_segments)}"
    )

    print()

    print(
        f"[info] Polynomial L : "
        f"{left.n_pixels} px "
        f"confidence={left.confidence:.2f}"
    )

    print(
        f"[info] Polynomial R : "
        f"{right.n_pixels} px "
        f"confidence={right.confidence:.2f}"
    )

    print()

    print(
        f"[info] Validation   : "
        f"L={validation.left_ok} "
        f"R={validation.right_ok} "
        f"PAIR={validation.pair_ok}"
    )

    print()

    print(
        f"[ok] combined image:"
    )

    print(
        f"     {combined_path}"
    )

    print("=" * 70)
    print()


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    main()