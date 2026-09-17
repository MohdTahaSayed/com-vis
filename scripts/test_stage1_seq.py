from __future__ import annotations

import argparse
import os
import sys
import time as _time

import cv2
import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.artifact_mask import ArtifactMask, ArtifactMaskConfig
from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.lane_state import LaneState, StateConfig
from src.io_video import VideoReader


def load_config(path):
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def draw_lane(
    frame,
    coeffs,
    y_range,
    color,
    thickness=3
):
    """Draw one polynomial lane curve on the frame."""

    if coeffs is None:
        return

    ys = np.linspace(
        y_range[0],
        y_range[1],
        80
    )

    xs = (
        coeffs[0] * ys * ys
        + coeffs[1] * ys
        + coeffs[2]
    )

    pts = np.stack(
        [xs, ys],
        axis=1
    ).astype(np.int32)

    # Keep points inside image
    h, w = frame.shape[:2]

    pts = pts[
        (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    ]

    if len(pts) >= 2:
        cv2.polylines(
            frame,
            [pts],
            False,
            color,
            thickness
        )


def put_text(frame, text, y, color=(255, 255, 255)):
    """Draw readable text with black outline."""

    cv2.putText(
        frame,
        text,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA
    )


def main():

    # ---------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------

    ap = argparse.ArgumentParser(
        description="Sequential Stage-1 lane detection + LaneState test"
    )

    ap.add_argument(
        "--input",
        required=True,
        help="Input video path"
    )

    ap.add_argument(
        "--time",
        type=float,
        default=900.0,
        help="Start time in seconds"
    )

    ap.add_argument(
        "--n",
        type=int,
        default=150,
        help="Number of consecutive frames"
    )

    ap.add_argument(
        "--config",
        default="config/default.yaml",
        help="YAML configuration"
    )

    ap.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory"
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_config(args.config)

    # ---------------------------------------------------------
    # Create pipeline modules
    # ---------------------------------------------------------

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

    lane_color = LaneColor(
        LaneColorConfig.from_dict(
            cfg.get("lane_color", {})
        )
    )

    roi = LaneROI(
        RoiConfig.from_dict(
            cfg.get("roi", {})
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

    lane_state = LaneState(
        StateConfig.from_dict(
            cfg.get("lane_state", {})
        )
    )

    # ---------------------------------------------------------
    # Open video
    # ---------------------------------------------------------

    video = VideoReader(args.input)

    fps = video.info.fps

    start_frame = int(
        round(args.time * fps)
    )

    end_frame = min(
        start_frame + args.n,
        video.info.frame_count
    )

    actual_n = end_frame - start_frame

    print()
    print("=" * 70)
    print("STAGE 1 SEQUENTIAL LANE STATE TEST")
    print("=" * 70)

    print(
        f"[info] video       : {args.input}"
    )

    print(
        f"[info] fps         : {fps:.2f}"
    )

    print(
        f"[info] start time  : {args.time:.2f}s"
    )

    print(
        f"[info] start frame : {start_frame}"
    )

    print(
        f"[info] end frame   : {end_frame}"
    )

    print(
        f"[info] frames      : {actual_n}"
    )

    print()

    # ---------------------------------------------------------
    # Output video
    # ---------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    output_video = os.path.join(
        args.outdir,
        "stage1_seq.mp4"
    )

    writer = None

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------

    counts = {
        "L_OK": 0,
        "L_HOLD": 0,
        "L_MISS": 0,

        "R_OK": 0,
        "R_HOLD": 0,
        "R_MISS": 0,
    }

    # Validation counters
    validation_counts = {
        "left_valid": 0,
        "right_valid": 0,
        "pair_valid": 0,
    }

    # ---------------------------------------------------------
    # Processing
    # ---------------------------------------------------------

    t0 = _time.time()

    processed = 0

    for idx, ts, frame in video.iter_frames(
        start=start_frame,
        end=end_frame,
        step=1
    ):

        # -----------------------------------------------------
        # 1. Artifact removal
        # -----------------------------------------------------

        frame_masked = artifact_mask.apply(
            frame
        )

        # -----------------------------------------------------
        # 2. Horizon detection
        # -----------------------------------------------------

        horizon_y = horizon.detect(
            frame_masked
        )

        # -----------------------------------------------------
        # 3. Canny + ROI + HSV reinforcement
        # -----------------------------------------------------

        edges_roi, edges_raw, hsv_hits = edges_module.compute(
            frame_masked,
            top_y_override=horizon_y
        )

        # -----------------------------------------------------
        # 4. Sliding-window lane fitting
        # -----------------------------------------------------

        left, right = fitter.fit(
            edges_roi
        )

        # -----------------------------------------------------
        # 5. Lane validation
        # -----------------------------------------------------

        h, w = frame.shape[:2]

        y_range = (
            int(h * 0.55),
            int(h * 0.95)
        )

        validation = validator.validate(
            left.coeffs,
            right.coeffs,
            y_range,
            w
        )

        if validation.left_ok:
            validation_counts["left_valid"] += 1

        if validation.right_ok:
            validation_counts["right_valid"] += 1

        if validation.pair_ok:
            validation_counts["pair_valid"] += 1

        # -----------------------------------------------------
        # 6. Temporal LaneState
        # -----------------------------------------------------

        (
            l_coeffs,
            l_status,
            l_conf
        ), (
            r_coeffs,
            r_status,
            r_conf
        ) = lane_state.update(

            left.coeffs
            if validation.left_ok
            else None,

            left.confidence
            if validation.left_ok
            else 0.0,

            right.coeffs
            if validation.right_ok
            else None,

            right.confidence
            if validation.right_ok
            else 0.0,
        )

        # -----------------------------------------------------
        # Count states
        # -----------------------------------------------------

        counts[f"L_{l_status}"] += 1
        counts[f"R_{r_status}"] += 1

        # -----------------------------------------------------
        # Terminal output
        # -----------------------------------------------------

        print(
            f"frame={idx:6d} "
            f"t={ts:7.2f}s  "
            f"L={l_status:5s} "
            f"R={r_status:5s}  "
            f"Lpx={left.n_pixels:4d} "
            f"Rpx={right.n_pixels:4d}  "
            f"Lconf={l_conf:.2f} "
            f"Rconf={r_conf:.2f}"
        )

        # -----------------------------------------------------
        # 7. Visualization
        # -----------------------------------------------------

        vis = frame.copy()

        # Draw raw/current detected lanes through LaneState output
        draw_lane(
            vis,
            l_coeffs,
            y_range,
            (0, 255, 255),
            3
        )

        draw_lane(
            vis,
            r_coeffs,
            y_range,
            (0, 255, 0),
            3
        )

        # -----------------------------------------------------
        # Status text
        # -----------------------------------------------------

        put_text(
            vis,
            f"Frame: {idx}   Time: {ts:.2f}s",
            24
        )

        put_text(
            vis,
            f"LEFT: {l_status}  conf={l_conf:.2f}",
            48,
            (0, 255, 255)
        )

        put_text(
            vis,
            f"RIGHT: {r_status}  conf={r_conf:.2f}",
            72,
            (0, 255, 0)
        )

        put_text(
            vis,
            f"Raw pixels: L={left.n_pixels} R={right.n_pixels}",
            96
        )

        if horizon_y is not None:

            put_text(
                vis,
                f"Horizon y={horizon_y}",
                120
            )

        # -----------------------------------------------------
        # Initialize video writer
        # -----------------------------------------------------

        if writer is None:

            writer = cv2.VideoWriter(
                output_video,
                fourcc,
                fps,
                (
                    vis.shape[1],
                    vis.shape[0]
                )
            )

        writer.write(vis)

        processed += 1

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    video.release()

    if writer is not None:
        writer.release()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    dt = _time.time() - t0

    processing_fps = (
        processed / dt
        if dt > 0
        else 0
    )

    print()
    print("=" * 70)
    print("SEQUENTIAL TEST COMPLETE")
    print("=" * 70)

    print(
        f"[seq] processed      : {processed} frames"
    )

    print(
        f"[seq] processing FPS : {processing_fps:.1f}"
    )

    print()

    print(
        "[seq] LEFT states:"
    )

    print(
        f"      OK   = {counts['L_OK']}"
    )

    print(
        f"      HOLD = {counts['L_HOLD']}"
    )

    print(
        f"      MISS = {counts['L_MISS']}"
    )

    print()

    print(
        "[seq] RIGHT states:"
    )

    print(
        f"      OK   = {counts['R_OK']}"
    )

    print(
        f"      HOLD = {counts['R_HOLD']}"
    )

    print(
        f"      MISS = {counts['R_MISS']}"
    )

    print()

    print(
        "[seq] validation:"
    )

    print(
        f"      left valid  = {validation_counts['left_valid']}"
    )

    print(
        f"      right valid = {validation_counts['right_valid']}"
    )

    print(
        f"      pair valid  = {validation_counts['pair_valid']}"
    )

    print()

    print(
        f"[seq] output video: {output_video}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()