from __future__ import annotations

import argparse
import csv
import os
import sys
import time

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

# ------------------------------------------------------------
# IMPORTANT:
# This diagnostic intentionally uses the SAME permanent
# EgoPosition implementation and the SAME upstream stage
# sequence/logic used by the original Stage 2 pipeline.
#
# The deleted artifact-mask stage is NOT used.
# ------------------------------------------------------------
from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.lane_state import LaneState, StateConfig
from src.ego_position import EgoPosition, EgoConfig
from src.io_video import VideoReader


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp_s",
            "frame",
            "y_eval",
            "lane_center_x",
            "offset_px",
            "lane_width_px",
            "offset_normalized",
            "left_status",
            "right_status",
            "left_confidence",
            "right_confidence",
            "status",
            "reason",
        ])

        writer.writerows(rows)


def save_plot(path, rows):
    """
    Diagnostic-only visualization.
    It does NOT participate in the ego-position calculation.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping plot.")
        return

    ok_rows = [
        r for r in rows
        if r[11] == "OK" and r[6] != ""
    ]

    if not ok_rows:
        print("[warn] no OK samples available for plot.")
        return

    t = [float(r[0]) for r in ok_rows]
    off = [float(r[4]) for r in ok_rows]
    norm = [float(r[6]) for r in ok_rows]

    fig = plt.figure(figsize=(11, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, off)
    ax1.axhline(0.0, linewidth=1)
    ax1.set_ylabel("Offset (px)")
    ax1.set_title("Ego Position Diagnostic — Permanent Logic")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, norm)
    ax2.axhline(0.0, linewidth=1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Normalized offset")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Temporary Stage 2 ego-position verification using "
            "the exact permanent EgoPosition logic."
        )
    )

    parser.add_argument("--input", required=True)

    parser.add_argument(
        "--config",
        default="config/default.yaml",
    )

    parser.add_argument(
        "--outdir",
        default="outputs/ego_position_check",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=3456,
        help=(
            "First frame to REPORT. Processing still starts at frame 0 "
            "so LaneState keeps the same temporal history as the pipeline."
        ),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=40,
        help="Number of consecutive frames to report.",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_yaml(args.config)

    # ========================================================
    # INITIALIZATION
    # Same module construction as the permanent Stage 2 flow.
    # ========================================================

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

    edges_mod = LaneEdges(
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

    state = LaneState(
        StateConfig.from_dict(
            cfg.get("lane_state", {})
        )
    )

    # EXACT permanent EgoPosition class.
    ego = EgoPosition(
        EgoConfig.from_dict(
            cfg.get("ego_position", {})
        )
    )

    MIN_CONF = float(
        cfg.get("lane_state", {}).get(
            "min_confidence",
            0.15,
        )
    )

    # ========================================================
    # VIDEO
    # ========================================================

    vr = VideoReader(args.input)

    fps = vr.info.fps
    width = int(vr.info.width)
    height = int(vr.info.height)

    # Permanent pipeline samples ego position at 1 Hz.
    sample_step = int(round(fps))

    report_start = args.start_frame
    report_end = report_start + args.num_frames - 1

    print("=" * 78)
    print("TEMPORARY EGO-POSITION VERIFICATION")
    print("=" * 78)
    print()
    print("This script does NOT modify permanent project files.")
    print()
    print("USING EXACT PERMANENT STAGE-2 LOGIC:")
    print("  EgoPosition.compute()")
    print("  y_eval = lane_eval_y_frac * frame_h")
    print("  ego_x  = ego_x_frac * frame_w")
    print("  lane_center = (x_left + x_right) / 2")
    print("  offset_px = ego_x - lane_center")
    print("  offset_normalized = offset_px / lane_width_px")
    print("  sampling = every round(FPS) frames")
    print()
    print("UPSTREAM LOGIC:")
    print("  Horizon -> Canny/ROI/HSV -> LaneFitter")
    print("  -> Validation -> LaneState -> EgoPosition")
    print()
    print(f"Input          : {args.input}")
    print(f"Resolution     : {width} x {height}")
    print(f"FPS            : {fps:.2f}")
    print(f"Sample step    : {sample_step} frames")
    print(f"Report frames  : {report_start} - {report_end}")
    print(f"Warm-up        : frame 0 -> {report_start - 1}")
    print()

    ego_cfg = cfg.get("ego_position", {})

    print("EGO CONFIG")
    print(f"  ego_x_frac       : {ego_cfg.get('ego_x_frac', 0.50)}")
    print(
        f"  lane_eval_y_frac : "
        f"{ego_cfg.get('lane_eval_y_frac', 0.85)}"
    )
    print(f"  MIN_CONF         : {MIN_CONF}")
    print()

    # ========================================================
    # PROCESSING
    #
    # This deliberately follows the permanent Stage 2 sequence:
    #
    # 1. Horizon
    # 2. Lane edges
    # 3. Lane fitting
    # 4. Validation
    # 5. Lane state
    # 6. Ego position at 1 Hz
    #
    # No temporary previous_left/right coefficients are added.
    # LaneFitter is called exactly as the Stage 2 code calls it.
    # ========================================================

    report_rows = []

    total_processed = 0
    total_report_samples = 0
    report_ok = 0
    report_miss = 0

    t0 = time.time()

    for idx, ts, frame in vr.iter_frames(step=1):
        # ----------------------------------------------------
        # 1. HORIZON
        # ----------------------------------------------------
        horizon_y = horizon.detect(frame)

        # ----------------------------------------------------
        # 2. CANNY + ROI + HSV
        # ----------------------------------------------------
        edges_roi, _, _ = edges_mod.compute(
            frame,
            top_y_override=horizon_y,
        )

        # ----------------------------------------------------
        # 3. LANE FIT
        #
        # Exact Stage-2 call: no temporary previous-fit logic.
        # ----------------------------------------------------
        left, right = fitter.fit(edges_roi)

        h, w = frame.shape[:2]

        # ----------------------------------------------------
        # 4. VALIDATION
        #
        # Exact Stage-2 y-range.
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # 5. TEMPORAL STATE
        #
        # Exact permanent state update.
        # ----------------------------------------------------
        (
            l_coeffs,
            l_status,
            l_conf,
        ), (
            r_coeffs,
            r_status,
            r_conf,
        ) = state.update(
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

        # ----------------------------------------------------
        # 6. EGO POSITION @ 1 Hz
        # ----------------------------------------------------
        if idx % sample_step == 0:
            measurement = ego.compute(
                l_coeffs,
                r_coeffs,
                w,
                h,
            )

            l_usable = (
                l_coeffs is not None
                and l_status in ("OK", "HOLD")
            )

            r_usable = (
                r_coeffs is not None
                and r_status in ("OK", "HOLD")
            )

            both_usable = (
                l_usable
                and r_usable
            )

            if both_usable:
                min_conf = float(
                    min(
                        l_conf,
                        r_conf,
                    )
                )
            else:
                min_conf = 0.0

            # EXACT permanent validity logic.
            if (
                both_usable
                and measurement.valid
                and min_conf >= MIN_CONF
            ):
                offset_px = measurement.offset_px
                lane_width_px = measurement.lane_width_px

                if (
                    lane_width_px is not None
                    and lane_width_px != 0
                ):
                    # IMPORTANT:
                    # This is the permanent project's definition.
                    # Do NOT divide by half the lane width.
                    offset_normalized = (
                        offset_px / lane_width_px
                    )
                else:
                    offset_normalized = None

                status = "OK"
                reason = measurement.reason

                report_ok += 1

            else:
                offset_px = None
                lane_width_px = None
                offset_normalized = None

                status = "MISS"

                if not both_usable:
                    reason = "lane_state_not_usable"
                elif not measurement.valid:
                    reason = measurement.reason
                else:
                    reason = "confidence_below_threshold"

                report_miss += 1

            # Only save/report the requested frame interval.
            if report_start <= idx <= report_end:
                report_rows.append([
                    f"{ts:.2f}",
                    idx,
                    measurement.y_eval,
                    (
                        f"{measurement.lane_center_x:.2f}"
                        if measurement.lane_center_x is not None
                        else ""
                    ),
                    (
                        f"{offset_px:.2f}"
                        if offset_px is not None
                        else ""
                    ),
                    (
                        f"{lane_width_px:.1f}"
                        if lane_width_px is not None
                        else ""
                    ),
                    (
                        f"{offset_normalized:.3f}"
                        if offset_normalized is not None
                        else ""
                    ),
                    l_status,
                    r_status,
                    f"{l_conf:.2f}",
                    f"{r_conf:.2f}",
                    status,
                    reason,
                ])

                total_report_samples += 1

                print(
                    f"{idx:6d}  "
                    f"t={ts:7.2f}s  "
                    f"L={l_status:5s}  "
                    f"R={r_status:5s}  "
                    f"y={measurement.y_eval:3d}  "
                    f"center="
                    f"{measurement.lane_center_x:7.2f}  "
                    f"width="
                    f"{measurement.lane_width_px:7.1f}  "
                    f"offset="
                    f"{offset_px:8.2f}  "
                    f"norm="
                    f"{offset_normalized:7.3f}"
                    if status == "OK"
                    else
                    f"{idx:6d}  "
                    f"t={ts:7.2f}s  "
                    f"L={l_status:5s}  "
                    f"R={r_status:5s}  "
                    f"y={measurement.y_eval:3d}  "
                    f"status=MISS  "
                    f"reason={reason}"
                )

        total_processed += 1

        if idx >= report_end:
            break

    vr.release()

    # ========================================================
    # OUTPUTS
    # ========================================================

    csv_path = os.path.join(
        args.outdir,
        "ego_position_diagnostic.csv",
    )

    plot_path = os.path.join(
        args.outdir,
        "ego_position_diagnostic.png",
    )

    write_csv(
        csv_path,
        report_rows,
    )

    save_plot(
        plot_path,
        report_rows,
    )

    dt = time.time() - t0

    print()
    print("=" * 78)
    print("EGO-POSITION VERIFICATION COMPLETE")
    print("=" * 78)
    print(f"Frames processed (warm-up + report): {total_processed}")
    print(f"Reported 1-Hz samples               : {total_report_samples}")
    print(f"Reported OK samples                 : {report_ok}")
    print(f"Reported MISS samples               : {report_miss}")
    print(f"Processing time                     : {dt:.1f}s")
    print()
    print(f"CSV   : {csv_path}")
    print(f"Plot  : {plot_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()