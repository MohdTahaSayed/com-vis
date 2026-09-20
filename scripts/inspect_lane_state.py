"""
LaneState stage inspector — STAGES 2-9.

Runs the front-end pipeline and visualizes the temporal state machine.

Stages run:
    Stage 2 — HorizonDetector.detect()    → horizon_y
    Stage 3 — LaneColor.masks()           → white, yellow, union
    Stage 4 — LaneROI.apply()             → trapezoid clip
    Stage 5 — LaneEdges.compute()         → edges_roi
    Stage 7 — LaneFitter.fit()            → left, right LaneFitResult
    Stage 8 — LaneValidation.validate()   → validation flags
    Stage 9 — LaneState.update()          → smoothed coeffs + status

Visualizes per frame:
    - raw fit (from Stage 7)
    - validation verdict (from Stage 8)
    - state machine status OK / HOLD / MISS (from Stage 9)
    - smoothed output line

Produces:
    - a per-frame PNG (5 panels)
    - an annotated MP4 (whole sequence)
    - a terminal trace of state transitions

All thresholds are read from config/default.yaml on every run.
Re-run after editing YAML to see updated behavior.

Usage:
    python scripts/inspect_lane_state.py --input data/VBOX0011_Trim.mp4 --time 100 --n 150
    python scripts/inspect_lane_state.py --input data/VBOX0011_Trim.mp4 --time 900 --n 300

Outputs:
    outputs/lane_state/
        frame_00001.png         (per-frame panel)
        state_trace.mp4         (annotated video)
        state_log.txt           (per-frame text log)
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.lane_state import LaneState, StateConfig
from src.io_video import VideoReader


# =====================================================================
# Config loading
# =====================================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =====================================================================
# Text helpers
# =====================================================================

def put_label(img, text, color=(0, 255, 255), y=26):
    cv2.putText(
        img, text, (8, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        img, text, (8, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        color, 1, cv2.LINE_AA,
    )
    return img


def status_color(status):
    """Consistent color for state statuses."""
    if status == "OK":
        return (0, 255, 0)
    if status == "HOLD":
        return (0, 165, 255)   # orange
    if status == "MISS":
        return (0, 0, 255)
    return (255, 255, 255)


def draw_curve(frame, coeffs, y_range, color, thickness=3):
    """Draw a polynomial curve on a frame."""
    if coeffs is None:
        return frame
    h, w = frame.shape[:2]
    ys = np.linspace(y_range[0], y_range[1], 100)
    xs = np.polyval(coeffs, ys)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
    if len(pts) >= 2:
        cv2.polylines(frame, [pts], False, color, thickness)
    return frame


# =====================================================================
# Build per-frame panels
# =====================================================================

def build_panels(frame, edges_roi, left_raw, right_raw, validation,
                 l_coeffs, l_status, l_conf,
                 r_coeffs, r_status, r_conf,
                 y_range, state):

    h, w = frame.shape[:2]

    # ---------------------------------------------------------------
    # Panel 0 — raw fits from Stage 7 (with validation verdict)
    # ---------------------------------------------------------------

    p0 = frame.copy()

    def draw_raw(coeffs, color):
        if coeffs is None:
            return
        ys = np.linspace(y_range[0], y_range[1], 100)
        xs = np.polyval(coeffs, ys)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
        if len(pts) >= 2:
            cv2.polylines(p0, [pts], False, color, 2)

    draw_raw(left_raw.coeffs, (0, 255, 255))
    draw_raw(right_raw.coeffs, (0, 255, 0))

    l_raw_tag = f"L raw ok={validation.left_ok}"
    r_raw_tag = f"R raw ok={validation.right_ok}"

    put_label(p0, f"0) Stage 7+8 raw fit | {l_raw_tag} | {r_raw_tag}",
              (255, 255, 255))
    put_label(p0,
              f"   reason L={validation.reason_left} R={validation.reason_right}",
              (200, 200, 200), y=50)

    # ---------------------------------------------------------------
    # Panel 1 — smoothed output from Stage 9
    # ---------------------------------------------------------------

    p1 = frame.copy()

    if l_coeffs is not None:
        ys = np.linspace(y_range[0], y_range[1], 100)
        xs = np.polyval(l_coeffs, ys)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
        if len(pts) >= 2:
            cv2.polylines(p1, [pts], False, status_color(l_status), 4)

    if r_coeffs is not None:
        ys = np.linspace(y_range[0], y_range[1], 100)
        xs = np.polyval(r_coeffs, ys)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
        if len(pts) >= 2:
            cv2.polylines(p1, [pts], False, status_color(r_status), 4)

    put_label(
        p1,
        f"1) Stage 9 output | L={l_status} ({l_conf:.2f}) "
        f"R={r_status} ({r_conf:.2f})",
        (255, 255, 255),
    )

    # ---------------------------------------------------------------
    # Panel 2 — edges_roi
    # ---------------------------------------------------------------

    if edges_roi.ndim == 2:
        p2 = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)
    else:
        p2 = edges_roi.copy()
    put_label(p2, "2) edges_roi (input to fit)", (255, 255, 255))

    # ---------------------------------------------------------------
    # Panel 3 — state machine diagram
    # ---------------------------------------------------------------

    p3 = np.zeros((360, 720, 3), dtype=np.uint8)

    def draw_side(y_offset, name, status, conf, fail_count,
                  ok_count, has_smoothed):
        # Title
        cv2.putText(p3, f"{name}", (20, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Current status box
        box_x = 200
        box_w = 160
        box_h = 50
        cv2.rectangle(p3,
                      (box_x, y_offset + 10),
                      (box_x + box_w, y_offset + 10 + box_h),
                      status_color(status), 2)
        cv2.putText(p3, status,
                    (box_x + 30, y_offset + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    status_color(status), 2, cv2.LINE_AA)

        # Right of box — counters
        cv2.putText(p3, f"conf={conf:.2f}",
                    (box_x + box_w + 20, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(p3, f"fail_count={fail_count}",
                    (box_x + box_w + 20, y_offset + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(p3, f"ok_count={ok_count}",
                    (box_x + box_w + 20, y_offset + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Smoothing flag
        smooth_tag = "smoothed_coeffs: yes" if has_smoothed else "smoothed_coeffs: no"
        cv2.putText(p3, smooth_tag,
                    (box_x + box_w + 20, y_offset + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255) if has_smoothed else (128, 128, 128),
                    1, cv2.LINE_AA)

    draw_side(40, "LEFT",
              l_status, l_conf,
              state.left.fail_count,
              state.left.ok_count,
              state.left.smoothed_coeffs is not None)

    draw_side(200, "RIGHT",
              r_status, r_conf,
              state.right.fail_count,
              state.right.ok_count,
              state.right.smoothed_coeffs is not None)

    put_label(p3, "3) State machine", (255, 255, 255))

    # ---------------------------------------------------------------
    # Panel 4 — text diagnostic
    # ---------------------------------------------------------------

    p4 = np.zeros((360, 720, 3), dtype=np.uint8)

    def coeffs_str(c):
        if c is None:
            return "None"
        return f"[{c[0]:+.2f}, {c[1]:+.4f}, {c[2]:+.2f}]"

    lines = [
        f"left_raw.coeffs  : {coeffs_str(left_raw.coeffs)}",
        f"right_raw.coeffs : {coeffs_str(right_raw.coeffs)}",
        "",
        f"valid L={validation.left_ok}  R={validation.right_ok}  "
        f"PAIR={validation.pair_ok}",
        f"reason_pair      : {validation.reason_pair}",
        "",
        f"l_coeffs (used)  : {coeffs_str(l_coeffs)}",
        f"r_coeffs (used)  : {coeffs_str(r_coeffs)}",
        "",
        f"state.left.status  = {state.left.status}",
        f"state.right.status = {state.right.status}",
    ]

    for i, line in enumerate(lines):
        y = 30 + i * 22
        cv2.putText(p4, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    put_label(p4, "4) Diagnostics", (255, 255, 255))

    return {
        "0_raw_fit":   p0,
        "1_smoothed":  p1,
        "2_edges":     p2,
        "3_state":     p3,
        "4_diag":      p4,
    }


# =====================================================================
# Grid builder
# =====================================================================

def build_grid(panels):
    order = ["0_raw_fit", "1_smoothed", "2_edges", "3_state", "4_diag"]
    target_w = 500

    resized = []
    for key in order:
        p = panels[key]
        h, w = p.shape[:2]
        new_h = int(h * target_w / w)
        resized.append(cv2.resize(p, (target_w, new_h)))

    max_h = max(p.shape[0] for p in resized)
    padded = []
    for p in resized:
        if p.shape[0] < max_h:
            pad = max_h - p.shape[0]
            p = cv2.copyMakeBorder(
                p, 0, pad, 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
        padded.append(p)

    # 2x2 grid of the first 4 panels + bottom row for panel 4
    top = np.hstack(padded[0:2])
    mid = np.hstack(padded[2:4])
    bot = cv2.resize(padded[4], (top.shape[1], int(padded[4].shape[0] * top.shape[1] / padded[4].shape[1])))

    return np.vstack([top, mid, bot])


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inspect STAGES 2-9 including LaneState temporal machine."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--time", type=float, default=900.0)
    parser.add_argument("--n", type=int, default=150,
                        help="Number of consecutive frames to process")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--outdir", default="outputs/lane_state")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save every Nth frame as PNG")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing the annotated MP4")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------------------------------------------
    # Config reloaded every run
    # -----------------------------------------------------------------
    cfg = load_config(args.config)

    horizon = HorizonDetector(HorizonConfig.from_dict(cfg.get("horizon", {})))
    lane_color = LaneColor(LaneColorConfig.from_dict(cfg.get("lane_color", {})))
    roi = LaneROI(RoiConfig.from_dict(cfg.get("roi", {})))
    edges_module = LaneEdges(
        CannyConfig.from_dict(cfg.get("canny", {})),
        roi, lane_color,
        reinforce_with_hsv=True,
    )
    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(cfg.get("sliding_window", {}))
    )
    validator = LaneValidation(
        ValidationConfig.from_dict(cfg.get("validation", {}))
    )
    state = LaneState(
        StateConfig.from_dict(cfg.get("lane_state", {}))
    )

    # -----------------------------------------------------------------
    # Video setup
    # -----------------------------------------------------------------
    vr = VideoReader(args.input)
    fps = vr.info.fps

    start_frame = int(round(args.time * fps))
    end_frame = min(start_frame + args.n, vr.info.frame_count)

    print()
    print("=" * 74)
    print("LANE STATE STAGE INSPECTION — STAGES 2-9")
    print("=" * 74)
    print(f"Input       : {args.input}")
    print(f"Config      : {args.config}")
    print(f"Resolution  : {vr.info.width}x{vr.info.height} @ {fps:.2f} fps")
    print(f"Start       : t={args.time:.2f}s  frame={start_frame}")
    print(f"End         : frame={end_frame}  ({end_frame - start_frame} frames)")
    print()
    print("Loaded from YAML (lane_state):")
    print(f"  hold_frames      : {state.cfg.hold_frames}")
    print(f"  miss_frames      : {state.cfg.miss_frames}")
    print(f"  smoothing_alpha  : {state.cfg.smoothing_alpha}")
    print(f"  min_confidence   : {state.cfg.min_confidence}")
    print("=" * 74)
    print()

    # -----------------------------------------------------------------
    # Open video writer
    # -----------------------------------------------------------------
    writer = None
    if not args.no_video:
        vpath = os.path.join(args.outdir, "state_trace.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            vpath, fourcc, fps,
            (int(vr.info.width), int(vr.info.height)),
        )
        print(f"[info] writing video: {vpath}")

    # -----------------------------------------------------------------
    # Log file
    # -----------------------------------------------------------------
    log_path = os.path.join(args.outdir, "state_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("frame,timestamp,L_raw_ok,R_raw_ok,L_status,R_status,L_conf,R_conf,L_fail,R_fail\n")

    # -----------------------------------------------------------------
    # Process
    # -----------------------------------------------------------------
    t0 = _time.time()

    processed = 0
    saved_frames = 0

    # Counters
    counts = {
        "L_OK": 0, "L_HOLD": 0, "L_MISS": 0,
        "R_OK": 0, "R_HOLD": 0, "R_MISS": 0,
    }

    # Track transitions for reporting
    last_l = None
    last_r = None

    h, w = vr.info.height, vr.info.width
    y_range = (int(h * 0.55), int(h * 0.95))

    for idx, ts, frame in vr.iter_frames(start=start_frame, end=end_frame, step=1):

        # -------------------------------------------------------------
        # Stage 2
        # -------------------------------------------------------------
        horizon_y = horizon.detect(frame)

        # -------------------------------------------------------------
        # Stage 5
        # -------------------------------------------------------------
        edges_roi, _, _ = edges_module.compute(
            frame, top_y_override=horizon_y
        )

        # -------------------------------------------------------------
        # Stage 7
        # -------------------------------------------------------------
        left_raw, right_raw = fitter.fit(edges_roi)

        # -------------------------------------------------------------
        # Stage 8
        # -------------------------------------------------------------
        validation = validator.validate(
            left_raw.coeffs,
            right_raw.coeffs,
            y_range,
            w,
        )

        # -------------------------------------------------------------
        # Stage 9
        # -------------------------------------------------------------
        (l_coeffs, l_status, l_conf), (r_coeffs, r_status, r_conf) = \
            state.update(
                left_raw.coeffs if validation.left_ok else None,
                left_raw.confidence if validation.left_ok else 0.0,
                right_raw.coeffs if validation.right_ok else None,
                right_raw.confidence if validation.right_ok else 0.0,
            )

        counts[f"L_{l_status}"] += 1
        counts[f"R_{r_status}"] += 1

        # -------------------------------------------------------------
        # Log transitions
        # -------------------------------------------------------------
        if l_status != last_l:
            print(f"  [t={ts:7.2f}s f={idx:6d}] LEFT  {last_l} → {l_status}")
            last_l = l_status
        if r_status != last_r:
            print(f"  [t={ts:7.2f}s f={idx:6d}] RIGHT {last_r} → {r_status}")
            last_r = r_status

        # -------------------------------------------------------------
        # Log file
        # -------------------------------------------------------------
        log_file.write(
            f"{idx},{ts:.2f},"
            f"{int(validation.left_ok)},{int(validation.right_ok)},"
            f"{l_status},{r_status},"
            f"{l_conf:.3f},{r_conf:.3f},"
            f"{state.left.fail_count},{state.right.fail_count}\n"
        )

        # -------------------------------------------------------------
        # Panels
        # -------------------------------------------------------------
        panels = build_panels(
            frame, edges_roi,
            left_raw, right_raw, validation,
            l_coeffs, l_status, l_conf,
            r_coeffs, r_status, r_conf,
            y_range, state,
        )

        grid = build_grid(panels)

        # Save every Nth frame PNG
        if processed % args.save_every == 0:
            png_path = os.path.join(
                args.outdir,
                f"frame_{idx:06d}.png"
            )
            cv2.imwrite(png_path, grid)
            saved_frames += 1

        # Write to video
        if writer is not None:
            writer.write(grid)

        processed += 1

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    log_file.close()
    vr.release()
    if writer is not None:
        writer.release()

    dt = _time.time() - t0

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print()
    print("=" * 74)
    print("SEQUENCE SUMMARY")
    print("=" * 74)
    print(f"Frames processed  : {processed}")
    print(f"Frames saved (PNG): {saved_frames}  → {args.outdir}/")
    print(f"Processing time   : {dt:.1f}s  ({processed/dt:.1f} FPS)")
    print()
    print("LEFT state counts :")
    print(f"  OK   = {counts['L_OK']}")
    print(f"  HOLD = {counts['L_HOLD']}")
    print(f"  MISS = {counts['L_MISS']}")
    print()
    print("RIGHT state counts:")
    print(f"  OK   = {counts['R_OK']}")
    print(f"  HOLD = {counts['R_HOLD']}")
    print(f"  MISS = {counts['R_MISS']}")
    print()
    if not args.no_video:
        print(f"Video : {os.path.join(args.outdir, 'state_trace.mp4')}")
    print(f"Log   : {log_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()