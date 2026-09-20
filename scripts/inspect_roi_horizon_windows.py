"""
ROI + Horizon + Sliding-Window inspector.

Visualizes THREE things together:
    1) ROI trapezoid (Stage 4)
    2) Horizon line (Stage 2)
    3) Sliding-window lane detection + histogram centre (Stage 7a/7b)

All parameters read live from config/default.yaml on every run.
Edit the YAML, re-run, and the output reflects your new values.

Usage:
    python scripts/inspect_roi_horizon_windows.py --input data/VBOX0011_Trim.mp4 --time 100
    python scripts/inspect_roi_horizon_windows.py --input data/VBOX0011_Trim.mp4 --frame 19250
    python scripts/inspect_roi_horizon_windows.py --input data/VBOX0011_Trim.mp4 --times 30 300 900

Outputs:
    outputs/roi_horizon_windows_<label>.png        ← 3x3 master grid
    outputs/roi_horizon_windows_<label>_roi.png    ← ROI panel alone
    outputs/roi_horizon_windows_<label>_hist.png   ← histogram panel alone
    outputs/roi_horizon_windows_<label>_left.png   ← left sliding windows
    outputs/roi_horizon_windows_<label>_right.png  ← right sliding windows
"""

from __future__ import annotations

import argparse
import os
import sys

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


def gray_to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


# =====================================================================
# HISTOGRAM computation (matches LaneFitter._histogram_base)
# =====================================================================

def compute_histogram(edges_roi, fitter):
    h, w = edges_roi.shape[:2]

    bottom_start = int(h * 0.70)

    hist = np.sum(
        edges_roi[bottom_start:, :] > 0,
        axis=0,
    ).astype(np.float32)

    kernel_size = 15
    if w >= kernel_size:
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        hist_smooth = np.convolve(hist, kernel, mode="same")
    else:
        hist_smooth = hist

    # LEFT
    left_end = int(w * 0.45)
    left_region = hist_smooth[:left_end]
    if left_region.max() > 0:
        left_base = int(np.argmax(left_region))
    else:
        left_base = int(w * 0.25)
    left_base += fitter.cfg.left_base_shift_px
    left_base = int(np.clip(left_base, 0, w - 1))

    # RIGHT
    right_start = int(w * 0.55)
    right_region = hist_smooth[right_start:]
    if right_region.max() > 0:
        right_base = int(np.argmax(right_region)) + right_start
    else:
        right_base = int(w * 0.75)
    right_base += fitter.cfg.right_base_shift_px
    right_base = int(np.clip(right_base, 0, w - 1))

    return left_base, right_base, hist, hist_smooth


# =====================================================================
# SLIDING-WINDOW replay (matches LaneFitter._collect_side_pixels)
# =====================================================================

def replay_sliding_windows(edges_roi, x_base, side, fitter):
    h, w = edges_roi.shape[:2]

    nonzero_y, nonzero_x = np.nonzero(edges_roi > 0)

    if len(nonzero_x) == 0:
        return [], np.array([]), np.array([])

    n_windows = fitter.cfg.n_windows
    window_height = max(1, h // n_windows)
    window_width = max(10, int(w * fitter.cfg.window_width_frac))

    current_x = int(x_base)
    records = []
    cx_all = []
    cy_all = []

    for window in range(n_windows):
        y_high = h - window * window_height
        y_low = max(0, h - (window + 1) * window_height)

        search_half = window_width // 2

        sx_low = max(0, current_x - search_half)
        sx_high = min(w, current_x + search_half)

        region = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= sx_low)
            & (nonzero_x < sx_high)
        )

        cx = nonzero_x[region]
        cy = nonzero_y[region]

        n_cand = len(cx)
        old_x = current_x

        if n_cand > 0:
            cx_all.append(cx)
            cy_all.append(cy)

        recentered = False
        if n_cand >= fitter.cfg.min_pixels_to_recenter:
            new_x = int(np.median(cx))
            max_jump = int(w * fitter.cfg.max_recenter_jump_frac)
            if abs(new_x - current_x) <= max_jump:
                current_x = new_x
                recentered = True

        records.append({
            "window": window + 1,
            "y_low": y_low,
            "y_high": y_high,
            "sx_low": sx_low,
            "sx_high": sx_high,
            "n_cand": n_cand,
            "x_before": old_x,
            "x_after": current_x,
            "recentered": recentered,
        })

    if cx_all:
        return records, np.concatenate(cx_all), np.concatenate(cy_all)

    return records, np.array([]), np.array([])


# =====================================================================
# PANEL BUILDERS
# =====================================================================

def panel_frame_roi_horizon(frame, roi, horizon_y):
    """Original frame + ROI polygon + horizon line."""
    h, w = frame.shape[:2]
    out = frame.copy()

    roi_pts = roi._pixel_vertices(w, h, top_y_override=horizon_y)

    # filled ROI (faint magenta)
    overlay = out.copy()
    cv2.fillPoly(overlay, [roi_pts], (255, 0, 255))
    out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)

    # outline
    cv2.polylines(out, [roi_pts], True, (255, 0, 255), 2)

    # horizon line
    if horizon_y is not None:
        cv2.line(out, (0, horizon_y), (w, horizon_y), (0, 255, 255), 2)

    # corner markers
    for i, (x, y) in enumerate(roi_pts):
        cv2.circle(out, (int(x), int(y)), 6, (0, 255, 255), -1)
        cv2.putText(
            out, f"P{i+1}", (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 255), 2, cv2.LINE_AA,
        )

    put_label(
        out,
        f"0) frame + ROI + horizon_y={horizon_y}",
        (255, 255, 255),
    )
    return out


def panel_roi_mask(frame, roi, horizon_y):
    """Binary ROI mask."""
    h, w = frame.shape[:2]
    out = frame.copy() * 0
    out[:] = (20, 20, 20)

    roi_pts = roi._pixel_vertices(w, h, top_y_override=horizon_y)
    cv2.fillPoly(out, [roi_pts], (255, 255, 255))
    cv2.polylines(out, [roi_pts], True, (255, 0, 255), 2)

    put_label(out, f"1) ROI mask | top_y={roi_pts[1][1]}", (255, 255, 255))
    return out


def panel_horizon_only(frame, horizon, horizon_y, cfg):
    """Frame with horizon line + search band."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # search band (from YAML)
    hz_cfg = cfg.get("horizon", {})
    top_frac = hz_cfg.get("search_top_frac", 0.50)
    bot_frac = hz_cfg.get("search_bot_frac", 0.70)

    y_top = int(top_frac * h)
    y_bot = int(bot_frac * h)

    # shaded search band
    overlay = out.copy()
    cv2.rectangle(
        overlay,
        (0, y_top), (w, y_bot),
        (100, 100, 100), -1,
    )
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

    cv2.line(out, (0, y_top), (w, y_top), (128, 128, 128), 1)
    cv2.line(out, (0, y_bot), (w, y_bot), (128, 128, 128), 1)

    # min/max clamp
    min_y = int(hz_cfg.get("min_y_frac", 0.62) * h)
    max_y = int(hz_cfg.get("max_y_frac", 0.62) * h)
    cv2.line(out, (0, min_y), (w, min_y), (0, 255, 0), 1)
    cv2.line(out, (0, max_y), (w, max_y), (0, 0, 255), 1)

    # detected horizon
    if horizon_y is not None:
        cv2.line(out, (0, horizon_y), (w, horizon_y), (0, 255, 255), 3)

    put_label(
        out,
        f"2) horizon_y={horizon_y} | band [{y_top},{y_bot}] "
        f"clamp [{min_y},{max_y}]",
        (255, 255, 255),
    )
    return out


def panel_edges(edges_roi, left_base, right_base):
    """edges_roi with left/right base dots."""
    h, w = edges_roi.shape[:2]
    out = gray_to_bgr(edges_roi)

    # base dots at bottom
    cv2.circle(out, (left_base, h - 10), 8, (0, 255, 255), -1)
    cv2.circle(out, (right_base, h - 10), 8, (0, 255, 0), -1)

    put_label(
        out,
        f"3) edges_roi | Lbase={left_base} Rbase={right_base}",
        (255, 255, 255),
    )
    return out


def panel_histogram(hist_raw, hist_smooth, left_base, right_base, w, h):
    """Standalone histogram plot."""
    canvas_h = 400
    canvas_w = w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    max_val = max(float(hist_smooth.max()), 1.0)

    bottom_y = canvas_h - 40

    # raw histogram (dim)
    for x in range(w):
        bar_h = int((hist_raw[x] / max_val) * (bottom_y - 30))
        cv2.line(
            canvas, (x, bottom_y), (x, bottom_y - bar_h),
            (60, 60, 60), 1,
        )

    # smoothed histogram (bright)
    for x in range(w):
        bar_h = int((hist_smooth[x] / max_val) * (bottom_y - 30))
        cv2.line(
            canvas, (x, bottom_y), (x, bottom_y - bar_h),
            (200, 200, 200), 1,
        )

    # search region boundaries
    cv2.line(canvas, (int(0.45 * w), 0), (int(0.45 * w), canvas_h),
             (255, 255, 0), 2)
    cv2.line(canvas, (int(0.55 * w), 0), (int(0.55 * w), canvas_h),
             (255, 255, 0), 2)

    cv2.putText(canvas, "45%", (int(0.45 * w) + 4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "55%", (int(0.55 * w) + 4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # LEFT base
    cv2.line(canvas, (left_base, 0), (left_base, canvas_h), (0, 255, 255), 3)
    cv2.putText(canvas, f"L={left_base}",
                (left_base + 6, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # RIGHT base
    cv2.line(canvas, (right_base, 0), (right_base, canvas_h), (0, 255, 0), 3)
    cv2.putText(canvas, f"R={right_base}",
                (right_base + 6, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    put_label(
        canvas,
        f"4) histogram | bottom 30% | Lbase={left_base} Rbase={right_base}",
        (255, 255, 255),
    )
    return canvas


def panel_sliding_windows(edges_roi, records, cx, cy, side):
    """Sliding-window rectangles + collected pixels."""
    out = gray_to_bgr(edges_roi)

    color_ok = (0, 255, 255) if side == "LEFT" else (0, 255, 0)
    color_dim = (100, 100, 100)

    for rec in records:
        color = color_ok if rec["recentered"] else color_dim
        cv2.rectangle(
            out,
            (rec["sx_low"], rec["y_low"]),
            (rec["sx_high"], rec["y_high"]),
            color, 1,
        )
        cv2.putText(
            out, f"W{rec['window']}:{rec['n_cand']}",
            (rec["sx_low"] + 2, rec["y_low"] + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    # collected pixels
    for x, y in zip(cx, cy):
        cv2.circle(out, (int(x), int(y)), 1, (0, 0, 255), -1)

    put_label(
        out,
        f"{'5' if side=='LEFT' else '6'}) {side} windows | "
        f"collected={len(cx)}px",
        color_ok,
    )
    return out


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--times", nargs="+", type=float, default=None)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------------------------------------------
    # LIVE YAML load
    # -----------------------------------------------------------------
    cfg = load_config(args.config)

    horizon = HorizonDetector(HorizonConfig.from_dict(cfg.get("horizon", {})))
    lane_color = LaneColor(LaneColorConfig.from_dict(cfg.get("lane_color", {})))
    roi = LaneROI(RoiConfig.from_dict(cfg.get("roi", {})))
    edges_module = LaneEdges(
        CannyConfig.from_dict(cfg.get("canny", {})),
        roi, lane_color, True,
    )
    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(cfg.get("sliding_window", {}))
    )

    # -----------------------------------------------------------------
    # Job list
    # -----------------------------------------------------------------
    vr = VideoReader(args.input)
    fps = vr.info.fps

    jobs = []
    if args.frame is not None:
        jobs.append((f"frame_{args.frame}", args.frame, args.frame / fps))
    elif args.time is not None:
        f = int(round(args.time * fps))
        jobs.append((f"{args.time:.2f}s", f, args.time))
    elif args.times is not None:
        for t in args.times:
            f = int(round(t * fps))
            jobs.append((f"{t:.2f}s", f, t))
    else:
        for t in (30.0, 300.0, 900.0):
            f = int(round(t * fps))
            jobs.append((f"{t:.2f}s", f, t))

    # -----------------------------------------------------------------
    # Banner
    # -----------------------------------------------------------------
    print()
    print("=" * 74)
    print("ROI + HORIZON + SLIDING WINDOWS INSPECTOR")
    print("=" * 74)
    print(f"Input     : {args.input}")
    print(f"Config    : {args.config}")
    print(f"Resolution: {vr.info.width}x{vr.info.height} @ {fps:.2f} fps")
    print()
    print("LIVE values loaded from YAML:")

    hz_cfg = cfg.get("horizon", {})
    print(f"  Horizon search band : [{hz_cfg.get('search_top_frac')}, "
          f"{hz_cfg.get('search_bot_frac')}]")
    print(f"  Horizon clamp       : [{hz_cfg.get('min_y_frac')}, "
          f"{hz_cfg.get('max_y_frac')}]")
    print(f"  Horizon threshold   : {hz_cfg.get('variance_threshold')}")

    roi_cfg = cfg.get("roi", {})
    print(f"  ROI enabled         : {roi_cfg.get('enabled')}")
    for i, v in enumerate(roi_cfg.get("vertices", [])):
        print(f"    P{i+1}: {v}")

    sw_cfg = cfg.get("sliding_window", {})
    print(f"  Sliding n_windows   : {sw_cfg.get('n_windows')}")
    print(f"  window_width_frac   : {sw_cfg.get('window_width_frac')}")
    print(f"  min_pixels_recenter : {sw_cfg.get('min_pixels_to_recenter')}")
    print(f"  L base shift        : {sw_cfg.get('left_base_shift_px')}")
    print(f"  R base shift        : {sw_cfg.get('right_base_shift_px')}")
    print("=" * 74)

    # -----------------------------------------------------------------
    # Process each job
    # -----------------------------------------------------------------
    for label, frame_idx, timestamp in jobs:

        frame = vr.read_frame(frame_idx)
        if frame is None:
            print(f"[WARN] cannot read frame {frame_idx}")
            continue

        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # Stage 2 — horizon
        # -------------------------------------------------------------
        horizon_y = horizon.detect(frame)

        # -------------------------------------------------------------
        # Stage 5 — edges_roi
        # -------------------------------------------------------------
        edges_roi, _, _ = edges_module.compute(
            frame, top_y_override=horizon_y
        )

        # -------------------------------------------------------------
        # Stage 7a — histogram
        # -------------------------------------------------------------
        left_base, right_base, hist_raw, hist_smooth = compute_histogram(
            edges_roi, fitter
        )

        # -------------------------------------------------------------
        # Stage 7b — sliding windows
        # -------------------------------------------------------------
        left_recs, lx, ly = replay_sliding_windows(
            edges_roi, left_base, "LEFT", fitter
        )
        right_recs, rx, ry = replay_sliding_windows(
            edges_roi, right_base, "RIGHT", fitter
        )

        # -------------------------------------------------------------
        # Panels
        # -------------------------------------------------------------
        p0 = panel_frame_roi_horizon(frame, roi, horizon_y)
        p1 = panel_roi_mask(frame, roi, horizon_y)
        p2 = panel_horizon_only(frame, horizon, horizon_y, cfg)
        p3 = panel_edges(edges_roi, left_base, right_base)
        p4 = panel_histogram(hist_raw, hist_smooth,
                             left_base, right_base, w, h)
        p5 = panel_sliding_windows(edges_roi, left_recs, lx, ly, "LEFT")
        p6 = panel_sliding_windows(edges_roi, right_recs, rx, ry, "RIGHT")

        # -------------------------------------------------------------
        # Save individual panels
        # -------------------------------------------------------------
        base = os.path.join(args.outdir, f"roi_horizon_windows_{label}")

        cv2.imwrite(f"{base}_roi.png", p0)
        cv2.imwrite(f"{base}_hist.png", p4)
        cv2.imwrite(f"{base}_left.png", p5)
        cv2.imwrite(f"{base}_right.png", p6)

        # -------------------------------------------------------------
        # 3x3 master grid
        # -------------------------------------------------------------
        target_w = 480

        def resize(p):
            hh, ww = p.shape[:2]
            nh = int(hh * target_w / ww)
            return cv2.resize(p, (target_w, nh))

        panels = [p0, p1, p2, p3, p4, p5, p6]
        resized = [resize(p) for p in panels]

        # pad all to same height
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

        # blank slot for panel 8 (keep grid 3x3)
        blank = np.zeros_like(padded[0])
        padded.append(blank)
        padded.append(blank)

        row1 = np.hstack(padded[0:3])
        row2 = np.hstack(padded[3:6])
        row3 = np.hstack(padded[6:9])

        grid = np.vstack([row1, row2, row3])

        grid_path = f"{base}.png"
        cv2.imwrite(grid_path, grid)

        # -------------------------------------------------------------
        # Terminal report
        # -------------------------------------------------------------
        print()
        print(f"--- frame={frame_idx}  t={timestamp:.2f}s ---")
        print(f"  horizon_y           : {horizon_y}")
        print(f"  ROI top_y           : "
              f"{roi._pixel_vertices(w, h, top_y_override=horizon_y)[1][1]}")
        print(f"  edges_roi px        : {int((edges_roi > 0).sum())}")
        print(f"  hist left_base      : {left_base}")
        print(f"  hist right_base     : {right_base}")
        print(f"  LEFT collected px   : {len(lx)}")
        print(f"  RIGHT collected px  : {len(rx)}")
        print(f"  -> {grid_path}")

    vr.release()

    print()
    print("=" * 74)
    print("Done.")
    print("=" * 74)


if __name__ == "__main__":
    main()