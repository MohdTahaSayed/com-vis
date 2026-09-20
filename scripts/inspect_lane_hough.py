"""
LaneHough stage inspector — STAGES 2-6 ONLY.

Runs and visualizes the front-end of the pipeline through Hough:

    Stage 2 — HorizonDetector.detect()    → horizon_y
    Stage 3 — LaneColor.masks()           → white, yellow, union
    Stage 4 — LaneROI.apply()             → trapezoid clip
    Stage 5 — LaneEdges.compute()         → edges_roi, edges_raw, hsv_hits
    Stage 6 — LaneHough.classify()        → left, right, discard segments

No sliding-window, no fit, no validation, no state, no ego.

All thresholds are read from config/default.yaml on every run.
Re-run after editing YAML to see updated behavior.

Usage:
    python scripts/inspect_lane_hough.py --input data/VBOX0011_Trim.mp4 --time 100
    python scripts/inspect_lane_hough.py --input data/VBOX0011_Trim.mp4 --frame 2500
    python scripts/inspect_lane_hough.py --input data/VBOX0011_Trim.mp4 --times 30 300 900

Outputs:
    outputs/lane_hough_<label>.png            — 3x3 grid of stage panels
    outputs/lane_hough_<label>_edges.png      — edges_roi alone
    outputs/lane_hough_<label>_classified.png — classified segments over frame
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
from src.lane_hough import LaneHough, HoughConfig
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
# Panel builders
# =====================================================================

def make_roi_panel(frame, roi, horizon_y):
    """Frame + ROI polygon + horizon line."""
    h, w = frame.shape[:2]
    out = frame.copy()

    roi_pts = roi._pixel_vertices(w, h, top_y_override=horizon_y)
    cv2.polylines(out, [roi_pts], True, (255, 0, 255), 2)

    if horizon_y is not None:
        cv2.line(out, (0, horizon_y), (w, horizon_y), (0, 255, 255), 2)

    put_label(
        out,
        f"0) frame | ROI + horizon_y={horizon_y}",
        (255, 255, 255),
    )
    return out


def make_edges_panel(edges_roi, edges_roi_px):
    """edges_roi alone."""
    out = gray_to_bgr(edges_roi)
    put_label(
        out,
        f"1) edges_roi | {edges_roi_px} px",
        (255, 255, 255),
    )
    return out


def make_hough_raw_panel(hough, edges_roi):
    """
    All raw Hough segments (no classification filter).
    Purely diagnostic — shows what HoughLinesP found BEFORE slope filtering.
    """
    h, w = edges_roi.shape[:2]
    out = gray_to_bgr(edges_roi)

    # Draw raw segments in white
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=hough.cfg.rho,
        theta=np.deg2rad(hough.cfg.theta_deg),
        threshold=hough.cfg.threshold,
        minLineLength=hough.cfg.min_line_length,
        maxLineGap=hough.cfg.max_line_gap,
    )

    n_raw = 0
    if lines is not None:
        n_raw = len(lines)
        for l in lines:
            if l.ndim == 2:
                x1, y1, x2, y2 = l[0]
            else:
                x1, y1, x2, y2 = l
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     (255, 255, 255), 1)

    put_label(
        out,
        f"2) raw HoughLinesP | {n_raw} segments",
        (255, 255, 255),
    )
    return out, n_raw


def make_classified_panel(frame, left, right, discard):
    """Classification result drawn on original frame."""
    out = frame.copy()

    for seg in left:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 255, 255), 2)   # yellow

    for seg in right:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 255, 0), 2)     # green

    for seg in discard:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 0, 255), 1)     # red

    put_label(
        out,
        f"3) classified | L={len(left)} R={len(right)} D={len(discard)}",
        (255, 255, 255),
    )
    return out


def make_left_panel(edges_roi, left):
    """Only LEFT-classified segments."""
    out = gray_to_bgr(edges_roi)
    for seg in left:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 255, 255), 3)
    put_label(out, f"4) LEFT only | {len(left)} segs", (0, 255, 255))
    return out


def make_right_panel(edges_roi, right):
    """Only RIGHT-classified segments."""
    out = gray_to_bgr(edges_roi)
    for seg in right:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 255, 0), 3)
    put_label(out, f"5) RIGHT only | {len(right)} segs", (0, 255, 0))
    return out


def make_discard_panel(edges_roi, discard):
    """Only DISCARD-classified segments."""
    out = gray_to_bgr(edges_roi)
    for seg in discard:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 0, 255), 3)
    put_label(out, f"6) DISCARD only | {len(discard)} segs", (0, 0, 255))
    return out


def make_slope_histogram_panel(left, right, discard, cfg):
    """
    Histogram of segment slopes, colored by class.
    Helps see whether the slope filter is cutting correctly.
    """
    canvas_h = 300
    canvas_w = 720
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Bin slopes into [-3.0 .. +3.0]
    bins = np.linspace(-3.0, 3.0, 61)   # 60 bins, width 0.1
    bin_w = canvas_w / (len(bins) - 1)

    def hist(segs, color):
        ys = np.zeros(len(bins) - 1, dtype=np.int32)
        for s in segs:
            if np.isfinite(s.slope):
                idx = np.searchsorted(bins, s.slope) - 1
                if 0 <= idx < len(ys):
                    ys[idx] += 1
        return ys

    hL = hist(left, (0, 255, 255))
    hR = hist(right, (0, 255, 0))
    hD = hist(discard, (0, 0, 255))

    max_h = max(1, max(hL.max(), hR.max(), hD.max()))

    for i in range(len(bins) - 1):
        x0 = int(i * bin_w)
        x1 = int((i + 1) * bin_w)
        for h, c in ((hL, (0, 255, 255)),
                     (hR, (0, 255, 0)),
                     (hD, (0, 0, 255))):
            bar_h = int((h[i] / max_h) * (canvas_h - 40))
            if bar_h > 0:
                cv2.rectangle(
                    canvas,
                    (x0, canvas_h - 1 - bar_h),
                    (x1 - 1, canvas_h - 1),
                    c, -1,
                )

    # Zero line
    zero_x = int((0 - bins[0]) / (bins[-1] - bins[0]) * canvas_w)
    cv2.line(canvas, (zero_x, 0), (zero_x, canvas_h),
             (255, 255, 255), 1)

    # Slope filter boundaries
    for bound in (cfg.slope_abs_min, cfg.slope_abs_max):
        for sign in (-1, 1):
            s = sign * bound
            xb = int((s - bins[0]) / (bins[-1] - bins[0]) * canvas_w)
            cv2.line(canvas, (xb, 0), (xb, canvas_h),
                     (128, 128, 128), 1, cv2.LINE_AA)

    put_label(
        canvas,
        f"7) slope histogram | L={len(left)} R={len(right)} D={len(discard)} "
        f"| filter={cfg.slope_abs_min}..{cfg.slope_abs_max}",
        (255, 255, 255),
    )
    return canvas


def make_overlay_panel(edges_roi, left, right, discard):
    """Classified segments over edges_roi (not the original frame)."""
    out = gray_to_bgr(edges_roi)

    for seg in discard:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 0, 255), 1)

    for seg in left:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 255, 255), 2)

    for seg in right:
        cv2.line(out, (seg.x1, seg.y1), (seg.x2, seg.y2),
                 (0, 255, 0), 2)

    put_label(
        out,
        "8) classified over edges_roi",
        (255, 255, 255),
    )
    return out


# =====================================================================
# Build all panels for one frame
# =====================================================================

def build_panels(frame, cfg, horizon, lane_color, roi, edges_module, hough):
    h, w = frame.shape[:2]

    # -----------------------------------------------------------------
    # Stage 2 — horizon
    # -----------------------------------------------------------------
    horizon_y = horizon.detect(frame)

    # -----------------------------------------------------------------
    # Stage 5 — LaneEdges
    # -----------------------------------------------------------------
    edges_roi, edges_raw, hsv_hits = edges_module.compute(
        frame, top_y_override=horizon_y
    )

    # -----------------------------------------------------------------
    # Stage 6 — LaneHough
    # -----------------------------------------------------------------
    left, right, discard = hough.classify(edges_roi)

    # -----------------------------------------------------------------
    # Build panels
    # -----------------------------------------------------------------
    p0 = make_roi_panel(frame, roi, horizon_y)
    p1 = make_edges_panel(edges_roi, int((edges_roi > 0).sum()))
    p2, n_raw = make_hough_raw_panel(hough, edges_roi)
    p3 = make_classified_panel(frame, left, right, discard)
    p4 = make_left_panel(edges_roi, left)
    p5 = make_right_panel(edges_roi, right)
    p6 = make_discard_panel(edges_roi, discard)
    p7 = make_slope_histogram_panel(left, right, discard, hough.cfg)
    p8 = make_overlay_panel(edges_roi, left, right, discard)

    panels = {
        "0_roi":        p0,
        "1_edges":      p1,
        "2_hough_raw":  p2,
        "3_classified": p3,
        "4_left":       p4,
        "5_right":      p5,
        "6_discard":    p6,
        "7_slope_hist": p7,
        "8_overlay":    p8,
    }

    stats = {
        "horizon_y":         horizon_y,
        "edges_roi_px":      int((edges_roi > 0).sum()),
        "raw_hough_segs":    n_raw,
        "left_segs":         len(left),
        "right_segs":        len(right),
        "discard_segs":      len(discard),
        "left_avg_slope":    (float(np.mean([s.slope for s in left]))
                              if left else 0.0),
        "right_avg_slope":   (float(np.mean([s.slope for s in right]))
                              if right else 0.0),
        "discard_avg_slope": (float(np.mean([s.slope for s in discard
                                              if np.isfinite(s.slope)]))
                              if any(np.isfinite(s.slope)
                                     for s in discard) else 0.0),
    }

    return panels, stats


# =====================================================================
# Grid builder (3x3)
# =====================================================================

def build_grid(panels):
    order = [
        "0_roi", "1_edges", "2_hough_raw",
        "3_classified", "4_left", "5_right",
        "6_discard", "7_slope_hist", "8_overlay",
    ]

    target_w = 400

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

    rows = []
    for i in range(0, 9, 3):
        rows.append(np.hstack(padded[i:i + 3]))

    return np.vstack(rows)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inspect STAGES 2-6 (horizon → HSV → ROI → edges → Hough)."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--times", nargs="+", type=float, default=None)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------------------------------------------
    # Config — reloaded every run
    # -----------------------------------------------------------------
    cfg = load_config(args.config)

    horizon = HorizonDetector(
        HorizonConfig.from_dict(cfg.get("horizon", {}))
    )

    lane_color = LaneColor(
        LaneColorConfig.from_dict(cfg.get("lane_color", {}))
    )

    roi = LaneROI(
        RoiConfig.from_dict(cfg.get("roi", {}))
    )

    edges_module = LaneEdges(
        CannyConfig.from_dict(cfg.get("canny", {})),
        roi,
        lane_color,
        reinforce_with_hsv=True,
    )

    hough = LaneHough(
        HoughConfig.from_dict(cfg.get("hough", {}))
    )

    # -----------------------------------------------------------------
    # Decide frames to inspect
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
    print("LANE HOUGH STAGE INSPECTION — STAGES 2-6 ONLY")
    print("=" * 74)
    print(f"Input       : {args.input}")
    print(f"Config      : {args.config}")
    print(f"Resolution  : {vr.info.width}x{vr.info.height} @ {fps:.2f} fps")
    print()
    print("Loaded from YAML:")
    print(f"  Canny              : {edges_module.canny_cfg.low_threshold}/"
          f"{edges_module.canny_cfg.high_threshold}")
    print(f"  White HSV range    : "
          f"H[{lane_color.cfg.white.h_min},{lane_color.cfg.white.h_max}] "
          f"S[{lane_color.cfg.white.s_min},{lane_color.cfg.white.s_max}] "
          f"V[{lane_color.cfg.white.v_min},{lane_color.cfg.white.v_max}]")
    print(f"  Yellow enabled     : {lane_color.cfg.yellow.enabled}")
    print(f"  ROI vertices       : {roi.cfg.vertices}")
    print(f"  Hough rho / theta  : {hough.cfg.rho} / {hough.cfg.theta_deg}°")
    print(f"  Hough threshold    : {hough.cfg.threshold}")
    print(f"  Hough minLen/gap   : {hough.cfg.min_line_length}/"
          f"{hough.cfg.max_line_gap}")
    print(f"  Slope filter       : |slope| ∈ "
          f"[{hough.cfg.slope_abs_min}, {hough.cfg.slope_abs_max}]")
    print("=" * 74)

    # -----------------------------------------------------------------
    # Process each job
    # -----------------------------------------------------------------
    for label, frame_idx, timestamp in jobs:

        frame = vr.read_frame(frame_idx)
        if frame is None:
            print(f"[WARN] cannot read frame {frame_idx}")
            continue

        panels, stats = build_panels(
            frame, cfg, horizon, lane_color,
            roi, edges_module, hough,
        )

        grid = build_grid(panels)
        grid_path = os.path.join(args.outdir, f"lane_hough_{label}.png")
        cv2.imwrite(grid_path, grid)

        cv2.imwrite(
            os.path.join(args.outdir, f"lane_hough_{label}_edges.png"),
            panels["1_edges"],
        )
        cv2.imwrite(
            os.path.join(args.outdir, f"lane_hough_{label}_classified.png"),
            panels["3_classified"],
        )

        # ---------------------------------------------------------
        # Console report
        # ---------------------------------------------------------
        print()
        print(f"--- frame={frame_idx}  t={timestamp:.2f}s ---")
        print(f"  horizon_y           : {stats['horizon_y']}")
        print(f"  edges_roi           : {stats['edges_roi_px']:6d} px")
        print(f"  raw Hough segments  : {stats['raw_hough_segs']:6d}")
        print(f"  LEFT    segments    : {stats['left_segs']:6d}  "
              f"(avg slope {stats['left_avg_slope']:+.3f})")
        print(f"  RIGHT   segments    : {stats['right_segs']:6d}  "
              f"(avg slope {stats['right_avg_slope']:+.3f})")
        print(f"  DISCARD segments    : {stats['discard_segs']:6d}  "
              f"(avg slope {stats['discard_avg_slope']:+.3f})")
        print(f"  -> {grid_path}")

    vr.release()

    print()
    print("=" * 74)
    print("Done.")
    print("=" * 74)


if __name__ == "__main__":
    main()