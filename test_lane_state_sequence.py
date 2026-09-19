from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.lane_state import LaneState, StateConfig
from src.io_video import VideoReader


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def draw_polyline(img, coeffs, y0, y1, color, thickness=4):
    if coeffs is None:
        return
    ys = np.linspace(y0, y1, 200)
    xs = np.polyval(coeffs, ys)
    h, w = img.shape[:2]
    pts = np.column_stack((xs, ys))
    pts = pts[
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    ].astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, color, thickness)


def label(img, text, y):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description="Temporary quadratic temporal lane-state diagnostic."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--start-frame", type=int, default=3456)
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs/lane_state_sequence_check")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config(args.config)

    horizon = HorizonDetector(HorizonConfig.from_dict(cfg.get("horizon", {})))
    lane_color = LaneColor(LaneColorConfig.from_dict(cfg.get("lane_color", {})))
    roi = LaneROI(RoiConfig.from_dict(cfg.get("roi", {})))
    edges_module = LaneEdges(
        CannyConfig.from_dict(cfg.get("canny", {})),
        roi,
        lane_color=lane_color,
        reinforce_with_hsv=True,
    )
    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(cfg.get("sliding_window", {}))
    )
    validator = LaneValidation(
        ValidationConfig.from_dict(cfg.get("validation", {}))
    )
    state = LaneState(StateConfig.from_dict(cfg.get("lane_state", {})))

    video = VideoReader(args.input)
    end_frame = args.start_frame + args.num_frames
    rows = []
    visual_count = 0

    print("=" * 125)
    print("TEMPORARY QUADRATIC TEMPORAL LANE-STATE DIAGNOSTIC")
    print("=" * 125)
    print(f"Input       : {args.input}")
    print(f"Start frame : {args.start_frame}")
    print(f"Frames      : {args.num_frames}")
    print(f"Resolution  : {video.info.width} x {video.info.height}")
    print("This test does NOT modify permanent project code.")
    print()

    sc = cfg.get("lane_state", {})
    print("LANE STATE CONFIG")
    for k in ["hold_frames", "miss_frames", "smoothing_alpha", "min_confidence"]:
        print(f"{k:20s}: {sc.get(k)}")
    print()
    print("State receives a side's polynomial only when that side passes validation.")
    print()
    print("frame | Raw L/R | State L/R | Lconf/Rconf | Lfail/Rfail | Lx_raw/Lx_smooth | Rx_raw/Rx_smooth")
    print("-" * 125)

    previous_left = None
    previous_right = None

    for idx, ts, frame in video.iter_frames(
        start=args.start_frame, end=end_frame, step=1
    ):
        h, w = frame.shape[:2]
        horizon_y = horizon.detect(frame)
        edges_roi, _, _ = edges_module.compute(frame, top_y_override=horizon_y)

        left, right = fitter.fit(
            edges_roi,
            previous_left=previous_left,
            previous_right=previous_right,
        )

        y_min = max(int(h * 0.62), int(horizon_y + 10)) if horizon_y is not None else int(h * 0.62)
        y_max = int(h * 0.95)
        y_range = (y_min, y_max)

        validation = validator.validate(left.coeffs, right.coeffs, y_range, w)

        # Only validated individual sides enter LaneState.
        left_in = left.coeffs if validation.left_ok else None
        right_in = right.coeffs if validation.right_ok else None
        left_conf = left.confidence if validation.left_ok else 0.0
        right_conf = right.confidence if validation.right_ok else 0.0

        (left_sm, left_status, left_state_conf), (right_sm, right_status, right_state_conf) = state.update(
            left_in, left_conf, right_in, right_conf
        )

        y_eval = y_max
        lx_raw = float(np.polyval(left.coeffs, y_eval)) if left.coeffs is not None else np.nan
        rx_raw = float(np.polyval(right.coeffs, y_eval)) if right.coeffs is not None else np.nan
        lx_smooth = float(np.polyval(left_sm, y_eval)) if left_sm is not None else np.nan
        rx_smooth = float(np.polyval(right_sm, y_eval)) if right_sm is not None else np.nan

        print(
            f"{idx:5d} | "
            f"{str(left.coeffs is not None)[0]}/{str(right.coeffs is not None)[0]}      | "
            f"{left_status:5s}/{right_status:5s}   | "
            f"{left_state_conf:5.2f}/{right_state_conf:5.2f}    | "
            f"{state.left.fail_count:2d}/{state.right.fail_count:2d}       | "
            f"{lx_raw:6.1f}/{lx_smooth:6.1f}      | "
            f"{rx_raw:6.1f}/{rx_smooth:6.1f}"
        )

        rows.append({
            "frame": idx,
            "timestamp": ts,
            "left_fit": left.coeffs is not None,
            "right_fit": right.coeffs is not None,
            "left_valid": validation.left_ok,
            "right_valid": validation.right_ok,
            "left_status": left_status,
            "right_status": right_status,
            "left_conf": left_state_conf,
            "right_conf": right_state_conf,
            "left_fail": state.left.fail_count,
            "right_fail": state.right.fail_count,
            "lx_raw": lx_raw,
            "lx_smooth": lx_smooth,
            "rx_raw": rx_raw,
            "rx_smooth": rx_smooth,
        })

        if left.coeffs is not None:
            previous_left = left.coeffs.copy()
        if right.coeffs is not None:
            previous_right = right.coeffs.copy()

        if visual_count == 0 or visual_count % 10 == 0:
            vis = frame.copy()
            if horizon_y is not None:
                cv2.line(vis, (0, int(horizon_y)), (w - 1, int(horizon_y)), (255, 255, 0), 2)

            # Raw fit: thin white, smoothed state: colored thick curve.
            draw_polyline(vis, left.coeffs, y_range[0], y_range[1], (255, 255, 255), 2)
            draw_polyline(vis, right.coeffs, y_range[0], y_range[1], (255, 255, 255), 2)
            draw_polyline(vis, left_sm, y_range[0], y_range[1], (0, 255, 255), 4)
            draw_polyline(vis, right_sm, y_range[0], y_range[1], (0, 255, 0), 4)

            label(vis, f"FRAME {idx} | L={left_status} R={right_status}", 24)
            label(vis, f"conf L/R={left_state_conf:.2f}/{right_state_conf:.2f} fail={state.left.fail_count}/{state.right.fail_count}", 48)
            label(vis, "white=raw fit | yellow/green=smoothed state", 72)

            cv2.imwrite(os.path.join(args.outdir, f"state_{idx:06d}.png"), vis)

        visual_count += 1

    video.release()

    csv_path = os.path.join(args.outdir, "lane_state_sequence_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,timestamp,left_fit,right_fit,left_valid,right_valid,left_status,right_status,left_conf,right_conf,left_fail,right_fail,lx_raw,lx_smooth,rx_raw,rx_smooth\n")
        for r in rows:
            f.write(
                f"{r['frame']},{r['timestamp']:.3f},{r['left_fit']},{r['right_fit']},"
                f"{r['left_valid']},{r['right_valid']},{r['left_status']},{r['right_status']},"
                f"{r['left_conf']:.6f},{r['right_conf']:.6f},{r['left_fail']},{r['right_fail']},"
                f"{r['lx_raw']:.3f},{r['lx_smooth']:.3f},{r['rx_raw']:.3f},{r['rx_smooth']:.3f}\n"
            )

    n = len(rows)
    print()
    print("=" * 125)
    print("SUMMARY")
    print("=" * 125)
    print(f"Frames processed : {n}")
    print(f"Left OK          : {sum(r['left_status'] == 'OK' for r in rows)}/{n}")
    print(f"Right OK         : {sum(r['right_status'] == 'OK' for r in rows)}/{n}")
    print(f"Left HOLD        : {sum(r['left_status'] == 'HOLD' for r in rows)}")
    print(f"Right HOLD       : {sum(r['right_status'] == 'HOLD' for r in rows)}")
    print(f"Left MISS        : {sum(r['left_status'] == 'MISS' for r in rows)}")
    print(f"Right MISS       : {sum(r['right_status'] == 'MISS' for r in rows)}")

    if rows:
        ldiff = [abs(r['lx_raw'] - r['lx_smooth']) for r in rows if np.isfinite(r['lx_raw']) and np.isfinite(r['lx_smooth'])]
        rdiff = [abs(r['rx_raw'] - r['rx_smooth']) for r in rows if np.isfinite(r['rx_raw']) and np.isfinite(r['rx_smooth'])]
        if ldiff:
            print(f"Left raw-vs-smooth x difference : mean={np.mean(ldiff):.2f}px max={np.max(ldiff):.2f}px")
        if rdiff:
            print(f"Right raw-vs-smooth x difference: mean={np.mean(rdiff):.2f}px max={np.max(rdiff):.2f}px")

    print()
    print(f"CSV     : {csv_path}")
    print(f"Visuals : {args.outdir}")
    print("=" * 125)


if __name__ == "__main__":
    main()
