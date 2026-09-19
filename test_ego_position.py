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


def draw_polyline(img, coeffs, y0, y1, color, thickness=3):
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
    cv2.putText(
        img, text, (8, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (8, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (255, 255, 255), 1, cv2.LINE_AA
    )


def main():
    ap = argparse.ArgumentParser(
        description="Temporary quadratic ego-position diagnostic."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--start-frame", type=int, default=3456)
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--outdir",
        default="outputs/ego_position_check"
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
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
        lane_color=lane_color,
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

    ego_cfg = cfg.get("ego_position", {})
    ego_x_frac = float(ego_cfg.get("ego_x_frac", 0.50))
    lane_eval_y_frac = float(
        ego_cfg.get("lane_eval_y_frac", 0.85)
    )

    video = VideoReader(args.input)

    end_frame = args.start_frame + args.num_frames
    rows = []
    visual_count = 0

    previous_left = None
    previous_right = None

    print("=" * 120)
    print("TEMPORARY QUADRATIC EGO-POSITION DIAGNOSTIC")
    print("=" * 120)
    print(f"Input          : {args.input}")
    print(f"Start frame    : {args.start_frame}")
    print(f"Frames         : {args.num_frames}")
    print(f"Resolution     : {video.info.width} x {video.info.height}")
    print("This test does NOT modify permanent project code.")
    print()
    print("EGO CONFIG")
    print(f"ego_x_frac       : {ego_x_frac}")
    print(f"lane_eval_y_frac : {lane_eval_y_frac}")
    print()
    print(
        "offset = ego_x - lane_center_x; "
        "normalized = offset / (lane_width / 2)"
    )
    print()
    print(
        "Columns: frame | state L/R | y_eval | "
        "Lx | Rx | width | center | ego_x | offset | norm | position"
    )
    print("-" * 120)

    for idx, ts, frame in video.iter_frames(
        start=args.start_frame,
        end=end_frame,
        step=1
    ):
        h, w = frame.shape[:2]

        horizon_y = horizon.detect(frame)

        edges_roi, _, _ = edges_module.compute(
            frame,
            top_y_override=horizon_y
        )

        left, right = fitter.fit(
            edges_roi,
            previous_left=previous_left,
            previous_right=previous_right
        )

        if horizon_y is not None:
            y_min = max(int(h * 0.62), int(horizon_y + 10))
        else:
            y_min = int(h * 0.62)

        y_max = int(h * 0.95)
        y_range = (y_min, y_max)

        validation = validator.validate(
            left.coeffs,
            right.coeffs,
            y_range,
            w
        )

        # Only validated fits are passed into LaneState.
        left_in = (
            left.coeffs
            if validation.left_ok
            else None
        )
        right_in = (
            right.coeffs
            if validation.right_ok
            else None
        )

        (left_state, right_state) = lane_state.update(
            left_in,
            left.confidence,
            right_in,
            right.confidence
        )

        left_coeffs, left_status, left_conf = left_state
        right_coeffs, right_status, right_conf = right_state

        y_eval = int(round(h * lane_eval_y_frac))
        ego_x = ego_x_frac * w

        lx = (
            float(np.polyval(left_coeffs, y_eval))
            if left_coeffs is not None
            else np.nan
        )
        rx = (
            float(np.polyval(right_coeffs, y_eval))
            if right_coeffs is not None
            else np.nan
        )

        if np.isfinite(lx) and np.isfinite(rx) and rx > lx:
            lane_width = rx - lx
            lane_center = (lx + rx) / 2.0
            offset = ego_x - lane_center

            # Negative = ego is left of lane center.
            # Positive = ego is right of lane center.
            normalized = offset / (lane_width / 2.0)

            if normalized < -0.05:
                position = "LEFT"
            elif normalized > 0.05:
                position = "RIGHT"
            else:
                position = "CENTER"
        else:
            lane_width = np.nan
            lane_center = np.nan
            offset = np.nan
            normalized = np.nan
            position = "UNAVAILABLE"

        print(
            f"{idx:5d} | "
            f"{left_status:4s}/{right_status:4s} | "
            f"{y_eval:5d} | "
            f"{lx:6.1f} | {rx:6.1f} | "
            f"{lane_width:6.1f} | "
            f"{lane_center:6.1f} | "
            f"{ego_x:5.1f} | "
            f"{offset:7.1f} | "
            f"{normalized:6.3f} | "
            f"{position}"
        )

        rows.append({
            "frame": idx,
            "timestamp": ts,
            "horizon_y": horizon_y,
            "y_eval": y_eval,
            "left_status": left_status,
            "right_status": right_status,
            "left_x": lx,
            "right_x": rx,
            "lane_width": lane_width,
            "lane_center": lane_center,
            "ego_x": ego_x,
            "offset": offset,
            "normalized_offset": normalized,
            "position": position,
        })

        if left_coeffs is not None:
            previous_left = left.coeffs.copy()
        if right_coeffs is not None:
            previous_right = right.coeffs.copy()

        if visual_count == 0 or visual_count % 10 == 0:
            vis = frame.copy()

            if horizon_y is not None:
                cv2.line(
                    vis,
                    (0, int(horizon_y)),
                    (w - 1, int(horizon_y)),
                    (255, 255, 0),
                    2
                )

            draw_polyline(
                vis, left_coeffs,
                y_range[0], y_range[1],
                (0, 255, 255), 4
            )
            draw_polyline(
                vis, right_coeffs,
                y_range[0], y_range[1],
                (0, 255, 0), 4
            )

            # Ego x line.
            cv2.line(
                vis,
                (int(round(ego_x)), y_range[0]),
                (int(round(ego_x)), h - 1),
                (255, 0, 255),
                2
            )

            # Evaluation points and lane center.
            if np.isfinite(lx) and np.isfinite(rx):
                cv2.circle(
                    vis, (int(round(lx)), y_eval),
                    6, (0, 255, 255), -1
                )
                cv2.circle(
                    vis, (int(round(rx)), y_eval),
                    6, (0, 255, 0), -1
                )
                cv2.circle(
                    vis, (int(round(lane_center)), y_eval),
                    6, (255, 255, 255), -1
                )

            label(
                vis,
                f"FRAME {idx} | L={left_status} R={right_status}",
                24
            )
            label(
                vis,
                f"Lane L/R={lx:.1f}/{rx:.1f} "
                f"center={lane_center:.1f} ego={ego_x:.1f}",
                48
            )
            label(
                vis,
                f"offset={offset:.1f}px "
                f"norm={normalized:.3f} "
                f"position={position}",
                72
            )

            out = os.path.join(
                args.outdir,
                f"ego_position_{idx:06d}.png"
            )
            cv2.imwrite(out, vis)

        visual_count += 1

    video.release()

    csv_path = os.path.join(
        args.outdir,
        "ego_position_results.csv"
    )

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "frame,timestamp,horizon_y,y_eval,"
            "left_status,right_status,left_x,right_x,"
            "lane_width,lane_center,ego_x,offset,"
            "normalized_offset,position\n"
        )

        for r in rows:
            f.write(
                f"{r['frame']},{r['timestamp']:.3f},"
                f"{r['horizon_y']},{r['y_eval']},"
                f"{r['left_status']},{r['right_status']},"
                f"{r['left_x']:.3f},{r['right_x']:.3f},"
                f"{r['lane_width']:.3f},{r['lane_center']:.3f},"
                f"{r['ego_x']:.3f},{r['offset']:.3f},"
                f"{r['normalized_offset']:.6f},"
                f"{r['position']}\n"
            )

    available = [
        r for r in rows
        if np.isfinite(r["normalized_offset"])
    ]

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Frames processed       : {len(rows)}")
    print(f"Ego position available : {len(available)}/{len(rows)}")

    if available:
        norms = np.array(
            [r["normalized_offset"] for r in available],
            dtype=float
        )
        offsets = np.array(
            [r["offset"] for r in available],
            dtype=float
        )
        widths = np.array(
            [r["lane_width"] for r in available],
            dtype=float
        )

        print(
            f"Lane width min/mean/max: "
            f"{widths.min():.1f} / {widths.mean():.1f} / {widths.max():.1f}"
        )
        print(
            f"Offset min/mean/max    : "
            f"{offsets.min():.1f} / {offsets.mean():.1f} / {offsets.max():.1f} px"
        )
        print(
            f"Norm offset min/mean/max: "
            f"{norms.min():.3f} / {norms.mean():.3f} / {norms.max():.3f}"
        )

        counts = {}
        for r in available:
            p = r["position"]
            counts[p] = counts.get(p, 0) + 1

        print()
        print("POSITION COUNTS:")
        for p in ["LEFT", "CENTER", "RIGHT", "UNAVAILABLE"]:
            if p in counts:
                print(f"  {p:12s}: {counts[p]}")

    print()
    print(f"CSV    : {csv_path}")
    print(f"Visuals: {args.outdir}")
    print("=" * 120)


if __name__ == "__main__":
    main()