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
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description="Temporary sliding-window polynomial lane-fit diagnostic."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--start-frame", type=int, default=3456)
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs/polynomial_fit_check")
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

    video = VideoReader(args.input)

    end_frame = args.start_frame + args.num_frames
    rows = []
    visual_count = 0

    print("=" * 95)
    print("TEMPORARY SLIDING-WINDOW POLYNOMIAL FIT DIAGNOSTIC")
    print("=" * 95)
    print(f"Input          : {args.input}")
    print(f"Start frame    : {args.start_frame}")
    print(f"Frames         : {args.num_frames}")
    print(f"FPS            : {video.info.fps:.2f}")
    print(f"Resolution     : {video.info.width} x {video.info.height}")
    print()
    print("This test does NOT modify permanent project code.")
    print()
    print("SLIDING-WINDOW CONFIG")
    sw = cfg.get("sliding_window", {})
    for k in [
        "n_windows", "window_width_frac", "min_pixels_to_recenter",
        "min_pixels_total", "min_pixels_per_window",
        "max_fit_rms_px", "vanishing_x_frac",
        "left_max_x_frac", "right_min_x_frac",
        "max_recenter_jump_frac", "tracking_enabled",
        "tracking_margin_px"
    ]:
        print(f"{k:25s}: {sw.get(k)}")
    print()
    print("Columns: frame | Lpx | Rpx | Lconf | Rconf | L_RMS | R_RMS | Lfit | Rfit")
    print("-" * 95)

    previous_left = None
    previous_right = None

    for idx, ts, frame in video.iter_frames(
        start=args.start_frame,
        end=end_frame,
        step=1
    ):
        horizon_y = horizon.detect(frame)

        edges_roi, edges_raw, hsv_hits = edges_module.compute(
            frame,
            top_y_override=horizon_y
        )

        left, right = fitter.fit(
            edges_roi,
            previous_left=previous_left,
            previous_right=previous_right
        )

        lvalid = left.coeffs is not None
        rvalid = right.coeffs is not None

        print(
            f"{idx:5d} | "
            f"{left.n_pixels:4d} | {right.n_pixels:4d} | "
            f"{left.confidence:5.2f} | {right.confidence:5.2f} | "
            f"{left.rms_error:6.2f} | {right.rms_error:6.2f} | "
            f"{str(lvalid):5s} | {str(rvalid):5s}"
        )

        rows.append([
            idx, ts, horizon_y,
            left.n_pixels, right.n_pixels,
            left.confidence, right.confidence,
            left.rms_error, right.rms_error,
            left.coeffs.tolist() if left.coeffs is not None else None,
            right.coeffs.tolist() if right.coeffs is not None else None,
        ])

        # Only use successful fits as temporal references.
        if left.coeffs is not None:
            previous_left = left.coeffs.copy()
        if right.coeffs is not None:
            previous_right = right.coeffs.copy()

        # Save first frame and every 10th frame.
        if visual_count == 0 or visual_count % 10 == 0:
            vis = frame.copy()
            h, w = vis.shape[:2]

            if horizon_y is not None:
                cv2.line(
                    vis, (0, int(horizon_y)), (w - 1, int(horizon_y)),
                    (255, 255, 0), 2
                )

            y0 = int(h * 0.62)
            y1 = int(h * 0.95)

            draw_polyline(vis, left.coeffs, y0, y1, (0, 255, 255), 4)
            draw_polyline(vis, right.coeffs, y0, y1, (0, 255, 0), 4)

            label(
                vis,
                f"FRAME {idx} | Lpx={left.n_pixels} Rpx={right.n_pixels}",
                24
            )
            label(
                vis,
                f"Lconf={left.confidence:.2f} Rconf={right.confidence:.2f}",
                48
            )
            label(
                vis,
                f"LRMS={left.rms_error:.1f} RRMS={right.rms_error:.1f}",
                72
            )

            out = os.path.join(
                args.outdir,
                f"poly_fit_{idx:06d}.png"
            )
            cv2.imwrite(out, vis)

        visual_count += 1

    video.release()

    csv_path = os.path.join(args.outdir, "polynomial_fit_results.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "frame,timestamp,horizon_y,"
            "left_pixels,right_pixels,"
            "left_conf,right_conf,"
            "left_rms,right_rms,"
            "left_coeffs,right_coeffs\n"
        )
        for r in rows:
            f.write(
                f"{r[0]},{r[1]:.3f},{r[2]},"
                f"{r[3]},{r[4]},"
                f"{r[5]:.6f},{r[6]:.6f},"
                f"{r[7]:.6f},{r[8]:.6f},"
                f"\"{r[9]}\",\"{r[10]}\"\n"
            )

    print()
    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"Frames processed : {len(rows)}")
    print(f"Left valid       : {sum(r[9] is not None for r in rows)}/{len(rows)}")
    print(f"Right valid      : {sum(r[10] is not None for r in rows)}/{len(rows)}")

    if rows:
        lpx = [r[3] for r in rows]
        rpx = [r[4] for r in rows]
        lc = [r[5] for r in rows]
        rc = [r[6] for r in rows]
        lr = [r[7] for r in rows]
        rr = [r[8] for r in rows]

        print()
        print("LEFT:")
        print(f"  pixels   min/mean/max : {min(lpx)} / {np.mean(lpx):.1f} / {max(lpx)}")
        print(f"  conf     min/mean/max : {min(lc):.2f} / {np.mean(lc):.2f} / {max(lc):.2f}")
        print(f"  RMS      min/mean/max : {min(lr):.2f} / {np.mean(lr):.2f} / {max(lr):.2f}")

        print()
        print("RIGHT:")
        print(f"  pixels   min/mean/max : {min(rpx)} / {np.mean(rpx):.1f} / {max(rpx)}")
        print(f"  conf     min/mean/max : {min(rc):.2f} / {np.mean(rc):.2f} / {max(rc):.2f}")
        print(f"  RMS      min/mean/max : {min(rr):.2f} / {np.mean(rr):.2f} / {max(rr):.2f}")

    print()
    print(f"CSV              : {csv_path}")
    print(f"Visuals           : {args.outdir}")
    print("=" * 95)


if __name__ == "__main__":
    main()