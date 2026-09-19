from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_hough import LaneHough, HoughConfig
from src.lane_state import LaneState, StateConfig
from src.io_video import VideoReader


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fit_linear_from_segments(segments):
    """Weighted linear fit: x = m*y + b."""
    if not segments:
        return None, None

    ys = []
    xs = []
    weights = []

    for seg in segments:
        ys.extend([float(seg.y1), float(seg.y2)])
        xs.extend([float(seg.x1), float(seg.x2)])

        w = max(float(seg.length), 1.0)
        weights.extend([w, w])

    if len(ys) < 2 or np.ptp(ys) < 1e-6:
        return None, None

    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    A = np.column_stack((ys, np.ones_like(ys)))
    W = np.sqrt(weights)[:, None]

    try:
        coeffs, _, _, _ = np.linalg.lstsq(
            A * W,
            xs * W[:, 0],
            rcond=None
        )
    except np.linalg.LinAlgError:
        return None, None

    predicted = coeffs[0] * ys + coeffs[1]
    rms = float(np.sqrt(np.mean((xs - predicted) ** 2)))

    return coeffs, rms


def confidence_from_fit(segments, rms):
    if not segments or rms is None:
        return 0.0

    total_length = sum(float(s.length) for s in segments)

    support = min(total_length / 400.0, 1.0)
    error_score = max(0.0, min(1.0, 1.0 - rms / 30.0))

    return float(0.5 * support + 0.5 * error_score)


def x_at(coeffs, y):
    if coeffs is None:
        return None
    return float(coeffs[0] * y + coeffs[1])


def classify_pair(left_coeffs, right_coeffs, h, w):
    """
    Diagnostic only.

    Reports the exact geometry at several y positions and
    identifies why the pair would fail a basic geometry check.
    """
    if left_coeffs is None or right_coeffs is None:
        return {
            "ok": False,
            "reason": "missing_side",
            "rows": []
        }

    ys = np.array([
        int(h * 0.62),
        int(h * 0.68),
        int(h * 0.74),
        int(h * 0.80),
        int(h * 0.86),
        int(h * 0.92),
    ])

    rows = []

    for y in ys:
        lx = x_at(left_coeffs, y)
        rx = x_at(right_coeffs, y)
        width = rx - lx

        rows.append((int(y), lx, rx, width))

    widths = np.array([r[3] for r in rows], dtype=np.float64)

    # Check crossing first.
    if np.any(widths <= 0):
        return {
            "ok": False,
            "reason": "crossing",
            "rows": rows
        }

    # Keep the same broad limits used by the previous
    # temporary diagnostic, but report which one fails.
    min_width = float(widths.min())
    max_width = float(widths.max())

    if min_width < 40:
        return {
            "ok": False,
            "reason": f"width_min={min_width:.1f}",
            "rows": rows
        }

    if max_width > 650:
        return {
            "ok": False,
            "reason": f"width_max={max_width:.1f}",
            "rows": rows
        }

    # Frame-bound checks.
    left_xs = np.array([r[1] for r in rows])
    right_xs = np.array([r[2] for r in rows])

    if np.any(left_xs < -100) or np.any(left_xs > w + 100):
        return {
            "ok": False,
            "reason": "left_out_of_frame",
            "rows": rows
        }

    if np.any(right_xs < -100) or np.any(right_xs > w + 100):
        return {
            "ok": False,
            "reason": "right_out_of_frame",
            "rows": rows
        }

    return {
        "ok": True,
        "reason": "ok",
        "rows": rows
    }


def main():
    ap = argparse.ArgumentParser(
        description="Temporary lane-pair geometry diagnostic"
    )

    ap.add_argument("--input", required=True)
    ap.add_argument("--start-frame", type=int, default=3456)
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--outdir",
        default="outputs/pair_geometry_check"
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
        lane_color,
        reinforce_with_hsv=True
    )

    hough = LaneHough(
        HoughConfig.from_dict(cfg.get("hough", {}))
    )

    state = LaneState(
        StateConfig.from_dict(cfg.get("lane_state", {}))
    )

    video = VideoReader(args.input)

    fps = video.info.fps

    end_frame = min(
        args.start_frame + args.num_frames,
        video.info.frame_count
    )

    actual_frames = end_frame - args.start_frame

    print()
    print("=" * 95)
    print("TEMPORARY LANE-PAIR GEOMETRY DIAGNOSTIC")
    print("=" * 95)
    print(f"Input          : {args.input}")
    print(f"Start frame    : {args.start_frame}")
    print(f"Frames         : {actual_frames}")
    print(f"FPS            : {fps:.2f}")
    print()
    print("This test does NOT modify permanent project code.")
    print()
    print("Lane equation: x = m*y + b")
    print()

    csv_path = os.path.join(
        args.outdir,
        "pair_geometry_results.csv"
    )

    csv = open(csv_path, "w", encoding="utf-8")

    csv.write(
        "frame,time,horizon_y,"
        "left_m,left_b,left_rms,left_conf,"
        "right_m,right_b,right_rms,right_conf,"
        "pair_ok,pair_reason,"
        "y0,left0,right0,width0,"
        "y1,left1,right1,width1,"
        "y2,left2,right2,width2,"
        "y3,left3,right3,width3,"
        "y4,left4,right4,width4,"
        "y5,left5,right5,width5\n"
    )

    reason_counts = {}
    pair_ok_count = 0
    processed = 0

    try:
        for frame_idx in range(args.start_frame, end_frame):

            frame = video.read_frame(frame_idx)

            if frame is None:
                continue

            processed += 1

            h, w = frame.shape[:2]
            timestamp = frame_idx / fps

            horizon_y = horizon.detect(frame)

            edges_roi, _, _ = edges_module.compute(
                frame,
                top_y_override=horizon_y
            )

            left_segments, right_segments, _ = hough.classify(
                edges_roi
            )

            left_coeffs, left_rms = fit_linear_from_segments(
                left_segments
            )

            right_coeffs, right_rms = fit_linear_from_segments(
                right_segments
            )

            left_conf = confidence_from_fit(
                left_segments,
                left_rms
            )

            right_conf = confidence_from_fit(
                right_segments,
                right_rms
            )

            (
                (left_smoothed, left_status, left_state_conf),
                (right_smoothed, right_status, right_state_conf)
            ) = state.update(
                left_coeffs,
                left_conf,
                right_coeffs,
                right_conf
            )

            result = classify_pair(
                left_smoothed,
                right_smoothed,
                h,
                w
            )

            pair_ok = result["ok"]
            reason = result["reason"]
            rows = result["rows"]

            if pair_ok:
                pair_ok_count += 1

            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            print()
            print(
                f"FRAME {frame_idx} | "
                f"L={left_status} "
                f"R={right_status} "
                f"PAIR={pair_ok} "
                f"REASON={reason}"
            )

            print(
                "  "
                "Y       LEFT X     RIGHT X     WIDTH"
            )

            for y, lx, rx, width in rows:
                print(
                    f"  "
                    f"{y:3d}   "
                    f"{lx:8.1f}   "
                    f"{rx:9.1f}   "
                    f"{width:8.1f}"
                )

            # CSV
            csv_values = [
                str(frame_idx),
                f"{timestamp:.4f}",
                "" if horizon_y is None else str(horizon_y),

                "" if left_coeffs is None else f"{left_coeffs[0]:.8f}",
                "" if left_coeffs is None else f"{left_coeffs[1]:.4f}",
                "" if left_rms is None else f"{left_rms:.6f}",
                f"{left_state_conf:.6f}",

                "" if right_coeffs is None else f"{right_coeffs[0]:.8f}",
                "" if right_coeffs is None else f"{right_coeffs[1]:.4f}",
                "" if right_rms is None else f"{right_rms:.6f}",
                f"{right_state_conf:.6f}",

                str(int(pair_ok)),
                reason,
            ]

            for y, lx, rx, width in rows:
                csv_values.extend([
                    str(y),
                    f"{lx:.3f}",
                    f"{rx:.3f}",
                    f"{width:.3f}",
                ])

            # If the pair is missing, pad the six geometry groups.
            while len(csv_values) < 37:
                csv_values.append("")

            csv.write(",".join(csv_values) + "\n")

    finally:
        csv.close()
        video.release()

    print()
    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"Frames processed : {processed}")
    print(f"Pair TRUE        : {pair_ok_count}/{processed}")
    print()
    print("PAIR FAILURE / RESULT REASONS:")

    for reason, count in sorted(reason_counts.items()):
        print(
            f"  {reason:25s}: {count}"
        )

    print()
    print(f"CSV              : {csv_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
