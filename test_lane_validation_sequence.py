from __future__ import annotations

import argparse, os, sys
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
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) &
              (pts[:, 1] >= 0) & (pts[:, 1] < h)].astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Temporary quadratic lane-validation sequence diagnostic.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--start-frame", type=int, default=3456)
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs/lane_validation_sequence_check")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config(args.config)

    horizon = HorizonDetector(HorizonConfig.from_dict(cfg.get("horizon", {})))
    lane_color = LaneColor(LaneColorConfig.from_dict(cfg.get("lane_color", {})))
    roi = LaneROI(RoiConfig.from_dict(cfg.get("roi", {})))
    edges = LaneEdges(CannyConfig.from_dict(cfg.get("canny", {})), roi, lane_color, True)
    fitter = LaneFitter(SlidingWindowConfig.from_dict(cfg.get("sliding_window", {})))
    validator = LaneValidation(ValidationConfig.from_dict(cfg.get("validation", {})))
    video = VideoReader(args.input)

    print("=" * 120)
    print("TEMPORARY QUADRATIC LANE-VALIDATION SEQUENCE DIAGNOSTIC")
    print("=" * 120)
    print(f"Input       : {args.input}")
    print(f"Start frame : {args.start_frame}")
    print(f"Frames      : {args.num_frames}")
    print(f"Resolution  : {video.info.width} x {video.info.height}")
    print("This test does NOT modify permanent project code.")
    print()
    print("VALIDATION CONFIG")
    print(f"max_sweep_frac    : {validator.cfg.max_sweep_frac}")
    print(f"min_lane_width_px : {validator.cfg.min_lane_width_px}")
    print(f"max_lane_width_px : {validator.cfg.max_lane_width_px}")
    print()
    print("frame | Lfit Rfit | Lok Rok Pair | Lreason Rreason PairReason | Wmin Wmax")
    print("-" * 120)

    previous_left = None
    previous_right = None
    rows = []
    visual_count = 0

    for idx, ts, frame in video.iter_frames(start=args.start_frame, end=args.start_frame + args.num_frames, step=1):
        h, w = frame.shape[:2]
        hy = horizon.detect(frame)
        edge_mask, _, _ = edges.compute(frame, top_y_override=hy)
        left, right = fitter.fit(edge_mask, previous_left=previous_left, previous_right=previous_right)

        y_min = max(int(h * 0.62), int(hy + 10)) if hy is not None else int(h * 0.62)
        y_range = (y_min, int(h * 0.95))
        vr = validator.validate(left.coeffs, right.coeffs, y_range, w)

        wmin = wmax = None
        if vr.left_ok and vr.right_ok:
            ys = np.linspace(y_range[0], y_range[1], 50)
            widths = np.polyval(right.coeffs, ys) - np.polyval(left.coeffs, ys)
            wmax = float(widths.max())
            wmin = float(widths[int(len(widths) * 0.30):].min())

        print(f"{idx:5d} | {str(left.valid):4s} {str(right.valid):4s} | "
              f"{str(vr.left_ok):3s} {str(vr.right_ok):3s} {str(vr.pair_ok):4s} | "
              f"{vr.reason_left or 'ok':12s} {vr.reason_right or 'ok':12s} {vr.reason_pair:14s} | "
              f"{'' if wmin is None else f'{wmin:.1f}':>5s} {'' if wmax is None else f'{wmax:.1f}':>5s}")

        rows.append((idx, left.valid, right.valid, vr.left_ok, vr.right_ok, vr.pair_ok,
                     vr.reason_left, vr.reason_right, vr.reason_pair, wmin, wmax))

        if left.coeffs is not None:
            previous_left = left.coeffs.copy()
        if right.coeffs is not None:
            previous_right = right.coeffs.copy()

        if visual_count == 0 or visual_count % 10 == 0:
            vis = frame.copy()
            if hy is not None:
                cv2.line(vis, (0, int(hy)), (w - 1, int(hy)), (255, 255, 0), 2)
            draw_polyline(vis, left.coeffs, *y_range, (0, 255, 255), 4)
            draw_polyline(vis, right.coeffs, *y_range, (0, 255, 0), 4)
            text = f"FRAME {idx} | L={vr.left_ok} R={vr.right_ok} PAIR={vr.pair_ok}"
            cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.outdir, f"validation_{idx:06d}.png"), vis)
        visual_count += 1

    video.release()

    csv_path = os.path.join(args.outdir, "lane_validation_sequence_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,left_fit,right_fit,left_ok,right_ok,pair_ok,reason_left,reason_right,reason_pair,wmin,wmax\n")
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},"{r[6]}","{r[7]}","{r[8]}",{r[9] if r[9] is not None else ""},{r[10] if r[10] is not None else ""}\n')

    n = len(rows)
    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Frames processed : {n}")
    print(f"Left fit         : {sum(r[1] for r in rows)}/{n}")
    print(f"Right fit        : {sum(r[2] for r in rows)}/{n}")
    print(f"Left valid       : {sum(r[3] for r in rows)}/{n}")
    print(f"Right valid      : {sum(r[4] for r in rows)}/{n}")
    print(f"Pair valid       : {sum(r[5] for r in rows)}/{n}")

    reasons = {}
    for r in rows:
        if not r[5]:
            reasons[r[8]] = reasons.get(r[8], 0) + 1
    print("\nPAIR FAILURE REASONS:")
    if reasons:
        for k, v in sorted(reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")
    else:
        print("  None")

    print(f"\nCSV     : {csv_path}")
    print(f"Visuals : {args.outdir}")
    print("=" * 120)


if __name__ == "__main__":
    main()