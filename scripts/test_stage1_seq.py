"""
Stage 1 sequence test — runs the full lane pipeline across N frames
and prints a per-frame status line so we can see the state machine
behave over time.

Usage:
    python scripts/test_stage1_seq.py --input data/VBOX0011_Trim.mp4 --time 900 --n 150
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifact_mask import ArtifactMask, ArtifactMaskConfig
from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.lane_state import LaneState, StateConfig
from src.io_video import VideoReader


def load_config(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--time", type=float, default=900.0,
                    help="start time in seconds")
    ap.add_argument("--n", type=int, default=150,
                    help="number of frames to process")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config(args.config)

    am = ArtifactMask(ArtifactMaskConfig.from_dict(cfg.get("artifact_mask", {})))
    hz = HorizonDetector(HorizonConfig.from_dict(cfg.get("horizon", {})))
    roi = LaneROI(RoiConfig.from_dict(cfg.get("roi", {})))
    lc = LaneColor(LaneColorConfig.from_dict(cfg.get("lane_color", {})))
    edges_mod = LaneEdges(
        CannyConfig.from_dict(cfg.get("canny", {})),
        roi, lc, reinforce_with_hsv=True,
    )
    fitter = LaneFitter(SlidingWindowConfig.from_dict(cfg.get("sliding_window", {})))
    validator = LaneValidation(ValidationConfig.from_dict(cfg.get("validation", {})))
    state = LaneState(StateConfig.from_dict(cfg.get("lane_state", {})))

    vr = VideoReader(args.input)
    fps = vr.info.fps
    start_frame = int(round(args.time * fps))
    end_frame = min(start_frame + args.n, vr.info.frame_count)

    print(f"[seq] fps={fps:.2f} start_frame={start_frame} "
          f"end_frame={end_frame} n={end_frame - start_frame}")
    print(f"     {'frame':>6} {'t':>8}  {'L_status':>9} {'R_status':>9}  "
          f"{'L_px':>5} {'R_px':>5}  {'L_conf':>6} {'R_conf':>6}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = os.path.join(args.outdir, "stage1_seq.mp4")
    writer = None

    t0 = _time.time()
    counts = {"L_OK": 0, "L_HOLD": 0, "L_MISS": 0,
              "R_OK": 0, "R_HOLD": 0, "R_MISS": 0}

    for idx, ts, frame in vr.iter_frames(start=start_frame, end=end_frame, step=1):
        f_masked = am.apply(frame)
        horizon_y = hz.detect(f_masked)
        edges_roi, _, _ = edges_mod.compute(f_masked, top_y_override=horizon_y)

        left, right = fitter.fit(edges_roi, base_left=None, base_right=None)
        h, w = frame.shape[:2]
        y_range = (int(h * 0.55), int(h * 0.95))
        v = validator.validate(left.coeffs, right.coeffs, y_range, w)

        (l_coeffs, l_status, l_conf), (r_coeffs, r_status, r_conf) = state.update(
            left.coeffs if v.left_ok else None, left.confidence if v.left_ok else 0.0,
            right.coeffs if v.right_ok else None, right.confidence if v.right_ok else 0.0,
        )

        counts[f"L_{l_status}"] += 1
        counts[f"R_{r_status}"] += 1

        print(f"     {idx:>6} {ts:>8.2f}  {l_status:>9} {r_status:>9}  "
              f"{left.n_pixels:>5} {right.n_pixels:>5}  "
              f"{l_conf:>6.2f} {r_conf:>6.2f}")

        vis = frame.copy()
        ys = np.linspace(y_range[0], y_range[1], 80)
        if l_coeffs is not None:
            xs = l_coeffs[0] * ys * ys + l_coeffs[1] * ys + l_coeffs[2]
            pts = np.stack([xs, ys], 1).astype(np.int32)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], False, (0, 255, 255), 3)
        if r_coeffs is not None:
            xs = r_coeffs[0] * ys * ys + r_coeffs[1] * ys + r_coeffs[2]
            pts = np.stack([xs, ys], 1).astype(np.int32)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], False, (0, 255, 0), 3)
        cv2.putText(vis, f"L={l_status} R={r_status}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, f"L={l_status} R={r_status}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if writer is None:
            writer = cv2.VideoWriter(out_video, fourcc, fps,
                                     (vis.shape[1], vis.shape[0]))
        writer.write(vis)

    if writer is not None:
        writer.release()

    dt = _time.time() - t0
    n = end_frame - start_frame
    print()
    print(f"[seq] done in {dt:.2f}s  ({n / dt:.1f} fps processing)")
    print(f"[seq] L counts: OK={counts['L_OK']} HOLD={counts['L_HOLD']} MISS={counts['L_MISS']}")
    print(f"[seq] R counts: OK={counts['R_OK']} HOLD={counts['R_HOLD']} MISS={counts['R_MISS']}")
    print(f"[seq] wrote {out_video}")


if __name__ == "__main__":
    main()