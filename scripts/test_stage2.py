from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.lane_state import LaneState, StateConfig
from src.ego_position import EgoPosition, EgoConfig
from src.csv_writers import EgoPositionCSV
from src.io_video import VideoReader


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="process only first N frames (for quick tests)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_yaml(args.config)

    hz = HorizonDetector(HorizonConfig.from_dict(cfg.get("horizon", {})))
    roi = LaneROI(RoiConfig.from_dict(cfg.get("roi", {})))
    lc = LaneColor(LaneColorConfig.from_dict(cfg.get("lane_color", {})))
    edges_mod = LaneEdges(CannyConfig.from_dict(cfg.get("canny", {})),
                          roi, lc, reinforce_with_hsv=True)
    fitter = LaneFitter(SlidingWindowConfig.from_dict(cfg.get("sliding_window", {})))
    validator = LaneValidation(ValidationConfig.from_dict(cfg.get("validation", {})))
    state = LaneState(StateConfig.from_dict(cfg.get("lane_state", {})))
    ego = EgoPosition(EgoConfig.from_dict(cfg.get("ego_position", {})))

    MIN_CONF = float(cfg.get("lane_state", {}).get("min_confidence", 0.15))

    csv_path = os.path.join(args.outdir, "ego_position.csv")
    csv_writer = EgoPositionCSV(csv_path)

    vr = VideoReader(args.input)
    fps = vr.info.fps
    sample_step = int(round(fps))
    print(f"[info] fps={fps} sample every {sample_step} frames")

    n_processed = 0
    n_rows = 0
    for idx, ts, frame in vr.iter_frames(step=1):
        horizon_y = hz.detect(frame)
        edges_roi, _, _ = edges_mod.compute(frame, top_y_override=horizon_y)

        left, right = fitter.fit(edges_roi)
        h, w = frame.shape[:2]
        y_range = (int(h * 0.55), int(h * 0.95))
        v = validator.validate(left.coeffs, right.coeffs, y_range, w)

        (l_coeffs, l_status, l_conf), (r_coeffs, r_status, r_conf) = state.update(
            left.coeffs if v.left_ok else None,
            left.confidence if v.left_ok else 0.0,
            right.coeffs if v.right_ok else None,
            right.confidence if v.right_ok else 0.0,
        )

        if idx % sample_step == 0:
            m = ego.compute(l_coeffs, r_coeffs, w, h)

            l_usable = (l_coeffs is not None) and (l_status in ("OK", "HOLD"))
            r_usable = (r_coeffs is not None) and (r_status in ("OK", "HOLD"))
            both_usable = l_usable and r_usable

            min_conf = float(min(l_conf, r_conf)) if both_usable else 0.0

            if both_usable and m.valid and min_conf >= MIN_CONF:
                off_px = m.offset_px
                lane_w = m.lane_width_px
                off_norm = (off_px / lane_w) if lane_w else None
                status = "OK"
            else:
                off_px = lane_w = off_norm = None
                status = "MISS"

            # [FIX] 7-column row, no 'side' column
            csv_writer.row(
                f"{ts:.2f}", idx,
                f"{off_px:.2f}" if off_px is not None else "",
                f"{lane_w:.1f}" if lane_w is not None else "",
                f"{off_norm:.3f}" if off_norm is not None else "",
                f"{min_conf:.2f}", status,
            )
            n_rows += 1

        n_processed += 1
        if args.max_frames and n_processed >= args.max_frames:
            break

    csv_writer.close()
    print(f"[ok] wrote {csv_path} ({n_processed} frames processed, {n_rows} csv rows)")


if __name__ == "__main__":
    main()