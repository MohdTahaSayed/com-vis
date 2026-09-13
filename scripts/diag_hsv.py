"""
Diagnostic: prints actual HSV values and component stats for one frame.
Usage:
    python scripts/diag_hsv.py --input data/VBOX0011_Trim.mp4 --time 900
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.io_video import VideoReader
from src.artifact_mask import ArtifactMask, ArtifactMaskConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_components import LaneComponents, ComponentsConfig
from src.lane_roi import LaneROI, RoiConfig


def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--time", type=float, required=True)
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    vr = VideoReader(args.input)
    idx = int(round(args.time * vr.info.fps))
    frame = vr.read_frame(idx)
    vr.release()
    h, w = frame.shape[:2]

    am = ArtifactMask(ArtifactMaskConfig.from_dict(cfg.get("artifact_mask", {})))
    lc = LaneColor(LaneColorConfig.from_dict(cfg.get("lane_color", {})))

    fmasked = am.apply(frame)
    hsv = cv2.cvtColor(fmasked, cv2.COLOR_BGR2HSV)
    white, yellow, union = lc.masks(fmasked)

    print(f"\n=== Frame {idx}, t={args.time}s, {w}x{h} ===")
    print(f"Config: white s_max={lc.cfg.white.s_max} v_min={lc.cfg.white.v_min}")

    samples = [
        ("left-edge-road",  0.18, 0.80),
        ("left-edge-road2", 0.22, 0.75),
        ("center-dash",     0.50, 0.82),
        ("right-edge-road", 0.82, 0.80),
        ("shoulder-left",   0.08, 0.75),
        ("shoulder-right",  0.92, 0.75),
        ("sky",             0.50, 0.20),
    ]

    print("\n--- HSV samples at chosen pixels ---")
    for label, xf, yf in samples:
        x, y = int(xf * w), int(yf * h)
        b, g, r = frame[y, x]
        hh, ss, vv = hsv[y, x]
        passed = "YES" if white[y, x] > 0 else "no"
        print(f"  {label:18s} ({x:3d},{y:3d})  BGR=({b:3d},{g:3d},{r:3d})"
              f"  HSV=({hh:3d},{ss:3d},{vv:3d})  in white mask: {passed}")

    print("\n--- Connected components (raw HSV) ---")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    fa = h * w
    print(f"  {'#':>3} {'x':>4} {'y':>4} {'w':>4} {'h':>4} {'area':>6}"
          f" {'a%':>6} {'aspect':>7} {'fill':>5}")
    rows = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 30:
            continue
        rows.append((i, x, y, bw, bh, area, area / fa * 100,
                     bw / max(bh, 1), area / max(bw * bh, 1)))
    rows.sort(key=lambda r: -r[5])
    for i, x, y, bw, bh, area, af, asp, fill in rows[:25]:
        print(f"  {i:>3} {x:>4} {y:>4} {bw:>4} {bh:>4} {area:>6}"
              f" {af:>6.3f} {asp:>7.2f} {fill:>5.2f}")

    lcomp = LaneComponents(ComponentsConfig.from_dict(cfg.get("components", {})))
    roi = LaneROI(RoiConfig.from_dict(cfg.get("roi", {})))
    comp_mask, kept = lcomp.filter(union)
    final = roi.apply(comp_mask)
    print(f"\n  Raw union pixels : {int((union>0).sum())}")
    print(f"  After components : {int((comp_mask>0).sum())} (kept {len(kept)})")
    print(f"  After ROI        : {int((final>0).sum())}")


if __name__ == "__main__":
    main()