"""
Stage 0 visual verification.

Loads a frame, applies the fixed artifact mask, writes side-by-side PNG:
    [original + bbox | masked + bbox]

Usage:
    python scripts/test_stage0.py --input data/VBOX0011_Trim.mp4 --frame 100
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifact_mask import ArtifactMask, ArtifactMaskConfig
from src.io_video import VideoReader, side_by_side


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_frame(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {args.image}")
        return frame, 0, 0.0

    vr = VideoReader(args.input)
    if args.time is not None:
        idx = int(round(args.time * vr.info.fps))
    else:
        idx = int(args.frame)
    frame = vr.read_frame(idx)
    if frame is None:
        raise RuntimeError(f"Cannot read frame {idx} from {args.input}")
    fps = vr.info.fps
    vr.release()
    return frame, idx, idx / fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="path to video")
    ap.add_argument("--image", help="path to a single PNG/JPG")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_config(args.config)
    am_cfg = ArtifactMaskConfig.from_dict(cfg.get("artifact_mask", {}))
    mask = ArtifactMask(am_cfg)

    frame, idx, ts = get_frame(args)

    masked = mask.apply(frame)
    overlay = mask.debug_render(frame, fill=True)
    masked_vis = mask.debug_render(masked, fill=False)

    row = side_by_side(
        overlay, masked_vis,
        label_a=f"orig+bbox (frame {idx}, t={ts:.2f}s)",
        label_b="after mask",
    )

    out_path = os.path.join(args.outdir, "stage0_verify.png")
    cv2.imwrite(out_path, row)
    print(f"[info] frame={idx} t={ts:.2f}s  bbox_px={mask._pixel_bbox(frame.shape[1], frame.shape[0])}")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()