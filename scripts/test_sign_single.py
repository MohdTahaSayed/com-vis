"""
One-shot sign detection on a single frame/image using the trained Indian model.
Usage:
    python scripts/test_sign_single.py --image frames/frame_t0040.00_f1000.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sign_detect import SignDetector, SignDetectConfig


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--conf", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_yaml(args.config)

    det_cfg = SignDetectConfig.from_dict(cfg.get("sign_detect", {}))
    if args.conf is not None:
        det_cfg.conf_threshold = args.conf

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)

    detector = SignDetector(det_cfg)

    print(f"[info] image: {args.image}  shape={frame.shape}")
    print(f"[info] model: {det_cfg.weights}  ({len(detector.class_names)} classes)")

    # --- low-confidence scan to see everything the model finds ---
    print(f"\n[low-conf scan at conf=0.05]")
    low_cfg = SignDetectConfig.from_dict(det_cfg.__dict__)
    low_cfg.conf_threshold = 0.05
    low_detector = SignDetector(low_cfg)
    low_dets = low_detector.detect(frame)
    if not low_dets:
        print("  (nothing above 0.05 either)")
    for d in low_dets:
        print(f"  {d.cls_name:42s}  conf={d.confidence:.3f}  bbox={d.bbox}")

    # --- run at configured threshold ---
    dets = detector.detect(frame)
    print(f"\n[configured conf={det_cfg.conf_threshold}]")
    if not dets:
        print("  (nothing above configured threshold)")
    for d in dets:
        print(f"  {d.cls_name:42s}  conf={d.confidence:.3f}  bbox={d.bbox}")

    vis = detector.debug_render(frame, low_dets)
    out = os.path.join(args.outdir, "sign_single.png")
    cv2.imwrite(out, vis)
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()