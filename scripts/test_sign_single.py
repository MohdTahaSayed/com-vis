"""
One-shot sign detection on a single frame or image.
Usage:
    python scripts/test_sign_single.py --image frames/frame_t0040.00_f01000.png
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
    ap.add_argument("--conf", type=float, default=None,
                    help="override confidence threshold")
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

    # --- run detector with a very low threshold to see EVERYTHING ---
    original_conf = det_cfg.conf_threshold
    det_cfg.conf_threshold = 0.05     # catch everything

    # direct call to YOLO with low conf to see all classes
    results = detector.model.predict(
        source=frame,
        conf=0.05,
        iou=det_cfg.iou_threshold,
        imgsz=det_cfg.imgsz,
        device=det_cfg.device,
        verbose=False,
    )

    print(f"[info] image: {args.image}  shape={frame.shape}")
    print(f"[info] running YOLO with conf=0.05 (show everything)")
    if results and results[0].boxes is not None and len(results[0].boxes):
        r = results[0]
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            cls_name = r.names[cls_id]
            conf = float(r.boxes.conf[i].item())
            xyxy = r.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            print(f"  detected: class={cls_name} (id={cls_id}) "
                  f"conf={conf:.3f}  bbox={xyxy}")
    else:
        print("  NO DETECTIONS AT ALL at conf=0.05")

    # --- now run at configured threshold, only kept classes ---
    detector.cfg.conf_threshold = original_conf
    dets = detector.detect(frame)
    print(f"\n[info] kept-classes detections at conf={original_conf}: {len(dets)}")
    for d in dets:
        print(f"  {d.cls_name} {d.confidence:.3f}  bbox={d.bbox}")

    vis = detector.debug_render(frame, dets)
    out = os.path.join(args.outdir, "sign_single.png")
    cv2.imwrite(out, vis)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()