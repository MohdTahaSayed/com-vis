
from __future__ import annotations

import argparse
import os
import sys
import time as _time

import cv2
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sign_detect import SignDetector, SignDetectConfig
from src.sign_track import SignTracker, SignTrackConfig
from src.sign_categories import category_of
from src.csv_writers import SignsCSV
from src.io_video import VideoReader


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--debug-video", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_yaml(args.config)

    det_cfg = SignDetectConfig.from_dict(cfg.get("sign_detect", {}))
    trk_cfg = SignTrackConfig.from_dict(cfg.get("sign_track", {}))

    print(f"[init] sign detect: {vars(det_cfg)}")
    print(f"[init] sign track : {vars(trk_cfg)}")

    detector = SignDetector(det_cfg)
    tracker = SignTracker(trk_cfg)

    csv_path = os.path.join(args.outdir, "signs.csv")
    writer = SignsCSV(csv_path)

    vr = VideoReader(args.input)
    fps = vr.info.fps
    print(f"[info] video fps={fps:.2f}, processing every {args.every}th frame")

    writer_video = None
    if args.debug_video:
        vpath = os.path.join(args.outdir, "signs_debug.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_video = cv2.VideoWriter(vpath, fourcc, fps / args.every,
                                       (int(vr.info.width), int(vr.info.height)))

    n_frames = 0
    n_dets = 0
    t0 = _time.time()
    for idx, ts, frame in vr.iter_frames(step=args.every):
        dets = detector.detect(frame)
        n_dets += len(dets)
        tracker.update(idx, ts, dets)
        n_frames += 1

        if writer_video is not None:
            vis = detector.debug_render(frame, dets)
            writer_video.write(vis)

        if args.max_frames and n_frames >= args.max_frames:
            break

    finalized = tracker.finalize()
    for tr in finalized:
        writer.row(
            f"{tr.first_ts:.2f}", tr.first_frame,
            tr.cls_name, category_of(tr.cls_name),
            tr.bbox[0], tr.bbox[1], tr.bbox[2], tr.bbox[3],
            f"{tr.confidence:.2f}",
        )
    writer.close()
    if writer_video is not None:
        writer_video.release()

    dt = _time.time() - t0
    print()
    print(f"[info] processed {n_frames} frames in {dt:.1f}s "
          f"({n_frames/dt:.1f} fps)")
    print(f"[info] raw detections: {n_dets}")
    print(f"[info] unique signs (confirmed tracks): {len(finalized)}")
    print(f"[ok] wrote {csv_path}")


if __name__ == "__main__":
    main()