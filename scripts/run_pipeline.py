
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time as _time

import cv2
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Stage 1
from src.artifact_mask import ArtifactMask, ArtifactMaskConfig
from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.lane_state import LaneState, StateConfig

# Stage 2
from src.ego_position import EgoPosition, EgoConfig
from src.csv_writers import EgoPositionCSV, LaneChangesCSV, SignsCSV

# Stage 3
from src.lane_change import LaneChangeDetector, LaneChangeConfig

# Stage 4
try:
    from src.sign_detect import SignDetector, SignDetectConfig
    from src.sign_track import SignTracker, SignTrackConfig
    from src.sign_categories import category_of
    _HAVE_SIGNS = True
except Exception as _e:
    _HAVE_SIGNS = False
    print(f"[warn] sign detection disabled: {_e}")

from src.io_video import VideoReader


def load_yaml(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_video(src: str, outdir: str) -> str:
    """Run scripts/normalize_video.py to produce a 720x576@25 input."""
    os.makedirs(outdir, exist_ok=True)
    norm_path = os.path.join(
        outdir,
        os.path.splitext(os.path.basename(src))[0] + "_720x576_25fps.mp4"
    )
    if os.path.isfile(norm_path):
        print(f"[pipeline] normalized video already exists: {norm_path}")
        return norm_path

    print(f"[pipeline] normalizing {src} -> {norm_path}")
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts", "normalize_video.py"),
        "--input", src,
        "--output", norm_path,
    ]
    subprocess.run(cmd, check=True)
    return norm_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="any video file")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--sign-every", type=int, default=5)
    ap.add_argument("--debug-video", action="store_true",
                    help="write annotated.mp4")
    ap.add_argument("--skip-normalize", action="store_true",
                    help="assume input is already 720x576@25")
    args = ap.parse_args()

    t_start = _time.time()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- normalize ----------
    if args.skip_normalize:
        proc_input = args.input
    else:
        proc_input = normalize_video(args.input, args.outdir)

    # ---------- build pipeline ----------
    cfg = load_yaml(args.config)

    am = ArtifactMask(ArtifactMaskConfig.from_dict(cfg.get("artifact_mask", {})))
    hz = HorizonDetector(HorizonConfig.from_dict(cfg.get("horizon", {})))
    roi = LaneROI(RoiConfig.from_dict(cfg.get("roi", {})))
    lc = LaneColor(LaneColorConfig.from_dict(cfg.get("lane_color", {})))
    edges_mod = LaneEdges(CannyConfig.from_dict(cfg.get("canny", {})),
                          roi, lc, reinforce_with_hsv=True)
    fitter = LaneFitter(SlidingWindowConfig.from_dict(cfg.get("sliding_window", {})))
    validator = LaneValidation(ValidationConfig.from_dict(cfg.get("validation", {})))
    state = LaneState(StateConfig.from_dict(cfg.get("lane_state", {})))
    ego = EgoPosition(EgoConfig.from_dict(cfg.get("ego_position", {})))
    lc_det = LaneChangeDetector(
        LaneChangeConfig.from_dict(cfg.get("lane_change", {}))
    )

    sign_det = None
    sign_trk = None
    if _HAVE_SIGNS and cfg.get("sign_detect", {}).get("enabled", True):
        try:
            sign_det = SignDetector(SignDetectConfig.from_dict(cfg.get("sign_detect", {})))
            sign_trk = SignTracker(SignTrackConfig.from_dict(cfg.get("sign_track", {})))
            print("[pipeline] sign detection ENABLED")
        except Exception as e:
            print(f"[warn] sign detector failed to init: {e}")

    # ---------- CSV writers ----------
    ego_csv = EgoPositionCSV(os.path.join(args.outdir, "ego_position.csv"))
    lc_csv = LaneChangesCSV(os.path.join(args.outdir, "lane_changes.csv"))
    sign_csv = SignsCSV(os.path.join(args.outdir, "signs.csv"))

    # ---------- video reader ----------
    vr = VideoReader(proc_input)
    fps = vr.info.fps
    sample_step = int(round(fps))
    print(f"[pipeline] fps={fps:.2f} sample_step={sample_step}")

    # ---------- writer ----------
    writer = None
    if args.debug_video:
        vpath = os.path.join(args.outdir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vpath, fourcc, fps,
                                 (int(vr.info.width), int(vr.info.height)))
        print(f"[pipeline] writing debug video to {vpath}")

    # ---------- main loop ----------
    n = 0
    for idx, ts, frame in vr.iter_frames(step=1):
        fmasked = am.apply(frame)
        horizon_y = hz.detect(fmasked)
        edges_roi, _, _ = edges_mod.compute(fmasked, top_y_override=horizon_y)
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

        # 1 Hz sample
        if idx % sample_step == 0:
            m = ego.compute(l_coeffs, r_coeffs, w, h)
            off_px = m.offset_px
            lane_w = m.lane_width_px
            off_norm = (off_px / lane_w) if (off_px is not None and lane_w) else None
            conf = float(min(l_conf, r_conf)) if m.valid else 0.0
            status = "OK" if m.valid else "MISS"
            ego_csv.row(
                f"{ts:.2f}", idx,
                f"{off_px:.2f}" if off_px is not None else "",
                f"{lane_w:.1f}" if lane_w is not None else "",
                f"{off_norm:.3f}" if off_norm is not None else "",
                f"{conf:.2f}", status,
            )
            ev = lc_det.feed(idx, ts, off_norm, status)
            if ev is not None:
                lc_csv.row(f"{ev.timestamp_s:.2f}", ev.frame,
                           ev.direction, f"{ev.magnitude:.3f}")

        # 5 Hz sign detection
        if sign_det is not None and idx % args.sign_every == 0:
            dets = sign_det.detect(frame)
            sign_trk.update(idx, ts, dets)

        # optional debug video
        if writer is not None:
            vis = frame.copy()
            ys = np.linspace(y_range[0], y_range[1], 80)
            if l_coeffs is not None:
                xs = l_coeffs[0]*ys*ys + l_coeffs[1]*ys + l_coeffs[2]
                pts = np.stack([xs, ys], 1).astype(np.int32)
                pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
                if len(pts) >= 2:
                    cv2.polylines(vis, [pts], False, (0, 255, 255), 3)
            if r_coeffs is not None:
                xs = r_coeffs[0]*ys*ys + r_coeffs[1]*ys + r_coeffs[2]
                pts = np.stack([xs, ys], 1).astype(np.int32)
                pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
                if len(pts) >= 2:
                    cv2.polylines(vis, [pts], False, (0, 255, 0), 3)
            if sign_trk is not None:
                vis = sign_trk.debug_render(vis, None)
            txt = f"L={l_status} R={r_status}"
            cv2.putText(vis, txt, (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, txt, (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(vis)

        n += 1
        if args.max_frames and n >= args.max_frames:
            break

    # finalize
    if sign_trk is not None:
        for tr in sign_trk.finalize():
            sign_csv.row(
                f"{tr.first_ts:.2f}", tr.first_frame,
                tr.cls_name, category_of(tr.cls_name),
                tr.bbox[0], tr.bbox[1], tr.bbox[2], tr.bbox[3],
                f"{tr.confidence:.2f}",
            )
    ego_csv.close()
    lc_csv.close()
    sign_csv.close()
    if writer is not None:
        writer.release()

    dt = _time.time() - t_start
    print()
    print(f"[pipeline] processed {n} frames in {dt:.1f}s ({n/dt:.1f} fps)")
    print(f"[pipeline] outputs in {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()