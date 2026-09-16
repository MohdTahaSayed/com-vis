
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.artifact_mask import ArtifactMask, ArtifactMaskConfig
from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_hough import LaneHough, HoughConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.io_video import VideoReader


def load_config(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_frame(args):
    vr = VideoReader(args.input)

    idx = (
        int(round(args.time * vr.info.fps))
        if args.time is not None
        else args.frame
    )

    frame = vr.read_frame(idx)
    vr.release()

    return frame, idx, idx / 25.0


def label(img, text, color=(0, 255, 255)):
    cv2.putText(
        img,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA
    )

    return img


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_config(args.config)

    am = ArtifactMask(
        ArtifactMaskConfig.from_dict(
            cfg.get("artifact_mask", {})
        )
    )

    hz = HorizonDetector(
        HorizonConfig.from_dict(
            cfg.get("horizon", {})
        )
    )

    roi = LaneROI(
        RoiConfig.from_dict(
            cfg.get("roi", {})
        )
    )

    lc = LaneColor(
        LaneColorConfig.from_dict(
            cfg.get("lane_color", {})
        )
    )

    edges_mod = LaneEdges(
        CannyConfig.from_dict(
            cfg.get("canny", {})
        ),
        roi,
        lc,
        reinforce_with_hsv=True,
    )

    hough = LaneHough(
        HoughConfig.from_dict(
            cfg.get("hough", {})
        )
    )

    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(
            cfg.get("sliding_window", {})
        )
    )

    validator = LaneValidation(
        ValidationConfig.from_dict(
            cfg.get("validation", {})
        )
    )

    # Get frame
    frame, idx, ts = get_frame(args)

    # Apply artifact mask
    f_masked = am.apply(frame)

    # Detect horizon
    horizon_y = hz.detect(f_masked)

    # ================= EDGE PIPELINE =================

    edges_roi, edges_raw, hsv_hits = edges_mod.compute(
        f_masked,
        top_y_override=horizon_y
    )

    # ================= HOUGH =================

    left_segs, right_segs, discard_segs = hough.classify(
        edges_roi
    )

    # Frame dimensions
    h, w = frame.shape[:2]

    # Use histogram base — Hough is used only for segment-count sanity.
    base_left = None
    base_right = None

    # ================= POLYNOMIAL FIT =================

    left, right = fitter.fit(
        edges_roi,
        base_left=base_left,
        base_right=base_right
    )

    # ================= VALIDATION =================

    y_range = (
        int(h * 0.55),
        int(h * 0.95)
    )

    v = validator.validate(
        left.coeffs,
        right.coeffs,
        y_range,
        w
    )

    # ================= PANELS =================

    # Panel 1: frame + ROI + Hough segments
    p1 = frame.copy()

    p1 = roi.debug_render(
        p1,
        top_y_override=horizon_y
    )

    p1 = hough.debug_render(
        p1,
        left_segs,
        right_segs,
        discard_segs
    )

    status = (
        f"L:{v.left_ok} "
        f"R:{v.right_ok} "
        f"pair:{v.pair_ok}"
    )

    cv2.putText(
        p1,
        status,
        (8, 552),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        p1,
        status,
        (8, 552),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    label(
        p1,
        f"t={ts:.1f}s  "
        f"L={len(left_segs)} "
        f"R={len(right_segs)} "
        f"Hough segments"
    )

    # Panel 2: Raw Canny edges
    p2 = cv2.cvtColor(
        edges_raw,
        cv2.COLOR_GRAY2BGR
    )

    label(
        p2,
        "Canny raw (whole frame)"
    )

    # Panel 3: Final ROI edges
    p3 = cv2.cvtColor(
        edges_roi,
        cv2.COLOR_GRAY2BGR
    )

    label(
        p3,
        f"edges ROI ({int((edges_roi > 0).sum())} px)"
    )

    # Panel 4: Polynomial fit
    p4 = frame.copy()

    p4 = fitter.debug_render(
        p4,
        left,
        right
    )

    label(
        p4,
        "poly fit"
    )

    # ================= COMBINE PANELS =================

    combo = np.hstack([
        p1,
        p2,
        p3,
        p4
    ])

    # ================= SAVE OUTPUTS =================

    out = os.path.join(
        args.outdir,
        "stage1_verify.png"
    )

    cv2.imwrite(
        out,
        combo
    )

    cv2.imwrite(
        os.path.join(
            args.outdir,
            "stage1_edges_raw.png"
        ),
        edges_raw
    )

    cv2.imwrite(
        os.path.join(
            args.outdir,
            "stage1_edges_roi.png"
        ),
        edges_roi
    )

    # ================= TERMINAL INFO =================

    print(
        f"[info] frame={idx} "
        f"t={ts:.2f}s "
        f"horizon_y={horizon_y}"
    )

    print(
        f"[info] Hough: "
        f"L={len(left_segs)} "
        f"R={len(right_segs)} "
        f"disc={len(discard_segs)}"
    )

    print(
        f"[info] "
        f"base_left={base_left} "
        f"base_right={base_right} "
        f"(histogram fallback)"
    )

    print(
        f"[info] edges: "
        f"raw={int((edges_raw > 0).sum())} "
        f"roi={int((edges_roi > 0).sum())}"
    )

    print(
        f"[info] fit: "
        f"L_px={left.n_pixels} "
        f"R_px={right.n_pixels} "
        f"L_ok={v.left_ok} "
        f"R_ok={v.right_ok} "
        f"pair={v.pair_ok}"
    )

    print(
        f"[ok] wrote {out}"
    )


if __name__ == "__main__":
    main()