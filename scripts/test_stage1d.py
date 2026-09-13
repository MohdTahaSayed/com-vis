"""
Stage 1d visual verification: sliding-window polynomial fit.

Pipeline (per frame):
    artifact mask → HSV → components → ROI → fit(left, right) → validation

Renders side-by-side: [fit overlay | final mask + fit]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from src.artifact_mask import ArtifactMask, ArtifactMaskConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_components import LaneComponents, ComponentsConfig
from src.lane_roi import LaneROI, RoiConfig
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


def stack_two(a, b, label_a, label_b):

    combo = np.hstack([a, b])

    for label, xoff in [
        (label_a, 0),
        (label_b, a.shape[1])
    ]:

        cv2.putText(
            combo,
            label,
            (xoff + 8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )

        cv2.putText(
            combo,
            label,
            (xoff + 8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return combo


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True
    )

    ap.add_argument(
        "--frame",
        type=int,
        default=None
    )

    ap.add_argument(
        "--time",
        type=float,
        default=None
    )

    ap.add_argument(
        "--config",
        default="config/default.yaml"
    )

    ap.add_argument(
        "--outdir",
        default="outputs"
    )

    args = ap.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    am = ArtifactMask(
        ArtifactMaskConfig.from_dict(
            cfg.get("artifact_mask", {})
        )
    )

    lc = LaneColor(
        LaneColorConfig.from_dict(
            cfg.get("lane_color", {})
        )
    )

    lcomp = LaneComponents(
        ComponentsConfig.from_dict(
            cfg.get("components", {})
        )
    )

    roi = LaneROI(
        RoiConfig.from_dict(
            cfg.get("roi", {})
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

    # Pipeline
    f_masked = am.apply(frame)

    white, _yellow, union = lc.masks(f_masked)

    comp_mask, _kept = lcomp.filter(union)

    final_mask = roi.apply(comp_mask)

    # Lane fitting
    left, right = fitter.fit(final_mask)

    h, w = frame.shape[:2]

    # UPDATED FIX 4:
    # Validate only useful lane region
    y_range = (
        int(h * 0.58),
        int(h * 0.92)
    )

    v = validator.validate(
        left.coeffs,
        right.coeffs,
        y_range,
        w
    )

    # --------------------------------------------------
    # Left panel: fit overlay on original frame
    # --------------------------------------------------

    p1 = fitter.debug_render(
        frame,
        left,
        right
    )

    txt = (
        f"L:{v.left_ok}({v.reason_left})  "
        f"R:{v.right_ok}({v.reason_right})  "
        f"pair:{v.pair_ok}({v.reason_pair})"
    )

    cv2.putText(
        p1,
        txt,
        (8, 552),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        p1,
        txt,
        (8, 552),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------
    # Right panel: mask + fitted curves
    # --------------------------------------------------

    p2 = cv2.cvtColor(
        final_mask,
        cv2.COLOR_GRAY2BGR
    )

    # UPDATED FIX 4:
    # Render only inside validation range
    ys = np.linspace(
        y_range[0],
        y_range[1],
        60
    )

    for res, color in (
        (left, (0, 255, 255)),
        (right, (0, 255, 0))
    ):

        if res.coeffs is None:
            continue

        xs = (
            res.coeffs[0] * ys * ys +
            res.coeffs[1] * ys +
            res.coeffs[2]
        )

        pts = np.stack(
            [xs, ys],
            axis=1
        ).astype(np.int32)

        pts = pts[
            (pts[:, 0] >= 0) &
            (pts[:, 0] < w)
        ]

        if len(pts) >= 2:

            cv2.polylines(
                p2,
                [pts],
                False,
                color,
                2
            )

    # --------------------------------------------------
    # Combine panels
    # --------------------------------------------------

    combo = stack_two(
        p1,
        p2,
        f"frame={idx} t={ts:.2f}s",
        "mask + fit"
    )

    out = os.path.join(
        args.outdir,
        "stage1d_verify.png"
    )

    cv2.imwrite(
        out,
        combo
    )

    print(
        f"[info] frame={idx} t={ts:.2f}s"
    )

    print(
        f"[info] L n_pixels={left.n_pixels} "
        f"conf={left.confidence:.2f} "
        f"ok={v.left_ok}({v.reason_left})"
    )

    print(
        f"[info] R n_pixels={right.n_pixels} "
        f"conf={right.confidence:.2f} "
        f"ok={v.right_ok}({v.reason_right})"
    )

    print(
        f"[info] pair ok={v.pair_ok}"
        f"({v.reason_pair})"
    )

    print(
        f"[ok] wrote {out}"
    )


if __name__ == "__main__":
    main()