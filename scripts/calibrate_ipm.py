"""
One-time IPM calibration.

Interactive helper. Usage:

    python scripts/calibrate_ipm.py --image frames/frame_t0900.00_f22500.png
    python scripts/calibrate_ipm.py --image frames/frame_t0900.00_f22500.png --interactive

Point order (left panel dots):
    0 = near-left   (bottom-left lane edge)
    1 = near-right  (bottom-right lane edge)
    2 = far-right   (right lane edge near horizon)
    3 = far-left    (left lane edge near horizon)
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ipm import IPM, IPMConfig


# --- Default road-plane points, tuned for 720x576 coded frames ---
# These 4 points trace the EGO LANE as a trapezoid on the road plane.
#   near pair (bottom of road, above the hood) is wide
#   far pair  (near horizon) is narrow
DEFAULT_SRC_PTS = [
    [200, 470],   # near-left  (left dashed lane edge at bottom of road)
    [520, 470],   # near-right (right dashed lane edge at bottom of road)
    [390, 330],   # far-right  (right lane edge near horizon)
    [330, 330],   # far-left   (left lane edge near horizon)
]

# --- Target positions in BEV ---
# Ego lane becomes a 300-px-wide, 300-px-tall rectangle.
# meters_per_pixel = 3.5 / 300 (standard lane width 3.5 m).
DEFAULT_DST_PTS = [
    [90,  470],   # near-left
    [390, 470],   # near-right
    [390, 170],   # far-right
    [90,  170],   # far-left
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="config/calibration.yaml")
    ap.add_argument("--meters_per_pixel", type=float, default=3.5 / 300.0,
                    help="3.5 m across 300 BEV px by default")
    ap.add_argument("--interactive", action="store_true",
                    help="pick 4 points by clicking (NOT recommended — easy to misplace)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    src_pts = np.array(DEFAULT_SRC_PTS, dtype=np.float32)
    dst_pts = np.array(DEFAULT_DST_PTS, dtype=np.float32)

    if args.interactive:
        print("Click 4 points in this order: "
              "near-left, near-right, far-right, far-left.")
        clicks = []
        display = img.copy()

        def cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(display, str(len(clicks) - 1), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("calib", display)

        cv2.imshow("calib", display)
        cv2.setMouseCallback("calib", cb)
        while len(clicks) < 4:
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        if len(clicks) == 4:
            src_pts = np.array(clicks, dtype=np.float32)
            print("using clicked points:", src_pts.tolist())

    H = IPM.calibrate(src_pts, dst_pts)

    cfg = IPMConfig(enabled=True, bev_width=480, bev_height=480,
                    meters_per_pixel=args.meters_per_pixel, H=H)
    ipm = IPM(cfg)
    bev = ipm.warp(img)

    # draw source points + connecting lines on the perspective image
    vis_src = img.copy()
    pts_int = src_pts.astype(int)
    # draw trapezoid edges
    for i in range(4):
        cv2.line(vis_src, tuple(pts_int[i]), tuple(pts_int[(i + 1) % 4]),
                 (0, 255, 255), 2)
    for i, (x, y) in enumerate(pts_int):
        cv2.circle(vis_src, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(vis_src, str(i), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # pad to same height and hstack
    h_vis = max(vis_src.shape[0], bev.shape[0])

    def pad_h(x):
        if x.shape[0] < h_vis:
            x = cv2.copyMakeBorder(x, 0, h_vis - x.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return x

    side = np.hstack([pad_h(vis_src), pad_h(bev)])
    os.makedirs("outputs", exist_ok=True)
    out_img = os.path.join("outputs", "calib_preview.png")
    cv2.imwrite(out_img, side)
    print(f"[ok] wrote {out_img}")

    # write calibration yaml
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = {"ipm": {
        "enabled": True,
        "bev_width": 480,
        "bev_height": 480,
        "meters_per_pixel": float(args.meters_per_pixel),
        "H": H.tolist(),
    }}
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    print(f"[ok] wrote {args.out}")

    # sanity
    nr_x = float(dst_pts[1, 0]); nl_x = float(dst_pts[0, 0])
    width_px_bev = nr_x - nl_x
    width_m_bev = width_px_bev * args.meters_per_pixel
    print(f"[sanity] BEV lane width near = {width_px_bev:.0f} px "
          f"-> {width_m_bev:.2f} m (target 3.5 m)")


if __name__ == "__main__":
    main()