"""
Stage 2: Inverse Perspective Mapping (IPM) via homography.

We compute a 3x3 homography H so that points on the road plane in the
perspective image map to a top-down (bird's-eye) view where lane markings
are parallel and distances are metric.

Two calibration approaches:

    A. From four manually chosen correspondences in a straight-road frame
       (Direct Linear Transform via cv2.findHomography).
    B. From camera intrinsics + extrinsics (K, R) — not used here.

We use (A). Calibration points are stored in config/calibration.yaml:
    src_pts: 4 points in the perspective image
    dst_pts: their target locations in the BEV image (chosen so that
             lane width becomes a known pixel count, e.g. 300 px for 3.5 m)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml


@dataclass
class IPMConfig:
    enabled: bool = True
    bev_width: int = 480
    bev_height: int = 480
    # meters-per-pixel in BEV. Derived from calibration (see yaml).
    meters_per_pixel: Optional[float] = None
    # homography matrix (3x3), loaded from yaml if present
    H: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "IPMConfig":
        d = d or {}
        c = cls()
        for k in ("enabled", "bev_width", "bev_height", "meters_per_pixel"):
            if k in d:
                setattr(c, k, d[k])
        if "H" in d and d["H"] is not None:
            c.H = np.array(d["H"], dtype=np.float64)
        return c


class IPM:
    def __init__(self, cfg: IPMConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    @staticmethod
    def calibrate(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """Compute H from 4 source→destination point pairs (DLT)."""
        src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, method=0)
        return H

    # ------------------------------------------------------------------
    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Warp frame to bird's-eye view using configured H."""
        if self.cfg.H is None:
            return frame.copy()
        return cv2.warpPerspective(
            frame, self.cfg.H,
            (self.cfg.bev_width, self.cfg.bev_height),
            flags=cv2.INTER_LINEAR,
        )

    def warp_points(self, pts: np.ndarray) -> np.ndarray:
        """Warp an (N,2) array of (x,y) points to BEV space."""
        if self.cfg.H is None:
            return pts.copy()
        pts_f = pts.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts_f, self.cfg.H).reshape(-1, 2)

    # ------------------------------------------------------------------
    def offset_px_to_meters(self, offset_px: float) -> Optional[float]:
        if self.cfg.meters_per_pixel is None:
            return None
        return offset_px * self.cfg.meters_per_pixel


# ---------------------------------------------------------------------
def load_calibration(path: str) -> IPMConfig:
    if not os.path.isfile(path):
        return IPMConfig(enabled=False)
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return IPMConfig.from_dict(d.get("ipm", {}))