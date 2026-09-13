"""
Stage 2: Ego position within lane.

Computes lateral offset in pixels first (camera-relative), then uses the
IPM homography to convert to meters via a bird's-eye-view transform.

The ego reference is frame_width / 2 (camera is at vehicle center).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.lane_fit import LaneFitter


@dataclass
class EgoConfig:
    ego_x_frac: float = 0.50        # camera-at-center assumption
    lane_eval_y_frac: float = 0.85  # y (fraction of frame height) at which we measure offset
    assumed_lane_width_m: float = 3.5

    @classmethod
    def from_dict(cls, d: Dict) -> "EgoConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


@dataclass
class EgoMeasurement:
    offset_px: Optional[float]       # signed; + = ego right of lane center
    lane_center_x: Optional[float]   # px
    lane_width_px: Optional[float]
    y_eval: int
    valid: bool
    reason: str


class EgoPosition:
    def __init__(self, cfg: EgoConfig):
        self.cfg = cfg

    def compute(self,
                left_coeffs: Optional[np.ndarray],
                right_coeffs: Optional[np.ndarray],
                frame_w: int,
                frame_h: int) -> EgoMeasurement:
        y_eval = int(self.cfg.lane_eval_y_frac * frame_h)

        if left_coeffs is None or right_coeffs is None:
            return EgoMeasurement(None, None, None, y_eval, False, "need_both_sides")

        xl = left_coeffs[0] * y_eval * y_eval + left_coeffs[1] * y_eval + left_coeffs[2]
        xr = right_coeffs[0] * y_eval * y_eval + right_coeffs[1] * y_eval + right_coeffs[2]

        if xr <= xl:
            return EgoMeasurement(None, None, None, y_eval, False, "curves_crossed")

        lane_center = (xl + xr) / 2.0
        lane_width = xr - xl
        ego_x = self.cfg.ego_x_frac * frame_w
        offset_px = ego_x - lane_center

        return EgoMeasurement(
            offset_px=float(offset_px),
            lane_center_x=float(lane_center),
            lane_width_px=float(lane_width),
            y_eval=y_eval,
            valid=True,
            reason="ok",
        )