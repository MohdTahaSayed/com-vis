"""
Stage 1a: HSV colour thresholding for lane markings.

Two independent masks:
    - WHITE  : low saturation, high value
    - YELLOW : narrow hue band around ~20-30 (OpenCV scale)

Output is the BITWISE OR of both.

Design note:
    This module returns a binary mask. It does NOT decide what is a lane.
    That is lane_components' job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class ColorRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int
    enabled: bool = True

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array(
            [self.h_min, self.s_min, self.v_min],
            dtype=np.uint8
        )

        hi = np.array(
            [self.h_max, self.s_max, self.v_max],
            dtype=np.uint8
        )

        return lo, hi


@dataclass
class LaneColorConfig:
    white: ColorRange
    yellow: ColorRange

    @classmethod
    def from_dict(cls, d: Dict) -> "LaneColorConfig":
        d = d or {}

        w = d.get("white", {})
        y = d.get("yellow", {})

        return cls(

            white=ColorRange(
                h_min=int(w.get("h_min", 0)),
                h_max=int(w.get("h_max", 179)),
                s_min=int(w.get("s_min", 0)),
                s_max=int(w.get("s_max", 90)),
                v_min=int(w.get("v_min", 130)),
                v_max=int(w.get("v_max", 255)),
                enabled=bool(w.get("enabled", True)),
            ),

            yellow=ColorRange(
                h_min=int(y.get("h_min", 10)),
                h_max=int(y.get("h_max", 40)),
                s_min=int(y.get("s_min", 60)),
                s_max=int(y.get("s_max", 255)),
                v_min=int(y.get("v_min", 100)),
                v_max=int(y.get("v_max", 255)),
                enabled=bool(y.get("enabled", False)),
            ),
        )


class LaneColor:
    """HSV thresholding for white + yellow lane markings."""

    def __init__(self, cfg: LaneColorConfig):
        self.cfg = cfg

    def masks(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Returns (white_mask, yellow_mask, union_mask)
        — all uint8 {0, 255}.
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White mask
        if self.cfg.white.enabled:
            w_lo, w_hi = self.cfg.white.to_arrays()
            white = cv2.inRange(hsv, w_lo, w_hi)
        else:
            white = np.zeros(
                frame.shape[:2],
                dtype=np.uint8
            )

        # Yellow mask
        if self.cfg.yellow.enabled:
            y_lo, y_hi = self.cfg.yellow.to_arrays()
            yellow = cv2.inRange(hsv, y_lo, y_hi)
        else:
            yellow = np.zeros(
                frame.shape[:2],
                dtype=np.uint8
            )

        # Combine both masks
        union = cv2.bitwise_or(white, yellow)

        return white, yellow, union


    def debug_render(
        self,
        frame: np.ndarray,
        white: np.ndarray,
        yellow: np.ndarray
    ) -> np.ndarray:

        """
        Overlay:
        white mask → green tint
        yellow mask → red tint
        """

        out = frame.copy()

        white_bool = white > 0
        yellow_bool = yellow > 0

        out[white_bool] = (0, 255, 0)
        out[yellow_bool] = (0, 0, 255)

        return out