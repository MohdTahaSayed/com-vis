"""
Stage 1 (rebuilt): Canny edge detection as the PRIMARY lane signal.

Pipeline:
    frame → grayscale → Gaussian blur → Canny
          → trapezoid ROI (removes sky, trees, billboards, hood)
          → [optional] HSV white/yellow reinforcement
          → edge mask (binary)

Why edges instead of HSV-primary:
    Sky, asphalt, and weathered lane paint share similar HSV ranges.
    No threshold cleanly separates them.
    But lane paint produces a sharp intensity transition vs asphalt.
    Sky/asphalt/grass are smooth. Canny exploits this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig


@dataclass
class CannyConfig:
    enabled: bool = True
    blur_kernel: int = 5
    low_threshold: int = 50
    high_threshold: int = 150

    @classmethod
    def from_dict(cls, d: Dict) -> "CannyConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class LaneEdges:
    """
    Produce a binary edge mask restricted to the road ROI.
    Optionally reinforce with HSV white/yellow.

    Public:
        compute(frame, top_y_override=None)
            → (edges_roi, edges_raw, hsv_reinforced)
    """

    def __init__(self,
                 canny_cfg: CannyConfig,
                 roi: LaneROI,
                 lane_color: Optional[LaneColor] = None,
                 reinforce_with_hsv: bool = True):
        self.canny_cfg = canny_cfg
        self.roi = roi
        self.lane_color = lane_color
        self.reinforce_with_hsv = reinforce_with_hsv

    def _canny(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.canny_cfg
        k = cfg.blur_kernel if cfg.blur_kernel % 2 == 1 else cfg.blur_kernel + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(blur, cfg.low_threshold, cfg.high_threshold)
        return edges

    def compute(self,
                frame: np.ndarray,
                top_y_override: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # 1. Canny on the full frame
        edges_raw = self._canny(frame)

        # 2. Mask by ROI (sky, hood, off-road regions gone)
        edges_roi = self.roi.apply(edges_raw, top_y_override=top_y_override)

        # 3. Optional HSV reinforcement
        hsv_hits: Optional[np.ndarray] = None
        if self.reinforce_with_hsv and self.lane_color is not None:
            white, yellow, union = self.lane_color.masks(frame)
            union_roi = self.roi.apply(union, top_y_override=top_y_override)
            # dilate the HSV hits so edge pixels near paint still pass
            k = np.ones((5, 5), np.uint8)
            union_dil = cv2.dilate(union_roi, k, iterations=1)

            # reinforce: keep edge pixels that ARE white/yellow
            # ALSO keep edge pixels immediately adjacent to white/yellow
            edges_reinf = cv2.bitwise_and(edges_roi, union_dil)

            # if the reinforced mask is too empty, fall back to edges_roi
            # (preserves signal on frames where HSV misses paint)
            if int((edges_reinf > 0).sum()) < 200:
                hsv_hits = union_roi
            else:
                edges_roi = edges_reinf
                hsv_hits = union_roi

        return edges_roi, edges_raw, hsv_hits

    def debug_render(self, frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Overlay edge pixels in bright green on a dimmed frame."""
        out = (frame * 0.5).astype(np.uint8)
        out[edges > 0] = (0, 255, 0)
        return out