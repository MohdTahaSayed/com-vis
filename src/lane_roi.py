"""
Stage 1c: Trapezoidal region-of-interest mask.

Keeps only the region where ego-lane markings actually live. Sky, trees,
billboards, adjacent lanes are erased. Top vertices can be overridden
per-frame by a detected horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class RoiConfig:
    enabled: bool = True
    vertices: List[List[float]] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "RoiConfig":
        d = d or {}
        verts = d.get("vertices") or [
            [0.10, 0.90], [0.32, 0.50], [0.68, 0.50], [0.90, 0.90],
        ]
        return cls(enabled=bool(d.get("enabled", True)), vertices=verts)


class LaneROI:
    def __init__(self, cfg: RoiConfig):
        self.cfg = cfg

    def _pixel_vertices(self, w: int, h: int,
                        top_y_override: Optional[int] = None) -> np.ndarray:
        pts = []
        for i, (x, y) in enumerate(self.cfg.vertices):
            px = int(round(x * w))
            py = int(round(y * h))
            if top_y_override is not None and i in (1, 2):
                py = top_y_override
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)

    def apply(self, mask: np.ndarray,
              top_y_override: Optional[int] = None) -> np.ndarray:
        if not self.cfg.enabled:
            return mask.copy()
        h, w = mask.shape[:2]
        roi = np.zeros_like(mask)
        pts = self._pixel_vertices(w, h, top_y_override)
        cv2.fillPoly(roi, [pts], 255)
        return cv2.bitwise_and(mask, roi)

    def debug_render(self, frame: np.ndarray,
                     top_y_override: Optional[int] = None) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()
        pts = self._pixel_vertices(w, h, top_y_override)
        cv2.polylines(out, [pts], isClosed=True,
                      color=(255, 0, 255), thickness=2)
        return out