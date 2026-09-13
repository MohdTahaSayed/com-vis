"""
Stage 0: Fixed lens-artifact suppression.

The dashcam has a permanent orange/yellow semi-transparent blob caused by
windshield reflection. After analysis, the blob is too faint for reliable
HSV separation (its saturation barely exceeds the road surface), and — 
critically — because it is that faint, it cannot fool the lane colour
threshold either.

Therefore: we use a single fixed normalized bbox that covers the blob in
every frame. Simple, deterministic, verifiable.

Design rule: artifact mask is applied BEFORE lane_color() every frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class ArtifactMaskConfig:
    enabled: bool = True
    x1: float = 0.64
    y1: float = 0.60
    x2: float = 0.75
    y2: float = 0.72

    @classmethod
    def from_dict(cls, d: Dict) -> "ArtifactMaskConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class ArtifactMask:
    """Blanks one fixed normalized rectangle on every frame."""

    def __init__(self, cfg: ArtifactMaskConfig):
        self.cfg = cfg

    def _pixel_bbox(self, w: int, h: int) -> Tuple[int, int, int, int]:
        x1 = int(round(self.cfg.x1 * w))
        y1 = int(round(self.cfg.y1 * h))
        x2 = int(round(self.cfg.x2 * w))
        y2 = int(round(self.cfg.y2 * h))
        x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
        y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
        return x1, y1, x2, y2

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.cfg.enabled:
            return frame.copy()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._pixel_bbox(w, h)
        out = frame.copy()
        out[y1:y2, x1:x2] = 0
        return out

    def debug_render(self, frame: np.ndarray, fill: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._pixel_bbox(w, h)
        out = frame.copy()
        if fill:
            ov = out.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), -1)
            out = cv2.addWeighted(ov, 0.40, out, 0.60, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, "artifact", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, "artifact", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return out