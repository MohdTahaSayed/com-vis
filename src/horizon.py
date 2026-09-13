"""
Horizon detection via row-gradient analysis.

Finds the topmost row above which the image is smooth (sky) and below which
it is textured (road, vegetation, vehicles).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class HorizonConfig:
    enabled: bool = True
    search_top_frac: float = 0.30
    search_bot_frac: float = 0.75
    variance_threshold: float = 60.0
    run_length: int = 8
    smoothing_alpha: float = 0.25
    min_y_frac: float = 0.35
    max_y_frac: float = 0.65

    @classmethod
    def from_dict(cls, d: Dict) -> "HorizonConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class HorizonDetector:
    """Row-gradient horizon detector with temporal smoothing."""

    def __init__(self, cfg: HorizonConfig):
        self.cfg = cfg
        self._last_y: Optional[int] = None

    def detect(self, frame: np.ndarray) -> Optional[int]:
        if not self.cfg.enabled:
            return None
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.abs(grad)
        row_score = grad.mean(axis=1)

        y_top = int(self.cfg.search_top_frac * h)
        y_bot = int(self.cfg.search_bot_frac * h)

        found: Optional[int] = None
        run = 0
        for y in range(y_top, y_bot):
            if row_score[y] > self.cfg.variance_threshold:
                run += 1
                if run >= self.cfg.run_length:
                    found = y - run + 1
                    break
            else:
                run = 0

        if found is None:
            return self._last_y

        min_y = int(self.cfg.min_y_frac * h)
        max_y = int(self.cfg.max_y_frac * h)
        found = max(min_y, min(max_y, found))

        if self._last_y is None:
            self._last_y = found
        else:
            a = self.cfg.smoothing_alpha
            self._last_y = int(round((1 - a) * self._last_y + a * found))

        return self._last_y

    def debug_render(self, frame: np.ndarray,
                     horizon_y: Optional[int]) -> np.ndarray:
        out = frame.copy()
        if horizon_y is not None:
            cv2.line(out, (0, horizon_y), (out.shape[1], horizon_y),
                     (0, 255, 255), 2)
            cv2.putText(out, f"horizon y={horizon_y}",
                        (10, max(20, horizon_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, f"horizon y={horizon_y}",
                        (10, max(20, horizon_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 1, cv2.LINE_AA)
        return out