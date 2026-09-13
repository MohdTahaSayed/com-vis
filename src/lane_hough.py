"""
Hough transform on the edge mask + slope-based left/right classification.

Returns two lists of line segments:
    left_segments  : slope > +slope_abs_min (line goes up-left)
    right_segments : slope < -slope_abs_min (line goes up-right)

Segments with near-horizontal slope (|slope| < slope_abs_min) or near-
vertical (|slope| > slope_abs_max) are discarded as noise.

The sliding-window fitter then uses the median x of each side's segments
as its starting base, instead of histogram peaks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class HoughConfig:
    enabled: bool = True
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 30
    min_line_length: int = 25
    max_line_gap: int = 20
    slope_abs_min: float = 0.35
    slope_abs_max: float = 3.0

    @classmethod
    def from_dict(cls, d: Dict) -> "HoughConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


@dataclass
class Seg:
    x1: int
    y1: int
    x2: int
    y2: int
    slope: float       # dy / dx (image coords, y down)
    length: float


class LaneHough:
    def __init__(self, cfg: HoughConfig):
        self.cfg = cfg

    def classify(self, edges: np.ndarray) -> Tuple[List[Seg], List[Seg], List[Seg]]:
        """
        Returns (left_segs, right_segs, discarded_segs).
        """
        if not self.cfg.enabled:
            return [], [], []

        lines = cv2.HoughLinesP(
            edges,
            rho=self.cfg.rho,
            theta=np.deg2rad(self.cfg.theta_deg),
            threshold=self.cfg.threshold,
            minLineLength=self.cfg.min_line_length,
            maxLineGap=self.cfg.max_line_gap,
        )

        left: List[Seg] = []
        right: List[Seg] = []
        discarded: List[Seg] = []

        if lines is None:
            return left, right, discarded

        for l in lines:
            if l.ndim == 2:
                x1, y1, x2, y2 = l[0]
            else:
                x1, y1, x2, y2 = l

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                discarded.append(
                    Seg(
                        x1,
                        y1,
                        x2,
                        y2,
                        np.inf,
                        float(np.hypot(dx, dy))
                    )
                )
                continue

            slope = dy / dx
            length = float(np.hypot(dx, dy))

            seg = Seg(
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                float(slope),
                length
            )

            a = abs(slope)

            if a < self.cfg.slope_abs_min or a > self.cfg.slope_abs_max:
                discarded.append(seg)
                continue

            # In image coords (y down):
            # Right lane: slope < 0
            # Left lane: slope > 0
            if slope < 0:
                right.append(seg)
            else:
                left.append(seg)

        return left, right, discarded

    @staticmethod
    def median_x_at_bottom(segs: List[Seg], y_query: int) -> Optional[int]:
        """
        For a list of Hough segments, compute the median x-coordinate where
        they would cross a given horizontal line y=y_query (using each
        segment's own slope). Used to seed sliding-window base positions.
        """
        if not segs:
            return None

        xs = []

        for s in segs:
            if not np.isfinite(s.slope):
                continue

            # x at y_query along this segment
            # param: x = x1 + (y_query - y1) / slope
            if abs(s.slope) < 1e-3:
                continue

            x_at = s.x1 + (y_query - s.y1) / s.slope
            xs.append(x_at)

        if not xs:
            return None

        return int(round(float(np.median(xs))))

    def debug_render(
        self,
        frame: np.ndarray,
        left: List[Seg],
        right: List[Seg],
        discarded: Optional[List[Seg]] = None
    ) -> np.ndarray:

        out = frame.copy()

        if discarded:
            for s in discarded:
                cv2.line(
                    out,
                    (s.x1, s.y1),
                    (s.x2, s.y2),
                    (60, 60, 60),
                    1
                )

        for s in left:
            cv2.line(
                out,
                (s.x1, s.y1),
                (s.x2, s.y2),
                (0, 255, 255),
                3
            )

        for s in right:
            cv2.line(
                out,
                (s.x1, s.y1),
                (s.x2, s.y2),
                (0, 255, 0),
                3
            )

        return out