"""
Stage 1b: Connected-component filtering of the lane colour mask.

Rejects non-lane blobs — trucks, road shoulder slabs, sky remnants,
guardrails, tiny asphalt specks.

Filters (per connected component):
    1. area (fraction of frame)          — rejects specks and huge slabs
    2. bbox aspect ratio (w/h)           — lane markings are tall & thin
    3. fill ratio (area / bbox_area)     — lane paint fills its bbox
    4. bbox HEIGHT cap                   — kills truck-side specular streaks
    5. bbox WIDTH cap                    — kills wide road-shoulder slabs

"Bottom relax": in the lowest 25% of the frame the perspective foreshortens
lane paint into wider blobs, so we relax thresholds there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class ComponentsConfig:
    min_area_frac: float = 0.0004
    max_area_frac: float = 0.03
    height_cap_frac: float = 0.22
    width_cap_frac: float = 0.30
    aspect_ratio_max: float = 6.0
    fill_ratio_min: float = 0.15
    bottom_relax_frac: float = 0.25
    bottom_aspect_ratio_max: float = 10.0
    bottom_fill_ratio_min: float = 0.10
    bottom_height_cap_frac: float = 0.50
    bottom_width_cap_frac: float = 0.50

    @classmethod
    def from_dict(cls, d: Dict) -> "ComponentsConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class LaneComponents:
    """Filter binary lane-colour mask by connected-component shape."""

    def __init__(self, cfg: ComponentsConfig):
        self.cfg = cfg

    def filter(self, mask: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        h, w = mask.shape[:2]
        frame_area = h * w
        bottom_y = int((1.0 - self.cfg.bottom_relax_frac) * h)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        kept: List[dict] = []

        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area <= 0:
                continue

            area_frac = area / frame_area
            aspect = bw / max(bh, 1)
            fill = area / max(bw * bh, 1)

            in_bottom = (y + bh) >= bottom_y

            ar_max = (self.cfg.bottom_aspect_ratio_max
                      if in_bottom else self.cfg.aspect_ratio_max)
            fill_min = (self.cfg.bottom_fill_ratio_min
                        if in_bottom else self.cfg.fill_ratio_min)
            hcap = (self.cfg.bottom_height_cap_frac
                    if in_bottom else self.cfg.height_cap_frac)
            wcap = (self.cfg.bottom_width_cap_frac
                    if in_bottom else self.cfg.width_cap_frac)

            if area_frac < self.cfg.min_area_frac or area_frac > self.cfg.max_area_frac:
                continue
            if aspect > ar_max:
                continue
            if fill < fill_min:
                continue
            if bh > hcap * h:
                continue
            if bw > wcap * w:
                continue

            out[labels == i] = 255
            kept.append({
                "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                "area": int(area), "aspect": float(aspect),
                "fill": float(fill), "in_bottom": bool(in_bottom),
            })

        return out, kept

    def debug_render(self, frame: np.ndarray,
                     kept: List[dict],
                     rejected: List[dict] | None = None) -> np.ndarray:
        out = frame.copy()
        for c in kept:
            cv2.rectangle(out, (c["x"], c["y"]),
                          (c["x"] + c["w"], c["y"] + c["h"]),
                          (0, 255, 0), 2)
        if rejected:
            for c in rejected:
                cv2.rectangle(out, (c["x"], c["y"]),
                              (c["x"] + c["w"], c["y"] + c["h"]),
                              (0, 0, 255), 1)
        return out