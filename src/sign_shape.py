"""
Stage 4b: Classical colour + shape sign detector.

Works when YOLOv8n cannot — at low resolutions where signs are 15-40 px
and blur reduces the deep-learning model's confidence below threshold.

Detects:
    - TRIANGLE_WARNING_SIGN   : red/white triangular warning markers
    - GREEN_DIRECTIONAL_SIGN  : large green rectangles (direction/destination)
    - BLUE_INFORMATIONAL_SIGN : blue rectangular informational signs
    - BLUE_CIRCLE_SIGN        : blue circles (mandatory direction)
    - RED_CIRCLE_SIGN         : red circles (regulatory)

Approach (based on Yadav et al. IJASST 2019, "Indian Traffic Signboard
Recognition and Driver Alert System"):
    1. Convert to HSV
    2. Build colour masks: red, white, green, blue
    3. Connected components on each
    4. Shape classification per component with STRICT gates:
        - circularity  = 4*pi*A / P^2
        - vertex count = len(approxPolyDP(contour, eps))
        - aspect ratio = bbox_w / bbox_h
        - fill ratio   = area / bbox_area
    5. Size filter, top-of-frame rejection, artifact bbox rejection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.sign_detect import SignDetection


@dataclass
class ShapeDetectConfig:
    enabled: bool = True

    # minimum bbox side in pixels — 18 px minimum to reduce noise
    min_bbox_px: int = 18
    # maximum bbox side in pixels — reject slabs
    max_bbox_px: int = 180
    # maximum bbox area fraction (reject huge backgrounds)
    max_area_frac: float = 0.15

    # --- HSV ranges (OpenCV H: 0..179) ---
    # red wraps around H=0/179
    red1_h_min: int = 0;   red1_h_max: int = 12
    red2_h_min: int = 168; red2_h_max: int = 179
    red_s_min: int = 60;   red_v_min: int = 50

    green_h_min: int = 40; green_h_max: int = 85
    green_s_min: int = 50; green_v_min: int = 50

    blue_h_min: int = 100; blue_h_max: int = 130
    blue_s_min: int = 120; blue_v_min: int = 80

    white_s_max: int = 80; white_v_min: int = 130

    # --- shape thresholds ---
    circle_circularity_min: float = 0.80
    triangle_eps_frac: float = 0.04
    rect_aspect_min: float = 1.0
    rect_aspect_max: float = 6.0
    rect_fill_min: float = 0.70

    # --- artifact bbox (same as YOLO filter) ---
    artifact_x1: float = 0.64
    artifact_y1: float = 0.60
    artifact_x2: float = 0.75
    artifact_y2: float = 0.72

    # reject detections in the top X% of the frame (sky contamination)
    top_reject_frac: float = 0.15

    @classmethod
    def from_dict(cls, d: Dict) -> "ShapeDetectConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class IndianSignShapeDetector:
    """Detect Indian signs by colour + shape (no ML)."""

    def __init__(self, cfg: ShapeDetectConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _inside_artifact(self, x1: int, y1: int, x2: int, y2: int,
                         w: int, h: int) -> bool:
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        return (self.cfg.artifact_x1 <= cx <= self.cfg.artifact_x2
                and self.cfg.artifact_y1 <= cy <= self.cfg.artifact_y2)

    # ------------------------------------------------------------------
    @staticmethod
    def _vertex_count(contour: np.ndarray, perimeter: float) -> int:
        eps = max(2.0, 0.04 * perimeter)
        approx = cv2.approxPolyDP(contour, eps, True)
        return len(approx)

    @staticmethod
    def _circularity(area: float, perim: float) -> float:
        if perim <= 1e-3:
            return 0.0
        return 4.0 * np.pi * area / (perim * perim)

    # ------------------------------------------------------------------
    def _classify(self, contour: np.ndarray,
                  x: int, y: int, bw: int, bh: int,
                  area: float) -> Optional[str]:
        per = cv2.arcLength(contour, True)
        circ = self._circularity(area, per)
        vcount = self._vertex_count(contour, per)
        aspect = bw / max(bh, 1)
        fill = area / max(bw * bh, 1)

        # ---- size sanity: reject components too large ----
        if bw > self.cfg.max_bbox_px or bh > self.cfg.max_bbox_px:
            return None

        # ---- aspect gate: real signs are roughly square-ish ----
        # (green directional signs are wide rectangles — handled later)
        is_wide_rectangle_candidate = (vcount >= 4
                                       and 1.5 <= aspect <= 6.0
                                       and fill >= self.cfg.rect_fill_min)

        if aspect < 0.4 or aspect > 2.5:
            if not is_wide_rectangle_candidate:
                return None

        # ---- triangle: strict gates ----
        if (vcount == 3
                and 0.50 <= circ <= 0.70
                and 0.35 <= fill <= 0.65
                and 0.6 <= aspect <= 1.6):
            return "TRIANGLE"

        # ---- circle: strict gates ----
        if (circ >= self.cfg.circle_circularity_min
                and vcount >= 6
                and 0.7 <= aspect <= 1.4
                and fill >= 0.70):
            return "CIRCLE"

        # ---- rectangle: strict gates ----
        if (vcount >= 4
                and self.cfg.rect_aspect_min <= aspect <= self.cfg.rect_aspect_max
                and fill >= self.cfg.rect_fill_min):
            return "RECTANGLE"

        return None

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[SignDetection]:
        c = self.cfg
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- build colour masks ---
        r1 = cv2.inRange(hsv,
                         (c.red1_h_min, c.red_s_min, c.red_v_min),
                         (c.red1_h_max, 255, 255))
        r2 = cv2.inRange(hsv,
                         (c.red2_h_min, c.red_s_min, c.red_v_min),
                         (c.red2_h_max, 255, 255))
        red = cv2.bitwise_or(r1, r2)
        green = cv2.inRange(hsv,
                            (c.green_h_min, c.green_s_min, c.green_v_min),
                            (c.green_h_max, 255, 255))
        blue = cv2.inRange(hsv,
                           (c.blue_h_min, c.blue_s_min, c.blue_v_min),
                           (c.blue_h_max, 255, 255))
        white = cv2.inRange(hsv,
                            (0, 0, c.white_v_min),
                            (179, c.white_s_max, 255))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        results: List[SignDetection] = []

        # --- process each colour ---
        plans = [
            ("RED",   red,
             {"TRIANGLE": "red_triangle_warning",
              "CIRCLE":   "red_circle_sign"}),
            ("GREEN", green,
             {"RECTANGLE": "green_directional_sign",
              "CIRCLE":    "green_directional_sign"}),
            ("BLUE",  blue,
             {"RECTANGLE": "blue_info_sign",
              "CIRCLE":    "blue_circle_sign"}),
            ("WHITE", white,
             {"TRIANGLE": "white_triangle_warning"}),
        ]

        for colour, mask, mapping in plans:
            m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area <= 0:
                    continue
                if bw < c.min_bbox_px or bh < c.min_bbox_px:
                    continue
                if bw > c.max_bbox_px or bh > c.max_bbox_px:
                    continue
                if area > c.max_area_frac * w * h:
                    continue
                if self._inside_artifact(x, y, x + bw, y + bh, w, h):
                    continue
                # reject top-of-frame (sky)
                if y < c.top_reject_frac * h:
                    continue

                comp = (labels[y:y+bh, x:x+bw] == i).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea)
                shape = self._classify(cnt, x, y, bw, bh, area)
                if shape is None:
                    continue
                cls_name = mapping.get(shape)
                if cls_name is None:
                    continue

                results.append(SignDetection(
                    cls_id=-1,
                    cls_name=cls_name,
                    confidence=0.5,
                    bbox=(int(x), int(y), int(x + bw), int(y + bh)),
                ))

        return results


if __name__ == "__main__":
    cfg = ShapeDetectConfig()
    print("classical shape detector config:", cfg)