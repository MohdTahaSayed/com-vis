from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class SignDetectConfig:
    enabled: bool = True
    weights: str = "models/indian_signs.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = "cpu"

    # Reject specks and landscape features
    min_bbox_px: int = 20
    max_bbox_px: int = 250

    # Fixed artifact bbox (windshield reflection / overlay)
    artifact_x1: float = 0.64
    artifact_y1: float = 0.60
    artifact_x2: float = 0.75
    artifact_y2: float = 0.72

    # ---------------------------------------------------------
    # NEW: tiled inference
    # ---------------------------------------------------------
    # When the model is trained on cropped sign images, it fails
    # on full dashcam frames. Splitting the frame into tiles
    # restores a "sign fills the view" scale.
    tiling_enabled: bool = True

    # Number of tiles along each axis (rows, cols).
    # 2x2 = 4 tiles, 3x2 = 6 tiles, 3x3 = 9 tiles.
    tile_rows: int = 2
    tile_cols: int = 3

    # Overlap between adjacent tiles (fraction of tile size).
    # Prevents signs on tile boundaries from being split.
    tile_overlap_frac: float = 0.20

    # Per-tile confidence threshold. Can be lower than the global
    # one because tiles have less noise (fewer unrelated objects).
    tile_conf_threshold: float = 0.20

    # Reject tile-local detections whose bbox is too small or too
    # big relative to the tile.
    tile_min_bbox_frac: float = 0.05
    tile_max_bbox_frac: float = 0.95

    @classmethod
    def from_dict(cls, d: Dict) -> "SignDetectConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


@dataclass
class SignDetection:
    cls_id: int
    cls_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 in frame coords


class SignDetector:
    def __init__(self, cfg: SignDetectConfig):
        self.cfg = cfg
        if YOLO is None:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(cfg.weights)
        self.class_names = list(self.model.names.values())

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _inside_artifact(self, x1, y1, x2, y2, w, h):
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        return (
            self.cfg.artifact_x1 <= cx <= self.cfg.artifact_x2
            and self.cfg.artifact_y1 <= cy <= self.cfg.artifact_y2
        )

    def _passes_global_filters(self, x1, y1, x2, y2, w, h) -> bool:
        bw = x2 - x1
        bh = y2 - y1

        if bw < self.cfg.min_bbox_px or bh < self.cfg.min_bbox_px:
            return False
        if bw > self.cfg.max_bbox_px or bh > self.cfg.max_bbox_px:
            return False

        aspect = bw / max(bh, 1)
        if aspect < 0.3 or aspect > 6.0:
            return False

        if self._inside_artifact(x1, y1, x2, y2, w, h):
            return False

        return True

    def _iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    def _nms(self, dets: List[SignDetection], iou_thr: float) -> List[SignDetection]:
        """
        Greedy NMS. Expects `dets` sorted by confidence descending.
        Only suppresses between detections of the SAME class.
        """
        keep: List[SignDetection] = []
        for d in dets:
            keep_flag = True
            for k in keep:
                if k.cls_id != d.cls_id:
                    continue
                if self._iou(k.bbox, d.bbox) > iou_thr:
                    keep_flag = False
                    break
            if keep_flag:
                keep.append(d)
        return keep

    # ==========================================================
    # TILE GRID
    # ==========================================================

    def _make_tiles(self, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """
        Return list of (x0, y0, x1, y1) tiles covering the frame
        with overlap.
        """
        rows = max(1, self.cfg.tile_rows)
        cols = max(1, self.cfg.tile_cols)
        ov = max(0.0, min(0.9, self.cfg.tile_overlap_frac))

        tile_w = w / cols
        tile_h = h / rows

        tiles = []
        for r in range(rows):
            for c in range(cols):
                x0 = int(max(0, c * tile_w - ov * tile_w))
                y0 = int(max(0, r * tile_h - ov * tile_h))
                x1 = int(min(w, (c + 1) * tile_w + ov * tile_w))
                y1 = int(min(h, (r + 1) * tile_h + ov * tile_h))
                if x1 > x0 and y1 > y0:
                    tiles.append((x0, y0, x1, y1))
        return tiles

    # ==========================================================
    # SINGLE-IMAGE INFERENCE
    # ==========================================================

    def _run_yolo(self, image: np.ndarray, conf: float) -> List[dict]:
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        out = []
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            c = float(r.boxes.conf[i].item())
            xyxy = r.boxes.xyxy[i].cpu().numpy().astype(int)
            out.append({
                "cls_id": cls_id,
                "conf": c,
                "xyxy": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
            })
        return out

    # ==========================================================
    # PUBLIC DETECT
    # ==========================================================

    def detect(self, frame: np.ndarray) -> List[SignDetection]:
        """
        Run sign detection on a frame. Uses tiling if enabled.

        Returns SignDetection objects in FRAME coordinates.
        """
        h, w = frame.shape[:2]

        candidates: List[SignDetection] = []

        # ----------------------------------------------------
        # FULL-FRAME INFERENCE (always runs first)
        # ----------------------------------------------------
        for raw in self._run_yolo(frame, self.cfg.conf_threshold):
            x1, y1, x2, y2 = raw["xyxy"]
            if not self._passes_global_filters(x1, y1, x2, y2, w, h):
                continue
            candidates.append(SignDetection(
                cls_id=raw["cls_id"],
                cls_name=self.class_names[raw["cls_id"]]
                if raw["cls_id"] < len(self.class_names)
                else f"cls_{raw['cls_id']}",
                confidence=raw["conf"],
                bbox=(x1, y1, x2, y2),
            ))

        # ----------------------------------------------------
        # TILED INFERENCE
        # ----------------------------------------------------
        if self.cfg.tiling_enabled:
            for (tx0, ty0, tx1, ty1) in self._make_tiles(w, h):
                tile = frame[ty0:ty1, tx0:tx1]
                th = ty1 - ty0
                tw = tx1 - tx0

                raw_list = self._run_yolo(
                    tile, self.cfg.tile_conf_threshold
                )

                for raw in raw_list:
                    x1, y1, x2, y2 = raw["xyxy"]

                    bw = x2 - x1
                    bh = y2 - y1

                    # Per-tile size sanity
                    if bw < self.cfg.tile_min_bbox_frac * tw:
                        continue
                    if bh < self.cfg.tile_min_bbox_frac * th:
                        continue
                    if bw > self.cfg.tile_max_bbox_frac * tw:
                        continue
                    if bh > self.cfg.tile_max_bbox_frac * th:
                        continue

                    # Convert to frame coordinates
                    fx1 = x1 + tx0
                    fy1 = y1 + ty0
                    fx2 = x2 + tx0
                    fy2 = y2 + ty0

                    if not self._passes_global_filters(fx1, fy1, fx2, fy2, w, h):
                        continue

                    cls_id = raw["cls_id"]
                    cls_name = (
                        self.class_names[cls_id]
                        if cls_id < len(self.class_names)
                        else f"cls_{cls_id}"
                    )

                    candidates.append(SignDetection(
                        cls_id=cls_id,
                        cls_name=cls_name,
                        confidence=raw["conf"],
                        bbox=(fx1, fy1, fx2, fy2),
                    ))

        # ----------------------------------------------------
        # DEDUPLICATE (NMS across classes, per-class suppression)
        # ----------------------------------------------------
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        kept = self._nms(candidates, self.cfg.iou_threshold)

        return kept

    # ==========================================================
    # DEBUG RENDER
    # ==========================================================

    def debug_render(self, frame: np.ndarray,
                     detections: List[SignDetection]) -> np.ndarray:
        import cv2
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{d.cls_name} {d.confidence:.2f}"
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)
        return out