
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

    # minimum bbox side in pixels (reject specks)
    min_bbox_px: int = 20
    # maximum bbox side in pixels (reject landscape features)
    max_bbox_px: int = 250

    # fixed artifact bbox (same as artifact_mask)
    artifact_x1: float = 0.64
    artifact_y1: float = 0.60
    artifact_x2: float = 0.75
    artifact_y2: float = 0.72

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
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2


class SignDetector:
    def __init__(self, cfg: SignDetectConfig):
        self.cfg = cfg
        if YOLO is None:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(cfg.weights)
        self.class_names = list(self.model.names.values())

    # ------------------------------------------------------------------
    def _inside_artifact(self, x1: int, y1: int, x2: int, y2: int,
                         w: int, h: int) -> bool:
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        return (self.cfg.artifact_x1 <= cx <= self.cfg.artifact_x2
                and self.cfg.artifact_y1 <= cy <= self.cfg.artifact_y2)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[SignDetection]:
        h, w = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=self.cfg.conf_threshold,
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

        out: List[SignDetection] = []
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            xyxy = r.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bw, bh = x2 - x1, y2 - y1

            # reject specks
            if bw < self.cfg.min_bbox_px or bh < self.cfg.min_bbox_px:
                continue
            # reject oversized (landscape / whole-frame false positives)
            if bw > self.cfg.max_bbox_px or bh > self.cfg.max_bbox_px:
                continue
            # reject extreme aspect ratios (signs are roughly square-ish)
            aspect = bw / max(bh, 1)
            if aspect < 0.3 or aspect > 6.0:
                continue
            # reject fixed artifact region
            if self._inside_artifact(x1, y1, x2, y2, w, h):
                continue

            out.append(SignDetection(
                cls_id=cls_id,
                cls_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"cls_{cls_id}",
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            ))
        return out

    # ------------------------------------------------------------------
    def debug_render(self, frame: np.ndarray,
                     detections: List[SignDetection]) -> np.ndarray:
        import cv2
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{d.cls_name} {d.confidence:.2f}"
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        return out