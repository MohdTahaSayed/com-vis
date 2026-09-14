"""
Stage 4a: Traffic sign detection.

Two detectors run in parallel:
    1. YOLOv8n fine-tuned on Indian signs (models/indian_signs.pt)
       → best when signs are large enough (>= ~30 px)
    2. Classical colour + shape detector
       → robust at low resolution (>= 15 px)

Results are merged with IOU deduplication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    min_bbox_px: int = 20

    artifact_x1: float = 0.64
    artifact_y1: float = 0.60
    artifact_x2: float = 0.75
    artifact_y2: float = 0.72

    # --- classical ---
    use_classical: bool = True
    classical_shape_cfg: Optional[Dict] = None

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
    bbox: Tuple[int, int, int, int]


def _iou(a: tuple, b: tuple) -> float:
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


class SignDetector:
    def __init__(self, cfg: SignDetectConfig):
        self.cfg = cfg
        if YOLO is None:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(cfg.weights)
        self.class_names = list(self.model.names.values())

        # classical detector
        self.classical = None
        if cfg.use_classical:
            try:
                from src.sign_shape import IndianSignShapeDetector, ShapeDetectConfig
                scfg = ShapeDetectConfig.from_dict(cfg.classical_shape_cfg or {})
                scfg.artifact_x1 = cfg.artifact_x1
                scfg.artifact_y1 = cfg.artifact_y1
                scfg.artifact_x2 = cfg.artifact_x2
                scfg.artifact_y2 = cfg.artifact_y2
                self.classical = IndianSignShapeDetector(scfg)
            except Exception as e:
                print(f"[warn] classical sign detector failed: {e}")

    # ------------------------------------------------------------------
    def _inside_artifact(self, x1: int, y1: int, x2: int, y2: int,
                         w: int, h: int) -> bool:
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        return (self.cfg.artifact_x1 <= cx <= self.cfg.artifact_x2
                and self.cfg.artifact_y1 <= cy <= self.cfg.artifact_y2)

    # ------------------------------------------------------------------
    def _detect_yolo(self, frame: np.ndarray) -> List[SignDetection]:
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
            if (x2 - x1) < self.cfg.min_bbox_px or (y2 - y1) < self.cfg.min_bbox_px:
                continue
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
    def detect(self, frame: np.ndarray) -> List[SignDetection]:
        yolo = self._detect_yolo(frame)
        classical = self.classical.detect(frame) if self.classical is not None else []

        # merge — YOLO first, then classical that don't overlap with YOLO
        merged = list(yolo)
        for cd in classical:
            keep = True
            for y in yolo:
                if _iou(cd.bbox, y.bbox) > 0.3:
                    keep = False
                    break
            if keep:
                merged.append(cd)
        return merged

    # ------------------------------------------------------------------
    def debug_render(self, frame: np.ndarray,
                     detections: List[SignDetection]) -> np.ndarray:
        import cv2
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            # classical (cls_id=-1) drawn in orange, YOLO in red
            color = (0, 165, 255) if d.cls_id == -1 else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.cls_name} {d.confidence:.2f}"
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        return out