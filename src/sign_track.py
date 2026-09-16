
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.sign_detect import SignDetection


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


@dataclass
class SignTrackConfig:
    iou_match_threshold: float = 0.35
    max_age: int = 15               # frames without a match before death
    min_hits: int = 3               # detections required to confirm a track

    @classmethod
    def from_dict(cls, d: Dict) -> "SignTrackConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


@dataclass
class SignTrack:
    track_id: int
    cls_id: int
    cls_name: str
    first_frame: int
    last_frame: int
    first_ts: float
    last_ts: float
    bbox: tuple                 # last known bbox (x1,y1,x2,y2)
    confidence: float           # max confidence seen
    hits: int = 0
    age: int = 0                # frames since last match
    confirmed: bool = False


class SignTracker:
    def __init__(self, cfg: SignTrackConfig):
        self.cfg = cfg
        self._next_id = 1
        self._tracks: List[SignTrack] = []
        self._finalized: List[SignTrack] = []

    # ------------------------------------------------------------------
    def update(self,
               frame_idx: int,
               timestamp_s: float,
               detections: List[SignDetection]) -> List[SignTrack]:
        """
        Feed one frame's detections. Returns tracks that were finalized
        on THIS update (i.e. just died and are confirmed).
        """
        newly_finalized: List[SignTrack] = []

        # --- match detections to existing tracks ---
        used_dets = set()
        for tr in self._tracks:
            best_iou = 0.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                if det.cls_id != tr.cls_id:
                    continue
                iou = _iou(tr.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self.cfg.iou_match_threshold and best_idx >= 0:
                det = detections[best_idx]
                tr.bbox = det.bbox
                tr.last_frame = frame_idx
                tr.last_ts = timestamp_s
                tr.confidence = max(tr.confidence, det.confidence)
                tr.hits += 1
                tr.age = 0
                if tr.hits >= self.cfg.min_hits:
                    tr.confirmed = True
                used_dets.add(best_idx)
            else:
                tr.age += 1

        # --- start new tracks for unmatched detections ---
        for i, det in enumerate(detections):
            if i in used_dets:
                continue
            self._tracks.append(SignTrack(
                track_id=self._next_id,
                cls_id=det.cls_id,
                cls_name=det.cls_name,
                first_frame=frame_idx,
                last_frame=frame_idx,
                first_ts=timestamp_s,
                last_ts=timestamp_s,
                bbox=det.bbox,
                confidence=det.confidence,
                hits=1,
                age=0,
                confirmed=False,
            ))
            self._next_id += 1

        # --- retire dead tracks ---
        alive: List[SignTrack] = []
        for tr in self._tracks:
            if tr.age >= self.cfg.max_age:
                if tr.confirmed:
                    self._finalized.append(tr)
                    newly_finalized.append(tr)
            else:
                alive.append(tr)
        self._tracks = alive

        return newly_finalized

    # ------------------------------------------------------------------
    def finalize(self) -> List[SignTrack]:
        """Flush remaining alive tracks at end of video."""
        out = list(self._finalized)
        for tr in self._tracks:
            if tr.confirmed:
                out.append(tr)
        self._finalized = []
        self._tracks = []
        return out

    # ------------------------------------------------------------------
    def debug_render(self, frame, cfg):
        import cv2
        out = frame.copy()
        for tr in self._tracks:
            x1, y1, x2, y2 = tr.bbox
            color = (0, 255, 0) if tr.confirmed else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"#{tr.track_id} {tr.cls_name} h={tr.hits}"
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out