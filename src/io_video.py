
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


class VideoReader:
    """Thin wrapper around cv2.VideoCapture with fixed-coded-dim semantics."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video not found: {path}")
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.info = VideoInfo(
            path=path,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_s=0.0,
        )
        if self.info.fps > 0:
            self.info.duration_s = self.info.frame_count / self.info.fps

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Seek to specific frame index. Returns BGR frame or None."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self.cap.read()
        return frame if ok else None

    def iter_frames(self, start: int = 0, end: Optional[int] = None,
                    step: int = 1) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield (frame_idx, timestamp_s, frame) from start to end."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        idx = start
        end = end if end is not None else self.info.frame_count
        while idx < end:
            ok, frame = self.cap.read()
            if not ok:
                break
            if (idx - start) % step == 0:
                yield idx, idx / self.info.fps, frame
            idx += 1

    def sample_1hz(self, start: int = 0,
                   end: Optional[int] = None) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield every 25th frame = 1 Hz exactly at 25 fps."""
        step = int(round(self.info.fps))  # 25
        yield from self.iter_frames(start=start, end=end, step=step)


# ---------------------------------------------------------------------
# Visual debug helpers
# ---------------------------------------------------------------------

def debug_render_info(frame: np.ndarray, info: VideoInfo,
                      frame_idx: int, timestamp_s: float) -> np.ndarray:
    """Overlay frame index, timestamp, and resolution on frame copy."""
    out = frame.copy()
    h, w = out.shape[:2]
    lines = [
        f"frame={frame_idx}  t={timestamp_s:.2f}s",
        f"res={w}x{h}  fps={info.fps:.2f}",
    ]
    for i, line in enumerate(lines):
        y = 20 + i * 22
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def side_by_side(a: np.ndarray, b: np.ndarray,
                 label_a: str = "A", label_b: str = "B") -> np.ndarray:
    """Concatenate two BGR frames horizontally with labels."""
    if a.shape[0] != b.shape[0]:
        # pad shorter side to match
        target_h = max(a.shape[0], b.shape[0])
        a = cv2.copyMakeBorder(a, 0, target_h - a.shape[0], 0, 0,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
        b = cv2.copyMakeBorder(b, 0, target_h - b.shape[0], 0, 0,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    combo = np.hstack([a, b])
    cv2.putText(combo, label_a, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(combo, label_a, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo, label_b, (a.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(combo, label_b, (a.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return combo