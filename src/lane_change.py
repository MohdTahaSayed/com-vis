
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class LaneChangeConfig:
    window_samples: int = 8
    lane_state_thresh: float = 0.15
    min_peak_magnitude: float = 0.25
    cooldown_samples: int = 10

    @classmethod
    def from_dict(cls, d: Dict) -> "LaneChangeConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


@dataclass
class LaneChangeEvent:
    frame: int
    timestamp_s: float
    direction: str
    magnitude: float


class LaneChangeDetector:
    def __init__(self, cfg: LaneChangeConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._window: Deque[float] = deque(maxlen=self.cfg.window_samples)
        self._last_state: str = "UNKNOWN"
        self._cooldown: int = 0
        self._peak_since_transition: float = 0.0

    def _classify(self, median: float) -> str:
        if median <= -self.cfg.lane_state_thresh:
            return "LEFT"
        if median >= self.cfg.lane_state_thresh:
            return "RIGHT"
        return "NEUTRAL"

    def feed(self,
             frame: int,
             timestamp_s: float,
             offset_norm: Optional[float],
             status: str) -> Optional[LaneChangeEvent]:
        if status != "OK" or offset_norm is None:
            return None

        offset_norm = max(-0.5, min(0.5, float(offset_norm)))

        if self._cooldown > 0:
            self._cooldown -= 1
            self._window.append(offset_norm)
            return None

        self._window.append(offset_norm)

        if len(self._window) < self.cfg.window_samples:
            return None

        med = float(np.median(self._window))
        state = self._classify(med)

        if state in ("LEFT", "RIGHT"):
            self._peak_since_transition = max(
                self._peak_since_transition, abs(offset_norm)
            )
        else:
            self._peak_since_transition = 0.0

        if (state in ("LEFT", "RIGHT")
                and self._last_state in ("LEFT", "RIGHT")
                and state != self._last_state):
            direction = "RIGHT" if state == "RIGHT" else "LEFT"

            if self._peak_since_transition >= self.cfg.min_peak_magnitude:
                ev = LaneChangeEvent(
                    frame=frame,
                    timestamp_s=timestamp_s,
                    direction=direction,
                    magnitude=float(self._peak_since_transition),
                )
                self._cooldown = self.cfg.cooldown_samples
                self._peak_since_transition = abs(offset_norm)
                self._last_state = state
                return ev
            else:
                self._last_state = state
                self._peak_since_transition = abs(offset_norm)
                return None

        if state in ("LEFT", "RIGHT"):
            self._last_state = state

        return None