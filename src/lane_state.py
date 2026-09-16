
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class StateConfig:
    hold_frames: int = 8
    miss_frames: int = 20
    smoothing_alpha: float = 0.30
    min_confidence: float = 0.15

    @classmethod
    def from_dict(cls, d: Dict) -> "StateConfig":
        d = d or {}
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


@dataclass
class SideState:
    status: str = "MISS"
    smoothed_coeffs: Optional[np.ndarray] = None
    last_valid_coeffs: Optional[np.ndarray] = None
    fail_count: int = 0
    ok_count: int = 0
    confidence: float = 0.0

    def reset(self):
        self.status = "MISS"
        self.smoothed_coeffs = None
        self.last_valid_coeffs = None
        self.fail_count = 0
        self.confidence = 0.0


class LaneState:
    """Two independent side states (left, right)."""

    def __init__(self, cfg: StateConfig):
        self.cfg = cfg
        self.left = SideState()
        self.right = SideState()

    def _update_side(self, state: SideState,
                     coeffs: Optional[np.ndarray],
                     confidence: float) -> Tuple[Optional[np.ndarray], str, float]:
        valid = (coeffs is not None) and (confidence >= self.cfg.min_confidence)

        if valid:
            state.fail_count = 0
            state.ok_count += 1

            if state.last_valid_coeffs is None:
                state.last_valid_coeffs = coeffs.copy()
                state.smoothed_coeffs = coeffs.copy()
            else:
                a = self.cfg.smoothing_alpha
                state.last_valid_coeffs = coeffs.copy()
                state.smoothed_coeffs = (1 - a) * state.smoothed_coeffs + a * coeffs

            state.status = "OK"
            state.confidence = float(confidence)
            return state.smoothed_coeffs.copy(), "OK", float(confidence)

        state.fail_count += 1
        if state.smoothed_coeffs is not None and state.fail_count <= self.cfg.hold_frames:
            state.status = "HOLD"
            state.confidence = max(0.0, state.confidence * 0.85)
            return state.smoothed_coeffs.copy(), "HOLD", state.confidence
        else:
            state.status = "MISS"
            state.confidence = 0.0
            if state.fail_count >= self.cfg.miss_frames:
                state.smoothed_coeffs = None
                state.last_valid_coeffs = None
            return None, "MISS", 0.0

    def update(self,
               left_coeffs: Optional[np.ndarray], left_conf: float,
               right_coeffs: Optional[np.ndarray], right_conf: float):
        l = self._update_side(self.left, left_coeffs, left_conf)
        r = self._update_side(self.right, right_coeffs, right_conf)
        return l, r

    def debug_summary(self) -> str:
        return (f"L={self.left.status}(c={self.left.confidence:.2f},"
                f"fail={self.left.fail_count})  "
                f"R={self.right.status}(c={self.right.confidence:.2f},"
                f"fail={self.right.fail_count})")