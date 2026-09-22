from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class LaneChangeConfig:
    # --- Trigger 1: total shift from stable baseline ---
    min_total_shift_frac: float = 0.25
    min_total_shift_px: float = 90.0

    # --- Baseline stability ---
    # Long buffer holds many samples. Baseline = median of the
    # OLDEST `baseline_old_frac` of that buffer.
    baseline_buffer_len: int = 80
    baseline_old_frac: float = 0.5

    # Short smoothing for the "current" position
    smooth_window: int = 3

    # Persistence — total-shift condition must hold for this many
    # consecutive samples before firing.
    persist_samples: int = 3

    # Cooldown
    cooldown_samples: int = 12

    min_confidence: float = 0.15

    lane_width_min_px: float = 100.0
    lane_width_max_px: float = 700.0

    enforce_alternation: bool = True

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
    delta_left_px: float
    delta_right_px: float


class LaneChangeDetector:
    """
    Baseline-shift lane-change detector with a stable baseline.

    Baseline = median of the OLD portion of a long history buffer.
    Current  = short median of recent samples.
    Trigger  = both boundaries shifted by > threshold in same
               direction, persisting for N samples.
    """

    def __init__(self, cfg: LaneChangeConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._smooth_left: Deque[float] = deque(maxlen=self.cfg.smooth_window)
        self._smooth_right: Deque[float] = deque(maxlen=self.cfg.smooth_window)

        self._buf_left: Deque[float] = deque(maxlen=self.cfg.baseline_buffer_len)
        self._buf_right: Deque[float] = deque(maxlen=self.cfg.baseline_buffer_len)

        self._cooldown: int = 0
        self._last_direction: Optional[str] = None

        self._candidate_direction: Optional[str] = None
        self._candidate_count: int = 0

    def feed(
        self,
        frame: int,
        timestamp_s: float,
        left_x: Optional[float],
        right_x: Optional[float],
        lane_width_px: Optional[float],
        status: str,
    ) -> Optional[LaneChangeEvent]:

        # ---- GATES ----
        if status != "OK":
            return None
        if left_x is None or right_x is None:
            return None

        if lane_width_px is not None:
            if (
                lane_width_px < self.cfg.lane_width_min_px
                or lane_width_px > self.cfg.lane_width_max_px
            ):
                return None

        # ---- SMOOTH ----
        self._smooth_left.append(float(left_x))
        self._smooth_right.append(float(right_x))

        s_left = float(np.median(self._smooth_left))
        s_right = float(np.median(self._smooth_right))

        # ---- COOLDOWN ----
        if self._cooldown > 0:
            self._cooldown -= 1
            self._buf_left.append(s_left)
            self._buf_right.append(s_right)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        # ---- COMPUTE BASELINE ----
        if len(self._buf_left) < 20:
            # Not enough history yet
            self._buf_left.append(s_left)
            self._buf_right.append(s_right)
            return None

        arr_l = np.array(self._buf_left)
        arr_r = np.array(self._buf_right)

        n = len(arr_l)
        old_end = max(5, int(n * self.cfg.baseline_old_frac))

        baseline_l = float(np.median(arr_l[:old_end]))
        baseline_r = float(np.median(arr_r[:old_end]))

        delta_l = s_left - baseline_l
        delta_r = s_right - baseline_r

        # ---- THRESHOLD ----
        if (
            self.cfg.min_total_shift_frac > 0
            and lane_width_px is not None
            and lane_width_px > 0
        ):
            threshold = self.cfg.min_total_shift_frac * float(lane_width_px)
        else:
            threshold = self.cfg.min_total_shift_px

        # ---- CANDIDATE ----
        same_sign = np.sign(delta_l) == np.sign(delta_r) and delta_l != 0
        both_big = abs(delta_l) >= threshold and abs(delta_r) >= threshold

        if not (same_sign and both_big):
            self._buf_left.append(s_left)
            self._buf_right.append(s_right)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        direction = "RIGHT" if delta_l > 0 else "LEFT"

        # ---- PERSISTENCE ----
        if direction == self._candidate_direction:
            self._candidate_count += 1
        else:
            self._candidate_direction = direction
            self._candidate_count = 1

        if self._candidate_count < self.cfg.persist_samples:
            self._buf_left.append(s_left)
            self._buf_right.append(s_right)
            return None

        # ---- ALTERNATION ----
        if (
            self.cfg.enforce_alternation
            and self._last_direction is not None
            and direction == self._last_direction
        ):
            self._buf_left.append(s_left)
            self._buf_right.append(s_right)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        # ---- EMIT ----
        event = LaneChangeEvent(
            frame=frame,
            timestamp_s=timestamp_s,
            direction=direction,
            magnitude=float(0.5 * (abs(delta_l) + abs(delta_r))),
            delta_left_px=float(delta_l),
            delta_right_px=float(delta_r),
        )

        self._last_direction = direction
        self._cooldown = self.cfg.cooldown_samples

        # Clear history so the transition doesn't re-fire
        self._buf_left.clear()
        self._buf_right.clear()
        self._buf_left.append(s_left)
        self._buf_right.append(s_right)

        self._candidate_direction = None
        self._candidate_count = 0

        return event