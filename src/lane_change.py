from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class LaneChangeConfig:
    # ----------------------------------------------------------
    # v3-style baseline-shift detector on lane_center_x
    # ----------------------------------------------------------
    # A lane change fires when the smoothed lane centre x moves
    # by more than `min_boundary_shift_frac * lane_width_px`
    # (or `min_boundary_shift_px` if width unavailable) from a
    # recent baseline, persisting for `persist_samples`.
    #
    # Direction:
    #   lane centre moves RIGHT → direction = RIGHT
    #   lane centre moves LEFT  → direction = LEFT
    #
    # (Verified against ground truth on VBOX0011_Trim.mp4.)
    min_boundary_shift_frac: float = 0.30
    min_boundary_shift_px: float = 90.0

    # Short smoothing window (samples)
    smooth_window: int = 3

    # Baseline window (samples)
    baseline_window: int = 5

    # Persistence
    persist_samples: int = 2

    # Cooldown
    cooldown_samples: int = 12

    min_confidence: float = 0.15

    lane_width_min_px: float = 100.0
    lane_width_max_px: float = 700.0

    # Alternation filter — after LEFT, next must be RIGHT (and vice versa)
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
    delta_left_px: float        # signed shift of lane centre (kept for API compat)
    delta_right_px: float       # always equals delta_left_px here
    magnitude_diff_px: float    # always 0 here


class LaneChangeDetector:
    """
    v3-style lane-change detector on lane_center_x with alternation.

    Baseline = median of recent valid lane-centre samples.
    Current  = smoothed median of the last few samples.
    Shift    = current - baseline.

    Fires when |shift| exceeds threshold, persists for N samples,
    and alternates with the previous direction.
    """

    def __init__(self, cfg: LaneChangeConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._smooth: Deque[float] = deque(maxlen=self.cfg.smooth_window)
        self._history: Deque[float] = deque(maxlen=self.cfg.baseline_window)
        self._cooldown: int = 0
        self._last_direction: Optional[str] = None
        self._candidate_direction: Optional[str] = None
        self._candidate_count: int = 0

    def feed(
        self,
        frame: int,
        timestamp_s: float,
        lane_center_x: Optional[float],
        lane_width_px: Optional[float],
        status: str,
    ) -> Optional[LaneChangeEvent]:

        # ---- GATES ----
        if status != "OK" or lane_center_x is None:
            return None

        if lane_width_px is not None:
            if (
                lane_width_px < self.cfg.lane_width_min_px
                or lane_width_px > self.cfg.lane_width_max_px
            ):
                return None

        # ---- SMOOTH ----
        self._smooth.append(float(lane_center_x))
        smoothed = float(np.median(self._smooth))

        # ---- COOLDOWN ----
        if self._cooldown > 0:
            self._cooldown -= 1
            self._history.append(smoothed)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        # ---- NEED HISTORY ----
        if len(self._history) < self.cfg.baseline_window:
            self._history.append(smoothed)
            return None

        baseline = float(np.median(self._history))

        # ---- THRESHOLD ----
        if (
            self.cfg.min_boundary_shift_frac > 0
            and lane_width_px is not None
            and lane_width_px > 0
        ):
            threshold = self.cfg.min_boundary_shift_frac * float(lane_width_px)
        else:
            threshold = self.cfg.min_boundary_shift_px

        shift = smoothed - baseline

        # ---- CANDIDATE ----
        if abs(shift) < threshold:
            self._history.append(smoothed)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        direction = "RIGHT" if shift > 0 else "LEFT"

        # ---- PERSISTENCE ----
        if direction == self._candidate_direction:
            self._candidate_count += 1
        else:
            self._candidate_direction = direction
            self._candidate_count = 1

        if self._candidate_count < self.cfg.persist_samples:
            self._history.append(smoothed)
            return None

        # ---- ALTERNATION ----
        if (
            self.cfg.enforce_alternation
            and self._last_direction is not None
            and direction == self._last_direction
        ):
            self._history.append(smoothed)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        # ---- EMIT ----
        event = LaneChangeEvent(
            frame=frame,
            timestamp_s=timestamp_s,
            direction=direction,
            magnitude=float(abs(shift)),
            delta_left_px=float(shift),
            delta_right_px=float(shift),
            magnitude_diff_px=0.0,
        )

        self._last_direction = direction
        self._cooldown = self.cfg.cooldown_samples

        self._history.clear()
        self._history.append(smoothed)
        self._candidate_direction = None
        self._candidate_count = 0

        return event