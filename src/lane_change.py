from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class LaneChangeConfig:
    # ----------------------------------------------------------
    # Jump-based detection
    # ----------------------------------------------------------
    # A lane change is detected when the *lane centre x* moves
    # by more than `jump_px_threshold` (or, if lane_width is
    # available, `relative_jump_frac * lane_width`) within
    # `baseline_window` consecutive valid samples.
    #
    # Direction semantics:
    #   lane centre moves RIGHT in image  →  direction = "LEFT"  (ego moved LEFT)
    #   lane centre moves LEFT  in image  →  direction = "RIGHT" (ego moved RIGHT)
    #
    # Why inverted? Because the lane centre x is measured in a
    # fixed image frame. When the ego moves left, the visible
    # scene (and hence the detected lane centre) shifts RIGHT.
    jump_px_threshold: float = 60.0

    # Median filter window for smoothing the raw lane-centre signal
    smooth_window: int = 3

    # How many recent samples to consider when computing the
    # baseline (median of the older end of this deque).
    baseline_window: int = 5

    # Cooldown (in samples) after firing an event.
    cooldown_samples: int = 10

    # Scale jump threshold by lane width. Set to 0 to use only
    # the fixed `jump_px_threshold`.
    relative_jump_frac: float = 0.18

    # Minimum confidence required to accept a sample.
    min_confidence: float = 0.15

    # Sanity bounds for lane width (px). If a measurement's
    # lane_width falls outside this range, we skip the sample.
    lane_width_min_px: float = 100.0
    lane_width_max_px: float = 700.0

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
    direction: str          # "LEFT" or "RIGHT" (ego motion)
    magnitude: float        # |jump| in pixels
    jump_px: float          # raw signed jump in image x


class LaneChangeDetector:
    """
    Lane-change detector based on lane-centre motion.

    Consumes per-sample (frame, ts, lane_center_x, lane_width_px,
    status) tuples and emits a LaneChangeEvent whenever the smoothed
    lane centre shifts by more than the configured threshold within
    a short window.

    Direction semantics (physical ego motion):
        lane centre moves RIGHT in image  →  ego moved LEFT
        lane centre moves LEFT  in image  →  ego moved RIGHT
    """

    def __init__(self, cfg: LaneChangeConfig):
        self.cfg = cfg
        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        self._smooth: Deque[float] = deque(maxlen=self.cfg.smooth_window)
        self._history: Deque[float] = deque(maxlen=self.cfg.baseline_window)
        self._cooldown: int = 0

    # ========================================================
    # MAIN FEED
    # ========================================================

    def feed(
        self,
        frame: int,
        timestamp_s: float,
        lane_center_x: Optional[float],
        lane_width_px: Optional[float],
        status: str,
    ) -> Optional[LaneChangeEvent]:
        """
        Returns a LaneChangeEvent if a lane change was just detected,
        otherwise None.
        """

        # ---------------------------------------------------
        # GATE — status
        # ---------------------------------------------------

        if status != "OK" or lane_center_x is None:
            return None

        # ---------------------------------------------------
        # GATE — lane width sanity
        # ---------------------------------------------------

        if lane_width_px is not None:
            if (
                lane_width_px < self.cfg.lane_width_min_px
                or lane_width_px > self.cfg.lane_width_max_px
            ):
                return None

        # ---------------------------------------------------
        # SMOOTH raw signal
        # ---------------------------------------------------

        self._smooth.append(float(lane_center_x))
        smoothed = float(np.median(self._smooth))

        # ---------------------------------------------------
        # COOLDOWN
        # ---------------------------------------------------

        if self._cooldown > 0:
            self._cooldown -= 1
            self._history.append(smoothed)
            return None

        # ---------------------------------------------------
        # BASELINE = median of the recent history
        # ---------------------------------------------------

        if len(self._history) < self.cfg.baseline_window:
            self._history.append(smoothed)
            return None

        baseline = float(np.median(self._history))

        # ---------------------------------------------------
        # THRESHOLD
        # ---------------------------------------------------

        if (
            self.cfg.relative_jump_frac > 0
            and lane_width_px is not None
            and lane_width_px > 0
        ):
            threshold = self.cfg.relative_jump_frac * float(lane_width_px)
        else:
            threshold = self.cfg.jump_px_threshold

        jump = smoothed - baseline

        # ---------------------------------------------------
        # DETECT
        # ---------------------------------------------------

        if abs(jump) < threshold:
            self._history.append(smoothed)
            return None

        # ---------------------------------------------------
        # DIRECTION (inverted because lane centre moves opposite
        # to ego motion when measured in the fixed image frame)
        # ---------------------------------------------------

        direction = "LEFT" if jump > 0 else "RIGHT"

        event = LaneChangeEvent(
            frame=frame,
            timestamp_s=timestamp_s,
            direction=direction,
            magnitude=float(abs(jump)),
            jump_px=float(jump),
        )

        # ---------------------------------------------------
        # POST-EVENT STATE
        # ---------------------------------------------------

        self._cooldown = self.cfg.cooldown_samples

        # Reset history so we don't immediately re-fire on the
        # same jump. Seed history with the current smoothed value.
        self._history.clear()
        self._history.append(smoothed)

        return event