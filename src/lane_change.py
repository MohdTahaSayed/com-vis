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
    # by more than `jump_px_threshold` within `jump_window_samples`
    # consecutive valid samples.
    #
    # In a typical dashcam view, one lane width is ~250-400 px.
    # Half of that (~150 px) is a safe minimum to distinguish a
    # real lane change from small lateral drift.
    jump_px_threshold: float = 150.0

    # Median filter window — smooths per-sample noise before
    # comparing current lane centre to a recent baseline.
    smooth_window: int = 5

    # How many recent samples to look back when computing the
    # baseline. A larger window means slower but more stable
    # detection.
    baseline_window: int = 15

    # Cooldown (in samples) after a fired event — prevents
    # double-firing on the same lane change.
    cooldown_samples: int = 15

    # If lane_width_px is available, we scale jump threshold by
    # this fraction of lane width. Set to 0 to use the fixed
    # jump_px_threshold instead.
    #
    # Recommended: 0.55  → jump must exceed ~55% of a lane width
    relative_jump_frac: float = 0.55

    # Minimum confidence required to accept a sample.
    min_confidence: float = 0.15

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
    direction: str          # "LEFT" or "RIGHT"
    magnitude: float        # |jump| in pixels (smoothed)
    jump_px: float          # raw signed jump in pixels


class LaneChangeDetector:
    """
    Lane-change detector based on lane-centre motion.

    Consumes per-sample (frame, ts, lane_center_x, lane_width_px,
    status) tuples and emits a LaneChangeEvent whenever the smoothed
    lane centre shifts by more than the configured threshold within
    a short window.

    Direction semantics:
        lane centre moves RIGHT  →  direction = "RIGHT"
        lane centre moves LEFT   →  direction = "LEFT"

    This matches the physical motion of the ego vehicle.
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
        Call once per sample.

        Returns a LaneChangeEvent if a lane change was just detected,
        otherwise None.
        """

        # ---------------------------------------------------
        # GATE
        # ---------------------------------------------------

        if status != "OK" or lane_center_x is None:
            return None

        # ---------------------------------------------------
        # SMOOTH the raw signal
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
        # NEED ENOUGH HISTORY
        # ---------------------------------------------------

        if len(self._history) < self.cfg.baseline_window:
            self._history.append(smoothed)
            return None

        # ---------------------------------------------------
        # BASELINE = median of the recent history (excluding now)
        # ---------------------------------------------------

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

        direction = "RIGHT" if jump > 0 else "LEFT"

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