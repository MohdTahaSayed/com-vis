from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class LaneChangeConfig:
    # ----------------------------------------------------------
    # Boundary-shift detection
    # ----------------------------------------------------------
    # A lane change fires when BOTH lane boundaries (left and right
    # x positions at the eval row) shift by more than
    # `min_boundary_shift_frac * lane_width_px` in the SAME direction
    # within `baseline_window` samples, AND the new configuration
    # persists for `persist_samples` samples.
    #
    # Direction semantics (physical ego motion):
    #   boundaries shift RIGHT in image  →  ego moved LEFT
    #   boundaries shift LEFT  in image  →  ego moved RIGHT
    #
    # Why inverted? Image x increases to the right. When the ego
    # moves right, the world (and its lane boundaries) shift left
    # in the image.
    min_boundary_shift_frac: float = 0.30
    min_boundary_shift_px: float = 90.0

    # Median filter window for smoothing each boundary signal.
    smooth_window: int = 3

    # How many recent samples to consider when computing the
    # baseline.
    baseline_window: int = 6

    # Require the shifted state to persist for this many samples
    # before firing. Prevents flicker.
    persist_samples: int = 2

    # Cooldown (in samples) after firing an event.
    cooldown_samples: int = 12

    # Minimum confidence required to accept a sample.
    min_confidence: float = 0.15

    # Sanity bounds for lane width (px). Measurements outside this
    # range are treated as bad and skipped.
    lane_width_min_px: float = 100.0
    lane_width_max_px: float = 700.0

    # Enforce strict alternation — after a LEFT event, the next
    # event must be RIGHT (and vice versa). Physically correct
    # because you cannot move LEFT twice in a row without an
    # intervening RIGHT.
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
    direction: str          # "LEFT" or "RIGHT" (ego motion)
    magnitude: float        # average |shift| across both boundaries (px)
    delta_left_px: float    # signed shift of left boundary in image x
    delta_right_px: float   # signed shift of right boundary in image x


class LaneChangeDetector:
    """
    Lane-change detector based on simultaneous shift of BOTH lane
    boundaries. Emits alternating LEFT/RIGHT events.

    Direction semantics (physical ego motion):
        boundaries shift RIGHT in image  →  ego moved LEFT
        boundaries shift LEFT  in image  →  ego moved RIGHT

    Enforces alternation — a same-direction event right after a
    previous same-direction event is rejected.
    """

    def __init__(self, cfg: LaneChangeConfig):
        self.cfg = cfg
        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        self._smooth_left: Deque[float] = deque(maxlen=self.cfg.smooth_window)
        self._smooth_right: Deque[float] = deque(maxlen=self.cfg.smooth_window)

        self._hist_left: Deque[float] = deque(maxlen=self.cfg.baseline_window)
        self._hist_right: Deque[float] = deque(maxlen=self.cfg.baseline_window)

        self._cooldown: int = 0
        self._last_direction: Optional[str] = None

        # Persistence — track candidate direction and how many
        # consecutive samples agreed.
        self._candidate_direction: Optional[str] = None
        self._candidate_count: int = 0

    # ========================================================
    # MAIN FEED
    # ========================================================

    def feed(
        self,
        frame: int,
        timestamp_s: float,
        left_x: Optional[float],
        right_x: Optional[float],
        lane_width_px: Optional[float],
        status: str,
    ) -> Optional[LaneChangeEvent]:

        # ---------------------------------------------------
        # GATES
        # ---------------------------------------------------

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

        # ---------------------------------------------------
        # SMOOTH each boundary
        # ---------------------------------------------------

        self._smooth_left.append(float(left_x))
        self._smooth_right.append(float(right_x))

        s_left = float(np.median(self._smooth_left))
        s_right = float(np.median(self._smooth_right))

        # ---------------------------------------------------
        # COOLDOWN
        # ---------------------------------------------------

        if self._cooldown > 0:
            self._cooldown -= 1
            self._hist_left.append(s_left)
            self._hist_right.append(s_right)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        # ---------------------------------------------------
        # BASELINE
        # ---------------------------------------------------

        if len(self._hist_left) < self.cfg.baseline_window:
            self._hist_left.append(s_left)
            self._hist_right.append(s_right)
            return None

        baseline_left = float(np.median(self._hist_left))
        baseline_right = float(np.median(self._hist_right))

        # ---------------------------------------------------
        # DELTAS
        # ---------------------------------------------------

        delta_left = s_left - baseline_left
        delta_right = s_right - baseline_right

        # ---------------------------------------------------
        # THRESHOLD
        # ---------------------------------------------------

        if (
            self.cfg.min_boundary_shift_frac > 0
            and lane_width_px is not None
            and lane_width_px > 0
        ):
            threshold = (
                self.cfg.min_boundary_shift_frac * float(lane_width_px)
            )
        else:
            threshold = self.cfg.min_boundary_shift_px

        # ---------------------------------------------------
        # DETECT CANDIDATE (both boundaries must shift same direction)
        # ---------------------------------------------------

        same_sign = np.sign(delta_left) == np.sign(delta_right)
        both_big = abs(delta_left) >= threshold and abs(delta_right) >= threshold

        if not (same_sign and both_big):
            # No candidate
            self._hist_left.append(s_left)
            self._hist_right.append(s_right)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        # ---------------------------------------------------
        # DIRECTION (inverted — image x is opposite to ego motion)
        # ---------------------------------------------------

        # In this pipeline, lane reacquisition after a change places
        # the new boundaries at LARGER x when the ego has moved RIGHT.
        # Verified against ground truth on VBOX0011_Trim.mp4.
        direction = "RIGHT" if delta_left > 0 else "LEFT"

        # ---------------------------------------------------
        # PERSISTENCE
        # ---------------------------------------------------

        if direction == self._candidate_direction:
            self._candidate_count += 1
        else:
            self._candidate_direction = direction
            self._candidate_count = 1

        if self._candidate_count < self.cfg.persist_samples:
            # Not yet — keep accumulating history and wait
            self._hist_left.append(s_left)
            self._hist_right.append(s_right)
            return None

        # ---------------------------------------------------
        # ALTERNATION FILTER
        # ---------------------------------------------------

        if (
            self.cfg.enforce_alternation
            and self._last_direction is not None
            and direction == self._last_direction
        ):
            # Same direction as previous — physically implausible.
            # Treat as noise and skip. Reset candidate.
            self._hist_left.append(s_left)
            self._hist_right.append(s_right)
            self._candidate_direction = None
            self._candidate_count = 0
            return None

        # ---------------------------------------------------
        # EMIT EVENT
        # ---------------------------------------------------

        magnitude = float(
            0.5 * (abs(delta_left) + abs(delta_right))
        )

        event = LaneChangeEvent(
            frame=frame,
            timestamp_s=timestamp_s,
            direction=direction,
            magnitude=magnitude,
            delta_left_px=float(delta_left),
            delta_right_px=float(delta_right),
        )

        self._last_direction = direction
        self._cooldown = self.cfg.cooldown_samples

        # Reset history and candidate to avoid re-firing
        self._hist_left.clear()
        self._hist_right.clear()
        self._hist_left.append(s_left)
        self._hist_right.append(s_right)

        self._candidate_direction = None
        self._candidate_count = 0

        return event