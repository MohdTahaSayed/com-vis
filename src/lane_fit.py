from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SlidingWindowConfig:
    n_windows: int = 9
    window_width_frac: float = 0.12

    min_pixels_to_recenter: int = 12
    min_pixels_per_window: int = 6
    min_pixels_total: int = 80

    max_fit_rms_px: float = 30.0

    # Keep left/right windows on their respective sides.
    left_max_x_frac: float = 0.58
    right_min_x_frac: float = 0.42

    # Prevent a window from jumping onto another object/edge.
    max_recenter_jump_frac: float = 0.12

    # Lane markings in this video may not cover 40% of the frame.
    min_vertical_coverage_frac: float = 0.25

    check_curve_direction: bool = True

    @classmethod
    def from_dict(cls, d: Dict) -> "SlidingWindowConfig":
        d = d or {}
        c = cls()

        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)

        return c


@dataclass
class FitResult:
    side: str
    coeffs: Optional[np.ndarray]
    pixels: np.ndarray
    confidence: float
    n_pixels: int
    x_base: float


class LaneFitter:

    def __init__(self, cfg: SlidingWindowConfig):
        self.cfg = cfg

    # ---------------------------------------------------------
    # HISTOGRAM BASE
    # ---------------------------------------------------------

    def _histogram_base(self, mask: np.ndarray) -> Tuple[int, int]:

        h, w = mask.shape[:2]

        bottom = mask[int(h * 0.70):, :]

        hist = np.sum(
            bottom > 0,
            axis=0
        ).astype(np.float32)

        kernel = np.ones(
            15,
            dtype=np.float32
        ) / 15.0

        hist = np.convolve(
            hist,
            kernel,
            mode="same"
        )

        left_lo = 0
        left_hi = int(0.45 * w)

        right_lo = int(0.55 * w)
        right_hi = w

        left_hist = hist[left_lo:left_hi]
        right_hist = hist[right_lo:right_hi]

        if left_hist.max() > 0:
            x_left = left_lo + int(
                np.argmax(left_hist)
            )
        else:
            x_left = int(0.25 * w)

        if right_hist.max() > 0:
            x_right = right_lo + int(
                np.argmax(right_hist)
            )
        else:
            x_right = int(0.75 * w)

        return x_left, x_right

    # ---------------------------------------------------------
    # SLIDING WINDOWS
    # ---------------------------------------------------------

    def _sliding_window(
        self,
        mask: np.ndarray,
        x_base: int,
        side: str,
    ) -> np.ndarray:

        h, w = mask.shape[:2]

        n = self.cfg.n_windows

        win_w = max(
            24,
            int(self.cfg.window_width_frac * w)
        )

        win_h = h // n

        max_jump = int(
            self.cfg.max_recenter_jump_frac * w
        )

        nonzero_y, nonzero_x = np.nonzero(mask)

        current_x = int(x_base)

        collected: List[np.ndarray] = []

        for i in range(n):

            y_low = h - (i + 1) * win_h
            y_high = h - i * win_h

            x_low = current_x - win_w // 2
            x_high = current_x + win_w // 2

            # Keep windows on their respective side.
            if side == "left":
                x_high = min(
                    x_high,
                    int(self.cfg.left_max_x_frac * w)
                )
            else:
                x_low = max(
                    x_low,
                    int(self.cfg.right_min_x_frac * w)
                )

            if x_high <= x_low:
                continue

            selection = (
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= x_low)
                & (nonzero_x < x_high)
            )

            ys = nonzero_y[selection]
            xs = nonzero_x[selection]

            if len(xs) < self.cfg.min_pixels_per_window:
                continue

            collected.append(
                np.stack(
                    [xs, ys],
                    axis=1
                ).astype(np.int32)
            )

            # Recenter only when enough pixels exist.
            if len(xs) >= self.cfg.min_pixels_to_recenter:

                proposed_x = int(
                    np.median(xs)
                )

                jump = proposed_x - current_x

                if abs(jump) <= max_jump:
                    current_x = proposed_x

        if not collected:
            return np.zeros(
                (0, 2),
                dtype=np.int32
            )

        pixels = np.concatenate(
            collected,
            axis=0
        )

        # -----------------------------------------------------
        # VERTICAL COVERAGE
        # -----------------------------------------------------

        y_span = (
            float(pixels[:, 1].max())
            - float(pixels[:, 1].min())
        )

        coverage = y_span / max(
            float(h),
            1.0
        )

        if coverage < self.cfg.min_vertical_coverage_frac:
            return np.zeros(
                (0, 2),
                dtype=np.int32
            )

        return pixels

    # ---------------------------------------------------------
    # POLYNOMIAL FIT
    # ---------------------------------------------------------

    def _fit_poly(
        self,
        pixels: np.ndarray,
        side: str,
        frame_height: int,
    ) -> Tuple[Optional[np.ndarray], float]:

        if pixels.shape[0] < 6:
            return None, float("inf")

        x = pixels[:, 0].astype(np.float64)
        y = pixels[:, 1].astype(np.float64)

        # Use the actual frame height instead of hardcoding 576.
        y_span = y.max() - y.min()

        if y_span < self.cfg.min_vertical_coverage_frac * frame_height:
            return None, float("inf")

        try:

            with warnings.catch_warnings():

                warnings.simplefilter("ignore")

                coeffs = np.polyfit(
                    y,
                    x,
                    deg=2
                )

        except (np.linalg.LinAlgError, ValueError):

            return None, float("inf")

        x_pred = (
            coeffs[0] * y * y
            + coeffs[1] * y
            + coeffs[2]
        )

        rms = float(
            np.sqrt(
                np.mean(
                    (x_pred - x) ** 2
                )
            )
        )

        if rms > self.cfg.max_fit_rms_px:
            return None, rms

        # -----------------------------------------------------
        # CORRECT CURVE DIRECTION
        # -----------------------------------------------------

        if self.cfg.check_curve_direction:

            derivative = (
                2.0 * coeffs[0] * y
                + coeffs[1]
            )

            median_slope = float(
                np.median(derivative)
            )

            # Image coordinates:
            #
            # LEFT lane:
            # bottom is farther LEFT than the top
            # therefore dx/dy < 0
            #
            # RIGHT lane:
            # bottom is farther RIGHT than the top
            # therefore dx/dy > 0

            if side == "left" and median_slope >= 0:
                return None, rms

            if side == "right" and median_slope <= 0:
                return None, rms

        return coeffs, rms

    # ---------------------------------------------------------
    # MAIN FIT
    # ---------------------------------------------------------

    def fit(
        self,
        mask: np.ndarray
    ) -> Tuple[FitResult, FitResult]:

        h, w = mask.shape[:2]

        x_left_base, x_right_base = (
            self._histogram_base(mask)
        )

        left_pixels = self._sliding_window(
            mask,
            x_left_base,
            "left"
        )

        right_pixels = self._sliding_window(
            mask,
            x_right_base,
            "right"
        )

        left_coeffs, left_rms = self._fit_poly(
            left_pixels,
            "left",
            h
        )

        right_coeffs, right_rms = self._fit_poly(
            right_pixels,
            "right",
            h
        )

        def confidence(
            n_pixels: int,
            rms: float
        ) -> float:

            if rms == float("inf"):
                return 0.0

            base = min(
                1.0,
                n_pixels /
                float(self.cfg.min_pixels_total)
            )

            rms_factor = max(
                0.0,
                1.0 - (
                    rms /
                    max(
                        self.cfg.max_fit_rms_px,
                        1.0
                    )
                )
            )

            return float(
                base *
                (0.5 + 0.5 * rms_factor)
            )

        left = FitResult(
            side="left",
            coeffs=left_coeffs,
            pixels=left_pixels,
            confidence=confidence(
                len(left_pixels),
                left_rms
            ),
            n_pixels=len(left_pixels),
            x_base=x_left_base,
        )

        right = FitResult(
            side="right",
            coeffs=right_coeffs,
            pixels=right_pixels,
            confidence=confidence(
                len(right_pixels),
                right_rms
            ),
            n_pixels=len(right_pixels),
            x_base=x_right_base,
        )

        return left, right

    # ---------------------------------------------------------
    # EVALUATE
    # ---------------------------------------------------------

    @staticmethod
    def evaluate(
        coeffs: np.ndarray,
        ys: np.ndarray
    ) -> np.ndarray:

        return (
            coeffs[0] * ys * ys
            + coeffs[1] * ys
            + coeffs[2]
        )

    # ---------------------------------------------------------
    # DEBUG RENDER
    # ---------------------------------------------------------

    def debug_render(
        self,
        frame: np.ndarray,
        left: FitResult,
        right: FitResult
    ) -> np.ndarray:

        out = frame.copy()

        # Selected left pixels
        for x, y in left.pixels:

            cv2.circle(
                out,
                (int(x), int(y)),
                1,
                (0, 180, 255),
                -1
            )

        # Selected right pixels
        for x, y in right.pixels:

            cv2.circle(
                out,
                (int(x), int(y)),
                1,
                (255, 120, 0),
                -1
            )

        ys = np.linspace(
            int(frame.shape[0] * 0.40),
            frame.shape[0] - 1,
            200
        )

        if left.coeffs is not None:

            xs = self.evaluate(
                left.coeffs,
                ys
            )

            pts = np.column_stack(
                [xs, ys]
            ).astype(np.int32)

            cv2.polylines(
                out,
                [pts],
                False,
                (0, 255, 255),
                3
            )

        if right.coeffs is not None:

            xs = self.evaluate(
                right.coeffs,
                ys
            )

            pts = np.column_stack(
                [xs, ys]
            ).astype(np.int32)

            cv2.polylines(
                out,
                [pts],
                False,
                (0, 255, 0),
                3
            )

        return out