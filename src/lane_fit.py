"""
Lane polynomial fitting using histogram + sliding windows.

This module:
1. Builds a bottom-region histogram of lane pixels.
2. Finds left/right starting positions.
3. Tracks lane pixels using sliding windows.
4. Fits a quadratic polynomial x = ay^2 + by + c.
5. Rejects only fits that have insufficient pixels or excessive RMS error.

Geometric lane validation is handled separately by lane_validation.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class LaneFitResult:
    """
    Result of fitting one lane boundary.
    """

    coeffs: Optional[np.ndarray] = None
    confidence: float = 0.0
    n_pixels: int = 0
    rms_error: float = 0.0

    x_base: Optional[int] = None

    # Debug information
    x_pixels: Optional[np.ndarray] = None
    y_pixels: Optional[np.ndarray] = None

    @property
    def valid(self) -> bool:
        return self.coeffs is not None


@dataclass
class SlidingWindowConfig:
    """
    Sliding-window configuration.
    """

    n_windows: int = 20
    window_width_frac: float = 0.15

    min_pixels_to_recenter: int = 12
    min_pixels_total: int = 100
    min_pixels_per_window: int = 6

    max_fit_rms_px: float = 25.0

    vanishing_x_frac: float = 0.50

    left_max_x_frac: float = 0.58
    right_min_x_frac: float = 0.42

    max_recenter_jump_frac: float = 0.10

    check_curve_direction: bool = True

    # These are kept configurable because the previous
    # implementation used shifted histogram centers.
    left_base_shift_px: int = 0
    right_base_shift_px: int = 0

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_windows=int(d.get("n_windows", 20)),
            window_width_frac=float(d.get("window_width_frac", 0.15)),
            min_pixels_to_recenter=int(
                d.get("min_pixels_to_recenter", 12)
            ),
            min_pixels_total=int(
                d.get("min_pixels_total", 100)
            ),
            min_pixels_per_window=int(
                d.get("min_pixels_per_window", 6)
            ),
            max_fit_rms_px=float(
                d.get("max_fit_rms_px", 25.0)
            ),
            vanishing_x_frac=float(
                d.get("vanishing_x_frac", 0.50)
            ),
            left_max_x_frac=float(
                d.get("left_max_x_frac", 0.58)
            ),
            right_min_x_frac=float(
                d.get("right_min_x_frac", 0.42)
            ),
            max_recenter_jump_frac=float(
                d.get("max_recenter_jump_frac", 0.10)
            ),
            check_curve_direction=bool(
                d.get("check_curve_direction", True)
            ),
            left_base_shift_px=int(
                d.get("left_base_shift_px", 0)
            ),
            right_base_shift_px=int(
                d.get("right_base_shift_px", 0)
            ),
        )


# ============================================================
# MAIN FITTER
# ============================================================

class LaneFitter:
    """
    Histogram + sliding-window lane detector.

    Input:
        Binary lane-pixel image.

    Output:
        Left and right quadratic lane fits.
    """

    def __init__(self, config=None):

        if config is None:
            self.cfg = SlidingWindowConfig()

        elif isinstance(config, SlidingWindowConfig):
            # test_stage1.py already created the config object
            self.cfg = config

        elif isinstance(config, dict):
            # Config was supplied as a dictionary
            self.cfg = SlidingWindowConfig.from_dict(config)

        else:
            raise TypeError(
                "LaneFitter config must be either "
                "SlidingWindowConfig, dict, or None."
            )

    # ========================================================
    # HISTOGRAM
    # ========================================================

    def _histogram_base(
        self,
        binary: np.ndarray
    ) -> Tuple[int, int, np.ndarray]:

        h, w = binary.shape[:2]

        # Use lower 30% of image.
        # This is where lane lines are usually strongest.
        bottom_start = int(h * 0.70)

        histogram = np.sum(
            binary[bottom_start:, :] > 0,
            axis=0
        ).astype(np.float32)

        # Smooth histogram to reduce isolated peaks.
        kernel_size = 15

        if w >= kernel_size:
            kernel = np.ones(kernel_size, dtype=np.float32)
            kernel /= kernel.sum()

            histogram_smooth = np.convolve(
                histogram,
                kernel,
                mode="same"
            )
        else:
            histogram_smooth = histogram

        # ----------------------------------------------------
        # LEFT HALF
        # ----------------------------------------------------

        left_end = int(w * 0.45)

        left_region = histogram_smooth[:left_end]

        if np.max(left_region) > 0:
            left_base = int(np.argmax(left_region))
        else:
            left_base = int(w * 0.25)

        # Optional configurable shift.
        left_base += self.cfg.left_base_shift_px

        # Keep inside image.
        left_base = int(
            np.clip(left_base, 0, w - 1)
        )

        # ----------------------------------------------------
        # RIGHT HALF
        # ----------------------------------------------------

        right_start = int(w * 0.55)

        right_region = histogram_smooth[right_start:]

        if np.max(right_region) > 0:
            right_base = (
                int(np.argmax(right_region))
                + right_start
            )
        else:
            right_base = int(w * 0.75)

        # Optional configurable shift.
        right_base += self.cfg.right_base_shift_px

        # Keep inside image.
        right_base = int(
            np.clip(right_base, 0, w - 1)
        )

        return (
            left_base,
            right_base,
            histogram_smooth
        )

    # ========================================================
    # SLIDING WINDOWS
    # ========================================================

    def _collect_side_pixels(
        self,
        binary: np.ndarray,
        x_base: int,
        side: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        h, w = binary.shape[:2]

        # Get coordinates of all non-zero pixels.
        nonzero_y, nonzero_x = np.nonzero(binary > 0)

        if len(nonzero_x) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32)
            )

        n_windows = self.cfg.n_windows

        window_height = max(
            1,
            h // n_windows
        )

        window_width = max(
            10,
            int(w * self.cfg.window_width_frac)
        )

        current_x = int(x_base)

        collected_x = []
        collected_y = []

        # ----------------------------------------------------
        # PROCESS FROM BOTTOM TO TOP
        # ----------------------------------------------------

        for window in range(n_windows):

            y_high = h - window * window_height
            y_low = max(
                0,
                h - (window + 1) * window_height
            )

            # Window boundaries.
            x_low = current_x - window_width // 2
            x_high = current_x + window_width // 2

            x_low = max(0, x_low)
            x_high = min(w, x_high)

            # Pixels inside current window.
            good = (
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= x_low)
                & (nonzero_x < x_high)
            )

            good_x = nonzero_x[good]
            good_y = nonzero_y[good]

            if len(good_x) > 0:
                collected_x.append(good_x)
                collected_y.append(good_y)

            # ------------------------------------------------
            # RECENTER
            # ------------------------------------------------

            if len(good_x) >= self.cfg.min_pixels_to_recenter:

                new_x = int(np.mean(good_x))

                # Prevent very large jumps.
                max_jump = int(
                    w * self.cfg.max_recenter_jump_frac
                )

                if abs(new_x - current_x) <= max_jump:
                    current_x = new_x

        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        if len(collected_x) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32)
            )

        x_pixels = np.concatenate(collected_x)
        y_pixels = np.concatenate(collected_y)

        return x_pixels, y_pixels

    # ========================================================
    # POLYNOMIAL FIT
    # ========================================================

    def _fit_poly(
        self,
        x_pixels: np.ndarray,
        y_pixels: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:

        if len(x_pixels) < self.cfg.min_pixels_total:
            return None, float("inf")

        try:
            # x = ay² + by + c
            coeffs = np.polyfit(
                y_pixels,
                x_pixels,
                2
            )
        except (np.linalg.LinAlgError, ValueError):
            return None, float("inf")

        predicted_x = np.polyval(
            coeffs,
            y_pixels
        )

        residuals = x_pixels - predicted_x

        rms = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        return coeffs, rms

    # ========================================================
    # CURVE DIRECTION
    # ========================================================

    def _check_curve_direction(
        self,
        coeffs: np.ndarray,
        height: int,
        side: str
    ) -> bool:

        if not self.cfg.check_curve_direction:
            return True

        y_bottom = int(height * 0.90)
        y_top = int(height * 0.55)

        x_bottom = float(
            np.polyval(coeffs, y_bottom)
        )

        x_top = float(
            np.polyval(coeffs, y_top)
        )

        if side == "left":

            # In image coordinates:
            # left lane should move toward the
            # vanishing point as y decreases.
            return x_bottom < x_top

        elif side == "right":

            return x_bottom > x_top

        return True

    # ========================================================
    # SIDE FIT
    # ========================================================

    def _fit_side(
        self,
        binary: np.ndarray,
        x_base: int,
        side: str
    ) -> LaneFitResult:

        h, w = binary.shape[:2]

        x_pixels, y_pixels = self._collect_side_pixels(
            binary,
            x_base,
            side
        )

        n_pixels = len(x_pixels)

        confidence = min(
            1.0,
            n_pixels / float(
                max(1, self.cfg.min_pixels_total * 3)
            )
        )

        result = LaneFitResult(
            coeffs=None,
            confidence=confidence,
            n_pixels=n_pixels,
            x_base=x_base,
            x_pixels=x_pixels,
            y_pixels=y_pixels
        )

        # ----------------------------------------------------
        # NOT ENOUGH PIXELS
        # ----------------------------------------------------

        if n_pixels < self.cfg.min_pixels_total:
            return result

        # ----------------------------------------------------
        # POLYNOMIAL FIT
        # ----------------------------------------------------

        coeffs, rms = self._fit_poly(
            x_pixels,
            y_pixels
        )

        result.rms_error = rms

        if coeffs is None:
            return result

        # ----------------------------------------------------
        # RMS CHECK
        # ----------------------------------------------------

        if rms > self.cfg.max_fit_rms_px:
            return result

        # ----------------------------------------------------
        # CURVE DIRECTION CHECK
        # ----------------------------------------------------

        if not self._check_curve_direction(
            coeffs,
            h,
            side
        ):
            return result

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT reject the polynomial using the old
        # side-position check here.
        #
        # lane_validation.py is responsible for checking
        # whether the resulting curve is geometrically
        # valid as a lane boundary.
        # ----------------------------------------------------

        result.coeffs = coeffs

        return result

    # ========================================================
    # PUBLIC FIT METHOD
    # ========================================================

    def fit(
        self,
        binary: np.ndarray
    ) -> Tuple[LaneFitResult, LaneFitResult]:

        if binary is None:
            return (
                LaneFitResult(),
                LaneFitResult()
            )

        if binary.ndim != 2:
            raise ValueError(
                "LaneFitter.fit() expects a binary 2D image."
            )

        # Ensure uint8 binary image.
        binary = (
            binary > 0
        ).astype(np.uint8) * 255

        # Find histogram bases.
        left_base, right_base, _ = (
            self._histogram_base(binary)
        )

        # Fit left lane.
        left_result = self._fit_side(
            binary,
            left_base,
            "left"
        )

        # Fit right lane.
        right_result = self._fit_side(
            binary,
            right_base,
            "right"
        )

        return (
            left_result,
            right_result
        )

    # ========================================================
    # DEBUG RENDER
    # ========================================================

    def debug_render(
        self,
        binary: np.ndarray,
        left: LaneFitResult,
        right: LaneFitResult
    ) -> np.ndarray:

        h, w = binary.shape[:2]

        # Convert binary image to BGR.
        if binary.ndim == 2:
            canvas = cv2.cvtColor(
                binary,
                cv2.COLOR_GRAY2BGR
            )
        else:
            canvas = binary.copy()

        # ----------------------------------------------------
        # DRAW COLLECTED PIXELS
        # ----------------------------------------------------

        if (
            left.x_pixels is not None
            and left.y_pixels is not None
        ):
            for x, y in zip(
                left.x_pixels,
                left.y_pixels
            ):
                cv2.circle(
                    canvas,
                    (int(x), int(y)),
                    1,
                    (255, 255, 255),
                    -1
                )

        if (
            right.x_pixels is not None
            and right.y_pixels is not None
        ):
            for x, y in zip(
                right.x_pixels,
                right.y_pixels
            ):
                cv2.circle(
                    canvas,
                    (int(x), int(y)),
                    1,
                    (255, 255, 255),
                    -1
                )

        # ----------------------------------------------------
        # DRAW POLYNOMIALS
        # ----------------------------------------------------

        y_values = np.linspace(
            int(h * 0.55),
            int(h * 0.95),
            100
        ).astype(np.int32)

        if left.coeffs is not None:

            x_values = np.polyval(
                left.coeffs,
                y_values
            )

            points = np.column_stack(
                (x_values, y_values)
            ).astype(np.int32)

            for i in range(len(points) - 1):

                p1 = tuple(points[i])
                p2 = tuple(points[i + 1])

                cv2.line(
                    canvas,
                    p1,
                    p2,
                    (0, 255, 0),
                    3
                )

        if right.coeffs is not None:

            x_values = np.polyval(
                right.coeffs,
                y_values
            )

            points = np.column_stack(
                (x_values, y_values)
            ).astype(np.int32)

            for i in range(len(points) - 1):

                p1 = tuple(points[i])
                p2 = tuple(points[i + 1])

                cv2.line(
                    canvas,
                    p1,
                    p2,
                    (0, 0, 255),
                    3
                )

        return canvas