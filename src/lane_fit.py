from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


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

    left_base_shift_px: int = 0
    right_base_shift_px: int = 0

    # Temporal tracking
    tracking_enabled: bool = True
    tracking_margin_px: int = 70

    # Debug verbosity
    debug: bool = False

    @classmethod
    def from_dict(cls, d):

        d = d or {}

        return cls(
            n_windows=int(d.get("n_windows", 20)),
            window_width_frac=float(d.get("window_width_frac", 0.15)),
            min_pixels_to_recenter=int(d.get("min_pixels_to_recenter", 12)),
            min_pixels_total=int(d.get("min_pixels_total", 100)),
            min_pixels_per_window=int(d.get("min_pixels_per_window", 6)),
            max_fit_rms_px=float(d.get("max_fit_rms_px", 25.0)),
            vanishing_x_frac=float(d.get("vanishing_x_frac", 0.50)),
            left_max_x_frac=float(d.get("left_max_x_frac", 0.58)),
            right_min_x_frac=float(d.get("right_min_x_frac", 0.42)),
            max_recenter_jump_frac=float(d.get("max_recenter_jump_frac", 0.10)),
            check_curve_direction=bool(d.get("check_curve_direction", True)),
            left_base_shift_px=int(d.get("left_base_shift_px", 0)),
            right_base_shift_px=int(d.get("right_base_shift_px", 0)),
            tracking_enabled=bool(d.get("tracking_enabled", True)),
            tracking_margin_px=int(d.get("tracking_margin_px", 70)),
            debug=bool(d.get("debug", False)),
        )


class LaneFitter:

    def __init__(self, config=None):

        if config is None:
            self.cfg = SlidingWindowConfig()

        elif isinstance(config, SlidingWindowConfig):
            self.cfg = config

        elif isinstance(config, dict):
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

        bottom_start = int(h * 0.70)

        histogram = np.sum(
            binary[bottom_start:, :] > 0,
            axis=0
        ).astype(np.float32)

        kernel_size = 15

        if w >= kernel_size:

            kernel = np.ones(
                kernel_size,
                dtype=np.float32
            )

            kernel /= kernel.sum()

            histogram_smooth = np.convolve(
                histogram,
                kernel,
                mode="same"
            )

        else:
            histogram_smooth = histogram

        # ----------------------------------------------------
        # LEFT base — peak in left 45% of frame
        # ----------------------------------------------------

        left_end = int(w * 0.45)

        left_region = histogram_smooth[:left_end]

        if np.max(left_region) > 0:
            left_base = int(np.argmax(left_region))
        else:
            left_base = int(w * 0.25)

        left_base += self.cfg.left_base_shift_px

        left_base = int(np.clip(left_base, 0, w - 1))

        # ----------------------------------------------------
        # RIGHT base — peak in right 45% of frame
        # ----------------------------------------------------

        right_start = int(w * 0.55)

        right_region = histogram_smooth[right_start:]

        if np.max(right_region) > 0:
            right_base = int(np.argmax(right_region)) + right_start
        else:
            right_base = int(w * 0.75)

        right_base += self.cfg.right_base_shift_px

        right_base = int(np.clip(right_base, 0, w - 1))

        return left_base, right_base, histogram_smooth

    # ========================================================
    # COLLECT SIDE PIXELS
    # ========================================================

    def _collect_side_pixels(
        self,
        binary: np.ndarray,
        x_base: int,
        side: str,
        previous_coeffs: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        h, w = binary.shape[:2]

        nonzero_y, nonzero_x = np.nonzero(binary > 0)

        if len(nonzero_x) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32)
            )

        n_windows = self.cfg.n_windows

        window_height = max(1, h // n_windows)

        window_width = max(
            10,
            int(w * self.cfg.window_width_frac)
        )

        collected_x = []
        collected_y = []

        current_x = int(x_base)

        for window in range(n_windows):

            y_high = h - window * window_height

            y_low = max(
                0,
                h - (window + 1) * window_height
            )

            # ------------------------------------------------
            # TEMPORAL TRACKING
            # ------------------------------------------------

            if (
                self.cfg.tracking_enabled
                and previous_coeffs is not None
            ):

                y_center = (y_low + y_high) / 2.0

                predicted_x = float(
                    np.polyval(previous_coeffs, y_center)
                )

                predicted_x = float(
                    np.clip(predicted_x, 0, w - 1)
                )

                current_x = int(round(predicted_x))

                search_half_width = self.cfg.tracking_margin_px

            else:

                search_half_width = window_width // 2

            search_x_low = max(0, current_x - search_half_width)

            search_x_high = min(w, current_x + search_half_width)

            region = (
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= search_x_low)
                & (nonzero_x < search_x_high)
            )

            candidate_x = nonzero_x[region]
            candidate_y = nonzero_y[region]

            if len(candidate_x) == 0:
                continue

            # ------------------------------------------------
            # TEMPORAL PIXEL FILTER
            # ------------------------------------------------

            if (
                self.cfg.tracking_enabled
                and previous_coeffs is not None
            ):

                predicted_for_pixels = np.polyval(
                    previous_coeffs,
                    candidate_y
                )

                distance = np.abs(candidate_x - predicted_for_pixels)

                pixel_limit = window_width / 2.0

                keep = distance <= pixel_limit

                candidate_x = candidate_x[keep]
                candidate_y = candidate_y[keep]

            if len(candidate_x) == 0:
                continue

            collected_x.append(candidate_x)
            collected_y.append(candidate_y)

            # ------------------------------------------------
            # RECENTER
            # ------------------------------------------------

            if len(candidate_x) >= self.cfg.min_pixels_to_recenter:

                new_x = int(np.median(candidate_x))

                max_jump = int(
                    w * self.cfg.max_recenter_jump_frac
                )

                if abs(new_x - current_x) <= max_jump:
                    current_x = new_x

        if len(collected_x) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32)
            )

        x_pixels = np.concatenate(collected_x)
        y_pixels = np.concatenate(collected_y)

        return x_pixels, y_pixels

    # ========================================================
    # LINEAR FIT
    # ========================================================

    def _fit_poly(
        self,
        x_pixels: np.ndarray,
        y_pixels: np.ndarray
    ):

        if len(x_pixels) < self.cfg.min_pixels_total:
            return None, float("inf")

        try:

            # ------------------------------------------------
            # LINEAR MODEL:  x = m*y + b
            # ------------------------------------------------

            linear_coeffs = np.polyfit(
                y_pixels,
                x_pixels,
                1
            )

            m, b = linear_coeffs

            # ------------------------------------------------
            # KEEP THE EXISTING 3-COEFFICIENT INTERFACE
            # x = 0*y² + m*y + b
            # ------------------------------------------------

            coeffs = np.array(
                [0.0, float(m), float(b)],
                dtype=np.float64
            )

        except (np.linalg.LinAlgError, ValueError):

            return None, float("inf")

        predicted_x = np.polyval(coeffs, y_pixels)

        residuals = x_pixels - predicted_x

        rms = float(np.sqrt(np.mean(residuals ** 2)))

        return coeffs, rms

    # ========================================================
    # LINE DIRECTION
    # ========================================================

    def _check_curve_direction(
        self,
        coeffs,
        height,
        side
    ):

        if not self.cfg.check_curve_direction:
            return True

        # ----------------------------------------------------
        # LINEAR MODEL
        #
        # coeffs = [0, m, b]
        #
        # For this camera/video:
        #   LEFT  lane boundary -> negative slope (m < 0)
        #   RIGHT lane boundary -> positive slope (m > 0)
        #
        # Verified against lane_hough.py which uses the
        # same convention.
        #
        # Reason: y increases downward in image coordinates.
        #   LEFT lane starts bottom-left, converges up-right.
        #     dy < 0, dx > 0  →  slope = dy/dx < 0.
        #   RIGHT lane starts bottom-right, converges up-left.
        #     dy < 0, dx < 0  →  slope = dy/dx > 0.
        # ----------------------------------------------------

        if len(coeffs) == 3:
            m = float(coeffs[1])
        elif len(coeffs) == 2:
            m = float(coeffs[0])
        else:
            return False

        if side == "left":
            return m < 0
        elif side == "right":
            return m > 0

        return True

    # ========================================================
    # TEMPORAL CONSISTENCY
    # ========================================================

    def _check_temporal_consistency(
        self,
        candidate_coeffs,
        previous_coeffs,
        height
    ):

        if previous_coeffs is None:
            return True

        y_values = np.linspace(
            int(height * 0.62),
            int(height * 0.95),
            30
        )

        previous_x = np.polyval(previous_coeffs, y_values)
        candidate_x = np.polyval(candidate_coeffs, y_values)

        displacement = np.abs(candidate_x - previous_x)

        max_displacement = float(np.max(displacement))

        return max_displacement <= self.cfg.tracking_margin_px

    # ========================================================
    # SIDE FIT
    # ========================================================

    def _fit_side(
        self,
        binary,
        x_base,
        side,
        previous_coeffs=None
    ):

        h, w = binary.shape[:2]

        x_pixels, y_pixels = self._collect_side_pixels(
            binary,
            x_base,
            side,
            previous_coeffs
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

        if self.cfg.debug:
            print(f"  [fit-{side}] collected n_pixels={n_pixels} "
                  f"x_base={x_base}")

        # ----------------------------------------------------
        # CHECK 1 — minimum pixels
        # ----------------------------------------------------

        if n_pixels < self.cfg.min_pixels_total:
            if self.cfg.debug:
                print(f"  [fit-{side}] REJECT: n_pixels {n_pixels} < "
                      f"min_pixels_total {self.cfg.min_pixels_total}")
            return result

        coeffs, rms = self._fit_poly(x_pixels, y_pixels)

        result.rms_error = rms

        if coeffs is None:
            if self.cfg.debug:
                print(f"  [fit-{side}] REJECT: polyfit returned None")
            return result

        if self.cfg.debug:
            print(f"  [fit-{side}] fitted m={coeffs[1]:+.4f} "
                  f"b={coeffs[2]:+.2f} rms={rms:.2f}")

        # ----------------------------------------------------
        # CHECK 2 — RMS
        # ----------------------------------------------------

        if rms > self.cfg.max_fit_rms_px:
            if self.cfg.debug:
                print(f"  [fit-{side}] REJECT: rms {rms:.2f} > "
                      f"max_fit_rms_px {self.cfg.max_fit_rms_px}")
            return result

        # ----------------------------------------------------
        # CHECK 3 — curve direction
        # ----------------------------------------------------

        if not self._check_curve_direction(coeffs, h, side):
            if self.cfg.debug:
                print(f"  [fit-{side}] REJECT: direction check failed "
                      f"(m={coeffs[1]:+.4f}, "
                      f"expected {'m<0' if side == 'left' else 'm>0'})")
            return result

        # ----------------------------------------------------
        # CHECK 4 — temporal consistency
        # ----------------------------------------------------

        if (
            self.cfg.tracking_enabled
            and previous_coeffs is not None
        ):

            if not self._check_temporal_consistency(
                coeffs,
                previous_coeffs,
                h
            ):
                if self.cfg.debug:
                    print(f"  [fit-{side}] REJECT: temporal consistency "
                          f"failed vs previous")
                return result

        # ----------------------------------------------------
        # ACCEPT
        # ----------------------------------------------------

        if self.cfg.debug:
            print(f"  [fit-{side}] ACCEPT")

        result.coeffs = coeffs

        return result

    # ========================================================
    # PUBLIC FIT
    # ========================================================

    def fit(
        self,
        binary,
        previous_left=None,
        previous_right=None
    ):

        if binary is None:
            return (LaneFitResult(), LaneFitResult())

        if binary.ndim != 2:
            raise ValueError(
                "LaneFitter.fit() expects a binary 2D image."
            )

        binary = (binary > 0).astype(np.uint8) * 255

        left_base, right_base, _ = self._histogram_base(binary)

        if self.cfg.debug:
            print(f"[LaneFitter] bases: left_base={left_base} "
                  f"right_base={right_base}")

        left_result = self._fit_side(
            binary,
            left_base,
            "left",
            previous_left
        )

        right_result = self._fit_side(
            binary,
            right_base,
            "right",
            previous_right
        )

        return (left_result, right_result)

    # ========================================================
    # DEBUG RENDER
    # ========================================================

    def debug_render(
        self,
        binary,
        left,
        right
    ):

        h, w = binary.shape[:2]

        if binary.ndim == 2:
            canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            canvas = binary.copy()

        if (left.x_pixels is not None
                and left.y_pixels is not None):

            for x, y in zip(left.x_pixels, left.y_pixels):
                cv2.circle(canvas, (int(x), int(y)), 1,
                           (255, 255, 255), -1)

        if (right.x_pixels is not None
                and right.y_pixels is not None):

            for x, y in zip(right.x_pixels, right.y_pixels):
                cv2.circle(canvas, (int(x), int(y)), 1,
                           (255, 255, 255), -1)

        y_values = np.linspace(
            int(h * 0.55),
            int(h * 0.95),
            100
        ).astype(np.int32)

        if left.coeffs is not None:
            x_values = np.polyval(left.coeffs, y_values)
            points = np.column_stack(
                (x_values, y_values)
            ).astype(np.int32)

            for i in range(len(points) - 1):
                cv2.line(canvas, tuple(points[i]),
                         tuple(points[i + 1]),
                         (0, 255, 0), 3)

        if right.coeffs is not None:
            x_values = np.polyval(right.coeffs, y_values)
            points = np.column_stack(
                (x_values, y_values)
            ).astype(np.int32)

            for i in range(len(points) - 1):
                cv2.line(canvas, tuple(points[i]),
                         tuple(points[i + 1]),
                         (0, 0, 255), 3)

        return canvas