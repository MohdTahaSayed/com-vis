

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class SlidingWindowConfig:
    n_windows: int = 9
    window_width_frac: float = 0.10

    min_pixels_to_recenter: int = 12
    min_pixels_total: int = 100
    min_pixels_per_window: int = 6

    max_fit_rms_px: float = 25.0

    # Geometry constraints
    vanishing_x_frac: float = 0.50
    left_max_x_frac: float = 0.58
    right_min_x_frac: float = 0.42

    # Prevent sudden sliding-window jumps
    max_recenter_jump_frac: float = 0.10

    # Check expected lane geometry
    check_curve_direction: bool = True

    @classmethod
    def from_dict(cls, d):
        """
        Build configuration from the YAML dictionary.

        Unknown/unused parameters are ignored for compatibility with
        older versions of the project configuration.
        """

        if d is None:
            return cls()

        valid_fields = {
            "n_windows",
            "window_width_frac",
            "min_pixels_to_recenter",
            "min_pixels_total",
            "min_pixels_per_window",
            "max_fit_rms_px",
            "vanishing_x_frac",
            "left_max_x_frac",
            "right_min_x_frac",
            "max_recenter_jump_frac",
            "check_curve_direction",
        }

        clean = {
            key: value
            for key, value in d.items()
            if key in valid_fields
        }

        return cls(**clean)


# =====================================================================
# Fit result
# =====================================================================

@dataclass
class FitResult:
    side: str
    coeffs: Optional[np.ndarray]
    pixels: int
    confidence: float
    n_pixels: int
    x_base: int


# =====================================================================
# Lane Fitter
# =====================================================================

class LaneFitter:

    def __init__(self, cfg: SlidingWindowConfig):
        self.cfg = cfg

    # -----------------------------------------------------------------
    # Histogram base detection
    # -----------------------------------------------------------------

    def _histogram_base(
        self,
        mask: np.ndarray
    ) -> Tuple[int, int]:
        """
        Find approximate starting x positions for the left and
        right lane lines.

        We use the bottom portion of the image because lane markings
        are generally strongest there.
        """

        h, w = mask.shape[:2]

        # Bottom 30% of image
        bottom_start = int(h * 0.70)

        bottom = mask[bottom_start:, :]

        # Count active pixels at each x-coordinate
        histogram = np.sum(
            bottom > 0,
            axis=0
        ).astype(np.float32)

        # Smooth histogram
        if len(histogram) >= 15:

            kernel = (
                np.ones(15, dtype=np.float32)
                / 15.0
            )

            histogram = np.convolve(
                histogram,
                kernel,
                mode="same"
            )

        # -------------------------------------------------------------
        # LEFT BASE
        # -------------------------------------------------------------

        left_end = int(w * 0.45)

        if left_end > 0:

            left_region = histogram[:left_end]

            if np.max(left_region) > 0:

                left_base = int(
                    np.argmax(left_region)
                )

            else:

                left_base = int(
                    w * 0.25
                )

        else:

            left_base = int(
                w * 0.25
            )

        # -------------------------------------------------------------
        # RIGHT BASE
        # -------------------------------------------------------------

        right_start = int(w * 0.55)

        if right_start < w:

            right_region = histogram[right_start:]

            if np.max(right_region) > 0:

                right_base = int(
                    right_start
                    + np.argmax(right_region)
                )

            else:

                right_base = int(
                    w * 0.75
                )

        else:

            right_base = int(
                w * 0.75
            )

        return left_base, right_base

    # -----------------------------------------------------------------
    # Sliding-window pixel collection
    # -----------------------------------------------------------------

    def _sliding_window(
        self,
        mask: np.ndarray,
        x_base: int,
        side: str,
    ) -> Tuple[np.ndarray, int]:
        """
        Collect lane pixels using sliding windows.

        Windows move from the bottom of the image toward the top.

        Returns:
            pixels: Nx2 array of [x, y]
            x_base: starting x-coordinate
        """

        h, w = mask.shape[:2]

        # Get coordinates of all non-zero pixels
        nonzero_y, nonzero_x = np.nonzero(mask)

        if len(nonzero_x) == 0:

            return (
                np.empty(
                    (0, 2),
                    dtype=np.int32
                ),
                x_base,
            )

        # Window width
        window_width = max(
            10,
            int(
                w * self.cfg.window_width_frac
            )
        )

        half_width = window_width // 2

        # Height of each window
        window_height = max(
            1,
            h // self.cfg.n_windows
        )

        current_x = int(x_base)

        previous_x = current_x

        collected_x = []
        collected_y = []

        # Maximum allowed recenter jump
        max_jump = int(
            w * self.cfg.max_recenter_jump_frac
        )

        # -------------------------------------------------------------
        # Bottom → top
        # -------------------------------------------------------------

        for window_idx in range(
            self.cfg.n_windows
        ):

            y_high = (
                h
                - window_idx * window_height
            )

            y_low = max(
                0,
                h
                - (window_idx + 1)
                * window_height
            )

            x_low = max(
                0,
                current_x - half_width
            )

            x_high = min(
                w,
                current_x + half_width
            )

            # Pixels inside current window
            good_indices = (
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= x_low)
                & (nonzero_x < x_high)
            )

            good_x = nonzero_x[
                good_indices
            ]

            good_y = nonzero_y[
                good_indices
            ]

            # ---------------------------------------------------------
            # Side-specific filtering
            # ---------------------------------------------------------

            if side == "left":

                max_x = int(
                    w * self.cfg.left_max_x_frac
                )

                keep = good_x <= max_x

                good_x = good_x[keep]
                good_y = good_y[keep]

            elif side == "right":

                min_x = int(
                    w * self.cfg.right_min_x_frac
                )

                keep = good_x >= min_x

                good_x = good_x[keep]
                good_y = good_y[keep]

            # ---------------------------------------------------------
            # Store pixels
            # ---------------------------------------------------------

            if len(good_x) > 0:

                collected_x.extend(
                    good_x.tolist()
                )

                collected_y.extend(
                    good_y.tolist()
                )

            # ---------------------------------------------------------
            # Recenter window
            # ---------------------------------------------------------

            if (
                len(good_x)
                >= self.cfg.min_pixels_to_recenter
            ):

                new_x = int(
                    np.median(good_x)
                )

                jump = abs(
                    new_x - previous_x
                )

                if jump <= max_jump:

                    current_x = new_x
                    previous_x = new_x

                else:

                    # Ignore sudden jump
                    current_x = previous_x

        # No pixels found
        if len(collected_x) == 0:

            return (
                np.empty(
                    (0, 2),
                    dtype=np.int32
                ),
                x_base,
            )

        pixels = np.column_stack(
            [
                np.asarray(
                    collected_x,
                    dtype=np.int32
                ),
                np.asarray(
                    collected_y,
                    dtype=np.int32
                ),
            ]
        )

        return pixels, x_base

    # -----------------------------------------------------------------
    # Polynomial fitting
    # -----------------------------------------------------------------

    def _fit_poly(
        self,
        pixels: np.ndarray,
        side: str,
    ) -> Tuple[
        Optional[np.ndarray],
        float
    ]:
        """
        Fit:

            x = a*y^2 + b*y + c

        using detected lane pixels.
        """

        if (
            pixels is None
            or len(pixels)
            < self.cfg.min_pixels_total
        ):

            return None, float("inf")

        x = pixels[:, 0].astype(
            np.float64
        )

        y = pixels[:, 1].astype(
            np.float64
        )

        # Need at least 3 unique y values
        if len(np.unique(y)) < 3:

            return None, float("inf")

        # -------------------------------------------------------------
        # Polynomial fit
        # -------------------------------------------------------------

        try:

            coeffs = np.polyfit(
                y,
                x,
                deg=2
            )

        except Exception:

            return None, float("inf")

        a, b, c = coeffs

        # Predicted x
        predicted_x = (
            a * y * y
            + b * y
            + c
        )

        # Residual error
        residuals = (
            x - predicted_x
        )

        rms = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        # Reject poor polynomial fit
        if not np.isfinite(rms):

            return None, float("inf")

        if (
            rms
            > self.cfg.max_fit_rms_px
        ):

            return None, rms

        # -------------------------------------------------------------
        # Curve direction check
        # -------------------------------------------------------------

        if self.cfg.check_curve_direction:

            # dx/dy = 2*a*y + b
            derivative = (
                2.0 * a * y
                + b
            )

            median_derivative = float(
                np.median(derivative)
            )

            if side == "left":

                # A left lane normally moves toward smaller x
                # as it approaches the top of the image.
                #
                # With y increasing downward, dx/dy should normally
                # be positive.
                if median_derivative < 0:

                    return None, rms

            elif side == "right":

                # Right-lane geometry is deliberately not rejected
                # using a strict derivative sign because camera
                # perspective and road curvature can change it.
                pass

        return coeffs, rms

    # -----------------------------------------------------------------
    # Confidence
    # -----------------------------------------------------------------

    def _confidence(
        self,
        n_pixels: int
    ) -> float:
        """
        Convert number of detected pixels to confidence.

        100+ pixels gives confidence 1.0.
        """

        if n_pixels <= 0:

            return 0.0

        return float(
            min(
                1.0,
                n_pixels
                / float(
                    self.cfg.min_pixels_total
                )
            )
        )

    # -----------------------------------------------------------------
    # Main fit
    # -----------------------------------------------------------------

    def fit(
        self,
        mask: np.ndarray,
    ) -> Tuple[
        FitResult,
        FitResult
    ]:
        """
        Detect and fit left/right lanes.

        Public API:

            left, right = fitter.fit(mask)
        """

        if mask is None:

            raise ValueError(
                "mask cannot be None"
            )

        if len(mask.shape) != 2:

            raise ValueError(
                "mask must be a single-channel "
                "binary image"
            )

        # Convert to binary mask
        binary = np.where(
            mask > 0,
            255,
            0
        ).astype(np.uint8)

        # -------------------------------------------------------------
        # Find lane bases
        # -------------------------------------------------------------

        left_base, right_base = (
            self._histogram_base(binary)
        )

        # -------------------------------------------------------------
        # Sliding windows
        # -------------------------------------------------------------

        left_pixels, left_base = (
            self._sliding_window(
                binary,
                left_base,
                "left",
            )
        )

        right_pixels, right_base = (
            self._sliding_window(
                binary,
                right_base,
                "right",
            )
        )

        # -------------------------------------------------------------
        # Polynomial fits
        # -------------------------------------------------------------

        left_coeffs, left_rms = (
            self._fit_poly(
                left_pixels,
                "left",
            )
        )

        right_coeffs, right_rms = (
            self._fit_poly(
                right_pixels,
                "right",
            )
        )

        # -------------------------------------------------------------
        # Confidence
        # -------------------------------------------------------------

        if left_coeffs is not None:

            left_conf = self._confidence(
                len(left_pixels)
            )

        else:

            left_conf = 0.0

        if right_coeffs is not None:

            right_conf = self._confidence(
                len(right_pixels)
            )

        else:

            right_conf = 0.0

        # -------------------------------------------------------------
        # Results
        # -------------------------------------------------------------

        left_result = FitResult(
            side="left",
            coeffs=left_coeffs,
            pixels=len(left_pixels),
            confidence=left_conf,
            n_pixels=len(left_pixels),
            x_base=int(left_base),
        )

        right_result = FitResult(
            side="right",
            coeffs=right_coeffs,
            pixels=len(right_pixels),
            confidence=right_conf,
            n_pixels=len(right_pixels),
            x_base=int(right_base),
        )

        return (
            left_result,
            right_result
        )

    # -----------------------------------------------------------------
    # Debug rendering
    # -----------------------------------------------------------------

    def debug_render(
        self,
        frame: np.ndarray,
        left: FitResult,
        right: FitResult,
    ) -> np.ndarray:
        """
        Draw detected lane curves on the frame.
        """

        vis = frame.copy()

        h, w = vis.shape[:2]

        # -------------------------------------------------------------
        # Helper to draw one polynomial
        # -------------------------------------------------------------

        def draw_curve(
            result: FitResult,
            color,
        ):

            if (
                result is None
                or result.coeffs is None
            ):

                return

            a, b, c = result.coeffs

            ys = np.linspace(
                int(h * 0.45),
                h - 1,
                100
            )

            xs = (
                a * ys * ys
                + b * ys
                + c
            )

            points = []

            for x, y in zip(
                xs,
                ys
            ):

                if (
                    0 <= x < w
                    and 0 <= y < h
                ):

                    points.append(
                        (
                            int(x),
                            int(y)
                        )
                    )

            if len(points) >= 2:

                pts = np.asarray(
                    points,
                    dtype=np.int32
                )

                cv2.polylines(
                    vis,
                    [pts],
                    False,
                    color,
                    3,
                )

        # Left = yellow
        draw_curve(
            left,
            (0, 255, 255)
        )

        # Right = green
        draw_curve(
            right,
            (0, 255, 0)
        )

        # -------------------------------------------------------------
        # Draw histogram base locations
        # -------------------------------------------------------------

        if left is not None:

            cv2.circle(
                vis,
                (
                    int(left.x_base),
                    h - 10
                ),
                7,
                (0, 165, 255),
                -1,
            )

        if right is not None:

            cv2.circle(
                vis,
                (
                    int(right.x_base),
                    h - 10
                ),
                7,
                (255, 0, 0),
                -1,
            )

        return vis