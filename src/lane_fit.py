

from __future__ import annotations

import warnings

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SlidingWindowConfig:
    n_windows: int = 9
    window_width_frac: float = 0.06
    min_pixels_to_recenter: int = 20
    min_pixels_total: int = 300
    min_pixels_per_window: int = 12
    max_fit_rms_px: float = 18.0

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

    # ------------------------------------------------------------------
    def _histogram_base(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Find starting x positions for left and right lane markings.

        Left search: 0% to 45% of frame width.
        Right search: 55% to 100% of frame width.
        """

        h, w = mask.shape[:2]

        # Use bottom band where lane paint is dense
        bottom = mask[int(h * 0.70):, :]

        # Column histogram
        hist = np.sum(bottom > 0, axis=0).astype(np.float32)

        # Smooth histogram
        k = np.ones(15, dtype=np.float32) / 15.0
        hist = np.convolve(hist, k, mode="same")

        # Search regions
        left_lo = 0
        left_hi = int(0.45 * w)

        right_lo = int(0.55 * w)
        right_hi = w

        left_seg = hist[left_lo:left_hi]
        right_seg = hist[right_lo:right_hi]

        # Left peak
        x_left = left_lo + (
            int(np.argmax(left_seg))
            if left_seg.max() > 0
            else int(0.25 * w)
        )

        # Right peak
        x_right = right_lo + (
            int(np.argmax(right_seg))
            if right_seg.max() > 0
            else int(0.75 * w)
        )

        return x_left, x_right

    # ------------------------------------------------------------------
    def _sliding_window(
        self,
        mask: np.ndarray,
        x_base: int
    ) -> np.ndarray:

        """
        Run vertical sliding windows from bottom to top.

        Windows with fewer than min_pixels_per_window
        pixels are skipped without recentering.
        """

        h, w = mask.shape[:2]

        n = self.cfg.n_windows
        win_w = max(
            20,
            int(self.cfg.window_width_frac * w)
        )

        win_h = h // n

        # np.nonzero returns y first, then x
        nonzero_y, nonzero_x = np.nonzero(mask)

        current_x = x_base

        collected: List[np.ndarray] = []

        for i in range(n):

            y_low = h - (i + 1) * win_h
            y_high = h - i * win_h

            x_low = current_x - win_w // 2
            x_high = current_x + win_w // 2

            # Pixels inside current window
            sel = (
                (nonzero_y >= y_low) &
                (nonzero_y < y_high) &
                (nonzero_x >= x_low) &
                (nonzero_x < x_high)
            )

            ys = nonzero_y[sel]
            xs = nonzero_x[sel]

            # Skip weak/empty window
            if len(xs) < self.cfg.min_pixels_per_window:
                continue

            # Only recenter if we have enough pixels to trust the median
            if len(xs) >= self.cfg.min_pixels_to_recenter:
                current_x = int(np.median(xs))

            # Store pixels for this window
            collected.append(
                np.stack([xs, ys], axis=1).astype(np.int32)
            )

        if not collected:
            return np.zeros(
                (0, 2),
                dtype=np.int32
            )

        return np.concatenate(collected, axis=0)

    # ------------------------------------------------------------------
    def _fit_poly(
        self,
        pixels: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:

        """
        Fit:

            x = a*y^2 + b*y + c

        Returns:
            (coeffs or None, rms_residual_px)
        """

        if pixels.shape[0] < 6:
            return None, float("inf")

        x = pixels[:, 0].astype(np.float64)
        y = pixels[:, 1].astype(np.float64)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(y, x, deg=2)
        except (np.linalg.LinAlgError, ValueError):
            return None, float("inf")

        # Predicted x coordinates
        x_pred = (
            coeffs[0] * y * y +
            coeffs[1] * y +
            coeffs[2]
        )

        # Root Mean Square residual
        rms = float(
            np.sqrt(
                np.mean(
                    (x_pred - x) ** 2
                )
            )
        )

        # Reject poor fit
        if rms > self.cfg.max_fit_rms_px:
            return None, rms

        return coeffs, rms

    # ------------------------------------------------------------------
    def fit(
        self,
        mask: np.ndarray
    ) -> Tuple[FitResult, FitResult]:

        """
        Run fitting on both lane sides.

        Returns:
            (left_result, right_result)
        """

        h, w = mask.shape[:2]

        # Starting positions
        x_left_base, x_right_base = self._histogram_base(mask)

        # Sliding window search
        left_pixels = self._sliding_window(
            mask,
            x_left_base
        )

        right_pixels = self._sliding_window(
            mask,
            x_right_base
        )

        # Polynomial fit + RMS
        left_coeffs, left_rms = self._fit_poly(
            left_pixels
        )

        right_coeffs, right_rms = self._fit_poly(
            right_pixels
        )

        def conf(
            n_pix: int,
            rms: float
        ) -> float:

            if rms == float("inf"):
                return 0.0

            if self.cfg.min_pixels_total <= 0:
                return 0.0

            return float(
                min(
                    1.0,
                    n_pix / float(
                        self.cfg.min_pixels_total
                    )
                )
            )

        # Left result
        lr = FitResult(
            side="left",
            coeffs=left_coeffs,
            pixels=left_pixels,
            confidence=conf(
                len(left_pixels),
                left_rms
            ),
            n_pixels=len(left_pixels),
            x_base=x_left_base,
        )

        # Right result
        rr = FitResult(
            side="right",
            coeffs=right_coeffs,
            pixels=right_pixels,
            confidence=conf(
                len(right_pixels),
                right_rms
            ),
            n_pixels=len(right_pixels),
            x_base=x_right_base,
        )

        return lr, rr

    # ------------------------------------------------------------------
    @staticmethod
    def evaluate(
        coeffs: np.ndarray,
        ys: np.ndarray
    ) -> np.ndarray:

        """
        Evaluate:

            x = a*y^2 + b*y + c
        """

        a, b, c = coeffs

        return (
            a * ys * ys +
            b * ys +
            c
        )

    # ------------------------------------------------------------------
    def debug_render(
        self,
        frame: np.ndarray,
        left: FitResult,
        right: FitResult,
        draw_pixels: bool = True
    ) -> np.ndarray:

        """
        Draw fitted curves and sliding-window pixels.
        """

        out = frame.copy()

        h = frame.shape[0]

        # Draw detected pixels
        if draw_pixels:

            for res, color in (
                (left, (255, 100, 0)),
                (right, (0, 100, 255))
            ):

                for x, y in res.pixels:

                    cv2.circle(
                        out,
                        (int(x), int(y)),
                        1,
                        color,
                        -1
                    )

        # Draw only in useful lane ROI range
        ys = np.linspace(
            int(h * 0.58),
            int(h * 0.92),
            60
        )

        # Draw fitted curves
        for res, color in (
            (left, (0, 255, 255)),
            (right, (0, 255, 0))
        ):

            if res.coeffs is None:
                continue

            xs = self.evaluate(
                res.coeffs,
                ys
            )

            pts = np.stack(
                [xs, ys],
                axis=1
            ).astype(np.int32)

            # Keep valid x positions
            pts = pts[
                (pts[:, 0] >= 0) &
                (pts[:, 0] < frame.shape[1])
            ]

            if len(pts) >= 2:

                cv2.polylines(
                    out,
                    [pts],
                    False,
                    color,
                    3
                )

        # Left information
        cv2.putText(
            out,
            f"L px={left.n_pixels} conf={left.confidence:.2f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            out,
            f"L px={left.n_pixels} conf={left.confidence:.2f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Right information
        cv2.putText(
            out,
            f"R px={right.n_pixels} conf={right.confidence:.2f}",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            out,
            f"R px={right.n_pixels} conf={right.confidence:.2f}",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        return out