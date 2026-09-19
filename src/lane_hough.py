from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class HoughConfig:

    enabled: bool = True

    rho: float = 1.0

    theta_deg: float = 1.0

    threshold: int = 20

    min_line_length: int = 20

    max_line_gap: int = 30

    slope_abs_min: float = 0.30

    slope_abs_max: float = 3.0


    @classmethod
    def from_dict(cls, d):

        d = d or {}

        c = cls()

        for k, v in d.items():

            if hasattr(c, k):

                setattr(c, k, v)

        return c


# ============================================================
# HOUGH SEGMENT
# ============================================================

@dataclass
class Seg:

    x1: int

    y1: int

    x2: int

    y2: int

    slope: float

    length: float


# ============================================================
# HOUGH DETECTOR
# ============================================================

class LaneHough:

    def __init__(self, cfg: HoughConfig):

        self.cfg = cfg


    # ========================================================
    # CLASSIFY HOUGH SEGMENTS
    # ========================================================

    def classify(self, edges):

        if not self.cfg.enabled:

            return [], [], []


        # ----------------------------------------------------
        # Probabilistic Hough Transform
        # ----------------------------------------------------

        lines = cv2.HoughLinesP(

            edges,

            rho=self.cfg.rho,

            theta=np.deg2rad(
                self.cfg.theta_deg
            ),

            threshold=self.cfg.threshold,

            minLineLength=self.cfg.min_line_length,

            maxLineGap=self.cfg.max_line_gap,

        )


        left = []

        right = []

        discarded = []


        if lines is None:

            return left, right, discarded


        # ====================================================
        # PROCESS EACH SEGMENT
        # ====================================================

        for l in lines:

            # ------------------------------------------------
            # OpenCV normally returns:
            #
            # [[[x1, y1, x2, y2]]]
            #
            # But support both possible formats.
            # ------------------------------------------------

            if l.ndim == 2:

                x1, y1, x2, y2 = l[0]

            else:

                x1, y1, x2, y2 = l


            # ------------------------------------------------
            # Direction
            # ------------------------------------------------

            dx = x2 - x1

            dy = y2 - y1


            # ------------------------------------------------
            # Vertical line
            # ------------------------------------------------

            if dx == 0:

                discarded.append(

                    Seg(

                        int(x1),

                        int(y1),

                        int(x2),

                        int(y2),

                        np.inf,

                        float(
                            np.hypot(
                                dx,
                                dy
                            )
                        )

                    )

                )

                continue


            # ------------------------------------------------
            # Slope
            # ------------------------------------------------

            slope = dy / dx


            # ------------------------------------------------
            # Segment length
            # ------------------------------------------------

            length = float(

                np.hypot(
                    dx,
                    dy
                )

            )


            seg = Seg(

                int(x1),

                int(y1),

                int(x2),

                int(y2),

                float(slope),

                length

            )


            # ------------------------------------------------
            # Absolute slope
            # ------------------------------------------------

            a = abs(slope)


            # ------------------------------------------------
            # Reject nearly-horizontal or excessively steep
            # segments.
            # ------------------------------------------------

            if (

                a < self.cfg.slope_abs_min

                or

                a > self.cfg.slope_abs_max

            ):

                discarded.append(seg)

                continue


            # =================================================
            # LEFT / RIGHT CLASSIFICATION
            #
            # IMPORTANT:
            #
            # For this camera/video:
            #
            #   negative slope -> LEFT lane boundary
            #   positive slope -> RIGHT lane boundary
            #
            # This was verified visually using the actual
            # fitted lane lines from frames 3456-3490.
            # =================================================

            if slope < 0:

                left.append(seg)

            else:

                right.append(seg)


        return (

            left,

            right,

            discarded

        )