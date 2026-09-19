from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class ValidationConfig:
    max_sweep_frac: float = 0.60
    min_lane_width_px: int = 30
    max_lane_width_px: int = 800
    max_width_variation: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict) -> "ValidationConfig":

        d = d or {}

        c = cls()

        for k, v in d.items():

            if hasattr(c, k):
                setattr(c, k, v)

        return c


@dataclass
class ValidationResult:
    left_ok: bool
    right_ok: bool
    pair_ok: bool

    reason_left: str = ""
    reason_right: str = ""
    reason_pair: str = ""


class LaneValidation:

    def __init__(
        self,
        cfg: ValidationConfig
    ):
        self.cfg = cfg

    # ---------------------------------------------------------
    # SIDE VALIDATION
    # ---------------------------------------------------------

    def _side_ok(
        self,
        coeffs: Optional[np.ndarray],
        y_range: Tuple[int, int],
        frame_w: int
    ) -> Tuple[bool, str]:

        if coeffs is None:
            return False, "no_fit"

        ys = np.linspace(
            y_range[0],
            y_range[1],
            50
        )

        a, b, c = coeffs

        xs = (
            a * ys * ys
            + b * ys
            + c
        )

        sweep = (
            xs.max()
            - xs.min()
        )

        if (
            sweep
            > self.cfg.max_sweep_frac
            * frame_w
        ):

            return (
                False,
                f"sweep={sweep:.0f}"
            )

        if (
            xs.min()
            < -0.15 * frame_w
            or
            xs.max()
            > 1.15 * frame_w
        ):

            return (
                False,
                "off_canvas"
            )

        return True, ""

    # ---------------------------------------------------------
    # PAIR VALIDATION
    # ---------------------------------------------------------

    def validate(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
        y_range: Tuple[int, int],
        frame_w: int
    ) -> ValidationResult:

        l_ok, l_reason = self._side_ok(
            left,
            y_range,
            frame_w
        )

        r_ok, r_reason = self._side_ok(
            right,
            y_range,
            frame_w
        )

        pair_ok = False

        pair_reason = "need_both"

        if l_ok and r_ok:

            ys = np.linspace(
                y_range[0],
                y_range[1],
                50
            )

            xs_l = (
                left[0] * ys * ys
                + left[1] * ys
                + left[2]
            )

            xs_r = (
                right[0] * ys * ys
                + right[1] * ys
                + right[2]
            )

            # -------------------------------------------------
            # 1. Curves must not cross
            # -------------------------------------------------

            if np.any(
                xs_r <= xs_l
            ):

                pair_reason = "crossing"

                pair_ok = False

            else:

                widths = (
                    xs_r - xs_l
                )

                # -------------------------------------------------
                # 2. Maximum lane width
                # -------------------------------------------------

                w_max = float(
                    widths.max()
                )

                if (
                    w_max
                    > self.cfg.max_lane_width_px
                ):

                    pair_reason = (
                        f"width_max={w_max:.0f}"
                    )

                else:

                    # -------------------------------------------------
                    # 3. Minimum lane width
                    #
                    # Ignore the first 30% near the horizon.
                    # -------------------------------------------------

                    width_start = int(
                        len(widths) * 0.30
                    )

                    usable_widths = (
                        widths[
                            width_start:
                        ]
                    )

                    w_min = float(
                        usable_widths.min()
                    )

                    if (
                        w_min
                        < self.cfg.min_lane_width_px
                    ):

                        pair_reason = (
                            f"width_min={w_min:.0f}"
                        )

                    else:

                        pair_ok = True

                        pair_reason = "ok"

        return ValidationResult(
            left_ok=l_ok,
            right_ok=r_ok,
            pair_ok=pair_ok,
            reason_left=l_reason,
            reason_right=r_reason,
            reason_pair=pair_reason,
        )