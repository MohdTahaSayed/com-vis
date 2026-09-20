from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.horizon import (
    HorizonDetector,
    HorizonConfig
)

from src.lane_color import (
    LaneColor,
    LaneColorConfig
)

from src.lane_roi import (
    LaneROI,
    RoiConfig
)

from src.lane_edges import (
    LaneEdges,
    CannyConfig
)

from src.lane_fit import (
    LaneFitter,
    SlidingWindowConfig,
    LaneFitResult
)

from src.io_video import VideoReader


# ============================================================
# CONFIG
# ============================================================

def load_config(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return yaml.safe_load(f) or {}


# ============================================================
# LABEL
# ============================================================

def label(img, text):

    cv2.putText(
        img,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return img


# ============================================================
# HISTOGRAM
# ============================================================

def draw_histogram(
    mask,
    x_left,
    x_right
):

    """
    Visualize the same histogram used by LaneFitter.
    """

    h, w = mask.shape

    # Same bottom 30%
    bottom = mask[
        int(h * 0.70):,
        :
    ]

    hist = np.sum(
        bottom > 0,
        axis=0
    ).astype(
        np.float32
    )

    # Same smoothing
    kernel = (
        np.ones(
            15,
            dtype=np.float32
        ) / 15.0
    )

    hist = np.convolve(
        hist,
        kernel,
        mode="same"
    )

    canvas_h = 300

    canvas = np.zeros(
        (
            canvas_h,
            w,
            3
        ),
        dtype=np.uint8
    )

    max_val = max(
        float(hist.max()),
        1.0
    )

    # Histogram
    for x in range(w):

        bar_h = int(
            (
                hist[x]
                / max_val
            )
            * (canvas_h - 30)
        )

        cv2.line(
            canvas,
            (
                x,
                canvas_h - 1
            ),
            (
                x,
                canvas_h - 1 - bar_h
            ),
            (255, 255, 255),
            1
        )

    # Search regions
    left_hi = int(
        0.45 * w
    )

    right_lo = int(
        0.55 * w
    )

    cv2.line(
        canvas,
        (
            left_hi,
            0
        ),
        (
            left_hi,
            canvas_h
        ),
        (0, 255, 255),
        1
    )

    cv2.line(
        canvas,
        (
            right_lo,
            0
        ),
        (
            right_lo,
            canvas_h
        ),
        (0, 255, 255),
        1
    )

    # Selected bases
    cv2.line(
        canvas,
        (
            int(x_left),
            0
        ),
        (
            int(x_left),
            canvas_h
        ),
        (0, 255, 255),
        3
    )

    cv2.line(
        canvas,
        (
            int(x_right),
            0
        ),
        (
            int(x_right),
            canvas_h
        ),
        (0, 255, 0),
        3
    )

    label(
        canvas,
        f"Histogram | Left base={x_left} | Right base={x_right}"
    )

    return canvas


# ============================================================
# SLIDING WINDOWS
# ============================================================

def draw_sliding_windows(
    mask,
    x_base,
    side,
    cfg
):

    """
    Visualize the EXACT same sliding-window logic
    used by LaneFitter.
    """

    h, w = mask.shape

    # --------------------------------------------------------
    # EXACT CONFIG VALUES
    # --------------------------------------------------------

    n = max(
        1,
        cfg.n_windows
    )

    win_w = max(
        20,
        int(
            cfg.window_width_frac
            * w
        )
    )

    win_h = max(
        1,
        h // n
    )

    max_jump = (
        cfg.max_recenter_jump_frac
        * w
    )

    # --------------------------------------------------------
    # NONZERO PIXELS
    # --------------------------------------------------------

    nonzero_y, nonzero_x = np.nonzero(
        mask
    )

    current_x = float(
        x_base
    )

    out = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )

    window_color = (
        (0, 255, 255)
        if side == "LEFT"
        else (0, 255, 0)
    )

    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    for i in range(n):

        y_low = max(
            0,
            h - (i + 1) * win_h
        )

        y_high = min(
            h,
            h - i * win_h
        )

        # Current window position
        x_low = int(
            current_x
            - win_w / 2
        )

        x_high = int(
            current_x
            + win_w / 2
        )

        # ----------------------------------------------------
        # DRAW WINDOW
        # ----------------------------------------------------

        cv2.rectangle(
            out,
            (
                x_low,
                y_low
            ),
            (
                x_high,
                y_high
            ),
            window_color,
            2
        )

        # ----------------------------------------------------
        # FIND PIXELS
        # ----------------------------------------------------

        selection = (
            (nonzero_y >= y_low)
            &
            (nonzero_y < y_high)
            &
            (nonzero_x >= x_low)
            &
            (nonzero_x < x_high)
        )

        xs = nonzero_x[
            selection
        ]

        ys = nonzero_y[
            selection
        ]

        # ----------------------------------------------------
        # DRAW DETECTED PIXELS
        # ----------------------------------------------------

        for x, y in zip(
            xs,
            ys
        ):

            cv2.circle(
                out,
                (
                    int(x),
                    int(y)
                ),
                1,
                (0, 0, 255),
                -1
            )

        # ----------------------------------------------------
        # EXACT SAME RECENTERING LOGIC
        # AS LaneFitter
        # ----------------------------------------------------

        if len(xs) >= (
            cfg.min_pixels_per_window
        ):

            if len(xs) >= (
                cfg.min_pixels_to_recenter
            ):

                new_x = float(
                    np.median(xs)
                )

                delta = (
                    new_x
                    - current_x
                )

                # Same maximum jump restriction
                if abs(delta) > max_jump:

                    if delta > 0:

                        new_x = (
                            current_x
                            + max_jump
                        )

                    else:

                        new_x = (
                            current_x
                            - max_jump
                        )

                current_x = new_x

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        cv2.putText(
            out,
            f"W{i + 1}: {len(xs)} px",
            (
                max(
                    5,
                    x_low
                ),
                max(
                    15,
                    y_low + 20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    label(
        out,
        f"Sliding Windows - {side} | base={x_base}"
    )

    return out


# ============================================================
# POLYNOMIAL FIT
# ============================================================

def draw_poly_fit(
    frame,
    left,
    right
):

    """
    Show the actual polynomial fit from LaneFitter.
    """

    out = frame.copy()

    h, w = frame.shape[:2]

    ys = np.linspace(
        int(h * 0.55),
        int(h * 0.95),
        100
    )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    if left.coeffs is not None:

        xs = (
            left.coeffs[0] * ys * ys
            + left.coeffs[1] * ys
            + left.coeffs[2]
        )

        pts = np.stack(
            [
                xs,
                ys
            ],
            axis=1
        ).astype(
            np.int32
        )

        pts = pts[
            (pts[:, 0] >= 0)
            &
            (pts[:, 0] < w)
        ]

        if len(pts) >= 2:

            cv2.polylines(
                out,
                [pts],
                False,
                (0, 255, 255),
                3
            )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    if right.coeffs is not None:

        xs = (
            right.coeffs[0] * ys * ys
            + right.coeffs[1] * ys
            + right.coeffs[2]
        )

        pts = np.stack(
            [
                xs,
                ys
            ],
            axis=1
        ).astype(
            np.int32
        )

        pts = pts[
            (pts[:, 0] >= 0)
            &
            (pts[:, 0] < w)
        ]

        if len(pts) >= 2:

            cv2.polylines(
                out,
                [pts],
                False,
                (0, 255, 0),
                3
            )

    label(
        out,
        "Polynomial Fit"
    )

    return out


# ============================================================
# COMPLETE DEBUG FRAME
# ============================================================

def make_frame_debug(
    frame,
    cfg
):

    # --------------------------------------------------------
    # 1. INITIALIZE MODULES
    # --------------------------------------------------------

    hz = HorizonDetector(
        HorizonConfig.from_dict(
            cfg.get(
                "horizon",
                {}
            )
        )
    )

    roi = LaneROI(
        RoiConfig.from_dict(
            cfg.get(
                "roi",
                {}
            )
        )
    )

    lc = LaneColor(
        LaneColorConfig.from_dict(
            cfg.get(
                "lane_color",
                {}
            )
        )
    )

    edges_mod = LaneEdges(
        CannyConfig.from_dict(
            cfg.get(
                "canny",
                {}
            )
        ),
        roi,
        lc,
        reinforce_with_hsv=True
    )

    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(
            cfg.get(
                "sliding_window",
                {}
            )
        )
    )

    # --------------------------------------------------------
    # 2. HORIZON
    # --------------------------------------------------------

    horizon_y = hz.detect(
        frame
    )

    # --------------------------------------------------------
    # 3. EDGES + ROI + HSV
    # --------------------------------------------------------

    edges_roi, edges_raw, hsv_hits = (
        edges_mod.compute(
            frame,
            top_y_override=horizon_y
        )
    )

    # --------------------------------------------------------
    # 4. HISTOGRAM BASES
    #
    # IMPORTANT:
    # LaneFitter calculates the shifted bases.
    # --------------------------------------------------------

    x_left, x_right = (
        fitter._histogram_base(
            edges_roi
        )
    )

    # --------------------------------------------------------
    # 5. SLIDING WINDOWS
    #
    # EXACT SAME LaneFitter logic
    # --------------------------------------------------------

    left_pixels = fitter._sliding_window(
        edges_roi,
        x_left
    )

    right_pixels = fitter._sliding_window(
        edges_roi,
        x_right
    )

    # --------------------------------------------------------
    # 6. POLYNOMIAL FIT
    # --------------------------------------------------------

    left_coeffs, left_rms = (
        fitter._fit_poly(
            left_pixels
        )
    )

    right_coeffs, right_rms = (
        fitter._fit_poly(
            right_pixels
        )
    )

    # --------------------------------------------------------
    # 7. FIT RESULT OBJECTS
    # --------------------------------------------------------

    left = LaneFitResult(
        coeffs=left_coeffs,
        confidence=min(
            1.0,
            len(left_pixels) / 100.0
        ),
        n_pixels=len(left_pixels),
        rms_error=left_rms,
        x_base=x_left,
        x_pixels=left_pixels[0] if isinstance(left_pixels, tuple) else left_pixels,
        y_pixels=left_pixels[1] if isinstance(left_pixels, tuple) else None
    )

    right = LaneFitResult(
        coeffs=right_coeffs,
        confidence=min(
            1.0,
            len(right_pixels) / 100.0
        ),
        n_pixels=len(right_pixels),
        rms_error=right_rms,
        x_base=x_right,
        x_pixels=right_pixels[0] if isinstance(right_pixels, tuple) else right_pixels,
        y_pixels=right_pixels[1] if isinstance(right_pixels, tuple) else None
    )

    # ========================================================
    # VISUALIZATIONS
    # ========================================================

    # --------------------------------------------------------
    # A. ORIGINAL
    # --------------------------------------------------------

    p1 = frame.copy()

    label(
        p1,
        f"Original | horizon={horizon_y}"
    )

    if horizon_y is not None:

        cv2.line(
            p1,
            (
                0,
                horizon_y
            ),
            (
                p1.shape[1],
                horizon_y
            ),
            (0, 255, 255),
            2
        )

    # --------------------------------------------------------
    # B. CANNY RAW
    # --------------------------------------------------------

    p2 = cv2.cvtColor(
        edges_raw,
        cv2.COLOR_GRAY2BGR
    )

    label(
        p2,
        "Canny Raw"
    )

    # --------------------------------------------------------
    # C. ROI EDGES
    # --------------------------------------------------------

    p3 = cv2.cvtColor(
        edges_roi,
        cv2.COLOR_GRAY2BGR
    )

    label(
        p3,
        "Canny + ROI + HSV"
        f" | {np.count_nonzero(edges_roi)} px"
    )

    # --------------------------------------------------------
    # D. HSV
    # --------------------------------------------------------

    if hsv_hits is not None:

        p4 = cv2.cvtColor(
            hsv_hits,
            cv2.COLOR_GRAY2BGR
        )

        label(
            p4,
            "HSV Reinforcement Mask"
        )

    else:

        p4 = np.zeros_like(
            p3
        )

        label(
            p4,
            "HSV Reinforcement: None"
        )

    # --------------------------------------------------------
    # E. HISTOGRAM
    # --------------------------------------------------------

    p5 = draw_histogram(
        edges_roi,
        x_left,
        x_right
    )

    # --------------------------------------------------------
    # F. LEFT WINDOWS
    # --------------------------------------------------------

    p6 = draw_sliding_windows(
        edges_roi,
        x_left,
        "LEFT",
        fitter.cfg
    )

    # --------------------------------------------------------
    # G. RIGHT WINDOWS
    # --------------------------------------------------------

    p7 = draw_sliding_windows(
        edges_roi,
        x_right,
        "RIGHT",
        fitter.cfg
    )

    # --------------------------------------------------------
    # H. POLYNOMIAL
    # --------------------------------------------------------

    p8 = draw_poly_fit(
        frame,
        left,
        right
    )

    # ========================================================
    # RESIZE PANELS
    # ========================================================

    target_h = 300

    panels = []

    for p in [
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8
    ]:

        new_w = int(
            p.shape[1]
            * target_h
            / p.shape[0]
        )

        panels.append(
            cv2.resize(
                p,
                (
                    new_w,
                    target_h
                )
            )
        )

    # ========================================================
    # 3 x 3 GRID (last cell empty)
    # ========================================================

    rows = []

    for i in range(
        0,
        9,
        3
    ):

        chunk = panels[
            i:i + 3
        ]

        # Pad final row with a blank panel if needed
        while len(chunk) < 3:

            blank = np.zeros_like(
                panels[0]
            )

            chunk.append(
                blank
            )

        row = np.hstack(
            chunk
        )

        rows.append(
            row
        )

    max_w = max(
        r.shape[1]
        for r in rows
    )

    fixed_rows = []

    for r in rows:

        if r.shape[1] < max_w:

            r = cv2.copyMakeBorder(
                r,
                0,
                0,
                0,
                max_w - r.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

        fixed_rows.append(
            r
        )

    final = np.vstack(
        fixed_rows
    )

    return final, {
        "left_base": x_left,
        "right_base": x_right,
        "left_pixels": len(left_pixels),
        "right_pixels": len(right_pixels),
        "left_rms": left_rms,
        "right_rms": right_rms,
        "horizon": horizon_y
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--time",
        type=float,
        default=None
    )

    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=None,
        help="timestamps in seconds"
    )

    parser.add_argument(
        "--config",
        default="config/default.yaml"
    )

    parser.add_argument(
        "--outdir",
        default="outputs/lane_debug"
    )

    args = parser.parse_args()

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    cfg = load_config(
        args.config
    )

    # --------------------------------------------------------
    # SELECT TIMES
    # --------------------------------------------------------

    if args.time is not None:

        times = [
            args.time
        ]

    elif args.times is not None:

        times = args.times

    else:

        times = [
            30.0,
            300.0,
            900.0
        ]

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    vr = VideoReader(
        args.input
    )

    print()
    print(
        "======================================"
    )
    print(
        " LANE STAGE DEBUG"
    )
    print(
        "======================================"
    )

    print(
        f"Video: {args.input}"
    )

    print(
        f"FPS:   {vr.info.fps}"
    )

    print(
        f"Size:  "
        f"{vr.info.width}x"
        f"{vr.info.height}"
    )

    print()

    # --------------------------------------------------------
    # PRINT ACTIVE CONFIG
    # --------------------------------------------------------

    sw = SlidingWindowConfig.from_dict(
        cfg.get(
            "sliding_window",
            {}
        )
    )

    print(
        "ACTIVE SLIDING WINDOW CONFIG"
    )

    print(
        f"  windows              : "
        f"{sw.n_windows}"
    )

    print(
        f"  width fraction       : "
        f"{sw.window_width_frac}"
    )

    print(
        f"  min pixels recenter  : "
        f"{sw.min_pixels_to_recenter}"
    )

    print(
        f"  min pixels/window    : "
        f"{sw.min_pixels_per_window}"
    )

    print(
        f"  min pixels total     : "
        f"{sw.min_pixels_total}"
    )

    print(
        f"  max recenter jump    : "
        f"{sw.max_recenter_jump_frac}"
    )

    print(
        f"  left base shift      : "
        f"-{sw.left_base_shift_px}px"
    )

    print(
        f"  right base shift     : "
        f"+{sw.right_base_shift_px}px"
    )

    print()

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for t in times:

        frame_idx = int(
            round(
                t * vr.info.fps
            )
        )

        frame = vr.read_frame(
            frame_idx
        )

        if frame is None:

            print(
                f"[WARN] Could not read "
                f"frame at {t}s"
            )

            continue

        debug, stats = make_frame_debug(
            frame,
            cfg
        )

        filename = os.path.join(
            args.outdir,
            f"lane_debug_{t:.2f}s.png"
        )

        cv2.imwrite(
            filename,
            debug
        )

        print(
            f"[OK] t={t:7.2f}s "
            f"frame={frame_idx:6d} "
            f"horizon={stats['horizon']} "
            f"Lbase={stats['left_base']} "
            f"Rbase={stats['right_base']} "
            f"Lpx={stats['left_pixels']} "
            f"Rpx={stats['right_pixels']}"
        )

        print(
            f"     -> {filename}"
        )

    vr.release()

    print()
    print(
        "Done."
    )


if __name__ == "__main__":
    main()