from __future__ import annotations

import os
import sys
import argparse

import cv2
import numpy as np
import yaml

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

from src.lane_edges import (
    LaneEdges,
    CannyConfig
)

from src.lane_roi import (
    LaneROI,
    RoiConfig
)

from src.io_video import VideoReader

from src.lane_fit import (
    SlidingWindowConfig
)


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)


# ============================================================
# READ FRAME
# ============================================================

def get_frame(
    video_path,
    time_s
):

    reader = VideoReader(
        video_path
    )

    fps = reader.info.fps

    frame_idx = int(
        round(
            time_s * fps
        )
    )

    frame = reader.read_frame(
        frame_idx
    )

    reader.release()

    if frame is None:

        raise RuntimeError(
            f"Could not read frame {frame_idx}"
        )

    return (
        frame,
        frame_idx,
        frame_idx / fps
    )


# ============================================================
# INSPECT WINDOWS
# ============================================================

def inspect_side(
    mask,
    x_base,
    cfg
):

    h, w = mask.shape[:2]

    n_windows = max(
        1,
        cfg.n_windows
    )

    window_width = max(
        20,
        int(
            cfg.window_width_frac
            * w
        )
    )

    window_height = max(
        1,
        h // n_windows
    )

    max_jump = (
        cfg.max_recenter_jump_frac
        * w
    )

    nonzero_y, nonzero_x = np.nonzero(
        mask
    )

    current_x = float(
        x_base
    )

    results = []

    for i in range(
        n_windows
    ):

        y_low = (
            h
            - (i + 1)
            * window_height
        )

        y_high = (
            h
            - i
            * window_height
        )

        y_low = max(
            0,
            y_low
        )

        y_high = min(
            h,
            y_high
        )

        x_low = int(
            current_x
            - window_width / 2
        )

        x_high = int(
            current_x
            + window_width / 2
        )

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

        pixel_count = len(xs)

        old_x = current_x

        recentered = False

        if pixel_count >= (
            cfg.min_pixels_per_window
        ):

            if pixel_count >= (
                cfg.min_pixels_to_recenter
            ):

                new_x = float(
                    np.median(xs)
                )

                delta = (
                    new_x
                    - current_x
                )

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

                recentered = True

        results.append({
            "window": i + 1,
            "x_before": old_x,
            "x_after": current_x,
            "y_low": y_low,
            "y_high": y_high,
            "pixels": pixel_count,
            "recentered": recentered
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/VBOX0011_Trim.mp4"
    )

    parser.add_argument(
        "--time",
        type=float,
        default=720.0
    )

    parser.add_argument(
        "--config",
        default="config/default.yaml"
    )

    parser.add_argument(
        "--output",
        default="outputs/sliding_windows_check.png"
    )

    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(
            args.output
        ) or ".",
        exist_ok=True
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    cfg = load_config(
        args.config
    )

    sw_cfg = SlidingWindowConfig.from_dict(
        cfg.get(
            "sliding_window",
            {}
        )
    )

    # --------------------------------------------------------
    # FRAME
    # --------------------------------------------------------

    frame, frame_idx, timestamp = get_frame(
        args.input,
        args.time
    )

    h, w = frame.shape[:2]

    window_width = max(
        20,
        int(
            sw_cfg.window_width_frac
            * w
        )
    )

    window_height = max(
        1,
        h // sw_cfg.n_windows
    )

    # --------------------------------------------------------
    # HORIZON
    # --------------------------------------------------------

    horizon_detector = HorizonDetector(
        HorizonConfig.from_dict(
            cfg.get(
                "horizon",
                {}
            )
        )
    )

    horizon_y = horizon_detector.detect(
        frame
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    roi = LaneROI(
        RoiConfig.from_dict(
            cfg.get(
                "roi",
                {}
            )
        )
    )

    # --------------------------------------------------------
    # LANE COLOR
    # --------------------------------------------------------

    lane_color = LaneColor(
        LaneColorConfig.from_dict(
            cfg.get(
                "lane_color",
                {}
            )
        )
    )

    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    edges = LaneEdges(
        CannyConfig.from_dict(
            cfg.get(
                "canny",
                {}
            )
        ),
        roi,
        lane_color,
        reinforce_with_hsv=True
    )

    edges_roi, _, _ = edges.compute(
        frame,
        top_y_override=horizon_y
    )

    # ========================================================
    # HISTOGRAM
    # ========================================================

    bottom = edges_roi[
        int(h * 0.70):,
        :
    ]

    histogram = np.sum(
        bottom > 0,
        axis=0
    ).astype(
        np.float32
    )

    kernel = (
        np.ones(
            15,
            dtype=np.float32
        ) / 15.0
    )

    histogram = np.convolve(
        histogram,
        kernel,
        mode="same"
    )

    # --------------------------------------------------------
    # LEFT BASE
    # --------------------------------------------------------

    left_hi = int(
        0.45 * w
    )

    left_segment = histogram[
        :left_hi
    ]

    if left_segment.max() > 0:

        left_base = int(
            np.argmax(
                left_segment
            )
        )

    else:

        left_base = int(
            0.25 * w
        )

    left_base = max(
        0,
        left_base
        - sw_cfg.left_base_shift_px
    )

    # --------------------------------------------------------
    # RIGHT BASE
    # --------------------------------------------------------

    right_lo = int(
        0.55 * w
    )

    right_segment = histogram[
        right_lo:
    ]

    if right_segment.max() > 0:

        right_base = (
            right_lo
            + int(
                np.argmax(
                    right_segment
                )
            )
        )

    else:

        right_base = int(
            0.75 * w
        )

    right_base = min(
        w - 1,
        right_base
        + sw_cfg.right_base_shift_px
    )

    # ========================================================
    # WINDOWS
    # ========================================================

    left_results = inspect_side(
        edges_roi,
        left_base,
        sw_cfg
    )

    right_results = inspect_side(
        edges_roi,
        right_base,
        sw_cfg
    )

    # ========================================================
    # DRAW
    # ========================================================

    output = frame.copy()

    if horizon_y is not None:

        cv2.line(
            output,
            (0, horizon_y),
            (w, horizon_y),
            (0, 255, 255),
            2
        )

    # --------------------------------------------------------
    # LEFT WINDOWS
    # --------------------------------------------------------

    for r in left_results:

        x = int(
            r["x_before"]
        )

        x1 = int(
            x - window_width / 2
        )

        x2 = int(
            x + window_width / 2
        )

        cv2.rectangle(
            output,
            (
                x1,
                r["y_low"]
            ),
            (
                x2,
                r["y_high"]
            ),
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"L{r['window']}: "
            f"{r['pixels']}px",
            (
                x1 + 4,
                r["y_low"] + 16
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # RIGHT WINDOWS
    # --------------------------------------------------------

    for r in right_results:

        x = int(
            r["x_before"]
        )

        x1 = int(
            x - window_width / 2
        )

        x2 = int(
            x + window_width / 2
        )

        cv2.rectangle(
            output,
            (
                x1,
                r["y_low"]
            ),
            (
                x2,
                r["y_high"]
            ),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"R{r['window']}: "
            f"{r['pixels']}px",
            (
                x1 + 4,
                r["y_low"] + 16
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # BASE DOTS
    # --------------------------------------------------------

    cv2.circle(
        output,
        (
            int(left_base),
            h - 10
        ),
        7,
        (0, 255, 255),
        -1
    )

    cv2.circle(
        output,
        (
            int(right_base),
            h - 10
        ),
        7,
        (0, 255, 0),
        -1
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"Horizon: {horizon_y}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Left base: {left_base}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Right base: {right_base}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Window: "
        f"{window_width} x "
        f"{window_height}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    cv2.imwrite(
        args.output,
        output
    )

    # ========================================================
    # TERMINAL
    # ========================================================

    print()
    print("=" * 60)
    print("SLIDING WINDOW INSPECTION")
    print("=" * 60)

    print(
        f"Input       : {args.input}"
    )

    print(
        f"Time        : {timestamp:.2f} s"
    )

    print(
        f"Frame       : {frame_idx}"
    )

    print(
        f"Resolution  : {w} x {h}"
    )

    print(
        f"Horizon     : {horizon_y}"
    )

    print(
        f"Left base   : {left_base}"
    )

    print(
        f"Right base  : {right_base}"
    )

    print(
        f"Window size : "
        f"{window_width} x "
        f"{window_height}"
    )

    print(
        f"Left shift  : "
        f"-{sw_cfg.left_base_shift_px} px"
    )

    print(
        f"Right shift : "
        f"+{sw_cfg.right_base_shift_px} px"
    )

    print(
        f"Output      : {args.output}"
    )

    print("=" * 60)

    print()
    print("LEFT WINDOWS")
    print("-" * 50)

    for r in left_results:

        print(
            f"W{r['window']:02d}: "
            f"x={r['x_before']:.0f}"
            f" -> "
            f"{r['x_after']:.0f} | "
            f"pixels={r['pixels']:4d} | "
            f"recenter={r['recentered']}"
        )

    print()
    print("RIGHT WINDOWS")
    print("-" * 50)

    for r in right_results:

        print(
            f"W{r['window']:02d}: "
            f"x={r['x_before']:.0f}"
            f" -> "
            f"{r['x_after']:.0f} | "
            f"pixels={r['pixels']:4d} | "
            f"recenter={r['recentered']}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()