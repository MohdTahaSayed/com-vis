from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_hough import LaneHough, HoughConfig
from src.lane_state import LaneState, StateConfig
from src.io_video import VideoReader


CONFIG_PATH = "config/default.yaml"


# ============================================================
# CONFIG
# ============================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================
# LINEAR FIT
# ============================================================

def fit_linear_from_segments(segments, min_points=2):
    """
    Fit:

        x = m*y + b

    from Hough Seg endpoints.

    Each Hough segment contributes:

        (x1, y1)
        (x2, y2)

    Longer segments receive larger weights.

    Returns:
        coeffs = [m, b]
        number of points
        RMS fitting error
    """

    if not segments:
        return None, 0, None

    points = []
    weights = []

    for seg in segments:

        points.append(
            (float(seg.y1), float(seg.x1))
        )

        points.append(
            (float(seg.y2), float(seg.x2))
        )

        weight = max(
            float(seg.length),
            1.0
        )

        weights.extend(
            [weight, weight]
        )

    if len(points) < min_points:
        return None, 0, None

    ys = np.asarray(
        [p[0] for p in points],
        dtype=np.float64
    )

    xs = np.asarray(
        [p[1] for p in points],
        dtype=np.float64
    )

    ws = np.asarray(
        weights,
        dtype=np.float64
    )

    # Cannot fit a line if all y values are identical.
    if np.ptp(ys) < 1e-6:
        return None, 0, None

    # --------------------------------------------------------
    # Weighted least squares
    #
    # x = m*y + b
    # --------------------------------------------------------

    A = np.column_stack(
        (
            ys,
            np.ones_like(ys)
        )
    )

    W = np.sqrt(ws)[:, None]

    try:

        coeffs, _, _, _ = np.linalg.lstsq(
            A * W,
            xs * W[:, 0],
            rcond=None
        )

    except np.linalg.LinAlgError:

        return None, 0, None

    m, b = coeffs

    # --------------------------------------------------------
    # RMS fitting error
    # --------------------------------------------------------

    predicted = m * ys + b

    rms = float(
        np.sqrt(
            np.mean(
                (xs - predicted) ** 2
            )
        )
    )

    return (
        np.array(
            [m, b],
            dtype=np.float64
        ),
        len(points),
        rms
    )


# ============================================================
# TEMPORARY CONFIDENCE
# ============================================================

def confidence_from_fit(segments, rms):
    """
    Temporary diagnostic confidence.

    This is NOT a permanent project scoring rule.
    """

    if not segments or rms is None:
        return 0.0

    total_length = sum(
        float(s.length)
        for s in segments
    )

    # More supporting line length
    # -> higher confidence.

    support = min(
        total_length / 400.0,
        1.0
    )

    # Lower RMS
    # -> higher confidence.

    error_score = max(
        0.0,
        min(
            1.0,
            1.0 - rms / 30.0
        )
    )

    return float(
        0.5 * support
        +
        0.5 * error_score
    )


# ============================================================
# LINE EVALUATION
# ============================================================

def line_x(coeffs, y):

    if coeffs is None:
        return None

    m, b = coeffs

    return float(
        m * y + b
    )


# ============================================================
# DRAW LINE
# ============================================================

def draw_linear_line(
    frame,
    coeffs,
    y1,
    y2,
    color,
    thickness=3
):

    if coeffs is None:
        return

    x1 = line_x(
        coeffs,
        y1
    )

    x2 = line_x(
        coeffs,
        y2
    )

    if x1 is None or x2 is None:
        return

    cv2.line(
        frame,
        (
            int(round(x1)),
            int(round(y1))
        ),
        (
            int(round(x2)),
            int(round(y2))
        ),
        color,
        thickness
    )


# ============================================================
# TEMPORARY PAIR VALIDATION
# ============================================================

def pair_validation(
    left_coeffs,
    right_coeffs,
    h,
    w
):

    """
    Basic temporary lane-pair geometry check.

    Checks:

    1. Both lines exist.
    2. Right line stays right of left line.
    3. Lane width is reasonable.
    4. Lines remain reasonably inside frame.

    This is ONLY for diagnosis.

    It does not modify the permanent
    LaneValidation module.
    """

    if (
        left_coeffs is None
        or right_coeffs is None
    ):

        return (
            False,
            "missing_side"
        )

    ys = np.linspace(
        int(h * 0.62),
        int(h * 0.92),
        20
    )

    xs_left = np.array(
        [
            line_x(
                left_coeffs,
                y
            )
            for y in ys
        ]
    )

    xs_right = np.array(
        [
            line_x(
                right_coeffs,
                y
            )
            for y in ys
        ]
    )

    if (
        np.any(~np.isfinite(xs_left))
        or
        np.any(~np.isfinite(xs_right))
    ):

        return (
            False,
            "non_finite"
        )

    widths = (
        xs_right
        -
        xs_left
    )

    # --------------------------------------------------------
    # Lines must not cross.
    # --------------------------------------------------------

    if np.any(widths <= 0):

        return (
            False,
            "crossing"
        )

    # --------------------------------------------------------
    # Minimum lane width.
    # --------------------------------------------------------

    min_width = float(
        widths.min()
    )

    if min_width < 40:

        return (
            False,
            f"width_min={min_width:.0f}"
        )

    # --------------------------------------------------------
    # Maximum lane width.
    # --------------------------------------------------------

    max_width = float(
        widths.max()
    )

    if max_width > 650:

        return (
            False,
            f"width_max={max_width:.0f}"
        )

    # --------------------------------------------------------
    # Left line should not be wildly outside frame.
    # --------------------------------------------------------

    if (
        np.any(xs_left < -100)
        or
        np.any(xs_left > w + 100)
    ):

        return (
            False,
            "left_out_of_frame"
        )

    # --------------------------------------------------------
    # Right line should not be wildly outside frame.
    # --------------------------------------------------------

    if (
        np.any(xs_right < -100)
        or
        np.any(xs_right > w + 100)
    ):

        return (
            False,
            "right_out_of_frame"
        )

    return (
        True,
        "ok"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Temporary linear Hough lane "
            "+ LaneState diagnostic."
        )
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=3456
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=40
    )

    parser.add_argument(
        "--config",
        default=CONFIG_PATH
    )

    parser.add_argument(
        "--outdir",
        default=(
            "outputs/"
            "linear_lane_corrected_check"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    cfg = load_config(
        args.config
    )

    # ========================================================
    # MODULE INITIALIZATION
    # ========================================================

    horizon = HorizonDetector(
        HorizonConfig.from_dict(
            cfg.get(
                "horizon",
                {}
            )
        )
    )

    lane_color = LaneColor(
        LaneColorConfig.from_dict(
            cfg.get(
                "lane_color",
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

    edges_module = LaneEdges(
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

    hough = LaneHough(
        HoughConfig.from_dict(
            cfg.get(
                "hough",
                {}
            )
        )
    )

    state = LaneState(
        StateConfig.from_dict(
            cfg.get(
                "lane_state",
                {}
            )
        )
    )

    # ========================================================
    # VIDEO
    # ========================================================

    video = VideoReader(
        args.input
    )

    fps = video.info.fps

    end_frame = min(
        args.start_frame
        +
        args.num_frames,
        video.info.frame_count
    )

    actual_frames = (
        end_frame
        -
        args.start_frame
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()

    print("=" * 78)

    print(
        "CORRECTED LINEAR LANE + TEMPORAL STATE TEST"
    )

    print("=" * 78)

    print(
        f"Input              : "
        f"{args.input}"
    )

    print(
        f"Start frame        : "
        f"{args.start_frame}"
    )

    print(
        f"Frames requested   : "
        f"{args.num_frames}"
    )

    print(
        f"Frames available   : "
        f"{actual_frames}"
    )

    print(
        f"FPS                : "
        f"{fps:.2f}"
    )

    print()

    print(
        "LINE EQUATION"
    )

    print(
        "x = m*y + b"
    )

    print()

    print(
        "This is a TEMPORARY diagnostic."
    )

    print(
        "No permanent project "
        "modules/configuration are modified."
    )

    print()

    print("-" * 78)

    print(
        f"{'FRAME':>6} | "
        f"{'L SEG':>5} "
        f"{'L m':>9} "
        f"{'L b':>9} "
        f"{'L RMS':>7} "
        f"{'L C':>5} "
        f"{'L STATE':>7} | "
        f"{'R SEG':>5} "
        f"{'R m':>9} "
        f"{'R b':>9} "
        f"{'R RMS':>7} "
        f"{'R C':>5} "
        f"{'R STATE':>7} | "
        f"PAIR"
    )

    print("-" * 78)

    # ========================================================
    # COUNTERS
    # ========================================================

    ok_left = 0
    hold_left = 0
    miss_left = 0

    ok_right = 0
    hold_right = 0
    miss_right = 0

    pair_ok_count = 0

    processed = 0

    # ========================================================
    # CSV
    # ========================================================

    csv_path = os.path.join(
        args.outdir,
        "linear_lane_results.csv"
    )

    csv = open(
        csv_path,
        "w",
        encoding="utf-8"
    )

    csv.write(
        "frame,time,horizon_y,"
        "left_segments,left_m,left_b,"
        "left_rms,left_conf,left_state,"
        "right_segments,right_m,right_b,"
        "right_rms,right_conf,right_state,"
        "pair_ok,pair_reason\n"
    )

    # ========================================================
    # PROCESS FRAMES
    # ========================================================

    try:

        for frame_idx in range(
            args.start_frame,
            end_frame
        ):

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            frame = video.read_frame(
                frame_idx
            )

            if frame is None:

                print(
                    f"WARNING: could not "
                    f"read frame {frame_idx}"
                )

                continue

            processed += 1

            h, w = frame.shape[:2]

            timestamp = (
                frame_idx / fps
            )

            # ------------------------------------------------
            # 1. Horizon
            # ------------------------------------------------

            horizon_y = horizon.detect(
                frame
            )

            # ------------------------------------------------
            # 2. Canny + ROI + HSV reinforcement
            # ------------------------------------------------

            edges_roi, edges_raw, hsv_hits = (
                edges_module.compute(
                    frame,
                    top_y_override=horizon_y
                )
            )

            # ------------------------------------------------
            # 3. Hough
            # ------------------------------------------------

            (
                left_segments,
                right_segments,
                discarded_segments
            ) = hough.classify(
                edges_roi
            )

            # ------------------------------------------------
            # 4. Temporary linear fit
            # ------------------------------------------------

            (
                left_coeffs,
                left_point_count,
                left_rms
            ) = fit_linear_from_segments(
                left_segments
            )

            (
                right_coeffs,
                right_point_count,
                right_rms
            ) = fit_linear_from_segments(
                right_segments
            )

            # ------------------------------------------------
            # 5. Temporary confidence
            # ------------------------------------------------

            left_conf = confidence_from_fit(
                left_segments,
                left_rms
            )

            right_conf = confidence_from_fit(
                right_segments,
                right_rms
            )

            # =================================================
            # 6. TEMPORAL STATE
            #
            # IMPORTANT:
            # LaneState uses one combined update() method.
            # =================================================

            (
                left_state_result,
                right_state_result
            ) = state.update(
                left_coeffs,
                left_conf,
                right_coeffs,
                right_conf
            )

            (
                left_smoothed,
                left_status,
                left_state_conf
            ) = left_state_result

            (
                right_smoothed,
                right_status,
                right_state_conf
            ) = right_state_result

            # ------------------------------------------------
            # 7. Temporary pair validation
            # ------------------------------------------------

            (
                pair_ok,
                pair_reason
            ) = pair_validation(
                left_smoothed,
                right_smoothed,
                h,
                w
            )

            if pair_ok:
                pair_ok_count += 1

            # ------------------------------------------------
            # 8. Count states
            # ------------------------------------------------

            if left_status == "OK":

                ok_left += 1

            elif left_status == "HOLD":

                hold_left += 1

            else:

                miss_left += 1

            if right_status == "OK":

                ok_right += 1

            elif right_status == "HOLD":

                hold_right += 1

            else:

                miss_right += 1

            # =================================================
            # 9. Terminal output
            # =================================================

            def fmt_coeff(
                coeffs,
                index
            ):

                if coeffs is None:
                    return "N/A"

                return (
                    f"{coeffs[index]:.4f}"
                )

            def fmt_rms(
                rms
            ):

                if rms is None:
                    return "N/A"

                return (
                    f"{rms:.2f}"
                )

            print(
                f"{frame_idx:6d} | "

                f"{len(left_segments):5d} "
                f"{fmt_coeff(left_coeffs, 0):>9} "
                f"{fmt_coeff(left_coeffs, 1):>9} "
                f"{fmt_rms(left_rms):>7} "
                f"{left_state_conf:5.2f} "
                f"{left_status:>7} | "

                f"{len(right_segments):5d} "
                f"{fmt_coeff(right_coeffs, 0):>9} "
                f"{fmt_coeff(right_coeffs, 1):>9} "
                f"{fmt_rms(right_rms):>7} "
                f"{right_state_conf:5.2f} "
                f"{right_status:>7} | "

                f"{str(pair_ok):>4}"
            )

            # =================================================
            # 10. CSV
            # =================================================

            left_m = (
                ""
                if left_coeffs is None
                else f"{left_coeffs[0]:.8f}"
            )

            left_b = (
                ""
                if left_coeffs is None
                else f"{left_coeffs[1]:.4f}"
            )

            left_rms_csv = (
                ""
                if left_rms is None
                else f"{left_rms:.6f}"
            )

            right_m = (
                ""
                if right_coeffs is None
                else f"{right_coeffs[0]:.8f}"
            )

            right_b = (
                ""
                if right_coeffs is None
                else f"{right_coeffs[1]:.4f}"
            )

            right_rms_csv = (
                ""
                if right_rms is None
                else f"{right_rms:.6f}"
            )

            horizon_csv = (
                ""
                if horizon_y is None
                else str(horizon_y)
            )

            csv.write(
                f"{frame_idx},"
                f"{timestamp:.4f},"
                f"{horizon_csv},"

                f"{len(left_segments)},"
                f"{left_m},"
                f"{left_b},"
                f"{left_rms_csv},"
                f"{left_state_conf:.6f},"
                f"{left_status},"

                f"{len(right_segments)},"
                f"{right_m},"
                f"{right_b},"
                f"{right_rms_csv},"
                f"{right_state_conf:.6f},"
                f"{right_status},"

                f"{int(pair_ok)},"
                f"{pair_reason}\n"
            )

            # =================================================
            # 11. VISUAL DIAGNOSTIC
            # =================================================

            # Save:
            #
            # - first frame
            # - every 10th frame
            #
            # so we can inspect the actual geometry.

            if (
                frame_idx == args.start_frame
                or
                frame_idx % 10 == 0
            ):

                visual = frame.copy()

                # ------------------------------------------------
                # Draw left Hough segments
                # ------------------------------------------------

                for seg in left_segments:

                    cv2.line(
                        visual,
                        (
                            seg.x1,
                            seg.y1
                        ),
                        (
                            seg.x2,
                            seg.y2
                        ),
                        (0, 255, 255),
                        1
                    )

                # ------------------------------------------------
                # Draw right Hough segments
                # ------------------------------------------------

                for seg in right_segments:

                    cv2.line(
                        visual,
                        (
                            seg.x1,
                            seg.y1
                        ),
                        (
                            seg.x2,
                            seg.y2
                        ),
                        (255, 0, 255),
                        1
                    )

                # ------------------------------------------------
                # Draw smoothed left line
                # ------------------------------------------------

                draw_y1 = int(
                    horizon_y
                    if horizon_y is not None
                    else h * 0.62
                )

                draw_y2 = int(
                    h * 0.95
                )

                draw_linear_line(
                    visual,
                    left_smoothed,
                    draw_y1,
                    draw_y2,
                    (0, 255, 0),
                    3
                )

                # ------------------------------------------------
                # Draw smoothed right line
                # ------------------------------------------------

                draw_linear_line(
                    visual,
                    right_smoothed,
                    draw_y1,
                    draw_y2,
                    (0, 0, 255),
                    3
                )

                # ------------------------------------------------
                # Draw horizon
                # ------------------------------------------------

                if horizon_y is not None:

                    cv2.line(
                        visual,
                        (
                            0,
                            int(horizon_y)
                        ),
                        (
                            w - 1,
                            int(horizon_y)
                        ),
                        (255, 255, 0),
                        2
                    )

                # ------------------------------------------------
                # Text
                # ------------------------------------------------

                cv2.putText(
                    visual,
                    (
                        f"frame={frame_idx} "
                        f"L={left_status} "
                        f"R={right_status} "
                        f"PAIR={pair_ok}"
                    ),
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    visual,
                    (
                        f"Lseg={len(left_segments)} "
                        f"Rseg={len(right_segments)} "
                        f"reason={pair_reason}"
                    ),
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                cv2.putText(
                    visual,
                    (
                        f"Lconf={left_state_conf:.2f} "
                        f"Rconf={right_state_conf:.2f}"
                    ),
                    (10, 69),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                output_image = os.path.join(
                    args.outdir,
                    f"frame_{frame_idx:06d}.png"
                )

                cv2.imwrite(
                    output_image,
                    visual
                )

    finally:

        csv.close()

        video.release()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 78)

    print("SUMMARY")

    print("=" * 78)

    print(
        f"Frames processed     : "
        f"{processed}"
    )

    print()

    print(
        f"LEFT  OK/HOLD/MISS   : "
        f"{ok_left}/{hold_left}/{miss_left}"
    )

    print(
        f"RIGHT OK/HOLD/MISS   : "
        f"{ok_right}/{hold_right}/{miss_right}"
    )

    print(
        f"Pair validation TRUE : "
        f"{pair_ok_count}/{processed}"
    )

    print()

    print(
        f"CSV                  : "
        f"{csv_path}"
    )

    print(
        f"Images               : "
        f"{args.outdir}"
    )

    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()