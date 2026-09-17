

import argparse
import os

import cv2
import numpy as np
import yaml


# =====================================================================
# Configuration
# =====================================================================

CONFIG_PATH = "config/default.yaml"


# =====================================================================
# Load YAML
# =====================================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =====================================================================
# Estimate horizon
# =====================================================================

def estimate_horizon(frame, cfg):
    """
    Estimate the horizon using horizontal image statistics.

    This is an inspection-only implementation intended to visualize
    approximately where the horizon detector searches.

    It follows the same configured search range:
        search_top_frac
        search_bot_frac

    Returns:
        horizon_y
    """

    h, w = frame.shape[:2]

    horizon_cfg = cfg.get("horizon", {})

    search_top_frac = float(
        horizon_cfg.get(
            "search_top_frac",
            0.30
        )
    )

    search_bot_frac = float(
        horizon_cfg.get(
            "search_bot_frac",
            0.75
        )
    )

    min_y_frac = float(
        horizon_cfg.get(
            "min_y_frac",
            0.35
        )
    )

    max_y_frac = float(
        horizon_cfg.get(
            "max_y_frac",
            0.65
        )
    )

    y_start = int(
        h * search_top_frac
    )

    y_end = int(
        h * search_bot_frac
    )

    # Convert to grayscale.
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur reduces small objects/noise.
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Horizontal gradient.
    #
    # The idea is to find a strong change in image statistics
    # around the road/sky/land transition.
    gy = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.abs(gy)

    # Calculate average vertical gradient for each row.
    row_score = np.mean(
        magnitude,
        axis=1
    )

    # Restrict to configured search region.
    search_scores = row_score[
        y_start:y_end
    ]

    if len(search_scores) == 0:
        return int(h * 0.50)

    # Smooth row scores.
    kernel_size = 15

    kernel = (
        np.ones(kernel_size)
        / kernel_size
    )

    smooth = np.convolve(
        search_scores,
        kernel,
        mode="same"
    )

    # Best row.
    best_relative = int(
        np.argmax(smooth)
    )

    horizon_y = (
        y_start
        + best_relative
    )

    # Apply configured allowed range.
    min_y = int(
        h * min_y_frac
    )

    max_y = int(
        h * max_y_frac
    )

    horizon_y = int(
        np.clip(
            horizon_y,
            min_y,
            max_y
        )
    )

    return horizon_y


# =====================================================================
# Draw ROI
# =====================================================================

def normalized_vertices_to_pixels(
    vertices,
    w,
    h
):
    points = []

    for x_frac, y_frac in vertices:

        x = int(
            float(x_frac) * w
        )

        y = int(
            float(y_frac) * h
        )

        points.append(
            [x, y]
        )

    return np.array(
        points,
        dtype=np.int32
    )


# =====================================================================
# Main visualization
# =====================================================================

def visualize(frame, cfg):
    h, w = frame.shape[:2]

    output = frame.copy()

    # ---------------------------------------------------------------
    # Read ROI configuration
    # ---------------------------------------------------------------

    roi_cfg = cfg.get(
        "roi",
        {}
    )

    vertices = roi_cfg.get(
        "vertices",
        [
            [0.10, 0.90],
            [0.30, 0.55],
            [0.70, 0.55],
            [0.90, 0.90],
        ]
    )

    roi_pts = normalized_vertices_to_pixels(
        vertices,
        w,
        h
    )

    # ---------------------------------------------------------------
    # Horizon configuration
    # ---------------------------------------------------------------

    horizon_cfg = cfg.get(
        "horizon",
        {}
    )

    search_top_frac = float(
        horizon_cfg.get(
            "search_top_frac",
            0.30
        )
    )

    search_bot_frac = float(
        horizon_cfg.get(
            "search_bot_frac",
            0.75
        )
    )

    min_y_frac = float(
        horizon_cfg.get(
            "min_y_frac",
            0.35
        )
    )

    max_y_frac = float(
        horizon_cfg.get(
            "max_y_frac",
            0.65
        )
    )

    # ---------------------------------------------------------------
    # Calculate horizon
    # ---------------------------------------------------------------

    horizon_y = estimate_horizon(
        frame,
        cfg
    )

    search_top_y = int(
        h * search_top_frac
    )

    search_bot_y = int(
        h * search_bot_frac
    )

    min_horizon_y = int(
        h * min_y_frac
    )

    max_horizon_y = int(
        h * max_y_frac
    )

    # ---------------------------------------------------------------
    # Create outside-ROI overlay
    # ---------------------------------------------------------------

    roi_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv2.fillPoly(
        roi_mask,
        [roi_pts],
        255
    )

    # Darken everything OUTSIDE the ROI.
    darkened = output.copy()

    darkened[:] = (
        darkened[:] * 0.30
    ).astype(np.uint8)

    outside = (
        roi_mask == 0
    )

    output[outside] = darkened[outside]

    # ---------------------------------------------------------------
    # Draw ROI interior
    # ---------------------------------------------------------------

    cv2.polylines(
        output,
        [roi_pts],
        True,
        (255, 255, 0),
        4
    )

    # ---------------------------------------------------------------
    # Draw ROI vertices
    # ---------------------------------------------------------------

    labels = [
        "BOTTOM-LEFT",
        "TOP-LEFT",
        "TOP-RIGHT",
        "BOTTOM-RIGHT",
    ]

    for point, label in zip(
        roi_pts,
        labels
    ):

        x, y = point

        cv2.circle(
            output,
            (int(x), int(y)),
            7,
            (255, 255, 0),
            -1
        )

        cv2.putText(
            output,
            label,
            (
                int(x) + 8,
                int(y) - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    # ---------------------------------------------------------------
    # Horizon search region
    # ---------------------------------------------------------------

    # Draw search region as horizontal lines.
    cv2.line(
        output,
        (0, search_top_y),
        (w, search_top_y),
        (255, 0, 255),
        2
    )

    cv2.line(
        output,
        (0, search_bot_y),
        (w, search_bot_y),
        (255, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"HORIZON SEARCH: {search_top_frac:.2f} - "
        f"{search_bot_frac:.2f}",
        (10, search_top_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA
    )

    # ---------------------------------------------------------------
    # Allowed horizon range
    # ---------------------------------------------------------------

    cv2.line(
        output,
        (0, min_horizon_y),
        (w, min_horizon_y),
        (0, 165, 255),
        1
    )

    cv2.line(
        output,
        (0, max_horizon_y),
        (w, max_horizon_y),
        (0, 165, 255),
        1
    )

    # ---------------------------------------------------------------
    # Detected horizon
    # ---------------------------------------------------------------

    cv2.line(
        output,
        (0, horizon_y),
        (w, horizon_y),
        (0, 0, 255),
        4
    )

    cv2.putText(
        output,
        f"DETECTED HORIZON y={horizon_y}px "
        f"({horizon_y / h:.3f})",
        (10, horizon_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # ---------------------------------------------------------------
    # Vanishing point
    # ---------------------------------------------------------------

    sliding_cfg = cfg.get(
        "sliding_window",
        {}
    )

    vanishing_x_frac = float(
        sliding_cfg.get(
            "vanishing_x_frac",
            0.50
        )
    )

    vanishing_x = int(
        w * vanishing_x_frac
    )

    # Place point at detected horizon.
    cv2.circle(
        output,
        (
            vanishing_x,
            horizon_y
        ),
        10,
        (0, 255, 255),
        -1
    )

    cv2.drawMarker(
        output,
        (
            vanishing_x,
            horizon_y
        ),
        (0, 0, 0),
        cv2.MARKER_CROSS,
        30,
        3
    )

    cv2.putText(
        output,
        f"VANISHING POINT "
        f"x={vanishing_x}px "
        f"({vanishing_x_frac:.2f}W)",
        (
            vanishing_x + 12,
            horizon_y + 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ---------------------------------------------------------------
    # ROI information
    # ---------------------------------------------------------------

    roi_top_y = int(
        h * vertices[1][1]
    )

    roi_bottom_y = int(
        h * vertices[0][1]
    )

    text_y = 25

    info = [
        f"FRAME: {w} x {h}",
        f"ROI TOP: y={roi_top_y}px "
        f"({vertices[1][1]:.2f}H)",
        f"ROI BOTTOM: y={roi_bottom_y}px "
        f"({vertices[0][1]:.2f}H)",
        f"VANISH X: {vanishing_x}px "
        f"({vanishing_x_frac:.2f}W)",
    ]

    for text in info:

        cv2.putText(
            output,
            text,
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        text_y += 22

    # ---------------------------------------------------------------
    # Legend
    # ---------------------------------------------------------------

    legend_x = 10
    legend_y = h - 115

    legend = [
        (
            (255, 255, 0),
            "CYAN: ROI boundary"
        ),
        (
            (255, 0, 255),
            "MAGENTA: horizon search region"
        ),
        (
            (0, 0, 255),
            "RED: detected horizon"
        ),
        (
            (0, 165, 255),
            "ORANGE: allowed horizon range"
        ),
        (
            (0, 255, 255),
            "YELLOW: vanishing point"
        ),
    ]

    for color, text in legend:

        cv2.line(
            output,
            (
                legend_x,
                legend_y
            ),
            (
                legend_x + 25,
                legend_y
            ),
            color,
            4
        )

        cv2.putText(
            output,
            text,
            (
                legend_x + 35,
                legend_y + 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        legend_y += 20

    # ---------------------------------------------------------------
    # "IGNORED" label outside ROI
    # ---------------------------------------------------------------

    cv2.putText(
        output,
        "DARK AREA = NOT USED BY ROI",
        (
            15,
            h // 2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        2,
        cv2.LINE_AA
    )

    return output, horizon_y


# =====================================================================
# Video frame extraction
# =====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input video"
    )

    parser.add_argument(
        "--time",
        type=float,
        default=900.0,
        help="Time in seconds"
    )

    parser.add_argument(
        "--output",
        default="outputs/roi_horizon_inspection.jpg",
        help="Output image"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load config
    # ---------------------------------------------------------------

    cfg = load_config(
        CONFIG_PATH
    )

    # ---------------------------------------------------------------
    # Open video
    # ---------------------------------------------------------------

    cap = cv2.VideoCapture(
        args.input
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {args.input}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    frame_number = int(
        args.time * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:

        raise RuntimeError(
            f"Could not read frame at "
            f"{args.time}s"
        )

    # ---------------------------------------------------------------
    # Visualize
    # ---------------------------------------------------------------

    result, horizon_y = visualize(
        frame,
        cfg
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    cv2.imwrite(
        args.output,
        result
    )

    # ---------------------------------------------------------------
    # Terminal information
    # ---------------------------------------------------------------

    h, w = frame.shape[:2]

    print()
    print("=" * 70)
    print("ROI + HORIZON INSPECTION")
    print("=" * 70)

    print(
        f"[info] video       : {args.input}"
    )

    print(
        f"[info] time        : {args.time:.2f}s"
    )

    print(
        f"[info] frame       : {frame_number}"
    )

    print(
        f"[info] resolution  : {w} x {h}"
    )

    print(
        f"[info] horizon     : y={horizon_y}px "
        f"({horizon_y / h:.3f}H)"
    )

    print()

    print(
        "[info] ROI vertices:"
    )

    for vertex in cfg["roi"]["vertices"]:

        x_frac, y_frac = vertex

        print(
            f"       x={x_frac:.2f}W "
            f"y={y_frac:.2f}H"
        )

    print()

    print(
        "[info] output      : "
        f"{args.output}"
    )

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()