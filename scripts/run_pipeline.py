from __future__ import annotations

import argparse
import os
import sys
import time as _time

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

from src.horizon import HorizonDetector, HorizonConfig
from src.lane_color import LaneColor, LaneColorConfig
from src.lane_roi import LaneROI, RoiConfig
from src.lane_edges import LaneEdges, CannyConfig
from src.lane_fit import LaneFitter, SlidingWindowConfig
from src.lane_validation import LaneValidation, ValidationConfig
from src.lane_state import LaneState, StateConfig

from src.ego_position import EgoPosition, EgoConfig

from src.csv_writers import (
    EgoPositionCSV,
    LaneChangesCSV,
    SignsCSV,
)

from src.lane_change import (
    LaneChangeDetector,
    LaneChangeConfig,
)

from src.io_video import VideoReader


# ============================================================
# OPTIONAL SIGN MODULES
# ============================================================

try:

    from src.sign_detect import (
        SignDetector,
        SignDetectConfig,
    )

    from src.sign_track import (
        SignTracker,
        SignTrackConfig,
    )

    from src.sign_categories import (
        category_of,
    )

    HAVE_SIGNS = True

except Exception as e:

    HAVE_SIGNS = False

    print(
        f"[warn] sign detection disabled: {e}"
    )


# ============================================================
# CONFIG LOADER
# ============================================================

def load_yaml(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f) or {}


# ============================================================
# DRAW POLYNOMIAL
# ============================================================

def draw_curve(
    frame,
    coeffs,
    y_range,
    color,
    thickness=3,
):

    if coeffs is None:
        return

    h, w = frame.shape[:2]

    ys = np.linspace(
        y_range[0],
        y_range[1],
        80,
    )

    xs = (
        coeffs[0] * ys * ys
        + coeffs[1] * ys
        + coeffs[2]
    )

    pts = np.stack(
        [xs, ys],
        axis=1,
    ).astype(np.int32)

    pts = pts[
        (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    ]

    if len(pts) >= 2:

        cv2.polylines(
            frame,
            [pts],
            False,
            color,
            thickness,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description="Complete IITB Lane Analytics pipeline"
    )

    ap.add_argument(
        "--input",
        required=True,
        help="Input video",
    )

    ap.add_argument(
        "--config",
        default="config/default.yaml",
        help="Configuration YAML",
    )

    ap.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory",
    )

    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for testing",
    )

    ap.add_argument(
        "--sign-every",
        type=int,
        default=5,
        help="Run sign detector every N frames",
    )

    ap.add_argument(
        "--debug-video",
        action="store_true",
        help="Write annotated debug video",
    )

    # ---------------------------------------------------------
    # Tracker reset threshold
    # ---------------------------------------------------------

    ap.add_argument(
        "--tracker-reset-frames",
        type=int,
        default=20,
        help=(
            "After this many consecutive MISS frames on a side, "
            "forget the previous coeffs so the fitter falls back "
            "to the histogram base and reacquires the lane."
        ),
    )

    # ---------------------------------------------------------
    # Ego CSV sampling rate (deliverable requirement: 1 Hz)
    # ---------------------------------------------------------

    ap.add_argument(
        "--ego-hz",
        type=float,
        default=1.0,
        help=(
            "Sampling rate for the ego_position.csv file. "
            "Default 1 Hz (deliverable requirement)."
        ),
    )

    # ---------------------------------------------------------
    # Lane-change detection sampling rate (internal)
    # ---------------------------------------------------------

    ap.add_argument(
        "--lane-change-hz",
        type=float,
        default=4.0,
        help=(
            "Sampling rate for internal lane-change detection. "
            "Higher than --ego-hz for accuracy. Default 4 Hz."
        ),
    )

    args = ap.parse_args()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    cfg = load_yaml(
        args.config
    )

    # --------------------------------------------------------
    # Lane pipeline modules
    # --------------------------------------------------------

    horizon = HorizonDetector(
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

    lane_color = LaneColor(
        LaneColorConfig.from_dict(
            cfg.get(
                "lane_color",
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
        reinforce_with_hsv=True,
    )

    fitter = LaneFitter(
        SlidingWindowConfig.from_dict(
            cfg.get(
                "sliding_window",
                {}
            )
        )
    )

    validator = LaneValidation(
        ValidationConfig.from_dict(
            cfg.get(
                "validation",
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

    ego = EgoPosition(
        EgoConfig.from_dict(
            cfg.get(
                "ego_position",
                {}
            )
        )
    )

    lane_change_detector = LaneChangeDetector(
        LaneChangeConfig.from_dict(
            cfg.get(
                "lane_change",
                {}
            )
        )
    )

    # --------------------------------------------------------
    # Minimum confidence
    # --------------------------------------------------------

    MIN_CONF = float(
        cfg.get(
            "lane_state",
            {}
        ).get(
            "min_confidence",
            0.15,
        )
    )

    # ========================================================
    # SIGN DETECTION
    # ========================================================

    sign_detector = None
    sign_tracker = None

    if (
        HAVE_SIGNS
        and cfg.get(
            "sign_detect",
            {}
        ).get(
            "enabled",
            True,
        )
    ):

        try:

            sign_detector = SignDetector(
                SignDetectConfig.from_dict(
                    cfg.get(
                        "sign_detect",
                        {}
                    )
                )
            )

            sign_tracker = SignTracker(
                SignTrackConfig.from_dict(
                    cfg.get(
                        "sign_track",
                        {}
                    )
                )
            )

            print(
                "[info] sign detection ENABLED"
            )

        except Exception as e:

            print(
                f"[warn] sign detector failed "
                f"to initialize: {e}"
            )

    # ========================================================
    # CSV OUTPUTS
    # ========================================================

    ego_csv = EgoPositionCSV(
        os.path.join(
            args.outdir,
            "ego_position.csv",
        )
    )

    lane_change_csv = LaneChangesCSV(
        os.path.join(
            args.outdir,
            "lane_changes.csv",
        )
    )

    sign_csv = SignsCSV(
        os.path.join(
            args.outdir,
            "signs.csv",
        )
    )

    # ========================================================
    # VIDEO
    # ========================================================

    vr = VideoReader(
        args.input
    )

    fps = vr.info.fps

    # ego position sample step (CSV output rate)
    if args.ego_hz <= 0:
        ego_step = int(round(fps))
    else:
        ego_step = max(1, int(round(fps / args.ego_hz)))

    # lane-change sample step (internal detection rate)
    if args.lane_change_hz <= 0:
        lc_step = int(round(fps))
    else:
        lc_step = max(1, int(round(fps / args.lane_change_hz)))

    print()
    print("=" * 70)
    print("IITB LANE ANALYTICS PIPELINE")
    print("=" * 70)

    print(
        f"video       = {args.input}"
    )

    print(
        f"resolution  = "
        f"{vr.info.width}x{vr.info.height}"
    )

    print(
        f"fps         = {fps:.2f}"
    )

    print(
        f"frames      = {vr.info.frame_count}"
    )

    print(
        f"duration    = "
        f"{vr.info.duration_s:.2f}s"
    )

    print(
        f"ego CSV     = every {ego_step} frames "
        f"({args.ego_hz:.2f} Hz)"
    )

    print(
        f"lane-change = every {lc_step} frames "
        f"({args.lane_change_hz:.2f} Hz, internal)"
    )

    print(
        f"tracker reset = after "
        f"{args.tracker_reset_frames} consecutive MISS"
    )

    print(
        f"lane-change detector = "
        f"baseline-shift "
        f"(shift frac={lane_change_detector.cfg.min_total_shift_frac}, "
        f"abs={lane_change_detector.cfg.min_total_shift_px}px, "
        f"buf={lane_change_detector.cfg.baseline_buffer_len}, "
        f"old_frac={lane_change_detector.cfg.baseline_old_frac}, "
        f"persist={lane_change_detector.cfg.persist_samples}, "
        f"alternation={lane_change_detector.cfg.enforce_alternation})"
    )

    print("=" * 70)

    # ========================================================
    # DEBUG VIDEO
    # ========================================================

    writer = None

    if args.debug_video:

        video_path = os.path.join(
            args.outdir,
            "annotated.mp4",
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            video_path,
            fourcc,
            fps,
            (
                int(vr.info.width),
                int(vr.info.height),
            ),
        )

        print(
            f"[info] debug video:"
            f" {video_path}"
        )

    # ========================================================
    # PROCESSING LOOP
    # ========================================================

    t0 = _time.time()

    n = 0

    lane_ok_samples = 0
    lane_miss_samples = 0

    raw_sign_detections = 0

    # --------------------------------------------------------
    # Temporal tracker state (for LaneFitter)
    # --------------------------------------------------------

    previous_left = None
    previous_right = None

    left_miss_count = 0
    right_miss_count = 0

    # --------------------------------------------------------
    # Every frame
    # --------------------------------------------------------

    for idx, ts, frame in vr.iter_frames(
        step=1
    ):

        # ====================================================
        # 1. HORIZON
        # ====================================================

        horizon_y = horizon.detect(
            frame
        )

        # ====================================================
        # 2. CANNY + ROI + HSV
        # ====================================================

        edges_roi, _, _ = (
            edges_module.compute(
                frame,
                top_y_override=horizon_y,
            )
        )

        # ====================================================
        # 3. LANE FIT (with temporal tracking)
        # ====================================================

        left, right = fitter.fit(
            edges_roi,
            previous_left=previous_left,
            previous_right=previous_right,
        )

        # ====================================================
        # 4. VALIDATION
        # ====================================================

        h, w = frame.shape[:2]

        y_range = (
            int(h * 0.55),
            int(h * 0.95),
        )

        validation = validator.validate(
            left.coeffs,
            right.coeffs,
            y_range,
            w,
        )

        # ====================================================
        # 4b. UPDATE TRACKER (miss-counter reset)
        # ====================================================

        # ---- LEFT ----
        if (
            validation.left_ok
            and left.coeffs is not None
        ):
            previous_left = left.coeffs.copy()
            left_miss_count = 0
        else:
            left_miss_count += 1
            if left_miss_count >= args.tracker_reset_frames:
                previous_left = None
                left_miss_count = 0

        # ---- RIGHT ----
        if (
            validation.right_ok
            and right.coeffs is not None
        ):
            previous_right = right.coeffs.copy()
            right_miss_count = 0
        else:
            right_miss_count += 1
            if right_miss_count >= args.tracker_reset_frames:
                previous_right = None
                right_miss_count = 0

        # ====================================================
        # 5. TEMPORAL STATE
        # ====================================================

        (
            l_coeffs,
            l_status,
            l_conf,
        ), (
            r_coeffs,
            r_status,
            r_conf,
        ) = state.update(

            left.coeffs
            if validation.left_ok
            else None,

            left.confidence
            if validation.left_ok
            else 0.0,

            right.coeffs
            if validation.right_ok
            else None,

            right.confidence
            if validation.right_ok
            else 0.0,
        )

        # ====================================================
        # 6. COMPUTE EGO MEASUREMENT (shared)
        # ====================================================

        is_lc_step = (idx % lc_step == 0)
        is_ego_step = (idx % ego_step == 0)

        if is_lc_step:

            measurement = ego.compute(
                l_coeffs,
                r_coeffs,
                w,
                h,
            )

            l_usable = (
                l_coeffs is not None
                and l_status in ("OK", "HOLD")
            )
            r_usable = (
                r_coeffs is not None
                and r_status in ("OK", "HOLD")
            )
            both_usable = l_usable and r_usable

            if both_usable:
                min_conf = float(min(l_conf, r_conf))
            else:
                min_conf = 0.0

            # ------------------------------------------------
            # Boundary x positions at ego eval row
            # ------------------------------------------------

            y_eval = int(ego.cfg.lane_eval_y_frac * h)

            left_x_eval = None
            right_x_eval = None

            if l_coeffs is not None:
                left_x_eval = float(
                    l_coeffs[0] * y_eval * y_eval
                    + l_coeffs[1] * y_eval
                    + l_coeffs[2]
                )

            if r_coeffs is not None:
                right_x_eval = float(
                    r_coeffs[0] * y_eval * y_eval
                    + r_coeffs[1] * y_eval
                    + r_coeffs[2]
                )

            # ------------------------------------------------
            # Valid / invalid
            # ------------------------------------------------

            if (
                both_usable
                and measurement.valid
                and min_conf >= MIN_CONF
            ):
                offset_px = measurement.offset_px
                lane_width_px = measurement.lane_width_px

                if lane_width_px is not None and lane_width_px != 0:
                    offset_normalized = offset_px / lane_width_px
                else:
                    offset_normalized = None

                status = "OK"

            else:
                offset_px = None
                lane_width_px = None
                offset_normalized = None
                status = "MISS"

            # ------------------------------------------------
            # Lane-change detector (internal, high-rate)
            # ------------------------------------------------

            event = lane_change_detector.feed(
                idx,
                ts,
                left_x_eval,
                right_x_eval,
                measurement.lane_width_px,
                status,
            )

            if event is not None:

                lane_change_csv.row(
                    f"{event.timestamp_s:.2f}",
                    event.frame,
                    event.direction,
                    f"{event.magnitude:.3f}",
                )

                print(
                    f"[LANE CHANGE] "
                    f"t={event.timestamp_s:.2f}s "
                    f"direction={event.direction} "
                    f"magnitude={event.magnitude:.1f}px "
                    f"(dL={event.delta_left_px:+.1f}, "
                    f"dR={event.delta_right_px:+.1f})"
                )

                state.reset()
                previous_left = None
                previous_right = None
                left_miss_count = 0
                right_miss_count = 0

                print(
                    "[LANE CHANGE] "
                    "LaneState + tracker reset"
                )

        # ====================================================
        # 7. WRITE EGO CSV (1 Hz)
        # ====================================================

        if is_ego_step:

            # If we didn't run the measurement this frame
            # (ego_step != lc_step alignment), compute it now.
            if not is_lc_step:

                measurement = ego.compute(
                    l_coeffs,
                    r_coeffs,
                    w,
                    h,
                )

                l_usable = (
                    l_coeffs is not None
                    and l_status in ("OK", "HOLD")
                )
                r_usable = (
                    r_coeffs is not None
                    and r_status in ("OK", "HOLD")
                )
                both_usable = l_usable and r_usable

                if both_usable:
                    min_conf = float(min(l_conf, r_conf))
                else:
                    min_conf = 0.0

                y_eval = int(ego.cfg.lane_eval_y_frac * h)
                left_x_eval = None
                right_x_eval = None

                if l_coeffs is not None:
                    left_x_eval = float(
                        l_coeffs[0] * y_eval * y_eval
                        + l_coeffs[1] * y_eval
                        + l_coeffs[2]
                    )

                if r_coeffs is not None:
                    right_x_eval = float(
                        r_coeffs[0] * y_eval * y_eval
                        + r_coeffs[1] * y_eval
                        + r_coeffs[2]
                    )

                if (
                    both_usable
                    and measurement.valid
                    and min_conf >= MIN_CONF
                ):
                    offset_px = measurement.offset_px
                    lane_width_px = measurement.lane_width_px
                    if lane_width_px is not None and lane_width_px != 0:
                        offset_normalized = offset_px / lane_width_px
                    else:
                        offset_normalized = None
                    status = "OK"
                else:
                    offset_px = None
                    lane_width_px = None
                    offset_normalized = None
                    status = "MISS"

            # Counters
            if status == "OK":
                lane_ok_samples += 1
            else:
                lane_miss_samples += 1

            # Write row
            ego_csv.row(
                f"{ts:.2f}",
                idx,
                f"{offset_px:.2f}" if offset_px is not None else "",
                f"{lane_width_px:.1f}" if lane_width_px is not None else "",
                f"{offset_normalized:.3f}" if offset_normalized is not None else "",
                f"{measurement.lane_center_x:.1f}" if measurement.lane_center_x is not None else "",
                f"{left_x_eval:.1f}" if left_x_eval is not None else "",
                f"{right_x_eval:.1f}" if right_x_eval is not None else "",
                f"{min_conf:.2f}",
                status,
            )

        # ====================================================
        # 8. SIGN DETECTION
        # ====================================================

        if (
            sign_detector is not None
            and sign_tracker is not None
            and args.sign_every > 0
            and idx % args.sign_every == 0
        ):

            detections = sign_detector.detect(
                frame
            )

            raw_sign_detections += len(
                detections
            )

            sign_tracker.update(
                idx,
                ts,
                detections,
            )

        # ====================================================
        # 9. DEBUG VIDEO
        # ====================================================

        if writer is not None:

            vis = frame.copy()

            if horizon_y is not None:
                cv2.line(
                    vis,
                    (0, horizon_y),
                    (w, horizon_y),
                    (0, 255, 255),
                    2,
                )

            draw_curve(vis, l_coeffs, y_range, (0, 255, 255), 3)
            draw_curve(vis, r_coeffs, y_range, (0, 255, 0), 3)

            ego_x = int(0.50 * w)
            cv2.line(
                vis,
                (ego_x, int(h * 0.75)),
                (ego_x, int(h * 0.95)),
                (255, 0, 0),
                2,
            )

            text_lines = [
                f"L={l_status} R={r_status}",
                f"Lconf={l_conf:.2f} Rconf={r_conf:.2f}",
                f"frame={idx} t={ts:.2f}s",
                f"Tracker misses: L={left_miss_count} R={right_miss_count}",
            ]

            for i, text in enumerate(text_lines):
                y = 25 + i * 23
                cv2.putText(
                    vis, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA,
                )
                cv2.putText(
                    vis, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

            if sign_tracker is not None:
                vis = sign_tracker.debug_render(vis, None)

            writer.write(vis)

        # ====================================================
        # FRAME COUNTER
        # ====================================================

        n += 1

        if (
            args.max_frames is not None
            and n >= args.max_frames
        ):
            break

    # ========================================================
    # FINALIZE SIGN TRACKS
    # ========================================================

    finalized_signs = []

    if sign_tracker is not None:

        finalized_signs = sign_tracker.finalize()

        for track in finalized_signs:

            sign_csv.row(
                f"{track.first_ts:.2f}",
                track.first_frame,
                track.cls_name,
                category_of(track.cls_name),
                track.bbox[0],
                track.bbox[1],
                track.bbox[2],
                track.bbox[3],
                f"{track.confidence:.2f}",
            )

    # ========================================================
    # CLOSE FILES
    # ========================================================

    ego_csv.close()
    lane_change_csv.close()
    sign_csv.close()

    vr.release()

    if writer is not None:
        writer.release()

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    dt = _time.time() - t0

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(f"frames processed = {n}")
    print(f"processing time  = {dt:.1f}s")

    if dt > 0:
        print(f"processing speed = {n / dt:.1f} FPS")

    print()
    print("Ego samples:")
    print(f"  OK   = {lane_ok_samples}")
    print(f"  MISS = {lane_miss_samples}")
    print()
    print(f"Raw sign detections = {raw_sign_detections}")
    print(f"Confirmed signs     = {len(finalized_signs)}")
    print()
    print("Outputs:")
    print(f"  {os.path.join(args.outdir, 'ego_position.csv')}")
    print(f"  {os.path.join(args.outdir, 'lane_changes.csv')}")
    print(f"  {os.path.join(args.outdir, 'signs.csv')}")
    if writer is not None:
        print(f"  {os.path.join(args.outdir, 'annotated.mp4')}")
    print("=" * 70)


if __name__ == "__main__":
    main()