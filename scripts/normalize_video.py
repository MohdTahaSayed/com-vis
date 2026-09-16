
from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2


TARGET_W = 720
TARGET_H = 576
TARGET_FPS = 25.0


def fmt_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def already_matches(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return (w == TARGET_W and h == TARGET_H and abs(fps - TARGET_FPS) < 0.5)


def center_crop_to_ratio(frame, target_ratio: float):
    """
    Center-crop frame to target aspect ratio (w/h).
    Returns cropped frame.
    """
    h, w = frame.shape[:2]
    src_ratio = w / h

    if abs(src_ratio - target_ratio) < 0.01:
        return frame

    if src_ratio > target_ratio:
        # too wide -> crop left/right
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        return frame[:, x0:x0 + new_w]
    else:
        # too tall -> crop top/bottom
        new_h = int(round(w / target_ratio))
        y0 = (h - new_h) // 2
        return frame[y0:y0 + new_h, :]


def normalize(input_path: str, output_path: str, verbose: bool = True) -> str:
    input_path = fmt_path(input_path)
    output_path = fmt_path(output_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input not found: {input_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = src_frames / src_fps if src_fps > 0 else 0.0

    if verbose:
        print(f"[norm] input : {input_path}")
        print(f"[norm]         {src_w}x{src_h} @ {src_fps:.2f} fps, "
              f"{src_frames} frames ({duration:.1f}s)")
        print(f"[norm] output: {output_path}")
        print(f"[norm]         {TARGET_W}x{TARGET_H} @ {TARGET_FPS:.2f} fps")

    target_ratio = TARGET_W / TARGET_H   # 1.25

    # We will produce TARGET_FPS frames. Strategy: iterate over source
    # frames and decide for each source frame how many output frames to
    # emit (nearest-neighbor in time).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, TARGET_FPS,
                             (TARGET_W, TARGET_H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer: {output_path}")

    out_frame_idx = 0
    total_out_est = int(round(duration * TARGET_FPS))
    if verbose:
        print(f"[norm] writing ~{total_out_est} output frames...")

    src_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Which output time window does this source frame cover?
        t_start = src_idx / src_fps
        t_end = (src_idx + 1) / src_fps

        # Output frames whose timestamp falls in [t_start, t_end)
        while out_frame_idx / TARGET_FPS < t_end:
            out_t = out_frame_idx / TARGET_FPS
            if out_t < t_start:
                out_frame_idx += 1
                continue

            # build the output frame
            cropped = center_crop_to_ratio(frame, target_ratio)
            resized = cv2.resize(cropped, (TARGET_W, TARGET_H),
                                 interpolation=cv2.INTER_AREA)
            writer.write(resized)
            out_frame_idx += 1

        src_idx += 1

        # progress every 500 source frames
        if verbose and src_idx % 500 == 0:
            pct = 100.0 * src_idx / max(src_frames, 1)
            print(f"[norm]   {pct:5.1f}%  src={src_idx}/{src_frames}  "
                  f"out={out_frame_idx}")

    cap.release()
    writer.release()

    if verbose:
        print(f"[norm] done. {out_frame_idx} frames written.")

    return output_path


def default_output_path(input_path: str, outdir: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(outdir, f"{base}_720x576_25fps.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="source video (any resolution/fps)")
    ap.add_argument("--output", default=None,
                    help="explicit output path; overrides --outdir")
    ap.add_argument("--outdir", default="outputs_normalized",
                    help="directory for auto-named output")
    ap.add_argument("--force", action="store_true",
                    help="re-encode even if input already matches")
    args = ap.parse_args()

    src = fmt_path(args.input)
    if args.output:
        dst = fmt_path(args.output)
    else:
        dst = fmt_path(default_output_path(src, args.outdir))

    if already_matches(src) and not args.force:
        print(f"[norm] input already {TARGET_W}x{TARGET_H}@{TARGET_FPS} fps")
        print(f"[norm] copying to {dst}")
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[norm] done.")
        return 0

    normalize(src, dst, verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())