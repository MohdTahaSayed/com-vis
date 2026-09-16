
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.lane_change import LaneChangeDetector, LaneChangeConfig
from src.csv_writers import LaneChangesCSV


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego", default="outputs/ego_position.csv")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--debug-every", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_cfg(args.config)

    det = LaneChangeDetector(
        LaneChangeConfig.from_dict(cfg.get("lane_change", {}))
    )
    print(f"[init] lane_change cfg: {vars(det.cfg)}")

    changes_path = os.path.join(args.outdir, "lane_changes.csv")
    writer = LaneChangesCSV(changes_path)

    n_samples = 0
    n_events = 0
    n_ok = 0
    n_miss = 0
    n_ok_since_debug = 0

    with open(args.ego, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_samples += 1
            frame = int(row["frame"])
            ts = float(row["timestamp_s"])
            status = row["status"]
            off_norm_str = row["offset_normalized"]
            off_norm = float(off_norm_str) if off_norm_str else None

            if status == "OK":
                n_ok += 1
            else:
                n_miss += 1

            ev = det.feed(frame, ts, off_norm, status)

            if ev is not None:
                writer.row(
                    f"{ev.timestamp_s:.2f}",
                    ev.frame,
                    ev.direction,
                    f"{ev.magnitude:.3f}",
                )
                n_events += 1
                print(f"[EVENT] t={ev.timestamp_s:7.2f}s  "
                      f"frame={ev.frame:6d}  {ev.direction:5s}  "
                      f"mag={ev.magnitude:.3f}")

            # debug print: every N valid (OK) samples
            if status == "OK" and off_norm is not None:
                n_ok_since_debug += 1
                if (args.debug_every > 0
                        and n_ok_since_debug % args.debug_every == 0):
                    win_len = len(det._window)
                    if win_len >= det.cfg.window_samples:
                        med = float(np.median(det._window))
                        state = det._classify(med)
                        print(f"  [dbg] t={ts:7.1f}s  off={off_norm:+.3f}  "
                              f"med={med:+.3f}  state={state:8s}  "
                              f"last={det._last_state}  "
                              f"peak={det._peak_since_transition:.3f}  "
                              f"cd={det._cooldown}")
                    else:
                        print(f"  [dbg] t={ts:7.1f}s  off={off_norm:+.3f}  "
                              f"(window only {win_len}/{det.cfg.window_samples})")

    writer.close()
    print()
    print(f"[info] processed {n_samples} samples (OK={n_ok}, MISS={n_miss})")
    print(f"[info] detected {n_events} lane-change events")
    print(f"[ok] wrote {changes_path}")


if __name__ == "__main__":
    main()