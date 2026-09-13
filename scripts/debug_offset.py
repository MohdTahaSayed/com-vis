"""
Quick diagnostic: print offset_normalized over time as a table.
Usage:
    python scripts/debug_offset.py --ego outputs/ego_position.csv
"""
from __future__ import annotations

import argparse
import csv
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego", default="outputs/ego_position.csv")
    ap.add_argument("--every", type=int, default=10,
                    help="print every N samples")
    args = ap.parse_args()

    rows = []
    with open(args.ego, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            off = r["offset_normalized"]
            rows.append({
                "t": float(r["timestamp_s"]),
                "frame": int(r["frame"]),
                "off": float(off) if off else None,
                "status": r["status"],
            })

    print(f"{'t(s)':>8} {'frame':>7} {'status':>5} {'off':>8}")
    print("-" * 34)
    for i, r in enumerate(rows):
        if i % args.every != 0:
            continue
        off_str = f"{r['off']:+.3f}" if r["off"] is not None else "  --  "
        print(f"{r['t']:>8.1f} {r['frame']:>7d} {r['status']:>5} {off_str:>8}")

    # summary of sign changes
    print()
    valid = [(r["t"], r["off"]) for r in rows if r["off"] is not None]
    print(f"[summary] {len(valid)} valid samples")
    if valid:
        xs = [v[1] for v in valid]
        print(f"  min={min(xs):+.3f}  max={max(xs):+.3f}")

        # count sign flips
        flips = 0
        prev_sign = 0
        flips_list = []
        for t, o in valid:
            s = 1 if o > 0.10 else (-1 if o < -0.10 else 0)
            if s != 0 and prev_sign != 0 and s != prev_sign:
                flips += 1
                flips_list.append((t, prev_sign, s))
            if s != 0:
                prev_sign = s
        print(f"  sign flips (|off|>0.10): {flips}")
        for t, ps, ns in flips_list[:20]:
            print(f"    t={t:7.1f}s  {ps:+d} -> {ns:+d}")


if __name__ == "__main__":
    main()