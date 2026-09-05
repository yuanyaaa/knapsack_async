#!/usr/bin/env python3
"""Re-clip the offline ``pass_rate`` prior of an evaluated dataset.

The evaluated parquet files ship ``pass_rate = clip(raw_pass_rate, 0.125, 0.875)``. That
lower bound is actively harmful for medium_focus adaptive n: a problem the reference model
never solved (raw_pass_rate == 0) is handed a prior of 0.125, whose 4p(1-p) is 0.4375, so
it gets allocated close to the peak n forever. Clipping to something like 0.01 instead
pushes those rows down to 4p(1-p) = 0.0396 and lets the budget go to genuinely mid-difficulty
problems. ``raw_pass_rate`` is preserved so the original signal stays recoverable.
"""

import argparse
import os
import sys

import pandas as pd


def describe(series, label):
    vc = series.value_counts().sort_index()
    print(f"    {label}: min={series.min():.4g} max={series.max():.4g} mean={series.mean():.4g} 唯一={len(vc)}")
    shown = list(vc.items())[:10]
    print("        " + ", ".join(f"{k:.4g}:{v}" for k, v in shown) + (" ..." if len(vc) > 10 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--low", type=float, default=0.01)
    ap.add_argument("--high", type=float, default=0.99)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst) and not args.overwrite:
        print(f"目标已存在,跳过: {args.dst}")
        return 0

    df = pd.read_parquet(args.src)
    print(f"src = {args.src}")
    print(f"      rows={len(df)}")

    if "extra_info" not in df.columns:
        print("[ERROR] 没有 extra_info 列", file=sys.stderr)
        return 1

    before = pd.to_numeric(df["extra_info"].map(lambda d: d.get("pass_rate")), errors="coerce")
    raw = pd.to_numeric(df["extra_info"].map(lambda d: d.get("raw_pass_rate")), errors="coerce")
    if raw.isna().all():
        print("[ERROR] extra_info 里没有 raw_pass_rate,无法重新 clip", file=sys.stderr)
        return 1
    if raw.isna().any():
        print(f"[ERROR] {int(raw.isna().sum())} 行缺少 raw_pass_rate", file=sys.stderr)
        return 1

    print("  改之前:")
    describe(before, "pass_rate")
    describe(raw, "raw_pass_rate")

    lo, hi = args.low, args.high

    def reclip(d):
        # Copy so we never mutate the object pandas handed us in place; the same dict can be
        # shared across views and an in-place edit would silently corrupt `before`.
        out = dict(d)
        out["pass_rate"] = min(hi, max(lo, float(out["raw_pass_rate"])))
        out["safe_low_bound"] = lo
        out["safe_upper_bound"] = hi
        return out

    df["extra_info"] = df["extra_info"].map(reclip)
    after = pd.to_numeric(df["extra_info"].map(lambda d: d.get("pass_rate")), errors="coerce")

    print(f"  改之后 (clip 到 [{lo}, {hi}]):")
    describe(after, "pass_rate")

    n_low = int((after <= lo + 1e-12).sum())
    n_high = int((after >= hi - 1e-12).sum())
    print(f"  触及下界的行 {n_low} ({n_low/len(df)*100:.1f}%), 触及上界的行 {n_high} ({n_high/len(df)*100:.1f}%)")

    # 4p(1-p) is what medium_focus actually consumes, so report the shift in that space.
    for name, s in (("改前", before), ("改后", after)):
        v = 4.0 * s * (1.0 - s)
        print(f"  {name} 4p(1-p): mean={v.mean():.4f} 中位={v.median():.4f}")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    tmp = args.dst + ".tmp"
    df.to_parquet(tmp, index=False)
    os.rename(tmp, args.dst)
    print(f"已写出: {args.dst} ({os.path.getsize(args.dst)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
