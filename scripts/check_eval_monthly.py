#!/usr/bin/env python3
"""Sanity-check eval_monthly.py output: physical ranges, seasonality, drift.

Global means are cos(lat)-WEIGHTED here. The unweighted mean over a 192x288
grid over-samples the poles and reads ~10 degC low, which is misleading.
"""
import glob
import os
import sys

import numpy as np
import xarray as xr

ROOT = "/scratch/project_2019839/eval_output/monthly"


def gmean(da):
    w = np.cos(np.deg2rad(da.lat))
    return da.weighted(w).mean(dim=("lat", "lon"))


def describe(path):
    d = xr.open_dataset(path)
    name = os.path.basename(path)
    print(f"\n--- {name}  {dict(d.sizes)}  {str(d.time.values[0])[:7]}..{str(d.time.values[-1])[:7]}")
    print(f"    prev_mode={d.attrs.get('prev_mode')} ckpt={d.attrs.get('checkpoint')}")
    out = {}
    for v in d.data_vars:
        a = d[v]
        g = gmean(a).mean("member")                       # (time,)
        clim = g.groupby("time.month").mean()             # seasonal cycle
        print(f"    {v}: weighted gmean={float(g.mean()):.3f}  "
              f"range=[{float(a.min()):.2f}, {float(a.max()):.2f}]  "
              f"seasonal amp={float(clim.max() - clim.min()):.3f}  "
              f"nan={int(np.isnan(a.values).sum())}")
        if v == "PRECT":
            neg = float((a < 0).sum()) / a.size * 100
            print(f"      negative precip: {neg:.3f}% of points, min {float(a.min()):.3f} mm/day")
        # drift: first vs last 24 months of the record
        d0, d1 = float(g[:24].mean()), float(g[-24:].mean())
        print(f"      first 24 mo={d0:.3f}  last 24 mo={d1:.3f}  drift={d1 - d0:+.3f}")
        out[v] = g.values
    return out


def main():
    for sub in sorted(os.listdir(ROOT)):
        files = sorted(glob.glob(os.path.join(ROOT, sub, "*.nc")))
        if not files:
            continue
        print(f"\n=== {sub} ===")
        for f in files:
            describe(f)

    # truth vs free-running on the same scenario/period = exposure bias
    t = os.path.join(ROOT, "test_ep11_truth", "monthly_hist.nc")
    f = os.path.join(ROOT, "test_ep11_free", "monthly_hist.nc")
    if os.path.exists(t) and os.path.exists(f):
        print("\n=== EXPOSURE BIAS: truth vs free-refine, hist ===")
        dt, df = xr.open_dataset(t), xr.open_dataset(f)
        n = min(dt.sizes["time"], df.sizes["time"])
        for v in dt.data_vars:
            gt = gmean(dt[v]).mean("member")[:n]
            gf = gmean(df[v]).mean("member")[:n]
            diff = float((gf - gt).mean())
            trend = float((gf - gt)[-24:].mean() - (gf - gt)[:24].mean())
            print(f"  {v}: mean(free - truth)={diff:+.4f}  "
                  f"growth over record={trend:+.4f}  "
                  f"corr={float(np.corrcoef(gt, gf)[0, 1]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
