#!/usr/bin/env python3
"""
Final-decade anomaly maps: CESM2, and how far each conditioning arm is from it.

One figure per variable. Columns are the four training experiments. The top row
is held-out CESM2's ensemble-mean anomaly (final decade minus 1850-1900); every
row below is ONE ARM MINUS CESM2, on a shared symmetric scale per variable so
the arms compare directly. Each difference panel is annotated with its
cos(lat)-weighted pattern correlation against CESM2's anomaly, and its RMSE.

Arms: the paper checkpoint (v1, clipped), asinh99 (CO2 v1, SUL/BC asinh),
minmax (linear to the smoothed max, no clip) and v1noclip (v1's anchors, clip
removed) -- each at its newest evaluated epoch.

The CESM2 reference and the paper checkpoint come from the cache written by
make_ensemble_mean_maps.py (held-out members, 25-member paper eval). The three
new arms come straight from their eval NetCDFs over the LUMI mount, which hold
FIVE members, so their maps carry more sampling noise than the paper row --
read small-scale speckle there as noise, not as an arm difference.

    ~/miniconda3/envs/plotting/bin/python scripts/make_arm_comparison_maps.py
"""
import argparse
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_ensemble_mean_maps import (
    SCENARIOS, META, BASELINE, apply_baselines, to_celsius, weighted_stats,
    make_axes, draw_map, panel_label, to_pm180, HAVE_CARTOPY,
)

EVAL_ROOT = os.path.expanduser("~/mnt/lumi_sc/eval_output")
ARMS = [  # label, eval run dir, epoch
    ("asinh99 (SUL/BC asinh)", "run_asinh99_co2fix", 470),
    ("minmax (no clip, max anchor)", "run_minmax_co2fix", 470),
    ("v1noclip (v1 anchors, no clip)", "run_v1noclip_co2fix", 240),
    ("v1noclip (v1 anchors, no clip)", "run_v1noclip_co2fix", 410),
]


def arm_maps(var, run, epoch, windows):
    """Final-decade and 1850-1900 ensemble-mean maps from one eval directory."""
    out = {}
    for key in SCENARIOS:
        path = f"{EVAL_ROOT}/{run}/best_ep{epoch:04d}/{var}_{key}.nc"
        with xr.open_dataset(path) as ds:
            da = ds[f"{var}_model"]
            yrs = np.asarray(ds["year"].values).astype(int)
            w0, w1 = windows[key]
            fin = np.asarray(da.isel(year=np.where((yrs >= w0) & (yrs <= w1))[0])
                             .mean(("member", "year")).values, float)
            b = (yrs >= BASELINE[0]) & (yrs <= BASELINE[1])
            base = (np.asarray(da.isel(year=np.where(b)[0]).mean(("member", "year")).values, float)
                    if b.any() else None)
            n = int(da.sizes["member"])
        out[key] = dict(final=fin, base=base, n=n)
        print(f"  [{run} ep{epoch}] {key}: {n} members, window {w0}-{w1}")
    return apply_baselines(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--outdir", default="plots/arm_maps")
    ap.add_argument("--arm", action="append", default=None, metavar="RUN:EPOCH",
                    help="override the arms, e.g. --arm run_minmax_co2fix:470 (repeatable)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for var in args.var:
        z = np.load(f"plots/ensmean_maps/cache_{var}_10y.npz", allow_pickle=True)
        emu = apply_baselines(to_celsius(z["emu"].item(), var, "paper"))
        ref = apply_baselines(to_celsius(z["ref"].item(), var, "CESM2"))
        windows = {k: ref[k]["window"] for k in SCENARIOS}
        lat, lon = ref["hist"]["lat"], ref["hist"]["lon"]
        unit = META[var]["unit"]

        anom = lambda d, k: d[k]["final"] - d[k]["base"]
        rows = [("v1 paper (clipped) ep860", {k: anom(emu, k) for k in SCENARIOS})]
        arms = ARMS if not args.arm else [
            (a.split(":")[0].replace("run_", "").replace("_co2fix", ""), a.split(":")[0], int(a.split(":")[1]))
            for a in args.arm]
        for label, run, ep in arms:
            m = arm_maps(var, run, ep, windows)
            rows.append((f"{label} ep{ep}", {k: anom(m, k) for k in SCENARIOS}))

        cesm = {k: anom(ref, k) for k in SCENARIOS}
        diffs = [np.ravel(r[k] - cesm[k]) for _, r in rows for k in SCENARIOS]
        dmax = float(np.nanpercentile(np.abs(np.concatenate(diffs)), 99))
        tmax = float(np.nanpercentile(np.abs(np.concatenate([np.ravel(v) for v in cesm.values()])), 99))

        nrow = 1 + len(rows)
        fig = plt.figure(figsize=(3.6 * len(SCENARIOS), 2.25 * nrow + 0.8), constrained_layout=True)
        axes = make_axes(fig, nrow, len(SCENARIOS))
        k_ = 0
        print(f"\n{var}: pattern r / RMSE ({unit}) of each arm's anomaly against CESM2")
        for j, (key, (label, _)) in enumerate(SCENARIOS.items()):
            ax = axes[0][j]
            im_top = draw_map(ax, to_pm180(cesm[key], lon)[0], cmap=META[var]["anom_cmap"],
                              vmin=-tmax, vmax=tmax)
            w = windows[key]
            ax.set_title(f"{label}\n{w[0]}-{w[1]} vs 1850-1900", fontsize=9)
            if j == 0:
                ax.text(-0.04, 0.5, "CESM2\nheld-out", transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=9)
            panel_label(ax, k_); k_ += 1
        for i, (name, fields) in enumerate(rows, start=1):
            line = []
            for j, key in enumerate(SCENARIOS):
                ax = axes[i][j]
                d = fields[key] - cesm[key]
                im_d = draw_map(ax, to_pm180(d, lon)[0], cmap=META[var]["dcmap"], vmin=-dmax, vmax=dmax)
                st = weighted_stats(fields[key], cesm[key], lat)
                ax.text(0.5, -0.04, f"r {st['r']:.3f}   RMSE {st['rmse']:.2f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7.5)
                if j == 0:
                    ax.text(-0.04, 0.5, name.replace(" (", "\n("), transform=ax.transAxes, rotation=90,
                            va="center", ha="right", fontsize=8)
                panel_label(ax, k_); k_ += 1
                line.append(f"{key} {st['r']:.3f}/{st['rmse']:.3f}")
            print(f"  {name:38s} " + "   ".join(line))
        fig.colorbar(im_top, ax=list(axes[0].ravel()), shrink=0.8, label=f"CESM2 anomaly ({unit})")
        fig.colorbar(im_d, ax=list(axes[1:].ravel()), shrink=0.5, label=f"arm minus CESM2 ({unit})")
        fig.suptitle(f"{var}: final-decade anomaly, CESM2 and each conditioning arm's difference "
                     f"(shared difference scale; r and RMSE of the arm's anomaly vs CESM2)", fontsize=11)
        out = os.path.join(args.outdir, f"arm_comparison_{var}")
        for ext in (".png", ".pdf"):
            fig.savefig(out + ext, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}.png/.pdf")


if __name__ == "__main__":
    main()
