#!/usr/bin/env python3
"""ABSOLUTE temperature (and precip) figures — what the emulator actually outputs.

Every existing eval figure is an ANOMALY re 1850-1900: global_mean_anomaly.png
and anomaly_maps_<scenario>.png. Anomalies hide whether the emulator's absolute
climate is right at all — a model can have a perfect warming curve on top of a
badly wrong mean state, and the anomaly plots would look fine.

Nothing needs re-running: eval_aero.py already writes the absolute fields
alongside the anomalies —

    <VAR>_model_mean        (year, lat, lon)   emulator ensemble mean
    <VAR>_model_gmean_mean  (year,)            already cos(lat)-weighted
    <VAR>_cesm_mean         (cesm_year, ...)   CESM2 reference, absolute
    <VAR>_cesm_gmean_mean   (cesm_year,)

— so this only plots them.

    absolute_series.png   global-mean absolute, emulator vs CESM2, per scenario
    absolute_maps.png     time-mean absolute maps: emulator | CESM2 | difference

Run it on LUMI, where the eval output lives (the files are ~800 MB each):

    python scripts/plot_absolute.py \
        --eval-dir /scratch/project_462001112/eval_output/run_mseyb_BCprect/best_ep0490 \
        --var TREFHT --out-dir plots/
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

UNITS = {"TREFHT": "°C", "PRECT": "mm/day"}
CMAP = {"TREFHT": "RdBu_r", "PRECT": "viridis"}


def scenario_period(d, n=20):
    """Last n years this scenario actually covers."""
    last = int(d.year.max())
    return last - n + 1, last


def area_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def load(eval_dir, var):
    """{scenario: Dataset} for every <var>_<scenario>.nc in the directory."""
    out = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, f"{var}_*.nc"))):
        scen = os.path.basename(f)[len(var) + 1:-3]
        out[scen] = xr.open_dataset(f)
    return out


def fig_series(runs, var, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (scen, d) in enumerate(runs.items()):
        c = colors[i % 10]
        mk = f"{var}_model_gmean_mean"
        if mk in d:
            ax.plot(d.year, d[mk], color=c, lw=1.6, label=f"{scen} — emulator")
        ck = f"{var}_cesm_gmean_mean"
        if ck in d:
            ax.plot(d.cesm_year, d[ck], color=c, lw=1.2, ls="--", alpha=0.8,
                    label=f"{scen} — CESM2")
    ax.set_xlabel("year")
    ax.set_ylabel(f"global-mean {var} [{UNITS.get(var, '')}] — ABSOLUTE")
    ax.set_title(f"Absolute global-mean {var}: emulator (solid) vs CESM2 (dashed)\n"
                 "cos(lat)-weighted; NOT an anomaly")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[WROTE] {path}")


def fig_maps(runs, var, path, period):
    scens = list(runs)
    fig, axes = plt.subplots(len(scens), 3, figsize=(15, 3.1 * len(scens)),
                             squeeze=False)
    for r, scen in enumerate(scens):
        d = runs[scen]
        mk, ck = f"{var}_model_mean", f"{var}_cesm_mean"
        # PER SCENARIO unless --period was given: hist ends in 2014 while the
        # SSPs run to 2100, so one window across all of them leaves the
        # historical rows empty (an all-NaN slice, not an error).
        y0, y1 = period or scenario_period(d)
        m = d[mk].sel(year=slice(y0, y1)).mean("year") if mk in d else None
        c = (d[ck].sel(cesm_year=slice(y0, y1)).mean("cesm_year")
             if ck in d else None)
        if m is None:
            for ax in axes[r]:
                ax.axis("off")
            continue

        w = area_weights(d.lat.values)[:, None]
        lo = float(min(m.min(), c.min() if c is not None else m.min()))
        hi = float(max(m.max(), c.max() if c is not None else m.max()))
        panels = [("emulator", m, CMAP.get(var, "viridis"), lo, hi)]
        if c is not None:
            diff = m - c
            lim = float(np.nanpercentile(np.abs(diff.values), 99))
            panels += [("CESM2", c, CMAP.get(var, "viridis"), lo, hi),
                       ("emulator − CESM2", diff, "RdBu_r", -lim, lim)]

        for col, (title, field, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r][col]
            ax.set_xlabel(f"{y0}-{y1}", fontsize=7)
            im = ax.imshow(field.values, origin="lower", aspect="auto",
                           extent=[0, 360, -90, 90], cmap=cmap,
                           vmin=vmin, vmax=vmax)
            gm = float((field.values * w).mean())
            ax.set_title(f"{scen} — {title}   (gmean {gm:.2f})", fontsize=9)
            if col == 0:
                ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, shrink=0.85)
        for col in range(len(panels), 3):
            axes[r][col].axis("off")

    span = (f"{period[0]}-{period[1]}" if period
            else "last 20 years of each scenario")
    fig.suptitle(f"Absolute {var} [{UNITS.get(var, '')}], {span}", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[WROTE] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--var", default="TREFHT")
    ap.add_argument("--period", nargs=2, type=int, default=None,
                    metavar=("Y0", "Y1"),
                    help="map averaging window (default: last 20 years present)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    runs = load(args.eval_dir, args.var)
    if not runs:
        sys.exit(f"[FATAL] no {args.var}_*.nc in {args.eval_dir}")
    print(f"[LOAD] {len(runs)} scenario(s): {', '.join(runs)}")

    out_dir = args.out_dir or args.eval_dir
    os.makedirs(out_dir, exist_ok=True)

    period = tuple(args.period) if args.period else None
    if period:
        print(f"[PERIOD] maps averaged over {period[0]}-{period[1]}")
    else:
        for scen, d in runs.items():
            print(f"[PERIOD] {scen}: {scenario_period(d)[0]}-{scenario_period(d)[1]}")

    fig_series(runs, args.var,
               os.path.join(out_dir, f"absolute_series_{args.var}.png"))
    fig_maps(runs, args.var,
             os.path.join(out_dir, f"absolute_maps_{args.var}.png"), period)
    return 0


if __name__ == "__main__":
    sys.exit(main())
