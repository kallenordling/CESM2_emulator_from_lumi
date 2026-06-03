#!/usr/bin/env python3
"""Overlay global-mean ΔT vs cumulative CO2 for all CO2-varying scenarios,
model (solid/filled) vs CESM2 (dashed/open), with per-scenario linear fits.

Reveals whether the model's ΔT-vs-cumCO2 response is convex (over-steep at low
forcing → hist/ssp126 over-warming) and how the per-scenario slopes compare to
CESM2.  Reads an eval output dir (global_mean_anomaly.csv) + the training cond
files for the cumulative-CO2 axis.

Usage:
    python plot_tcre_curve.py <eval_output_dir>
e.g. python plot_tcre_curve.py /mnt/lumi_sc2/eval_output/run_slope-tcre/best_ep0100
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMIS = "/mnt/lumi_sc2/emulator_data"
COND = {
    "hist":   "emissions_hist_only_timefixed.nc",
    "ssp370": "emissions_ssp370_only_timefixed.nc",
    "ssp126": "emissions_ssp126_only_timefixed.nc",
    "ghg":    "emissions_ghg_only_timefixed.nc",
}
COL = {"hist": "#1f77b4", "ssp370": "#d62728", "ssp126": "#9467bd", "ghg": "#2ca02c"}


def cumco2(fn):
    ds = xr.open_dataset(f"{EMIS}/{fn}")
    td = "time" if "time" in ds.dims else "year"
    sp = [d for d in ds["CO2"].dims if d != td]
    s = ds["CO2"].sum(dim=sp)            # total cumulative CO2 per step
    yrs = ds[td].values
    if hasattr(yrs[0], "year") or "datetime" in str(type(yrs[0])).lower():
        yrs = np.array([int(str(y)[:4]) for y in yrs])
    else:
        yrs = np.asarray(yrs, int)
    return dict(zip(yrs, s.values.astype(float)))


def main(eval_dir):
    csv = os.path.join(eval_dir, "global_mean_anomaly.csv")
    if not os.path.exists(csv):
        sys.exit(f"no global_mean_anomaly.csv in {eval_dir}")
    m = re.search(r"ep(\d+)", eval_dir)
    ep = m.group(1) if m else "?"
    df = pd.read_csv(csv)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    summary = []
    for sc, fn in COND.items():
        try:
            c = cumco2(fn)
        except Exception as e:
            print(f"[skip {sc}] {e}")
            continue
        sub = df[df.experiment == sc].copy()
        sub["cum"] = sub.year.map(c)
        sub = sub.dropna(subset=["cum"]).sort_values("cum")
        if len(sub) < 3:
            continue
        x = sub.cum.values
        ax.scatter(x, sub.model_anom_degC, s=10, color=COL[sc], alpha=.45)
        ax.scatter(x, sub.cesm_anom_degC, s=10, facecolors="none",
                   edgecolors=COL[sc], alpha=.45)
        mm = np.polyfit(x, sub.model_anom_degC, 1)
        cc = np.polyfit(x, sub.cesm_anom_degC, 1)
        xr_ = np.linspace(x.min(), x.max(), 50)
        ax.plot(xr_, np.polyval(mm, xr_), color=COL[sc], lw=2,
                label=f"{sc} MODEL  slope={mm[0]:.2e}")
        ax.plot(xr_, np.polyval(cc, xr_), color=COL[sc], lw=1.4, ls="--",
                label=f"{sc} CESM2  slope={cc[0]:.2e}")
        ratio = mm[0] / cc[0] if cc[0] != 0 else float("nan")
        summary.append(f"  {sc:7s} model={mm[0]:.2e} CESM2={cc[0]:.2e} ratio={ratio:.2f}")

    ax.set_xlabel("cumulative CO2 (sum over gridpoints, scenario units)")
    ax.set_ylabel("global-mean ΔT vs 1850-1900 (°C)")
    ax.set_title(f"ΔT vs cumulative CO2 — model (solid) vs CESM2 (dashed)\n"
                 f"run_slope-tcre best_ep{ep}")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = f"tcre_curve_ep{ep}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print("per-scenario MODEL/CESM2 slope ratios:")
    print("\n".join(summary))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/mnt/lumi_sc2/eval_output/run_slope-tcre/best_ep0030")
