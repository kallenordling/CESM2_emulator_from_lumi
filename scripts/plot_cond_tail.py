#!/usr/bin/env python3
"""Where the normalised conditioning actually exceeds [-1, +1].

The stage maps hide this: with a percentile anchor ~1% (CO2) to ~5% (SUL/BC) of
cells sit above +1 BY CONSTRUCTION, which is a sprinkle of cells invisible at
global scale, and a p99.5 colour cap paints them the same shade as +1. Here the
distribution is on a log count axis, and the map shows ONLY the exceeding cells.
"""
import os, sys
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ensemble_mean_maps import make_axes, draw_map, to_pm180, panel_label

DATA = os.path.expanduser("~/mnt/lumi_sc2/emulator_data")
SPECIES, SIGMA = ["CO2", "SUL", "BC"], 4.0
PCTL = {"CO2": (1, 99), "SUL": (5, 95), "BC": (5, 95)}

def smooth(a, s):
    a = gaussian_filter1d(a, sigma=s, axis=-1, mode="wrap")
    return gaussian_filter1d(a, sigma=s, axis=-2, mode="reflect")

raw = {}
for scen in ("hist", "ssp370"):
    with xr.open_dataset(f"{DATA}/emissions_{scen}_only_timefixed_bc_co2fix.nc") as ds:
        raw[scen] = {v: smooth(np.asarray(ds[v].values, float), SIGMA) for v in SPECIES}
        lat, lon = ds["lat"].values, ds["lon"].values

fig = plt.figure(figsize=(13.5, 3.1 * len(SPECIES) + 0.7), constrained_layout=True)
gs = fig.add_gridspec(len(SPECIES), 2, width_ratios=[1, 1.35])
k = 0
import cartopy.crs as ccrs
for i, v in enumerate(SPECIES):
    pooled = np.concatenate([raw[s][v].ravel() for s in raw])
    plo, phi = PCTL[v]
    lo, hi = np.percentile(pooled, plo), np.percentile(pooled, phi)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    z = {s: (raw[s][v] - mid) / half for s in raw}
    allz = np.concatenate([a.ravel() for a in z.values()])

    ax = fig.add_subplot(gs[i, 0])
    ax.hist(allz, bins=200, color="tab:blue")
    ax.set_yscale("log")
    ax.axvline(1, color="crimson", ls=":", lw=1.2)
    ax.axvline(-1, color="crimson", ls=":", lw=1.2)
    frac = 100 * (allz > 1).mean()
    ax.set_title(f"{v}: {frac:.2f}% of cell-years above +1, max {allz.max():.1f}",
                 fontsize=9)
    ax.set_xlabel("model units"); ax.set_ylabel("count (log)")
    panel_label(ax, k); k += 1

    axm = fig.add_subplot(gs[i, 1], projection=ccrs.Robinson())
    peak = z["ssp370"][-1]                      # final year of ssp370
    masked = np.where(peak > 1, peak, np.nan)
    d, _ = to_pm180(masked, lon)
    im = draw_map(axm, d, cmap="autumn_r", vmin=1,
                  vmax=float(np.nanpercentile(peak[peak > 1], 99)) if (peak > 1).any() else 2)
    fig.colorbar(im, ax=axm, shrink=0.75, label="model units (>1 only)")
    axm.set_title(f"{v}: cells ABOVE +1, ssp370 final year "
                  f"({100*(peak>1).mean():.2f}% of map)", fontsize=9)
    panel_label(axm, k); k += 1

fig.suptitle("The tail the stage maps hide: percentile anchors put ~1% (CO2) to ~5% "
             "(SUL/BC) of cells above +1 by construction\n"
             f"smoothed sigma={SIGMA:g}, hist+ssp370 anchors, v1_noclip (NO clip)",
             fontsize=11)
os.makedirs("plots/cond_stages", exist_ok=True)
for ext in (".png", ".pdf"):
    fig.savefig("plots/cond_stages/cond_tail" + ext, dpi=150, bbox_inches="tight")
print("wrote plots/cond_stages/cond_tail.png/.pdf")
