#!/usr/bin/env python3
"""Conditioning maps across smoothing sigmas, as the model would receive them.

Runs the REAL pipeline from data/climate_dataset.py for each sigma --
gaussian smooth, then min/max normalise with the anchors refit on the smoothed
hist+ssp370 fields -- rather than smoothing a already-normalised field. The
anchors move with sigma (smoothing cuts the peak, so the ceiling drops), which
is the whole reason the sweep is not just a blur comparison: at sigma 0 the
single hottest raw cell sets +1 and the bulk collapses onto -1, and that
collapse is what sigma buys back.

Two diagnostics accompany the maps, both measured on the normalised field:

  roughness   mean |x - 5x5 box mean|, cos-lat weighted. The grid-scale
              texture that showed up as speckle in the emulator output.
  stranded    share of EMITTING cells (raw value > 0) landing within 0.06 of
              -1. The cost of a min/max map: with the ceiling at the single
              most extreme cell, the bulk of real emitters is crushed onto the
              floor, and smoothing is what makes that affordable.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_cond_sigma_sweep.py
"""
import argparse
import os
import sys

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Coastlines + country borders. Optional: a cartopy/shapely geometry failure
# must not cost the whole figure -- the eval on LUMI hits exactly that
# (GEOSException: Points of LinearRing do not form a closed linestring), so
# every call here is guarded and the maps degrade to plain axes.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:                                            # noqa: BLE001
    HAVE_CARTOPY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Light background: a dark-background map hides the near -1 bulk, which is
# exactly the quantity this figure is about.
CMAP = "YlOrRd"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sigmas", default="0,0.5,1,1.5,2,2.5,3,3.5",
                help="comma-separated; sigma 4 is the arm already trained")
ap.add_argument("--data", default=os.path.expanduser("~/mnt/lumi_sc2/emulator_data"))
ap.add_argument("--scenario", default="ssp370")
ap.add_argument("--year", type=int, default=2050)
ap.add_argument("--vars", default="CO2,SUL,BC")
ap.add_argument("--outdir", default="plots/cond_sigma_sweep")
args = ap.parse_args()

SIGMAS = [float(s) for s in args.sigmas.split(",")]
VARS = [v.strip() for v in args.vars.split(",")]
os.makedirs(args.outdir, exist_ok=True)

import data.climate_dataset as cd

# Point the module at the locally mounted copies and fit anchors the way the
# training arm does.
cd.EMISSIONS_PATHS = [f"{args.data}/emissions_{k}_only_timefixed_bc_co2fix.nc"
                      for k in ("hist", "ssp370", "aaer", "ghg")]
cd.set_anchor_scenarios("hist_ssp370")

path = f"{args.data}/emissions_{args.scenario}_only_timefixed_bc_co2fix.nc"
raw = xr.open_dataset(path)
tdim = "time" if "time" in raw.dims else "year"
years = np.asarray(raw[tdim].values).astype(int)
yi = int(np.argmin(np.abs(years - args.year)))
lat = raw["lat"].values
lon = raw["lon"].values
stacked = (raw[VARS].to_stacked_array("var", sample_dims=[tdim, "lon", "lat"])
           .transpose("var", tdim, "lat", "lon"))
base = torch.tensor(stacked.values, dtype=torch.float32)
raw.close()
print(f"[sweep] {args.scenario} {years[yi]}  tensor {tuple(base.shape)}")

wgt = np.cos(np.deg2rad(lat))[:, None]


def roughness(field):
    """Mean |x - 5x5 box mean|, cos-lat weighted; lon wraps, lat reflects."""
    from scipy.ndimage import uniform_filter
    p = 2
    padded = np.pad(field, ((0, 0), (p, p)), mode="wrap")
    padded = np.pad(padded, ((p, p), (0, 0)), mode="reflect")
    box = uniform_filter(padded, size=5, mode="nearest")[p:-p, p:-p]
    return float(np.average(np.abs(field - box), weights=np.broadcast_to(wgt, field.shape)))


rows = []
panels = {}
for sig in SIGMAS:
    sigs = [sig] * len(VARS)
    # Each sigma gets its OWN anchors: clear the cache or every later sigma
    # silently reuses the first one's ceiling.
    cd.set_processed_minmax_override(None)
    cd._PROCESSED_MINMAX_CACHE = None
    sm = torch.from_numpy(cd.smooth_cond_spatial(base.numpy(), sigs, "gaussian", VARS))
    norm = cd.normalize_tensor_cond(sm, VARS, sigs)
    anchors = cd.get_processed_minmax_state()
    for vi, var in enumerate(VARS):
        f = norm[vi, yi].numpy()
        panels[(sig, var)] = f
        emitting = base[vi, yi].numpy() > 0
        stranded = (float(np.mean(f[emitting] < -1 + 0.06) * 100)
                    if emitting.any() else np.nan)
        # Absolute roughness is NOT comparable across sigma: the min/max
        # anchor falls as sigma rises (SUL 1.15e-3 -> 3.2e-5 over this sweep),
        # so the normalised field is amplified and its absolute texture grows
        # even as smoothing removes structure. Dividing by the field's own
        # spread makes it scale-free and so actually a function of smoothing.
        rough = roughness(f)
        spread = float(np.sqrt(np.average((f - np.average(f, weights=np.broadcast_to(wgt, f.shape))) ** 2,
                                          weights=np.broadcast_to(wgt, f.shape))))
        rows.append(dict(sigma=sig, var=var, hi=anchors[var][1],
                         roughness=rough, spread=spread,
                         rough_rel=rough / spread if spread > 0 else np.nan,
                         stranded=stranded,
                         median_emitting=(float(np.median(f[emitting]))
                                          if emitting.any() else np.nan)))
    print(f"[sweep] sigma={sig} done")

# ── maps: one figure per variable, one panel per sigma ──────────────────────
for var in VARS:
    n = len(SIGMAS)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    subplot_kw = {"projection": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.3 * nrow),
                             constrained_layout=True, subplot_kw=subplot_kw)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for k, sig in enumerate(SIGMAS):
        ax = axes[k]
        # Shared scale: these are all normalised to [-1, 1] by construction,
        # so a per-panel scale would hide the very collapse being measured.
        kw = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
        im = ax.pcolormesh(lon, lat, panels[(sig, var)], cmap=CMAP,
                           vmin=-1, vmax=1, shading="auto", **kw)
        if HAVE_CARTOPY:
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.22,
                               edgecolor="0.35")
                ax.coastlines(linewidth=0.35, color="0.15")
                ax.set_global()
            except Exception as exc:                          # noqa: BLE001
                print(f"[sweep] border draw failed for {var} sigma={sig}: {exc}")
        r = next(x for x in rows if x["sigma"] == sig and x["var"] == var)
        # rough/spread, NOT absolute roughness: the anchor shrinks as sigma
        # grows, so absolute texture rises even where smoothing removes
        # structure. Printing the confounded number on the figure invites
        # exactly the misreading the caption warns about.
        ax.set_title(f"$\\sigma$={sig:g}   rough/spread {r['rough_rel']:.3f}   "
                     f"stranded {r['stranded']:.0f}%", fontsize=9)
        if not HAVE_CARTOPY:
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label=f"{var} (normalised)")
    fig.suptitle(f"{var} conditioning as the model receives it — "
                 f"{args.scenario} {years[yi]}, smooth then min/max normalise",
                 fontsize=12)
    out = os.path.join(args.outdir, f"cond_sigma_sweep_{var}")
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png/.pdf")

# ── the two diagnostics against sigma ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for var in VARS:
    sel = [r for r in rows if r["var"] == var]
    axes[0].plot([r["sigma"] for r in sel], [r["rough_rel"] for r in sel],
                 "o-", label=var)
    axes[1].plot([r["sigma"] for r in sel], [r["stranded"] for r in sel],
                 "o-", label=var)
axes[0].set_xlabel("smoothing $\\sigma$ (gridpoints)")
axes[0].set_ylabel("roughness / field spread")
axes[0].set_title("texture, scale-free\n(absolute roughness is confounded by the anchor)")
axes[1].set_xlabel("smoothing $\\sigma$ (gridpoints)")
axes[1].set_ylabel("% of emitting cells within 0.06 of $-1$")
axes[1].set_title("bulk crushed onto the floor")
for a in axes:
    a.grid(alpha=0.3); a.legend(fontsize=8)
fig.suptitle("The sigma trade-off: smoothing cuts texture, but the min/max "
             "ceiling still strands the bulk", fontsize=11)
out = os.path.join(args.outdir, "cond_sigma_tradeoff")
for ext in (".png", ".pdf"):
    fig.savefig(out + ext, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}.png/.pdf")

import csv
csv_path = os.path.join(args.outdir, "cond_sigma_sweep.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {csv_path}")
print(f"\n{'sigma':>6} {'var':>5} {'hi anchor':>12} {'spread':>9} "
      f"{'rough/spread':>13} {'stranded%':>10} {'median emit':>12}")
for r in rows:
    print(f"{r['sigma']:6g} {r['var']:>5} {r['hi']:12.4e} {r['spread']:9.4f} "
          f"{r['rough_rel']:13.4f} {r['stranded']:10.1f} {r['median_emitting']:12.3f}")
