#!/usr/bin/env python3
"""
================================================================================
 FIGURE 21 — JACOBIAN ROWS: WHICH CONDITIONING CELLS MOVE N, AND WHERE
================================================================================

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig19_jacobian_maps.py

WHAT IS PLOTTED, PRECISELY
--------------------------
Not the Jacobian. The full Jacobian of this model is 55296 x 55296 — every
output cell against every input cell — and is neither computable nor readable.
What is plotted is one ROW per output region:

    d N(region) / d cond(x)          for x over the whole 192x288 input grid

That row is what `integrated_gradients` already averages along the IG path, so
it comes for free. Two versions are written because they answer different
questions:

  GRADIENT     d N / d cond(x). "If this conditioning cell were nudged, how much
               would N over that region move?" Full resolution, and it is
               SENSITIVITY — it says nothing about whether the cell actually
               changed between 1850 and 2040.

  ATTRIBUTION  (cond - baseline) * gradient. The gradient weighted by the
               displacement actually travelled. This is the IG quantity, it sums
               to N exactly, and it is what "this contributed that much" means.

A cell can carry a large gradient and near-zero attribution — the model is
sensitive there but nothing happened — and that difference is usually the
interesting part.

THE CEILING THAT STILL APPLIES
------------------------------
These maps are full-resolution, but the perturbations the emission pipeline can
actually produce are not. Conditioning is PCA-denoised to five modes per aerosol
species, and a single-region emission change survives that projection at 9.4%
(SUL) or 4.6% (CO2) of its variance. So a bright cell here does NOT license
"changing emissions at this cell changes N" — the model would never receive such
a change. Read these as the model's internal sensitivity structure, and make
quantitative claims in mode space (figure 18).
"""

# =============================================================================
#  SETTINGS
# =============================================================================

RESULT = "plots/ig_nonlinear_2040_maps.npz"
OUT = "plots/xai19/xai19_{kind}.png"     # the .pdf sibling is written alongside

# Output regions to show as rows. The full run covers eight; these three carry
# the structure — the global mean, the strongest region, and the one that
# reverses sign.
SHOW_REGIONS = ["Global", "Arctic", "N. Atlantic"]

KINDS = {"gradient": ("grad_maps", "dN / d cond  (sensitivity)"),
         "attribution": ("attr_maps", "(cond - baseline) x dN/dcond  (IG)")}

# =============================================================================

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
except ImportError:
    sys.exit("[error] cartopy missing — use the plotting env")

if not os.path.exists(RESULT):
    sys.exit(f"[error] {RESULT} not found — rerun scripts/ig_nonlinear.py on "
             f"LUMI (it now saves grad_maps/attr_maps) and scp it into plots/")

store = np.load(RESULT, allow_pickle=True)
regions = [str(r) for r in store["regions"]]
cond_vars = [str(v) for v in store["cond_vars"]]
lat, lon = store["lat"], store["lon"]
year = int(store["year"])
n_values = store["N"]

rows = [regions.index(r) for r in SHOW_REGIONS if r in regions]
if not rows:
    sys.exit(f"[error] none of {SHOW_REGIONS} present in {regions}")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})

for kind, (key, description) in KINDS.items():
    maps = store[key]                              # (region, channel, lat, lon)
    fig = plt.figure(figsize=(4.6 * len(cond_vars), 2.7 * len(rows) + 1.1))
    grid = fig.add_gridspec(len(rows), len(cond_vars), hspace=0.30, wspace=0.06)
    projection = ccrs.Robinson(central_longitude=0)

    for r, region_index in enumerate(rows):
        # ONE SCALE PER ROW, across the three species. The channels are on the
        # same normalised footing, so a shared scale shows which species the
        # model is actually leaning on — rescaling each panel to fill its own
        # range would hide exactly that.
        row_max = float(np.nanpercentile(np.abs(maps[region_index]), 99.5))
        axes_in_row = []
        for c, channel in enumerate(cond_vars):
            ax = fig.add_subplot(grid[r, c], projection=projection)
            image = ax.pcolormesh(lon, lat, maps[region_index, c],
                                  cmap="RdBu_r", vmin=-row_max, vmax=row_max,
                                  shading="auto", transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.35, color="0.25")
            ax.set_global()
            if r == 0:
                ax.set_title(channel, fontsize=11, pad=6)
            if c == 0:
                ax.text(-0.05, 0.5,
                        f"{regions[region_index]}\nN = {n_values[region_index]:+.4f}",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=9.5)
            axes_in_row.append(ax)
        fig.colorbar(image, ax=axes_in_row, orientation="vertical",
                     fraction=0.016, pad=0.012, extend="both")

    fig.suptitle(
        f"Jacobian rows for N = ALL $-$ GHG $-$ AAER, year {year} — {description}\n"
        f"one row per OUTPUT region, showing sensitivity to every INPUT "
        f"conditioning cell; not the full Jacobian",
        fontsize=10.5, y=0.99)

    path = OUT.format(kind=kind)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for p in (path, os.path.splitext(path)[0] + ".pdf"):
        fig.savefig(p, bbox_inches="tight")
        print(f"[xai19] wrote {p}")
    plt.close(fig)

print("\n[xai19] share of the attribution magnitude carried by each channel:")
area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon)))
for region_index in rows:
    parts = [float(np.average(np.abs(store["attr_maps"][region_index, c]),
                              weights=area)) for c in range(len(cond_vars))]
    total = sum(parts) or 1.0
    print(f"  {regions[region_index]:12s} " + ",  ".join(
        f"{v} {100 * p / total:.0f}%" for v, p in zip(cond_vars, parts)))
