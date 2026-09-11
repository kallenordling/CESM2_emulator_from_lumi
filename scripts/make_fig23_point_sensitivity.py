#!/usr/bin/env python3
"""
================================================================================
 FIGURE 23 — HOW EACH EMISSION CELL AFFECTS THE RESPONSE AT ONE OUTPUT CELL
================================================================================

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig21_point_sensitivity.py \
        --result plots/ig_point_55N30W.npz

Renders the single-output-cell run of scripts/ig_nonlinear.py:

    d Y(one output cell) / d E(x)      for x over the whole 192x288 input grid

One row per species. Two quantities are shown because they are different
questions and get confused constantly:

  TEMPERATURE CHANGE   attribution of dALL, the response at that cell to the
                       1850->2040 forcing. The straightforward question.
  NONLINEAR TERM       attribution of N = dALL - dGHG - dAAER. Only the part
                       that the single-forcing runs fail to explain.

READ THE SCALE, NOT THE PATTERN ALONE
-------------------------------------
A single output cell is a far noisier target than a regional mean: the scalar is
one number rather than an average over thousands of cells. The per-cell
attribution already shows grid-scale dipoles at regional resolution, so at a
point they are stronger still. Individual input cells are NOT robust here; what
is robust is the large-scale organisation and the totals per species, both of
which are printed.
"""

import argparse
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result", default="plots/ig_point_55N30W.npz")
parser.add_argument("--out", default="plots/fig23/fig23.png")
parser.add_argument("--index", type=int, default=0)
args = parser.parse_args()

store = np.load(args.result, allow_pickle=True)
cond_vars = [str(v) for v in store["cond_vars"]]
lat, lon = store["lat"], store["lon"]
label = str(store["regions"][args.index])
# `years` is the span actually averaged; `year` is the single-year fallback for
# files written before multi-year averaging existed. Using the wrong one labels
# a 21-year mean as a single year.
span = str(store["years"]) if "years" in store.files else str(int(store["year"]))

QUANTITIES = [
    ("Temperature change  ($\\Delta$ALL)", store["attr_all"][args.index]),
    ("Nonlinear term  (N = ALL $-$ GHG $-$ AAER)", store["attr_maps"][args.index]),
]

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})
fig = plt.figure(figsize=(5.0 * len(QUANTITIES) + 0.6, 2.7 * len(cond_vars) + 1.2))
grid = fig.add_gridspec(len(cond_vars), len(QUANTITIES), hspace=0.26, wspace=0.06)
projection = ccrs.Robinson(central_longitude=0)

# Marker for the OUTPUT cell — the map is about input cells, so the point the
# question is asked at has to be visible or the figure is ambiguous.
parts = label.replace("point ", "").replace("N", "").replace("E", "").split()
point_lat, point_lon = (float(parts[0]), float(parts[1])) if len(parts) == 2 \
    else (None, None)

print(f"[fig23] output cell: {label}, years {span}")
for column, (title, maps) in enumerate(QUANTITIES):
    scale = float(np.nanpercentile(np.abs(maps), 99.5))
    total = maps.sum()
    parts_sum = sum(abs(maps[r].sum()) for r in range(len(cond_vars)))
    note = ("  (species nearly cancel — shares suppressed)"
            if abs(total) <= 0.25 * parts_sum else "")
    print(f"  {title.split('(')[0].strip():22s} total {total:+.5f}{note}")
    column_axes = []            # collected explicitly: colour-bar axes have no
                                # subplotspec, so filtering fig.axes fails
    for row, species in enumerate(cond_vars):
        ax = fig.add_subplot(grid[row, column], projection=projection)
        column_axes.append(ax)
        image = ax.pcolormesh(lon, lat, maps[row], cmap="RdBu_r",
                              vmin=-scale, vmax=scale, shading="auto",
                              transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.3, color="0.3")
        ax.set_global()
        if point_lat is not None:
            ax.plot(point_lon, point_lat, marker="*", markersize=13,
                    markerfacecolor="#2E7D6B", markeredgecolor="black",
                    markeredgewidth=0.7, transform=ccrs.PlateCarree(), zorder=6)
        # A share is only meaningful when the total is large compared with the
        # parts. Here the temperature-change column nearly cancels between CO2
        # and SUL, so "% of total" explodes to nonsense (+1065%) and is
        # suppressed rather than printed.
        parts = sum(abs(maps[r].sum()) for r in range(len(cond_vars)))
        if abs(total) > 0.25 * parts:
            annotation = f" ({100 * maps[row].sum() / total:+.0f}% of total)"
        else:
            annotation = ""
        ax.set_title(f"{species} — {maps[row].sum():+.4f}{annotation}",
                     fontsize=9.5, loc="left", pad=3)
        if column == 0:
            ax.text(-0.04, 0.5, species, transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=11)
        if row == 0:
            ax.text(0.5, 1.32, title, transform=ax.transAxes, ha="center",
                    fontsize=11)
        print(f"      {species:4s} {maps[row].sum():+.5f}  "
              f"(|.| {np.abs(maps[row]).sum():.5f})")
    fig.colorbar(image, ax=column_axes, orientation="horizontal",
                 fraction=0.045, pad=0.04, aspect=32, extend="both",
                 label="contribution per input cell")

fig.suptitle(
    f"Sensitivity of the response at {label} to emissions at every input cell, "
    f"{span}\n"
    f"star = the output cell; colour = how much that INPUT cell contributes",
    fontsize=10.5, y=1.0)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
for p in (args.out, os.path.splitext(args.out)[0] + ".pdf"):
    fig.savefig(p, bbox_inches="tight")
    print(f"[fig23] wrote {p}")
