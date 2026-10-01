#!/usr/bin/env python3
"""Paper figure: the conditioning pipeline, raw to what the network receives.

Columns: raw inventory -> gaussian smoothed (sigma 4, RAW units) -> normalised
under each candidate transform. Rows: CO2, SUL, BC. Both transforms anchor on
the MIN/MAX of the smoothed hist+ssp370 field, so both are bounded to [-1, +1]
with nothing clipped and nothing pinned; they differ only in the shape of the
map between the anchors.

    mmlin     y = 2u - 1                                (linear)
    mmasinh   y = 2*asinh(u/s)/asinh(1/s) - 1, s=0.001  (concave stretch)

Each panel keeps its own colourbar: the stages are in different units, and a
shared scale across transforms would hide the very difference the figure is
about. The `max` annotation is the panel's true maximum -- a percentile cap
would hide the tail, which is how an earlier version of this figure misled.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_cond_pipeline_paper.py
"""
import argparse, os, sys
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ensemble_mean_maps import make_axes, draw_map, to_pm180, panel_label, HAVE_CARTOPY

DATA = os.path.expanduser("~/mnt/lumi_sc2/emulator_data")
SPECIES = ["CO2", "SUL", "BC"]
RAW_UNIT = {"CO2": "kg m$^{-2}$ s$^{-1}$ (cumulative)",
            "SUL": "kg m$^{-2}$ s$^{-1}$", "BC": "kg m$^{-2}$ s$^{-1}$"}
S_ASINH = 0.001
# Emission fields are mostly zero (ocean, and most land), so a colourmap that is
# BLACK at the low end fills the panel with ink and buries the coastlines. This
# one puts near-white at "no emission" and spends its colour where emissions are.
CMAP = "YlOrRd"


def smooth(a, s):
    if s <= 0:
        return a
    a = gaussian_filter1d(a, sigma=s, axis=-1, mode="wrap")
    return gaussian_filter1d(a, sigma=s, axis=-2, mode="reflect")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--decade", type=int, nargs=2, default=[2005, 2014])
    ap.add_argument("--outdir", default="plots/cond_pipeline")
    ap.add_argument("--transforms", nargs="+", default=["mmlin", "mmasinh"],
                    choices=["mmlin", "mmasinh"],
                    help="which normalisations to show. One gives the clean "
                         "three-column paper figure: raw -> smoothed -> normalised.")
    ap.add_argument("--name", default=None, help="output basename")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    raw, sm, anchors = {}, {}, {}
    for scen in ("hist", "ssp370"):
        with xr.open_dataset(f"{DATA}/emissions_{scen}_only_timefixed_bc_co2fix.nc") as ds:
            t = "time" if "time" in ds.dims else "year"
            yrs = np.asarray(ds[t].values).astype(int)
            for v in SPECIES:
                a = np.asarray(ds[v].values, float)
                raw.setdefault(v, {})[scen] = a
                sm.setdefault(v, {})[scen] = smooth(a, args.sigma)
            lat, lon = ds["lat"].values, ds["lon"].values
            if scen == "hist":
                sel = np.where((yrs >= args.decade[0]) & (yrs <= args.decade[1]))[0]
    for v in SPECIES:
        pooled = np.concatenate([sm[v][s].ravel() for s in sm[v]])
        anchors[v] = (float(pooled.min()), float(pooled.max()))
        print(f"[anchor] {v}: lo={anchors[v][0]:.4e} hi={anchors[v][1]:.4e} (min/max of smoothed)")

    LABEL = {"mmlin": "normalised (min/max, linear)",
             "mmasinh": "normalised (min/max + asinh, $s$=0.001)"}
    cols = ["raw inventory", f"smoothed ($\\sigma$={args.sigma:g})"] + \
           [LABEL[t] for t in args.transforms]
    fig = plt.figure(figsize=(4.6 * len(cols), 2.9 * len(SPECIES) + 1.0),
                     constrained_layout=True)
    axes = make_axes(fig, len(SPECIES), len(cols))
    k = 0
    for i, v in enumerate(SPECIES):
        r = raw[v]["hist"][sel].mean(0)
        s_ = sm[v]["hist"][sel].mean(0)
        lo, hi = anchors[v]
        u = np.clip((s_ - lo) / (hi - lo), 0, None)
        lin = 2 * u - 1
        asnh = 2 * np.arcsinh(u / S_ASINH) / np.arcsinh(1 / S_ASINH) - 1
        avail = {"mmlin": lin, "mmasinh": asnh}
        panels = [(r, RAW_UNIT[v]), (s_, RAW_UNIT[v])] + \
                 [(avail[t], "model units") for t in args.transforms]
        for j, (field, unit) in enumerate(panels):
            ax = axes[i][j]
            d, _ = to_pm180(field, lon)
            if j < 2:
                # Each of raw/smoothed gets its OWN scale. Sharing one leaves
                # the smoothed panel black, because the raw field's point
                # sources are ~12x its smoothed peak -- the figure would then
                # hide the very structure the smoothing creates.
                im = draw_map(ax, d, cmap=CMAP, vmin=0,
                              vmax=float(np.nanpercentile(field, 99.9)))
            else:
                im = draw_map(ax, d, cmap=CMAP, vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax, shrink=0.74, label=unit)
            if i == 0:
                ax.set_title(cols[j], fontsize=10)
            if j == 0:
                ax.text(-0.04, 0.5, v, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=12)
            ax.text(0.5, -0.06, f"max {float(np.nanmax(field)):.3g}",
                    transform=ax.transAxes, fontsize=8, ha="center", va="top")
            panel_label(ax, k); k += 1
    proj = " (Robinson)" if HAVE_CARTOPY else ""
    tail = ("Both transforms are bounded to [-1, +1] with NO clipping and nothing "
            "pinned; they differ only in shape." if len(args.transforms) > 1 else
            "The result is bounded to [-1, +1] with NO clipping and nothing pinned: "
            "+1 IS the largest cell in the record.")
    fig.suptitle(
        f"Conditioning pipeline, historical {args.decade[0]}-{args.decade[1]} mean{proj}. "
        "Smoothing acts on RAW units; normalisation follows, anchored on the\n"
        f"MIN/MAX of the smoothed hist+ssp370 field. {tail}", fontsize=11)
    out = os.path.join(args.outdir, args.name or
                       ("cond_pipeline_paper" if len(args.transforms) > 1
                        else f"cond_pipeline_{args.transforms[0]}"))
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=200, bbox_inches="tight")
    print("wrote", out + ".png/.pdf")


if __name__ == "__main__":
    main()
