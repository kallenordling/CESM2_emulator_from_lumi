#!/usr/bin/env python3
"""The conditioning field at each stage of the nopca pipeline.

    cond_file -> gaussian smooth (sigma 4, RAW units) -> normalize -> model

with v1_noclip (no clip) and anchors fitted on hist+ssp370 only. Two figures:

  maps    one row per species, one column per stage, for a chosen decade.
          Each stage has its OWN units, so each panel carries its own colourbar
          -- a shared one would be meaningless across kg/m2/s and model units.
  series  cos(lat)-weighted global mean per stage over the full record, one
          panel per species, hist and ssp370 drawn continuously.

Smoothing conserves the area mean almost exactly, so the raw and smoothed
series overlie; the point of the series is the NORMALISED curve, which is what
the model is actually driven by.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_cond_stages.py
"""
import argparse, os, sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ensemble_mean_maps import make_axes, draw_map, to_pm180, panel_label, HAVE_CARTOPY

DATA = os.path.expanduser("~/mnt/lumi_sc2/emulator_data")
SPECIES = ["CO2", "SUL", "BC"]
PCTL = {"CO2": (1, 99), "SUL": (5, 95), "BC": (5, 95)}
RAW_UNIT = {"CO2": "kg m$^{-2}$ s$^{-1}$ (cum.)", "SUL": "kg m$^{-2}$ s$^{-1}$",
            "BC": "kg m$^{-2}$ s$^{-1}$"}


def smooth(a, s):
    if s <= 0:
        return a
    a = gaussian_filter1d(a, sigma=s, axis=-1, mode="wrap")      # lon periodic
    return gaussian_filter1d(a, sigma=s, axis=-2, mode="reflect")


def load(scen):
    out = {}
    with xr.open_dataset(f"{DATA}/emissions_{scen}_only_timefixed_bc_co2fix.nc") as ds:
        t = "time" if "time" in ds.dims else "year"
        yrs = ds[t].values
        yrs = np.asarray([int(str(v)[:4]) for v in yrs] if not np.issubdtype(
            np.asarray(yrs).dtype, np.number) else yrs).astype(int)
        for v in SPECIES:
            if v in ds:
                out[v] = np.asarray(ds[v].values, float)
        lat, lon = ds["lat"].values, ds["lon"].values
    return out, yrs, lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--decade", type=int, nargs=2, default=[2005, 2014])
    ap.add_argument("--outdir", default="plots/cond_stages")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    hist, yh, lat, lon = load("hist")
    ssp, ys, _, _ = load("ssp370")
    w = np.cos(np.deg2rad(lat))[:, None]

    # Anchors: hist+ssp370 pooled, on the SMOOTHED field (normalise runs after)
    anchors = {}
    for v in SPECIES:
        pooled = np.concatenate([smooth(hist[v], args.sigma).ravel(),
                                 smooth(ssp[v], args.sigma).ravel()])
        plo, phi = PCTL[v]
        lo, hi = np.percentile(pooled, plo), np.percentile(pooled, phi)
        anchors[v] = (lo, hi, (lo + hi) / 2, (hi - lo) / 2)
        print(f"[anchor] {v}: lo={lo:.4e} hi={hi:.4e}")

    # ── maps ────────────────────────────────────────────────────────────────
    lo_y, hi_y = args.decade
    fig = plt.figure(figsize=(14.5, 3.0 * len(SPECIES) + 0.8), constrained_layout=True)
    axes = make_axes(fig, len(SPECIES), 3)
    k = 0
    for i, v in enumerate(SPECIES):
        sel = np.where((yh >= lo_y) & (yh <= hi_y))[0]
        raw = hist[v][sel].mean(0)
        sm = smooth(hist[v], args.sigma)[sel].mean(0)
        _, _, mid, half = anchors[v]
        nm = (sm - mid) / half
        for j, (field, title, unit) in enumerate((
                (raw, f"{v} raw", RAW_UNIT[v]),
                (sm, f"{v} smoothed ($\\sigma$={args.sigma:g})", RAW_UNIT[v]),
                (nm, f"{v} normalised", "model units"))):
            ax = axes[i][j]
            d, _ = to_pm180(field, lon)
            if j < 2:
                vmax = float(np.nanpercentile(raw, 99.5)) or 1.0
                im = draw_map(ax, d, cmap="inferno", vmin=0, vmax=vmax)
            else:
                im = draw_map(ax, d, cmap="inferno", vmin=-1,
                              vmax=float(np.nanpercentile(nm, 99.5)))
            fig.colorbar(im, ax=ax, shrink=0.72, label=unit)
            ax.set_title(title, fontsize=9)
            panel_label(ax, k); k += 1
            ax.text(0.5, -0.06, f"max {float(field.max()):.3g}", transform=ax.transAxes,
                    fontsize=7.5, ha="center", va="top")
    proj = f" ({'Robinson'})" if HAVE_CARTOPY else ""
    fig.suptitle(f"Conditioning at each pipeline stage, historical {lo_y}-{hi_y} mean{proj}\n"
                 f"smooth (raw units) -> normalise (hist+ssp370 anchors, NO clip); PCA disabled",
                 fontsize=11)
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(args.outdir, "cond_stages_maps" + ext), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(args.outdir, "cond_stages_maps.png/.pdf"))

    # ── global-mean series ──────────────────────────────────────────────────
    fig, axs = plt.subplots(len(SPECIES), 2, figsize=(12, 2.9 * len(SPECIES)),
                            constrained_layout=True)
    for i, v in enumerate(SPECIES):
        _, _, mid, half = anchors[v]
        for src, yrs, lab in ((hist, yh, "historical"), (ssp, ys, "SSP3-7.0")):
            # np.average takes 1-D weights only for a multi-axis mean, so do
            # the cos(lat) weighting explicitly over (lat, lon).
            def gm(f, _w=w):
                ww = np.broadcast_to(_w, f.shape[1:])
                return (f * ww).sum(axis=(1, 2)) / ww.sum()
            raw_s = gm(src[v])
            sm_s = gm(smooth(src[v], args.sigma))
            nm_s = (sm_s - mid) / half
            ls = "-" if lab == "historical" else "--"
            axs[i, 0].plot(yrs, raw_s, ls, color="0.2", label=f"raw, {lab}")
            axs[i, 0].plot(yrs, sm_s, ls, color="tab:orange", alpha=.8,
                           label=f"smoothed, {lab}")
            axs[i, 1].plot(yrs, nm_s, ls, color="tab:blue", label=f"normalised, {lab}")
        axs[i, 0].set_ylabel(f"{v}\n{RAW_UNIT[v]}", fontsize=8)
        axs[i, 1].set_ylabel("model units", fontsize=8)
        axs[i, 1].axhline(1, color="crimson", lw=0.8, ls=":")
        axs[i, 1].axhline(-1, color="crimson", lw=0.8, ls=":")
        for a in axs[i]:
            a.grid(alpha=.3); a.legend(fontsize=6.5)
        panel_label(axs[i, 0], 2 * i); panel_label(axs[i, 1], 2 * i + 1)
    axs[0, 0].set_title("raw vs smoothed (they overlie: smoothing conserves the area mean)",
                        fontsize=9)
    axs[0, 1].set_title("after normalisation — what the model is driven by\n"
                        "(dotted red = the nominal [-1, +1] range)", fontsize=9)
    for a in axs[-1]:
        a.set_xlabel("year")
    fig.suptitle("Conditioning global mean at each stage "
                 "(cos-lat weighted; hist solid, ssp370 dashed)", fontsize=11)
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(args.outdir, "cond_stages_series" + ext), dpi=150,
                    bbox_inches="tight")
    print("wrote", os.path.join(args.outdir, "cond_stages_series.png/.pdf"))


if __name__ == "__main__":
    main()
