#!/usr/bin/env python3
"""Normalised conditioning field as DECADAL means, 1850s to 2090s.

One figure per species: 25 panels, each the 10-year mean of the field the
network is actually fed, on a shared [-1, +1] scale so the industrial rise, the
aerosol peak and its post-2000 decline are comparable panel to panel.

Record is continuous: hist supplies 1850-2014, ssp370 2015-2100. Smoothing acts
on RAW units at sigma 4, then the min/max normalisation of the smoothed
hist+ssp370 field -- the arms' real pipeline, no clipping, nothing pinned.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_cond_decades.py --transform mmasinh
"""
import argparse, os, sys
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ensemble_mean_maps import make_axes, draw_map, to_pm180, HAVE_CARTOPY

DATA = os.path.expanduser("~/mnt/lumi_sc2/emulator_data")
S_ASINH = 0.001


def smooth(a, s):
    if s <= 0:
        return a
    a = gaussian_filter1d(a, sigma=s, axis=-1, mode="wrap")
    return gaussian_filter1d(a, sigma=s, axis=-2, mode="reflect")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=["CO2", "SUL", "BC"])
    ap.add_argument("--transform", default="mmasinh", choices=["mmlin", "mmasinh"])
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--outdir", default="plots/cond_decades")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for v in args.species:
        fields, years = [], []
        for scen, lo_y, hi_y in (("hist", 1850, 2014), ("ssp370", 2015, 2100)):
            with xr.open_dataset(f"{DATA}/emissions_{scen}_only_timefixed_bc_co2fix.nc") as ds:
                t = "time" if "time" in ds.dims else "year"
                yr = np.asarray(ds[t].values).astype(int)
                keep = (yr >= lo_y) & (yr <= hi_y)
                fields.append(smooth(np.asarray(ds[v].values, float)[keep], args.sigma))
                years.append(yr[keep])
                lat, lon = ds["lat"].values, ds["lon"].values
        full = np.concatenate(fields, axis=0)
        yr = np.concatenate(years)
        lo, hi = float(full.min()), float(full.max())
        u = np.clip((full - lo) / (hi - lo), 0, None)
        if args.transform == "mmlin":
            z = 2 * u - 1
        else:
            z = 2 * np.arcsinh(u / S_ASINH) / np.arcsinh(1 / S_ASINH) - 1
        print(f"[{v}] anchors lo={lo:.4e} hi={hi:.4e}  range [{z.min():.2f}, {z.max():.2f}]")

        decades = [(d, d + 9) for d in range(1850, 2100, 10)]
        ncol = 5
        nrow = int(np.ceil(len(decades) / ncol))
        fig = plt.figure(figsize=(3.3 * ncol, 1.95 * nrow + 1.0), constrained_layout=True)
        axes = make_axes(fig, nrow, ncol)
        im = None
        for k, (d0, d1) in enumerate(decades):
            ax = axes[k // ncol][k % ncol]
            sel = (yr >= d0) & (yr <= d1)
            if not sel.any():
                ax.set_visible(False)
                continue
            m = z[sel].mean(0)
            d, _ = to_pm180(m, lon)
            im = draw_map(ax, d, cmap="YlOrRd", vmin=-1, vmax=1)  # light at "no emission"
            ax.set_title(f"{d0}s", fontsize=8)
            ax.text(0.5, -0.04, f"max {float(m.max()):.2f}", transform=ax.transAxes,
                    fontsize=6.5, ha="center", va="top")
        for k in range(len(decades), nrow * ncol):
            axes[k // ncol][k % ncol].set_visible(False)
        fig.colorbar(im, ax=list(axes.ravel()), shrink=0.45, label="model units")
        proj = " (Robinson)" if HAVE_CARTOPY else ""
        fig.suptitle(f"{v}: normalised conditioning, decadal means 1850s-2090s"
                     f"{proj}\nsmoothed $\\sigma$={args.sigma:g} on raw units, then "
                     f"{args.transform} min/max normalisation (hist+ssp370, no clip); "
                     f"hist to 2014 then SSP3-7.0", fontsize=11)
        out = os.path.join(args.outdir, f"cond_decades_{v}_{args.transform}")
        for ext in (".png", ".pdf"):
            fig.savefig(out + ext, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out + ".png/.pdf")


if __name__ == "__main__":
    main()
