#!/usr/bin/env python3
"""Raw emissions across the scenarios, as the conditioning files hold them.

Left: cos(lat)-weighted global mean per year, hist then each SSP, so the
divergence after 2015 is visible on one axis. Right: the 2091-2100 mean map per
scenario for one species, sharing a scale so scenarios are comparable.

RAW units throughout -- no smoothing, no normalisation. This is the input
before any of the pipeline touches it. The single-forcing runs (aaer, ghg) are
included because they are training scenarios too, and their flat lines are the
point: aaer holds CO2 near pre-industrial and ghg holds the aerosols at zero.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_ssp_emissions.py
"""
import argparse, os, sys
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ensemble_mean_maps import make_axes, draw_map, to_pm180, panel_label, HAVE_CARTOPY

DATA = os.path.expanduser("~/mnt/lumi_sc2/emulator_data")
SCEN = {"hist":   ("Historical",   "0.25",      "-"),
        "ssp126": ("SSP1-2.6",     "tab:green", "-"),
        "ssp245": ("SSP2-4.5",     "tab:blue",  "-"),
        "ssp370": ("SSP3-7.0",     "tab:red",   "-"),
        "aaer":   ("Aerosol-only", "tab:orange", "--"),
        "ghg":    ("GHG-only",     "tab:purple", "--")}
UNIT = {"CO2": "kg m$^{-2}$ s$^{-1}$ (cumulative)",
        "SUL": "kg m$^{-2}$ s$^{-1}$", "BC": "kg m$^{-2}$ s$^{-1}$"}


def load(scen, species):
    p = f"{DATA}/emissions_{scen}_only_timefixed_bc_co2fix.nc"
    if not os.path.exists(p):
        return None
    with xr.open_dataset(p) as ds:
        t = "time" if "time" in ds.dims else "year"
        yrs = np.asarray(ds[t].values)
        yrs = np.asarray([int(str(v)[:4]) for v in yrs]) if not np.issubdtype(
            yrs.dtype, np.number) else yrs.astype(int)
        out = {v: np.asarray(ds[v].values, float) for v in species if v in ds}
        return out, yrs, ds["lat"].values, ds["lon"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=["CO2", "SUL", "BC"])
    ap.add_argument("--map-species", default="SUL")
    ap.add_argument("--decade", type=int, nargs=2, default=[2091, 2100])
    ap.add_argument("--outdir", default="plots/ssp_emissions")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data, lat, lon = {}, None, None
    for s in SCEN:
        got = load(s, args.species)
        if got is None:
            print(f"[skip] {s}: no file"); continue
        data[s], yrs, lat, lon = got[0], got[1], got[2], got[3]
        data[s] = (data[s], yrs)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))

    # ── global-mean series, one panel per species ───────────────────────────
    fig, axs = plt.subplots(len(args.species), 1, figsize=(9, 2.9 * len(args.species)),
                            constrained_layout=True, sharex=True)
    for i, v in enumerate(args.species):
        ax = axs[i]
        for s, (label, col, ls) in SCEN.items():
            if s not in data or v not in data[s][0]:
                continue
            arr, yrs = data[s][0][v], data[s][1]
            g = (arr * w).sum(axis=(1, 2)) / w.sum()
            ax.plot(yrs, g, ls, color=col, lw=1.6, label=label)
        ax.set_ylabel(f"{v}\n{UNIT[v]}", fontsize=8)
        ax.grid(alpha=.3)
        panel_label(ax, i)
        if i == 0:
            ax.legend(fontsize=7, ncol=3)
    axs[-1].set_xlabel("year")
    fig.suptitle("Raw emissions by scenario, cos(lat)-weighted global mean "
                 "(conditioning files, before smoothing or normalisation)", fontsize=11)
    out = os.path.join(args.outdir, "ssp_emissions_series")
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out + ".png/.pdf")

    # ── maps of one species, final decade, shared scale ─────────────────────
    v = args.map_species
    keys = [s for s in SCEN if s in data and v in data[s][0]]
    fields = {}
    for s in keys:
        arr, yrs = data[s][0][v], data[s][1]
        sel = (yrs >= args.decade[0]) & (yrs <= args.decade[1])
        if not sel.any():                      # hist ends 2014: use its last decade
            sel = yrs >= (yrs.max() - 9)
        fields[s] = arr[sel].mean(0)
    vmax = float(np.nanpercentile(np.concatenate([f.ravel() for f in fields.values()]), 99.9))
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 2.7 * nrow + 0.9), constrained_layout=True)
    axes = make_axes(fig, nrow, ncol)
    im = None
    for k, s in enumerate(keys):
        ax = axes[k // ncol][k % ncol]
        d, _ = to_pm180(fields[s], lon)
        im = draw_map(ax, d, cmap="YlOrRd", vmin=0, vmax=vmax)
        yrs = data[s][1]
        win = (f"{args.decade[0]}-{args.decade[1]}" if yrs.max() >= args.decade[1]
               else f"{yrs.max()-9}-{yrs.max()}")
        ax.set_title(f"{SCEN[s][0]}  {win}", fontsize=9)
        ax.text(0.5, -0.05, f"max {float(fields[s].max()):.3g}", transform=ax.transAxes,
                fontsize=7.5, ha="center", va="top")
        panel_label(ax, k)
    for k in range(len(keys), nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    fig.colorbar(im, ax=list(axes.ravel()), shrink=0.6, label=f"{v} ({UNIT[v]})")
    proj = " (Robinson)" if HAVE_CARTOPY else ""
    fig.suptitle(f"{v}: raw emissions by scenario, final-decade mean{proj} — shared scale",
                 fontsize=11)
    out = os.path.join(args.outdir, f"ssp_emissions_map_{v}")
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=150, bbox_inches="tight")
    print("wrote", out + ".png/.pdf")


if __name__ == "__main__":
    main()
