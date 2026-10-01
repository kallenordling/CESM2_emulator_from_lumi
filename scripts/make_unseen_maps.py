#!/usr/bin/env python3
"""Ensemble-mean maps for the scenarios the emulator never trained on.

Columns: SSP1-2.6, SSP2-4.5 (both unseen forcing combinations) and the RAMIP
aerosol-removal run ssp370-126aer. Rows: emulator, CESM2, difference. Final
decade anomaly vs 1850-1900, matching fig05/fig06 so the unseen maps are read
on the same terms as the trained ones.

TWO DIFFERENCES FROM fig05/fig06, both forced by the data:
  * No significance stippling. The unseen-scenario eval NetCDFs store only the
    member MEAN (`TREFHT_model_mean`), with no member dimension, so there is no
    spread to test. Do not read the absence of stippling here as a stronger
    claim than fig05/fig06 make -- it is simply not computed.
  * RAMIP is TREFHT-only and 1 member of CESM2 (2015-2079). Its "CESM2" column
    is one realisation, so small-scale structure there is internal variability,
    not a model-data disagreement. See memory ramip_ssp370_126aer_inventory.

The baseline is each side's OWN 1850-1900: the emulator's historical run for the
emulator, CESM2's historical ensemble for CESM2. Mixing them would fold a
mean-state offset into the anomaly.

    ~/miniconda3/envs/plotting/bin/python scripts/make_unseen_maps.py
"""
import argparse, os, sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ensemble_mean_maps import (META, BASELINE, weighted_stats, make_axes,
                                     draw_map, to_pm180, panel_label, PROJECTION,
                                     HAVE_CARTOPY)

EVAL = os.path.expanduser("~/mnt/lumi_sc/eval_output/manual/ep0860_ens25")
REF = os.path.expanduser("~/mnt/lumi_sc2/emulator_data/cmip6")
SCEN = {                       # key: (label, window, ref file stem, ref var)
    "ssp126":         ("SSP1-2.6 (unseen)",      (2091, 2100), "ssp126",        {"TREFHT": "tas", "PRECT": "pr"}),
    "ssp245":         ("SSP2-4.5 (unseen)",      (2091, 2100), "ssp245",        {"TREFHT": "tas", "PRECT": "pr"}),
    "ssp370-126aer":  ("RAMIP ssp370-126aer",    (2070, 2079), "ssp370-126aer", {"TREFHT": "tas", "PRECT": "pr"}),
}

# The RAMIP reference files are named with a ramip_ prefix to keep them apart
# from the 3-member CMIP6 ssp370-126aer tas file that predates them. Temperature
# still resolves to the old 1-member file; precipitation is the 10-member
# download of 2026-09-22, so the two RAMIP columns do NOT have the same ensemble
# size and the precipitation one is the better constrained of the two.
STEM_OVERRIDE = {("ssp370-126aer", "PRECT"): "ramip_ssp370-126aer"}


def emu_maps(var, key, window):
    """Emulator final-decade mean and its own 1850-1900 baseline."""
    def _mean_over(ds, sel_idx):
        """Ensemble+time mean, from either eval schema.

        Older evals stored a precomputed `{var}_model_mean` (year, lat, lon).
        Current ones store `{var}_model` (member, year, lat, lon) and no
        precomputed mean (eval_aero.py:1325), so average the member axis here.
        """
        if f"{var}_model_mean" in ds:
            da = ds[f"{var}_model_mean"]
            return np.asarray(da.isel(year=sel_idx).mean("year").values, float)
        da = ds[f"{var}_model"]
        dims = ["year"] + (["member"] if "member" in da.dims else [])
        return np.asarray(da.isel(year=sel_idx).mean(dims).values, float)

    with xr.open_dataset(f"{EVAL}/{var}_{key}.nc") as ds:
        yrs = np.asarray(ds["year"].values).astype(int)
        sel = (yrs >= window[0]) & (yrs <= window[1])
        fin = _mean_over(ds, np.where(sel)[0])
        lat, lon = ds["lat"].values, ds["lon"].values
    with xr.open_dataset(f"{EVAL}/{var}_hist.nc") as ds:
        yrs = np.asarray(ds["year"].values).astype(int)
        b = (yrs >= BASELINE[0]) & (yrs <= BASELINE[1])
        base = _mean_over(ds, np.where(b)[0])
    return fin - base, lat, lon


def ref_maps(var, key, window):
    """CESM2 final-decade mean minus CESM2's own historical 1850-1900."""
    stem = STEM_OVERRIDE.get((key, var), SCEN[key][2])
    names = SCEN[key][3]
    if var not in names:
        return None
    suffix = "" if var == "TREFHT" else "_pr"
    path = f"{REF}/{stem}{suffix}.nc"
    if not os.path.exists(path):
        return None
    with xr.open_dataset(path) as ds:
        da = ds[names[var]]
        yrs = np.asarray(ds["year"].values).astype(int)
        sel = (yrs >= window[0]) & (yrs <= window[1])
        dims = ["year"] + (["member"] if "member" in da.dims else [])
        fin = np.asarray(da.isel(year=np.where(sel)[0]).mean(dims).values, float)
    hist_path = f"{REF}/historical{suffix}.nc"
    if os.path.exists(hist_path):
        with xr.open_dataset(hist_path) as ds:
            da = ds[names[var]]
            yrs = np.asarray(ds["year"].values).astype(int)
            b = (yrs >= BASELINE[0]) & (yrs <= BASELINE[1])
            dims = ["year"] + (["member"] if "member" in da.dims else [])
            base = np.asarray(da.isel(year=np.where(b)[0]).mean(dims).values, float)
        if var == "PRECT" and np.nanmax(np.abs(base)) < 0.01:
            base = base * 86400.0
        if var == "PRECT" and np.nanmax(np.abs(fin)) < 0.01:
            fin = fin * 86400.0
        return fin - base
    # No CMIP6 historical for this variable (there is no historical_pr.nc).
    # Fall back to the LENS2 historical ensemble baseline that fig05/fig06
    # already use, via make_ensemble_mean_maps' cache. Both are CESM2 1850-1900
    # with identical forcing, so they differ only by internal variability, which
    # an ensemble mean over 51 years makes small -- but they ARE different
    # ensembles (CMIP6 r4/r10/r11 vs LENS2), so say so rather than hide it.
    cache = f"plots/ensmean_maps/cache_{var}_10y_v2.npz"
    if not os.path.exists(cache):
        cache = f"plots/ensmean_maps/cache_{var}_10y.npz"
    if not os.path.exists(cache):
        return None
    z = np.load(cache, allow_pickle=True)
    base = z["ref"].item()["hist"]["base"]
    print(f"  [baseline] {var} {key}: no historical{suffix}.nc -- using the LENS2 "
          f"1850-1900 baseline from {os.path.basename(cache)}")
    if var == "PRECT" and np.nanmax(np.abs(fin)) < 0.01:
        fin = fin * 86400.0
    out = fin - np.asarray(base, float)
    if var == "PRECT":          # kg m-2 s-1 -> mm/day, detected not assumed
        if np.nanmax(np.abs(out)) < 0.01:
            out = out * 86400.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--outdir", default="plots/unseen_maps")
    ap.add_argument("--eval-dir", default=None,
                    help="emulator eval dir; default is the paper's ep0860 run. "
                         "Point it at another arm to rebuild for that arm -- "
                         "without this the script silently plots the PAPER "
                         "checkpoint into whatever outdir you name.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    global EVAL
    if args.eval_dir:
        EVAL = os.path.expanduser(args.eval_dir)
        print(f"[unseen] emulator eval dir: {EVAL}")

    for var in args.var:
        cols, lat, lon = [], None, None
        for key, (label, win, _, names) in SCEN.items():
            if var not in names:
                continue
            e, lat, lon = emu_maps(var, key, win)
            c = ref_maps(var, key, win)
            if c is None:
                print(f"[skip] {var} {key}: no CESM2 reference"); continue
            cols.append((key, label, win, e, c))
        if not cols:
            print(f"[skip] {var}: nothing to plot"); continue

        unit = META[var]["unit"]
        top = np.concatenate([np.ravel(x) for _, _, _, e, c in cols for x in (e, c)])
        tmax = float(np.nanpercentile(np.abs(top), 99))
        dmax = float(np.nanpercentile(np.abs(np.concatenate(
            [np.ravel(e - c) for _, _, _, e, c in cols])), 99))

        rows = ["Emulator", "CESM2", "Emulator - CESM2"]
        fig = plt.figure(figsize=(3.7 * len(cols), 6.8), constrained_layout=True)
        axes = make_axes(fig, 3, len(cols))
        weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], cols[0][3].shape)
        for j, (key, label, win, e, c) in enumerate(cols):
            for i, field in enumerate((e, c, e - c)):
                ax = axes[i][j]
                panel_label(ax, i * len(cols) + j)
                d = to_pm180(field, lon)[0]
                if i < 2:
                    im_top = draw_map(ax, d, cmap=META[var]["anom_cmap"],
                                      vmin=-tmax, vmax=tmax)
                else:
                    im_diff = draw_map(ax, d, cmap=META[var]["dcmap"],
                                       vmin=-dmax, vmax=dmax)
                    st = weighted_stats(e, c, lat)
                    ax.text(0.5, -0.12, f"r {st['r']:.3f}  RMSE {st['rmse']:.2f}",
                            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
                if i == 0:
                    ax.set_title(f"{label}\n{win[0]}-{win[1]} vs 1850-1900", fontsize=9)
                if j == 0:
                    ax.text(-0.04, 0.5, rows[i], transform=ax.transAxes, rotation=90,
                            va="center", ha="right", fontsize=10)
                gm = np.average(field, weights=weights)
                ax.text(0.5, -0.05, f"{gm:+.2f} {unit}", transform=ax.transAxes,
                        fontsize=8, ha="center", va="top")
        fig.colorbar(im_top, ax=list(axes[:2].ravel()), shrink=0.62,
                     label=f"{var} anomaly ({unit})")
        fig.colorbar(im_diff, ax=list(axes[2].ravel()), shrink=0.8,
                     label=f"difference ({unit})")
        proj = f" ({PROJECTION})" if HAVE_CARTOPY else ""
        fig.suptitle(f"{var}: scenarios OUTSIDE the training set — ensemble-mean "
                     f"anomaly{proj}", fontsize=12)
        out = os.path.join(args.outdir, f"unseen_map_{var}")
        for ext in (".png", ".pdf"):
            fig.savefig(out + ext, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}.png/.pdf")
        for key, label, win, e, c in cols:
            st = weighted_stats(e, c, lat)
            print(f"  [skill] {var:6s} {key:14s} r={st['r']:.4f} rmse={st['rmse']:.3f} "
                  f"bias={st['bias']:+.3f} {unit}")


if __name__ == "__main__":
    main()
