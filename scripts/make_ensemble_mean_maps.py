#!/usr/bin/env python3
"""
Ensemble-mean MAPS over the same window the histogram figure pools.

paper_fig_histograms.py pools the final N years of every member into one
distribution. This draws the maps behind that number: the ensemble mean over the
same final-decade window, per experiment, emulator against held-out CESM2, with
their difference underneath.

Two forms of every figure, because they answer different questions:

  absolute  the mean state itself, degC and mm/day. The difference row then
            contains the emulator's mean-state offset (about -0.1 degC globally
            for this checkpoint), which is what the anomaly form removes.
  anomaly   each side minus ITS OWN 1850-1900 mean map, so a shared offset
            cancels and the difference row shows disagreement about the
            RESPONSE. Same convention as the timeseries and histogram figures:
            ssp370 has no pre-industrial of its own and inherits the historical
            baseline, on both sides.

REFERENCE = HELD-OUT MEMBERS ONLY, read from the training trees and truncated to
the emulator's ensemble size, exactly as the other paper figures do. The eval
NetCDF's own CESM arrays are not used: for aaer/ghg most of those members are
training data.

Maps are cached per variable — reading ~37 member trees over the mount takes
tens of minutes. Delete the .npz to force a re-read.

Usage
-----
    python scripts/make_ensemble_mean_maps.py                 # both vars, both forms
    python scripts/make_ensemble_mean_maps.py --var TREFHT
    python scripts/make_ensemble_mean_maps.py --n-years 20
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

EVAL_DIR = "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25_absolute"
TREE_ROOT = "/home/nordling/mnt/lumi_sc/emulator_data/training_data"
DATA_CONFIG = "configs/config_data_ybias_BCprect.yaml"
BASELINE = (1850, 1900)

SCENARIOS = {"hist": ("Historical", "hist"),
             "ssp370": ("SSP3-7.0", "ssp370"),
             "aaer": ("Aerosol-only", "AAER"),
             "ghg": ("GHG-only", "GHG")}

META = {
    "TREFHT": dict(unit="degC", cmap="RdYlBu_r", dcmap="RdBu_r",
                   anom_cmap="RdBu_r", tree_scale=None),
    # LENS2 stores m/s, eval_aero writes mm/day: 1 m/s = 1000*86400 mm/day.
    "PRECT": dict(unit="mm/day", cmap="YlGnBu", dcmap="BrBG",
                  anom_cmap="BrBG", tree_scale=1000.0 * 86400.0),
}


def heldout_members(var):
    config = yaml.safe_load(open(DATA_CONFIG))
    trained = {e["scenario_name"]: set(e.get("realizations", []))
               for e in config["experiment_configs"]}
    out = {}
    for key, (_, subdir) in SCENARIOS.items():
        d = f"{TREE_ROOT}/{var}/{subdir}"
        on_disk = {n for n in os.listdir(d)
                   if n != "diagnostics" and os.path.isdir(f"{d}/{n}")}
        out[key] = sorted(on_disk - trained.get(key, set()))
    return out


def year_axis(da):
    return np.asarray(da["time" if "time" in da.dims else "year"].values).astype(int)


def window_mean(da, years, lo, hi):
    """Mean map over [lo, hi], or NaNs when the record does not reach it."""
    m = (years >= lo) & (years <= hi)
    if not m.any():
        return None
    dim = "time" if "time" in da.dims else "year"
    return np.asarray(da.isel({dim: np.where(m)[0]}).mean(dim).values, dtype=float)


def read_cesm(var, members, n_years):
    """Per experiment: (final-decade mean map, 1850-1900 mean map, window, n)."""
    scale = META[var]["tree_scale"]
    out = {}
    for key, (_, subdir) in SCENARIOS.items():
        finals, bases, wins = [], [], None
        for i, member in enumerate(members[key], 1):
            files = sorted(glob.glob(f"{TREE_ROOT}/{var}/{subdir}/{member}/*.nc"))
            print(f"  [{key}] {i}/{len(members[key])} {member}", flush=True)
            ds = xr.open_mfdataset(files, combine="by_coords", decode_times=False)
            da = ds[var]
            years = year_axis(da)
            hi = int(years.max())
            wins = (hi - n_years + 1, hi)
            f = window_mean(da, years, *wins)
            b = window_mean(da, years, *BASELINE)
            if scale is not None:
                f = f * scale
                b = b * scale if b is not None else None
            finals.append(f)
            bases.append(b)
            lat, lon = ds["lat"].values, ds["lon"].values
            ds.close()
        good_b = [b for b in bases if b is not None]
        out[key] = dict(final=np.mean(finals, axis=0),
                        base=np.mean(good_b, axis=0) if good_b else None,
                        window=wins, n=len(finals), lat=lat, lon=lon)
    return out


def read_emulator(var, n_years, n_cap):
    out = {}
    for key in SCENARIOS:
        ds = xr.open_dataset(f"{EVAL_DIR}/{var}_{key}.nc")
        da = ds[f"{var}_model"].isel(member=slice(None, n_cap[key]))
        years = np.asarray(ds["year"].values).astype(int)
        hi = int(years.max())
        wins = (hi - n_years + 1, hi)
        sel = (years >= wins[0]) & (years <= wins[1])
        final = np.asarray(da.isel(year=np.where(sel)[0]).mean(("member", "year")).values,
                           dtype=float)
        bsel = (years >= BASELINE[0]) & (years <= BASELINE[1])
        base = (np.asarray(da.isel(year=np.where(bsel)[0]).mean(("member", "year")).values,
                           dtype=float) if bsel.any() else None)
        out[key] = dict(final=final, base=base, window=wins,
                        n=int(da.sizes["member"]),
                        lat=ds["lat"].values, lon=ds["lon"].values)
        print(f"  [{key}] emulator {out[key]['n']} members, window {wins}")
        ds.close()
    return out


def to_celsius(side, var, name):
    """LENS2 stores kelvin; eval_aero writes Celsius. Put both in Celsius.

    The anomaly form hides this — a constant offset cancels in a difference of
    anomalies — so it has to be handled here or the absolute difference row is
    a flat -273. Detected, not assumed: a global mean near 287 is kelvin and one
    near 14 is Celsius, and nothing between 40 and 100 is either.
    """
    if var != "TREFHT":
        return side
    for key, d in side.items():
        typical = float(np.nanmean(d["final"]))
        if typical > 100:
            for field in ("final", "base"):
                if d[field] is not None:
                    d[field] = d[field] - 273.15
            print(f"[units] {name:8s} {key:7s} K -> degC (mean was {typical:.1f})")
        elif typical > 40:
            raise SystemExit(f"[units] {name} {key}: mean {typical:.1f} is "
                             "neither kelvin (~287) nor Celsius (~14)")
    return side


def apply_baselines(side):
    """ssp370 has no pre-industrial of its own: it inherits the historical one."""
    if side["ssp370"]["base"] is None:
        side["ssp370"]["base"] = side["hist"]["base"]
    for key, d in side.items():
        if d["base"] is None:
            raise SystemExit(f"{key}: no 1850-1900 window on this side")
    return side


def draw(var, mode, emu, ref, outdir):
    unit = META[var]["unit"]
    lat, lon = emu["hist"]["lat"], emu["hist"]["lon"]
    ext = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    fields = {}
    for key in SCENARIOS:
        e = emu[key]["final"] - (emu[key]["base"] if mode == "anomaly" else 0)
        c = ref[key]["final"] - (ref[key]["base"] if mode == "anomaly" else 0)
        fields[key] = (e, c, e - c)

    # One scale for the top two rows so emulator and CESM2 are comparable, and
    # a separate symmetric scale for the difference row.
    top = np.concatenate([np.ravel(v[:2]) for v in fields.values()])
    diff = np.concatenate([np.ravel(v[2]) for v in fields.values()])
    if mode == "anomaly" or var == "TREFHT":
        tmax = np.nanpercentile(np.abs(top), 99)
        tlim = (-tmax, tmax) if mode == "anomaly" else (np.nanpercentile(top, 1),
                                                        np.nanpercentile(top, 99))
    else:
        tlim = (0, np.nanpercentile(top, 99))
    dmax = np.nanpercentile(np.abs(diff), 99)
    tcmap = META[var]["anom_cmap"] if mode == "anomaly" else META[var]["cmap"]

    rows = ["Emulator", "CESM2 held-out", "Emulator - CESM2"]
    fig, axes = plt.subplots(3, len(SCENARIOS), figsize=(3.5 * len(SCENARIOS), 7.2),
                             constrained_layout=True, squeeze=False)
    for j, (key, (label, _)) in enumerate(SCENARIOS.items()):
        for i in range(3):
            ax = axes[i][j]
            data = fields[key][i]
            if i < 2:
                im_top = ax.imshow(data, origin="lower", extent=ext, cmap=tcmap,
                                   vmin=tlim[0], vmax=tlim[1], aspect="auto")
            else:
                im_diff = ax.imshow(data, origin="lower", extent=ext,
                                    cmap=META[var]["dcmap"], vmin=-dmax, vmax=dmax,
                                    aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                w = emu[key]["window"]
                ax.set_title(f"{label}\n{w[0]}-{w[1]}  "
                             f"(n = {emu[key]['n']} / {ref[key]['n']})", fontsize=9)
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=10)
            gm = np.average(data, weights=np.broadcast_to(
                np.cos(np.deg2rad(lat))[:, None], data.shape))
            ax.text(0.02, 0.05, f"{gm:+.2f} {unit}" if i == 2 else f"{gm:.2f} {unit}",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.4))
    # Flat lists: matplotlib wants axes, and axes[:2].tolist() is a list OF
    # lists, which it silently mis-handles into an AttributeError.
    fig.colorbar(im_top, ax=list(axes[:2].ravel()), shrink=0.7,
                 label=f"{var} {'anomaly ' if mode == 'anomaly' else ''}({unit})")
    fig.colorbar(im_diff, ax=list(axes[2].ravel()), shrink=0.85,
                 label=f"difference ({unit})")
    title = (f"{var} ensemble mean, final decade — "
             f"{'anomaly vs 1850-1900' if mode == 'anomaly' else 'absolute'}")
    fig.suptitle(title, fontsize=12)
    path = os.path.join(outdir, f"ensmean_map_{var}_{mode}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", nargs="+", default=["TREFHT", "PRECT"],
                    choices=["TREFHT", "PRECT"])
    ap.add_argument("--n-years", type=int, default=10,
                    help="final-decade window, matching paper_fig_histograms.py")
    ap.add_argument("--outdir", default="plots/ensmean_maps")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for var in args.var:
        cache = os.path.join(args.outdir, f"cache_{var}_{args.n_years}y.npz")
        members = heldout_members(var)
        n_cap = {k: len(v) for k, v in members.items()}
        if os.path.exists(cache):
            print(f"[{var}] reusing {cache}")
            z = np.load(cache, allow_pickle=True)
            emu, ref = z["emu"].item(), z["ref"].item()
        else:
            print(f"[{var}] emulator maps from {EVAL_DIR}")
            emu = read_emulator(var, args.n_years, n_cap)
            print(f"[{var}] CESM2 held-out maps from the trees (slow)")
            ref = read_cesm(var, members, args.n_years)
            np.savez(cache, emu=emu, ref=ref)
            print(f"[{var}] cached to {cache}")
        emu = apply_baselines(to_celsius(emu, var, "emulator"))
        ref = apply_baselines(to_celsius(ref, var, "CESM2"))
        for mode in ("absolute", "anomaly"):
            print("wrote", draw(var, mode, emu, ref, args.outdir))


if __name__ == "__main__":
    main()
