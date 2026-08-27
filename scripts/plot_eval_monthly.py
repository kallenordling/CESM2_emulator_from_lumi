#!/usr/bin/env python3
"""Figures for eval_monthly.py output — no cartopy, so it runs on Roihu.

Three per run directory:
  <dir>_series.png    weighted global-mean monthly series, one panel per var
  <dir>_seasonal.png  monthly climatology, scenarios overlaid
  <dir>_maps.png      time-mean map per scenario and variable

Global means are cos(lat)-WEIGHTED. Unweighted means over this grid read ~10 degC
low because the poles are over-sampled.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
UNITS = {"TREFHT": "degC", "PRECT": "mm/day"}


def gmean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(dim=("lat", "lon"))


def decimal_year(ds):
    """Calendar year as a float, so scenarios share one 1850-2100 axis.

    "Years into record" put hist and ssp370 on top of each other at x=0, which
    is exactly wrong when the point is that one follows the other.
    """
    t = ds.time.values
    yr = np.array([int(str(v)[:4]) for v in t], dtype=float)
    mo = np.array([int(str(v)[5:7]) for v in t], dtype=float)
    return yr + (mo - 0.5) / 12.0


def running_annual(x, n=12):
    """12-month running mean; the seasonal cycle otherwise hides the trend."""
    if len(x) < n:
        return None
    k = np.ones(n) / n
    return np.convolve(x, k, mode="valid")


def load(run_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "monthly_*.nc"))):
        out[os.path.basename(f)[len("monthly_"):-3]] = xr.open_dataset(f)
    return out


# Scenario name in the eval files -> directory name in the monthly training
# tree. The single-forcing runs are upper-case on disk and lower-case in the
# eval output, which is the sort of mismatch that silently yields "no truth
# found" if you assume they match.
SCEN_DIR = {"hist": "hist", "ssp370": "ssp370", "aaer": "AAER", "ghg": "GHG"}

# The training tree stores RAW CESM2 output; the emulator writes denormalised
# fields. These are exactly PREPROCESS_FN from data/climate_dataset.py, repeated
# here so this script stays importable without the project's dependencies.
TRUTH_CONV = {
    "TREFHT": lambda x: x - 273.15,          # K -> degC
    "PRECT":  lambda x: x * 8.64e7,          # m/s -> mm/day (x1000 x86400)
}


def _chunk_files(d):
    """chunk_*.nc sorted NUMERICALLY. Lexical order gives 0, 1, 10, 11, 2 ...
    and a non-monotonic time axis that xarray will happily concatenate."""
    fs = glob.glob(os.path.join(d, "chunk_*.nc"))
    return sorted(fs, key=lambda f: int(os.path.basename(f)[len("chunk_"):-3]))


def _open_chunks(files, var):
    """Concatenate a member's chunks. Falls back to plain open_dataset when dask
    is unavailable, since open_mfdataset requires it."""
    try:
        ds = xr.open_mfdataset(files, combine="nested", concat_dim="time")
    except (ImportError, ValueError):
        ds = xr.concat([xr.open_dataset(f) for f in files], dim="time")
    return ds[var]


def load_truth(truth_root, runs, n_members=1):
    """CESM2 monthly data matching each emulated scenario, same units and shape.

    The eval conditions on ONE realization (recorded in the file's attrs), and
    with --prev-mode truth it is fed that realization's own previous state — so
    that member is the like-for-like reference and is preferred over an
    arbitrary one. n_members > 1 adds others for spread.
    """
    truth = {}
    for name, d in runs.items():
        sub = SCEN_DIR.get(name)
        if sub is None:
            print(f"[truth] {name}: no tree mapping — skipped")
            continue
        var_out = {}
        for var in d.data_vars:
            root = os.path.join(truth_root, var, sub)
            if not os.path.isdir(root):
                print(f"[truth] {name}/{var}: {root} missing — skipped")
                continue
            avail = sorted(m for m in os.listdir(root)
                           if os.path.isdir(os.path.join(root, m))
                           and m != "diagnostics")
            want = d.attrs.get("realization")
            picks = ([want] if want in avail else []) + [m for m in avail if m != want]
            picks = picks[:n_members]
            if not picks:
                print(f"[truth] {name}/{var}: no members under {root}")
                continue
            arrs = []
            for m in picks:
                files = _chunk_files(os.path.join(root, m))
                if not files:
                    continue
                arrs.append(TRUTH_CONV.get(var, lambda x: x)(_open_chunks(files, var)))
            if not arrs:
                continue
            da = xr.concat(arrs, dim="member").assign_coords(
                member=np.arange(len(arrs)))
            # Clip to the emulated period so the two lines cover the same years.
            t = d.time.values
            da = da.sel(time=slice(t.min(), t.max()))
            var_out[var] = da
            print(f"[truth] {name}/{var}: {len(picks)} member(s) "
                  f"{picks[0]}{' (matched)' if picks[0] == want else ''}, "
                  f"{da.sizes.get('time', 0)} months")
        if var_out:
            truth[name] = xr.Dataset(var_out)
    return truth


def fig_series(runs, path, xlim=None):
    vars_ = sorted({v for d in runs.values() for v in d.data_vars})
    fig, axes = plt.subplots(len(vars_), 1, figsize=(11, 3.2 * len(vars_)), squeeze=False)
    for ax, v in zip(axes[:, 0], vars_):
        for name, d in runs.items():
            if v not in d:
                continue
            g = gmean(d[v])
            t = decimal_year(d)
            mu = g.mean("member").values
            line, = ax.plot(t, mu, lw=0.5, alpha=0.35)
            # The 12-month mean carries the signal; the raw monthly trace is
            # kept faint behind it so the seasonal amplitude stays visible.
            sm = running_annual(mu)
            if sm is not None:
                ax.plot(t[11:], sm, lw=1.6, color=line.get_color(),
                        label=f"{name} ({int(t[0])}-{int(t[-1])})")
            else:
                line.set_label(f"{name} ({int(t[0])}-{int(t[-1])})")
                line.set_alpha(1.0)
            if d.sizes["member"] > 1:
                ax.fill_between(t, g.min("member"), g.max("member"),
                                alpha=0.15, lw=0, color=line.get_color())
        ax.set_ylabel(f"{v} [{UNITS.get(v, '')}]")
        ax.set_xlabel("year")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[0, 0].set_title("Weighted global mean — thin: monthly, thick: 12-month mean, "
                         "shading: member spread")
    if xlim:
        for ax in axes[:, 0]:
            ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[WROTE] {path}")


def fig_seasonal(runs, path):
    vars_ = sorted({v for d in runs.values() for v in d.data_vars})
    fig, axes = plt.subplots(1, len(vars_), figsize=(5.5 * len(vars_), 3.8), squeeze=False)
    for ax, v in zip(axes[0], vars_):
        for name, d in runs.items():
            if v not in d:
                continue
            clim = gmean(d[v]).mean("member").groupby("time.month").mean()
            ax.plot(clim.month, clim, marker="o", ms=3, lw=1.2, label=name)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTHS)
        ax.set_ylabel(f"{v} [{UNITS.get(v, '')}]")
        ax.set_title(f"{v} seasonal cycle")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[WROTE] {path}")


def fig_maps(runs, path):
    vars_ = sorted({v for d in runs.values() for v in d.data_vars})
    names = list(runs)
    fig, axes = plt.subplots(len(vars_), len(names),
                             figsize=(4.4 * len(names), 2.6 * len(vars_)), squeeze=False)
    for r, v in enumerate(vars_):
        # One colour scale per variable so scenarios are comparable by eye.
        fields = {n: runs[n][v].mean(dim=("member", "time")) for n in names if v in runs[n]}
        lo = min(float(f.min()) for f in fields.values())
        hi = max(float(f.max()) for f in fields.values())
        for c, n in enumerate(names):
            ax = axes[r, c]
            if n not in fields:
                ax.axis("off")
                continue
            im = ax.imshow(fields[n].values, origin="lower", aspect="auto",
                           cmap="RdBu_r" if v == "TREFHT" else "viridis",
                           vmin=lo, vmax=hi,
                           extent=[0, 360, -90, 90])
            ax.set_title(f"{n} — {v}", fontsize=9)
            if c == 0:
                ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Time-mean fields (member mean)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[WROTE] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="a directory of monthly_*.nc")
    ap.add_argument("--compare-dir", default=None,
                    help="second run dir to overlay in the series plot "
                         "(e.g. the free-running one against teacher forcing)")
    ap.add_argument("--truth-root", default=None, metavar="DIR",
                    help="monthly training tree (<VAR>/<scenario>/<member>/"
                         "chunk_*.nc). Overlays the CESM2 data the emulator is "
                         "imitating, converted to the same units.")
    ap.add_argument("--truth-members", type=int, default=1,
                    help="how many CESM2 members to load (default 1: the "
                         "realization the eval was conditioned on)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("Y0", "Y1"),
                    help="fix the year axis, e.g. --xlim 1850 2100")
    args = ap.parse_args()

    runs = load(args.run_dir)
    if not runs:
        sys.exit(f"[FATAL] no monthly_*.nc in {args.run_dir}")
    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.run_dir))

    # CESM2 goes in the SAME dict, so every figure gets it without special
    # cases; the key carries the label the legend shows.
    plot_sets = dict(runs)
    if args.truth_root:
        truth = load_truth(args.truth_root, runs, args.truth_members)
        if not truth:
            print("[truth] nothing loaded — figures will show the emulator only")
        plot_sets = {}
        for name in runs:
            plot_sets[f"{name} emulator"] = runs[name]
            if name in truth:
                plot_sets[f"{name} CESM2"] = truth[name]

    fig_series(plot_sets, os.path.join(out_dir, f"{tag}_series.png"), args.xlim)
    fig_seasonal(plot_sets, os.path.join(out_dir, f"{tag}_seasonal.png"))
    fig_maps(plot_sets, os.path.join(out_dir, f"{tag}_maps.png"))

    if args.compare_dir:
        other = load(args.compare_dir)
        shared = set(runs) & set(other)
        if not shared:
            print(f"[WARN] no shared scenarios with {args.compare_dir}")
        else:
            merged = {f"{k} truth": runs[k] for k in shared}
            merged.update({f"{k} free": other[k] for k in shared})
            fig_series(merged, os.path.join(out_dir, f"{tag}_vs_free_series.png"),
                       args.xlim)
            fig_seasonal(merged, os.path.join(out_dir, f"{tag}_vs_free_seasonal.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
