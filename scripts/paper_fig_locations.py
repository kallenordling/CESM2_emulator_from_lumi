#!/usr/bin/env python3
"""Local variability at named cities over the last N years, per experiment.

A global mean hides the thing a user of a climate emulator usually cares about:
what happens at ONE place. This takes the nearest grid cell to each city and
shows the distribution of values there over the final N years of every
experiment, emulator against CESM2.

TWO DATA LAYOUTS, --mode
------------------------
annual   the LUMI eval output: <VAR>_<scenario>.nc holding <VAR>_model_m<N>
         and <VAR>_cesm_m<N> on a `year` axis. Spread = interannual variability
         pooled across members.

monthly  the Roihu eval output: monthly_<scenario>.nc with dims
         (member, time, lat, lon) and NO CESM2 inside, so the reference comes
         from the monthly training tree via --truth-root. Spread here is
         DOMINATED BY THE SEASONAL CYCLE, which is why the CSV also reports a
         deseasonalised standard deviation: the same numbers after removing
         each month's own climatology. Read the boxes for range and the CSV's
         sd_deseason for "variability" in the sense of year-to-year noise.

The box is the interquartile range, the whiskers the 5th-95th percentile, and
the notch the median. Points beyond the whiskers are drawn individually.

Usage
-----
    python scripts/paper_fig_locations.py --mode annual \\
        --eval-dir <eval>/ep0860_ens25 --out plots/locations_annual.png

    python scripts/paper_fig_locations.py --mode monthly \\
        --eval-dir <eval>/monthly/full_ep163 \\
        --truth-root <data>/training_data_monthly \\
        --out plots/locations_monthly.png
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

# lat, lon in degrees east (the grid is 0-360, and all three are already positive)
LOCATIONS = {
    "Helsinki": (60.17, 24.94),
    "Paris":    (48.86, 2.35),
    "Sydney":   (-33.87, 151.21),
}

VARS = {"TREFHT": dict(label="Temperature", unit="°C"),
        "PRECT":  dict(label="Precipitation", unit="mm/day")}

# Raw CESM2 -> the emulator's units. Mirrors PREPROCESS_FN in
# data/climate_dataset.py; repeated so this script needs no project imports.
TRUTH_CONV = {"TREFHT": lambda x: x - 273.15, "PRECT": lambda x: x * 8.64e7}
SCEN_DIR = {"hist": "hist", "ssp370": "ssp370", "aaer": "AAER", "ghg": "GHG"}

C_EMU, C_CESM = "#D55E00", "#0072B2"


def nearest(da, lat, lon):
    """Nearest grid cell. Longitudes are 0-360 here, so wrap negatives."""
    return da.sel(lat=lat, lon=lon % 360, method="nearest")


# ── annual ────────────────────────────────────────────────────────────────────
def load_annual(eval_dir, var, scen, n_years):
    p = os.path.join(eval_dir, f"{var}_{scen}.nc")
    if not os.path.exists(p):
        return None
    ds = xr.open_dataset(p)
    out = {}
    for side, tag in (("model", "emulator"), ("cesm", "CESM2")):
        names = sorted([v for v in ds.data_vars
                        if re.fullmatch(rf"{var}_{side}_m\d+", v)],
                       key=lambda x: int(x.rsplit("_m", 1)[1]))
        if not names:
            continue
        yv = "year" if side == "model" else ("cesm_year" if "cesm_year" in ds.dims
                                             else "year")
        yrs = ds[yv].values.astype(int)
        keep = yrs >= yrs.max() - n_years + 1
        per_loc = {}
        for city, (la, lo) in LOCATIONS.items():
            vals = [nearest(ds[n].isel({yv: keep}), la, lo).values for n in names]
            per_loc[city] = np.asarray(vals).ravel()        # member x year
        out[tag] = dict(data=per_loc, n=len(names),
                        years=(int(yrs[keep].min()), int(yrs[keep].max())))
    ds.close()
    return out or None


# ── monthly ───────────────────────────────────────────────────────────────────
def _chunk_files(d):
    fs = glob.glob(os.path.join(d, "chunk_*.nc"))
    return sorted(fs, key=lambda f: int(os.path.basename(f)[len("chunk_"):-3]))


def load_monthly(eval_dir, truth_root, scen, n_years):
    p = os.path.join(eval_dir, f"monthly_{scen}.nc")
    if not os.path.exists(p):
        return None
    ds = xr.open_dataset(p)
    n_months = n_years * 12
    out, months = {}, None
    for var in [v for v in VARS if v in ds.data_vars]:
        da = ds[var].isel(time=slice(-n_months, None))
        months = da.time.values
        per_loc = {city: nearest(da, la, lo).values.ravel()
                   for city, (la, lo) in LOCATIONS.items()}
        mon_idx = np.tile(np.array([int(str(t)[5:7]) for t in months]),
                          ds.sizes["member"])
        out.setdefault("emulator", {})[var] = dict(
            data=per_loc, n=int(ds.sizes["member"]), month=mon_idx)

    # CESM2 from the training tree — only the LAST few chunk files are opened,
    # which is what makes this cheap over a network mount.
    if truth_root:
        want = ds.attrs.get("realization")
        for var in [v for v in VARS if v in ds.data_vars]:
            root = os.path.join(truth_root, var, SCEN_DIR.get(scen, scen))
            if not os.path.isdir(root):
                continue
            mem = [m for m in sorted(os.listdir(root))
                   if os.path.isdir(os.path.join(root, m)) and m != "diagnostics"]
            pick = want if want in mem else (mem[0] if mem else None)
            if pick is None:
                continue
            files = _chunk_files(os.path.join(root, pick))
            if not files:
                continue
            # ~49 months per chunk; take enough from the end, then trim exactly.
            need = int(np.ceil(n_months / 49)) + 1
            sub = xr.concat([xr.open_dataset(f)[var] for f in files[-need:]],
                            dim="time").isel(time=slice(-n_months, None))
            sub = TRUTH_CONV.get(var, lambda x: x)(sub)
            per_loc = {city: nearest(sub, la, lo).values.ravel()
                       for city, (la, lo) in LOCATIONS.items()}
            mon_idx = np.array([int(str(t)[5:7]) for t in sub.time.values])
            out.setdefault("CESM2", {})[var] = dict(
                data=per_loc, n=1, month=mon_idx, member=pick)
    ds.close()
    return out or None


def deseason_sd(x, month):
    """sd after removing each calendar month's own mean — the seasonal cycle is
    not 'variability' in the sense anyone means when they ask for it."""
    x = np.asarray(x, float)
    m = np.asarray(month)
    if m.size != x.size:
        return float(np.nanstd(x, ddof=1))
    r = x.copy()
    for k in np.unique(m):
        sel = m == k
        r[sel] -= np.nanmean(x[sel])
    return float(np.nanstd(r, ddof=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["annual", "monthly"], required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--truth-root", default=None,
                    help="monthly mode: the training tree for the CESM2 side")
    ap.add_argument("--scenarios", nargs="+",
                    default=["hist", "ssp370", "aaer", "ghg"])
    ap.add_argument("--vars", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--n-years", type=int, default=10)
    ap.add_argument("--out", default="plots/paper_fig_locations.png")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    # data[(var, scen, side)][city] -> 1-D sample
    data, meta, rows = {}, {}, []
    for scen in args.scenarios:
        if args.mode == "annual":
            for var in args.vars:
                got = load_annual(args.eval_dir, var, scen, args.n_years)
                if not got:
                    print(f"[{scen}/{var}] no data — skipped")
                    continue
                for side, d in got.items():
                    data[(var, scen, side)] = d["data"]
                    meta[(var, scen, side)] = dict(n=d["n"], month=None,
                                                   window=d["years"])
        else:
            got = load_monthly(args.eval_dir, args.truth_root, scen, args.n_years)
            if not got:
                print(f"[{scen}] no data — skipped")
                continue
            for side, per_var in got.items():
                for var, d in per_var.items():
                    if var not in args.vars:
                        continue
                    data[(var, scen, side)] = d["data"]
                    meta[(var, scen, side)] = dict(n=d["n"], month=d.get("month"),
                                                   window=None)

    if not data:
        print("no data", file=sys.stderr)
        return 1

    for (var, scen, side), per_loc in sorted(data.items()):
        for city, x in per_loc.items():
            mth = meta[(var, scen, side)]["month"]
            rows.append(dict(
                mode=args.mode, var=var, scenario=scen, side=side, city=city,
                n_members=meta[(var, scen, side)]["n"], n_values=int(x.size),
                mean=round(float(np.nanmean(x)), 3),
                sd=round(float(np.nanstd(x, ddof=1)), 3),
                sd_deseason=(round(deseason_sd(x, mth), 3)
                             if mth is not None else np.nan),
                p05=round(float(np.nanpercentile(x, 5)), 3),
                p95=round(float(np.nanpercentile(x, 95)), 3),
                unit=VARS[var]["unit"]))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # ── figure: rows = variables, columns = cities ───────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})

    vars_ = [v for v in args.vars if any(k[0] == v for k in data)]
    cities = list(LOCATIONS)
    fig, axes = plt.subplots(len(vars_), len(cities),
                             figsize=(4.5 * len(cities), 3.6 * len(vars_)),
                             squeeze=False)
    for r, var in enumerate(vars_):
        for c, city in enumerate(cities):
            ax = axes[r][c]
            pos, ticks, labels = [], [], []
            for i, scen in enumerate(args.scenarios):
                for j, (side, col) in enumerate((("emulator", C_EMU),
                                                 ("CESM2", C_CESM))):
                    key = (var, scen, side)
                    if key not in data or city not in data[key]:
                        continue
                    x = data[key][city]
                    x = x[np.isfinite(x)]
                    if x.size == 0:
                        continue
                    p = i * 1.0 + (j - 0.5) * 0.34
                    bp = ax.boxplot([x], positions=[p], widths=0.30,
                                    patch_artist=True, notch=True,
                                    whis=(5, 95), showfliers=True,
                                    flierprops=dict(marker=".", ms=2,
                                                    alpha=0.35,
                                                    markerfacecolor=col,
                                                    markeredgecolor="none"),
                                    medianprops=dict(color="k", lw=1.2))
                    for b in bp["boxes"]:
                        b.set(facecolor=col, alpha=0.55, lw=0.8)
                    pos.append(p)
                ticks.append(i * 1.0)
                labels.append(scen)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=8.5)
            ax.set_xlim(-0.7, len(args.scenarios) - 0.3)
            if c == 0:
                ax.set_ylabel(f"{VARS[var]['label']} ({VARS[var]['unit']})")
            if r == 0:
                la, lo = LOCATIONS[city]
                ax.set_title(f"{city}  ({la:.1f}°, {lo:.1f}°E)", fontsize=10)
    axes[0][0].legend(handles=[Patch(facecolor=C_EMU, alpha=0.55, label="Emulator"),
                               Patch(facecolor=C_CESM, alpha=0.55, label="CESM2")],
                      frameon=False, fontsize=8.5, loc="best")

    what = ("last %d years, all members pooled" % args.n_years
            if args.mode == "annual"
            else "last %d years of MONTHLY values — the spread includes the "
                 "seasonal cycle" % args.n_years)
    fig.suptitle(f"Local variability at the nearest grid cell — {what}\n"
                 "box: interquartile range, whiskers: 5th-95th percentile, "
                 "notch: median", fontsize=10.5, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"\nwrote {args.out}")
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
