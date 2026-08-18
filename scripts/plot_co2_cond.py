#!/usr/bin/env python3
"""
Plot the CO2 forcing that the emulator actually receives, per scenario.

This reads the CONDITIONING FILES — the same NetCDFs eval_aero.py feeds to the
model — not the published scenario data. That distinction is the whole point of
the script: several transformations sit between the two, and the emulator only
ever sees the far end of them.

WHAT THE CONDITIONING FILES CONTAIN
-----------------------------------
CO2 is CUMULATIVE per gridpoint (Gt CO2), SUL and BC are per-year. Do NOT trust
the `units` attribute — it is wrong in most of these files. emissions_ssp370_*
labels its cumulative CO2 "Gt CO2 / year / gridpoint" while labelling per-year
SUL "(cumulative)", and the cmip7 files label the same fields the other way
round. This script therefore DETECTS the convention: a field whose global sum
is monotone non-decreasing is treated as cumulative. Both are reported.

ssp126 and the CMIP7 very-low scenario are legitimately non-monotone — they go
net-negative late in the century, so cumulative CO2 turns over. The detector
handles that with a tolerance on the total decline rather than a strict
monotonicity test, and prints which rule fired for every file.

THE ABSOLUTE NUMBERS ARE NOT REAL-WORLD GtCO2
---------------------------------------------
The regrid onto the CESM2 grid does not conserve the extensive sum: it deflates
totals by roughly 4.7x (see the cond_regrid_extensive_deflation note). ssp370
here reaches ~2100 Gt cumulative by 2100 where the published scenario is
~10000 GtCO2. The emulator is self-consistent in this deflated space — it was
trained on it — so the curves are the right thing to compare against EACH OTHER
and the wrong thing to quote as emissions. --scale-to-published applies a single
constant if you need approximate real-world units for a figure caption.

Usage
-----
    python scripts/plot_co2_cond.py                       # cmip7 + ssp370 + ssp126
    python scripts/plot_co2_cond.py --scenarios ssp370 ssp126
    python scripts/plot_co2_cond.py --annual              # add the per-year panel
    python scripts/plot_co2_cond.py --dump-data plots/co2_cond.csv
"""
import argparse
import os
import sys

import numpy as np

# Conditioning files, by scenario label. The CMIP7 set lives on the 462001112
# scratch and the CMIP6-era set on 462001328 — they were built in different
# campaigns and were never consolidated.
SC2 = "/home/nordling/mnt/lumi_sc/emulator_data"     # project 462001112
SC1 = "/home/nordling/mnt/lumi_sc2/emulator_data"    # project 462001328

# Each entry: (label, [(file, style-role), ...]). Historical is prepended where
# a scenario file starts at 2015/2024, so the cumulative curve is continuous
# rather than starting from a bare offset.
SCENARIOS = {
    "ssp370":    [f"{SC2}/emissions_hist_only_timefixed_bc.nc",
                  f"{SC2}/emissions_ssp370_only_timefixed_bc.nc"],
    "ssp126":    [f"{SC2}/emissions_hist_only_timefixed_bc.nc",
                  f"{SC2}/emissions_ssp126_only_timefixed_co2fix_bc.nc"],
    "ssp245":    [f"{SC2}/emissions_hist_only_timefixed_bc.nc",
                  f"{SC2}/emissions_ssp245_only_timefixed_bc.nc"],
    "cmip7_vl":  [f"{SC1}/emissions_hist_cmip7_only_timefixed_bc.nc",
                  f"{SC1}/emissions_vl_cmip7_only_timefixed_bc.nc"],
    "cmip7_h":   [f"{SC1}/emissions_hist_cmip7_only_timefixed_bc.nc",
                  f"{SC1}/emissions_h_cmip7_only_timefixed_bc.nc"],
}
DEFAULT = ["cmip7_vl", "cmip7_h", "ssp370", "ssp126"]

COLORS = {"ssp370": "#c0392b", "ssp126": "#2471a3", "ssp245": "#e08e0b",
          "cmip7_h": "#7d3c98", "cmip7_vl": "#148f77"}
NICE = {"ssp370": "SSP3-7.0", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
        "cmip7_h": "CMIP7 high", "cmip7_vl": "CMIP7 very low"}

# Rough constant from the extensive-regrid deflation note. Approximate, and
# only ever applied on request.
DEFLATION = 4.7


def time_coord(ds):
    """These files use 'year' (historical) or 'time' (scenarios) — see the
    cond_files_only_variant / eval_year_to_time notes; both appear in the same
    pipeline and nothing normalised them."""
    for c in ("year", "time"):
        if c in ds.coords:
            return c
    raise KeyError(f"no year/time coordinate in {list(ds.coords)}")


def global_series(ds, var):
    """Global sum over gridpoints. The fields are per-gridpoint extensive
    quantities, so a plain sum is the global total — NOT an area-weighted mean,
    which would be the right operator for an intensive field and the wrong one
    here."""
    t = time_coord(ds)
    return ds[t].values.astype(int), ds[var].sum(dim=("lat", "lon")).values


def is_cumulative(series):
    """Detect rather than trust the units attribute.

    Verified empirically across all six conditioning files (2026-08-18): CO2 is
    cumulative in EVERY one, including the historical files that start near zero
    and the mitigation scenarios that turn over. The units attributes disagree
    with each other and with the data, so they are ignored entirely.

    Two signatures, either of which is sufficient:

      1. monotone non-decreasing — the ordinary case (hist, ssp370, CMIP7 high)
      2. starts far above its own year-to-year increment — a running total
         carried in from history. ssp370 opens at 324.6 with ~20/yr increments;
         a per-year series opens AT its typical magnitude.

    Rule 2 is what makes the net-negative scenarios work. ssp126 peaks in 2078
    and CMIP7 very-low grows only 25% across the century, so any threshold on
    total growth misclassifies precisely the scenarios this script exists to
    show — an earlier version used 50% growth and cumsum'd CMIP7 very-low into
    37533 Gt, an 80x error that was obvious only because the number was absurd.
    """
    s = np.asarray(series, dtype=float)
    if s.size < 3:
        return True
    d = np.diff(s)
    if (d >= -abs(s).max() * 1e-9).all():
        return True                                  # rule 1: monotone
    step = np.median(np.abs(d))
    return bool(step > 0 and s[0] > 10 * step)       # rule 2: starts high


def load(label, paths, scale):
    """Concatenate the files for one scenario into a single cumulative series."""
    import xarray as xr
    years, cum = [], []
    how = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [SKIP] {label}: missing {p}", file=sys.stderr)
            return None
        ds = xr.open_dataset(p)
        y, g = global_series(ds, "CO2")
        cumulative = is_cumulative(g)
        how.append(f"{os.path.basename(p)}={'cumulative' if cumulative else 'per-year'}"
                   f" (attr '{ds['CO2'].attrs.get('units', '?')}')")
        if not cumulative:
            g = np.cumsum(g)
        ds.close()
        years.append(y)
        cum.append(g)

    # Splice: drop overlap, and offset the scenario onto the end of the
    # historical curve. The scenario files already carry the historical
    # cumulative total in their first value (ssp370 starts at 324.6, not 0), so
    # rebasing would double-count — check before shifting.
    y0, c0 = years[0], cum[0]
    for y1, c1 in zip(years[1:], cum[1:]):
        keep = y1 > y0[-1]
        y1, c1 = y1[keep], c1[keep]
        if c1.size and c1[0] < c0[-1] * 0.5:
            c1 = c1 + c0[-1]        # scenario restarts from zero → splice on
        y0 = np.concatenate([y0, y1])
        c0 = np.concatenate([c0, c1])
    print(f"  [{label}] {' | '.join(how)}")
    return y0, c0 * scale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT,
                    choices=sorted(SCENARIOS), metavar="NAME")
    ap.add_argument("--annual", action="store_true",
                    help="add a second panel with the implied per-year emissions")
    ap.add_argument("--scale-to-published", action="store_true",
                    help=f"multiply by {DEFLATION} to approximate real-world GtCO2")
    ap.add_argument("--out", default="plots/co2_cond")
    ap.add_argument("--dump-data", metavar="CSV")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required — use the plotting env:\n"
              "  /home/nordling/miniconda3/envs/plotting/bin/python "
              "scripts/plot_co2_cond.py", file=sys.stderr)
        return 2

    scale = DEFLATION if args.scale_to_published else 1.0
    unit = ("Gt CO$_2$ (approx. published scale)" if args.scale_to_published
            else "Gt CO$_2$ (emulator input space)")

    print("[co2] reading conditioning files (detecting cumulative vs per-year):")
    data = {}
    for s in args.scenarios:
        r = load(s, SCENARIOS[s], scale)
        if r is not None:
            data[s] = r
    if not data:
        print("[co2] nothing loaded", file=sys.stderr)
        return 1

    n = 2 if args.annual else 1
    fig, axes = plt.subplots(n, 1, figsize=(9, 4.2 * n), sharex=True, squeeze=False)
    ax = axes[0][0]
    for s, (y, c) in data.items():
        ax.plot(y, c, color=COLORS.get(s), lw=2, label=NICE.get(s, s))
    ax.set_ylabel(f"cumulative CO$_2$\n{unit}")
    ax.set_title("CO$_2$ forcing as the emulator receives it")
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    ax.axvline(2015, color="k", lw=.8, ls=":", alpha=.6)
    ax.annotate("historical → scenario", xy=(2015, ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", fontsize=8, alpha=.7)

    if args.annual:
        ax2 = axes[1][0]
        for s, (y, c) in data.items():
            # The model is conditioned on the cumulative field; the annual rate
            # is its first difference and is shown only to make the scenario
            # shapes (peak, decline, net-negative) legible.
            ax2.plot(y[1:], np.diff(c), color=COLORS.get(s), lw=1.5,
                     label=NICE.get(s, s))
        ax2.axhline(0, color="k", lw=.8)
        ax2.set_ylabel(f"implied annual CO$_2$\n{unit.replace('Gt', 'Gt')}/yr")
        ax2.grid(alpha=.3)
        ax2.set_xlabel("year")
    else:
        ax.set_xlabel("year")

    if not args.scale_to_published:
        fig.text(0.01, 0.005,
                 f"Conditioning-file totals: the extensive regrid deflates these "
                 f"by ~{DEFLATION}x vs published scenario emissions. "
                 f"Self-consistent for the emulator; not real-world GtCO2.",
                 fontsize=7, alpha=.65)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"[co2] wrote {args.out}.{ext}")

    if args.dump_data:
        import csv
        os.makedirs(os.path.dirname(args.dump_data) or ".", exist_ok=True)
        with open(args.dump_data, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "year", "cumulative_GtCO2", "annual_GtCO2",
                        "scaled_to_published"])
            for s, (y, c) in data.items():
                d = np.concatenate([[np.nan], np.diff(c)])
                for yi, ci, di in zip(y, c, d):
                    w.writerow([s, int(yi), f"{ci:.6g}",
                                "" if np.isnan(di) else f"{di:.6g}",
                                int(args.scale_to_published)])
        print(f"[co2] wrote {args.dump_data}")

    print("\n[co2] endpoint cumulative CO2 by scenario:")
    for s, (y, c) in data.items():
        pk = int(y[int(np.argmax(c))])
        print(f"   {NICE.get(s, s):16s} {int(y[0])}-{int(y[-1])}: "
              f"{c[-1]:8.1f}   peak {c.max():8.1f} at {pk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
