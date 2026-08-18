#!/usr/bin/env python3
"""
Plot the forcing the emulator actually receives: CO2, SUL and BC, per scenario.

This reads the CONDITIONING FILES — the same NetCDFs eval_aero.py feeds to the
model — not the published scenario data. That distinction is the point: several
transformations sit between the two and the emulator only sees the far end.

WHAT THE CONDITIONING FILES CONTAIN
-----------------------------------
The three channels are stored in DIFFERENT conventions, and each is plotted the
way the model receives it:

    CO2   cumulative per gridpoint (Gt CO2)      — the level, not the rate
    SUL   per year   per gridpoint (Gt SO2/yr)
    BC    per year   per gridpoint (Gt BC/yr)

Do NOT trust the `units` attribute — it is wrong in most of these files, and
wrong in different directions. emissions_ssp370_* labels its cumulative CO2
"Gt CO2 / year / gridpoint" and its per-year SUL "(cumulative)"; the cmip7 files
label CO2 correctly but describe the same SUL field as per-year. The script
therefore DETECTS the convention from the data and prints what it decided for
every file and species, so the choice is auditable rather than assumed.

THE ABSOLUTE NUMBERS ARE NOT REAL-WORLD EMISSIONS
-------------------------------------------------
The regrid onto the CESM2 grid does not conserve the extensive sum: it deflates
totals by roughly 4.7x (see the cond_regrid_extensive_deflation note). ssp370
reaches ~2100 Gt cumulative CO2 here where the published scenario is ~10000
GtCO2. The emulator is self-consistent in this deflated space — it was trained
in it — so these curves are the right thing to compare against each other and
the wrong thing to quote as emissions. --scale-to-published applies the single
constant if a figure caption needs approximate real-world units.

Usage
-----
    python scripts/plot_cond_emissions.py                    # all three species
    python scripts/plot_cond_emissions.py --species CO2      # one panel
    python scripts/plot_cond_emissions.py --species SUL BC
    python scripts/plot_cond_emissions.py --co2-annual       # CO2 as a rate too
    python scripts/plot_cond_emissions.py --dump-data plots/cond_emissions.csv
"""
import argparse
import os
import sys

import numpy as np

# The CMIP7 set lives on the 462001112 scratch and the CMIP6-era set on
# 462001328 — different campaigns, never consolidated.
SC2 = "/home/nordling/mnt/lumi_sc/emulator_data"     # project 462001112
SC1 = "/home/nordling/mnt/lumi_sc2/emulator_data"    # project 462001328

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
DEFAULT_SCEN = ["cmip7_vl", "cmip7_h", "ssp370", "ssp126"]

COLORS = {"ssp370": "#c0392b", "ssp126": "#2471a3", "ssp245": "#e08e0b",
          "cmip7_h": "#7d3c98", "cmip7_vl": "#148f77"}
NICE = {"ssp370": "SSP3-7.0", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
        "cmip7_h": "CMIP7 high", "cmip7_vl": "CMIP7 very low"}
# Dashed for CMIP7. The two families are DIFFERENT DATA, not variants of one
# series: CMIP7 uses CEDS-CMIP-2025 historical and branches in 2024, the CMIP6
# SSPs use CEDS-2017 and branch in 2015. Their historical halves genuinely
# disagree — BC most visibly — so the linestyle keeps a reader from reading the
# gap as an error in one of them.
STYLE = {s: ("--" if s.startswith("cmip7") else "-") for s in NICE}

# How each channel is STORED, i.e. what the model is conditioned on.
SPECIES = {
    "CO2": dict(stored="cumulative", label="cumulative CO$_2$", unit="Gt CO$_2$"),
    "SUL": dict(stored="per-year",   label="SO$_2$ emissions",  unit="Gt SO$_2$ yr$^{-1}$"),
    "BC":  dict(stored="per-year",   label="BC emissions",      unit="Gt BC yr$^{-1}$"),
}
DEFLATION = 4.7


def time_coord(ds):
    """'year' (historical) or 'time' (scenarios) — both appear in this pipeline
    and nothing normalised them (see the eval_year_to_time note)."""
    for c in ("year", "time"):
        if c in ds.coords:
            return c
    raise KeyError(f"no year/time coordinate in {list(ds.coords)}")


def global_series(ds, var):
    """Global sum over gridpoints. These are per-gridpoint EXTENSIVE fields, so
    a plain sum is the global total — an area-weighted mean would be the right
    operator for an intensive field and the wrong one here."""
    t = time_coord(ds)
    return ds[t].values.astype(int), ds[var].sum(dim=("lat", "lon")).values


def is_cumulative(series):
    """Detect whether a CUMULATIVE convention is in use. CO2 ONLY.

    This test is not general and must not be applied to SUL or BC. Rule 2 below
    keys on "starts far above its own year-to-year increment", which is also
    true of any smoothly-varying per-year series: SO2 opens at 0.0217 with
    ~2e-4 steps, a ratio of ~100, HIGHER than cumulative CO2's ~21. Applying it
    to the aerosols classified them as cumulative and differenced them, turning
    SSP3-7.0 SO2 into a series ending at -1.25e-4 — negative emissions, and
    obviously wrong only because the sign is impossible.

    The aerosol channels are therefore not detected at all: they are per-year in
    every conditioning file (verified 2026-08-18 across hist, ssp370, ssp126,
    ssp245 and both CMIP7 files) and are validated for negative values instead.

    For CO2 the test is sound, because a cumulative CO2 series is either
    monotone or dominated by a large historical offset. Two signatures, either
    sufficient:
      1. monotone non-decreasing — the ordinary case (hist, ssp370, CMIP7 high)
      2. starts far above its own year-to-year increment — a running total
         carried in from history (ssp370 opens at 324.6 with ~15/yr steps; a
         per-year series opens AT its typical magnitude)

    Rule 2 is what makes the net-negative scenarios work: ssp126 peaks in 2078
    and CMIP7 very-low grows only 25% across the century, so any threshold on
    total growth misclassifies exactly the scenarios worth showing. An earlier
    version used "grew by 50%" and cumsum'd CMIP7 very-low to 37533 Gt instead
    of 494 — an 80x error caught only because the number was absurd.
    """
    s = np.asarray(series, dtype=float)
    if s.size < 3:
        return True
    d = np.diff(s)
    if (d >= -abs(s).max() * 1e-9).all():
        return True
    step = np.median(np.abs(d))
    return bool(step > 0 and s[0] > 10 * step)


def load(label, paths, var, scale, verbose=True):
    """Concatenate one scenario's files into a single series, in the convention
    the channel is stored in."""
    import xarray as xr
    want_cum = SPECIES[var]["stored"] == "cumulative"
    years, vals, how = [], [], []
    # Detection is CO2-only (see is_cumulative). The aerosol channels are
    # per-year in every file and are taken as stored.
    detect = var == "CO2"
    for p in paths:
        if not os.path.exists(p):
            print(f"  [SKIP] {label}/{var}: missing {p}", file=sys.stderr)
            return None
        ds = xr.open_dataset(p)
        if var not in ds:
            ds.close()
            print(f"  [SKIP] {label}: no {var} in {os.path.basename(p)}", file=sys.stderr)
            return None
        y, g = global_series(ds, var)
        cum = is_cumulative(g) if detect else False
        how.append(f"{os.path.basename(p).replace('emissions_', '')}="
                   f"{'cum' if cum else 'per-yr'}"
                   f"{'' if detect else ' (assumed)'}")
        if not detect and (g < 0).any():
            print(f"  [WARN] {label}/{var}: {int((g < 0).sum())} negative value(s) "
                  f"in {os.path.basename(p)} — a per-year emission field should "
                  f"never be negative; the stored convention may not be per-year.",
                  file=sys.stderr)
        if want_cum and not cum:
            g = np.cumsum(g)
        elif not want_cum and cum:
            g = np.concatenate([[g[0]], np.diff(g)])
        ds.close()
        years.append(y)
        vals.append(g)

    y0, v0 = years[0], vals[0]
    for y1, v1 in zip(years[1:], vals[1:]):
        keep = y1 > y0[-1]
        y1, v1 = y1[keep], v1[keep]
        # Only cumulative channels need rebasing onto the historical total, and
        # only when the scenario file restarts from zero. These scenario files
        # already carry the historical total in their first value (ssp370 opens
        # at 324.6, not 0), so shifting unconditionally would double-count.
        if want_cum and v1.size and v1[0] < v0[-1] * 0.5:
            v1 = v1 + v0[-1]
        y0 = np.concatenate([y0, y1])
        v0 = np.concatenate([v0, v1])
    if verbose:
        print(f"  [{label:9s} {var:3s}] {' | '.join(how)}")
    return y0, v0 * scale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCEN,
                    choices=sorted(SCENARIOS), metavar="NAME")
    ap.add_argument("--species", nargs="+", default=["CO2", "SUL", "BC"],
                    choices=list(SPECIES), metavar="VAR")
    ap.add_argument("--co2-annual", action="store_true",
                    help="add a panel with CO2 as an annual rate (its first "
                         "difference) — the model sees the cumulative field, "
                         "but the rate makes scenario shapes legible")
    ap.add_argument("--scale-to-published", action="store_true",
                    help=f"multiply by {DEFLATION} to approximate real-world totals")
    ap.add_argument("--data-dir", metavar="DIR",
                    help="override the CMIP6-era cond directory (the ssp*/hist "
                         "files). Use it to plot a locally rebuilt set before it "
                         "is copied back to LUMI. The CMIP7 files are unaffected "
                         "and still read from their own project.")
    ap.add_argument("--out", default="plots/cond_emissions")
    ap.add_argument("--dump-data", metavar="CSV")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required — use the plotting env:\n"
              "  /home/nordling/miniconda3/envs/plotting/bin/python "
              "scripts/plot_cond_emissions.py", file=sys.stderr)
        return 2

    scale = DEFLATION if args.scale_to_published else 1.0
    space = ("approx. published scale" if args.scale_to_published
             else "emulator input space")

    if args.data_dir:
        d = args.data_dir.rstrip("/")
        for k, paths in SCENARIOS.items():
            SCENARIOS[k] = [p.replace(SC2, d) for p in paths]
        print(f"[cond] CMIP6-era files from {d}")

    print("[cond] reading conditioning files (detecting cumulative vs per-year):")
    data = {}
    for var in args.species:
        for s in args.scenarios:
            r = load(s, SCENARIOS[s], var, scale)
            if r is not None:
                data[(var, s)] = r
    if not data:
        print("[cond] nothing loaded", file=sys.stderr)
        return 1

    panels = list(args.species) + (["CO2_annual"] if args.co2_annual else [])
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3.4 * len(panels)),
                             sharex=True, squeeze=False)
    for i, pan in enumerate(panels):
        ax = axes[i][0]
        var = "CO2" if pan == "CO2_annual" else pan
        meta = SPECIES[var]
        for s in args.scenarios:
            if (var, s) not in data:
                continue
            y, v = data[(var, s)]
            if pan == "CO2_annual":
                ax.plot(y[1:], np.diff(v), color=COLORS.get(s), lw=1.5,
                        ls=STYLE.get(s, "-"), label=NICE.get(s, s))
            else:
                ax.plot(y, v, color=COLORS.get(s), lw=2,
                        ls=STYLE.get(s, "-"), label=NICE.get(s, s))
        if pan == "CO2_annual":
            ax.set_ylabel(f"implied annual CO$_2$\n{meta['unit']} yr$^{{-1}}$")
            ax.axhline(0, color="k", lw=.8)
        else:
            ax.set_ylabel(f"{meta['label']}\n{meta['unit']}")
        ax.grid(alpha=.3)
        ax.axvline(2015, color="k", lw=.8, ls=":", alpha=.5)
        if any(k.startswith("cmip7") for k in args.scenarios):
            ax.axvline(2024, color="k", lw=.8, ls=":", alpha=.25)
        if i == 0:
            ax.legend(frameon=False, fontsize=9)
            ax.set_title(f"Conditioning fields as the emulator receives them "
                         f"({space})")
            ax.text(0.985, 0.05, "solid: CMIP6 SSPs (branch 2015)\n"
                                 "dashed: CMIP7 (branch 2024)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, alpha=.7)
    axes[-1][0].set_xlabel("year")

    if not args.scale_to_published:
        fig.text(0.01, 0.004,
                 f"Conditioning-file totals: the extensive regrid deflates these by "
                 f"~{DEFLATION}x vs published emissions. Self-consistent for the "
                 f"emulator; not real-world totals.", fontsize=7, alpha=.65)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"[cond] wrote {args.out}.{ext}")

    if args.dump_data:
        import csv
        os.makedirs(os.path.dirname(args.dump_data) or ".", exist_ok=True)
        with open(args.dump_data, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["species", "scenario", "year", "value", "stored_as",
                        "scaled_to_published"])
            for (var, s), (y, v) in data.items():
                for yi, vi in zip(y, v):
                    w.writerow([var, s, int(yi), f"{vi:.6g}",
                                SPECIES[var]["stored"],
                                int(args.scale_to_published)])
        print(f"[cond] wrote {args.dump_data}")

    print("\n[cond] endpoints (2100) and peaks:")
    for var in args.species:
        print(f"  {var}:")
        for s in args.scenarios:
            if (var, s) not in data:
                continue
            y, v = data[(var, s)]
            print(f"    {NICE.get(s, s):16s} {int(y[0])}-{int(y[-1])}: "
                  f"end {v[-1]:10.5g}   peak {v.max():10.5g} at {int(y[int(np.argmax(v))])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
