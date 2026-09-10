#!/usr/bin/env python3
"""
Raw input4MIPs emissions beside the conditioning channels the emulator is fed.

Three species (CO2, BC, SUL) x two spaces:

  LEFT   what the published input4MIPs files say, global totals in physical
         units (Gt CO2/yr cumulative, Tg BC/yr, Tg SO2/yr), historical spliced
         to each scenario at 2015 exactly as the pipeline does it -- scenario
         files are DECADAL, so they are linearly interpolated to annual and the
         anchors are marked.
  RIGHT  the same quantity read straight out of the cond files, summed over the
         grid, in the units the model actually sees.

The two columns are NOT on the same scale: the pipeline's bilinear regrid of a
per-gridpoint (extensive) field discards mass, and how much it discards depends
on the spatial pattern. Each right-hand panel is annotated with the measured
retained fraction so the difference is stated rather than implied.

What the figure is for: every known defect in the conditioning shows up as a
disagreement in shape between the two columns, or as a step at 2015 within one.

Usage:
    python scripts/plot_raw_vs_cond_emissions.py \
        --raw-dir ~/data_staging/inputs4mips --cond-dir ~/data_staging/bc_rebuild
"""
import argparse
import glob
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_EARTH = 6.371e6
SPY = 365.25 * 24 * 3600
SCEN = ["ssp370", "ssp245", "ssp126"]
COLOR = {"hist": "0.25", "ssp370": "#c1121f", "ssp245": "#e07a00", "ssp126": "#0077b6"}

# Same source globs the builders use: make_co2_files.py:42-55 for CO2,
# make_aerosol_files.py:_ANTHRO_PATTERNS for BC and SO2.
RAW = {
    "CO2": {
        "hist":   ["CO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2024-11-25_gn_*.nc",
                   "CO2-em-AIR-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2024-10-21_gn_*.nc"],
        "ssp370": ["CO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc",
                   "CO2-em-AIR-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc"],
        "ssp245": ["CO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc",
                   "CO2-em-AIR-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc"],
        "ssp126": ["CO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc",
                   "CO2-em-AIR-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc"],
    },
    "BC": {
        "hist":   ["BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc"],
        "ssp370": ["BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc"],
        "ssp245": ["BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc"],
        "ssp126": ["BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc"],
    },
    "SUL": {
        "hist":   ["SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc"],
        "ssp370": ["SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc"],
        "ssp245": ["SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc"],
        "ssp126": ["SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc"],
    },
}
# CO2 is stored cumulatively in the cond files; the aerosols are annual rates.
CUMULATIVE = {"CO2": True, "BC": False, "SUL": False}
# Divisor from kg to the unit each row is plotted in, and that unit's name.
UNIT = {"CO2": (1e12, "Gt CO$_2$"), "BC": (1e9, "Tg BC"), "SUL": (1e9, "Tg SO$_2$")}
# The cond channels are stored as "Gt per gridpoint", so summing the grid already
# gives Gt. Only the aerosol rows need Gt -> Tg to match the left column's unit.
COND_TO_UNIT = {"CO2": 1.0, "BC": 1e3, "SUL": 1e3}


def cell_area(lat, lon):
    """Mirror of make_aerosol_files.py:compute_grid_cell_area."""
    dlat = np.abs(np.diff(lat).mean())
    dlon = np.abs(np.diff(lon).mean())
    edges = np.deg2rad(np.clip(np.concatenate([
        [lat[0] - dlat / 2], (lat[:-1] + lat[1:]) / 2, [lat[-1] + dlat / 2]]), -90, 90))
    a = np.abs(np.sin(edges[1:]) - np.sin(edges[:-1])) * np.deg2rad(dlon) * R_EARTH ** 2
    return xr.DataArray(np.broadcast_to(a[:, None], (len(lat), len(lon))),
                        dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def raw_series(raw_dir, patterns, divisor):
    """Global annual total from one or more input4MIPs globs, summed together.

    Sums over sector (surface files) or level (aircraft files), takes the annual
    MEAN of the rate, then area-integrates. Returns {year: value}.
    """
    total = {}
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(raw_dir, pat)))
        if not files:
            print(f"    MISSING: {pat}")
            continue
        ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal",
                               coords="minimal", compat="override")
        var = [v for v in ds.data_vars if "bnds" not in v and "bound" not in v][0]
        da = ds[var]
        for d in ("sector", "level", "lev"):
            if d in da.dims:
                da = da.sum(d)
        da = da.groupby("time.year").mean()
        g = (da * cell_area(ds.lat.values, ds.lon.values)).sum(("lat", "lon")).compute()
        g = g * SPY / divisor
        for y, v in zip(np.asarray(g.year.values, int), g.values):
            total[int(y)] = total.get(int(y), 0.0) + float(v)
    return total


def to_annual(d):
    """Decadal -> annual by linear interpolation, as concat_and_regrid does."""
    yrs = np.array(sorted(d))
    full = np.arange(yrs[0], yrs[-1] + 1)
    return full, np.interp(full, yrs, [d[y] for y in yrs]), yrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.expanduser("~/data_staging/inputs4mips"))
    ap.add_argument("--cond-dir", default=os.path.expanduser("~/data_staging/bc_rebuild"))
    ap.add_argument("--out", default="plots/raw_vs_cond_emissions.png")
    ap.add_argument("--cache", default="plots/raw_vs_cond_emissions.npz")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── raw side (slow; cached) ──────────────────────────────────────────────
    if os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        raw = z["raw"].item()
        print(f"[raw] loaded cache {args.cache}")
    else:
        raw = {}
        for sp, per_exp in RAW.items():
            div = UNIT[sp][0]
            for exp, pats in per_exp.items():
                print(f"[raw] {sp} {exp}")
                raw[(sp, exp)] = raw_series(args.raw_dir, pats, div)
        np.savez(args.cache, raw=np.array(raw, dtype=object))
        print(f"[raw] cached to {args.cache}")

    # ── cond side ────────────────────────────────────────────────────────────
    cond = {}
    for sp in RAW:
        h = xr.open_dataset(os.path.join(args.cond_dir,
                                         "emissions_hist_only_timefixed_bc_co2fix.nc"))
        cond[(sp, "hist")] = (np.asarray(h.year.values, int),
                              h[sp].sum(("lat", "lon")).values)
        for s in SCEN:
            d = xr.open_dataset(os.path.join(
                args.cond_dir, f"emissions_{s}_only_timefixed_bc_co2fix.nc"))
            cond[(sp, s)] = (np.asarray(d.year.values, int),
                             d[sp].sum(("lat", "lon")).values)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), sharex=True)
    for row, sp in enumerate(["CO2", "BC", "SUL"]):
        div, uname = UNIT[sp]
        axL, axR = axes[row]

        # LEFT: raw, spliced hist + each scenario the way the pipeline splices.
        hy = np.array(sorted(y for y in raw[(sp, "hist")] if 1850 <= y <= 2014))
        hv = np.array([raw[(sp, "hist")][y] for y in hy])
        for s in SCEN:
            fy, fv, anchors = to_annual(raw[(sp, s)])
            keep = fy > 2014
            yy = np.concatenate([hy, fy[keep]])
            vv = np.concatenate([hv, fv[keep]])
            if CUMULATIVE[sp]:
                vv = np.cumsum(vv)
            axL.plot(yy, vv, color=COLOR[s], lw=1.4, label=s, zorder=2)
            if not CUMULATIVE[sp]:
                av = np.array([raw[(sp, s)][a] for a in anchors])
                axL.plot(anchors, av, "o", ms=3.5, color=COLOR[s], zorder=3)
        axL.plot(hy, np.cumsum(hv) if CUMULATIVE[sp] else hv,
                 color=COLOR["hist"], lw=1.8, label="historical", zorder=4)

        # RIGHT: the cond channel in the SAME physical unit as the left column,
        # with the published curve behind it. Both are Gt/Tg, so the vertical gap
        # between the pale and solid lines IS the regrid's mass loss.
        k = COND_TO_UNIT[sp]
        for s in SCEN:
            fy, fv, _ = to_annual(raw[(sp, s)])
            keep = fy > 2014
            yy = np.concatenate([hy, fy[keep]])
            vv = np.concatenate([hv, fv[keep]])
            if CUMULATIVE[sp]:
                vv = np.cumsum(vv)
            # Draw only the scenario leg per colour; the shared historical leg is
            # drawn once below, or three overlapping pale colours muddy it.
            axR.plot(yy[yy > 2014], vv[yy > 2014], color=COLOR[s], lw=1.0,
                     alpha=0.3, zorder=1)
        axR.plot(hy, np.cumsum(hv) if CUMULATIVE[sp] else hv,
                 color=COLOR["hist"], lw=1.0, alpha=0.3, zorder=1,
                 label="as published")
        for s in SCEN:
            y, v = cond[(sp, s)]
            axR.plot(y, v * k, color=COLOR[s], lw=1.4, label=s, zorder=2)
        y, v = cond[(sp, "hist")]
        axR.plot(y, v * k, color=COLOR["hist"], lw=1.8, label="historical", zorder=3)

        # Retained fraction, stated not implied, at both ends of the scenario so
        # its DRIFT is visible: the bilinear regrid of an extensive field keeps
        # a pattern-dependent fraction, so this is not one constant.
        def retained(s, year):
            r = to_annual(raw[(sp, s)])
            rv = float(np.interp(year, r[0], r[1]))
            if CUMULATIVE[sp]:
                rv = sum(raw[(sp, "hist")][y] for y in hy) + float(
                    np.interp(year, r[0], np.cumsum(r[1]) - r[1][0]))
            cy, cv = cond[(sp, s)]
            return float(np.interp(year, cy, cv)) / rv

        lines = [f"cond/raw  {s}: {retained(s, 2015):.3e} -> {retained(s, 2100):.3e}"
                 f"  ({100 * (retained(s, 2100) / retained(s, 2015) - 1):+.0f}%)"
                 for s in SCEN]
        if not CUMULATIVE[sp]:
            cy, cv = cond[(sp, "hist")]
            a = float(cv[list(cy).index(2014)])
            b = float(cond[(sp, "ssp370")][1][0])
            lines.append(f"step at 2015: {100 * (b / a - 1):+.1f}%")
        axR.text(0.03, 0.955, "\n".join(lines), transform=axR.transAxes, va="top",
                 fontsize=7.6, color="0.25", family="monospace", zorder=5,
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.5))

        for ax in (axL, axR):
            ax.axvline(2015, color="r", ls="--", lw=0.8, alpha=0.55, zorder=1)
            ax.grid(alpha=0.25)
            ax.set_xlim(1850, 2100)
        kind = "cumulative" if CUMULATIVE[sp] else "annual"
        axL.set_ylabel(f"{uname}  ({kind})")
        axR.set_ylabel(f"{uname}  ({kind})")
        axL.set_title(f"{sp} — raw input4MIPs" + ("" if row else "   (dots = decadal anchors)"),
                      fontsize=10)
        axR.set_title(f"{sp} — conditioning file fed to the emulator "
                      f"(pale = as published)", fontsize=10)
    axes[0][0].legend(fontsize=8, loc="upper left")
    axes[2][0].set_xlabel("Year")
    axes[2][1].set_xlabel("Year")
    fig.suptitle("Emissions as published vs as fed to the emulator — "
                 "historical spliced to each scenario at 2015", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
