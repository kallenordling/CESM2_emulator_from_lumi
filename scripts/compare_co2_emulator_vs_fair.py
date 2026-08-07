#!/usr/bin/env python3
"""
Compare the CO2 emissions the emulator is conditioned on against the CO2
emissions FaIR is driven with, for the CMIP7 scenarios.

Why this matters
----------------
The emulator's CO2 channel is a CUMULATIVE per-gridpoint inventory built from
gridded input4MIPs files; FaIR is driven by global stylised CO2 pathways. If the
two disagree in absolute magnitude, any comparison of their temperature
responses is meaningless — and there is precedent: the CMIP6-era cond files
carry a ~5x inflated CO2 inventory (cumulative 11,132 GtCO2 at 2014 vs ~2,000
real-world), which is why the emulator's absolute TCRE cannot be compared to
AR6 without that caveat. This script checks whether the CMIP7 files have the
same problem.

Emulator side — two possible sources
------------------------------------
--cond-dir  the built cond files (emissions_*_cmip7_only_timefixed_bc.nc).
            PREFERRED: this is literally what the model eats, and it includes
            aircraft CO2. The CO2 variable is CUMULATIVE per gridpoint, so the
            global total is a plain SUM over (lat, lon) — never area-weighted
            (cond_co2_units) — and the annual rate is its year-to-year diff.
--raw-dir   the raw input4MIPs gridded files. Fallback for before the cond files
            are built. Applies make_cmip7_cond.py's exact conversion
            (sum sectors -> annual mean of the rate -> * area * s/yr / 1e12).
            Includes aircraft only if the *-em-AIR-anthro files are present.

FaIR side
---------
emissions_adjusted.csv from run_scenarios.py (species x year, GtCO2/yr).
CEDS/IIASA `*_em_anthro` is fossil-fuel + industry, i.e. the counterpart of
FaIR's "CO2 FFI"; FaIR's AFOLU (land use) has no gridded counterpart here, so
BOTH FaIR curves are plotted and FFI is the like-for-like one.

Usage
-----
    python scripts/compare_co2_emulator_vs_fair.py \
        --raw-dir /home/nordling/mnt/lumi_sc2/emulator_data/emission_data/inputs4mips \
        --fair-emissions ~/Downloads/chrisroadmap-cmip7-scenariomip-3129623/output/emissions_adjusted.csv \
        --out plots/co2_emulator_vs_fair.png
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

R_EARTH = 6.371e6
SECONDS_PER_YEAR = 365.25 * 24 * 3600
KG_PER_GT = 1e12

# emulator scenario -> FaIR scenario (see scripts/make_fair_reference.py for the
# reasoning; h is unambiguous within 2100, vl is a documented assumption)
DEFAULT_MAP = {"h": "high-extension", "vl": "verylow"}
COLORS = {"h": "#d62728", "vl": "#2ca02c", "hist": "#1f77b4"}


def cell_area(lat, lon):
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    dlat = abs(np.diff(lat).mean()); dlon = abs(np.diff(lon).mean())
    edges = np.deg2rad(np.clip(np.concatenate(
        [[lat[0] - dlat / 2], (lat[:-1] + lat[1:]) / 2, [lat[-1] + dlat / 2]]), -90, 90))
    a = np.abs(np.sin(edges[1:]) - np.sin(edges[:-1])) * np.deg2rad(dlon) * R_EARTH ** 2
    return np.broadcast_to(a[:, None], (len(lat), len(lon)))


def from_raw(raw_dir, pattern, year_lo, year_hi):
    """Global annual GtCO2/yr from raw input4MIPs files (make_cmip7_cond.py recipe).

    Streams ONE FILE AT A TIME and reduces to a global scalar per year before
    moving on. open_mfdataset across the whole historical set instead makes dask
    re-read during rechunking (~12.5 GB of reads for ~4 GB of files) and holds
    several GB resident — painful over sshfs. Each file here is read once and
    collapses immediately to a per-year series.
    """
    import xarray as xr
    files = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not files:
        return None
    parts = []
    for i, fp in enumerate(files, 1):
        print(f"      [{i}/{len(files)}] {os.path.basename(fp)}", flush=True)
        with xr.open_dataset(fp, chunks={"time": 12}) as ds:
            ds = ds.drop_vars([v for v in ds if "bnds" in str(v) or "bound" in str(v)],
                              errors="ignore")
            var = [n for n in ds.data_vars if "bnds" not in n and "bound" not in n][0]
            da = ds[var]
            for d in ("sector", "sectors", "level", "lev"):
                if d in da.dims:
                    da = da.sum(dim=d)
            yrs = da["time.year"]
            if int(yrs.max()) < year_lo or int(yrs.min()) > year_hi:
                continue                      # file entirely outside the window
            da = da.sel(time=(yrs >= year_lo) & (yrs <= year_hi))
            if da.sizes.get("time", 0) == 0:
                continue
            ann = da.groupby("time.year").mean()
            area = xr.DataArray(cell_area(ann.lat.values, ann.lon.values),
                                dims=["lat", "lon"],
                                coords={"lat": ann.lat.values, "lon": ann.lon.values})
            gt = (ann * area * SECONDS_PER_YEAR / KG_PER_GT).sum(dim=["lat", "lon"])
            parts.append(gt.compute().to_series())
    if not parts:
        return None
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="first")]


def emulator_from_raw(raw_dir, scenarios, hist_source, scen_version,
                      hist_end, y0, y1):
    """Per-scenario global annual GtCO2/yr, hist spliced with each scenario."""
    print(f"[emulator] reading raw input4MIPs from {raw_dir}", flush=True)
    print("[emulator] historical (CEDS) …", flush=True)
    hist = from_raw(raw_dir, f"CO2-em-anthro_*CMIP_{hist_source}_gn_*.nc", y0, hist_end)
    if hist is None:
        print(f"ERROR: no CO2-em-anthro CEDS files in {raw_dir}", file=sys.stderr)
        return None, None
    hist_air = from_raw(raw_dir, f"CO2-em-AIR-anthro_*CMIP_{hist_source}_gn_*.nc",
                        y0, hist_end)
    if hist_air is not None:
        print("[emulator]   + aircraft", flush=True)
        hist = hist.add(hist_air, fill_value=0)
    else:
        print("[emulator]   (no aircraft files — surface anthro only)", flush=True)

    out = {}
    for sc in scenarios:
        print(f"[emulator] scenario {sc} …", flush=True)
        s = from_raw(raw_dir,
                     f"CO2-em-anthro_*ScenarioMIP_IIASA-IAMC-{sc}-{scen_version}_gn_*.nc",
                     hist_end + 1, y1)
        if s is None:
            print(f"  WARNING: no files for scenario {sc}; skipping")
            continue
        s_air = from_raw(raw_dir,
                         f"CO2-em-AIR-anthro_*ScenarioMIP_IIASA-IAMC-{sc}-{scen_version}_gn_*.nc",
                         hist_end + 1, y1)
        if s_air is not None:
            s = s.add(s_air, fill_value=0)
        # IIASA ScenarioMIP files are NOT annual: they carry monthly data for
        # only ~17 sampled years over 2022-2100 (decadal-ish). Interpolate to
        # annual before any cumulative sum, exactly as data/make_cmip7_cond.py
        # does — otherwise cumsum() adds 17 sparse points as if consecutive
        # years and under-counts cumulative CO2 by a factor of ~4.
        full = np.arange(int(s.index.min()), int(s.index.max()) + 1)
        if len(full) != len(s):
            print(f"  [{sc}] scenario is sparse ({len(s)} yrs over "
                  f"{s.index.min()}-{s.index.max()}) — interpolating to annual",
                  flush=True)
            s = s.reindex(full).interpolate(method="index")
        out[sc] = pd.concat([hist, s]).sort_index()
    return out, hist


def emulator_from_cond(cond_dir, scenarios, suffix):
    """Global annual GtCO2/yr from the built cond files.

    The cond CO2 variable is CUMULATIVE per gridpoint, so global total = plain
    SUM over lat/lon (cond_co2_units: area-weighting it is wrong), and the
    annual rate is the year-to-year difference of that cumulative curve.
    """
    import xarray as xr
    print(f"[emulator] reading cond files from {cond_dir}", flush=True)
    hist_f = os.path.join(cond_dir, f"emissions_hist{suffix}.nc")
    if not os.path.exists(hist_f):
        print(f"ERROR: {hist_f} not found. Build the cond files first "
              f"(run_make_cmip7_cond.sh), or use --raw-dir.", file=sys.stderr)
        return None, None
    def cum(path):
        ds = xr.open_dataset(path)
        t = "time" if "time" in ds.dims else "year"
        s = ds["CO2"].sum(dim=["lat", "lon"]).to_series()
        s.index = [int(x) for x in ds[t].values]
        return s
    hist_cum = cum(hist_f)
    out = {}
    for sc in scenarios:
        p = os.path.join(cond_dir, f"emissions_{sc}{suffix}.nc")
        if not os.path.exists(p):
            print(f"  WARNING: {p} not found; skipping {sc}")
            continue
        full = pd.concat([hist_cum, cum(p)]).sort_index()
        out[sc] = full.diff().dropna()      # cumulative -> annual
    return out, hist_cum


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare emulator-conditioning CO2 against FaIR's driving CO2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cond-dir", help="Dir with emissions_*_cmip7_*.nc (preferred)")
    src.add_argument("--raw-dir", help="Dir with raw input4MIPs .nc (fallback)")
    ap.add_argument("--fair-emissions", required=True,
                    help="emissions_adjusted.csv from run_scenarios.py")
    ap.add_argument("--scenarios", nargs="+", default=["h", "vl"])
    ap.add_argument("--suffix", default="_cmip7_only_timefixed_bc")
    ap.add_argument("--hist-source", default="CEDS-CMIP-2025-04-18")
    ap.add_argument("--scen-version", default="1-1-0")
    ap.add_argument("--hist-end", type=int, default=2023)
    ap.add_argument("--year-min", type=int, default=1850)
    ap.add_argument("--year-max", type=int, default=2100)
    ap.add_argument("--out", default="plots/co2_emulator_vs_fair.png")
    ap.add_argument("--csv", default=None, help="Optional CSV of the compared series")
    args = ap.parse_args()

    if args.cond_dir:
        emu, _ = emulator_from_cond(args.cond_dir, args.scenarios, args.suffix)
        src_label = "cond files"
    else:
        emu, _ = emulator_from_raw(args.raw_dir, args.scenarios, args.hist_source,
                                   args.scen_version, args.hist_end,
                                   args.year_min, args.year_max)
        src_label = "raw input4MIPs"
    if not emu:
        return 1

    # ── FaIR side ───────────────────────────────────────────────────────────
    if not os.path.exists(args.fair_emissions):
        print(f"ERROR: {args.fair_emissions} not found. Run the FaIR experiment "
              f"first (run_scenarios.py).", file=sys.stderr)
        return 1
    df = pd.read_csv(args.fair_emissions, index_col=0)
    ycols = [c for c in df.columns if c != "Scenario"]
    fyears = np.array([float(c) for c in ycols])

    def fair_series(scen, specie):
        r = df[(df.index == specie) & (df.Scenario == scen)]
        if not len(r):
            return None
        s = pd.Series(r[ycols].values[0].astype(float), index=np.floor(fyears).astype(int))
        return s[(s.index >= args.year_min) & (s.index <= args.year_max)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    rows = []
    for sc in args.scenarios:
        if sc not in emu:
            continue
        fsc = DEFAULT_MAP.get(sc, sc)
        c = COLORS.get(sc, "#7f7f7f")
        e = emu[sc]
        e = e[(e.index >= args.year_min) & (e.index <= args.year_max)]
        ffi = fair_series(fsc, "CO2 FFI")
        afo = fair_series(fsc, "CO2 AFOLU")

        ax[0].plot(e.index, e.values, color=c, lw=2.0,
                   label=f"emulator {sc} ({src_label})")
        if ffi is not None:
            ax[0].plot(ffi.index, ffi.values, color=c, lw=1.4, ls="--",
                       label=f"FaIR {fsc} — CO2 FFI")
            if afo is not None:
                ax[0].plot(ffi.index, (ffi + afo).values, color=c, lw=1.0, ls=":",
                           alpha=0.7, label=f"FaIR {fsc} — FFI+AFOLU")

        ax[1].plot(e.index, e.cumsum().values, color=c, lw=2.0, label=f"emulator {sc}")
        if ffi is not None:
            ax[1].plot(ffi.index, ffi.cumsum().values, color=c, lw=1.4, ls="--",
                       label=f"FaIR {fsc} FFI")

        for yr in (2000, 2020, 2050, 2100):
            if yr in e.index and ffi is not None and yr in ffi.index:
                ev, fv = float(e.loc[yr]), float(ffi.loc[yr])
                rows.append(dict(scenario=sc, fair_scenario=fsc, year=yr,
                                 emulator_GtCO2_yr=round(ev, 3),
                                 fair_ffi_GtCO2_yr=round(fv, 3),
                                 ratio=round(ev / fv, 4) if fv else np.nan))

    ax[0].set_ylabel("CO$_2$ emissions, GtCO$_2$ yr$^{-1}$")
    ax[1].set_ylabel("Cumulative CO$_2$ since %d, GtCO$_2$" % args.year_min)
    for a in ax:
        a.set_xlabel("year"); a.grid(alpha=0.3)
        a.axhline(0, ls=":", color="k", lw=0.5)
        a.legend(fontsize=7)
    ax[0].set_title("(a) annual")
    ax[1].set_title("(b) cumulative")
    fig.suptitle("CO$_2$ driving the emulator vs driving FaIR — CMIP7 scenarios\n"
                 "gridded input4MIPs (fossil+industry, +aircraft) vs FaIR stylised pathways",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")

    if rows:
        tab = pd.DataFrame(rows)
        print("\nemulator vs FaIR CO2 FFI (GtCO2/yr), ratio = emulator/FaIR")
        print(tab.to_string(index=False))
        if args.csv:
            tab.to_csv(args.csv, index=False)
            print(f"\nwrote {args.csv}")
        bad = tab[(tab.ratio < 0.5) | (tab.ratio > 2.0)]
        if len(bad):
            print("\nWARNING: emulator/FaIR CO2 differs by >2x at some years — the "
                  "two are NOT on a common absolute scale, so their temperature "
                  "responses are not directly comparable. (The CMIP6-era cond "
                  "files had a ~5x inflated CO2 inventory; see cond_co2_units.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
