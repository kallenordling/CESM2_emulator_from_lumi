#!/usr/bin/env python3
"""
Rebuild the SUL (sulfate/SO2) conditioning channel for the CMIP6 scenarios.

THE DEFECT
----------
The SUL channel steps DOWN 11.8% at the 2014->2015 splice, and the step is
almost entirely East China. Measured from raw input4MIPs (global anthro SO2,
Tg/yr, and the 20-45N/100-125E box):

    CEDS-2017 2014   global 111.66   E.China 35.45
    IAMC ssp*  2015  global  98.47   E.China 23.01     0.882 / 0.649
    CEDS-2025 2014   global  95.73   E.China 24.16
    CEDS-2025 2015   global  89.79   E.China 20.70     0.938 / 0.857

East China alone supplies 12.4 of the 13.2 Tg global drop; India RISES across
the junction. The cause is not sectors (both sides carry the same 8 sector ids
in the same order) and not the harmonisation (all three SSP files hold a
byte-identical 2015 field, i.e. one shared base year). It is that
CEDS-2017-05-18 is FLAT over China after 2011 -- E.China 33.7 / 35.2 / 35.3 /
36.8 / 35.5 Tg for 2010-2014 -- because that release extrapolated the post-2011
years and never saw the Chinese scrubber rollout. CEDS-2025 over the same years
falls 31.0 -> 24.2. The SSP 2015 base (23.0) sits near the modern value, so the
trajectory steps off a stale flat history onto a corrected base year.

About half the global step is real: CEDS-2025 spans the junction continuously
and declines 6.2% from 2014 to 2015. The other half is the CEDS-2017 China
overestimate being discarded in a single year.

THE FIX AND ITS TRADE
---------------------
Take the historical SUL from CEDS-CMIP-2025-04-18, which is continuous from
1750 to 2023 and captures the real Chinese decline. That removes the stale
plateau but leaves a small junction against the CEDS-2017-harmonised SSPs:
95.73 (2014) vs 98.47 (2015) = +2.9%, against -11.8% today, and with the
opposite sign. Pre-2005 the two vintages agree closely (2005: 122.56 vs
122.63 Tg), so only the last ~10 historical years move materially.

This is the OPPOSITE of the choice made for BC ([[bc_ceds_vintage_mismatch]]),
where history was moved ONTO CEDS-2017 to match the scenarios. BC's problem was
a pure vintage offset; SUL's is that CEDS-2017 itself is wrong over China.
Pass --hist-vintage ceds2017 to reproduce the shipped behaviour instead.

THE PROCEDURE
-------------
    1. historical annual SO2 (surface anthro, summed over sectors), NATIVE grid,
       area-integrated to Gt/yr per gridpoint exactly as make_aerosol_files.py
       does it, clipped to <= 2014
    2. scenario annual SO2, native grid, >= 2015 -- the ScenarioMIP files are
       DECADAL (2015, 2020, ...), so interpolate to annual FIRST
    3. concat. NO cumsum: SUL is an annual rate, unlike CO2
    4. xesmf bilinear periodic regrid to the 192x288 target grid
    5. inject SUL into copies of the existing cond files, leaving CO2 and BC
       untouched

Unlike the CO2 rebuild this reads the RAW input4MIPs files, not the
CO2/SO2_cumulative_Gt_per_gridpoint_*.nc intermediates, so its provenance does
not stop at whenever those were last written.

Usage:
    python data/rebuild_cmip6_sul_cond.py --data-dir DIR --raw-dir DIR --check
    python data/rebuild_cmip6_sul_cond.py --data-dir DIR --raw-dir DIR \
        --out-dir DIR --target <a cond file>
"""
import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.provenance import stamp  # noqa: E402

Y0, HIST_END, Y1 = 1850, 2014, 2100
R_EARTH = 6.371e6
SECONDS_PER_YEAR = 365.25 * 24 * 3600
KG_PER_GT = 1e12

# Raw input4MIPs globs. The hist entry is the whole point of this script; the
# scenario entries are the same files make_aerosol_files.py reads.
HIST_PATTERNS = {
    "ceds2025": "SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_*.nc",
    "ceds2017": "SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc",
}
SSP_PATTERNS = {
    "ssp370": "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc",
    "ssp126": "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc",
    "ssp245": "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc",
}

# Which cond file takes SUL from which spliced series, and over what years.
# Verified against the shipped files before writing this table:
#   aaer[1850:2014] == hist, aaer[2015:2050] == ssp370, ghg SUL constant and
#   equal to hist[1850], ramip SUL == ssp126[2015:2079].
# mode: "series" = take those years from the spliced series;
#       "pin"    = constant field, the series at PIN_YEAR, broadcast over all years.
COND = [
    # (filename,                                          scenario, mode)
    ("emissions_hist_only_timefixed_bc_co2fix.nc",        "ssp370", "series"),
    ("emissions_ssp370_only_timefixed_bc_co2fix.nc",      "ssp370", "series"),
    ("emissions_ssp126_only_timefixed_bc_co2fix.nc",      "ssp126", "series"),
    ("emissions_ssp245_only_timefixed_bc_co2fix.nc",      "ssp245", "series"),
    ("emissions_aaer_only_timefixed_bc_co2fix.nc",        "ssp370", "series"),
    ("emissions_ghg_only_timefixed_bc_co2fix.nc",         "ssp370", "pin"),
    ("emissions_ssp370co2_ssp126aer_bc_2015-2079_co2fix.nc", "ssp126", "series"),
]
PIN_YEAR = 1850


def cell_area(lat, lon):
    """Grid-cell area in m^2. Byte-for-byte the same formula as
    make_aerosol_files.py:compute_grid_cell_area -- the rebuilt channel has to
    land on the same scale as the CO2/BC channels beside it."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dlat = np.abs(np.diff(lat).mean())
    dlon = np.abs(np.diff(lon).mean())
    edges = np.deg2rad(np.clip(np.concatenate([
        [lat[0] - dlat / 2], (lat[:-1] + lat[1:]) / 2, [lat[-1] + dlat / 2]]), -90, 90))
    area = np.abs(np.sin(edges[1:]) - np.sin(edges[:-1])) * np.deg2rad(dlon) * R_EARTH ** 2
    return xr.DataArray(np.broadcast_to(area[:, None], (len(lat), len(lon))),
                        dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def native(raw_dir, pattern, lo, hi, label):
    """Annual SO2 on the native grid, Gt/yr per gridpoint, clipped to [lo, hi].

    Sum over sectors, annual MEAN of the rate (an annual sum of kg/m2/s is not
    a physical quantity), then area * seconds/year / 1e12."""
    files = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"{label}: no files matching\n  {os.path.join(raw_dir, pattern)}")
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal",
                           coords="minimal", compat="override")
    var = [v for v in ds.data_vars if "bnds" not in v and "bound" not in v][0]
    ds = ds.drop_vars([v for v in ds.variables if "bnds" in str(v) or "bound" in str(v)],
                      errors="ignore")
    da = ds[var].sum(dim="sector")
    da = da.groupby("time.year").mean()
    da = da.sel(year=(da.year >= lo) & (da.year <= hi))
    da = (da * cell_area(ds.lat.values, ds.lon.values) * SECONDS_PER_YEAR / KG_PER_GT)
    da = da.compute().rename("SUL")
    print(f"    [{label}] {len(files)} file(s), years "
          f"{int(da.year.values[0])}-{int(da.year.values[-1])}, "
          f"{float(da.isel(year=-1).sum()):.5f} Gt/yr at the end")
    return da


def to_annual(da, label):
    """Interpolate a decadal series onto every year. The ScenarioMIP SO2 files
    carry 2015, 2020, 2030 ... only; the year in between is simply absent, so
    without this the splice would drop 80 of 86 scenario years."""
    full = np.arange(int(da.year.values[0]), int(da.year.values[-1]) + 1)
    if len(full) != len(da.year):
        print(f"    [{label}] interpolating {len(da.year)} -> {len(full)} years "
              f"(decadal -> annual)")
        da = da.interp(year=full, method="linear")
    return da


def build_series(raw_dir, scenario, hist_vintage):
    """Spliced ANNUAL SUL 1850-2100 on the native grid. No cumsum."""
    hist = native(raw_dir, HIST_PATTERNS[hist_vintage], Y0, HIST_END, f"hist/{hist_vintage}")
    scen = to_annual(native(raw_dir, SSP_PATTERNS[scenario], HIST_END + 1, Y1, scenario),
                     scenario)
    hg = float(hist.sel(year=HIST_END).sum())
    sg = float(scen.sel(year=HIST_END + 1).sum())
    print(f"    junction {HIST_END}->{HIST_END + 1}: {hg:.5f} -> {sg:.5f} Gt/yr "
          f"({100 * (sg / hg - 1):+.2f}%)")
    return xr.concat([hist, scen], dim="year").sortby("year")


def report(raw_dir, scenarios, data_dir):
    """Diagnose without writing: the junction each vintage produces, native
    grid, against the junction the shipped cond files actually contain."""
    for vintage in ("ceds2017", "ceds2025"):
        print(f"\n=== hist vintage {vintage} ===")
        for sc in scenarios:
            print(f"  {sc}:")
            build_series(raw_dir, sc, vintage)
    print("\n=== junction in the shipped cond files (global sum, cond units) ===")
    p = os.path.join(data_dir, "emissions_aaer_only_timefixed_bc_co2fix.nc")
    if os.path.exists(p):
        ds = xr.open_dataset(p)
        g = ds["SUL"].sum(dim=("lat", "lon"))
        a, b = float(g.sel(year=2014)), float(g.sel(year=2015))
        print(f"  aaer (the only file spanning the junction): "
              f"{a:.6g} -> {b:.6g}  ({100 * (b / a - 1):+.2f}%)")
        ds.close()
    else:
        print(f"  (not found: {p})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True,
                    help="directory holding the raw input4MIPs SO2-em-anthro files")
    ap.add_argument("--data-dir", required=True,
                    help="directory holding the cond files to rebuild SUL into")
    ap.add_argument("--out-dir", default=None,
                    help="where to write (default: --data-dir; never overwrites the input)")
    ap.add_argument("--hist-vintage", default="ceds2025", choices=list(HIST_PATTERNS),
                    help="historical SO2 release. ceds2025 is the fix; ceds2017 "
                         "reproduces the shipped -11.8%% junction")
    ap.add_argument("--scenarios", nargs="+", default=list(SSP_PATTERNS),
                    choices=list(SSP_PATTERNS))
    ap.add_argument("--target", help="grid template (lat/lon only); required to write")
    ap.add_argument("--out-suffix", default="_sulfix",
                    help="written alongside the input, never over it")
    ap.add_argument("--check", action="store_true", help="diagnose only, write nothing")
    args = ap.parse_args()

    print(f"[sul] raw-dir  {args.raw_dir}")
    print(f"[sul] data-dir {args.data_dir}")
    print(f"[sul] vintage  {args.hist_vintage}")

    if args.check:
        report(args.raw_dir, args.scenarios, args.data_dir)
        return 0

    if not args.target:
        ap.error("--target is required to write (a cond file whose grid to match)")
    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    import xesmf as xe
    target = xr.open_dataset(args.target)
    tgrid = xr.Dataset({"lat": target["lat"], "lon": target["lon"]})

    # One regridded series per scenario, reused by every cond file that needs it.
    regridded = {}
    for sc in args.scenarios:
        print(f"\n=== {sc} ===")
        ds = build_series(args.raw_dir, sc, args.hist_vintage).to_dataset(name="SUL")
        # Match the CO2/BC paths' convention fix so all three channels land on
        # identical geography (rebuild_cmip6_co2_cond.py:236, concat_and_regrid.py:201).
        if float(ds.lon.min()) < 0:
            ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
        rg = xe.Regridder(ds, tgrid, method="bilinear", periodic=True)
        out = rg(ds["SUL"], keep_attrs=True)
        # Unmapped pole points -> 0, matching the BC path (concat_and_regrid.py:233).
        regridded[sc] = out.fillna(0.0).compute()
        print(f"  regridded to {regridded[sc].shape[1]}x{regridded[sc].shape[2]}, "
              f"global {float(regridded[sc].sel(year=HIST_END).sum()):.6g} -> "
              f"{float(regridded[sc].sel(year=HIST_END + 1).sum()):.6g} at the junction")

    print()
    for fname, sc, mode in COND:
        if sc not in regridded:
            print(f"  skip {fname} (needs {sc})")
            continue
        src = os.path.join(args.data_dir, fname)
        if not os.path.exists(src):
            print(f"  skip {fname} (absent)")
            continue
        dst = os.path.join(out_dir, fname.replace(".nc", f"{args.out_suffix}.nc"))
        cd = xr.open_dataset(src)
        c = "year" if "year" in cd.coords else "time"
        yrs = np.asarray(cd[c].values).astype(int)
        series = regridded[sc]
        if mode == "pin":
            new = np.broadcast_to(series.sel(year=PIN_YEAR).values, cd["SUL"].shape).copy()
        else:
            pos = {int(v): i for i, v in enumerate(np.asarray(series.year.values).astype(int))}
            missing = [int(y) for y in yrs if int(y) not in pos]
            assert not missing, f"{fname}: years absent from the series: {missing[:5]}"
            new = series.isel(year=[pos[int(y)] for y in yrs]).values
        assert new.shape == cd["SUL"].shape, f"{fname}: {new.shape} != {cd['SUL'].shape}"
        old_g = cd["SUL"].sum(dim=("lat", "lon")).values
        cd["SUL"] = xr.DataArray(new, dims=cd["SUL"].dims, coords=cd["SUL"].coords)
        new_g = cd["SUL"].sum(dim=("lat", "lon")).values
        stamp(cd, __file__,
              sources=[(os.path.join(args.raw_dir, HIST_PATTERNS[args.hist_vintage]), None),
                       (os.path.join(args.raw_dir, SSP_PATTERNS[sc]), None),
                       (src, None)],
              extra={"cond_channel_rebuilt": "SUL only",
                     "cond_channels_untouched": "CO2, BC (copied verbatim from "
                                                "the input cond file)",
                     "sul_hist_vintage": args.hist_vintage,
                     "sul_scenario": sc,
                     "sul_mode": mode,
                     "splice_hist_end_year": HIST_END,
                     "cumsum": "none (SUL is an annual rate)",
                     "scenario_interp": "decadal -> annual",
                     "global_first_year_before": f"{old_g[0]:.6g}",
                     "global_first_year_after": f"{new_g[0]:.6g}",
                     "global_last_year_before": f"{old_g[-1]:.6g}",
                     "global_last_year_after": f"{new_g[-1]:.6g}"},
              note="SUL conditioning rebuilt because the shipped channel steps "
                   "-11.8% at 2015, almost entirely over East China, where "
                   "CEDS-2017-05-18 is flat after 2011 and the SSP 2015 base "
                   "year already carries the real decline. CO2 and BC are NOT "
                   "rebuilt here: they are inherited from the input cond file "
                   "and carry its provenance, not this one.")
        cd.to_netcdf(dst)
        cd.close()
        print(f"  {fname}")
        print(f"    global SUL {old_g[0]:.6g} -> {new_g[0]:.6g} (first year), "
              f"{old_g[-1]:.6g} -> {new_g[-1]:.6g} (last)")
        print(f"    wrote {dst}")
    print("\n[sul] CO2 and BC were not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
