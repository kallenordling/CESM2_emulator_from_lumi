#!/usr/bin/env python3
"""
Rebuild the cumulative CO2 conditioning channel for the CMIP6 SSP scenarios.

THE DEFECT
----------
emissions_ssp370_only_timefixed*.nc accumulates CO2 at roughly TWICE the correct
rate. Measured in the shipped cond files:

    historical 2014 step   +7.25     (per year, cond units)
    ssp126     2016 step   +7.847
    ssp245     2016 step   +7.87
    ssp370     2016 step  +15.28     <- ~1.95x its siblings

The raw ESGF data says this cannot be right. All three SSPs branch from an
IDENTICAL 2015 state — 34.900 Gt CO2/yr surface anthro plus 0.764 aircraft,
byte-identical across the three files — and the historical hands off at
34.811 Gt/yr in 2014, a continuous 2.4% rise. The native-grid intermediate is
also correct: CO2_cumulative_Gt_per_gridpoint_ssp370.nc holds 35.66 at 2015,
matching the raw total. Scaled by the ~4.7x extensive-regrid deflation that
gives 7.59 per year — which is what ssp126 and ssp245 show and what ssp370
should show.

So the doubling is introduced between the native intermediate and the cond
file, by code that NO LONGER RUNS: concat_and_regrid.py lines 76-153 (the CO2
splice, the hist-endpoint addition and the cumsum) sit inside a triple-quoted
string, and make_co2_files.py:200 has its cumsum commented out. The shipped
ssp370 file is a legacy artifact of an earlier pipeline, which is precisely why
nothing re-exposed the bug.

WHY IT MATTERS
--------------
CO2 is the dominant forcing and the emulator's response is calibrated against
cumulative CO2. With ssp370's axis stretched ~2x, the model learns roughly half
the true warming-per-unit-CO2 FOR THAT SCENARIO while ssp126/ssp245 sit on the
correct scale. TCRE cannot see it — it regresses temperature against cumulative
CO2 from the same file, so the error is self-consistent — but any cross-scenario
comparison inherits it. Note ssp370 scores +1% on TCRE while ssp126 is -26%.

THE PROCEDURE (from make_cmip7_cond.py, the reference implementation)
---------------------------------------------------------------------
    1. historical annual CO2 (surface anthro + aircraft), native grid, <= 2014
    2. scenario annual CO2, native grid, >= 2015 — the ScenarioMIP files are
       DECADAL, so interpolate to annual FIRST
    3. concat, then cumsum over the FULL spliced record from 1850
       (the cumsum must come after the splice, never per-segment)
    4. xesmf bilinear periodic regrid to the 192x288 target grid
    5. inject CO2 into copies of the existing *_bc.nc cond files, leaving SUL
       and BC untouched

Step 3 is where the old path went wrong: cumsum before/around the splice, or a
decadal series summed as if annual, doubles the rate exactly this way.

Usage:
    python data/rebuild_cmip6_co2_cond.py --check          # diagnose, write nothing
    python data/rebuild_cmip6_co2_cond.py --data-dir DIR
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Y0, HIST_END, Y1 = 1850, 2014, 2100

# Which cond file each scenario's CO2 is written into. Mirrors
# concat_and_regrid.py's BC mapping so the two channels stay consistent.
COND = {
    "ssp370": ("emissions_ssp370_only_timefixed_bc.nc",
               "emissions_ssp370_only_timefixed_bc.nc"),
    "ssp245": ("emissions_ssp245_only_timefixed_bc.nc",
               "emissions_ssp245_only_timefixed_bc.nc"),
    "ssp126": ("emissions_ssp126_only_timefixed_co2fix_bc.nc",
               "emissions_ssp126_only_timefixed_co2fix_bc.nc"),
}


def at_year(da, y):
    """Select a single year positionally.

    The source files carry a proper `year` index, but it does not survive every
    concat/interp/sortby combination here — .sel(year=2016) failed with
    "KeyError: no index found for coordinate 'year'" on the spliced series even
    though both inputs were indexed. Looking the value up in the coordinate
    array sidesteps the question entirely and cannot silently mis-select.
    """
    yrs = np.asarray(da["year"].values, dtype=int)
    idx = np.where(yrs == int(y))[0]
    if idx.size == 0:
        raise KeyError(f"year {y} not present (range {yrs.min()}-{yrs.max()})")
    return da.isel(year=int(idx[0]))


def at_years(da, wanted):
    """Positional equivalent of .sel(year=[...]) — same reasoning as at_year."""
    yrs = np.asarray(da["year"].values, dtype=int)
    pos = {int(v): i for i, v in enumerate(yrs)}
    missing = [int(w) for w in wanted if int(w) not in pos]
    if missing:
        raise KeyError(f"years absent from the rebuilt series: {missing[:5]}"
                       f"{' …' if len(missing) > 5 else ''}")
    return da.isel(year=[pos[int(w)] for w in wanted])


def native(data_dir, exp):
    """Annual CO2 on the native grid. Despite the filename these hold ANNUAL
    rates, not a cumulative series — make_co2_files.py:200 has its cumsum
    commented out. Verified: the ssp370 file's 2015 value (35.66) equals the
    raw ESGF annual total, and the historical series is non-monotone, which a
    cumulative series could not be."""
    p = os.path.join(data_dir, f"CO2_cumulative_Gt_per_gridpoint_{exp}.nc")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p}\n  Regenerate with: python data/make_co2_files.py {exp}")
    da = xr.open_dataset(p)["CO2"]
    # Some of these files carry `year` as a plain variable rather than a
    # dimension coordinate with an index. Everything downstream (.sel(year=...),
    # concat, interp) then fails with
    #     KeyError: no index found for coordinate 'year'
    # so normalise it once, here, instead of guarding every call site.
    if "year" in da.dims and "year" not in da.indexes:
        da = da.assign_coords(year=("year", np.asarray(da["year"].values, dtype=int)))
    return da.sortby("year")


def to_annual(da, lo, hi, label=""):
    """Clip to [lo, hi] and interpolate a decadal series onto every year.

    ScenarioMIP CO2 is published decadally (ssp370 has 10 samples for 86 years).
    Interpolating the RATE and cumsumming afterwards is correct; summing decadal
    samples as though each represented one year is not, and inflates the total
    by roughly the sampling interval.
    """
    da = da.sel(year=(da.year >= lo) & (da.year <= hi))
    full = np.arange(int(da.year.values[0]), int(da.year.values[-1]) + 1)
    if len(full) != len(da.year):
        print(f"    [{label}] interpolating {len(da.year)} -> {len(full)} years "
              f"(decadal -> annual)")
        da = da.interp(year=full, method="linear")
    return da


def build_series(data_dir, scenario):
    """Spliced ANNUAL CO2 1850-2100 on the native grid, then cumulative."""
    print(f"  [{scenario}] historical …")
    hist = to_annual(native(data_dir, "hist"), Y0, HIST_END, "hist")
    print(f"  [{scenario}] scenario …")
    scen = to_annual(native(data_dir, scenario), HIST_END + 1, Y1, scenario)

    hg = float(at_year(hist, HIST_END).sum())
    sg = float(at_year(scen, HIST_END + 1).sum())
    print(f"    junction {HIST_END}->{HIST_END+1}: {hg:.4g} -> {sg:.4g} Gt/yr "
          f"({100*(sg/hg-1):+.2f}%)")

    series = xr.concat([hist, scen], dim="year").sortby("year")
    # THE CUMSUM MUST BE HERE: once, over the whole spliced record. Doing it
    # per-segment and adding an offset is what the dead code did, and is the
    # likeliest origin of the ssp370 doubling.
    cum = series.cumsum(dim="year", keep_attrs=True)
    cum["year"] = series["year"]
    return series, cum


def report(data_dir, scenarios):
    """Diagnose without writing: what the corrected annual step would be,
    against what the shipped cond files contain."""
    print("\n=== corrected vs shipped (global, cond units after /deflation) ===")
    print(f"  {'scenario':9s} {'native 2016 step':>17s} {'shipped 2016 step':>18s} "
          f"{'shipped/expected':>17s}")
    DEFLATION = 4.7
    for sc in scenarios:
        series, _ = build_series(data_dir, sc)
        step = float(at_year(series, 2016).sum())
        exp_step = step / DEFLATION
        cond = os.path.join(data_dir, COND[sc][0])
        if os.path.exists(cond):
            ds = xr.open_dataset(cond)
            c = "year" if "year" in ds.coords else "time"
            g = ds["CO2"].sum(dim=("lat", "lon"))
            got = float(g.sel({c: 2016}) - g.sel({c: 2015}))
            ds.close()
            print(f"  {sc:9s} {step:17.4g} {got:18.4g} {got/exp_step:17.2f}")
        else:
            print(f"  {sc:9s} {step:17.4g} {'(no cond file)':>18s}")
    print(f"\n  deflation assumed {DEFLATION}x (extensive regrid). A ratio near "
          f"1.0 is correct;\n  near 2.0 is the doubling.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="directory holding CO2_cumulative_Gt_per_gridpoint_*.nc "
                         "and the cond files")
    ap.add_argument("--scenarios", nargs="+",
                    default=["ssp370", "ssp126", "ssp245"], choices=list(COND))
    ap.add_argument("--target", help="grid template (lat/lon only); required to write")
    ap.add_argument("--out-suffix", default="_co2fix",
                    help="written alongside the input, never over it")
    ap.add_argument("--check", action="store_true",
                    help="diagnose only, write nothing")
    args = ap.parse_args()

    if args.data_dir is None:
        try:
            import lumi_paths as L
            args.data_dir = L.DATA
        except Exception:
            ap.error("--data-dir is required off-cluster")

    print(f"[co2] data-dir  {args.data_dir}")
    print(f"[co2] scenarios {', '.join(args.scenarios)}")

    if args.check:
        report(args.data_dir, args.scenarios)
        return 0

    if not args.target:
        ap.error("--target is required to write (a cond file whose grid to match)")

    import xesmf as xe
    target = xr.open_dataset(args.target)
    tgrid = xr.Dataset({"lat": target["lat"], "lon": target["lon"]})

    for sc in args.scenarios:
        print(f"\n=== {sc} ===")
        _, cum = build_series(args.data_dir, sc)
        ds = cum.to_dataset(name="CO2")
        # Match the BC path's convention fix (concat_and_regrid.py:201-206) so
        # both channels land on identical geography.
        if float(ds.lon.min()) < 0:
            ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
        rg = xe.Regridder(ds, tgrid, method="bilinear", periodic=True)
        out = rg(ds["CO2"])

        src = os.path.join(args.data_dir, COND[sc][0])
        dst = os.path.join(args.data_dir,
                           COND[sc][1].replace(".nc", f"{args.out_suffix}.nc"))
        cd = xr.open_dataset(src)
        c = "year" if "year" in cd.coords else "time"
        yrs = cd[c].values.astype(int)
        sel = at_years(out, yrs).values
        assert sel.shape == cd["CO2"].shape, f"{sel.shape} != {cd['CO2'].shape}"
        old = float(cd["CO2"].sum(dim=("lat", "lon")).isel({c: 1})
                    - cd["CO2"].sum(dim=("lat", "lon")).isel({c: 0}))
        cd["CO2"] = xr.DataArray(sel, dims=cd["CO2"].dims, coords=cd["CO2"].coords)
        new = float(cd["CO2"].sum(dim=("lat", "lon")).isel({c: 1})
                    - cd["CO2"].sum(dim=("lat", "lon")).isel({c: 0}))
        cd.to_netcdf(dst)
        cd.close()
        print(f"  first annual step: {old:.4g} -> {new:.4g}  "
              f"(factor {new/old:.3f})")
        print(f"  wrote {dst}")
    print("\n[co2] SUL and BC were not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
