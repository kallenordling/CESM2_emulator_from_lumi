#!/usr/bin/env python3
"""Aggregate flat CMIP6 monthly files into the annual multi-member reference
file the eval and the paper figures read.

WHY THIS EXISTS
---------------
`cmip6/ssp126.nc` and `cmip6/ssp245.nc` — the CESM2 temperature reference for
the out-of-training scenarios — were NOT produced by anything in this repo.
Their global attributes carry `status: ...created; by gcs.cmip6.ldeo@gmail.com`,
i.e. they came out of the Pangeo Google-Cloud CMIP6 archive. That left no way
to build the same thing for a variable those files do not contain, and
`cmip6/` has no precipitation at all, so the unseen scenarios could only ever
get emulator-only precip panels.

This script closes that gap from the ESGF downloads instead:

    download_cmip6_cesm2.py --experiment ssp126 --variables pr \
        --members r4i1p1f1 r10i1p1f1 r11i1p1f1
    scripts/build_cmip6_annual_ref.py --experiment ssp126 --variable pr

OUTPUT LAYOUT — matched to the existing files on purpose
--------------------------------------------------------
    dims   (year, member, lat, lon)
    coords year (int), member (str), lat, lon
    var    <variable_id>, native CMIP6 units, attrs preserved

`year` is a plain integer coordinate, not a time axis: the readers
(`paper_fig_maps.cesm_fields_cmip6`, `paper_fig_timeseries.read_cmip6_ensemble`,
eval_aero's reference loader) all index by calendar year.

UNITS ARE NOT CONVERTED. `pr` stays kg m-2 s-1 as CMIP6 defines it; the figure
scripts hold the conversion tables and multiply by 86400 to reach mm/day. Baking
a conversion in here would leave a file whose `units` attribute lies about its
own contents.

ANNUAL MEAN WEIGHTING — defaults to matching the shipped files, not to the
better estimator. Verified 2026-08-13 against `cmip6/ssp126.nc`: a plain
unweighted mean of the 12 monthly values reproduces it to 1.2e-5 K (float32
rounding), while day-weighting by month length differs by 0.035 K mean /
0.19 K max. Day-weighting is the more correct annual mean, but the temperature
reference already in use is unweighted, and the paper reads temperature and
precipitation biases side by side — a reference built one way and another built
the other way would put a ~0.03 K inconsistency between them, ~15% of the
ssp126/ssp245 temperature bias being reported. Pass --weighting days to opt in,
and then rebuild BOTH variables.
"""
import os
import re
import glob
import argparse

import numpy as np
import xarray as xr

try:
    import lumi_paths as L
    CMIP6_DIR = f"{L.DATA}/cmip6"
except Exception:                      # runnable off-LUMI with --cmip6-dir
    CMIP6_DIR = "cmip6"

MEMBER_RE = re.compile(r"_(r\d+i\d+p\d+f\d+)_")


def annual_mean(da: xr.DataArray, weighting: str) -> xr.DataArray:
    """Annual mean of a monthly series, indexed by integer year.

    weighting="none" reproduces the shipped GCS-derived reference exactly;
    "days" weights each month by its length (see the module docstring).
    """
    yr = da["time"].dt.year
    if weighting == "none":
        return da.groupby(yr).mean("time")
    days = da["time"].dt.days_in_month
    return ((da * days).groupby(yr).sum("time", skipna=False)
            / days.groupby(yr).sum("time"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmip6-dir", default=CMIP6_DIR)
    ap.add_argument("--model", default="CESM2")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--variable", default="pr")
    ap.add_argument("--table", default="Amon")
    ap.add_argument("--members", nargs="+", default=None,
                    help="restrict to these member_ids (default: all found)")
    ap.add_argument("--weighting", choices=["none", "days"], default="none",
                    help="monthly->annual weighting; 'none' matches the shipped "
                         "cmip6/<exp>.nc exactly, 'days' weights by month length")
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2100)
    ap.add_argument("--out", default=None,
                    help="default <cmip6-dir>/<experiment>_<variable>.nc, or "
                         "<experiment>.nc for tas (the existing convention)")
    args = ap.parse_args()

    if args.out is None:
        stem = (args.experiment if args.variable == "tas"
                else f"{args.experiment}_{args.variable}")
        args.out = os.path.join(args.cmip6_dir, f"{stem}.nc")

    # RECURSIVE on purpose: download_cmip6_cesm2.py writes
    # <outdir>/<experiment>/<variable>/<member>/<file>, while the legacy files
    # that came from the Google-Cloud archive sit flat in cmip6/. `**` with
    # recursive=True matches zero directories too, so one pattern covers both
    # and neither layout has to be reorganised.
    name = f"{args.variable}_{args.table}_{args.model}_{args.experiment}_r*_gn_*.nc"
    pat = os.path.join(args.cmip6_dir, "**", name)
    files = sorted(set(glob.glob(pat, recursive=True)))
    if not files:
        return _fail(f"no files match {pat}\n"
                     f"Download them first:\n"
                     f"  python download_cmip6_cesm2.py --experiment "
                     f"{args.experiment} --variables {args.variable}")

    by_member = {}
    for f in files:
        m = MEMBER_RE.search(os.path.basename(f))
        if not m:
            print(f"  [skip] cannot parse member: {os.path.basename(f)}")
            continue
        by_member.setdefault(m.group(1), []).append(f)

    wanted = sorted(by_member) if args.members is None else list(args.members)
    missing = [m for m in wanted if m not in by_member]
    if missing:
        return _fail(f"requested members with no files: {missing}\n"
                     f"available: {sorted(by_member)}")

    das, kept, var_attrs = [], [], None
    for mem in wanted:
        ds = xr.open_mfdataset(sorted(by_member[mem]), combine="by_coords",
                               use_cftime=True)
        if args.variable not in ds:
            return _fail(f"{mem}: no {args.variable!r} in "
                         f"{[os.path.basename(p) for p in by_member[mem]]}")
        a = annual_mean(ds[args.variable], args.weighting)
        a = a.sel(year=slice(args.start, args.end))
        yrs = a["year"].values
        # A short member would silently truncate the whole ensemble on the
        # concat below, so it is dropped loudly instead.
        if yrs.min() > args.start or yrs.max() < args.end:
            print(f"  [skip partial] {mem}: {int(yrs.min())}-{int(yrs.max())} "
                  f"(need {args.start}-{args.end})")
            ds.close()
            continue
        var_attrs = var_attrs or dict(ds[args.variable].attrs)
        das.append(a.compute())
        kept.append(mem)
        print(f"  [{len(kept)}] {mem}: {int(yrs.min())}-{int(yrs.max())} "
              f"({len(yrs)} yr)")
        ds.close()

    if not das:
        return _fail("no full-coverage members — nothing to write")

    out = xr.concat(das, dim="member").assign_coords(member=("member", kept))
    out = out.transpose("year", "member", "lat", "lon")
    out.attrs = var_attrs or {}
    ds_out = out.to_dataset(name=args.variable)
    ds_out.attrs.update(
        experiment_id=args.experiment, source_id=args.model,
        table_id=args.table, variable_id=args.variable,
        frequency="yr",
        history=f"annual means (weighting={args.weighting}) of ESGF monthly "
                f"files, {len(kept)} members, built by "
                f"scripts/build_cmip6_annual_ref.py")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    ds_out.to_netcdf(args.out)
    print(f"\nwrote {args.out}")
    print(f"  {args.variable} {dict(ds_out.sizes)}  units="
          f"{var_attrs.get('units') if var_attrs else '?'}  members={kept}")
    return 0


def _fail(msg: str) -> int:
    import sys
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
