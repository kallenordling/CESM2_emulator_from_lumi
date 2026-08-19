#!/usr/bin/env python3
"""
Stage the CESM2 Single-Forcing archive into monthly training trees.

download_cesm_lens_sf.py fetches RAW CESM tseries files and does no aggregation,
so its output is not something ClimateDataset can read:

    sf/<ENS>/<VAR>/b.e21...CESM2-SF-<ENS>.<mmm>.cam.h0.<VAR>.<span>.nc

This turns that into the layout config_data_monthly.yaml expects:

    training_data_monthly/<TREFHT|PRECT>/<ENS>/<mmm>/chunk_<i>.nc

Two things have to happen on the way, and neither is in the downloader:

1. PRECT IS NOT AN ARCHIVED VARIABLE. CESM2 stores precipitation as PRECC
   (convective) and PRECL (large-scale), both m/s; total precipitation is their
   sum. get_data.py:93 does this for the AWS path — the same rule, and the same
   refusal to add components whose units disagree, applies here.

2. THE FIVE TIME SLICES ARE ONE RECORD. Each member ships as 1850-1899,
   1900-1949, 1950-1999, 2000-2014 and a separate SSP370 2015-2050 file. They
   concatenate into a contiguous 1850-2050 monthly axis, which is what the
   previous-target channel requires (it raises on an unevenly spaced axis).

Chunking matches the existing trees: `chunk_<i>.nc`, count set by --num-chunks,
written whole so a partial run leaves no half-written member behind.

Usage:
    python stage_sf_monthly.py --sf-dir  /scratch/project_2019839/emulator_data/sf \
                               --out-dir /scratch/project_2019839/emulator_data/training_data_monthly
    python stage_sf_monthly.py ... --ensemble GHG --members 001 002 --dry-run
"""
import argparse
import os
import re
import sys
from glob import glob

import xarray as xr

# PRECT is derived; TREFHT is archived as-is. Mirrors get_data.py's DERIVED map.
DERIVED = {"PRECT": ("PRECC", "PRECL")}
MEMBER_RE = re.compile(r"CESM2-SF-[A-Za-z]+(?:-SSP370)?\.(\d{3})\.cam\.h0\.")


def members_present(sf_dir, ens, var):
    """Member ids that have at least one file for this ensemble/variable."""
    found = set()
    for p in glob(os.path.join(sf_dir, ens, var, "*.nc")):
        m = MEMBER_RE.search(os.path.basename(p))
        if m:
            found.add(m.group(1))
    return found


def load_member(sf_dir, ens, var, member):
    """One member's full 1850-2050 monthly record for one ARCHIVED variable.

    Sorted by the span encoded in the filename rather than lexically: the
    SSP370 file breaks lexical order (its name carries an extra '-SSP370'), and
    concatenating out of order would silently produce a non-monotonic time axis.
    """
    pat = os.path.join(sf_dir, ens, var, f"*.{member}.cam.h0.{var}.*.nc")
    files = glob(pat)
    if not files:
        raise FileNotFoundError(f"{ens}/{var}/{member}: no files match {pat}")
    files.sort(key=lambda p: os.path.basename(p).rsplit(".", 2)[-2])
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 120})
    if var not in ds:
        raise KeyError(f"{ens}/{var}/{member}: {var} not in {files[0]}")
    return ds


def build_var(sf_dir, ens, var, member):
    """One member's record for VAR, deriving it from components when needed."""
    if var not in DERIVED:
        ds = load_member(sf_dir, ens, var, member)
        return ds[[var]]

    parts = DERIVED[var]
    das, units = [], set()
    for part in parts:
        ds = load_member(sf_dir, ens, part, member)
        units.add(ds[part].attrs.get("units", "?"))
        das.append(ds[part])
    if len(units) > 1:
        raise ValueError(f"{ens}/{member}/{var}: components disagree on units "
                         f"{units} — refusing to add them")
    total = sum(das[1:], das[0]).to_dataset(name=var)
    total[var].attrs.update(units=units.pop(),
                            long_name="Total precipitation rate (PRECC + PRECL)",
                            derived_from=" + ".join(parts))
    return total


def write_chunks(ds, out_dir, num_chunks):
    """Write chunk_<i>.nc, same convention as get_data.py:save_dataset."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    n = ds.sizes["time"]
    size = n // num_chunks
    parts, paths = [], []
    for i in range(num_chunks):
        lo = i * size
        hi = lo + size if i < num_chunks - 1 else None
        parts.append(ds.isel(time=slice(lo, hi)))
        paths.append(os.path.join(out_dir, f"chunk_{i}.nc"))
    xr.save_mfdataset(parts, paths, compute=True)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ensemble", nargs="+", default=["AAER", "GHG"])
    ap.add_argument("--variable", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--members", nargs="+",
                    help="member ids like 001 002 (default: every member that "
                         "has files for ALL required variables)")
    ap.add_argument("--num-chunks", type=int, default=40,
                    help="files per member (default 40, matching the existing trees)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave a member alone if its output dir already holds "
                         "num-chunks files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rc = 0
    for ens in args.ensemble:
        # Intersect across every source variable: a member missing one component
        # of PRECT cannot be staged, and finding that out mid-write leaves a
        # half-built tree behind.
        need = set()
        for var in args.variable:
            need.update(DERIVED.get(var, (var,)))
        common = None
        for src in sorted(need):
            have = members_present(args.sf_dir, ens, src)
            common = have if common is None else common & have
            print(f"[{ens}] {src}: {len(have)} members")
        common = sorted(common or [])
        if args.members:
            common = [m for m in common if m in args.members]
        print(f"[{ens}] staging {len(common)} members: "
              f"{', '.join(common) if common else '(none)'}")

        for member in common:
            for var in args.variable:
                out = os.path.join(args.out_dir, var, ens, member)
                if args.skip_existing and \
                        len(glob(os.path.join(out, "chunk_*.nc"))) == args.num_chunks:
                    print(f"  [skip] {var}/{ens}/{member}")
                    continue
                if args.dry_run:
                    print(f"  [dry-run] {var}/{ens}/{member} -> {out}")
                    continue
                try:
                    ds = build_var(args.sf_dir, ens, var, member)
                    n = write_chunks(ds, out, args.num_chunks)
                    t0 = str(ds.time.values[0])[:7]
                    t1 = str(ds.time.values[-1])[:7]
                    print(f"  [ok] {var}/{ens}/{member}: {n} months "
                          f"{t0}..{t1} -> {args.num_chunks} chunks")
                    ds.close()
                except Exception as e:
                    print(f"  [FAIL] {var}/{ens}/{member}: "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
                    rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
