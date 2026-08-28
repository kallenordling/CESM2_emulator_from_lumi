#!/usr/bin/env python3
"""Repair NaN years in a staged CESM2-LE member from the AWS source.

WHY
---
Staging can drop whole years silently. LE2-1231.012 has 1931-1939 written as
100% NaN in TREFHT/hist — nine entire fields — while the same member's PRECT is
clean and every other member's TREFHT is clean. Averaging that into a reference
ensemble drags the mean and inflates the spread exactly where a bias panel is
read, which is why paper_fig_timeseries.py carries qc_ensemble() to mask it.
Masking is a workaround; this fixes the data.

The source is fine: r12i1231p1f2 over 1931-1939 has 0% NaN at AWS. So the fix is
to re-fetch those years and write them back.

WHAT IT DOES
------------
Surgical by default: it opens only the chunk files that contain bad years and
replaces the values for those years, leaving the file's structure, coordinates,
chunking and every good year untouched. A whole-member rewrite would change
chunk boundaries and risk far more than it repairs.

MEMBER NAMING is the trap. The tree uses LE2-<seed>.<nnn>; the AWS catalog uses
r<n>i<seed>p1f<v>, split across TWO forcing variants:

    nnn 001-010  ->  forcing_variant "cmip6",  r<n>i<seed>p1f1
    nnn 011-020  ->  forcing_variant "smbb",   r<n>i<seed>p1f2

so LE2-1231.012 is an smbb member and cannot be found in the cmip6 catalog at
all. Getting this wrong yields "not all values found in index 'member_id'".

Usage
-----
    # report only
    python scripts/refetch_member.py --member LE2-1231.012 --variable TREFHT \\
        --scenario hist --tree-root /path/to/training_data --dry-run

    # repair, in two trees at once
    python scripts/refetch_member.py --member LE2-1231.012 --variable TREFHT \\
        --scenario hist --tree-root /path/a/training_data /path/b/training_data
"""

import argparse
import glob
import os
import re
import sys
import tempfile

import numpy as np
import xarray as xr

SCEN_KEY = {"hist": "historical", "ssp370": "ssp370"}


def parse_member(name: str):
    """LE2-<seed>.<nnn> -> (catalog member_id, forcing_variant)."""
    m = re.fullmatch(r"LE2-(\d+)\.(\d+)", name)
    if not m:
        return name, None                      # already a catalog id
    seed, nnn = m.group(1), int(m.group(2))
    if 1 <= nnn <= 10:
        return f"r{nnn}i{seed}p1f1", "cmip6"
    if 11 <= nnn <= 20:
        return f"r{nnn}i{seed}p1f2", "smbb"
    raise SystemExit(f"member number {nnn} outside the documented 1-20 range")


def chunk_files(d):
    return sorted(glob.glob(os.path.join(d, "chunk_*.nc")),
                  key=lambda f: int(os.path.basename(f)[len("chunk_"):-3]))


def years_of(ds, var):
    dim = ds[var].dims[0]
    if dim == "year":
        return dim, [int(v) for v in ds[dim].values]
    return dim, [int(str(v)[:4]) for v in ds[dim].values]


def scan(member_dir, var, force_years=()):
    """{chunk path: [years to replace]}.

    Detects years that are entirely NaN. That catches dropped fields but NOT a
    year that is present and wrong: LE2-1231.012 also carries 1930 = 286.02 K,
    1.3 K below its neighbours (~10 sigma), which is finite and so invisible to
    a NaN test. Pass such years explicitly in `force_years`.
    """
    bad = {}
    for f in chunk_files(member_dir):
        with xr.open_dataset(f) as ds:
            dim, yrs = years_of(ds, var)
            a = ds[var].values
        for i, y in enumerate(yrs):
            if not np.isfinite(a[i]).all() or y in force_years:
                bad.setdefault(f, []).append(y)
    return bad


def fetch_annual(var, member, variant, scenario):
    """Annual means for one member, straight from the AWS catalog."""
    import intake
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    cat = intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/"
        "intake-catalogs/aws-cesm2-le.json")
    sub = cat.search(variable=var, frequency="monthly",
                     forcing_variant=variant)
    dsets = sub.to_dataset_dict(storage_options={"anon": True},
                                progressbar=False)
    key = f"atm.{SCEN_KEY[scenario]}.monthly.{variant}"
    if key not in dsets:
        raise SystemExit(f"{key} not in catalog: {sorted(dsets)}")
    da = dsets[key][var].sel(member_id=member)
    # Same aggregation as get_data.py: monthly -> annual mean.
    return da.groupby("time.year").mean().load()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", required=True, help="e.g. LE2-1231.012")
    ap.add_argument("--variable", default="TREFHT")
    ap.add_argument("--scenario", default="hist", choices=sorted(SCEN_KEY))
    ap.add_argument("--tree-root", nargs="+", required=True,
                    help="one or more training_data roots to repair")
    ap.add_argument("--years", nargs="+", type=int, default=[],
                    help="replace these years as well, even though they are "
                         "finite — for values that are present but wrong, "
                         "which a NaN scan cannot see")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    member_id, variant = parse_member(args.member)
    print(f"[member] {args.member} -> {member_id} (forcing_variant={variant})")

    targets = {}
    for root in args.tree_root:
        d = os.path.join(root, args.variable, args.scenario, args.member)
        if not os.path.isdir(d):
            print(f"[skip] {d} does not exist")
            continue
        bad = scan(d, args.variable, set(args.years))
        n = sum(len(v) for v in bad.values())
        print(f"[scan] {d}\n       {n} bad year(s) in {len(bad)} chunk(s): "
              f"{sorted(y for v in bad.values() for y in v)}")
        if bad:
            targets[d] = bad
    if not targets:
        print("nothing to repair")
        return 0
    if args.dry_run:
        print("\n--dry-run: stopping before any download or write")
        return 0

    need = sorted({y for bad in targets.values()
                   for ys in bad.values() for y in ys})
    print(f"\n[fetch] {args.variable} {args.scenario} {member_id} "
          f"for {need[0]}-{need[-1]} …")
    src = fetch_annual(args.variable, member_id, variant, args.scenario)
    have = {int(v) for v in src.year.values}
    missing = [y for y in need if y not in have]
    if missing:
        raise SystemExit(f"source lacks {missing} — aborting, nothing written")
    src_nan = {y: float(np.isnan(src.sel(year=y).values).mean()) for y in need}
    if any(v > 0 for v in src_nan.values()):
        raise SystemExit(f"source itself has NaN: {src_nan} — aborting")
    print(f"[fetch] ok, {len(need)} year(s), 0% NaN at source")

    for d, bad in targets.items():
        print(f"\n[write] {d}")
        for f, yrs in bad.items():
            with xr.open_dataset(f) as ds:
                ds = ds.load()
            dim, all_yrs = years_of(ds, args.variable)
            arr = ds[args.variable].values
            for y in yrs:
                i = all_yrs.index(y)
                arr[i] = src.sel(year=y).values
                print(f"        {os.path.basename(f)}  year {y} replaced "
                      f"(mean {float(np.nanmean(arr[i])):.3f})")
            ds[args.variable].values = arr
            # Write beside the target and rename, so an interrupted write
            # cannot leave a half-written chunk in the tree.
            tmp = tempfile.NamedTemporaryFile(
                dir=os.path.dirname(f), suffix=".tmp.nc", delete=False)
            tmp.close()
            ds.to_netcdf(tmp.name)
            ds.close()
            os.replace(tmp.name, f)

    print("\n[verify]")
    ok = True
    for d in targets:
        left = scan(d, args.variable)   # NaN-only: forced years are
                                        # finite by construction
        n = sum(len(v) for v in left.values())
        print(f"  {d}: {n} bad year(s) remaining")
        ok &= (n == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
