#!/usr/bin/env python3
"""Build a per-realization symlink directory of CESM2 members for one variable,
for the eval's multi-member reference loader.

eval_aero.py loads the ssp126 CESM2 reference as ``data_dir/<realization>/*.nc``
(open_mfdataset combine="by_coords", then resample to annual, then year-
intersect across members). This script scans the flat ESGF downloads in the
cmip6 dir, groups them by member, keeps only members with FULL time coverage
(start..end, default 2015-2100), and symlinks each member's files into
``<out_name>/<member>/``.

Partial members are skipped on purpose: the loader aligns members by year
intersection, so one short member (e.g. r11 = 2065-2100) would truncate the
whole ensemble and drop the low-forcing early period.

Run ON LUMI (so the symlink targets resolve to real /scratch paths):
    python build_ssp126_ensemble.py
    python build_ssp126_ensemble.py --start 2015 --end 2100   # explicit window
    python build_ssp126_ensemble.py --experiment ssp245 --variable pr

The paper figures do not read these directories — they read the single
aggregated `cmip6/<experiment>[_<variable>].nc`. Build that with
`scripts/build_cmip6_annual_ref.py` after downloading.

Then point eval_aero.py's ssp126 experiment at the directory:
    data_dir     = os.path.join(SCRATCH, "cmip6", "CESM2_ssp126_ens")
    realizations = [<members printed below>]
"""
import lumi_paths as L
import os
import re
import glob
import argparse
from collections import defaultdict

CMIP6_DIR = f"{L.DATA}/cmip6"
MEMBER_RE = re.compile(r"_(r\d+i\d+p\d+f\d+)_")
RANGE_RE = re.compile(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmip6-dir", default=CMIP6_DIR)
    ap.add_argument("--model", default="CESM2")
    ap.add_argument("--experiment", default="ssp126")
    ap.add_argument("--variable", default="tas",
                    help="CMIP6 variable_id to group, e.g. tas or pr")
    ap.add_argument("--table", default="Amon", help="CMIP6 table_id")
    ap.add_argument("--out-name", default=None,
                    help="ensemble dir created under --cmip6-dir "
                         "(default CESM2_<experiment>_ens for tas, "
                         "CESM2_<experiment>_<variable>_ens otherwise)")
    ap.add_argument("--start", type=int, default=2015, help="required first year")
    ap.add_argument("--end", type=int, default=2100, help="required last year")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.out_name is None:
        # tas keeps the historical directory name so existing eval_aero.py
        # data_dir settings and already-built ensembles stay valid.
        args.out_name = (f"CESM2_{args.experiment}_ens" if args.variable == "tas"
                         else f"CESM2_{args.experiment}_{args.variable}_ens")

    # Recursive: download_cmip6_cesm2.py writes a nested
    # <experiment>/<variable>/<member>/ tree while the older files sit flat in
    # cmip6/. `**` with recursive=True matches zero directories, so one pattern
    # handles both.
    name = f"{args.variable}_{args.table}_{args.model}_{args.experiment}_r*_gn_*.nc"
    pat = os.path.join(args.cmip6_dir, "**", name)
    files = sorted(set(glob.glob(pat, recursive=True)))
    if not files:
        raise SystemExit(f"no files match {pat}")

    # Group files by member, track combined year coverage from filenames.
    members = defaultdict(list)
    coverage = defaultdict(lambda: [9999, 0])  # member -> [min_start, max_end]
    for f in files:
        mm = MEMBER_RE.search(os.path.basename(f))
        rr = RANGE_RE.search(os.path.basename(f))
        if not mm or not rr:
            print(f"  [skip] cannot parse member/range: {os.path.basename(f)}")
            continue
        mem = mm.group(1)
        y0, y1 = int(rr.group(1)), int(rr.group(2))
        members[mem].append(f)
        coverage[mem][0] = min(coverage[mem][0], y0)
        coverage[mem][1] = max(coverage[mem][1], y1)

    ens_dir = os.path.join(args.cmip6_dir, args.out_name)
    full, partial = [], []
    for mem in sorted(members):
        c0, c1 = coverage[mem]
        if c0 <= args.start and c1 >= args.end:
            full.append(mem)
        else:
            partial.append(mem)
            print(f"  [skip partial] {mem}: covers {c0}-{c1} "
                  f"(need {args.start}-{args.end})")

    if not full:
        raise SystemExit("no full-coverage members found — nothing to build")

    print(f"\nFull-coverage members ({len(full)}): {full}")
    if args.dry_run:
        print("[dry-run] not creating symlinks")
    else:
        for mem in full:
            mdir = os.path.join(ens_dir, mem)
            os.makedirs(mdir, exist_ok=True)
            for f in sorted(members[mem]):
                link = os.path.join(mdir, os.path.basename(f))
                if os.path.islink(link) or os.path.exists(link):
                    os.remove(link)
                os.symlink(os.path.realpath(f), link)
        print(f"\nBuilt {ens_dir}/")

    print("\nSet in eval_aero.py ssp126 experiment:")
    print(f'    data_dir     = os.path.join(SCRATCH, "cmip6", "{args.out_name}"),')
    print(f"    realizations = {full},")


if __name__ == "__main__":
    main()
