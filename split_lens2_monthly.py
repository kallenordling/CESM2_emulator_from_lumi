#!/usr/bin/env python3
"""
Split the continuous LENS2 monthly trees into hist and ssp370 scenario trees.

get_data.py --monthly writes ONE record per member spanning 1850-2100, under

    training_data_monthly/<VAR>/<member>/chunk_<i>.nc      member = r1i1001p1f1

but config_data_monthly.yaml asks for a scenario level and the LE2 naming the
annual trees use:

    training_data_monthly/<VAR>/hist/LE2-1001.001/chunk_<i>.nc      1850-2014
    training_data_monthly/<VAR>/ssp370/LE2-1001.001/chunk_<i>.nc    2015-2100

Three things are reconciled here.

1. THE SPLIT. CMIP6 historical ends 2014 and ScenarioMIP starts 2015, and the
   cond files are built on that boundary (emissions_hist_* is 1850-2014,
   emissions_ssp370_* is 2015-2100). Cutting anywhere else would misalign the
   conditioning from the target.

2. THE NAMING. `r{run}i{base}p1f1` -> `LE2-{base}.{run:03d}`. Mechanical, but it
   has to happen or the realization lists never match.

3. THE HELD-OUT MEMBER. LE2-1231.001 is the validation member
   (config_data.yaml marks it so). It is PRESENT in the monthly download, and
   staging it into a training tree would leak validation data into training.
   --val-members keeps it out of the training tree by default; pass
   --allow-val-in-train only if you genuinely mean to.

Chunking matches get_data.py:save_dataset and stage_sf_monthly.py.

Usage:
    python split_lens2_monthly.py --root /scratch/project_2019839/emulator_data/training_data_monthly
    python split_lens2_monthly.py --root ... --dry-run
"""
import argparse
import os
import re
import shutil
import sys
from glob import glob

import xarray as xr

MEMBER_RE = re.compile(r"^r(\d+)i(\d+)p\d+f\d+$")
HIST_END = 2014                       # CMIP6 historical / ScenarioMIP boundary
# LE2-1231.001 is the validation member in config_data.yaml. Keeping the default
# here rather than in the caller means the leak has to be opted INTO.
DEFAULT_VAL = ["LE2-1231.001"]


def le2_name(member):
    """r1i1001p1f1 -> LE2-1001.001, or None if it is not an intake member id."""
    m = MEMBER_RE.match(member)
    if not m:
        return None
    run, base = m.group(1), m.group(2)
    return f"LE2-{base}.{int(run):03d}"


def write_chunks(ds, out_dir, num_chunks):
    """Same convention as get_data.py:save_dataset — chunk_<i>.nc."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    n = ds.sizes["time"]
    if n < num_chunks:
        num_chunks = max(1, n)
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
    ap.add_argument("--root", required=True,
                    help="training_data_monthly directory")
    ap.add_argument("--variable", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--num-chunks", type=int, default=40)
    ap.add_argument("--val-members", nargs="+", default=DEFAULT_VAL,
                    help=f"members to keep OUT of the training trees "
                         f"(default: {' '.join(DEFAULT_VAL)})")
    ap.add_argument("--allow-val-in-train", action="store_true",
                    help="stage the validation member into the training trees "
                         "anyway — this leaks validation data, so it must be "
                         "asked for explicitly")
    ap.add_argument("--keep-source", action="store_true",
                    help="leave the original <VAR>/<member>/ dirs in place "
                         "(default: leave them; they are never deleted)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rc = 0
    for var in args.variable:
        vdir = os.path.join(args.root, var)
        if not os.path.isdir(vdir):
            print(f"[skip] {vdir} does not exist", file=sys.stderr)
            continue
        members = sorted(d for d in os.listdir(vdir)
                         if MEMBER_RE.match(d)
                         and os.path.isdir(os.path.join(vdir, d)))
        print(f"\n[{var}] {len(members)} continuous member(s): "
              f"{', '.join(members) if members else '(none)'}")

        for member in members:
            le2 = le2_name(member)
            files = sorted(glob(os.path.join(vdir, member, "chunk_*.nc")),
                           key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
            if not files:
                print(f"  [skip] {member}: no chunks")
                continue
            held = le2 in args.val_members and not args.allow_val_in_train
            try:
                ds = xr.open_mfdataset(files, combine="by_coords",
                                       chunks={"time": 120})
                yrs = ds["time.year"] if "time" in ds.coords else None
                if yrs is None:
                    raise KeyError("no time coordinate")
                hist = ds.sel(time=yrs <= HIST_END)
                ssp = ds.sel(time=yrs > HIST_END)
                spans = (f"hist {int(hist['time.year'][0])}-"
                         f"{int(hist['time.year'][-1])} ({hist.sizes['time']} mo), "
                         f"ssp370 {int(ssp['time.year'][0])}-"
                         f"{int(ssp['time.year'][-1])} ({ssp.sizes['time']} mo)")
                if args.dry_run:
                    print(f"  [dry-run] {member} -> {le2}: {spans}"
                          + ("   [VAL — hist/ssp370 training trees SKIPPED]" if held else ""))
                    ds.close()
                    continue
                for scen, part in (("hist", hist), ("ssp370", ssp)):
                    if held:
                        continue
                    out = os.path.join(vdir, scen, le2)
                    n = write_chunks(part, out, args.num_chunks)
                    print(f"  [ok] {var}/{scen}/{le2}: {n} months")
                if held:
                    print(f"  [VAL] {member} -> {le2}: held out, not staged into "
                          f"hist/ssp370 (pass --allow-val-in-train to override)")
                ds.close()
            except Exception as e:
                print(f"  [FAIL] {var}/{member}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                rc = 1

    print("\nSource <VAR>/<member>/ directories are left untouched; remove them "
          "yourself once the split trees are verified.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
