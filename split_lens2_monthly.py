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
   (config_data.yaml and config_data_monthly.yaml both mark it so). It IS
   staged, into the same hist/ssp370 trees as everything else, because that is
   how this codebase separates train from val: ONE tree per scenario holding
   every member, and the split lives in the realization LISTS of
   experiment_configs vs val_experiment_configs. The AAER/GHG trees already
   work that way (all 20/15 members on disk, 001-009 train, 010 val).
   Withholding it from disk does not prevent a leak — it makes
   val_experiment_configs crash on an empty glob in
   ClimateDataset.load_data. --omit-members exists for the rare case where a
   member must genuinely not exist on disk; it is NOT the leak guard.

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
# LE2-1231.001 is the validation member in config_data.yaml. It is still STAGED
# (see the header): the train/val separation is the realization lists in the
# config, not the presence of files on disk. Named here only so the run prints a
# reminder that it must appear in val_experiment_configs and nowhere else.
VAL_MEMBERS = ["LE2-1231.001"]


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
    ap.add_argument("--omit-members", nargs="+", default=[],
                    help="LE2-style member ids to leave off disk entirely. "
                         "Not the leak guard — validation members ARE staged "
                         "by default and are excluded from training by the "
                         "realization lists in the config. See the header.")
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
            held = le2 in args.omit_members
            is_val = le2 in VAL_MEMBERS
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
                    note = ""
                    if held:
                        note = "   [OMITTED — not written]"
                    elif is_val:
                        note = "   [validation member — staged; keep it out of "\
                               "experiment_configs]"
                    print(f"  [dry-run] {member} -> {le2}: {spans}{note}")
                    ds.close()
                    continue
                for scen, part in (("hist", hist), ("ssp370", ssp)):
                    if held:
                        continue
                    out = os.path.join(vdir, scen, le2)
                    n = write_chunks(part, out, args.num_chunks)
                    print(f"  [ok] {var}/{scen}/{le2}: {n} months")
                if held:
                    print(f"  [omit] {member} -> {le2}: --omit-members, nothing "
                          f"written")
                elif is_val:
                    print(f"  [VAL] {le2} is the validation member: staged like "
                          f"any other, and must appear ONLY in "
                          f"val_experiment_configs")
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
