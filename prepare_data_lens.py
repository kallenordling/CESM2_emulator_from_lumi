#!/usr/bin/env python3
"""
Preprocess CESM2 Large Ensemble (LENS2) target data into training chunks.

Identifies hist vs SSP370 files by compset in filename:
  BHIST*   → historical experiment (1850-2014)
  BSSP370* → ssp370 experiment    (2015-2100)

Each experiment is processed separately and saved to:
  {output_dir}/{variable}/{experiment}/{member_label}/chunk_{i}.nc

Where experiment is "hist" or "ssp370".

Input layout (from download_lens2.py):
  {data_dir}/LENS2/{variable}/{member_label}/
      b.e21.BHISTsmbb.f09_g17.LE2-1301.012.cam.h0.{variable}.185001-185912.nc
      ...
      b.e21.BSSP370smbb.f09_g17.LE2-1301.012.cam.h0.{variable}.201501-202412.nc
      ...

Output layout:
  {output_dir}/{variable}/hist/{member_label}/chunk_0.nc ... chunk_{N-1}.nc
  {output_dir}/{variable}/ssp370/{member_label}/chunk_0.nc ... chunk_{N-1}.nc

Usage:
  python prepare_data_lens.py --data-dir /scratch/.../emulator_data/lens2 \
      --output-dir /scratch/.../emulator_data/training_data --variable PRECT
  python prepare_data_lens.py --data-dir /scratch/.../lens2 --members LE2-1301.012 --dry-run
  python prepare_data_lens.py --data-dir /scratch/.../lens2 --experiments hist ssp370
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CHUNKS = 40

EXPERIMENT_COMPSET = {
    "hist":   re.compile(r"b\.e21\.BHIST[^.]*\."),
    "ssp370": re.compile(r"b\.e21\.BSSP370[^.]*\."),
}

YEAR_RANGE = {
    "hist":   (1850, 2014),
    "ssp370": (2015, 2100),
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_member_files(data_dir: Path, member_label: str, experiment: str,
                      variable: str) -> list[Path]:
    """
    Find all `variable` .nc files for a member matching the given experiment
    compset, sorted by start date.
    """
    member_dir = data_dir / "LENS2" / variable / member_label
    if not member_dir.exists():
        return []

    compset_re = EXPERIMENT_COMPSET[experiment]
    date_re    = re.compile(r"\.(\d{6})-(\d{6})\.nc$")

    files = []
    for f in member_dir.iterdir():
        if compset_re.search(f.name) and date_re.search(f.name):
            m = date_re.search(f.name)
            files.append((int(m.group(1)), f))

    files.sort(key=lambda x: x[0])
    return [f for _, f in files]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def load_and_concat(files: list[Path]) -> xr.Dataset:
    datasets = [xr.open_dataset(f, use_cftime=True) for f in files]
    ds = xr.concat(datasets, dim="time").sortby("time")
    for d in datasets:
        d.close()
    return ds


def check_monthly_coverage(ds: xr.Dataset) -> None:
    """
    Fail loudly if interior years don't have 12 months. resample('1YE')
    silently NaN-fills missing years AND averages partial years from a
    single month at gap edges — this poisoned the first PRECT staging.
    Edge years are exempt (CESM h0 month labels can shift them to 11/1).
    """
    counts = {}
    for t in ds["time"].values:
        counts[t.year] = counts.get(t.year, 0) + 1
    y0, y1 = min(counts), max(counts)
    bad = {y: counts.get(y, 0) for y in range(y0 + 1, y1)
           if counts.get(y, 0) != 12}
    if bad:
        raise ValueError(
            f"incomplete monthly coverage in {y0}-{y1} (year: n_months) "
            f"{dict(sorted(bad.items()))} — missing source segments? "
            f"Re-run run_download_data.sh before staging."
        )


def annual_mean(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    ds = ds.drop_vars(
        [v for v in ["time_bnds", "lat_bnds", "lon_bnds"] if v in ds],
        errors="ignore"
    )
    check_monthly_coverage(ds)
    ds_annual = ds.resample(time="1YE").mean(dim="time")
    years = np.array([t.year for t in ds_annual.time.values])
    ds_annual = ds_annual.assign_coords(time=years)
    ds_annual["time"].attrs["units"] = "year"
    ds_annual["time"].attrs["long_name"] = "year"
    return ds_annual


def save_chunks(ds: xr.Dataset, out_dir: Path, num_chunks: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("chunk_*.nc"):
        f.unlink()

    total_time = len(ds["time"])
    chunk_size = total_time // num_chunks

    split_datasets, paths = [], []
    for idx in range(num_chunks):
        start = idx * chunk_size
        end   = start + chunk_size if idx < num_chunks - 1 else None
        split_datasets.append(ds.isel(time=slice(start, end)))
        paths.append(str(out_dir / f"chunk_{idx}.nc"))

    xr.save_mfdataset(split_datasets, paths, compute=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess CESM2 LENS2 target → hist + ssp370 → annual means → chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir",    required=True, type=Path,
                   help="Root input dir containing LENS2/{variable}/{member}/ subdirs")
    p.add_argument("--output-dir",  type=Path, default=None,
                   help="Root output dir (default: data-dir + '_processed')")
    p.add_argument("--variable",    default="TREFHT",
                   help="Target variable to process (default: TREFHT; e.g. PRECT)")
    p.add_argument("--experiments", nargs="+", default=["hist", "ssp370"],
                   choices=["hist", "ssp370"],
                   help="Experiments to process (default: both)")
    p.add_argument("--members",     nargs="+", default=None,
                   help="Specific member labels e.g. LE2-1301.012 (default: all found)")
    p.add_argument("--num-chunks",  type=int, default=NUM_CHUNKS,
                   help=f"Time chunks per member/experiment (default: {NUM_CHUNKS})")
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    variable   = args.variable
    data_dir   = args.data_dir
    output_dir = args.output_dir or Path(str(data_dir) + "_processed")

    # Discover members
    if args.members:
        members = args.members
    else:
        lens2_dir = data_dir / "LENS2" / variable
        if not lens2_dir.exists():
            sys.exit(f"ERROR: {lens2_dir} does not exist. Check --data-dir/--variable.")
        members = sorted([d.name for d in lens2_dir.iterdir() if d.is_dir()])

    print(f"\nCESM2 LENS2 preprocessing")
    print(f"  Variable   : {variable}")
    print(f"  Input      : {data_dir}/LENS2/{variable}/{{member}}/")
    print(f"  Output     : {output_dir}/{variable}/{{experiment}}/{{member}}/chunk_{{i}}.nc")
    print(f"  Experiments: {args.experiments}")
    print(f"  Members    : {len(members)}")
    print(f"  Chunks     : {args.num_chunks} per member/experiment\n")

    total_ok = total_skip = total_fail = 0

    for experiment in args.experiments:
        start_year, end_year = YEAR_RANGE[experiment]
        print(f"--- {experiment.upper()}  ({start_year}–{end_year}) ---")

        for member_label in members:
            out_dir = output_dir / variable / experiment / member_label
            label   = f"{experiment}/{member_label}"

            if not args.overwrite and out_dir.exists() and any(out_dir.glob("chunk_*.nc")):
                print(f"  [SKIP]  {label}")
                total_skip += 1
                continue

            files = find_member_files(data_dir, member_label, experiment, variable)
            if not files:
                print(f"  [MISS]  {label}  — no {experiment} files found")
                total_fail += 1
                continue

            date_range = f"{files[0].name.split('.')[-2]} → {files[-1].name.split('.')[-2]}"
            print(f"  [PROC]  {label}  ({len(files)} chunks: {date_range})")

            if args.dry_run:
                for f in files:
                    print(f"           {f.name}")
                print(f"           → {out_dir}/chunk_0.nc ... chunk_{args.num_chunks-1}.nc")
                continue

            try:
                ds        = load_and_concat(files)
                ds_annual = annual_mean(ds, start_year, end_year)
                ds.close()

                n_years = len(ds_annual["time"])
                y0, y1  = int(ds_annual.time.values[0]), int(ds_annual.time.values[-1])
                print(f"          → {n_years} years ({y0}–{y1}), shape: {ds_annual[variable].shape}")

                save_chunks(ds_annual, out_dir, args.num_chunks)
                print(f"          → {out_dir}/chunk_*.nc")
                total_ok += 1

            except Exception as e:
                print(f"  [FAIL]  {label}: {e}")
                import traceback; traceback.print_exc()
                total_fail += 1

        print()

    print(f"Done: {total_ok} processed, {total_skip} skipped, {total_fail} failed")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
