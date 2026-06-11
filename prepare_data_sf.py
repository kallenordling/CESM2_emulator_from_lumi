#!/usr/bin/env python3
"""
Preprocess CESM2 Single Forcing Large Ensemble (SF-LENS) target data into chunks.

For each ensemble × member:
  1. Glob all downloaded monthly .nc chunks (hist + SSP370)
  2. Concatenate into one continuous time series
  3. Compute annual means
  4. Split into NUM_CHUNKS equal time chunks and save as netCDF4

Input layout:
  {data_dir}/{ensemble}/{variable}/
      b.e21.*.CESM2-SF-GHG.001.cam.h0.{variable}.185001-185912.nc
      ...

Output layout:
  {output_dir}/{variable}/{ensemble}/{member}/
      chunk_{member}_0.nc
      chunk_{member}_1.nc
      ...
      chunk_{member}_{NUM_CHUNKS-1}.nc

Usage:
  python prepare_data_sf.py --data-dir /scratch/.../emulator_data/sf \
      --output-dir /scratch/.../emulator_data/training_data --variable PRECT
  python prepare_data_sf.py --data-dir /scratch/.../sf --ensembles GHG AAER --dry-run
  python prepare_data_sf.py --data-dir /scratch/.../sf --output-dir /scratch/.../out --members 001 002
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Dataset knowledge
# ---------------------------------------------------------------------------

ENSEMBLE_MEMBERS = {
    "GHG":  [f"{i:03d}" for i in range(1, 16)],   # 001-015
    "AAER": [f"{i:03d}" for i in range(1, 21)],   # 001-020
    "BMB":  [f"{i:03d}" for i in range(1, 16)],   # 001-015
    "EE":   [f"1{i:02d}" for i in range(1, 16)],  # 101-115
    "xAER": [f"{i:03d}" for i in range(1, 11)],   # 001-010
}

NUM_CHUNKS = 40


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_member_files(data_dir: Path, ensemble: str, member: str,
                      variable: str) -> list[Path]:
    var_dir = data_dir / ensemble / variable
    if not var_dir.exists():
        return []

    pattern = re.compile(
        rf"b\.e21\.[^.]+\.f09_g17\.CESM2-SF-{re.escape(ensemble)}[^.]*"
        rf"\.{re.escape(member)}\.cam\.h0\.{re.escape(variable)}\.(\d{{6}})-(\d{{6}})\.nc"
    )
    files = []
    for f in var_dir.iterdir():
        m = pattern.match(f.name)
        if m:
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


def annual_mean(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    ds = ds.drop_vars(
        [v for v in ["time_bnds", "lat_bnds", "lon_bnds"] if v in ds],
        errors="ignore"
    )
    ds_annual = ds.resample(time="1YE").mean(dim="time")
    years = np.array([t.year for t in ds_annual.time.values])
    ds_annual = ds_annual.assign_coords(time=years)
    ds_annual["time"].attrs["units"] = "year"
    ds_annual["time"].attrs["long_name"] = "year"
    return ds_annual


def save_chunks(ds: xr.Dataset, out_dir: Path, member: str, num_chunks: int):
    """Save dataset as chunk_{member}_0.nc ... chunk_{member}_{N-1}.nc inside out_dir."""
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
        paths.append(str(out_dir / f"chunk_{member}_{idx}.nc"))

    xr.save_mfdataset(split_datasets, paths, compute=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess CESM2 SF-LENS target → annual means → chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir",   required=True, type=Path,
                   help="Root input dir containing {ensemble}/{variable}/ subdirs")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Root output dir (default: data-dir + '_processed'). "
                        "Chunks saved to output-dir/{variable}/{ensemble}/chunk_{member}_{i}.nc")
    p.add_argument("--variable",   default="TREFHT",
                   help="Target variable to process (default: TREFHT; e.g. PRECT)")
    p.add_argument("--ensembles",  nargs="+", default=list(ENSEMBLE_MEMBERS.keys()),
                   choices=list(ENSEMBLE_MEMBERS.keys()),
                   help="Ensembles to process (default: all)")
    p.add_argument("--members",    nargs="+", default=None,
                   help="Specific members e.g. 001 002 (default: all per ensemble)")
    p.add_argument("--start-year", type=int, default=1850)
    p.add_argument("--end-year",   type=int, default=2050)
    p.add_argument("--num-chunks", type=int, default=NUM_CHUNKS,
                   help=f"Time chunks per member (default: {NUM_CHUNKS})")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-process already completed members")
    return p.parse_args()


def main():
    args = parse_args()

    variable   = args.variable
    data_dir   = args.data_dir
    output_dir = args.output_dir or Path(str(args.data_dir) + "_processed")

    print(f"\nCESM2 SF-LENS preprocessing")
    print(f"  Variable   : {variable}")
    print(f"  Input      : {data_dir}")
    print(f"  Output     : {output_dir}/{variable}/{{ensemble}}/{{member}}/chunk_{{i}}.nc")
    print(f"  Period     : {args.start_year}–{args.end_year} (annual means)")
    print(f"  Chunks     : {args.num_chunks} per member")
    print(f"  Ensembles  : {args.ensembles}\n")

    total_ok = total_skip = total_fail = 0

    for ensemble in args.ensembles:
        members = args.members if args.members else ENSEMBLE_MEMBERS[ensemble]
        for member in members:
            label   = f"{ensemble}/{member}"
            out_dir = output_dir / variable / ensemble / member

            # Skip check
            if not args.overwrite and out_dir.exists() and any(out_dir.glob("chunk_*.nc")):
                print(f"  [SKIP]  {label}  (already done, use --overwrite to redo)")
                total_skip += 1
                continue

            files = find_member_files(data_dir, ensemble, member, variable)
            if not files:
                print(f"  [MISS]  {label}  — no files in {data_dir}/{ensemble}/{variable}/")
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
                ds_annual = annual_mean(ds, args.start_year, args.end_year)
                ds.close()

                n_years = len(ds_annual["time"])
                y0, y1  = int(ds_annual.time.values[0]), int(ds_annual.time.values[-1])
                print(f"          → {n_years} years ({y0}–{y1}), shape: {ds_annual[variable].shape}")

                save_chunks(ds_annual, out_dir, member, args.num_chunks)
                print(f"          → {out_dir}/chunk_*.nc")
                total_ok += 1

            except Exception as e:
                print(f"  [FAIL]  {label}: {e}")
                import traceback; traceback.print_exc()
                total_fail += 1

    print(f"\nDone: {total_ok} processed, {total_skip} skipped, {total_fail} failed")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
