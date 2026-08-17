import lumi_paths as L
import argparse
import intake
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
import os
import glob
import re
from joblib import Parallel, delayed

NUM_CHUNKS = 40

parser = argparse.ArgumentParser()
parser.add_argument("--variable",   nargs="+", default=["TREFHT", "PRECT"],
                    help="CESM2-LE variable(s) to download (default: TREFHT PRECT)")
parser.add_argument("--output-dir", default=f"{L.DATA}/training_data",
                    help="Root output directory; data saved to <output-dir>/<variable>/")
parser.add_argument("--n-jobs",     type=int, default=4,
                    help="Parallel save workers")
parser.add_argument("--monthly", action="store_true",
                    help="Keep MONTHLY resolution instead of collapsing to annual "
                         "means. The AWS catalog is monthly either way; the "
                         "default groupby('time.year').mean() is what throws the "
                         "seasonal cycle away. Needed for seq_len>1 training, "
                         "and for aerosols in particular, whose forcing depends "
                         "on WHEN they are emitted (insolation, monsoon washout, "
                         "BC-on-snow are all seasonal). ~12x the data: budget "
                         "~80 MB per member-year per variable against ~200 kB "
                         "annual.")
parser.add_argument("--max-members", type=int, default=0, metavar="N",
                    help="use only the first N common members (0 = all). "
                         "SIZE: monthly is 2.65 MB per member-year per variable "
                         "(192x288 x 12 x 4 B); hist+ssp370 is 251 years, so "
                         "each member costs ~1.33 GB for TREFHT+PRECT. The "
                         "catalog has 50 members = ~66 GB. Selection is the "
                         "first N sorted member ids, deterministic.")
parser.add_argument("--skip-existing", action="store_true",
                    help="Skip any realization whose output directory already "
                         "holds chunk files, so an interrupted download resumes "
                         "instead of re-fetching everything.")
args = parser.parse_args()

N_JOBS = args.n_jobs


def save_dataset(dataset: xr.Dataset, realization: str, save_dir: str, num_chunks):
    """Saves the dataset in chunks to many netCDF4 files for parallel loading later."""

    full_save_dir = os.path.join(save_dir, realization)
    os.makedirs(full_save_dir, exist_ok=True)

    for file in os.listdir(full_save_dir):
        os.remove(os.path.join(full_save_dir, file))

    dataset = dataset.sel(member_id=realization)
    # The time dimension is "year" for annual output (groupby('time.year')) and
    # "time" for --monthly, which keeps the native axis. Hardcoding "year" made
    # every monthly save raise KeyError: 'year'.
    tdim = "year" if "year" in dataset.dims else "time"
    total_time_points = len(dataset[tdim])
    chunk_size = total_time_points // num_chunks

    split_datasets = []
    paths = []
    for idx in range(num_chunks):
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size if idx < num_chunks - 1 else None
        split_datasets.append(dataset.isel({tdim: slice(start_idx, end_idx)}))
        paths.append(os.path.join(full_save_dir, f"chunk_{idx}.nc"))

    xr.save_mfdataset(split_datasets, paths, compute=True)


os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

catalog = intake.open_esm_datastore(
    'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
)
print(catalog)

# ── Load all variables and find common members ────────────────────────────────
merged_by_var = {}
for variable in args.variable:
    print(f"\n[LOAD] {variable}")
    catalog_subset = catalog.search(variable=variable, frequency='monthly', forcing_variant="cmip6")
    dsets = catalog_subset.to_dataset_dict(storage_options={'anon': True})
    historical = dsets['atm.historical.monthly.cmip6']
    future     = dsets['atm.ssp370.monthly.cmip6']
    if args.monthly:
        # Keep every month. The concat dim stays 'time', which is also what
        # ClimateDataset expects when time_dim="time".
        merged_by_var[variable] = xr.concat([historical, future], dim='time')
        print(f"  [{variable}] MONTHLY: "
              f"{merged_by_var[variable].sizes.get('time', '?')} timesteps")
    else:
        historical = historical.groupby('time.year').mean()
        future     = future.groupby('time.year').mean()
        merged_by_var[variable] = xr.concat([historical, future], dim='year')

# Intersect member_id across all variables to guarantee alignment
common_members = None
for variable, ds in merged_by_var.items():
    members = set(ds["member_id"].values)
    common_members = members if common_members is None else common_members & members

common_members = sorted(common_members)
_n_all = len(common_members)
if args.max_members and args.max_members < _n_all:
    common_members = common_members[:args.max_members]
print(f"\n[MEMBERS] {_n_all} common members across {args.variable}"
      + (f" — using the first {len(common_members)}" if len(common_members) != _n_all else ""))
_yrs = 251   # hist 1850-2014 + ssp370 2015-2100
_gb = len(common_members) * _yrs * len(args.variable) * (192*288*4*(12 if args.monthly else 1)) / 1e9
print(f"[SIZE]    ~{_gb:.1f} GB uncompressed "
      f"({'monthly' if args.monthly else 'annual'}, {len(args.variable)} variables)")
for v, ds in merged_by_var.items():
    n = len(ds["member_id"].values)
    print(f"  {v}: {n} available → {len(common_members)} used")

# ── Download each variable for the common member set ─────────────────────────
for variable, merged in merged_by_var.items():
    print(f"\n{'='*60}")
    print(f"[VARIABLE] {variable}")
    output_dir = os.path.join(args.output_dir, variable)
    merged = merged.sel(member_id=common_members)

    todo = common_members
    if args.skip_existing:
        # A member counts as done only if its directory holds chunk files; an
        # empty directory from a killed run is retried rather than skipped.
        done = [m for m in common_members
                if glob.glob(os.path.join(output_dir, m, "chunk_*.nc"))]
        todo = [m for m in common_members if m not in set(done)]
        print(f"  --skip-existing: {len(done)} already present, {len(todo)} to fetch")
    if not todo:
        print(f"  nothing to do for {variable}")
        continue

    print(f"  {len(todo)} members → {output_dir}")
    Parallel(n_jobs=N_JOBS, backend="multiprocessing")(
        delayed(save_dataset)(merged, m, output_dir, NUM_CHUNKS) for m in todo
    )
    print(f"  Saved {len(todo)} members to {output_dir}")
