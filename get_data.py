import argparse
import intake
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
import os
import re
from joblib import Parallel, delayed

NUM_CHUNKS = 40

parser = argparse.ArgumentParser()
parser.add_argument("--variable",   nargs="+", default=["TREFHT", "PRECT"],
                    help="CESM2-LE variable(s) to download (default: TREFHT PRECT)")
parser.add_argument("--output-dir", default="/scratch/project_462001328/emulator_data/training_data",
                    help="Root output directory; data saved to <output-dir>/<variable>/")
parser.add_argument("--n-jobs",     type=int, default=4,
                    help="Parallel save workers")
args = parser.parse_args()

N_JOBS = args.n_jobs


def save_dataset(dataset: xr.Dataset, realization: str, save_dir: str, num_chunks):
    """Saves the dataset in chunks to many netCDF4 files for parallel loading later."""

    full_save_dir = os.path.join(save_dir, realization)
    os.makedirs(full_save_dir, exist_ok=True)

    for file in os.listdir(full_save_dir):
        os.remove(os.path.join(full_save_dir, file))

    dataset = dataset.sel(member_id=realization)
    total_time_points = len(dataset["year"])
    chunk_size = total_time_points // num_chunks

    split_datasets = []
    paths = []
    for idx in range(num_chunks):
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size if idx < num_chunks - 1 else None
        split_datasets.append(dataset.isel(year=slice(start_idx, end_idx)))
        paths.append(os.path.join(full_save_dir, f"chunk_{idx}.nc"))

    xr.save_mfdataset(split_datasets, paths, compute=True)


os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

catalog = intake.open_esm_datastore(
    'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
)
print(catalog)

for variable in args.variable:
    print(f"\n{'='*60}")
    print(f"[VARIABLE] {variable}")
    output_dir = os.path.join(args.output_dir, variable)

    catalog_subset = catalog.search(variable=variable, frequency='monthly', forcing_variant="cmip6")
    print(catalog_subset.df)

    dsets = catalog_subset.to_dataset_dict(storage_options={'anon': True})

    historical = dsets['atm.historical.monthly.cmip6'].groupby('time.year').mean()
    future     = dsets['atm.ssp370.monthly.cmip6'].groupby('time.year').mean()
    merged     = xr.concat([historical, future], dim='year')

    members = list(merged["member_id"].values)
    print(f"  {len(members)} members → {output_dir}")

    Parallel(n_jobs=N_JOBS, backend="multiprocessing")(
        delayed(save_dataset)(merged, m, output_dir, NUM_CHUNKS) for m in members
    )

    print(f"  Saved {len(members)} members to {output_dir}")
