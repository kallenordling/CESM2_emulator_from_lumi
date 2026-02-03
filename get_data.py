import intake
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
import os
import re
from joblib import Parallel, delayed

NUM_CHUNKS = 40


def save_dataset(dataset: xr.Dataset, realization: str, save_dir: str, num_chunks):
    """Saves the dataset in chunks to many netCDF4 files for parallel loading later."""

    # Create the save directory if it doesn't already exist
    full_save_dir = os.path.join(save_dir, realization)
    os.makedirs(full_save_dir, exist_ok=True)

    # Delete all the files in the save directory
    for file in os.listdir(full_save_dir):
        os.remove(os.path.join(full_save_dir, file))
    dataset = dataset.sel(member_id=realization)
    # Determine the number of chunks based on the length of the 'time' dimension
    total_time_points = len(dataset["year"])
    chunk_size = total_time_points // num_chunks
    enc = {"zlib": True, "complevel": 9}
    split_datasets = []
    paths = []
    # 1. Split the dataset into chunks
    for idx in range(num_chunks):
        # 1. Determine the start and end indices for the chunk
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size

        # Make sure we cover the remainder if we're on the last chunk
        if idx == num_chunks - 1:
            end_idx = None

        # Slice that chunk from the dataset
        split_datasets.append(dataset.isel(year=slice(start_idx, end_idx)))

        # Save the chunk to an indexed file
        paths.append(os.path.join(full_save_dir, f"chunk_{idx}.nc"))

    xr.save_mfdataset(split_datasets, paths, compute=True)

os.environ['AWS_NO_SIGN_REQUEST'] = 'YES' 

catalog = intake.open_esm_datastore(
    'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
)
print(catalog)
catalog_subset = catalog.search(variable='TREFHT', frequency='monthly',forcing_variant="cmip6")
print(catalog_subset.df)
dsets = catalog_subset.to_dataset_dict(storage_options={'anon':True})
print(dsets)


historical_cmip6 = dsets['atm.historical.monthly.cmip6'].groupby('time.year').mean()
future_cmip6 = dsets['atm.ssp370.monthly.cmip6'].groupby('time.year').mean()
merge_ds_cmip6= xr.concat([historical_cmip6, future_cmip6], dim='year')
print(merge_ds_cmip6)
members = list(merge_ds_cmip6["member_id"].values)
n_jobs=4
results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(save_dataset)(merge_ds_cmip6,m,"/fmi/scratch/project_2001927/nordlin1/emulator_data",NUM_CHUNKS) for m in members
    )

print(f"Saved {len(results)} files to {out_dir}")

#merge_ds_cmip6.to_netcdf("CESM_yearly.nc")
