import xarray as xr
import numpy as np

ds = xr.open_dataset("/scratch/project_462001112/emulator_data/emissions_co2_so2_regridded.nc")

comp = dict(
    zlib=True,
    complevel=9,      # max compression
    shuffle=True,     # improves compression for floats
)

encoding = {}

for var in ds.data_vars:
    dtype = ds[var].dtype

    # reduce precision if float
    if np.issubdtype(dtype, np.floating):
        encoding[var] = {
            **comp,
            "dtype": "float32",   # float64 → float32 saves 50%
            "_FillValue": None
        }
    else:
        encoding[var] = comp

ds.to_netcdf(
    "compressed.nc",
    format="NETCDF4",
    engine="netcdf4",
    encoding=encoding
)