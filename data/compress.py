import xarray as xr
import numpy as np

ds = xr.open_dataset("/scratch/project_462001112/emulator_data/emissions_co2_so2_regridded.nc")

comp = dict(
    zlib=True,
    complevel=9,      # max compression
    shuffle=True,     # improves compression for floats
)

encoding = {}

for var in ['CO2']:
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

ds['CO2'].to_netcdf(
    "compressed.nc",
    format="NETCDF4",
    engine="netcdf4",
    encoding=encoding
)

import numpy as np

arr = ds['CO2'].values   # 3D array

np.savetxt(
    "array.txt",
    arr.reshape(arr.shape[0], -1)  # flatten last dims
)

# save shape separately
with open("array_shape.txt", "w") as f:
    f.write(" ".join(map(str, arr.shape)))