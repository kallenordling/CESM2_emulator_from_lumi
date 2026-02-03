import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def calcmean(ds):
	weights = np.cos(np.deg2rad(ds.lat))
	ds_weighted = ds.weighted(weights)
	return ds_weighted.mean(("lon", "lat"))	
	

data_dir=Path("/fmi/scratch/project_2001927/nordlin1/emulator_data/gen_te1t_TREFHT_1850-2100")
nc_files = sorted(data_dir.glob("*.nc"))
for nc_path in nc_files:
	print("open",nc_path)
	ds=xr.open_dataset(nc_path)['TREFHT']
	ds = calcmean(ds)
	anom=ds-ds.sel(year=slice(1850,1900)).mean()
	anom.plot(color="#df0000")
	
data_dir=Path("/fmi/scratch/project_2001927/nordlin1/emulator_data/gen_ssp126_TREFHT_1850-2100")
nc_files = sorted(data_dir.glob("*.nc"))
for nc_path in nc_files:
	print("open",nc_path)
	ds=xr.open_dataset(nc_path)['TREFHT']
	ds = calcmean(ds)
	anom=ds-ds.sel(year=slice(1850,1900)).mean()
	anom.plot(color="#003466")
	
plt.savefig('timeser.png')
plt.close()

#cumulative plots
co2=xr.open_dataset('/fmi/scratch/project_2001927/nordlin1/emulator_data/co2_final_ssp370l.nc')['CO2_em_anthro'].sum(['lat','lon'])
for nc_path in nc_files:
	print("open",nc_path)
	ds=xr.open_dataset(nc_path)['TREFHT']
	ds = calcmean(ds)
	anom=ds-ds.sel(year=slice(1850,1900)).mean()
	anom=anom.isel(time=0)
	plt.plot(co2.isel(year=slice(0,250)),anom.isel(year=slice(0,250)))
plt.savefig('emissions_temp.png')
