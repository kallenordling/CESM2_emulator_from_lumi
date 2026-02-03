import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

#emissions_ssp126.nc #emissions.nc

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

emis_ssp370 = Path("/scratch/project_462001112/emulator_data/emissions_new.nc")
emis_ssp126 = Path("/scratch/project_462001112/emulator_data/emissions.nc")

ds_ssp126 = xr.open_dataset(emis_ssp126)
ds_ssp370 = xr.open_dataset(emis_ssp370)

fig, ax = plt.subplots(2,1,figsize=(10, 7))

#for i,v in enumerate(["CO2_em_anthro",'sul']):
	#s126 = ds_ssp126[v].sum(['lat','lon'])
	#s370 = ds_ssp370[v].mean(['lat','lon'])
	#ax[i].plot(s126.year,s126.values,label="ssp126")
	#ax[i].plot(s370.year,s370.values,label='ssp370')
	
	
for i,v in enumerate(["CO2",'SO2']):
	#s126 = ds_ssp126[v].mean(['lat','lon'])
	s370 = ds_ssp370[v].sum(['lat','lon'])
	#ax[i].plot(s126.year,s126.values,label="ssp126")
	ax[i].plot(s370.year,s370.values,label='ssp370')	
plt.legend()
plt.savefig('emis.png')
