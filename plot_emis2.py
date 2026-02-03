import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from data.climate_dataset import ClimateDataset
from omegaconf import OmegaConf
#emissions_ssp126.nc #emissions.nc
from hydra.utils import instantiate

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
realization_dict = {"gen": "r10i1181p1f1", "val": "r10i1181p1f1", "test": "r10i1181p1f1"}
config = OmegaConf.load('./configs/generate_aero.yaml')
dataset: ClimateDataset = instantiate(
        config.dataset,
        data_dir=config.data_dir,
        realizations=[realization_dict[config.gen_mode]],
        target_vars=config.variables,cond_vars=["CO2_em_anthro",'sul'],cond_file="emissions_ssp126_v2.nc"
)

xr_ds_ssp126 = dataset.dataset_cond.load()


dataset: ClimateDataset = instantiate(
        config.dataset,
        data_dir=config.data_dir,
        realizations=[realization_dict[config.gen_mode]],
        target_vars=config.variables,cond_vars=["CO2_em_anthro",'sul'],cond_file="emissions.nc"
)

xr_ds_ssp370 = dataset.dataset_cond.load()

dataset: ClimateDataset = instantiate(
        config.dataset,
        data_dir=config.data_dir,
        realizations=[realization_dict[config.gen_mode]],
        target_vars=config.variables,cond_vars=["CO2_em_anthro",'sul'],cond_file="emissions_ssp370_ssp126aer_v2.nc"
)

xr_ds_ramip = dataset.dataset_cond.load()

emis_ssp370 = Path("/scratch/project_462001112/emulator_data/emissions.nc")
emis_ssp126 = Path("/scratch/project_462001112/emulator_data/emissions_ssp126_v2.nc")

ds_ssp126 = xr_ds_ssp126#xr.open_dataset(emis_ssp126)
ds_ssp370 = xr_ds_ssp370 # xr.open_dataset(emis_ssp370)
ds_ramip=xr_ds_ramip

fig, ax = plt.subplots(2,1,figsize=(10, 7))

for i,v in enumerate(["CO2_em_anthro",'sul']):
	s126 = ds_ssp126[v].mean(['lat','lon'])
	s370 = ds_ssp370[v].mean(['lat','lon'])
	sramip = ds_ramip[v].mean(['lat','lon'])
	ax[i].plot(s126.year,s126.values,label="ssp126")
	ax[i].plot(s370.year,s370.values,label='ssp370')
	ax[i].plot(sramip.year,sramip.values,label='ramip')
ax[1].legend()
ax[0].legend()
plt.savefig('emis2.png')

