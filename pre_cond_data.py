import numpy as np
import xarray as xr
import xarray as xr
import numpy as np

def xr_cumsum_manual(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    Manually compute cumulative sum along a given dimension, preserving coordinates.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    dim : str
        Name of the dimension to sum along.

    Returns
    -------
    xr.DataArray
        Cumulative sum along `dim`, with coordinates preserved.
    """
    # Get axis index for the given dimension
    axis = da.get_axis_num(dim)

    # Convert to NumPy and manually compute cumulative sum
    data_np = da.values
    out_np = np.empty_like(data_np)

    # Loop cumulative sum manually
    running_total = np.zeros_like(data_np.take(indices=0, axis=axis))
    for i in range(data_np.shape[axis]):
        slice_i = np.take(data_np, i, axis=axis)
        running_total = running_total + slice_i
        out_np = np.insert(np.delete(out_np, i, axis=axis), i, running_total, axis=axis)

    # Wrap back into DataArray with same coords
    result = xr.DataArray(
        out_np,
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs
    )

    return result


path="/fmi/scratch/project_2001927/nordlin1/emulator_data/emission_data/"
hist_file="emissions-cmip6_CO2_anthro_surface_175001-201412_0.9x1.25_kgm2s_c20180516.nc"
exp_file="emissions-cmip6_CO2_anthro_surface_ScenarioMIP_IAMC-IMAGE-ssp126_201401-210112_fv_0.9x1.25_c20190207.nc"

ds_hist=xr.open_dataset(path+hist_file).groupby('time.year').sum()
ds_exp=xr.open_dataset(path+exp_file).groupby('time.year').sum()
ds = xr.concat([ds_hist,ds_exp],dim='year')['CO2_flux'].sel(year=slice(1850,2100))

ds=xr_cumsum_manual(ds,'year')


#ds.to_netcdf("co2_1990_2010.nc")
#ds_tmp.sel(year=slice(1990,2010)).to_netcdf("temp_1990_2010.nc")
ds.name="CO2_em_anthro"
ds.to_netcdf(path+"co2_final_ssp126l.nc")
print("save",path+"co2_final_ssp126l.nc")

#area=xr.open_dataset("gridarea.nc")
#ds = (ds*area*31536000)/1e12
#print(ds_repeated.to_netcdf('co2_final.nc'))

#/fmi/scratch/project_2001927/nordlin1/emulator_data/co2_final_ssp126l.nc
