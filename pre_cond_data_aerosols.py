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
NA = 6.022e23
MW_SO2 = 64.066e-3  # kg/mol

path="//scratch/project_462001112/emulator_data/emission_data/"
files=['emissions-cmip6-ScenarioMIP_IAMC-AIM-ssp370-1-1_SO2_anthro-ag-ship-res_surface_mol_175001-210101_0.9x1.25_c20200924.nc','emissions-cmip6-ScenarioMIP_IAMC-AIM-ssp370-1-1_SO2_anthro-ene_surface_mol_175001-210101_0.9x1.25_c20190222.nc']
files=['emissions-cmip6-ScenarioMIP_IAMC-IMAGE-ssp126-1-1_SO2_anthro-ag-ship-res_surface_mol_175001-210101_0.9x1.25_c20190225.nc','emissions-cmip6-ScenarioMIP_IAMC-IMAGE-ssp126-1-1_SO2_anthro-ene_surface_mol_175001-210101_0.9x1.25_c20190225.nc']
ds_sum = None

for f in files:
    print("adding:", f)
    ds = xr.open_dataset(path+f).groupby('time.year').sum().sel(year=slice(1850,2100))
    for var_name, da in ds.data_vars.items():
        if var_name == "date":
           continue
        print(var_name)

        if ds_sum is None:
            ds_sum = ds[var_name].copy()
        else:
            # sum all data variables element-wise
            ds_sum = ds_sum + ds[var_name]
        #print(ds_sum.year)



ds = ds_sum
#ds=xr_cumsum_manual(ds,'year')


#ds.to_netcdf("co2_1990_2010.nc")
#ds_tmp.sel(year=slice(1990,2010)).to_netcdf("temp_1990_2010.nc")
ds.name="sul"
path="/scratch/project_462001112/emulator_data/"
ds.to_netcdf(path+"so2_final_ssp126l.nc")
print("save",path+"so2_final_ssp126l.nc")
ds2 = xr.open_dataset(path+"co2_final_ssp126l.nc")["CO2_em_anthro"].sortby('lat')
ds=ds.sortby('lat')
ds=ds.interp(year=ds2.year)
ds=ds.interp(lat=ds2.lat)
ds=ds * 1e4 * MW_SO2 / NA
ds=ds.fillna(0)
ds_merged = xr.merge([ds, ds2])
print(ds.year)
print(ds.lat)
print('#########')
print(ds2.year)
print(ds2.lat)
ds_merged.to_netcdf(path+"emissions_ssp126.nc")
print(ds_merged)
#area=xr.open_dataset("gridarea.nc")
#ds = (ds*area*31536000)/1e12
#print(ds_repeated.to_netcdf('co2_final.nc'))

#/fmi/scratch/project_2001927/nordlin1/emulator_data/co2_final_ssp126l.nc
