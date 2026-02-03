import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs

# ---------- styling (journal-friendly) ----------
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

BASELINE = slice(1850, 1900)        # reference period
YR_MIN, YR_MAX = 1850, 2100        # x-axis limits
VAR = "TREFHT"

def calcmean(da: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over lat/lon (cos(lat) weighting)."""
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lon", "lat"))

def load_anomalies(dir_path: Path) -> xr.DataArray:
    """Load all members from dir, compute area mean + baseline anomalies."""
    series = []
    years = None
    for nc_path in sorted(dir_path.glob("member_[0-9].nc")):
        print("open", nc_path)
        da = xr.open_dataset(nc_path)[VAR].isel(time=0)#+273.15
        #da = calcmean(ds[VAR])

        # guard for baseline existence
        base = da.sel(year=BASELINE).mean('year')
        anom = da - base
        #print(anom)
        #print(base)
        #input()
        # name each member by filename stem
        series.append(anom)

    # stack to (member, year)
    ens = xr.concat(series, dim="member")
    return ens

def plot_xr_map(
    da,ax,
    time=None,
    projection=ccrs.PlateCarree(),
    extent=None,             # (west, east, south, north)
    cmap="viridis",
    add_coastlines=True,
    add_gridlines=True,
    figsize=(9, 5),
    title=None,
):
    """
    Plot xarray DataArray on a Cartopy map using .plot().
    Works for both 1D and 2D lat/lon coordinate grids.
    """

    # If time dimension present and a time index is given
    if time is not None:
        da = da.sel(time=time) if "time" in da.dims else da.isel(time=time)

    #fig = plt.figure(figsize=figsize)
    #ax = plt.axes(projection=projection)

    # Set map extent if requested
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add common map features
    if add_coastlines:
        ax.coastlines()
    if add_gridlines:
        ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.6)

    # Plot using xarray's built-in plot function
    # NOTE: We explicitly set transform so cartopy knows the coordinate system.
    m=da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,vmin=-11,vmax=11,
        add_colorbar=False
    )

    ax.set_title(title)
    return m

def plot_xr_map2(
    da,ax,
    time=None,
    projection=ccrs.PlateCarree(),
    extent=None,             # (west, east, south, north)
    cmap="viridis",
    add_coastlines=True,
    add_gridlines=True,
    figsize=(9, 5),
    title=None,
):
    """
    Plot xarray DataArray on a Cartopy map using .plot().
    Works for both 1D and 2D lat/lon coordinate grids.
    """

    # If time dimension present and a time index is given
    if time is not None:
        da = da.sel(time=time) if "time" in da.dims else da.isel(time=time)

    #fig = plt.figure(figsize=figsize)
    #ax = plt.axes(projection=projection)

    # Set map extent if requested
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add common map features
    if add_coastlines:
        ax.coastlines()
    if add_gridlines:
        ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.6)

    # Plot using xarray's built-in plot function
    # NOTE: We explicitly set transform so cartopy knows the coordinate system.
    m=da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,vmin=-2,vmax=2,
        add_colorbar=False
    )

    ax.set_title(title)
    return m

# ---------- paths ----------

dir_ssp370_aero = Path("/scratch/project_462001112/emulator_data/gen_ssp370_aero_v3_TREFHT_1850-2100")
dir_ssp126_aero = Path("/scratch/project_462001112/emulator_data/gen_ssp370_aero_standard_boosting2_TREFHT_1850-2100")


# ---------- load ----------
ens_te1t  = load_anomalies(dir_ssp370_aero).sel(year=slice(2080,2100)).mean('year').mean('member')
ens_ssp126 = load_anomalies(dir_ssp126_aero).sel(year=slice(2080,2100)).mean('year').mean('member')
#ens_aero = load_anomalies(dir_aero).sel(year=slice(2080,2100)).mean('year').mean('member')



# Align years across scenarios (intersection to be safe)
#common_years = np.intersect1d(ens_te1t["year"].values, ens_ssp126["year"].values)
#ens_te1t = ens_te1t.sel(year=common_years)
#ens_ssp126 = ens_ssp126.sel(year=common_years)

# ---------- plot ----------
fig, ax = plt.subplots(2,2,figsize=(10, 7),subplot_kw={"projection": ccrs.PlateCarree()} )
print(ens_te1t)
m1=plot_xr_map(
    ens_te1t,ax[0,0],
    cmap="coolwarm",title="SSP370"
)

plot_xr_map(
    ens_ssp126 ,ax[0,1],
    cmap="coolwarm",title="SSP126"
)




hist_cmip=xr.open_dataset("/scratch/project_462001112/emulator_data/cmip6/historical.nc")
ssp1=xr.open_dataset("/scratch/project_462001112/emulator_data/cmip6/ssp126.nc")
ssp3=xr.open_dataset("/scratch/project_462001112/emulator_data/cmip6/ssp370.nc")

common_members3 = np.intersect1d(hist_cmip.member.values, ssp3.member.values)
common_members1 = np.intersect1d(hist_cmip.member.values, ssp1.member.values)

ds_merged_ssp3 = xr.concat([hist_cmip.sel(member=common_members3), ssp3.sel(member=common_members3)], dim="year")
ds_merged_ssp1 = xr.concat([hist_cmip.sel(member=common_members1), ssp1.sel(member=common_members1)], dim="year")
cmip_ssp3=(ds_merged_ssp3)
cmip_ssp1=(ds_merged_ssp1)
cmip_ssp3=cmip_ssp3-cmip_ssp3.sel(year=BASELINE).mean('year')
cmip_ssp1=cmip_ssp1-cmip_ssp1.sel(year=BASELINE).mean('year')

diff1=ens_te1t-cmip_ssp3.tas.sel(year=slice(2080,2100)).mean('year').mean('member')
diff2=ens_ssp126-cmip_ssp3.tas.sel(year=slice(2080,2100)).mean('year').mean('member')

print(diff1)
m2=plot_xr_map2(
    diff1,ax[1,0],
    cmap="coolwarm",title="SSP370 difference"
)

plot_xr_map2(
    diff2,ax[1,1],
    cmap="coolwarm",title="SSP126 difference"
)

cax = fig.add_axes([0.2, 0.5, 0.6, 0.03])   # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(m1, cax=cax, orientation="horizontal")
cbar.set_label("Temperature anomaly (°C)")

cax = fig.add_axes([0.2, 0.08, 0.6, 0.03])   # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(m2, cax=cax, orientation="horizontal")
cbar.set_label("Temperature differnce between emulated and CMIP6 CESM2 (°C)")#fig.tight_layout()
plt.title("Emulated global temperature, trainin set includes only ssp370")
fig.savefig("temperature_anomaly2.png")
fig.savefig("temperature_anomaly2.svg")  # vector for journals
plt.close(fig)
