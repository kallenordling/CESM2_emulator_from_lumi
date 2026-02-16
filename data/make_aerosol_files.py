"""
Process CO2 emission input files from CEDS/input4MIPs:
1. Open AIR-anthro files, sum along 'level'
2. Open anthro files, sum along 'sector'
3. Calculate annual mean flux for both
4. Add AIR + anthro together
5. Convert kg/m2/s -> Gt CO2 per grid point per year (using grid cell areas)
6. Compute cumulative sum over time
"""

import xarray as xr
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", help="Experiment name, e.g. hist or ssp370")
args = parser.parse_args()

exp = args.exp# ── Configure paths ──────────────────────────────────────────────────────────
INPUT_DIR = "/scratch/project_462001112/emulator_data/emission_data/inputs4mips/"
OUTPUT_DIR = "/scratch/project_462001112/emulator_data/"

if exp == "hist":
    #AIR_PATTERN = os.path.join(INPUT_DIR, "CO2-em-AIR-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2024-10-21_gn_*.nc")
    ANTHRO_PATTERN = os.path.join(INPUT_DIR, "SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc")
if  exp =="ssp370":
    #AIR_PATTERN = os.path.join(INPUT_DIR, "CO2-em-AIR-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc")
    ANTHRO_PATTERN = os.path.join(INPUT_DIR, "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc")
R_EARTH = 6.371e6  # Earth radius in meters
SECONDS_PER_YEAR = 365.25 * 24 * 3600
KG_PER_GT = 1e12  # 1 Gt = 1e12 kg


def compute_grid_cell_area(lat, lon):
    """
    Compute area of each grid cell in m² assuming a regular lat/lon grid.
    Returns a 2D array of shape (len(lat), len(lon)).
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Infer cell edges from midpoints
    dlat = np.abs(np.diff(lat).mean())
    dlon = np.abs(np.diff(lon).mean())

    lat_edges = np.deg2rad(np.clip(
        np.concatenate([
            [lat[0] - dlat / 2],
            (lat[:-1] + lat[1:]) / 2,
            [lat[-1] + dlat / 2]
        ]),
        -90, 90
    ))

    dlon_rad = np.deg2rad(dlon)

    # Area = R^2 * |sin(lat_north) - sin(lat_south)| * dlon
    area = np.abs(np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])) * dlon_rad * R_EARTH**2
    # Broadcast to 2D (lat, lon)
    area_2d = np.broadcast_to(area[:, np.newaxis], (len(lat), len(lon)))
    return area_2d


def find_dim(ds, candidates):
    """Find the first matching dimension name from candidates."""
    for c in candidates:
        if c in ds.dims:
            return c
    raise ValueError(f"None of {candidates} found in dims: {list(ds.dims)}")


def rename_to_co2(ds):
    """Rename the main data variable to 'CO2'."""
    data_vars = [v for v in ds.data_vars if "bnds" not in v and "bound" not in v]
    assert len(data_vars) >= 1, f"No data variables found: {list(ds.data_vars)}"
    var_name = data_vars[0]
    print(f"  Renaming '{var_name}' -> 'CO2'")
    return ds.rename({var_name: "CO2"})


# ── 1. AIR-anthro: open & sum along level ────────────────────────────────────
print("Opening AIR-anthro files...")
#air_files = sorted(glob.glob(AIR_PATTERN))
#assert len(air_files) > 0, f"No AIR-anthro files found matching:\n  {AIR_PATTERN}"
#print(f"  Found {len(air_files)} files")

#ds_air = xr.open_mfdataset(air_files, combine="by_coords").load()
#print(f"  Variables: {list(ds_air.data_vars)}")
#print(f"  Dimensions: {dict(ds_air.dims)}")

#level_dim = find_dim(ds_air, ["level", "lev", "levels"])
#print(f"  Summing along '{level_dim}'...")
# Drop cftime bound variables early to prevent serialization issues
#ds_air = ds_air.drop_vars([v for v in ds_air if "bnds" in str(v) or "bound" in str(v)], errors="ignore")
#ds_air_summed = ds_air.sum(dim=level_dim)

# ── 2. Anthro: open & sum along sector ───────────────────────────────────────
print("\nOpening anthro files...")
anthro_files = sorted(glob.glob(ANTHRO_PATTERN))
assert len(anthro_files) > 0, f"No anthro files found matching:\n  {ANTHRO_PATTERN}"
print(f"  Found {len(anthro_files)} files")

ds_anthro = xr.open_mfdataset(anthro_files, combine="by_coords").load()
print(f"  Variables: {list(ds_anthro.data_vars)}")
print(f"  Dimensions: {dict(ds_anthro.dims)}")

sector_dim = find_dim(ds_anthro, ["sector", "sectors"])
print(f"  Summing along '{sector_dim}'...")
ds_anthro = ds_anthro.drop_vars([v for v in ds_anthro if "bnds" in str(v) or "bound" in str(v)], errors="ignore")
ds_anthro_summed = ds_anthro.sum(dim=sector_dim)

# ── 3. Annual mean flux for both (kg/m2/s averaged over each year) ───────────
# We take the annual MEAN of the rate, then multiply by seconds/year to get
# total mass per year. (Annual sum of a rate in kg/m2/s is not physical.)
print("\nCalculating annual mean flux...")

#ds_air_annual = ds_air_summed.groupby('time.year').mean()#.resample(time="YE").mean()
ds_anthro_annual = ds_anthro_summed.groupby('time.year').mean()#.resample(time="YE").mean()

#print(f"  AIR annual shape: {dict(ds_air_annual.dims)}")
print(f"  Anthro annual shape: {dict(ds_anthro_annual.dims)}")

# Replace cftime with pandas timestamps to avoid cftime arithmetic errors
import pandas as pd

#n_air = len(ds_air_annual.year)
n_anthro = len(ds_anthro_annual.year)

# Extract start year from first time value
#air_start = int(str(ds_air_annual.year.values[0])[:4])
anthro_start = int(str(ds_anthro_annual.year.values[0])[:4])

#ds_air_annual["time"] = pd.date_range(f"{air_start}-01-01", periods=n_air, freq="YS")
#ds_anthro_annual["time"] = pd.date_range(f"{anthro_start}-01-01", periods=n_anthro, freq="YS")

#print(f"  AIR time: {ds_air_annual.year.values[0]} to {ds_air_annual.year.values[-1]}")
print(f"  Anthro time: {ds_anthro_annual.year.values[0]} to {ds_anthro_annual.year.values[-1]}")

# ── 4. Rename to CO2 ────────────────────────────────────────────────────────
print("\nRenaming variables to 'CO2'...")
#ds_air_annual = rename_to_co2(ds_air_annual)
ds_anthro_annual = rename_to_co2(ds_anthro_annual)

# ── 5. Sum AIR + anthro on shared time range ─────────────────────────────────
print("\nCombining AIR + anthro emissions...")
#ds_air_aligned, ds_anthro_aligned = xr.align(ds_air_annual, ds_anthro_annual, join="inner")
ds_total =  ds_anthro_aligned
print(f"  Combined time range: {str(ds_total.year.values[0])[:10]} to {str(ds_total.year.values[-1])[:10]}")

# ── 6. Convert kg/m²/s -> Gt CO2 per grid point per year ────────────────────
print("\nConverting kg/m²/s -> Gt CO2 per grid point...")

lat = ds_total.lat.values
lon = ds_total.lon.values
area_m2 = compute_grid_cell_area(lat, lon)  # (nlat, nlon)

# Make it an xarray DataArray aligned with the dataset
area_da = xr.DataArray(area_m2, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

# Conversion:
#   kg/m²/s  *  m²  *  s/yr  /  (kg/Gt)  =  Gt/yr per grid cell
ds_total["CO2"] = ds_total["CO2"] * area_da * SECONDS_PER_YEAR / KG_PER_GT

ds_total["CO2"].attrs["units"] = "Gt CO2 / year / gridpoint"
ds_total["CO2"].attrs["long_name"] = "Annual CO2 emissions per grid point"

print(f"  Global total first year: {float(ds_total['CO2'].isel(year=0).sum()):.4f} Gt CO2/yr")
print(f"  Global total last year:  {float(ds_total['CO2'].isel(year=-1).sum()):.4f} Gt CO2/yr")

# ── 7. Cumulative sum over time ─────────────────────────────────────────────
print("\nComputing cumulative sum over time...")
print(ds_total)
#ds_total["CO2"] = ds_total["CO2"].cumsum(dim="year")

ds_total["CO2"].attrs["units"] = "Gt CO2 (cumulative)"
ds_total["CO2"].attrs["long_name"] = "Cumulative CO2 emissions per grid point"

print(f"  Global cumulative at last timestep: {float(ds_total['CO2'].isel(year=-1).sum()):.4f} Gt CO2")

# ── 8. Save ─────────────────────────────────────────────────────────────────
# Drop any leftover cftime-based variables (time_bnds, etc.) that cause serialization errors
drop_vars = [v for v in ds_total.coords if "bnds" in str(v) or "bound" in str(v)]
drop_vars += [v for v in ds_total.data_vars if "bnds" in str(v) or "bound" in str(v)]
if drop_vars:
    print(f"  Dropping leftover variables: {drop_vars}")
    ds_total = ds_total.drop_vars(drop_vars)

# Force compute from dask to numpy before writing
ds_total = ds_total.compute()

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "CO2_cumulative_Gt_per_gridpoint_"+exp+".nc")
print(ds_total)
ds_total.to_netcdf(out_path)
print(f"\nSaved: {out_path}")
print("Done!")