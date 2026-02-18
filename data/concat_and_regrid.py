"""
Concat hist + ssp370 CO2 and SO2 files, merge into one dataset,
and regrid to match the target CESM2 grid using xesmf.

Usage:
    python concat_and_regrid.py --target /path/to/target_file.nc
"""

import argparse
import xarray as xr
import xesmf as xe
import os
import numpy as np

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Concat hist+ssp370 emissions and regrid to target grid")
parser.add_argument("--target", required=True, help="Path to a target NetCDF file to extract the grid from")
parser.add_argument("--data_dir", default="/mnt/lumi_sc/emulator_data/",
                    help="Directory containing the CO2/SO2 files")
parser.add_argument("--output", default=None, help="Output file path (default: <data_dir>/emissions_co2_so2_regridded.nc)")
args = parser.parse_args()

DATA_DIR = args.data_dir
TARGET_FILE = args.target
OUTPUT_FILE = args.output or os.path.join(DATA_DIR, "emissions_co2_so2_regridded.nc")

# ── File paths (adjust naming to match your make_co2_files.py output) ────────
CO2_HIST = os.path.join(DATA_DIR, "CO2_cumulative_Gt_per_gridpoint_hist.nc")
CO2_SSP = os.path.join(DATA_DIR, "CO2_cumulative_Gt_per_gridpoint_ssp370.nc")
SO2_HIST = os.path.join(DATA_DIR, "SO2_cumulative_Gt_per_gridpoint_hist.nc")
SO2_SSP = os.path.join(DATA_DIR, "SO2_cumulative_Gt_per_gridpoint_ssp370.nc")

# ── 1. Load and concat CO2 (hist + ssp370) ──────────────────────────────────
print("Loading CO2 files...")
ds_co2_hist = xr.open_dataset(CO2_HIST)
ds_co2_ssp = xr.open_dataset(CO2_SSP)
ds_co2_ssp = ds_co2_ssp.interp(year=np.arange(ds_co2_ssp.year.values[0], ds_co2_ssp.year.values[-1] + 1), method="linear")

print(f"  hist: {ds_co2_hist.year.values[0]}–{ds_co2_hist.year.values[-1]}")
print(f"  ssp370: {ds_co2_ssp.year.values[0]}–{ds_co2_ssp.year.values[-1]}")

# Drop overlapping years from ssp (keep hist as authoritative)
last_hist_year = int(ds_co2_hist.year.values[-1])
ds_co2_ssp = ds_co2_ssp.sel(year=ds_co2_ssp.year > last_hist_year)

# For cumulative CO2: ssp values need to continue from hist endpoint
# Add the last hist cumulative value to all ssp timesteps
if len(ds_co2_ssp.year) > 0:
    hist_endpoint = ds_co2_hist["CO2"].isel(year=-1)
    ds_co2_ssp["CO2"] = ds_co2_ssp["CO2"] + hist_endpoint

ds_co2 = xr.concat([ds_co2_hist, ds_co2_ssp], dim="year").sel(year=slice(1850,2100)).cumsum(dim="year")
print(f"  Combined CO2: {ds_co2.year.values[0]}–{ds_co2.year.values[-1]} ({len(ds_co2.year)} years)")

# ── 2. Load and concat SO2 (hist + ssp370) ──────────────────────────────────
print("\nLoading SO2 files...")
ds_so2_hist = xr.open_dataset(SO2_HIST)).sel(year=slice(1850,2014))
ds_so2_ssp = xr.open_dataset(SO2_SSP).sel(year=slice(2015,2100))
ds_so2_ssp = ds_so2_ssp.interp(year=np.arange(ds_so2_ssp.year.values[0], ds_so2_ssp.year.values[-1] + 1), method="linear")

print(f"  hist: {ds_so2_hist.year.values[0]}–{ds_so2_hist.year.values[-1]}")
print(f"  ssp370: {ds_so2_ssp.year.values[0]}–{ds_so2_ssp.year.values[-1]}")

# Drop overlapping years from ssp
ds_so2_ssp = ds_so2_ssp.sel(year=ds_so2_ssp.year > last_hist_year)

# SO2 is annual rate (not cumulative), so just concat directly
ds_so2 = xr.concat([ds_so2_hist, ds_so2_ssp], dim="year").sel(year=slice(1850,2100))
print(f"  Combined SO2: {ds_so2.year.values[0]}–{ds_so2.year.values[-1]} ({len(ds_so2.year)} years)")
print(ds_co2)
print(ds_so2)
# ── 3. Merge CO2 and SO2 into one dataset ───────────────────────────────────
print("\nMerging CO2 and SO2...")
ds_merged = xr.merge([ds_co2, ds_so2])
print(f"  Variables: {list(ds_merged.data_vars)}")
print(f"  Dimensions: {dict(ds_merged.dims)}")

# ── 4. Load target grid ─────────────────────────────────────────────────────
print(f"\nLoading target grid from: {TARGET_FILE}")
ds_target = xr.open_dataset(TARGET_FILE)

# Extract target lat/lon (try common names)
target_lat = None
target_lon = None
for lat_name in ["lat", "latitude"]:
    if lat_name in ds_target:
        target_lat = ds_target[lat_name]
        break
for lon_name in ["lon", "longitude"]:
    if lon_name in ds_target:
        target_lon = ds_target[lon_name]
        break

assert target_lat is not None and target_lon is not None, \
    f"Could not find lat/lon in target file. Coords: {list(ds_target.coords)}"

target_grid = xr.Dataset({"lat": target_lat, "lon": target_lon})
print(f"  Target grid: {len(target_lat.values)} lat x {len(target_lon.values)} lon")
print(f"  Target lat range: {float(target_lat.min()):.2f} to {float(target_lat.max()):.2f}")
print(f"  Target lon range: {float(target_lon.min()):.2f} to {float(target_lon.max()):.2f}")

# ── 5. Ensure compatible lon convention (0-360 vs -180-180) ──────────────────
src_lon = ds_merged.lon.values
tgt_lon = target_lon.values

src_is_360 = float(src_lon.max()) > 180
tgt_is_360 = float(tgt_lon.max()) > 180

if src_is_360 != tgt_is_360:
    if tgt_is_360 and not src_is_360:
        print("  Converting source lon: -180..180 -> 0..360")
        ds_merged = ds_merged.assign_coords(lon=(ds_merged.lon % 360))
        ds_merged = ds_merged.sortby("lon")
    elif not tgt_is_360 and src_is_360:
        print("  Converting source lon: 0..360 -> -180..180")
        ds_merged = ds_merged.assign_coords(lon=((ds_merged.lon + 180) % 360 - 180))
        ds_merged = ds_merged.sortby("lon")
ds_merged.to_netcdf(OUTPUT_FILE)

# ── 6. Regrid with xesmf ────────────────────────────────────────────────────
print("\nBuilding xesmf regridder (bilinear)...")
regridder = xe.Regridder(ds_merged, target_grid, method="bilinear", periodic=True)
print(f"  {regridder}")

print("Regridding...")
ds_regridded = regridder(ds_merged, keep_attrs=True)

print(f"  Output shape: {dict(ds_regridded.dims)}")

# ── 7. Save ─────────────────────────────────────────────────────────────────
print(f"\nSaving to: {OUTPUT_FILE}")
ds_regridded = ds_regridded.compute().sel(year=slice(1850,2100))
ds_regridded.to_netcdf(OUTPUT_FILE)
print("Done!")
