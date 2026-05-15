"""
Concat hist + SSP CO2 and SO2 files, merge into one dataset,
and regrid to match the target CESM2 grid using xesmf.

Runs once per scenario (default: ssp370 and ssp126) and writes
one regridded file per scenario.

Usage:
    python concat_and_regrid.py --target /path/to/target_file.nc
    python concat_and_regrid.py --target ... --scenarios ssp370 ssp126
"""

import argparse
import xarray as xr
import xesmf as xe
import os
import numpy as np

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Concat hist+SSP emissions and regrid to target grid")
parser.add_argument("--target", required=True, help="Path to a target NetCDF file to extract the grid from")
parser.add_argument("--data_dir", default="/mnt/lumi_sc2/emulator_data/",
                    help="Directory containing the CO2/SO2 files")
parser.add_argument("--output_dir", default=None,
                    help="Output directory (default: same as --data_dir)")
parser.add_argument("--scenarios", nargs="+", default=["ssp370", "ssp126"],
                    help="SSP scenarios to process (default: ssp370 ssp126)")
args = parser.parse_args()

DATA_DIR = args.data_dir
TARGET_FILE = args.target
OUTPUT_DIR = args.output_dir or DATA_DIR
SCENARIOS = args.scenarios

CO2_HIST = os.path.join(DATA_DIR, "CO2_cumulative_Gt_per_gridpoint_hist.nc")
SO2_HIST = os.path.join(DATA_DIR, "SO2_cumulative_Gt_per_gridpoint_hist.nc")

# ── Load hist once (shared across scenarios) ─────────────────────────────────
print("Loading hist files...")
ds_co2_hist = xr.open_dataset(CO2_HIST).sel(year=slice(1850, 2014))
ds_so2_hist = xr.open_dataset(SO2_HIST).sel(year=slice(1850, 2014))
last_hist_year = int(ds_co2_hist.year.values[-1])
print(f"  CO2 hist: {ds_co2_hist.year.values[0]}–{ds_co2_hist.year.values[-1]}")
print(f"  SO2 hist: {ds_so2_hist.year.values[0]}–{ds_so2_hist.year.values[-1]}")

# ── Load target grid once ────────────────────────────────────────────────────
print(f"\nLoading target grid from: {TARGET_FILE}")
ds_target = xr.open_dataset(TARGET_FILE)
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

tgt_lon = target_lon.values
tgt_is_360 = float(tgt_lon.max()) > 180


def process_scenario(scenario: str) -> None:
    print(f"\n=== Processing scenario: {scenario} ===")
    '''
    CO2_SSP = os.path.join(DATA_DIR, f"CO2_cumulative_Gt_per_gridpoint_{scenario}.nc")
    SO2_SSP = os.path.join(DATA_DIR, f"SO2_cumulative_Gt_per_gridpoint_{scenario}.nc")

    # ── 1. CO2 ────────────────────────────────────────────────────────────────
    print("Loading CO2...")
    ds_co2_ssp = xr.open_dataset(CO2_SSP).sel(year=slice(2015, 2100))
    ds_co2_ssp = ds_co2_ssp.interp(
        year=np.arange(ds_co2_ssp.year.values[0], ds_co2_ssp.year.values[-1] + 1),
        method="linear",
    )
    ds_co2_ssp = ds_co2_ssp.sel(year=ds_co2_ssp.year > last_hist_year)
    if len(ds_co2_ssp.year) > 0:
        hist_endpoint = ds_co2_hist["CO2"].isel(year=-1)
        ds_co2_ssp["CO2"] = ds_co2_ssp["CO2"] + hist_endpoint

    ds_co2 = xr.concat([ds_co2_hist, ds_co2_ssp], dim="year").sel(year=slice(1850, 2100))
    ds_co2.to_netcdf(os.path.join(OUTPUT_DIR, f"emissions_co2_concat_{scenario}.nc"))
    ds_co2_cumsum = ds_co2.cumsum(dim="year", keep_attrs=True)
    ds_co2_cumsum["year"] = ds_co2["year"]
    ds_co2 = ds_co2_cumsum
    ds_co2.to_netcdf(os.path.join(OUTPUT_DIR, f"emissions_co2_cumsum_{scenario}.nc"))
    print(f"  Combined CO2: {ds_co2.year.values[0]}–{ds_co2.year.values[-1]} ({len(ds_co2.year)} years)")

    # ── 2. SO2 ────────────────────────────────────────────────────────────────
    print("Loading SO2...")
    ds_so2_ssp = xr.open_dataset(SO2_SSP).sel(year=slice(2015, 2100))
    ds_so2_ssp = ds_so2_ssp.interp(
        year=np.arange(ds_so2_ssp.year.values[0], ds_so2_ssp.year.values[-1] + 1),
        method="linear",
    )
    ds_so2_ssp = ds_so2_ssp.sel(year=ds_so2_ssp.year > last_hist_year)
    ds_so2 = xr.concat([ds_so2_hist, ds_so2_ssp], dim="year").sel(year=slice(1850, 2100))
    print(f"  Combined SO2: {ds_so2.year.values[0]}–{ds_so2.year.values[-1]} ({len(ds_so2.year)} years)")

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    ds_merged = xr.merge([ds_co2, ds_so2])
    print(f"  Variables: {list(ds_merged.data_vars)}")

    # ── 4. Lon convention ─────────────────────────────────────────────────────
    src_lon = ds_merged.lon.values
    src_is_360 = float(src_lon.max()) > 180
    if src_is_360 != tgt_is_360:
        if tgt_is_360 and not src_is_360:
            print("  Converting source lon: -180..180 -> 0..360")
            ds_merged = ds_merged.assign_coords(lon=(ds_merged.lon % 360)).sortby("lon")
        else:
            print("  Converting source lon: 0..360 -> -180..180")
            ds_merged = ds_merged.assign_coords(lon=((ds_merged.lon + 180) % 360 - 180)).sortby("lon")

    pre_regrid_path = os.path.join(OUTPUT_DIR, f"emissions_co2_so2_merged_{scenario}.nc")
    ds_merged.to_netcdf(pre_regrid_path)

    # ── 5. Regrid ─────────────────────────────────────────────────────────────
    print("Building xesmf regridder (bilinear)...")
    regridder = xe.Regridder(ds_merged, target_grid, method="bilinear", periodic=True)
    ds_regridded = regridder(ds_merged, keep_attrs=True)
    print(f"  Output shape: {dict(ds_regridded.dims)}")
    
    # ── 6. Save ───────────────────────────────────────────────────────────────
    output_file = os.path.join(OUTPUT_DIR, f"emissions_co2_so2_regridded_{scenario}.nc")
    #ds_regridded = ds_regridded.compute().sel(year=slice(1850, 2100))
    #ds_regridded.to_netcdf(output_file)
    ds_regridded=xr.open_datset(output_file)
    print(f"Saved: {output_file}")
    
    # ── 7. SSP-only file (2015–2100, dim renamed year→time) ──────────────────
    # Cumulative values preserved — same integration from 1850, just clipped
    # in year range. Dim renamed so eval_aero.py's time_dim="time" works.
    ssp_only = (
        ds_regridded.sel(year=slice(2015, 2100))
                    .rename({"year": "time"})
    )
    
    ssp_only_file = os.path.join(OUTPUT_DIR, f"emissions_{scenario}_only_timefixed.nc")
    #ssp_only.to_netcdf(ssp_only_file)
    #print(f"Saved: {ssp_only_file}  ({ssp_only.dims['time']} years, 2015–2100)")
    '''
    if scenario == 'ssp370':
        output_file = os.path.join(OUTPUT_DIR, f"emissions_co2_so2_regridded_{scenario}.nc")
        ds_regridded=xr.open_dataset(output_file)
        #----splot file to ghg and hist only ans hist
        ds=ds_regridded.sel(year=slice(1850,2050))
        ds_ghg=ds.copy()
        print(ds_ghg)
        ds_aero=ds.copy()
        ds_ghg['SUL'] *=0
        ds_ghg['SUL'] += ds['SUL'].isel(year=0)
        
        ds_aero['CO2'] *=0
        ds_aero['CO2'] += ds['CO2'].isel(year=0)
        
        ds_hist = ds_regridded.sel(year=slice(1850,2014))
    
        hist_only_file = os.path.join(OUTPUT_DIR, f"emissions_hist_only_timefixed.nc")
        ghg_only_file = os.path.join(OUTPUT_DIR, f"emissions_ghg_only_timefixed.nc")
        aaer_only_file = os.path.join(OUTPUT_DIR, f"emissions_aaer_only_timefixed.nc")
        
        ds_ghg.to_netcdf(ghg_only_file)
        ds_aero.to_netcdf(aaer_only_file)
        ds_hist.to_netcdf(hist_only_file)  

for scenario in SCENARIOS:
    process_scenario(scenario)

print("\nDone!")
