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
parser.add_argument("--build-bc", action="store_true",
                    help="BC mode: regrid BC_per_gridpoint_*.nc to the target grid and "
                         "inject it as a 3rd data_var into COPIES of the existing "
                         "emissions_*_only_timefixed.nc files (written as *_bc.nc). "
                         "CO2/SUL are carried through byte-identical; originals untouched.")
args = parser.parse_args()

DATA_DIR = args.data_dir
TARGET_FILE = args.target
OUTPUT_DIR = args.output_dir or DATA_DIR
SCENARIOS = args.scenarios

CO2_HIST = os.path.join(DATA_DIR, "CO2_cumulative_Gt_per_gridpoint_hist.nc")
SO2_HIST = os.path.join(DATA_DIR, "SO2_cumulative_Gt_per_gridpoint_hist.nc")
BC_HIST = os.path.join(DATA_DIR, "BC_per_gridpoint_hist.nc")  # 3rd cond channel (CEDS-2025, ≤2014 clipped upstream)

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
        has_bc = "BC" in ds_regridded.data_vars   # BC merged in upstream (3rd channel)
        #----splot file to ghg and hist only ans hist
        ds=ds_regridded.sel(year=slice(1850,2050))
        ds_ghg=ds.copy()
        print(ds_ghg)
        ds_aero=ds.copy()
        ds_ghg['SUL'] *=0
        ds_ghg['SUL'] += ds['SUL'].isel(year=0)

        ds_aero['CO2'] *=0
        ds_aero['CO2'] += ds['CO2'].isel(year=0)

        # BC (3rd cond channel): GHG holds ALL aerosols fixed → zero BC and pin to
        # year-0 (mirror SUL above). AAER varies ALL anthropogenic aerosols incl.
        # BC, so ds_aero KEEPS BC varying (only CO2 is zeroed). hist carries BC
        # as-is. Guarded on has_bc so CO2/SUL stay byte-identical when BC is absent.
        if has_bc:
            ds_ghg['BC'] *= 0
            ds_ghg['BC'] += ds['BC'].isel(year=0)
            print("  [BC] ghg: zeroed + pinned to year-0; aaer: kept varying")

        ds_hist = ds_regridded.sel(year=slice(1850,2014))
    
        hist_only_file = os.path.join(OUTPUT_DIR, f"emissions_hist_only_timefixed.nc")
        ghg_only_file = os.path.join(OUTPUT_DIR, f"emissions_ghg_only_timefixed.nc")
        aaer_only_file = os.path.join(OUTPUT_DIR, f"emissions_aaer_only_timefixed.nc")
        
        ds_ghg.to_netcdf(ghg_only_file)
        ds_aero.to_netcdf(aaer_only_file)
        ds_hist.to_netcdf(hist_only_file)  

# ── BC (3rd cond channel) build path ─────────────────────────────────────────
# Reconstructs the original CO2/SO2 recipe (concat hist≤2014 + ssp≥2015, lon
# convention fix, xesmf bilinear periodic regrid to the f09 192x288 target grid —
# :109-126) for BC, then INJECTS the regridded BC into copies of the existing
# emissions_*_only_timefixed.nc cond files. CO2/SUL are carried through unchanged
# so old TREFHT runs stay reproducible against the untouched originals.

def _regrid_to_target(ds):
    """Lon-convention fix + bilinear periodic regrid to target grid (mirror :109-126)."""
    src_lon = ds.lon.values
    src_is_360 = float(src_lon.max()) > 180
    if src_is_360 != tgt_is_360:
        if tgt_is_360 and not src_is_360:
            print("  Converting source lon: -180..180 -> 0..360")
            ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
        else:
            print("  Converting source lon: 0..360 -> -180..180")
            ds = ds.assign_coords(lon=((ds.lon + 180) % 360 - 180)).sortby("lon")
    regridder = xe.Regridder(ds, target_grid, method="bilinear", periodic=True)
    return regridder(ds, keep_attrs=True)


def _build_bc_concat(scenario):
    """hist CEDS BC (≤2014) + scenario IAMC BC (≥2015), regridded to target grid.
    Mirrors the SO2 recipe EXACTLY: annual emissions, NO cumsum, and NO
    hist-endpoint add (the :83-84 add was a CO2-cumulative-only artefact and must
    NOT be applied to an annual aerosol channel). Returns the BC DataArray on the
    target grid with a 'year' coord spanning 1850-2100."""
    print(f"\n=== Building BC concat: {scenario} ===")
    bc_hist = xr.open_dataset(BC_HIST).sel(year=slice(1850, 2014))
    bc_ssp_path = os.path.join(DATA_DIR, f"BC_per_gridpoint_{scenario}.nc")
    bc_ssp = xr.open_dataset(bc_ssp_path).sel(year=slice(2015, 2100))
    bc_ssp = bc_ssp.interp(
        year=np.arange(int(bc_ssp.year.values[0]), int(bc_ssp.year.values[-1]) + 1),
        method="linear",
    )
    bc_ssp = bc_ssp.sel(year=bc_ssp.year > last_hist_year)
    bc = xr.concat([bc_hist, bc_ssp], dim="year").sel(year=slice(1850, 2100))
    # Measure the CEDS→IAMC 2015 junction on the NATIVE grid (do NOT smooth here;
    # σ-smoothing + 5-EOF PCA downstream handle it, same as SUL — aaer_2015_spike).
    g14 = float(bc["BC"].sel(year=2014).sum())
    g15 = float(bc["BC"].sel(year=2015).sum())
    print(f"  [{scenario}] BC global sum 2014->2015 junction: "
          f"{g14:.5f} -> {g15:.5f} Gt/yr ({100 * (g15 - g14) / g14:+.2f}%)")
    bc = _regrid_to_target(bc)
    bc["BC"] = bc["BC"].fillna(0.0)  # unmapped pole points -> 0 (matches SO2/SUL path)
    return bc["BC"]


def _inject_bc(src_file, out_file, bc_da, *, zero_pin=False):
    """Copy an existing cond file and add BC as a 3rd var on SUL's exact
    dims/coords. CO2/SUL pass through untouched. bc_da: (T, lat, lon)
    DataArray on the regrid target grid (coords used for alignment check)."""
    ds = xr.open_dataset(src_file)
    sul = ds["SUL"]
    bc_arr = bc_da.values
    assert bc_arr.shape == sul.shape, f"{out_file}: BC {bc_arr.shape} != SUL {sul.shape}"
    # bc_da was regridded to TARGET_FILE's grid; stamping its values onto this
    # file's coords is only valid if the grids are truly identical (rounding
    # noise aside) — otherwise BC would be geographically misaligned with
    # CO2/SUL while every shape/NaN check still passes.
    if not (np.allclose(bc_da.lat.values, sul.lat.values, atol=1e-6)
            and np.allclose(bc_da.lon.values, sul.lon.values, atol=1e-6)):
        raise ValueError(
            f"{out_file}: lat/lon differ from the BC regrid target grid "
            f"(max lat diff {np.abs(bc_da.lat.values - sul.lat.values).max():g}, "
            f"max lon diff {np.abs(bc_da.lon.values - sul.lon.values).max():g})"
        )
    if zero_pin:  # GHG holds ALL aerosols fixed -> constant year-0 field (mirror SUL)
        bc_arr = np.broadcast_to(bc_arr[0:1], bc_arr.shape).copy()
    ds["BC"] = xr.DataArray(bc_arr, dims=sul.dims, coords=sul.coords)
    ds["BC"].attrs["units"] = "Gt BC / year / gridpoint"
    ds["BC"].attrs["long_name"] = "Annual BC emissions per grid point (3rd cond channel)"
    ds.to_netcdf(out_file)
    print(f"  wrote {out_file}  (vars now {list(ds.data_vars)})")


def build_bc_cond_files():
    bc = {sc: _build_bc_concat(sc) for sc in ["ssp370", "ssp126", "ssp245"]}
    bc370 = bc["ssp370"]
    D = OUTPUT_DIR
    j = lambda n: os.path.join(D, n)
    print("\n=== Injecting BC into cond files ===")
    # hist (year 1850-2014): CEDS hist is scenario-independent
    _inject_bc(j("emissions_hist_only_timefixed.nc"),
               j("emissions_hist_only_timefixed_bc.nc"),
               bc370.sel(year=slice(1850, 2014)))
    # ssp-only files (time 2015-2100). ssp126 eval uses the CO2-fixed base
    # (ssp126_co2_cond_construction_bug → emissions_ssp126_only_timefixed_co2fix.nc,
    # eval_aero.py:99), so inject BC into THAT so CO2 matches what eval consumes.
    ssp_sources = {
        "ssp370": ("emissions_ssp370_only_timefixed.nc",        "emissions_ssp370_only_timefixed_bc.nc"),
        "ssp245": ("emissions_ssp245_only_timefixed.nc",        "emissions_ssp245_only_timefixed_bc.nc"),
        "ssp126": ("emissions_ssp126_only_timefixed_co2fix.nc", "emissions_ssp126_only_timefixed_co2fix_bc.nc"),
    }
    for sc, (src, out) in ssp_sources.items():
        _inject_bc(j(src), j(out), bc[sc].sel(year=slice(2015, 2100)))
    # ghg (year 1850-2050): BC held fixed at year-0 (mirror SUL zero+pin)
    _inject_bc(j("emissions_ghg_only_timefixed.nc"),
               j("emissions_ghg_only_timefixed_bc.nc"),
               bc370.sel(year=slice(1850, 2050)), zero_pin=True)
    # aaer (year 1850-2050): BC kept varying (AAER varies ALL anthro aerosols incl BC)
    _inject_bc(j("emissions_aaer_only_timefixed.nc"),
               j("emissions_aaer_only_timefixed_bc.nc"),
               bc370.sel(year=slice(1850, 2050)))


if args.build_bc:
    build_bc_cond_files()
else:
    for scenario in SCENARIOS:
        process_scenario(scenario)

print("\nDone!")
