"""
Process CO2 emission input files from CEDS/input4MIPs:
1. Open AIR-anthro files, sum along 'level'
2. Open anthro files, sum along 'sector'
3. Calculate annual mean flux for both
4. Add AIR + anthro together
5. Convert kg/m2/s -> Gt CO2 per grid point per year (using grid cell areas)
6. Compute cumulative sum over time
"""

import lumi_paths as L
import xarray as xr
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", help="Experiment name, e.g. hist or ssp370")
parser.add_argument("--species", default="SO2", choices=["SO2", "BC"],
                    help="Aerosol species to process. SO2→var 'SUL' (existing), "
                         "BC→var 'BC' (3rd cond channel). Drives the input glob, "
                         "output var name, and output filename.")
args = parser.parse_args()

exp = args.exp
SPECIES = args.species
# Output variable name: SO2 keeps the legacy 'SUL' name; BC is its own channel.
OUT_VAR = "SUL" if SPECIES == "SO2" else "BC"
# ── Configure paths ──────────────────────────────────────────────────────────
# Override via env to run off a local mount (e.g. /mnt/lumi_sc2) instead of /scratch.
INPUT_DIR = os.environ.get(
    "EMUL_INPUT_DIR", f"{L.DATA}/emission_data/inputs4mips/")
OUTPUT_DIR = os.environ.get(
    "EMUL_OUTPUT_DIR", f"{L.DATA}/")

# Per-(species, exp) surface-anthro input glob. AIR-anthro stays omitted for both
# species (matches the existing SO2/SUL channel, which uses surface anthro only).
#
# BOTH SPECIES USE CEDS-2017 FOR hist, AND THAT MATTERS.
# The IAMC ScenarioMIP files (all SSPs, both species) are harmonised to CEDS-2017,
# so the historical source must be CEDS-2017 or the hist→ssp junction steps.
# BC previously used CEDS-CMIP-2025-04-18 — eight years newer, with anthropogenic
# BC revised substantially downward — which put a +35% discontinuity at 2015 into
# the BC conditioning channel, at exactly the year every scenario branches.
# Measured global anthro BC at the junction (Tg/yr):
#     CEDS-2017 2014 = 8.012  vs IAMC 2015 = 7.986  -> ratio 0.997
#     CEDS-2025 2014 = 5.917  vs IAMC 2015 = 7.986  -> ratio 1.350
# 0.997 is identical for ssp370/ssp126/ssp245, confirming the harmonisation.
# Sector structure is unchanged between the two BC releases (8 sectors, same ids,
# 720x360, kg m-2 s-1), so the sector sum below is unaffected by the swap.
# CEDS-2017 has two extra files (1750-1849); concat_and_regrid.py:217 clips BC to
# slice(1850, 2014), so they are harmless.
_ANTHRO_PATTERNS = {
    ("SO2", "hist"):   "SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc",
    ("SO2", "ssp370"): "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc",
    ("SO2", "ssp126"): "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc",
    ("SO2", "ssp245"): "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc",
    ("BC",  "hist"):   "BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc",
    ("BC",  "ssp370"): "BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc",
    ("BC",  "ssp126"): "BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc",
    ("BC",  "ssp245"): "BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc",
}
if (SPECIES, exp) not in _ANTHRO_PATTERNS:
    raise SystemExit(f"No input glob configured for species={SPECIES} exp={exp}")
ANTHRO_PATTERN = os.path.join(INPUT_DIR, _ANTHRO_PATTERNS[(SPECIES, exp)])
# Clip BC CEDS-2025 hist (runs to 2023) to ≤2014 so the hist channel ends where
# the SO2 channel does; ssp370 IAMC BC supplies ≥2015 downstream.
CLIP_HIST_2014 = (SPECIES == "BC" and exp == "hist")
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


def rename_to_out(ds):
    """Rename the main data variable to the output var name (OUT_VAR)."""
    data_vars = [v for v in ds.data_vars if "bnds" not in v and "bound" not in v]
    assert len(data_vars) >= 1, f"No data variables found: {list(ds.data_vars)}"
    var_name = data_vars[0]
    print(f"  Renaming '{var_name}' -> '{OUT_VAR}'")
    return ds.rename({var_name: OUT_VAR})


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

ds_anthro = xr.open_mfdataset(anthro_files, combine="by_coords")#.load()
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

# Clip BC CEDS-2025 hist (extends to 2023) to ≤2014 so it ends where the SO2
# hist channel does; ssp370 IAMC BC supplies ≥2015 in concat_and_regrid.py.
if CLIP_HIST_2014:
    ds_anthro_annual = ds_anthro_annual.sel(year=ds_anthro_annual.year <= 2014)
    print(f"  [BC] clipped hist to ≤2014 → {ds_anthro_annual.year.values[-1]}")

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

# ── 4. Rename to output var ─────────────────────────────────────────────────
print(f"\nRenaming variables to '{OUT_VAR}'...")
#ds_air_annual = rename_to_out(ds_air_annual)
ds_anthro_annual = rename_to_out(ds_anthro_annual)

# ── 5. Sum AIR + anthro on shared time range ─────────────────────────────────
print("\nCombining AIR + anthro emissions...")
#ds_air_aligned, ds_anthro_aligned = xr.align(ds_air_annual, ds_anthro_annual, join="inner")
ds_total =  ds_anthro_annual#ds_anthro_aligned
print(f"  Combined time range: {str(ds_total.year.values[0])[:10]} to {str(ds_total.year.values[-1])[:10]}")

# ── 6. Convert kg/m²/s -> Gt species per grid point per year ────────────────
print(f"\nConverting kg/m²/s -> Gt {SPECIES} per grid point...")

lat = ds_total.lat.values
lon = ds_total.lon.values
area_m2 = compute_grid_cell_area(lat, lon)  # (nlat, nlon)

# Make it an xarray DataArray aligned with the dataset
area_da = xr.DataArray(area_m2, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

# Conversion:
#   kg/m²/s  *  m²  *  s/yr  /  (kg/Gt)  =  Gt/yr per grid cell
ds_total[OUT_VAR] = ds_total[OUT_VAR] * area_da * SECONDS_PER_YEAR / KG_PER_GT

ds_total[OUT_VAR].attrs["units"] = f"Gt {SPECIES} / year / gridpoint"
ds_total[OUT_VAR].attrs["long_name"] = f"Annual {SPECIES} emissions per grid point"

print(f"  Global total first year: {float(ds_total[OUT_VAR].isel(year=0).sum()):.4f} Gt {SPECIES}/yr")
print(f"  Global total last year:  {float(ds_total[OUT_VAR].isel(year=-1).sum()):.4f} Gt {SPECIES}/yr")

# ── 7. Cumulative sum over time (DISABLED — aerosols are annual, like SUL) ───
# BC is NOT cumulative (annual emissions, same as SUL); cumsum stays commented.
print("\n(annual emissions; cumulative sum intentionally disabled)")
print(ds_total)
#ds_total[OUT_VAR] = ds_total[OUT_VAR].cumsum(dim="year")

# ── 8. Save ─────────────────────────────────────────────────────────────────
# Drop any leftover cftime-based variables (time_bnds, etc.) that cause serialization errors
drop_vars = [v for v in ds_total.coords if "bnds" in str(v) or "bound" in str(v)]
drop_vars += [v for v in ds_total.data_vars if "bnds" in str(v) or "bound" in str(v)]
if drop_vars:
    print(f"  Dropping leftover variables: {drop_vars}")
    ds_total = ds_total.drop_vars(drop_vars)

# Force compute from dask to numpy before writing
#ds_total = ds_total.compute()

os.makedirs(OUTPUT_DIR, exist_ok=True)
# SO2 keeps its legacy filename; BC writes BC_per_gridpoint_<exp>.nc.
if SPECIES == "SO2":
    out_path = os.path.join(OUTPUT_DIR, "SO2_cumulative_Gt_per_gridpoint_" + exp + ".nc")
else:
    out_path = os.path.join(OUTPUT_DIR, "BC_per_gridpoint_" + exp + ".nc")
print(ds_total)
ds_total.to_netcdf(out_path)
print(f"\nSaved: {out_path}")
print("Done!")
