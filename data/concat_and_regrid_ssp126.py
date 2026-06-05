"""
Concat hist + ssp126 CO2/SO2, integrate, merge, and regrid to the CESM2 grid.

This is a CLEANED, CO2-FIXED version of concat_and_regrid.py for ssp126.

WHAT WAS WRONG (see ssp126_co2_cond_construction_bug):
  make_co2_files.py writes ANNUAL Gt/gridpoint/yr (its line-181 cumsum is
  commented out, despite the "CO2_cumulative_*" filename). The old
  concat_and_regrid.py then did, for the SSP segment:

      hist_endpoint = ds_co2_hist["CO2"].isel(year=-1)   # = hist 2014 ANNUAL value
      ds_co2_ssp["CO2"] = ds_co2_ssp["CO2"] + hist_endpoint
      ...
      ds_co2 = cumsum(concat(hist_annual, ssp_annual))

  The "+ hist_endpoint" offset is a LEFTOVER from when the inputs were already
  cumulative (then you offset the SSP cumulative to continue from the hist
  endpoint). With ANNUAL inputs it adds the 2014 annual value to EVERY future
  year before the single cumsum, injecting a spurious constant annual "floor"
  -> a fake linear ramp in the cumulative. ssp126 is hit hardest (its real
  emissions are smallest and go net-negative late century), so its cumulative
  CO2 keeps climbing (1014 @2070 -> 1216 @2100) instead of plateauing.

THE FIX:
  Inputs are annual, so the single ``cumsum`` over the concatenated annual
  series IS the integration from 1850 — no per-year offset is needed. The
  "+ hist_endpoint" line is removed. CO2 is cumsum'd (warming ~ cumulative
  CO2); SO2/SUL is NOT cumsum'd (instantaneous aerosol loading) — same
  asymmetry as the original.

Output: by default writes to a NEW filename (``emissions_ssp126_only_timefixed
{suffix}.nc`` with suffix ``_co2fix``) so it does NOT clobber the existing cond
file. Repoint config_data.yaml / eval_aero.py at it (and retrain) when ready.

Usage:
    python concat_and_regrid_ssp126.py --target /path/to/cesm2_grid_file.nc
    # overwrite the live filename instead (forces retrain):
    python concat_and_regrid_ssp126.py --target ... --out-suffix ""
"""

import argparse
import xarray as xr
import os
import numpy as np

# xesmf is optional: only needed for --regrid xesmf. The --regrid interp fallback
# (xarray bilinear) needs no extra deps, so the script runs on a plain local mount.
try:
    import xesmf as xe
    _HAVE_XESMF = True
except Exception:
    xe = None
    _HAVE_XESMF = False

# Default grid template + data dir on the local LUMI scratch mount, so this runs
# locally with no args. (The target file is read only for its lat/lon.)
_MOUNT = "/mnt/lumi_sc2/emulator_data"

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Concat hist+ssp126 emissions and regrid (CO2-fixed)")
parser.add_argument("--target", default=os.path.join(_MOUNT, "emissions_ssp126_only_timefixed.nc"),
                    help="NetCDF file to read the target lat/lon grid from "
                         "(default: the existing regridded ssp126 file on /mnt/lumi_sc2)")
parser.add_argument("--data_dir", default=_MOUNT + "/",
                    help="Directory containing the CO2/SO2 files (default: /mnt/lumi_sc2/emulator_data/)")
parser.add_argument("--output_dir", default=None,
                    help="Output directory (default: same as --data_dir)")
parser.add_argument("--scenarios", nargs="+", default=["ssp126"],
                    help="SSP scenarios to process (default: ssp126)")
parser.add_argument("--out-suffix", default="_co2fix",
                    help="Suffix on output filenames so the live cond file is not "
                         "clobbered. Pass '' to overwrite the live names (forces retrain).")
parser.add_argument("--regrid", choices=["auto", "xesmf", "interp"], default="auto",
                    help="Regridder: 'xesmf' (bilinear, periodic, needs xesmf), "
                         "'interp' (xarray bilinear, periodic-padded, no extra deps), "
                         "'auto' = xesmf if importable else interp (default).")
args = parser.parse_args()

REGRID = args.regrid if args.regrid != "auto" else ("xesmf" if _HAVE_XESMF else "interp")
if REGRID == "xesmf" and not _HAVE_XESMF:
    raise SystemExit("--regrid xesmf requested but xesmf is not importable. "
                     "Use --regrid interp (no extra deps) or run in the LUMI container.")
print(f"[regrid] method = {REGRID}" + ("" if _HAVE_XESMF else "  (xesmf not available)"))

DATA_DIR = args.data_dir
TARGET_FILE = args.target
OUTPUT_DIR = args.output_dir or DATA_DIR
SCENARIOS = args.scenarios
SUFFIX = args.out_suffix

CO2_HIST = os.path.join(DATA_DIR, "CO2_cumulative_Gt_per_gridpoint_hist.nc")
SO2_HIST = os.path.join(DATA_DIR, "SO2_cumulative_Gt_per_gridpoint_hist.nc")

# ── Load hist once (shared across scenarios) ─────────────────────────────────
print("Loading hist files...")
ds_co2_hist = xr.open_dataset(CO2_HIST).sel(year=slice(1850, 2014))
ds_so2_hist = xr.open_dataset(SO2_HIST).sel(year=slice(1850, 2014))
last_hist_year = int(ds_co2_hist.year.values[-1])
print(f"  CO2 hist: {ds_co2_hist.year.values[0]}-{ds_co2_hist.year.values[-1]} (ANNUAL)")
print(f"  SO2 hist: {ds_so2_hist.year.values[0]}-{ds_so2_hist.year.values[-1]} (ANNUAL)")

# ── Load target grid once ────────────────────────────────────────────────────
print(f"\nLoading target grid from: {TARGET_FILE}")
ds_target = xr.open_dataset(TARGET_FILE)
target_lat = next((ds_target[n] for n in ["lat", "latitude"] if n in ds_target), None)
target_lon = next((ds_target[n] for n in ["lon", "longitude"] if n in ds_target), None)
assert target_lat is not None and target_lon is not None, \
    f"Could not find lat/lon in target file. Coords: {list(ds_target.coords)}"
target_grid = xr.Dataset({"lat": target_lat, "lon": target_lon})
print(f"  Target grid: {len(target_lat.values)} lat x {len(target_lon.values)} lon")

tgt_lon = target_lon.values
tgt_is_360 = float(tgt_lon.max()) > 180


def _global_sum_co2(ds):
    """Diagnostic: CO2 summed over all non-year dims, per year (for plateau check)."""
    g = ds["CO2"].sum(dim=[d for d in ds["CO2"].dims if d != "year"]).values
    yr = ds["year"].values
    return yr, g


def _interp_regrid(ds, tlat, tlon):
    """Bilinear regrid via xarray linear interp — no xesmf/ESMF dependency.

    Both source and target are regular lat/lon grids, so xarray's 2-D linear
    interp == bilinear. We pad the source longitude periodically (one wrapped
    column on each side) so points near the 0/360 (or ±180) seam interpolate
    across the wrap instead of clamping. Matches the original xe.Regridder
    (method="bilinear", periodic=True) intent; not conservative, same as before.
    """
    ds = ds.sortby("lat").sortby("lon")
    lon = ds["lon"].values
    span = 360.0
    left = ds.isel(lon=-1).assign_coords(lon=float(lon[-1]) - span)
    right = ds.isel(lon=0).assign_coords(lon=float(lon[0]) + span)
    ds_pad = xr.concat([left, ds, right], dim="lon")
    out = ds_pad.interp(lat=tlat, lon=tlon, method="linear")
    return out


def process_scenario(scenario: str) -> None:
    print(f"\n=== Processing scenario: {scenario} (CO2-fixed) ===")
    CO2_SSP = os.path.join(DATA_DIR, f"CO2_cumulative_Gt_per_gridpoint_{scenario}.nc")
    SO2_SSP = os.path.join(DATA_DIR, f"SO2_cumulative_Gt_per_gridpoint_{scenario}.nc")

    # ── 1. CO2 (ANNUAL inputs → concat hist+ssp → single cumsum = integration) ─
    print("Loading CO2 (annual)...")
    ds_co2_ssp = xr.open_dataset(CO2_SSP).sel(year=slice(2015, 2100))
    ds_co2_ssp = ds_co2_ssp.interp(
        year=np.arange(ds_co2_ssp.year.values[0], ds_co2_ssp.year.values[-1] + 1),
        method="linear",
    )
    ds_co2_ssp = ds_co2_ssp.sel(year=ds_co2_ssp.year > last_hist_year)

    # FIX: NO "+ hist_endpoint" offset. Inputs are annual; the cumsum below
    # integrates hist+ssp from 1850 in one pass, which already carries the hist
    # accumulation into the future years correctly. Adding the 2014 annual value
    # to every future year (the old bug) injected a spurious cumulative ramp.

    ds_co2 = xr.concat([ds_co2_hist, ds_co2_ssp], dim="year").sel(year=slice(1850, 2100))
    ds_co2_cumsum = ds_co2.cumsum(dim="year", keep_attrs=True)
    ds_co2_cumsum["year"] = ds_co2["year"]
    ds_co2 = ds_co2_cumsum
    print(f"  Combined CO2: {ds_co2.year.values[0]}-{ds_co2.year.values[-1]} "
          f"({len(ds_co2.year)} years, cumulative)")

    # Plateau sanity check (native grid, pre-regrid).
    yr, g = _global_sum_co2(ds_co2)
    def at(y):  # cumulative CO2 at year y
        return float(g[int(np.argmin(np.abs(yr - y)))])
    print("  [check] cumulative CO2 global-sum: "
          f"2015={at(2015):.1f} 2050={at(2050):.1f} 2070={at(2070):.1f} "
          f"2090={at(2090):.1f} 2100={at(2100):.1f}")
    print(f"  [check] late slope 2070->2100 = {(at(2100)-at(2070))/30:.3f}/yr "
          f"(should be ~0 / declining for ssp126's net-negative emissions)")

    # ── 2. SO2 (annual, NOT cumsum'd — instantaneous aerosol loading) ─────────
    print("Loading SO2 (annual)...")
    ds_so2_ssp = xr.open_dataset(SO2_SSP).sel(year=slice(2015, 2100))
    ds_so2_ssp = ds_so2_ssp.interp(
        year=np.arange(ds_so2_ssp.year.values[0], ds_so2_ssp.year.values[-1] + 1),
        method="linear",
    )
    ds_so2_ssp = ds_so2_ssp.sel(year=ds_so2_ssp.year > last_hist_year)
    ds_so2 = xr.concat([ds_so2_hist, ds_so2_ssp], dim="year").sel(year=slice(1850, 2100))
    print(f"  Combined SO2: {ds_so2.year.values[0]}-{ds_so2.year.values[-1]} "
          f"({len(ds_so2.year)} years)")

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

    # ── 5. Regrid ─────────────────────────────────────────────────────────────
    if REGRID == "xesmf":
        print("Regridding with xesmf (bilinear, periodic)...")
        regridder = xe.Regridder(ds_merged, target_grid, method="bilinear", periodic=True)
        ds_regridded = regridder(ds_merged, keep_attrs=True).compute()
    else:
        print("Regridding with xarray bilinear interp (periodic-padded; no xesmf)...")
        ds_regridded = _interp_regrid(ds_merged, target_lat, target_lon).compute()
    ds_regridded = ds_regridded.sel(year=slice(1850, 2100))
    print(f"  Output dims: {dict(ds_regridded.sizes)}")

    regridded_file = os.path.join(OUTPUT_DIR, f"emissions_co2_so2_regridded_{scenario}{SUFFIX}.nc")
    ds_regridded.to_netcdf(regridded_file)
    print(f"  Saved: {regridded_file}")

    # ── 6. SSP-only file (2015-2100, dim renamed year→time for eval_aero) ──────
    ssp_only = ds_regridded.sel(year=slice(2015, 2100)).rename({"year": "time"})
    ssp_only_file = os.path.join(OUTPUT_DIR, f"emissions_{scenario}_only_timefixed{SUFFIX}.nc")
    ssp_only.to_netcdf(ssp_only_file)
    print(f"  Saved: {ssp_only_file}  ({ssp_only.sizes['time']} years, 2015-2100)")


for scenario in SCENARIOS:
    process_scenario(scenario)

print("\nDone!")
if SUFFIX:
    print(f"NOTE: outputs carry suffix '{SUFFIX}' so the live cond file is untouched. "
          f"Repoint config_data.yaml / eval_aero.py at the new file and retrain when ready.")
