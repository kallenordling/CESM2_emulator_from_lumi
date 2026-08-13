"""Build a mixed-scenario conditioning file: CO2 from ssp370 (high/rising
forcing), SUL + BC from ssp126 (declining aerosol trajectory). Mirrors the
real CMIP6 "ssp370-126aer" experiment design (high CO2/GHG forcing combined
with ssp126's aggressive aerosol cleanup) used to isolate the aerosol-removal
warming signal.

Sources (production cond files, both time/lat/lon-identical up to float
noise -- see coordinate check in the commit this script ships with):
  CO2      <- emissions_ssp370_only_timefixed_bc.nc
  SUL, BC  <- emissions_ssp126_only_timefixed_co2fix_bc.nc  (the ssp126 file
             eval_aero.py actually uses, post co2-construction-bug fix --
             see memory ssp126_co2_cond_construction_bug)

Output matches the production _bc.nc format exactly (time dim 2015-2100,
CO2/SUL/BC vars, same per-variable units/long_name attrs, regrid_method
attr) so it drops straight into an eval_aero.py experiment entry or a
config_data.yaml experiment_configs block with no other changes needed.
"""
import argparse

import xarray as xr

EMIS_DIR = "/home/nordling/mnt/lumi_sc2/emulator_data"
SSP370_FILE = f"{EMIS_DIR}/emissions_ssp370_only_timefixed_bc.nc"
SSP126_FILE = f"{EMIS_DIR}/emissions_ssp126_only_timefixed_co2fix_bc.nc"
OUT_FILE = f"{EMIS_DIR}/emissions_ssp370co2_ssp126aer_bc.nc"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--emis-dir", default=EMIS_DIR)
ap.add_argument("--start", type=int, default=None,
                help="first year to keep (default: all)")
ap.add_argument("--end", type=int, default=None,
                help="last year to keep. eval_aero.py takes the GENERATION "
                     "window straight off this file's time axis "
                     "(eval_aero.py:363), so capping it here is what stops the "
                     "emulator generating years the reference cannot cover — "
                     "e.g. --end 2079 for the RAMIP ssp370-126aer run, whose "
                     "truth stops in 2079 while the forcing runs to 2100")
ap.add_argument("--out", default=None,
                help="default emissions_ssp370co2_ssp126aer_bc[_<start>-<end>].nc")
args = ap.parse_args()

EMIS_DIR = args.emis_dir
SSP370_FILE = f"{EMIS_DIR}/emissions_ssp370_only_timefixed_bc.nc"
SSP126_FILE = f"{EMIS_DIR}/emissions_ssp126_only_timefixed_co2fix_bc.nc"
OUT_FILE = args.out or (
    f"{EMIS_DIR}/emissions_ssp370co2_ssp126aer_bc.nc"
    if args.start is None and args.end is None else
    f"{EMIS_DIR}/emissions_ssp370co2_ssp126aer_bc_"
    f"{args.start or 'start'}-{args.end or 'end'}.nc")

ds370 = xr.open_dataset(SSP370_FILE)
ds126 = xr.open_dataset(SSP126_FILE)

if args.start is not None or args.end is not None:
    # Slice BOTH sources identically, before any assertions, so the time-axis
    # equality check below still validates what actually gets written.
    _sl = slice(args.start, args.end)
    ds370 = ds370.sel(time=_sl)
    ds126 = ds126.sel(time=_sl)
    print(f"[window] {int(ds370.time.values[0])}-{int(ds370.time.values[-1])} "
          f"({ds370.sizes['time']} steps)")

assert (ds370.time.values == ds126.time.values).all(), "time axis mismatch"
assert abs(ds370.lat.values - ds126.lat.values).max() < 1e-6, "lat grid mismatch"
assert (ds370.lon.values == ds126.lon.values).all(), "lon grid mismatch"

def _clean(da: xr.DataArray) -> xr.DataArray:
    # Drop any stray non-dimensional coords (e.g. ssp126's member_id) and
    # snap onto ssp370's lat/lon exactly (source grids match to float noise,
    # see the assert above -- this just removes that noise from the output).
    da = da.reset_coords(drop=True) if da.coords.keys() - {"time", "lat", "lon"} else da
    return da.assign_coords(lat=ds370.lat, lon=ds370.lon)

out = xr.Dataset(
    {
        "CO2": ds370["CO2"],
        "SUL": _clean(ds126["SUL"]),
        "BC":  _clean(ds126["BC"]),
    },
    coords={"time": ds370.time, "lat": ds370.lat, "lon": ds370.lon},
)
out.attrs["regrid_method"] = "bilinear"
out.attrs["description"] = (
    "Mixed-scenario cond file: CO2 from ssp370, SUL+BC from ssp126 "
    "(mirrors the CMIP6 ssp370-126aer experiment design)."
)
out.attrs["co2_source"] = "emissions_ssp370_only_timefixed_bc.nc"
out.attrs["aerosol_source"] = "emissions_ssp126_only_timefixed_co2fix_bc.nc"
if args.start is not None or args.end is not None:
    out.attrs["time_window"] = (f"{int(out.time.values[0])}-"
                                f"{int(out.time.values[-1])}")

out.to_netcdf(OUT_FILE)
print(f"wrote {OUT_FILE}")
print(out)
