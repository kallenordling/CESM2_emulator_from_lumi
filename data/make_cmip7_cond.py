"""
Build CMIP7 conditioning files (CO2 + SUL + BC) for the h / vl ScenarioMIP
scenarios, in the SAME normalized space as the CMIP6 training cond files so the
existing trained checkpoints can be evaluated on them without retraining.

This is a self-contained rebuild of the make_co2_files.py + make_aerosol_files.py
+ concat_and_regrid.py chain for CMIP7 inputs. It is deliberately separate rather
than an edit of those scripts, because:
  * the CMIP6 cond files must stay byte-reproducible for existing runs, and
  * concat_and_regrid.py's CO2/SO2 body is currently dead code (lines 76-153 sit
    inside a ''' ''' string literal), so there is nothing there to extend.

Conventions replicated EXACTLY from the CMIP6 path (do not "improve" these --
the model was trained on them and eval consumes them):
  CO2  = AIR-anthro (summed over level) + surface anthro (summed over sector),
         annual MEAN of the kg/m2/s rate * seconds/year * cell area / 1e12,
         then CUMULATIVE SUM over year from 1850   (make_co2_files.py:163,178 +
         concat_and_regrid.py:94 -- note the cumsum lives in the CONCAT step,
         make_co2_files.py:189 has it commented out)
  SUL  = SO2 surface anthro only, ANNUAL, no AIR, no cumsum (make_aerosol_files.py)
  BC   = BC  surface anthro only, ANNUAL, no AIR, no cumsum
  grid = xesmf bilinear periodic regrid to the target file's grid (192x288 f09),
         NaN -> 0 at unmapped points
  dims = 'year' internally, renamed to 'time' on write (eval_aero.py time_dim)

CMIP7 vs CMIP6 junction -- THE ONE REAL DIFFERENCE:
  CMIP6 spliced hist <=2014 / ssp >=2015 with no overlap. CMIP7 CEDS historical
  runs to 2023 and the IIASA scenarios start in 2022, a 2-year OVERLAP. Default
  here is hist <=2023 / scenario >=2024 (--hist-end), i.e. prefer the inventory
  wherever it exists and drop the scenarios' 2022-2023 back-cast. Change with
  --hist-end 2021 to let each scenario run from its own start year instead.

Inputs (flat dir, as downloaded by download_input4mips_cmip7.py --layout flat):
  {CO2,SO2,BC}-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_*.nc
  CO2-em-AIR-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_*.nc
  {CO2,SO2,BC}-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-<sc>-1-1-0_gn_202201-210012.nc
  CO2-em-AIR-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-<sc>-1-1-0_gn_202201-210012.nc

Outputs (OUTPUT_DIR, suffix keeps them clear of the CMIP6 files):
  emissions_hist_cmip7_only_timefixed_bc.nc   time 1850-2023
  emissions_h_cmip7_only_timefixed_bc.nc      time 2024-2100
  emissions_vl_cmip7_only_timefixed_bc.nc     time 2024-2100

Usage:
  python data/make_cmip7_cond.py --target /path/to/cesm2_grid.nc
  python data/make_cmip7_cond.py --target ... --scenarios h vl --hist-end 2023
  python data/make_cmip7_cond.py --target ... --dry-run     # check inputs only
"""

import argparse
import glob
import os

import numpy as np
import xarray as xr

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Build CMIP7 (h/vl) cond files matching the CMIP6 training conventions")
parser.add_argument("--target", required=True,
                    help="NetCDF file to take the output grid from (192x288 CESM2 f09)")
parser.add_argument("--input-dir", default=os.environ.get(
    "EMUL_INPUT_DIR", "/scratch/project_462001328/emulator_data/emission_data/inputs4mips/"),
                    help="Flat dir holding the downloaded input4MIPs files")
parser.add_argument("--output-dir", default=os.environ.get(
    "EMUL_OUTPUT_DIR", "/scratch/project_462001328/emulator_data/"))
parser.add_argument("--scenarios", nargs="+", default=["h", "vl"],
                    help="CMIP7 warming-level scenarios (only h and vl have "
                         "gridded emissions published as of 2026-08)")
parser.add_argument("--hist-source", default="CEDS-CMIP-2025-04-18")
parser.add_argument("--scen-version", default="1-1-0",
                    help="IIASA-IAMC-<scen>-<version> source_id suffix")
parser.add_argument("--hist-end", type=int, default=2023,
                    help="Last year taken from CEDS historical; scenarios supply "
                         "hist_end+1 onward. CEDS ends 2023, IIASA starts 2022.")
parser.add_argument("--start-year", type=int, default=1850,
                    help="First year of the cumulative CO2 integration (must match "
                         "training: 1850)")
parser.add_argument("--end-year", type=int, default=2100)
parser.add_argument("--suffix", default="_cmip7_only_timefixed_bc",
                    help="Output filename suffix after emissions_<scenario>")
parser.add_argument("--dry-run", action="store_true",
                    help="Resolve and report input files, then exit without computing")
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
HIST_END = args.hist_end
Y0, Y1 = args.start_year, args.end_year

R_EARTH = 6.371e6
SECONDS_PER_YEAR = 365.25 * 24 * 3600
KG_PER_GT = 1e12

# Output variable names: SO2 keeps the legacy 'SUL' channel name used in training.
OUT_VAR = {"CO2": "CO2", "SO2": "SUL", "BC": "BC"}


# ── Input file patterns ──────────────────────────────────────────────────────
def hist_pattern(species: str, air: bool = False) -> str:
    kind = "em-AIR-anthro" if air else "em-anthro"
    return os.path.join(
        INPUT_DIR,
        f"{species}-{kind}_input4MIPs_emissions_CMIP_{args.hist_source}_gn_*.nc")


def scen_pattern(species: str, scenario: str, air: bool = False) -> str:
    kind = "em-AIR-anthro" if air else "em-anthro"
    src = f"IIASA-IAMC-{scenario}-{args.scen_version}"
    return os.path.join(
        INPUT_DIR,
        f"{species}-{kind}_input4MIPs_emissions_ScenarioMIP_{src}_gn_*.nc")


def _resolve(pattern: str) -> list:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            f"No input files matching:\n  {pattern}\n"
            f"Download them first:  sbatch run_download_input4mips_slurm.sh")
    return files


# ── Grid cell area (identical to make_co2_files.py / make_aerosol_files.py) ──
def compute_grid_cell_area(lat, lon):
    """Area of each cell in m^2 on a regular lat/lon grid -> (nlat, nlon)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dlat = np.abs(np.diff(lat).mean())
    dlon = np.abs(np.diff(lon).mean())
    lat_edges = np.deg2rad(np.clip(
        np.concatenate([[lat[0] - dlat / 2],
                        (lat[:-1] + lat[1:]) / 2,
                        [lat[-1] + dlat / 2]]), -90, 90))
    area = np.abs(np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])) \
        * np.deg2rad(dlon) * R_EARTH ** 2
    return np.broadcast_to(area[:, np.newaxis], (len(lat), len(lon)))


def find_dim(ds, candidates):
    for c in candidates:
        if c in ds.dims:
            return c
    raise ValueError(f"None of {candidates} in dims: {list(ds.dims)}")


def _main_var(ds):
    v = [n for n in ds.data_vars if "bnds" not in n and "bound" not in n]
    assert v, f"No data variables in {list(ds.data_vars)}"
    return v[0]


def load_annual_gt(pattern: str, *, air: bool, year_lo: int, year_hi: int) -> xr.DataArray:
    """input4MIPs monthly kg/m2/s -> annual Gt/gridpoint/year on the native grid.

    Sums the sector dim (surface anthro) or level dim (aircraft), takes the
    annual MEAN of the rate, then converts with cell area * seconds/year.
    Annual mean (not sum) of a rate is the physical choice -- this mirrors
    make_co2_files.py:130-134 exactly.
    """
    files = _resolve(pattern)
    print(f"    {len(files)} file(s): {os.path.basename(files[0])}"
          + (f" … +{len(files)-1}" if len(files) > 1 else ""))
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.drop_vars([v for v in ds if "bnds" in str(v) or "bound" in str(v)],
                      errors="ignore")

    collapse = find_dim(ds, ["level", "lev", "levels"]) if air \
        else find_dim(ds, ["sector", "sectors"])
    ds = ds.sum(dim=collapse)

    # Trim before the annual groupby -- CEDS starts in 1750 and we only ever use
    # >=1850, so this drops ~40% of the work.
    ds = ds.sel(time=ds["time.year"] >= year_lo)
    annual = ds.groupby("time.year").mean()
    annual = annual.sel(year=(annual.year >= year_lo) & (annual.year <= year_hi))

    var = _main_var(annual)
    area = xr.DataArray(
        compute_grid_cell_area(annual.lat.values, annual.lon.values),
        dims=["lat", "lon"],
        coords={"lat": annual.lat.values, "lon": annual.lon.values})
    da = annual[var] * area * SECONDS_PER_YEAR / KG_PER_GT
    return da.compute()


def species_series(species: str, scenario: str) -> xr.DataArray:
    """Full spliced 1850-2100 annual Gt/gridpoint series for one species.

    hist (CEDS) <= HIST_END, scenario (IIASA) >= HIST_END+1. CO2 additionally
    includes aircraft emissions; SO2/BC are surface anthro only.
    """
    print(f"  [{species}] historical ({args.hist_source}) …")
    hist = load_annual_gt(hist_pattern(species), air=False,
                          year_lo=Y0, year_hi=HIST_END)
    if species == "CO2":
        hist_air = load_annual_gt(hist_pattern(species, air=True), air=True,
                                  year_lo=Y0, year_hi=HIST_END)
        hist, hist_air = xr.align(hist, hist_air, join="inner")
        hist = hist + hist_air

    print(f"  [{species}] scenario (IIASA-IAMC-{scenario}-{args.scen_version}) …")
    scen = load_annual_gt(scen_pattern(species, scenario), air=False,
                          year_lo=HIST_END + 1, year_hi=Y1)
    if species == "CO2":
        scen_air = load_annual_gt(scen_pattern(species, scenario, air=True),
                                  air=True, year_lo=HIST_END + 1, year_hi=Y1)
        scen, scen_air = xr.align(scen, scen_air, join="inner")
        scen = scen + scen_air

    # IIASA scenario files are decadal in places -> interpolate to annual, the
    # same treatment the CMIP6 path applied (concat_and_regrid.py:103-106).
    if len(scen.year) > 1:
        full = np.arange(int(scen.year.values[0]), int(scen.year.values[-1]) + 1)
        if len(full) != len(scen.year):
            print(f"    interpolating scenario to annual "
                  f"({len(scen.year)} -> {len(full)} years)")
            scen = scen.interp(year=full, method="linear")

    series = xr.concat([hist, scen], dim="year").sortby("year")
    series = series.sel(year=(series.year >= Y0) & (series.year <= Y1))

    # Junction diagnostic on the NATIVE grid (mirrors the BC path in
    # concat_and_regrid.py:228-231). A large step here shows up downstream as a
    # cond discontinuity -- see aaer_2015_spike / the CEDS->IAMC junction note.
    if HIST_END in series.year and (HIST_END + 1) in series.year:
        a = float(series.sel(year=HIST_END).sum())
        b = float(series.sel(year=HIST_END + 1).sum())
        pct = 100 * (b - a) / a if a else float("nan")
        print(f"    junction {HIST_END}->{HIST_END+1}: "
              f"{a:.5f} -> {b:.5f} Gt/yr ({pct:+.2f}%)")
    return series


# ── Dry run: resolve inputs only (before touching the target grid, so this
#    works as a pre-flight check on a machine that has nothing else set up) ───
if args.dry_run:
    print("=== dry run: resolving inputs ===")
    ok = True
    for sp in ("CO2", "SO2", "BC"):
        checks = [("hist", hist_pattern(sp))] + \
                 [(f"scen {sc}", scen_pattern(sp, sc)) for sc in args.scenarios]
        if sp == "CO2":
            checks += [("hist AIR", hist_pattern(sp, air=True))] + \
                      [(f"scen {sc} AIR", scen_pattern(sp, sc, air=True))
                       for sc in args.scenarios]
        for label, pat in checks:
            n = len(glob.glob(pat))
            print(f"  {sp:4s} {label:12s} {n} file(s)"
                  + ("" if n else f"   MISSING: {os.path.basename(pat)}"))
            ok &= n > 0
    print("\nAll inputs present." if ok else
          "\nMissing inputs — run: sbatch run_download_input4mips_slurm.sh")
    raise SystemExit(0 if ok else 1)


# ── Target grid ──────────────────────────────────────────────────────────────
print(f"Loading target grid: {args.target}")
ds_target = xr.open_dataset(args.target)
tgt_lat = next((ds_target[n] for n in ("lat", "latitude") if n in ds_target), None)
tgt_lon = next((ds_target[n] for n in ("lon", "longitude") if n in ds_target), None)
assert tgt_lat is not None and tgt_lon is not None, \
    f"No lat/lon in target file. Coords: {list(ds_target.coords)}"
target_grid = xr.Dataset({"lat": tgt_lat, "lon": tgt_lon})
tgt_is_360 = float(tgt_lon.values.max()) > 180
print(f"  {len(tgt_lat)} lat x {len(tgt_lon)} lon  (lon 0-360: {tgt_is_360})")


def regrid_to_target(ds):
    """Lon-convention fix + xesmf bilinear periodic regrid (concat_and_regrid.py:195-207)."""
    import xesmf as xe
    src_is_360 = float(ds.lon.values.max()) > 180
    if src_is_360 != tgt_is_360:
        if tgt_is_360:
            print("  lon: -180..180 -> 0..360")
            ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
        else:
            print("  lon: 0..360 -> -180..180")
            ds = ds.assign_coords(lon=((ds.lon + 180) % 360 - 180)).sortby("lon")
    regridder = xe.Regridder(ds, target_grid, method="bilinear", periodic=True)
    return regridder(ds, keep_attrs=True)


# ── Build ────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
hist_written = False

for scenario in args.scenarios:
    print(f"\n{'='*70}\nScenario: {scenario}  (hist <={HIST_END}, scenario >={HIST_END+1})\n{'='*70}")

    series = {sp: species_series(sp, scenario) for sp in ("CO2", "SO2", "BC")}

    # CO2 -> cumulative from Y0. MUST happen on the full spliced record BEFORE
    # clipping, so the scenario file's magnitudes carry the whole 1850-onward
    # integration and match the training distribution (concat_and_regrid.py:94).
    print("\n  CO2: cumulative sum over year (from "
          f"{Y0}; annual -> cumulative)")
    co2_annual_total = float(series["CO2"].sum(dim=["lat", "lon"]).sel(year=2100)) \
        if 2100 in series["CO2"].year else float("nan")
    series["CO2"] = series["CO2"].cumsum(dim="year", keep_attrs=True)
    series["CO2"]["year"] = series["CO2"].year  # cumsum drops the coord otherwise

    ds = xr.Dataset({OUT_VAR[sp]: series[sp] for sp in ("CO2", "SO2", "BC")})

    print(f"\n  regridding {dict(ds.sizes)} -> target grid …")
    ds = regrid_to_target(ds)
    for v in ds.data_vars:
        ds[v] = ds[v].fillna(0.0)  # unmapped pole points -> 0 (matches SO2/SUL path)

    ds["CO2"].attrs.update(units="Gt CO2 / gridpoint (cumulative)",
                           long_name="Cumulative CO2 emissions per grid point")
    ds["SUL"].attrs.update(units="Gt SO2 / year / gridpoint",
                           long_name="Annual SO2 emissions per grid point")
    ds["BC"].attrs.update(units="Gt BC / year / gridpoint",
                          long_name="Annual BC emissions per grid point")
    ds.attrs.update(
        mip_era="CMIP7",
        hist_source=args.hist_source,
        scenario_source=f"IIASA-IAMC-{scenario}-{args.scen_version}",
        junction=f"hist <={HIST_END}, scenario >={HIST_END+1}",
        note="Built by data/make_cmip7_cond.py to match CMIP6 training conventions "
             "(CO2 cumulative from 1850 incl. aircraft; SUL/BC annual surface anthro).")

    # ── hist file (written once; CEDS historical is scenario-independent) ────
    if not hist_written:
        hist_out = os.path.join(OUTPUT_DIR, f"emissions_hist{args.suffix}.nc")
        hist_ds = ds.sel(year=slice(Y0, HIST_END)).rename({"year": "time"})
        hist_ds.to_netcdf(hist_out)
        print(f"\n  wrote {hist_out}  "
              f"({hist_ds.sizes['time']} yr, {Y0}-{HIST_END}, vars {list(hist_ds.data_vars)})")
        hist_written = True

    # ── scenario-only file (HIST_END+1 .. Y1) ───────────────────────────────
    out = os.path.join(OUTPUT_DIR, f"emissions_{scenario}{args.suffix}.nc")
    scen_ds = ds.sel(year=slice(HIST_END + 1, Y1)).rename({"year": "time"})
    scen_ds.to_netcdf(out)
    print(f"  wrote {out}  "
          f"({scen_ds.sizes['time']} yr, {HIST_END+1}-{Y1}, vars {list(scen_ds.data_vars)})")

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\n  --- {scenario} summary ---")
    print(f"  cumulative CO2 {HIST_END+1}: "
          f"{float(scen_ds['CO2'].isel(time=0).sum()):10.2f} Gt")
    print(f"  cumulative CO2 {Y1}: "
          f"{float(scen_ds['CO2'].isel(time=-1).sum()):10.2f} Gt")
    print(f"  annual CO2 {Y1} (pre-cumsum): {co2_annual_total:10.4f} Gt/yr")
    for v, unit in (("SUL", "Gt SO2/yr"), ("BC", "Gt BC/yr")):
        print(f"  {v} {HIST_END+1}: {float(scen_ds[v].isel(time=0).sum()):8.5f} {unit}"
              f"   {Y1}: {float(scen_ds[v].isel(time=-1).sum()):8.5f} {unit}")
    for v in scen_ds.data_vars:
        n_nan = int(np.isnan(scen_ds[v].values).sum())
        if n_nan:
            print(f"  WARNING: {v} has {n_nan} NaNs after regrid+fillna")

print("\nDone.")
print("Next: point eval_aero.py's EXPERIMENTS at these cond files. Note h/vl are "
      "OOD scenarios with no persisted PCA basis, so eval_aero.py will fit a fresh "
      "per-scenario basis (the 'fit' sentinel path, eval_aero.py:2307).")
