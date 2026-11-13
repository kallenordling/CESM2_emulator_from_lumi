#!/usr/bin/env python3
"""
Add open biomass burning to the SUL and BC cond channels  ->  *_bb.nc

WHY
---
The shipped cond channels are fossil CO2 + CEDS *anthropogenic* SO2/BC. CEDS
excludes open burning, so the emulator cannot see biomass burning at all:

    CESM2-SF BMB   (biomass burning aerosols only)  -> cond = constant 1850
    CESM2-SF xAER  (all but anthropogenic aerosols) -> cond = GHG's cond

i.e. the same inputs paired with different climates. Adding open burning to
SUL/BC makes those channels TOTAL emissions, gives BMB and xAER their own
cond, and lets both join the training set. EE (land use, ozone, solar,
volcanic) is not emissions-driven and stays eval-only.

WHAT IS BUILT (all on the shipped 192x288 grid, same units as SUL/BC)
--------------------------------------------------------------------
    hist, ssp370, ssp126, ssp245   SUL/BC += their own open burning
    aaer, ghg, pi1850_const (EE)   SUL/BC += open burning pinned at 1850
                                   (CESM2-SF AAER explicitly EXCLUDES burning)
    bmb   (NEW)                    CO2 + anthro SUL/BC pinned at 1850
                                   + SMOOTHED open burning (hist|ssp370)
    xaer                           existing xaer cond + SMOOTHED open burning
    ssp370co2_ssp126aer (RAMIP)    += ssp370 open burning  (see NOTE)

Every input channel that is NOT open burning is carried over byte-for-byte
from the existing *_bc_co2fix.nc file; only the addition is new.

THE PIPELINE (identical to how SUL/BC were made — verify with --verify-anthro)
-----------------------------------------------------------------------------
    raw kg m-2 s-1, monthly -> sum sectors -> annual MEAN
      -> * cell area * s/yr / 1e12  = Gt / yr / native cell
      -> bilinear periodic regrid (xesmf) to the cond grid

The bilinear regrid of a per-cell TOTAL is what deflates SUL/BC ~4.7x
(see cond_regrid_extensive_deflation). The deflation depends on the native
cell size, so the 0.25 deg BB4CMIP6 historical is first SUMMED to the 0.5 deg
CEDS cells (exact); regridding it at 0.25 deg would deflate it ~4x more than
the anthro channel it is added to. The 2014->2015 hist->ssp370 junction in
the log is the check that the two grids agree.

SMOOTHED BURNING (bmb, xaer)
----------------------------
CESM2-SF uses the smoothed biomass burning of LE2 members 51-100
(11-yr running mean over 1990-2020). The training hist/ssp370 members
(LE2-1001..1231) are the first 50, with the UNSMOOTHED CMIP6 burning, so
hist/ssp370 get the raw series and bmb/xaer the smoothed one.

NOTE (unverified assumptions, flagged in the file attrs):
  * xAER: anthropogenic aerosols pinned at 1850 (inherited from
    make_sf_extra_cond.py) — the SF docs do not state the pin year.
  * RAMIP ssp370-126aer: burning follows ssp370.

Usage:
    python data/add_openburning_cond.py \
        --cond-dir ~/data_staging/bc_rebuild \
        --raw-dir ~/data_staging/inputs4mips_bb ~/data_staging/inputs4mips \
        --target ~/data_staging/bc_rebuild/target.nc \
        --out-dir ~/data_staging/bb_rebuild [--verify-anthro]
"""
import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr

R_EARTH = 6.371e6
SECONDS_PER_YEAR = 365.25 * 24 * 3600
KG_PER_GT = 1e12
Y0, HIST_END, Y1 = 1850, 2014, 2100
SMOOTH_WIN, SMOOTH_LO, SMOOTH_HI = 11, 1990, 2020

RAW_BB = {
    ("SO2", "hist"):   "SO2-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_185001-201512.nc",
    ("BC",  "hist"):   "BC-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_185001-201512.nc",
    ("SO2", "ssp370"): "SO2-em-openburning_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc",
    ("BC",  "ssp370"): "BC-em-openburning_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc",
    ("SO2", "ssp126"): "SO2-em-openburning_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc",
    ("BC",  "ssp126"): "BC-em-openburning_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc",
    ("SO2", "ssp245"): "SO2-em-openburning_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc",
    ("BC",  "ssp245"): "BC-em-openburning_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc",
}
# Only for --verify-anthro: the files the shipped SUL/BC hist channels came from.
RAW_ANTHRO = {
    ("SO2", "hist"):   "SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc",
    ("BC",  "hist"):   "BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc",
    ("SO2", "ssp370"): "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc",
    ("BC",  "ssp370"): "BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc",
}
CHAN = {"SO2": "SUL", "BC": "BC"}

SRC = "emissions_{}_only_timefixed_bc_co2fix.nc"
SRC_SPECIAL = {"pi1850_const": "emissions_pi1850_const_bc_co2fix.nc",
               "ramip": "emissions_ssp370co2_ssp126aer_bc_2015-2079_co2fix.nc"}


# ── raw -> native annual Gt/yr per cell ──────────────────────────────────────
def _bounds(ds, dim):
    name = ds[dim].attrs.get("bounds")
    for cand in ([name] if name else []) + [f"{dim}_bnds", f"{dim}_bounds"]:
        if cand and cand in ds:
            b = np.asarray(ds[cand].values, dtype=np.float64)
            if b.ndim == 2 and b.shape == (ds[dim].size, 2):
                return b
    return None


def _edges(c):
    c = np.asarray(c, dtype=np.float64)
    d = np.abs(np.diff(c).mean())
    return np.stack([c - d / 2, c + d / 2], axis=1)


def cell_area(lat, lon, lat_b, lon_b):
    lat_b = _edges(lat) if lat_b is None else lat_b
    lon_b = _edges(lon) if lon_b is None else lon_b
    lat_b = np.clip(lat_b, -90, 90)
    slat = np.abs(np.sin(np.deg2rad(lat_b[:, 1])) - np.sin(np.deg2rad(lat_b[:, 0])))
    dlon = np.deg2rad(np.abs(lon_b[:, 1] - lon_b[:, 0]))
    return slat[:, None] * dlon[None, :] * R_EARTH ** 2


def _resolve(dirs, pattern):
    for d in dirs:
        f = sorted(glob.glob(os.path.join(os.path.expanduser(d), pattern)))
        if f:
            return f
    raise FileNotFoundError(f"no files match {pattern} in {dirs}")


def native_annual(dirs, pattern, label):
    """Annual Gt/yr per 0.5 deg cell, year-indexed."""
    files = _resolve(dirs, pattern)
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 12})
    # BB4CMIP6 spells its axes latitude/longitude; CEDS/ScenarioMIP use lat/lon.
    ds = ds.rename({k: v for k, v in (("latitude", "lat"), ("longitude", "lon"))
                    if k in ds.variables})
    for d in ("lat", "lon"):
        b = ds[d].attrs.get("bounds")
        if b and b not in ds and f"{d}_bnds" in ds:
            ds[d].attrs["bounds"] = f"{d}_bnds"
    bnds ={d: _bounds(ds, d) for d in ("lat", "lon")}
    ds = ds.drop_vars([v for v in ds.variables
                       if "bnds" in str(v) or "bound" in str(v)], errors="ignore")
    var = [v for v in ds.data_vars if v not in ("sector", "time_bnds")][0]
    da = ds[var]
    for s in ("sector", "sectors"):
        if s in da.dims:
            da = da.sum(s)
    # Annual MEAN of a rate; years with <12 months would bias it, so refuse them.
    n = da["time"].groupby("time.year").count()
    da = da.groupby("time.year").mean().compute()
    area = xr.DataArray(cell_area(da.lat.values, da.lon.values, bnds["lat"], bnds["lon"]),
                        dims=["lat", "lon"], coords={"lat": da.lat, "lon": da.lon})
    out = da * area * SECONDS_PER_YEAR / KG_PER_GT
    dlat = float(np.abs(np.diff(out.lat.values).mean()))
    if dlat < 0.4:                       # 0.25 deg BB4CMIP6 -> sum to 0.5 deg cells
        k = int(round(0.5 / dlat))
        out = out.coarsen(lat=k, lon=k, boundary="exact").sum()
        out = out.assign_coords(lat=da.lat.coarsen(lat=k).mean().values,
                                lon=da.lon.coarsen(lon=k).mean().values)
    # Sort AFTER area + coarsen (the bounds arrays are in file order); BB4CMIP6
    # stores latitude north->south.
    out = out.sortby("year").sortby("lat").sortby("lon")
    bad =[int(y) for y, c in zip(n.year.values, n.values) if c != 12]
    print(f"  [{label}] {len(files)} file(s) {int(out.year[0])}-{int(out.year[-1])} "
          f"grid {out.sizes['lat']}x{out.sizes['lon']}  "
          f"global {float(out.isel(year=0).sum())*1e3:.3f} -> "
          f"{float(out.isel(year=-1).sum())*1e3:.3f} Tg/yr"
          + (f"  (partial years ignored downstream: {bad})" if bad else ""))
    return out, set(bad)


def to_annual(da, lo, hi, partial=()):
    """Clip to [lo, hi], drop partial years, interpolate decadal samples."""
    keep = [(int(y) >= lo) and (int(y) <= hi) and int(y) not in partial
            for y in da.year.values]
    da = da.isel(year=np.where(keep)[0])
    full = np.arange(int(da.year[0]), int(da.year[-1]) + 1)
    if len(full) != da.sizes["year"]:
        print(f"    interpolating {da.sizes['year']} -> {len(full)} years")
        da = da.interp(year=full)
    return da


def spliced(dirs, species, scen, raw=RAW_BB):
    h, hb = native_annual(dirs, raw[(species, "hist")], f"hist {species}")
    s, sb = native_annual(dirs, raw[(species, scen)], f"{scen} {species}")
    h = to_annual(h, Y0, HIST_END, hb)
    s = to_annual(s, HIST_END + 1, Y1, sb)
    a, b = float(h.isel(year=-1).sum()), float(s.isel(year=0).sum())
    print(f"    junction {HIST_END}->{HIST_END+1}: {a*1e3:.3f} -> {b*1e3:.3f} Tg/yr "
          f"({100*(b/a-1):+.1f}%)")
    return xr.concat([h, s], dim="year")


def smooth_bb(da):
    """LE2 'smbb': 11-yr centred running mean, applied over 1990-2020 only."""
    rm = da.rolling(year=SMOOTH_WIN, center=True, min_periods=1).mean()
    yrs = da.year.values
    m = xr.DataArray((yrs >= SMOOTH_LO) & (yrs <= SMOOTH_HI), dims="year",
                     coords={"year": yrs})
    return xr.where(m, rm, da).transpose(*da.dims)


# ── cond files ───────────────────────────────────────────────────────────────
def open_cond(cond_dir, name):
    fn = SRC_SPECIAL.get(name, SRC.format(name))
    ds = xr.open_dataset(os.path.join(cond_dir, fn)).load()
    if "time" in ds.dims and "year" not in ds.dims:
        ds = ds.rename({"time": "year"})
    ds = ds.assign_coords(year=np.asarray(ds.year.values, dtype=int))
    return fn, ds


def sel_years(da, yrs):
    pos = {int(y): i for i, y in enumerate(da.year.values)}
    miss = [int(y) for y in yrs if int(y) not in pos]
    if miss:
        raise KeyError(f"years missing from burning series: {miss[:5]}")
    return da.isel(year=[pos[int(y)] for y in yrs]).assign_coords(year=np.asarray(yrs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond-dir", required=True)
    ap.add_argument("--raw-dir", nargs="+", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-suffix", default="_bb")
    ap.add_argument("--verify-anthro", action="store_true",
                    help="reproduce the shipped hist/ssp370 SUL/BC from raw CEDS first")
    args = ap.parse_args()

    import xesmf as xe
    tgt = xr.open_dataset(args.target)
    tgrid = xr.Dataset({"lat": tgt["lat"], "lon": tgt["lon"]})
    _rg = {}

    def regrid(da):
        d = da.to_dataset(name="V")
        if float(d.lon.min()) < 0:
            d = d.assign_coords(lon=(d.lon % 360)).sortby("lon")
        key = (d.sizes["lat"], d.sizes["lon"], float(d.lat[0]), float(d.lon[0]))
        if key not in _rg:
            _rg[key] = xe.Regridder(d, tgrid, method="bilinear", periodic=True)
        out = _rg[key](d["V"])
        return out.assign_coords(lat=tgt.lat.values, lon=tgt.lon.values)

    if args.verify_anthro:
        print("\n=== verify: shipped SUL/BC from raw CEDS through this pipeline ===")
        for scen in ("hist", "ssp370"):
            _, ship = open_cond(args.cond_dir, scen)
            for sp in ("SO2", "BC"):
                nat, part = native_annual(args.raw_dir, RAW_ANTHRO[(sp, scen)],
                                          f"anthro {scen} {sp}")
                lo, hi = (Y0, HIST_END) if scen == "hist" else (HIST_END + 1, Y1)
                nat = to_annual(nat, lo, hi, part)
                yrs = [y for y in (1850, 1950, 1980, 2014, 2015, 2050, 2100)
                       if lo <= y <= hi]
                mine = regrid(sel_years(nat, yrs))
                ref = sel_years(ship[CHAN[sp]], yrs)
                rel = float(np.abs(mine - ref).max() / np.abs(ref).max())
                tot = float(mine.sum() / ref.sum())
                print(f"    {scen} {CHAN[sp]} years {yrs}: max|diff|/max = {rel:.2e}, "
                      f"total ratio {tot:.6f}")
                if rel > 1e-3:
                    sys.exit(f"ABORT: pipeline does not reproduce shipped {scen} "
                             f"{CHAN[sp]} (rel {rel:.2e}); burning would not be "
                             f"on the same scale")

    print("\n=== open burning, native ===")
    bb = {}
    for scen in ("ssp370", "ssp126", "ssp245"):
        for sp in ("SO2", "BC"):
            print(f" {scen} {sp}")
            bb[(scen, sp)] = regrid(spliced(args.raw_dir, sp, scen))
    bb_smooth = {sp: smooth_bb(bb[("ssp370", sp)]) for sp in ("SO2", "BC")}
    bb1850 = {sp: bb[("ssp370", sp)].sel(year=Y0, drop=True) for sp in ("SO2", "BC")}

    os.makedirs(args.out_dir, exist_ok=True)
    common_attrs = {
        "openburning_added": "SUL += SO2 open burning, BC += BC open burning, "
                             "same raw->Gt/yr/cell->bilinear pipeline as the anthro "
                             "channels; BB4CMIP6 0.25deg summed to 0.5deg first",
        "openburning_sources": "VUA-CMIP-BB4CMIP6-1-2 (<=2014), "
                               "IAMC ScenarioMIP *-em-openburning (>=2015)",
        "built_by": "data/add_openburning_cond.py",
    }

    def write(name, ds, how, src):
        fn = (f"emissions_{name}_only_timefixed_bc_co2fix{args.out_suffix}.nc"
              if name not in SRC_SPECIAL else
              SRC_SPECIAL[name].replace(".nc", f"{args.out_suffix}.nc"))
        ds = ds.copy()
        ds.attrs.update(common_attrs)
        ds.attrs["openburning_mode"] = how
        ds.attrs["base_cond_file"] = src
        for v in ds.data_vars:
            ds[v] = ds[v].astype("float64")
        dst = os.path.join(args.out_dir, fn)
        ds.to_netcdf(dst)
        y = ds.year.values
        print(f"  [{name:13s}] {y[0]}-{y[-1]}  last-year SUL {float(ds.SUL.isel(year=-1).sum()):.5g} "
              f"BC {float(ds.BC.isel(year=-1).sum()):.5g}  ({how})  -> {fn}")

    print("\n=== writing ===")
    # time-varying burning, own scenario
    for name, scen in (("hist", "ssp370"), ("ssp370", "ssp370"),
                       ("ssp126", "ssp126"), ("ssp245", "ssp245"),
                       ("ramip", "ssp370")):
        src, ds = open_cond(args.cond_dir, name)
        for sp in ("SO2", "BC"):
            ds[CHAN[sp]] = ds[CHAN[sp]] + sel_years(bb[(scen, sp)], ds.year.values)
        write(name, ds, f"{scen} burning, unsmoothed" + (
            " (ASSUMPTION: RAMIP keeps ssp370 burning)" if name == "ramip" else ""), src)

    # burning pinned at 1850
    for name in ("aaer", "ghg", "pi1850_const"):
        src, ds = open_cond(args.cond_dir, name)
        for sp in ("SO2", "BC"):
            ds[CHAN[sp]] = ds[CHAN[sp]] + bb1850[sp]
        write(name, ds, "burning pinned at 1850", src)

    # BMB: everything at 1850 except smoothed burning. Built from the EE file
    # (all channels pinned at 1850), so CO2/anthro aerosols are its exact values.
    src, ds = open_cond(args.cond_dir, "pi1850_const")
    for sp in ("SO2", "BC"):
        ds[CHAN[sp]] = ds[CHAN[sp]] + sel_years(bb_smooth[sp], ds.year.values)
    write("bmb", ds, "smoothed burning (LE2 smbb, 11-yr mean 1990-2020); CO2 + "
                     "anthro aerosols pinned at 1850", src)

    src, ds = open_cond(args.cond_dir, "xaer")
    for sp in ("SO2", "BC"):
        ds[CHAN[sp]] = ds[CHAN[sp]] + sel_years(bb_smooth[sp], ds.year.values)
    write("xaer", ds, "smoothed burning (LE2 smbb); ASSUMPTION: anthro aerosols "
                      "pinned at 1850", src)
    print("\ndone")


if __name__ == "__main__":
    main()
