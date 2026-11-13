#!/usr/bin/env python3
"""
Add a VOLCANIC cond channel (VOLC) to the open-burning cond set  ->  *_bbv.nc

WHY
---
CESM2-SF EE ("everything else") varies volcanic aerosol (plus solar, land use,
ozone) while GHG/aerosols stay at 1850. No cond channel carries any of it, so
EE's cond is constant and EE cannot be trained on. Volcanoes also act in the
hist/ssp370 TRAINING runs (Krakatoa 1883, Pinatubo 1991) with nothing in the
cond to explain the cooling. VOLC makes the volcanic part visible. Solar,
land use and ozone remain unseen (user choice 2026-11-13: volcanic only).

WHAT VOLC IS
------------
Stratospheric sulfate column load, kg H2SO4 / m^2, from the CMIP6 forcing
dataset IACETH-SAGE3lambda-2-1-0 (H2SO4_mass, molecules/cm^3 on 36 lat x 70
altitude, 1850-2014): integrated over altitude, annual mean, interpolated to
the cond latitudes and broadcast over longitude (the dataset is zonal mean).

    years <= 2014   the dataset
    2015-2024       linear ramp from 2014 to BACKGROUND
    >= 2025         BACKGROUND
    BACKGROUND      the 1850-2014 mean (per latitude)

The ramp + background is the CMIP6 ScenarioMIP volcanic convention, and the
background is also the CMIP6 piControl volcanic forcing — so experiments that
hold "natural forcing at 1850" (GHG, AAER, BMB) get BACKGROUND, not the 1850
value of the series. ASSUMPTIONS (flagged in file attrs): CESM2 follows these
conventions; CESM2 used its own WACCM-derived volcanic aerosol, whose eruption
timing/latitudes match this dataset but whose amplitudes need not.

Per experiment:
    hist                       series 1850-2014
    ssp370/126/245, RAMIP      ramp + background
    ee   (NEW name)            series 1850-2014 + ramp to 2050; every other
                               channel pinned at 1850 (from pi1850_const_bb)
    xaer                       series from 1920 + ramp
    aaer, ghg, bmb             BACKGROUND constant

Usage:
    python data/add_volcanic_channel.py \
        --bb-dir ~/data_staging/bb_rebuild \
        --volc-file ~/data_staging/inputs4mips_volc/multiple_input4MIPs_aerosolProperties_CMIP_IACETH-SAGE3lambda-2-1-0_gn_185001-201412.nc \
        --out-dir ~/data_staging/bbv_rebuild
"""
import argparse
import os

import numpy as np
import xarray as xr

M_H2SO4 = 0.09808          # kg/mol
N_A = 6.02214076e23
RAMP_START, RAMP_END = 2015, 2024
HIST_END = 2014


def column_load(path):
    """(year, lat) annual-mean stratospheric H2SO4 column, kg/m^2."""
    ds = xr.open_dataset(path, decode_times=False)
    dz_m = np.diff(ds["bounds_altitude"].values, axis=1)[:, 0] * 1e3
    n = ds["H2SO4_mass"]                       # molecules / cm^3
    if ds["H2SO4_mass"].attrs.get("units") != "molecules/cm3air":
        raise ValueError(f"unexpected H2SO4_mass units {n.attrs.get('units')}")
    kg_m3 = n * 1e6 * M_H2SO4 / N_A
    col = (kg_m3 * xr.DataArray(dz_m, dims="altitude")).sum("altitude")
    nt = col.sizes["time"]
    assert nt % 12 == 0, nt
    yrs = 1850 + np.arange(nt // 12)
    ann = col.values.reshape(nt // 12, 12, -1).mean(1)
    assert yrs[-1] == HIST_END, yrs[-1]
    return xr.DataArray(ann, dims=("year", "lat"),
                        coords={"year": yrs, "lat": ds["latitude"].values})


def extend(hist, y_end):
    """hist series, then the ScenarioMIP ramp to the 1850-2014 background."""
    bg = hist.mean("year")
    last = hist.sel(year=HIST_END)
    fut_years = np.arange(HIST_END + 1, y_end + 1)
    w = np.clip((fut_years - (RAMP_START - 1)) / (RAMP_END - RAMP_START + 1), 0, 1)
    w = xr.DataArray(w, dims="year", coords={"year": fut_years})
    fut = last * (1 - w) + bg * w
    return xr.concat([hist, fut.transpose("year", "lat")], dim="year"), bg


def to_grid(da_ylat, lat, lon, years):
    """(year, lat_src) -> (year, lat, lon) on the cond grid, by year label."""
    src_lat = da_ylat["lat"].values
    pos = {int(y): i for i, y in enumerate(da_ylat["year"].values)}
    v = da_ylat.values[[pos[int(y)] for y in years]]                  # (T, nlat_src)
    out = np.stack([np.interp(lat, src_lat, row) for row in v])       # edge-held
    out = np.broadcast_to(out[:, :, None], (len(years), len(lat), len(lon)))
    return xr.DataArray(out.astype("float64"), dims=("year", "lat", "lon"),
                        coords={"year": np.asarray(years), "lat": lat, "lon": lon},
                        attrs={"units": "kg m-2", "long_name":
                               "stratospheric H2SO4 column load (volcanic)"})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bb-dir", required=True)
    ap.add_argument("--volc-file", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    hist = column_load(args.volc_file)
    series, bg = extend(hist, 2100)
    w = np.cos(np.deg2rad(hist.lat))
    gm = (series * w).sum("lat") / w.sum()
    print("global-mean VOLC (kg/m2): background %.3e, 1884 %.3e, 1992 %.3e, "
          "2014 %.3e, 2020 %.3e, 2030 %.3e" % tuple(
              float(x) for x in ((bg * w).sum() / w.sum(), gm.sel(year=1884),
                                 gm.sel(year=1992), gm.sel(year=2014),
                                 gm.sel(year=2020), gm.sel(year=2030))))
    bg_series = xr.concat([bg.expand_dims(year=[y]) for y in range(1850, 2101)],
                          dim="year")

    base = "emissions_{}_only_timefixed_bc_co2fix_bb.nc"
    special = {"pi1850_const": "emissions_pi1850_const_bc_co2fix_bb.nc",
               "ramip": "emissions_ssp370co2_ssp126aer_bc_2015-2079_co2fix_bb.nc"}
    plan = [  # (output name, source cond file key, VOLC source, mode)
        ("hist", "hist", series, "series"),
        ("ssp370", "ssp370", series, "ramp to background"),
        ("ssp126", "ssp126", series, "ramp to background"),
        ("ssp245", "ssp245", series, "ramp to background"),
        ("ramip", "ramip", series, "ramp to background"),
        ("xaer", "xaer", series, "series + ramp"),
        ("ee", "pi1850_const", series, "series + ramp; other channels 1850"),
        ("aaer", "aaer", bg_series, "background (piControl) constant"),
        ("ghg", "ghg", bg_series, "background (piControl) constant"),
        ("bmb", "bmb", bg_series, "background (piControl) constant"),
    ]
    for name, src_key, vsrc, mode in plan:
        src = special.get(src_key, base.format(src_key))
        ds = xr.open_dataset(os.path.join(args.bb_dir, src)).load()
        tdim = "year" if "year" in ds.dims else "time"
        years = np.asarray(ds[tdim].values, dtype=int)
        volc = to_grid(vsrc, ds.lat.values, ds.lon.values, years)
        if tdim != "year":
            volc = volc.rename({"year": tdim})
        volc = volc.assign_coords({tdim: ds[tdim].values})
        ds["VOLC"] = volc
        ds.attrs.update({
            "volc_added": "VOLC = stratospheric H2SO4 column (kg/m2), "
                          "IACETH-SAGE3lambda-2-1-0, zonal mean broadcast in lon",
            "volc_mode": mode,
            "volc_assumption": "post-2014 linear ramp 2015-2024 to the 1850-2014 "
                               "mean; fixed-natural experiments get that mean",
            "volc_built_by": "data/add_volcanic_channel.py",
        })
        if name in special:
            fn = special[name].replace("_bb.nc", "_bbv.nc")
        else:
            fn = f"emissions_{name}_only_timefixed_bc_co2fix_bbv.nc"
        ds.to_netcdf(os.path.join(args.out_dir, fn))
        g = float((ds.VOLC.mean("lon") * np.cos(np.deg2rad(ds.lat))).sum()
                  / np.cos(np.deg2rad(ds.lat)).sum() / 1)
        print(f"  [{name:7s}] {years[0]}-{years[-1]}  VOLC gm(first) "
              f"{float((ds.VOLC.isel({tdim: 0}).mean('lon') * np.cos(np.deg2rad(ds.lat))).sum() / np.cos(np.deg2rad(ds.lat)).sum()):.3e}"
              f"  max {float(ds.VOLC.max()):.3e}  ({mode}) -> {fn}")
    print("done")


if __name__ == "__main__":
    main()
