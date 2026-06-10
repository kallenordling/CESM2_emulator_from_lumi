"""Compute decadal means from an eval-generated TREFHT_<experiment>.nc.

The eval (eval_aero.py) writes one NetCDF per experiment containing the
ensemble-mean model and CESM2 anomaly fields (re 1850-1900):

  <VAR>_model_mean_anom        (year, lat, lon)   model ensemble-mean anomaly
  <VAR>_model_gmean_mean_anom  (year,)            its area-weighted global mean
  <VAR>_cesm_mean_anom         (cesm_year, lat, lon)
  <VAR>_cesm_gmean_mean_anom   (cesm_year,)

This groups years into decades (label = START year; 2050 = mean over 2050-2059)
and writes, for ONE experiment:

  * printed + CSV table of decadal GLOBAL-MEAN anomaly: model / cesm / bias
    → <output_dir>/<exp>_decadal_gmean.csv
  * decadal-mean MAPS (model_anom / cesm_anom / bias) per decade
    → <output_dir>/<exp>_decadal_anom.nc   (dims: decade, lat, lon)

Usage:
  python decadal_means_from_nc.py <output_dir> <experiment> [--var TREFHT] [--out FILE]
  e.g.
  python decadal_means_from_nc.py /mnt/lumi_sc2/eval_output/manual/ep0852_v2 ssp126
"""
import os
import csv
import argparse
import numpy as np
import xarray as xr


def decadal_mean(da: xr.DataArray, ydim: str) -> xr.DataArray:
    """Mean within each decade (start year). Reduces `ydim` → new `decade` dim.

    Works for both a global-mean series (ydim,) and a field (ydim, lat, lon).
    """
    yrs = da[ydim].values.astype(int)
    dec_of = (yrs // 10) * 10
    decades = sorted(set(dec_of.tolist()))
    parts = []
    for d in decades:
        idx = np.where(dec_of == d)[0]
        parts.append(da.isel({ydim: idx}).mean(ydim))
    out = xr.concat(parts, dim="decade")
    return out.assign_coords(decade=("decade", np.array(decades, dtype=int)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="eval output dir containing TREFHT_<exp>.nc")
    ap.add_argument("experiment", help="experiment name, e.g. ssp126 / ssp245 / hist")
    ap.add_argument("--var", default="TREFHT", help="variable prefix (default TREFHT)")
    ap.add_argument("--out", default=None, help="output NetCDF (default <exp>_decadal_anom.nc)")
    args = ap.parse_args()
    V = args.var

    path = os.path.join(args.output_dir, f"{V}_{args.experiment}.nc")
    if not os.path.isfile(path):
        raise SystemExit(f"not found: {path}")
    ds = xr.open_dataset(path)

    # ── decadal GLOBAL-MEAN table (already area-weighted gmean vars) ──────────
    m_g = decadal_mean(ds[f"{V}_model_gmean_mean_anom"], "year")
    c_g = (decadal_mean(ds[f"{V}_cesm_gmean_mean_anom"], "cesm_year")
           if f"{V}_cesm_gmean_mean_anom" in ds else None)

    m_decs = {int(d): float(m_g.sel(decade=d)) for d in m_g.decade.values}
    c_decs = ({int(d): float(c_g.sel(decade=d)) for d in c_g.decade.values}
              if c_g is not None else {})

    print(f"\n{args.experiment} — decadal global-mean anomaly (°C):")
    print(f'{"decade":>7} {"model":>8} {"cesm":>8} {"bias":>8}')
    rows = []
    for d in sorted(set(m_decs) | set(c_decs)):
        m = m_decs.get(d, float("nan"))
        c = c_decs.get(d, float("nan"))
        b = m - c if not (np.isnan(m) or np.isnan(c)) else float("nan")
        cs = f"{c:8.3f}" if not np.isnan(c) else f"{'--':>8}"
        bs = f"{b:+8.3f}" if not np.isnan(b) else f"{'--':>8}"
        print(f"{d:>7} {m:8.3f} {cs} {bs}")
        rows.append({
            "decade":          d,
            "model_anom_degC": round(m, 4),
            "cesm_anom_degC":  round(c, 4) if not np.isnan(c) else "",
            "bias_degC":       round(b, 4) if not np.isnan(b) else "",
        })
    csv_out = os.path.join(args.output_dir, f"{args.experiment}_decadal_gmean.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["decade", "model_anom_degC",
                                          "cesm_anom_degC", "bias_degC"])
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {csv_out}")

    # ── decadal-mean MAPS (model / cesm / bias) ──────────────────────────────
    m_map = decadal_mean(ds[f"{V}_model_mean_anom"], "year")
    out_vars = {"model_anom": m_map}
    if f"{V}_cesm_mean_anom" in ds:
        c_map = decadal_mean(ds[f"{V}_cesm_mean_anom"], "cesm_year")
        common = np.intersect1d(m_map.decade.values, c_map.decade.values)
        out_vars["cesm_anom"] = c_map
        out_vars["bias"] = (m_map.sel(decade=common) - c_map.sel(decade=common))
    out = xr.Dataset(out_vars)
    nc_out = args.out or os.path.join(args.output_dir, f"{args.experiment}_decadal_anom.nc")
    out.to_netcdf(nc_out)
    print(f"[nc]  {nc_out}  (decadal-mean maps: {list(out_vars)}; dims decade,lat,lon)")


if __name__ == "__main__":
    main()
