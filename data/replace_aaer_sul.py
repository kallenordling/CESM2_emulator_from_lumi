"""Replace SUL in the AAER cond file with hist+ssp370 SUL (1850-2050).

The CESM2-LE2 AAER single-forcing experiment evolves aerosols along the
ssp370 trajectory while holding GHGs fixed.  The ssp370 cond file already
contains hist+ssp370 SO2 stitched and regridded onto the model grid, so we
just lift its SUL field and write it into the existing AAER cond file for
the requested year range.  Everything else (CO2 channel, coords, attrs) is
left untouched.

Usage:
    python data/replace_aaer_sul.py \
        --aaer   /scratch/.../emissions_aero_only_timefixed.nc \
        --ssp370 /scratch/.../emissions_ssp370_timefixed.nc \
        --out    /scratch/.../emissions_aero_only_sulreplaced_timefixed.nc \
        [--y0 1850] [--y1 2050] [--var SUL]
"""
import argparse
import numpy as np
import xarray as xr


def _time_dim(da: xr.DataArray) -> str:
    return next(d for d in da.dims if d not in ("lat", "lon"))


def _years(da: xr.DataArray, dim: str) -> np.ndarray:
    return np.array([int(str(t)[:4]) for t in da[dim].values])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aaer",   required=True, help="Existing AAER cond NetCDF (target)")
    p.add_argument("--ssp370", required=True, help="ssp370 cond NetCDF (source of new SUL)")
    p.add_argument("--out",    required=True, help="Output NetCDF path")
    p.add_argument("--var",    default="SUL", help="Variable to replace (default: SUL)")
    p.add_argument("--y0",     type=int, default=1850, help="First year to replace")
    p.add_argument("--y1",     type=int, default=2050, help="Last year to replace (inclusive)")
    p.add_argument("--zero_co2", action="store_true",
                   help="Also overwrite the CO2 channel with zeros (AAER physics: GHGs held fixed)")
    args = p.parse_args()

    ds_a = xr.open_dataset(args.aaer)
    ds_s = xr.open_dataset(args.ssp370)

    if args.var not in ds_a:
        raise KeyError(f"{args.var} not in {args.aaer}; vars={list(ds_a.data_vars)}")
    if args.var not in ds_s:
        raise KeyError(f"{args.var} not in {args.ssp370}; vars={list(ds_s.data_vars)}")

    if not (ds_a.lat.equals(ds_s.lat) and ds_a.lon.equals(ds_s.lon)):
        raise ValueError("lat/lon grids differ between AAER and ssp370 cond files")

    tdim_a = _time_dim(ds_a[args.var])
    tdim_s = _time_dim(ds_s[args.var])
    years_a = _years(ds_a[args.var], tdim_a)
    years_s = _years(ds_s[args.var], tdim_s)

    src = ds_s[args.var].values   # (T_s, lat, lon)
    new = ds_a[args.var].values.copy()

    s_index = {int(y): i for i, y in enumerate(years_s)}
    replaced, missing = 0, []
    for i, yr in enumerate(years_a):
        if not (args.y0 <= yr <= args.y1):
            continue
        j = s_index.get(int(yr))
        if j is None:
            missing.append(int(yr))
            continue
        new[i] = src[j]
        replaced += 1

    out = ds_a.copy(deep=True)
    out[args.var].values = new
    out[args.var].attrs["history"] = (
        out[args.var].attrs.get("history", "") +
        f" | replaced years {args.y0}-{args.y1} from {args.ssp370}"
    ).strip(" |")

    if args.zero_co2:
        co2_name = "CO2" if "CO2" in out else ("co2" if "co2" in out else None)
        if co2_name is None:
            print("[warn] --zero_co2 set but no CO2/co2 var in AAER file; skipping")
        else:
            out[co2_name].values = np.zeros_like(out[co2_name].values)
            out[co2_name].attrs["history"] = (
                out[co2_name].attrs.get("history", "") +
                " | zeroed for AAER single-forcing"
            ).strip(" |")
            print(f"[zero]    {co2_name}: set to 0 everywhere")

    out.to_netcdf(args.out)

    print(f"[load]    aaer:   {args.aaer}  years={years_a.min()}-{years_a.max()}")
    print(f"[load]    ssp370: {args.ssp370}  years={years_s.min()}-{years_s.max()}")
    print(f"[replace] {args.var}: {replaced} years in [{args.y0}, {args.y1}]"
          + (f"  missing={missing}" if missing else ""))
    print(f"[save]    {args.out}")


if __name__ == "__main__":
    main()
