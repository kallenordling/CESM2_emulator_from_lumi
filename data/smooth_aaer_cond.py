f"""Smooth the CEDS→IAMC SO2 inventory junction in the aaer conditioning file.

The aaer cond trajectory stitches CEDS-2017 historical SO2 (1850-2014) to
IAMC-AIM ssp370 SO2 (2015-2100). At 2014→2015 the spatial pattern
correlation drops from ~0.9996 to 0.9732 — a regional reshuffle (NA/EU
emissions ↓, E/S Asia ↑) that CESM2 itself smooths through ocean lag but
that our annual-mean conditioning exposes as a step. The model learns to
reproduce that step as a temperature spike at 2015 (emerged ~epoch 610).

Fix: replace years 2013-2016 with linear interpolation between 2012 and
2017 at every gridpoint. CO2 and other vars are left untouched.

Usage:
    python smooth_aaer_cond.py \\
        --in  {L.DATA}/emissions_aero_only_timefixed.nc \\
        --out {L.DATA}/emissions_aero_only_smoothed_timefixed.nc
"""
import lumi_paths as L
import argparse
import numpy as np
import xarray as xr


def smooth_junction(ds: xr.Dataset, var: str = "SUL",
                    anchor_before: int = 2012, anchor_after: int = 2017) -> xr.Dataset:
    tdim = next(d for d in ds[var].dims if d not in ("lat", "lon"))
    years = np.array([int(str(t)[:4]) for t in ds[tdim].values])
    if anchor_before not in years or anchor_after not in years:
        raise ValueError(
            f"Anchor years {anchor_before}/{anchor_after} not both present "
            f"(range {years.min()}-{years.max()})"
        )
    i0 = int(np.where(years == anchor_before)[0][0])
    i1 = int(np.where(years == anchor_after)[0][0])

    out = ds.copy(deep=True)
    arr = out[var].values.copy()
    a, b = arr[i0], arr[i1]
    for i in range(i0 + 1, i1):
        alpha = (i - i0) / (i1 - i0)
        arr[i] = (1 - alpha) * a + alpha * b
    out[var].values = arr

    pre_drop = (ds[var].isel({tdim: i0 + 3}).values.mean() -
                ds[var].isel({tdim: i0 + 2}).values.mean())
    post_drop = (arr[i0 + 3].mean() - arr[i0 + 2].mean())
    print(f"[smooth] {var} y{anchor_before}→y{anchor_after}: replaced years "
          f"{years[i0+1]}-{years[i1-1]} with linear interpolation")
    print(f"  2014→2015 gmean step before: {pre_drop:.3e}")
    print(f"  2014→2015 gmean step after:  {post_drop:.3e}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True, help="Input cond NetCDF")
    p.add_argument("--out", dest="dst", required=True, help="Output smoothed NetCDF")
    p.add_argument("--var", default="SUL", help="Variable to smooth (default: SUL)")
    p.add_argument("--before", type=int, default=2012, help="Anchor year BEFORE gap")
    p.add_argument("--after", type=int, default=2017, help="Anchor year AFTER gap")
    args = p.parse_args()

    ds = xr.open_dataset(args.src)
    print(f"[load] {args.src}  vars={list(ds.data_vars)}  "
          f"time={len(ds[next(d for d in ds[args.var].dims if d not in ('lat','lon'))])}")
    out = smooth_junction(ds, var=args.var,
                          anchor_before=args.before, anchor_after=args.after)
    out.to_netcdf(args.dst)
    print(f"[save] {args.dst}")


if __name__ == "__main__":
    main()
