"""Trim a conditioning NetCDF to a year range and write a new file.

Use case: ssp370 / ssp126 cond files stored at /scratch/.../emulator_data
were built by stitching hist (1850-2014) + SSP (2015-2100) and stored as
cumsum-from-1850 over the full range.  For an SSP scenario the historical
years are redundant — the simulation members only run 2015-2100 — so the
cond file should be trimmed to start where hist ends.

This script slices on the time-like dim by year and writes a new NetCDF.
The cumulative CO2 values are *not* re-zeroed — the cumsum integration
window stays anchored at 1850 so absolute magnitudes match the training
normalization.

Usage:
    python data/trim_cond_years.py \
        --in  /mnt/lumi_sc2/emulator_data/emissions_ssp370_timefixed.nc \
        --out /mnt/lumi_sc2/emulator_data/emissions_ssp370_timefixed.nc.trimmed \
        --y0 2015 [--y1 2100]
"""
import argparse
import numpy as np
import xarray as xr


def _years(da: xr.DataArray, dim: str) -> np.ndarray:
    return np.array([int(str(t)[:4]) for t in da[dim].values])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--y0",  type=int, required=True, help="First year to keep")
    p.add_argument("--y1",  type=int, default=None,  help="Last year to keep (default: file end)")
    args = p.parse_args()

    ds = xr.open_dataset(args.src)
    time_dim = next(d for d in ds.dims if d not in ("lat", "lon"))
    years = _years(ds[time_dim], time_dim)

    y1 = args.y1 if args.y1 is not None else int(years.max())
    mask = (years >= args.y0) & (years <= y1)
    if not mask.any():
        raise ValueError(f"No years in [{args.y0}, {y1}] within file range "
                         f"{int(years.min())}-{int(years.max())}")

    out = ds.isel({time_dim: np.where(mask)[0]})
    out.attrs["history"] = (
        ds.attrs.get("history", "") +
        f" | trimmed to years [{args.y0}, {y1}] from {args.src}"
    ).strip(" |")
    out.to_netcdf(args.dst)

    kept = years[mask]
    print(f"[load]  {args.src}  range={int(years.min())}-{int(years.max())} ({len(years)})")
    print(f"[trim]  kept {len(kept)} years: {int(kept.min())}-{int(kept.max())}")
    print(f"[save]  {args.dst}")


if __name__ == "__main__":
    main()
