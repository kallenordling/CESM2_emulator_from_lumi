# build_normalizer.py

import json
import numpy as np
import xarray as xr

EMISSIONS_PATH = "/scratch/project_462001112/emulator_data/emissions.nc"

def compute_zscore_stats(x: xr.DataArray, log1p: bool = True, eps: float = 1e-6):
    x = x.astype("float64")
    if log1p:
        x = xr.apply_ufunc(np.log1p, x)

    xv = x.values
    xv = xv[np.isfinite(xv)]
    if xv.size == 0:
        raise ValueError("All values non-finite")

    mean = float(xv.mean())
    std = float(xv.std())

    return {
        "method": "zscore",
        "log1p": bool(log1p),
        "mean": mean,
        "std": std,
        "eps": float(eps),
        "clip": None,          # IMPORTANT: no clipping
    }

def compute_sparse_percentile_stats_raw(
    x: xr.DataArray,
    p_high=95.0,
    eps=1e-12,
):
    xv = x.astype("float64").values
    xv = xv[np.isfinite(xv)]

    # positives only to avoid oceans
    xv_pos = xv[xv > 0.0]
    if xv_pos.size < 1000:
        xv_use = xv  # fallback
    else:
        xv_use = xv_pos

    # anchor low at 0 in RAW space
    lo = 0.0
    hi = float(np.percentile(xv_use, p_high))

    return {
        "method": "percentile",
        "log1p": True,      # applied in normalize(), exactly once
        "p_low": lo,
        "p_high": hi,
        "eps": float(eps),
        "clip": None,
        "p_high_q": float(p_high),
        "notes": "sparse raw percentiles; low anchored at 0; log1p applied at normalize()",
    }

def compute_zscore_stats_posonly(x: xr.DataArray, log1p: bool = True, eps: float = 1e-6):
    x = x.astype("float64")
    if log1p:
        x = xr.apply_ufunc(np.log1p, x)

    xv = x.values
    xv = xv[np.isfinite(xv)]

    # Use only positives to avoid oceans/zeros collapsing the distribution
    xv_pos = xv[xv > 0.0]
    if xv_pos.size > 1000:
        xv_use = xv_pos
    else:
        xv_use = xv  # fallback

    mean = float(xv_use.mean())
    std = float(xv_use.std())

    return {
        "method": "zscore",
        "log1p": bool(log1p),
        "mean": mean,
        "std": std,
        "eps": float(eps),
        "clip": None,
        "notes": "zscore computed on x>0 after log1p (fallback to all if too few)",
    }

def compute_percentile_stats_sparse_emissions(
    x: xr.DataArray,
    log1p=True,
    p_high=99.0,
    eps=1e-12,
):
    x = x.astype("float64")
    if log1p:
        x = xr.apply_ufunc(np.log1p, x)

    xv = x.values
    xv = xv[np.isfinite(xv)]

    # use positives to define the "upper scale"
    xv_pos = xv[xv > 0.0]
    if xv_pos.size == 0:
        raise ValueError("No positive values found for sparse-emissions scaling.")

    hi = float(np.percentile(xv_pos, p_high))

    # IMPORTANT: keep the low anchor at 0 so small positives aren't clipped to -1
    lo = 0.0

    return {
        "method": "percentile",
        "log1p": False,  # already applied above
        "p_low": lo,
        "p_high": hi,
        "eps": float(eps),
        "clip": None,
        "p_high_q": float(p_high),
        "sparse_low_anchor": "0.0_after_log1p",
    }


def main():
    ds = xr.open_dataset(EMISSIONS_PATH)

    stats = {}
    stats["CO2_em_anthro"] = compute_zscore_stats(ds["CO2_em_anthro"], log1p=True, eps=1e-6)
    stats["sul"]           = compute_zscore_stats_posonly(ds["sul"],           log1p=True, eps=1e-6)

    with open("norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Wrote norm_stats.json:", list(stats.keys()))

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
