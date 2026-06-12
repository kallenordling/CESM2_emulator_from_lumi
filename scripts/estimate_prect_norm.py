"""
estimate_prect_norm.py
======================
Estimate the PRECT log-space normalisation constants (PRECT_LOG_MEAN /
PRECT_LOG_STD) used by data/climate_dataset.py.

PRECT is monthly CAM h0 in m/s; we annual-mean it, convert m/s → mm/day
(×8.64e7), apply log1p, and report the mean + std over the pooled hist+ssp370
training years. Those two numbers must be baked into climate_dataset.py BEFORE
training so the precip channel sits at ~unit variance (balancing its MSE against
TREFHT's (x-4.5)/21). Run once on a CLEAN realization — the current re-download
was NaN-poisoned; the placeholder constants (mu≈0.7, sigma≈0.9) are GUESSES.

Usage:
    python scripts/estimate_prect_norm.py \
        --trees /scratch/.../training_data/PRECT/hist \
                /scratch/.../training_data/PRECT/ssp370 \
        --realization LE2-1001.001 --var PRECT
"""
import argparse
import glob
import os

import numpy as np
import xarray as xr

PREPROCESS = 8.64e7   # m/s → mm/day  (×1000 m→mm ×86400 s→day)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trees", nargs="+", required=True,
                   help="One or more PRECT scenario trees (e.g. hist ssp370).")
    p.add_argument("--realization", default=None,
                   help="Realization dir name to pool; default = every realization found.")
    p.add_argument("--var", default="PRECT")
    p.add_argument("--time-dim", default="time")
    args = p.parse_args()

    logs = []
    for tree in args.trees:
        reals = ([args.realization] if args.realization
                 else sorted(d for d in os.listdir(tree)
                             if os.path.isdir(os.path.join(tree, d))))
        for real in reals:
            files = sorted(glob.glob(os.path.join(tree, real, "*.nc")))
            if not files:
                print(f"  [skip] no files in {tree}/{real}")
                continue
            ds = xr.open_mfdataset(files, combine="by_coords")[args.var]
            # Monthly → annual mean (matches the training annual-mean target).
            if "time" in ds.dims and ds.sizes.get("time", 0) > len(set(
                    int(str(v)[:4]) for v in ds["time"].values)):
                ds = ds.groupby(f"{args.time_dim}.year").mean()
            mmday = (ds.values * PREPROCESS).astype(np.float64)
            mmday = mmday[np.isfinite(mmday)]
            mmday = mmday[mmday >= 0]           # guard tiny negatives
            logs.append(np.log1p(mmday).ravel())
            print(f"  {tree}/{real}: n={mmday.size}  "
                  f"mm/day [{mmday.min():.3f}, {mmday.max():.3f}]")

    if not logs:
        raise SystemExit("No PRECT data found — check --trees / --realization.")
    pooled = np.concatenate(logs)
    mu, sigma = float(pooled.mean()), float(pooled.std())
    print("\n================ RESULT ================")
    print(f"PRECT_LOG_MEAN = {mu:.4f}")
    print(f"PRECT_LOG_STD  = {sigma:.4f}")
    print("Bake these into data/climate_dataset.py (replace the placeholders).")


if __name__ == "__main__":
    main()
