#!/usr/bin/env python3
"""Decompose the model's error vs the CESM2 ensemble, per scenario, from a
saved eval directory (eval_output/<run>/best_ep<N>/TREFHT_<scen>.nc).

For each scenario (late-decade mean anomaly) prints:
  GMbias   — area-weighted global-mean (model − CESM2 ensemble mean), °C
  patcorr  — area-weighted spatial pattern correlation of the anomaly fields
  spRMSE   — area-weighted spatial RMSE of (model − CESM2), °C
  IVfloor  — internal-variability floor (RMS spread of CESM2 members), °C
  err/IV   — spRMSE / IVfloor (how far above the achievable noise floor)
  latitude-band bias (Arctic / NHmid / Trop / SHmid / Antarctic)

Reads only NetCDF, so it runs anywhere with xarray (no model/GPU). The eval
NetCDF carries TREFHT_model_mean_anom, TREFHT_cesm_mean_anom, cesm_m1..m10_anom.

Usage:
    python diag_model_skill.py [eval_dir]
    (default: newest best_ep* under eval_output/run_slope-tcre)
"""
import os
import re
import sys
import glob
import warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
SCENARIOS = ["hist", "ssp370", "ghg", "aaer", "ssp126"]


def main(eval_dir):
    print(f"[skill] {eval_dir}")
    print(f"{'scen':7s} {'GMbias':>7s} {'patcorr':>7s} {'spRMSE':>7s} {'IVfloor':>7s} "
          f"{'err/IV':>6s}  Arctic  NHmid  Trop   SHmid  Antarc  (Nmem)")
    for sc in SCENARIOS:
        f = os.path.join(eval_dir, f"TREFHT_{sc}.nc")
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f)
        if "TREFHT_cesm_mean_anom" not in ds:
            print(f"{sc:7s}  (no CESM2 ref)")
            continue
        lat = ds.lat.values
        m = ds["TREFHT_model_mean_anom"].values[-10:].mean(0)
        c = ds["TREFHT_cesm_mean_anom"].values[-10:].mean(0)
        w = np.cos(np.deg2rad(lat))[:, None]; w = w / w.mean()
        mems = [v for v in ds.data_vars if re.match(r"TREFHT_cesm_m\d+_anom$", v)]
        if mems:
            arr = np.stack([ds[v].values[-10:].mean(0) for v in mems])
            iv = np.sqrt(((arr - arr.mean(0)) ** 2).mean(0))
            ivf = float(np.sqrt(((iv ** 2) * w).mean()))
        else:
            ivf = float("nan")
        err = m - c
        gmb = float((err * w).mean())
        sprmse = float(np.sqrt(((err ** 2) * w).mean()))
        ma = m - (m * w).mean(); ca = c - (c * w).mean()
        pat = float(((ma * ca) * w).mean()
                    / np.sqrt(((ma ** 2) * w).mean() * ((ca ** 2) * w).mean()))

        def band(lo, hi):
            s = (lat >= lo) & (lat < hi)
            wl = np.cos(np.deg2rad(lat[s]))[:, None]; wl = wl / wl.mean()
            return float((err[s] * wl).mean())

        bands = [band(60, 91), band(30, 60), band(-30, 30), band(-60, -30), band(-91, -60)]
        ratio = sprmse / ivf if ivf == ivf and ivf > 0 else float("nan")
        print(f"{sc:7s} {gmb:+7.2f} {pat:7.2f} {sprmse:7.2f} {ivf:7.2f} {ratio:6.1f}  "
              + " ".join(f"{b:+5.2f}" for b in bands) + f"  ({len(mems)})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        d = sys.argv[1]
    else:
        cands = sorted(glob.glob("/mnt/lumi_sc2/eval_output/run_slope-tcre/best_ep*"),
                       key=os.path.getmtime)
        d = cands[-1] if cands else sys.exit("no eval dir given and none found")
    main(d)
