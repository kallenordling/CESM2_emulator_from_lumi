#!/usr/bin/env python3
"""Pull TREFHT at four city gridpoints, for every TRAINING experiment.

Point series rather than the usual global means: a global mean averages away
both the aerosol fingerprint and the ensemble spread, and the question these
figures answer is whether the emulator gets the LOCAL climate right in places
with very different forcing histories.

Training experiments only -- hist, ssp370, aaer, ghg. ssp126/ssp245 are held
out and belong in a different comparison.

For each city and experiment it saves the emulator's per-member series, the
CESM2 per-member series, and the year axis. Nearest gridpoint on the 192x288
grid, so a "city" is really a ~1x1.25 degree cell containing it.

    python3 analysis/nonlinear_emission_interaction/dump_city_series.py \\
        --eval-dir /scratch/.../best_ep0860
"""

import argparse
import os
import re
import sys

import numpy as np
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))

CITIES = [
    ("Helsinki",  60.17,   24.94),
    ("Tokyo",     35.68,  139.65),
    ("Sydney",   -33.87,  151.21),
    ("Sao Paulo", -23.55, -46.63),
]
EXPERIMENTS = ["hist", "ssp370", "aaer", "ghg"]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--eval-dir", required=True)
ap.add_argument("--var", default="TREFHT")
ap.add_argument("--out", default=os.path.join(_HERE, "results", "city_series.npz"))
args = ap.parse_args()

payload = {"cities": np.array([c[0] for c in CITIES], dtype=object),
           "city_lat": np.array([c[1] for c in CITIES]),
           "city_lon": np.array([c[2] for c in CITIES]),
           "experiments": np.array(EXPERIMENTS, dtype=object)}

for exp in EXPERIMENTS:
    f = os.path.join(args.eval_dir, f"{args.var}_{exp}.nc")
    if not os.path.exists(f):
        print(f"[city] {exp}: no {os.path.basename(f)} — skipped")
        continue
    d = xr.open_dataset(f)
    lat = d["lat"].values
    lon = d["lon"].values
    lon180 = ((lon + 180) % 360) - 180

    def members(prefix, ydim):
        """Per-member (member, year) at each city, from *_m1, *_m2, ... vars."""
        pat = re.compile(rf"^{args.var}_{prefix}_m(\d+)$")
        names = sorted((v for v in d.data_vars if pat.match(v)),
                       key=lambda v: int(pat.match(v).group(1)))
        if not names:
            return None, None
        out = np.full((len(names), len(CITIES), d.sizes[ydim]), np.nan)
        for mi, v in enumerate(names):
            arr = d[v].values                       # (year, lat, lon)
            for ci, (_, la, lo) in enumerate(CITIES):
                i = int(np.argmin(np.abs(lat - la)))
                k = int(np.argmin(np.abs(lon180 - lo)))
                out[mi, ci] = arr[:, i, k]
        return out, names

    mod, mnames = members("model", "year")
    ces, cnames = members("cesm", "cesm_year" if "cesm_year" in d.sizes else "year")
    if mod is not None:
        payload[f"model_{exp}"] = mod
        payload[f"years_{exp}"] = d["year"].values
    if ces is not None:
        payload[f"cesm_{exp}"] = ces
        payload[f"cesm_years_{exp}"] = d["cesm_year" if "cesm_year" in d.coords
                                         else "year"].values
    print(f"[city] {exp}: model {0 if mod is None else mod.shape[0]} members, "
          f"cesm {0 if ces is None else ces.shape[0]} members, "
          f"{d.sizes['year']} years")
    # what cell each city actually landed in — worth recording, a 1.25 deg cell
    # near a coast can be mostly ocean and that changes the variance a lot
    if exp == EXPERIMENTS[0]:
        cell = []
        for _, la, lo in CITIES:
            i = int(np.argmin(np.abs(lat - la)))
            k = int(np.argmin(np.abs(lon180 - lo)))
            cell.append((float(lat[i]), float(lon180[k])))
        payload["cell_latlon"] = np.array(cell)
        print("[city] nearest cells: " + ", ".join(
            f"{c[0]} {v[0]:.2f},{v[1]:.2f}" for c, v in zip(CITIES, cell)))
    d.close()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, **payload, allow_pickle=True)
print(f"[city] wrote {args.out}")
