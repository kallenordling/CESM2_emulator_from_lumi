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
ap.add_argument("--tree-root",
                default="/home/nordling/mnt/lumi_sc/emulator_data/training_data",
                help="root holding <var>/hist, /ssp370, /AAER, /GHG member trees, "
                     "used for the CESM2 side when the eval files carry none")
ap.add_argument("--data-config", default="configs/config_data_ybias_BCprect_nopca.yaml",
                help="defines which members were TRAINED on; the rest are held out")
ap.add_argument("--no-cesm-trees", action="store_true",
                help="skip the tree fallback and emit emulator-only output")
args = ap.parse_args()

_SUBDIR = {"hist": "hist", "ssp370": "ssp370", "aaer": "AAER", "ghg": "GHG"}


def cesm_from_trees(exp, lat, lon180):
    """Held-out CESM2 members at each city cell, read from the member trees.

    The evaluation NetCDFs used to carry `{var}_cesm_m*`; current ones carry no
    CESM2 at all, so the reference has to come from the same trees every other
    paper figure reads. Held-out = on disk but absent from the data config's
    experiment_configs, matching make_ensemble_mean_maps.heldout_members.
    """
    import glob
    import yaml
    d_ = os.path.join(args.tree_root, args.var, _SUBDIR[exp])
    if not os.path.isdir(d_):
        print(f"[city] {exp}: no tree at {d_}")
        return None, None
    try:
        cfg = yaml.safe_load(open(args.data_config))
        trained = {e["scenario_name"]: set(e.get("realizations", []))
                   for e in cfg["experiment_configs"]}.get(exp, set())
    except Exception as e:                                   # noqa: BLE001
        print(f"[city] {exp}: cannot read {args.data_config} ({e}); using ALL members")
        trained = set()
    on_disk = sorted(n for n in os.listdir(d_)
                     if n != "diagnostics" and os.path.isdir(os.path.join(d_, n)))
    members_ = [m for m in on_disk if m not in trained]
    if not members_:
        print(f"[city] {exp}: no held-out members on disk")
        return None, None
    series, years = [], None
    for mi, m in enumerate(members_, 1):
        files = sorted(glob.glob(os.path.join(d_, m, "*.nc")))
        if not files:
            continue
        with xr.open_mfdataset(files, combine="by_coords", decode_times=False) as ds:
            da = ds[args.var]
            tdim = "time" if "time" in da.dims else "year"
            yy = np.asarray(ds[tdim].values).astype(int)
            la_, lo_ = ds["lat"].values, ds["lon"].values
            lo180_ = ((lo_ + 180) % 360) - 180
            row = np.full((len(CITIES), len(yy)), np.nan)
            for ci, (_, cla, clo) in enumerate(CITIES):
                i = int(np.argmin(np.abs(la_ - cla)))
                k = int(np.argmin(np.abs(lo180_ - clo)))
                row[ci] = np.asarray(da.isel({"lat": i, "lon": k}).values, float)
            series.append(row)
            years = yy
        print(f"  [city-cesm] {exp} {mi}/{len(members_)} {m}", flush=True)
    if not series:
        return None, None
    return np.stack(series), years

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

    def _cells(arr):
        """(..., year) at each city cell, from a (..., year, lat, lon) array."""
        # (..., year, lat, lon) -> (..., city, year): drop the two spatial axes,
        # insert the city axis, keep year last.
        out = np.full(arr.shape[:-3] + (len(CITIES), arr.shape[-3]), np.nan)
        for ci, (_, la, lo) in enumerate(CITIES):
            i = int(np.argmin(np.abs(lat - la)))
            k = int(np.argmin(np.abs(lon180 - lo)))
            out[..., ci, :] = arr[..., :, i, k]
        return out

    def members(prefix, ydim):
        """Per-member (member, city, year), from either eval schema.

        Older evals exposed one variable per member (`*_m1`, `*_m2`, ...).
        Current ones write a single `{var}_model` with a member DIMENSION and
        no CESM2 arrays at all (eval_aero.py:1325), so the cesm side returns
        None here and the caller falls back to the member trees.
        """
        pat = re.compile(rf"^{args.var}_{prefix}_m(\d+)$")
        names = sorted((v for v in d.data_vars if pat.match(v)),
                       key=lambda v: int(pat.match(v).group(1)))
        if names:
            out = np.full((len(names), len(CITIES), d.sizes[ydim]), np.nan)
            for mi, v in enumerate(names):
                arr = d[v].values                   # (year, lat, lon)
                for ci, (_, la, lo) in enumerate(CITIES):
                    i = int(np.argmin(np.abs(lat - la)))
                    k = int(np.argmin(np.abs(lon180 - lo)))
                    out[mi, ci] = arr[:, i, k]
            return out, names
        var = f"{args.var}_{'model' if prefix == 'model' else 'cesm'}"
        if prefix == "model" and var in d.data_vars:
            da = d[var]
            arr = np.asarray(da.values, float)       # (member, year, lat, lon)
            if "member" not in da.dims:
                arr = arr[None]
            return _cells(arr), [f"member{i+1}" for i in range(arr.shape[0])]
        return None, None

    mod, mnames = members("model", "year")
    ces, cnames = members("cesm", "cesm_year" if "cesm_year" in d.sizes else "year")
    if ces is None and not args.no_cesm_trees:
        # Current evals carry NO CESM2 arrays, so read the held-out members
        # from the training trees -- the same source and the same held-out
        # split every other paper figure uses.
        ces, cyears = cesm_from_trees(exp, lat, lon180)
        if ces is not None:
            payload[f"cesm_{exp}"] = ces
            payload[f"cesm_years_{exp}"] = cyears
            cnames = [f"heldout{i+1}" for i in range(ces.shape[0])]
            ces = None          # already stored; keep the block below from redoing it
            print(f"[city] {exp}: cesm from trees, {len(cnames)} held-out members")
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
