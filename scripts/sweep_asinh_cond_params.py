#!/usr/bin/env python3
"""
Sweep the two asinh conditioning knobs and price what each setting buys.

The asinh transform has exactly two free numbers (data/climate_dataset.py):

    _ASINH_TOP_PCTL     percentile of the POSITIVE cells that maps to +1
    _ASINH_SCALE_FRAC   s, as a fraction of the positive-cell median

Raising the top percentile un-saturates the heaviest emitters. It is not free:
everything below the ceiling is compressed toward -1 at the same time. This
script measures both sides on the same grid, so the choice is a trade-off with
numbers rather than a preference.

METRICS (per species, on the training fit, evaluated on hist + ssp370)

    mass_pinned_<region>  share of the region's EMISSIONS sitting in cells the
                          transform flattens onto +1. This is the saturation
                          number that matters -- a low pinned CELL count with
                          most of the mass inside it is still a destroyed field.
    pinned_global         share of positive cells at +1, globally, 2100.
    contrast              spatial std of the normalised field over positive
                          cells, 2100. The within-map dynamic range the network
                          can actually use.
    span                  peak-to-peak of the global-mean normalised value over
                          1850-2100, as % of the full [-1, 1] range. The
                          temporal signal the conditioning carries.

Usage
-----
    python scripts/sweep_asinh_cond_params.py
    python scripts/sweep_asinh_cond_params.py --species BC --tops 99.5 99.9 100
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr

from plot_asinh_cond_maps import (  # same directory
    DEFAULT_DATA_DIR, REGIONS, SPECIES, CLIP_PCTL, FIT_FILES,
    cond_path, region_mask, apply_v1, apply_asinh,
)

SWEEP_REGIONS = ["E China", "India"]


def load_fit_pool(data_dir, species):
    """The positive-cell distribution the parameters are fitted on."""
    chunks = []
    for tag in FIT_FILES:
        with xr.open_dataset(cond_path(data_dir, tag)) as ds:
            a = np.asarray(ds[species].values, dtype=float).ravel()
        chunks.append(a[np.isfinite(a) & (a > 0)])
    return np.concatenate(chunks)


def load_record(data_dir, scenario, species):
    """hist + scenario stacked as (year, lat, lon), plus the axes."""
    parts, years = [], []
    for tag in ("hist", scenario):
        with xr.open_dataset(cond_path(data_dir, tag)) as ds:
            parts.append(np.asarray(ds[species].values, dtype=float))
            years.append(np.asarray(ds["year"].values))
            lat, lon = ds["lat"].values, ds["lon"].values
    return np.concatenate(parts), np.concatenate(years), lat, lon


def metrics(field, years, masks, s, top, transform="asinh", v1_hi=None):
    last = field[-1]
    if transform == "asinh":
        z_last = apply_asinh(last, s, top)
        gmean = np.array([apply_asinh(f, s, top).mean() for f in field])
    else:
        z_last = apply_v1(last, 0.0, v1_hi)
        gmean = np.array([apply_v1(f, 0.0, v1_hi).mean() for f in field])
    pos = last > 0
    pin = z_last >= 1.0 - 1e-9
    out = {
        "pinned_global": 100.0 * pin[pos].mean(),
        "contrast": float(z_last[pos].std()),
        "span": 100.0 * float(gmean.max() - gmean.min()) / 2.0,
    }
    for name, m in masks.items():
        r, p = last[m], pin[m]
        out[f"mass_{name}"] = 100.0 * float(r[p].sum() / r.sum()) if r.sum() > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--scenario", default="ssp370")
    ap.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    ap.add_argument("--tops", type=float, nargs="+",
                    default=[99.5, 99.9, 99.99, 100.0])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.02, 0.10, 0.50])
    ap.add_argument("--out", default="plots/asinh_cond/param_sweep_ssp370.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for v in args.species:
        pool = load_fit_pool(args.data_dir, v)
        med = float(np.percentile(pool, 50))
        v1_hi = None
        field, years, lat, lon = load_record(args.data_dir, args.scenario, v)
        masks = {n: region_mask(lat, lon, REGIONS[n]) for n in SWEEP_REGIONS}

        # the shipped v1 fit needs the all-cell distribution, not just positives
        allpool = []
        for tag in FIT_FILES:
            with xr.open_dataset(cond_path(args.data_dir, tag)) as ds:
                a = np.asarray(ds[v].values, dtype=float).ravel()
            allpool.append(a[np.isfinite(a)])
        v1_hi = float(np.percentile(np.concatenate(allpool), CLIP_PCTL[v][1]))
        del allpool

        base = metrics(field, years, masks, None, None, transform="v1", v1_hi=v1_hi)
        rows.append((v, "v1", float("nan"), float("nan"), base))
        for top_p in args.tops:
            top = float(pool.max()) if top_p >= 100 else float(np.percentile(pool, top_p))
            for sf in args.scales:
                rows.append((v, "asinh", top_p, sf,
                             metrics(field, years, masks, med * sf, top,
                                     transform="asinh")))
        del field, pool

    keys = ["pinned_global", "contrast", "span"] + [f"mass_{n}" for n in SWEEP_REGIONS]
    with open(args.out, "w") as fh:
        fh.write("species,transform,top_pctl,scale_frac," + ",".join(keys) + "\n")
        for v, t, tp, sf, mm in rows:
            fh.write(f"{v},{t},{tp},{sf}," + ",".join(f"{mm[k]:.3f}" for k in keys) + "\n")

    hdr = f"{'sp':4s}{'transform':>10s}{'top%':>8s}{'s/med':>7s}"
    hdr += "".join(f"{k:>16s}" for k in keys)
    print(hdr)
    for v, t, tp, sf, mm in rows:
        tps = "  -" if np.isnan(tp) else f"{tp:.2f}"
        sfs = "  -" if np.isnan(sf) else f"{sf:.2f}"
        print(f"{v:4s}{t:>10s}{tps:>8s}{sfs:>7s}"
              + "".join(f"{mm[k]:16.2f}" for k in keys))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
