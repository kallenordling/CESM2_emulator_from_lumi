#!/usr/bin/env python3
"""
Saturation measured at the END of the conditioning pipeline, not after normalise.

The earlier numbers (plot_asinh_cond_maps.py, sweep_asinh_cond_params.py) stop
at `normalize()`. The model does not see that field. Two more stages follow, and
both destroy regional amplitude on their own:

    normalise -> gaussian smoothing (cond_smooth_sigma, default [0, 2, 2])
              -> PCA truncation      (n_components_cond, default [30, 5, 5])

PCA is the reason a "pinned cell" count is not the end of the story: with 5 EOFs
the SUL and BC channels are a five-number-per-year summary of the globe, so a
region's history can be flattened even where the clip never bit. This script
therefore reports, per region and per stage, the quantity that actually matters:

    r      correlation of the region-mean cond value with the region's TRUE
           emissions over 1850-2100 -- does the channel still track the region?
    amp    peak-to-peak of that region-mean series, in [-1, 1] units -- how much
           of the network's input range the region's history moves.
    srho   Spearman rank correlation, ACROSS THE CELLS INSIDE THE REGION at
           2100, between the cond field and the true emissions. This is the
           saturation metric: pinning ties the biggest cells together and ties
           cost rank correlation, while a region-mean series can look healthy
           throughout because the unpinned majority of cells carries it.
    mass   share of the region's emission mass in cells pinned at +1
           (normalise stage only; after PCA there is no ceiling to sit on).

Configurations compared: v1 everywhere (the precip-bc branch and every shipped
checkpoint), asinh everywhere (the running arm), and the mixed spec that keeps
CO2 on v1 -- the one this repo now supports as
`cond_transform="CO2=v1,SUL=asinh,BC=asinh"`.

Usage
-----
    python scripts/measure_cond_pipeline_saturation.py
    python scripts/measure_cond_pipeline_saturation.py --species BC --no-pca
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from plot_asinh_cond_maps import (
    DEFAULT_DATA_DIR, REGIONS, SPECIES, CLIP_PCTL, FIT_FILES,
    cond_path, region_mask, apply_v1, apply_asinh, fit_params,
)
from sweep_asinh_cond_params import load_record

# config_data_ybias_BCprect.yaml — [CO2, SUL, BC]
SMOOTH_SIGMA = {"CO2": 0, "SUL": 2, "BC": 2}
N_COMPONENTS = {"CO2": 30, "SUL": 5, "BC": 5}

# Each config is {modes per species} plus an optional per-species sigma override
# of SMOOTH_SIGMA. The two v1_noclip rows are the arms that removed the clip:
# v1noclip keeps the shipped sigma (CO2 unsmoothed, which is what speckles), and
# co2smooth smooths CO2 too. The point of running them here is that smoothing is
# what buys the speckle fix, and smoothing is also what destroys regional
# amplitude -- these rows say whether co2smooth can have both.
CONFIGS = {
    "v1 (precip-bc)":   dict(modes={"CO2": "v1",    "SUL": "v1",    "BC": "v1"}),
    "asinh (arm)":      dict(modes={"CO2": "asinh", "SUL": "asinh", "BC": "asinh"}),
    "CO2=v1 + asinh":   dict(modes={"CO2": "v1",    "SUL": "asinh", "BC": "asinh"}),
    "v1noclip (CO2 s=0)": dict(modes={k: "v1_noclip" for k in SPECIES}),
    "co2smooth (CO2 s=2)": dict(modes={k: "v1_noclip" for k in SPECIES},
                                sigma={"CO2": 2}),
}


def normalise(field, species, mode, params):
    if mode == "asinh":
        s, top = params[species]["asinh"]
        return apply_asinh(field, s, top)
    lo, hi = params[species]["v1"]
    if mode == "v1_noclip":
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        return np.zeros_like(field) if half == 0 else (field - mid) / half
    return apply_v1(field, lo, hi)


def smooth(field, sigma):
    if sigma <= 0:
        return field
    out = gaussian_filter1d(field, sigma=sigma, axis=-1, mode="wrap")     # lon periodic
    return gaussian_filter1d(out, sigma=sigma, axis=-2, mode="reflect")   # lat


def pca_truncate(field, n_components):
    T, H, W = field.shape
    flat = field.reshape(T, H * W).astype(np.float64)
    if flat.std() < 1e-8:
        return field
    p = PCA(n_components=min(n_components, T, H * W), whiten=False)
    recon = p.inverse_transform(p.fit_transform(flat))
    return recon.reshape(T, H, W), float(p.explained_variance_ratio_.sum())


def region_series(field, mask):
    return field[:, mask].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--scenario", default="ssp370")
    ap.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    ap.add_argument("--regions", nargs="+", default=["E China", "India"])
    ap.add_argument("--no-pca", action="store_true", help="stop after smoothing")
    ap.add_argument("--out", default="plots/asinh_cond/pipeline_saturation_ssp370.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    params = fit_params(args.data_dir, args.species)
    rows = []
    for v in args.species:
        field, years, lat, lon = load_record(args.data_dir, args.scenario, v)
        masks = {n: region_mask(lat, lon, REGIONS[n]) for n in args.regions}
        truth = {n: region_series(field, m) for n, m in masks.items()}
        for cfg_name, cfg in CONFIGS.items():
            mode = cfg["modes"][v]
            sigma = cfg.get("sigma", {}).get(v, SMOOTH_SIGMA[v])
            z = normalise(field, v, mode, params)
            pin = z >= 1.0 - 1e-9
            stages = [("normalise", z)]
            zs = smooth(z, sigma)
            stages.append((f"+smooth s={sigma}", zs))
            if not args.no_pca:
                zp, var_kept = pca_truncate(zs, N_COMPONENTS[v])
                stages.append((f"+PCA {N_COMPONENTS[v]} ({var_kept*100:.1f}% var)", zp))
            for stage, arr in stages:
                for n, m in masks.items():
                    ser = region_series(arr, m)
                    r = float(np.corrcoef(ser, truth[n])[0, 1])
                    amp = float(ser.max() - ser.min())
                    srho = float(spearmanr(arr[-1][m], field[-1][m]).statistic)
                    if stage == "normalise":
                        last, p = field[-1][m], pin[-1][m]
                        mass = 100.0 * float(last[p].sum() / last.sum()) if last.sum() > 0 else np.nan
                    else:
                        mass = np.nan
                    rows.append((v, cfg_name, mode, stage, n, r, amp, srho, mass))
        del field

    with open(args.out, "w") as fh:
        fh.write("species,config,mode,stage,region,r,amplitude,spatial_srho,mass_pinned_pct\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")

    print(f"{'sp':4s}{'config':>18s}{'stage':>24s}{'region':>9s}{'r':>8s}{'amp':>8s}{'srho':>8s}{'mass%':>8s}")
    for v, cfg_name, mode, stage, n, r, amp, srho, mass in rows:
        ms = "   -" if np.isnan(mass) else f"{mass:8.1f}"
        print(f"{v:4s}{cfg_name:>18s}{stage:>24s}{n:>9s}{r:8.3f}{amp:8.3f}{srho:8.3f}{ms:>8s}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
