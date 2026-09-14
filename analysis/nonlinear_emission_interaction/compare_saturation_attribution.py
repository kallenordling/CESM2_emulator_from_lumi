#!/usr/bin/env python3
"""
Did asinh remove the saturation artefact in the nonlinear-term attribution?

Reads two integrated-map files from 02_interaction_maps.py -- one checkpoint
trained on the v1 clip, one on asinh for SUL/BC -- and answers in three steps.

1. THE INPUT DISTORTION. delta(x), the 1850->2040 aerosol conditioning move as
   the model received it. Under v1 a minor emitter pinned at +1 in 2040 moves
   as far as East China, so its delta is inflated relative to its emissions.
   The test is the ratio delta(Middle East) / delta(East Asia) against the same
   ratio in RAW emissions: a faithful transform tracks the raw ratio.

2. THE ATTRIBUTION. G(x) * delta(x) summed over a source box is that box's
   share of N in a response region (21 K per model unit). A saturation artefact
   shows up as a Middle East share out of proportion to its emissions.

3. A FIGURE. delta and the Global attribution density for both checkpoints,
   with the Arabian Peninsula boxed, on the paper's projection.

READ AS A DECOMPOSITION, NOT A COUNTERFACTUAL. And the two checkpoints differ
in more than the transform -- training length (863 vs ~300 epochs) and the
paper checkpoint's pre-co2fix CO2 basis -- so a difference in N itself is not
attributable to the transform. The delta ratio in step 1 is, because it is a
property of the conditioning pipeline and not of the network.

    ~/miniconda3/envs/plotting/bin/python \\
        analysis/nonlinear_emission_interaction/compare_saturation_attribution.py \\
        --v1 intmaps_2040_v1paper_ep863.npz --asinh intmaps_2040_asinh99_ep307.npz
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from make_ensemble_mean_maps import make_axes, draw_map, panel_label, to_pm180, HAVE_CARTOPY  # noqa: E402

K_PER_UNIT = 21.0      # DENORM_FN TREFHT = x*21.0 + 4.5
COND_DIR = os.path.expanduser("~/mnt/lumi_sc/emulator_data")
SOURCE = [  # name, lat0, lat1, lon0, lon1 (0-360)
    ("East Asia",     20.0, 50.0, 100.0, 145.0),
    ("South Asia",     5.0, 30.0,  65.0, 100.0),
    ("Middle East",   12.0, 40.0,  35.0,  65.0),
    ("Europe",        35.0, 65.0,   0.0,  40.0),
    ("N. America",    30.0, 60.0, 230.0, 300.0),
    ("Africa",       -35.0, 35.0,   0.0,  50.0),
    ("S. America",   -55.0, 12.0, 280.0, 325.0),
]
ARABIA = (12.0, 32.0, 35.0, 60.0)


def box(lat, lon, lat0, lat1, lon0, lon1):
    return ((lat >= lat0) & (lat <= lat1))[:, None] & ((lon >= lon0) & (lon <= lon1))[None, :]


def raw_move(species):
    """Raw emission change 1850 -> 2040, same files the conditioning comes from."""
    with xr.open_dataset(f"{COND_DIR}/emissions_hist_only_timefixed_bc_co2fix.nc") as h, \
         xr.open_dataset(f"{COND_DIR}/emissions_ssp370_only_timefixed_bc_co2fix.nc") as s:
        return (np.asarray(s[species].sel(year=2040).values, float)
                - np.asarray(h[species].sel(year=1850).values, float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--asinh", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "figure_40_saturation_attribution"))
    args = ap.parse_args()

    runs = {"v1 (paper ep863)": np.load(args.v1, allow_pickle=True),
            "asinh_aero (asinh99)": np.load(args.asinh, allow_pickle=True)}
    first = next(iter(runs.values()))
    lat, lon = first["lat"], first["lon"]
    species = [str(s) for s in first["aerosol_names"]]
    regions = [str(r) for r in first["regions"]]
    for name, z in runs.items():
        print(f"[read] {name}: transform={z['cond_transform'] if 'cond_transform' in z.files else '?'} "
              f"kind='{z['source_map_kind']}'")

    # ---- 1. the input distortion ------------------------------------------
    masks = {n: box(lat, lon, *b) for n, *b in SOURCE}
    print("\n1. CONDITIONING MOVE, Middle East relative to East Asia "
          "(1.0 would mean Arabia moves as far as East China)")
    print(f"   {'':22s}" + "".join(f"{sp:>12s}" for sp in species))
    raw = {sp: raw_move(sp) for sp in species}
    row = "".join(f"{raw[sp][masks['Middle East']].mean() / raw[sp][masks['East Asia']].mean():12.3f}"
                  for sp in species)
    print(f"   {'RAW emissions':22s}{row}")
    ratios = {}
    for name, z in runs.items():
        d = z["delta"]
        r = [d[i][masks["Middle East"]].mean() / d[i][masks["East Asia"]].mean()
             for i in range(len(species))]
        ratios[name] = r
        print(f"   {name:22s}" + "".join(f"{v:12.3f}" for v in r))

    # ---- 2. the attribution ------------------------------------------------
    print("\n2. SHARE OF N BY SOURCE REGION (K), SUL+BC, per response region")
    table = {}
    for name, z in runs.items():
        d = z["delta"]
        for reg in regions:
            g = z[f"source_mean_{reg}"]
            dens = (g * d).sum(axis=0) * K_PER_UNIT          # both species, K per cell
            total = float(dens.sum())
            parts = {n: float(dens[m].sum()) for n, m in masks.items()}
            table[(name, reg)] = (total, parts)
    for reg in regions:
        print(f"\n   {reg}")
        print(f"   {'source':14s}" + "".join(f"{n:>24s}" for n in runs))
        for n, *_ in SOURCE:
            print(f"   {n:14s}" + "".join(f"{table[(r, reg)][1][n]:+24.3f}" for r in runs))
        print(f"   {'TOTAL':14s}" + "".join(f"{table[(r, reg)][0]:+24.3f}" for r in runs))
        me = "".join(f"{100 * abs(table[(r, reg)][1]['Middle East']) / (sum(abs(v) for v in table[(r, reg)][1].values()) + 1e-12):23.1f}%"
                     for r in runs)
        print(f"   {'M.East |share|':14s}{me}")

    # ---- 3. the figure -----------------------------------------------------
    fig = plt.figure(figsize=(14.5, 6.6), constrained_layout=True)
    axes = make_axes(fig, 2, 4)
    k = 0
    for i, (name, z) in enumerate(runs.items()):
        d = z["delta"]
        dens = z["source_mean_Global"] * d * K_PER_UNIT
        panels = [(d[0], f"delta {species[0]}", "magma", (0, np.nanpercentile(np.abs(d[0]), 99))),
                  (d[1], f"delta {species[1]}", "magma", (0, np.nanpercentile(np.abs(d[1]), 99))),
                  (dens[0], f"Global N density {species[0]} (K)", "RdBu_r", None),
                  (dens[1], f"Global N density {species[1]} (K)", "RdBu_r", None)]
        for j, (field, title, cmap, lim) in enumerate(panels):
            ax = axes[i][j]
            if lim is None:
                v = np.nanpercentile(np.abs(field), 99.5)
                lim = (-v, v)
            im = draw_map(ax, to_pm180(field, lon)[0], cmap=cmap, vmin=lim[0], vmax=lim[1],
                          outline="0.85" if cmap == "magma" else "0.15")
            if HAVE_CARTOPY:
                import cartopy.crs as ccrs
                lat0, lat1, lon0, lon1 = ARABIA
                ax.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0],
                        color="lime", lw=1.2, transform=ccrs.PlateCarree())
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.text(-0.04, 0.5, name, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=10)
            panel_label(ax, k); k += 1
            fig.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.02)
    fig.suptitle("Saturation test: aerosol conditioning move 1850->2040 and the Global nonlinear-term "
                 "density, v1 vs asinh (Arabian Peninsula boxed)", fontsize=11)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(args.out + ext, dpi=160, bbox_inches="tight")
    print(f"\nwrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
