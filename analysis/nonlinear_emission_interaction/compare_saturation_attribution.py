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


def smooth_pca_move(species, n_comp=5, sigma=2.0, year=2040, base_year=1850):
    """The emission change after the pipeline's smoothing AND PCA truncation, in Gt.

    Same basis construction as cond_basis.joint_basis -- hist <= 2014 joined to
    ssp370 >= 2015, n_comp EOFs per aerosol species -- but fitted on the SMOOTHED
    PHYSICAL fields rather than on normalised ones. That shows what the rank-5
    truncation does to the emission change independently of any transform, so
    the loss can be told apart from the saturation.

    Returns (reconstructed change, change removed by the truncation).
    """
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA

    def sm(a):
        a = gaussian_filter1d(a, sigma, axis=-1, mode="wrap")
        return gaussian_filter1d(a, sigma, axis=-2, mode="reflect")

    with xr.open_dataset(f"{COND_DIR}/emissions_hist_only_timefixed_bc_co2fix.nc") as h, \
         xr.open_dataset(f"{COND_DIR}/emissions_ssp370_only_timefixed_bc_co2fix.nc") as s:
        hy, sy = h["year"].values, s["year"].values
        hf = sm(np.asarray(h[species].values, float))
        sf = sm(np.asarray(s[species].values, float))
    record = np.concatenate([hf[hy <= 2014], sf[sy >= 2015]])
    years = np.concatenate([hy[hy <= 2014], sy[sy >= 2015]])
    T, H, W = record.shape
    pca = PCA(n_components=n_comp).fit(record.reshape(T, H * W))
    recon = pca.inverse_transform(pca.transform(record.reshape(T, H * W))).reshape(T, H, W)
    i1, i0 = int(np.where(years == year)[0][0]), int(np.where(years == base_year)[0][0])
    full = record[i1] - record[i0]
    kept = recon[i1] - recon[i0]
    print(f"[pca] {species}: {n_comp} EOFs keep {100 * pca.explained_variance_ratio_.sum():.2f}% "
          f"of variance; the 1850->2040 change keeps "
          f"{100 * (1 - np.sum((full - kept) ** 2) / np.sum(full ** 2)):.1f}% of its energy")
    return kept, full - kept


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
    # Row 0 is the PHYSICAL change the conditioning is supposed to represent:
    # emissions straight from the cond files, 2040 minus 1850, in the files' own
    # units. Left pair unsmoothed, right pair after the same sigma=2 gaussian the
    # pipeline applies -- the smoothed field is what normalisation actually sees.
    # Log colour scale, because the change spans five decades and a linear scale
    # would show three bright pixels. Comparing this row with the delta panels
    # below is the saturation test by eye: a faithful map keeps Arabia dimmer
    # than East Asia, as it is here.
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter1d

    def smooth2(a):
        a = gaussian_filter1d(a, 2.0, axis=-1, mode="wrap")
        return gaussian_filter1d(a, 2.0, axis=-2, mode="reflect")

    raw_panels = [(raw[species[0]], f"emission change {species[0]}, raw"),
                  (raw[species[1]], f"emission change {species[1]}, raw"),
                  (smooth2(raw[species[0]]), f"emission change {species[0]}, smoothed (sigma 2)"),
                  (smooth2(raw[species[1]]), f"emission change {species[1]}, smoothed (sigma 2)")]

    fig = plt.figure(figsize=(14.5, 13.0), constrained_layout=True)
    axes = make_axes(fig, 4, 4)
    k = 0
    for j, (field, title) in enumerate(raw_panels):
        ax = axes[0][j]
        pos = field[field > 0]
        # anchored per panel: from the median emitting cell to the 99.9th percentile
        norm = LogNorm(vmin=float(np.percentile(pos, 50)), vmax=float(np.percentile(pos, 99.9)))
        shown = np.where(field > 0, field, np.nan)
        # A log scale cannot show a decline, so those cells are masked. Paint
        # them a labelled grey: left white they read as missing data.
        cmap_raw = plt.get_cmap("magma").copy()
        cmap_raw.set_bad("0.62")
        im = draw_map(ax, to_pm180(shown, lon)[0], cmap=cmap_raw, norm=norm, outline="0.85")
        ax.text(0.5, -0.02, "grey: emissions fell or unchanged", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="0.3")
        if HAVE_CARTOPY:
            import cartopy.crs as ccrs
            lat0, lat1, lon0, lon1 = ARABIA
            ax.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0],
                    color="lime", lw=1.2, transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=9)
        if j == 0:
            ax.text(-0.04, 0.5, "cond files\n2040 - 1850", transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontsize=10)
        panel_label(ax, k); k += 1
        unit = "Gt SO2/yr" if species[j % 2] == "SUL" else "Gt BC/yr"
        cb = fig.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.02)
        cb.set_label(f"{unit} per gridpoint (log)", fontsize=7)

    # Row 1: the same change after smoothing AND the rank-5 PCA, still physical.
    pca_stage = {sp: smooth_pca_move(sp) for sp in species}
    for j, sp in enumerate(species):
        kept, lost = pca_stage[sp]
        unit = "Gt SO2/yr" if sp == "SUL" else "Gt BC/yr"
        for col, (field, title, kind) in enumerate((
                (kept, f"{sp} change, smoothed + 5-EOF PCA", "log"),
                (lost, f"{sp} change REMOVED by the PCA", "div"))):
            ax = axes[1][2 * j + col]
            if kind == "log":
                pos = field[field > 0]
                norm = LogNorm(vmin=float(np.percentile(pos, 50)),
                               vmax=float(np.percentile(pos, 99.9)))
                cmap_pca = plt.get_cmap("magma").copy()
                cmap_pca.set_bad("0.62")
                im = draw_map(ax, to_pm180(np.where(field > 0, field, np.nan), lon)[0],
                              cmap=cmap_pca, norm=norm, outline="0.85")
                ax.text(0.5, -0.02, "grey: reconstruction <= 0", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7, color="0.3")
                label = f"{unit} per gridpoint (log)"
            else:
                v = float(np.nanpercentile(np.abs(field), 99.5))
                im = draw_map(ax, to_pm180(field, lon)[0], cmap="RdBu_r", vmin=-v, vmax=v)
                label = f"{unit} per gridpoint"
            if HAVE_CARTOPY:
                import cartopy.crs as ccrs
                lat0, lat1, lon0, lon1 = ARABIA
                ax.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0],
                        color="lime", lw=1.2, transform=ccrs.PlateCarree())
            ax.set_title(title, fontsize=9)
            if j == 0 and col == 0:
                ax.text(-0.04, 0.5, "cond files after\nsmoothing + PCA", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)
            panel_label(ax, k); k += 1
            cb = fig.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.02)
            cb.set_label(label, fontsize=7)

    for i0, (name, z) in enumerate(runs.items()):
        i = i0 + 2
        d = z["delta"]
        dens = z["source_mean_Global"] * d * K_PER_UNIT
        panels = [(d[0], f"{species[0]} move: normalised + smoothed + PCA", "magma",
                   (0, np.nanpercentile(np.abs(d[0]), 99))),
                  (d[1], f"{species[1]} move: normalised + smoothed + PCA", "magma",
                   (0, np.nanpercentile(np.abs(d[1]), 99))),
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
            ax.set_title(title, fontsize=9)
            if j == 0:
                ax.text(-0.04, 0.5, name, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=10)
            panel_label(ax, k); k += 1
            fig.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.02)
    fig.suptitle("Saturation test: the emission change in the cond files, the conditioning move the model "
                 "receives, and the Global nonlinear-term density (Arabian Peninsula boxed)", fontsize=11)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(args.out + ext, dpi=160, bbox_inches="tight")
    print(f"\nwrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
