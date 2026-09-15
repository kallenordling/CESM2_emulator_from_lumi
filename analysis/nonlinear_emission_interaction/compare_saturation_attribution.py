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
from make_ensemble_mean_maps import make_axes, draw_map, panel_label, to_pm180, HAVE_CARTOPY, PANEL_LETTERS  # noqa: E402
sys.path.insert(0, HERE)

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


class CondOnly(dict):
    """A figure row with no model behind it: only delta and a transform name."""
    @property
    def files(self):
        return list(self.keys())


def conditioning_only_row(species, pctl, n_comp=5, year=2040, base_year=1850):
    """v1's clipped linear map with the clip at [100-pctl, pctl], pushed through
    normalise -> smooth -> joint PCA exactly as the model rows are. Returns the
    row (delta) and a callable giving each species' normalised record, so the
    time-series panel reuses the same numbers."""
    from asian_saturation_by_transform import smooth as sm, path as cpath, FIT_FILES as FF
    from sklearn.decomposition import PCA

    with xr.open_dataset(cpath("hist")) as h, xr.open_dataset(cpath("ssp370")) as s_:
        hy, sy = h["year"].values, s_["year"].values
        rec = {sp: np.concatenate([np.asarray(h[sp].values, float)[hy <= 2014],
                                   np.asarray(s_[sp].values, float)[sy >= 2015]]) for sp in species}
    years = np.concatenate([hy[hy <= 2014], sy[sy >= 2015]])
    deltas, processed = [], {}
    for sp in species:
        pool = []
        for t in FF:
            with xr.open_dataset(cpath(t)) as ds:
                a = np.asarray(ds[sp].values, float).ravel()
            pool.append(a[np.isfinite(a)])
        lo, hi = np.percentile(np.concatenate(pool), [100.0 - pctl, pctl])
        z = sm(np.clip((rec[sp] - (lo + hi) / 2) / ((hi - lo) / 2), -1, 1))
        T, H, W = z.shape
        pca = PCA(n_components=n_comp).fit(z.reshape(T, -1))
        zp = pca.inverse_transform(pca.transform(z.reshape(T, -1))).reshape(T, H, W)
        processed[sp] = zp
        i1, i0 = int(np.where(years == year)[0][0]), int(np.where(years == base_year)[0][0])
        deltas.append(zp[i1] - zp[i0])
        print(f"[v1 p{pctl:g}] {sp}: lo={lo:.3e} hi={hi:.3e}")
    row = CondOnly(delta=np.stack(deltas).astype(np.float32),
                   cond_transform=np.array(f"v1_p{pctl:g}"))
    return row, processed


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
    ap.add_argument("--minmax", default=None,
                    help="optional third arm: linear to the smoothed maximum, no clip")
    ap.add_argument("--v1-clip-pctl", type=float, default=99.0,
                    help="add a CONDITIONING-ONLY row: v1's linear map with its clip "
                         "moved to [100-p, p] percentiles for SUL and BC (precip-bc uses "
                         "5-95). No model was trained this way, so the row has no N. "
                         "Pass 0 to omit it.")
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "figure_40_saturation_attribution"))
    args = ap.parse_args()

    def epoch_of(z):
        ck = str(z["checkpoint"]) if "checkpoint" in z.files else ""
        tail = os.path.splitext(os.path.basename(ck))[0].rsplit("_", 1)[-1]
        return f"ep{tail}" if tail.isdigit() else "ep?"

    runs = {}
    for label, path in (("v1 (paper)", args.v1), ("asinh_aero (asinh99)", args.asinh),
                        ("minmax, no clip", args.minmax)):
        if path:
            z = np.load(path, allow_pickle=True)
            runs[f"{label} {epoch_of(z)}"] = z
    first = next(iter(runs.values()))
    lat, lon = first["lat"], first["lon"]
    species = [str(s) for s in first["aerosol_names"]]
    regions = [str(r) for r in first["regions"]]
    for name, z in runs.items():
        print(f"[read] {name}: transform={z['cond_transform'] if 'cond_transform' in z.files else '?'} "
              f"kind='{z['source_map_kind']}'")

    # ---- 1. the input distortion ------------------------------------------
    masks = {n: box(lat, lon, *b) for n, *b in SOURCE}
    # rows = everything the figure shows; runs = only what has a model behind it.
    rows, extra_series = {}, {}
    for name, z in runs.items():
        rows[name] = z
        if name.startswith("v1") and args.v1_clip_pctl:
            label = f"v1, clip p{100 - args.v1_clip_pctl:g}-p{args.v1_clip_pctl:g} (no model)"
            rows[label], extra_series[label] = conditioning_only_row(species, args.v1_clip_pctl)

    print("\n1. CONDITIONING MOVE, Middle East relative to East Asia "
          "(1.0 would mean Arabia moves as far as East China)")
    print(f"   {'':22s}" + "".join(f"{sp:>12s}" for sp in species))
    raw = {sp: raw_move(sp) for sp in species}
    row = "".join(f"{raw[sp][masks['Middle East']].mean() / raw[sp][masks['East Asia']].mean():12.3f}"
                  for sp in species)
    print(f"   {'RAW emissions':22s}{row}")
    ratios = {}
    for name, z in rows.items():
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

    nrows = 2 + len(rows)
    fig = plt.figure(figsize=(14.5, 3.25 * nrows), constrained_layout=True)
    # Maps everywhere except columns 3-4 of each transform row, which hold time
    # series and so must be ordinary axes, not projected ones.
    gs = fig.add_gridspec(nrows, 4)
    axes = np.empty((nrows, 4), dtype=object)
    for r_ in range(nrows):
        for c_ in range(4):
            if r_ >= 2 and c_ >= 2:
                axes[r_, c_] = fig.add_subplot(gs[r_, c_])
            elif HAVE_CARTOPY:
                import cartopy.crs as ccrs
                axes[r_, c_] = fig.add_subplot(gs[r_, c_], projection=ccrs.Robinson())
            else:
                axes[r_, c_] = fig.add_subplot(gs[r_, c_])
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
                label = f"{unit} per gridpoint (x1e{int(np.floor(np.log10(v)))})"
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
            if kind != "log":
                # Scale into the label: an offset exponent drawn at the bar's
                # end collides with the label text on these narrow panels.
                ex = 10.0 ** int(np.floor(np.log10(v)))
                from matplotlib.ticker import FuncFormatter, MaxNLocator
                cb.locator = MaxNLocator(nbins=3, symmetric=True)
                cb.formatter = FuncFormatter(lambda x, _p, ex=ex: f"{x / ex:.0f}")
                cb.update_ticks()
            cb.set_label(label, fontsize=7)

    # Global-mean NORMALISED conditioning, 1850-2100, through the same stages as
    # the maps (normalise -> smooth -> joint PCA), so the two halves of a row
    # show one object. The anchors come from asian_saturation_by_transform,
    # which fits them exactly as training does. Grey, on its own axis: the
    # global total emission from the same files, for shape.
    from asian_saturation_by_transform import anchors as fit_anchors, \
        transforms as build_maps, smooth as sm2, path as cond_path_of
    from sklearn.decomposition import PCA as _PCA
    KEY = {"v1": "v1 (clip)", "asinh_aero": "asinh (p99.9)", "minmax": "minmax (no clip)"}
    with xr.open_dataset(cond_path_of("hist")) as h_, xr.open_dataset(cond_path_of("ssp370")) as s_:
        hy_, sy_ = h_["year"].values, s_["year"].values
        record = {sp: np.concatenate([np.asarray(h_[sp].values, float)[hy_ <= 2014],
                                      np.asarray(s_[sp].values, float)[sy_ >= 2015]])
                  for sp in species}
    yrs = np.concatenate([hy_[hy_ <= 2014], sy_[sy_ >= 2015]])
    wts = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    fits = {sp: build_maps(fit_anchors(sp)) for sp in species}

    def global_series(sp, key):
        zz = sm2(fits[sp][key](record[sp]))
        T, H, W = zz.shape
        pca = _PCA(n_components=5).fit(zz.reshape(T, -1))
        zp = pca.inverse_transform(pca.transform(zz.reshape(T, -1))).reshape(T, H, W)
        return np.array([np.average(zp[t], weights=wts) for t in range(T)])

    emis_global = {sp: np.array([np.average(record[sp][t], weights=wts) for t in range(len(yrs))])
                   for sp in species}

    for i0, (name, z) in enumerate(rows.items()):
        i = i0 + 2
        d = z["delta"]
        key = KEY.get(str(z["cond_transform"]) if "cond_transform" in z.files else "v1", "v1 (clip)")
        for j in range(2):
            ax = axes[i][j]
            # Diverging around zero. A floor at 0 painted every DECREASE the same
            # black as no change; 0.2-0.6% of cells do fall 1850->2040 (old
            # industrial Europe), by up to -0.47. Two slopes, because rises
            # reach +2 while the deepest fall is under -0.5.
            from matplotlib.colors import TwoSlopeNorm
            vpos = float(np.nanpercentile(d[j][d[j] > 0], 99)) if (d[j] > 0).any() else 1.0
            vneg = float(min(np.nanmin(d[j]), -0.05 * vpos))
            im = draw_map(ax, to_pm180(d[j], lon)[0], cmap="RdBu_r",
                          norm=TwoSlopeNorm(vmin=vneg, vcenter=0.0, vmax=vpos), outline="0.15")
            if HAVE_CARTOPY:
                import cartopy.crs as ccrs
                lat0, lat1, lon0, lon1 = ARABIA
                ax.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0],
                        color="lime", lw=1.2, transform=ccrs.PlateCarree())
            ax.set_title(f"{species[j]} move 1850->2040", fontsize=9)
            if j == 0:
                ax.text(-0.04, 0.5, f"{name}\nnormalised + smoothed + PCA", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
            panel_label(ax, k); k += 1
            cb = fig.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.02)
            # Label the deepest fall, zero and the top: with two slopes an
            # automatic locator put every tick on the positive side.
            cb.set_ticks([vneg, 0.0, vpos])
            cb.set_ticklabels([f"{vneg:.2f}", "0", f"{vpos:.2f}"])
        for j, sp in enumerate(species):
            ax = axes[i][2 + j]
            ax.set_box_aspect(0.55)          # about a Robinson map's height/width
            if name in extra_series:
                zp_ = extra_series[name][sp]
                ser = np.array([np.average(zp_[t], weights=wts) for t in range(zp_.shape[0])])
            else:
                ser = global_series(sp, key)
            ax.plot(yrs, ser, color="#0072B2", lw=1.8, label="normalised, global mean")
            ax.set_ylabel("normalised value", fontsize=8, color="#0072B2")
            ax.tick_params(axis="y", labelcolor="#0072B2", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax.axvline(2014.5, color="0.8", lw=0.8)
            tw = ax.twinx()
            tw.plot(yrs, emis_global[sp], color="0.55", lw=1.1, ls="--", label="emissions, global mean")
            tw.set_ylabel(f"{'Gt SO2' if sp == 'SUL' else 'Gt BC'}/yr per gridpoint", fontsize=7, color="0.45")
            tw.tick_params(axis="y", labelcolor="0.45", labelsize=6)
            tw.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.set_title(f"{sp}: global mean over time", fontsize=9)
            ax.text(0.02, 0.95, f"({PANEL_LETTERS[k]})", transform=ax.transAxes,
                    fontweight="bold", va="top", fontsize=9); k += 1
            if i == 2 and j == 0:
                h1, l1 = ax.get_legend_handles_labels(); h2, l2 = tw.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    fig.suptitle("Saturation test: the emission change in the cond files, and the conditioning each "
                 "normalisation delivers -- its 1850->2040 move and its global mean over time "
                 "(Arabian Peninsula boxed; each panel has its own scale)", fontsize=11)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(args.out + ext, dpi=160, bbox_inches="tight")
    print(f"\nwrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
