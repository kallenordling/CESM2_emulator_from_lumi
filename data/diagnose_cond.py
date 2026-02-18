"""
diagnose_cond.py  –  Conditioning-data diagnostic script
=========================================================
Plots three views of each conditioning variable (CO2, SO2):
  1. Raw values (log10 for SO2 to handle dynamic range)
  2. Normalised values  (norm_zscore / the same path the model sees)
  3. PCA-filtered values (if n_components is set)

For every view the script produces:
  • Global-mean time-series  (full record + highlighted 1850 / 2000 / 2100)
  • Spatial maps at 1850, 2000, 2100  (or nearest available years)
  • A side-by-side comparison strip  (raw | normalised | PCA) at each snapshot year

Usage
-----
  python diagnose_cond.py \
      --cond_file /path/to/emissions.nc \
      --cond_vars CO2 SO2 \
      --n_components 10 40 \
      --out_dir ./cond_diagnostics

All arguments have sensible defaults – just set --cond_file to your file.
"""

import argparse
import os
import warnings
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import SymLogNorm
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers  (mirrors climate_dataset.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def norm_zscore(da: xr.DataArray) -> xr.DataArray:
    vals = da.values.flatten()
    if da.name == "SUL":
        mask = vals > 1e-13
    elif da.name == "CO2":
        mask = vals > 1e-8
    else:
        mask = vals > 0
    vals_log = np.log1p(vals)
    mu    = float(vals_log[mask].mean())
    sigma = float(vals_log[mask].std())
    result = ((da - mu) / max(sigma, 1e-30)).astype("float32")
    # Store params on the DataArray for display
    result.attrs["norm_mu"]    = mu
    result.attrs["norm_sigma"] = sigma
    return result


def normalize_var(da: xr.DataArray) -> xr.DataArray:
    if da.name in ("CO2", "SO2", "SUL"):
        return norm_zscore(da).fillna(0)
    raise ValueError(f"No normalization defined for variable '{da.name}'")


# ──────────────────────────────────────────────────────────────────────────────
# PCA helpers
# ──────────────────────────────────────────────────────────────────────────────

def fit_and_apply_pca(data: np.ndarray, n_components: int, var_name: str = ""):
    """data: (T, H, W) float32 → returns (denoised (T,H,W), pca object)"""
    T, H, W = data.shape
    flat = data.reshape(T, H * W).astype(np.float64)
    n_components = min(n_components, T, H * W)
    pca = PCA(n_components=n_components, whiten=False)
    scores = pca.fit_transform(flat)
    recon  = pca.inverse_transform(scores)
    pct    = pca.explained_variance_ratio_.sum() * 100
    print(f"  [PCA] {var_name}: {n_components} components → {pct:.2f}% variance explained")
    return recon.reshape(T, H, W).astype(np.float32), pca


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

TARGET_YEARS = [1850, 2000, 2100]

def nearest_year(years: np.ndarray, target: int) -> Optional[int]:
    diffs = np.abs(years - target)
    idx   = int(np.argmin(diffs))
    return int(years[idx]) if diffs[idx] <= 20 else None


def _symlog_lims(data: np.ndarray, pct=99.5):
    """Symmetric percentile bounds, useful for raw SO2 with huge dynamic range."""
    hi = float(np.nanpercentile(np.abs(data), pct))
    return -hi, hi


def plot_timeseries(axes_row, years, ts_raw, ts_norm, ts_pca,
                    var_name, snap_years, snap_colors):
    """Three time-series panels (raw | norm | pca) in the supplied axes row."""
    datasets = [
        (ts_raw,  "Raw",        "steelblue",  None),
        (ts_norm, "Normalised", "darkorange",  None),
        (ts_pca,  "PCA-filtered","forestgreen", None),
    ]
    for ax, (ts, label, color, _) in zip(axes_row, datasets):
        if ts is None:
            ax.set_visible(False)
            continue
        ax.plot(years, ts, color=color, lw=1.6, label=label)
        for yr, col in zip(snap_years, snap_colors):
            if yr is not None:
                ax.axvline(yr, color=col, ls="--", lw=1.0, alpha=0.8)
        ax.set_title(f"{var_name}  –  {label} global mean", fontsize=10)
        ax.set_xlabel("Year");  ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        # Annotate range
        ax.annotate(
            f"min={ts.min():.3g}  max={ts.max():.3g}  std={ts.std():.3g}",
            xy=(0.02, 0.05), xycoords="axes fraction", fontsize=7.5,
            color="dimgray",
        )


def plot_spatial_maps(fig, outer_gs, row_idx, snap_data_list,
                      snap_labels, title_prefix, cmap, vmin, vmax,
                      lats, lons, norm_instance=None):
    """
    One row of spatial maps.
    snap_data_list : list of (H,W) arrays – one per snapshot year
    snap_labels    : list of strings
    """
    n = len(snap_data_list)
    inner = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=outer_gs[row_idx],
                                             wspace=0.04)
    for col, (data, label) in enumerate(zip(snap_data_list, snap_labels)):
        ax = fig.add_subplot(inner[col])
        if norm_instance is not None:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                               norm=norm_instance, shading="auto")
        else:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                               vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(
            f"{label}\nmin={data.min():.3g} max={data.max():.3g}",
            fontsize=8,
        )
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat" if col == 0 else "")
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    # Row label on the left
    fig.add_subplot(outer_gs[row_idx]).set_axis_off()          # placeholder
    fig.text(0.01, 1 - (row_idx + 0.5) / outer_gs.get_geometry()[0],
             title_prefix, va="center", ha="left",
             fontsize=9, rotation=90, color="dimgray")


# ──────────────────────────────────────────────────────────────────────────────
# Main per-variable diagnostic
# ──────────────────────────────────────────────────────────────────────────────

def diagnose_variable(
    da_raw    : xr.DataArray,       # original (un-normalised) DataArray, dims (year, lat, lon)
    var_name  : str,
    n_components: Optional[int],
    out_dir   : str,
    years     : np.ndarray,
):
    print(f"\n{'='*60}")
    print(f"  Diagnosing: {var_name}")
    print(f"{'='*60}")

    lats = da_raw.lat.values
    lons = da_raw.lon.values
    raw_np  = da_raw.values.astype(np.float32)   # (T, H, W)
    T, H, W = raw_np.shape

    # ── 1. Normalize ──────────────────────────────────────────────────────────
    da_norm  = normalize_var(da_raw)
    norm_np  = da_norm.values.astype(np.float32)  # (T, H, W)
    mu       = da_norm.attrs.get("norm_mu", float("nan"))
    sigma    = da_norm.attrs.get("norm_sigma", float("nan"))
    print(f"  Norm params: log1p(x) → (x – {mu:.4f}) / {sigma:.4f}")
    print(f"  Raw   stats: min={raw_np.min():.4g}  max={raw_np.max():.4g}  "
          f"std={raw_np.std():.4g}")
    print(f"  Norm  stats: min={norm_np.min():.4f}  max={norm_np.max():.4f}  "
          f"std={norm_np.std():.4f}")

    # ── 2. PCA ────────────────────────────────────────────────────────────────
    pca_np  = None
    pca_obj = None
    if n_components is not None and n_components > 0:
        pca_np, pca_obj = fit_and_apply_pca(norm_np, n_components, var_name)
        print(f"  PCA   stats: min={pca_np.min():.4f}  max={pca_np.max():.4f}  "
              f"std={pca_np.std():.4f}")

    # ── 3. Snapshot year indices ───────────────────────────────────────────────
    snap_actual = [nearest_year(years, y) for y in TARGET_YEARS]
    snap_idx    = [int(np.where(years == y)[0][0]) if y is not None else None
                   for y in snap_actual]
    snap_labels = [str(y) if y is not None else "N/A" for y in snap_actual]
    snap_colors = ["navy", "crimson", "darkorchid"]

    valid_snaps = [(i, lbl, col)
                   for i, lbl, col in zip(snap_idx, snap_labels, snap_colors)
                   if i is not None]

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 1: Global-mean time-series  (raw | norm | pca)
    # ──────────────────────────────────────────────────────────────────────────
    ts_raw  = raw_np.mean(axis=(1, 2))
    ts_norm = norm_np.mean(axis=(1, 2))
    ts_pca  = pca_np.mean(axis=(1, 2))  if pca_np  is not None else None

    n_panels = 3 if pca_np is not None else 2
    fig_ts, axes_ts = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4),
                                   constrained_layout=True)
    if n_panels == 2:
        axes_ts = list(axes_ts) + [None]

    panel_data = [
        (axes_ts[0], ts_raw,  "Raw",          "steelblue"),
        (axes_ts[1], ts_norm, "Normalised",   "darkorange"),
        (axes_ts[2], ts_pca,  "PCA-filtered", "forestgreen"),
    ]
    for ax, ts, label, color in panel_data:
        if ax is None or ts is None:
            continue
        ax.plot(years, ts, color=color, lw=1.8)
        for (si, slbl, scol) in valid_snaps:
            ax.axvline(years[si], color=scol, ls="--", lw=1.2, alpha=0.85,
                       label=slbl)
        ax.set_title(f"{var_name}  –  {label}  (global mean)", fontsize=11)
        ax.set_xlabel("Year");  ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, title="Snapshot years")
        ax.annotate(
            f"min={ts.min():.3g}  max={ts.max():.3g}  std={ts.std():.3g}",
            xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8.5,
            color="dimgray", style="italic",
        )

    fig_ts.suptitle(
        f"{var_name} — Global-mean time series  "
        f"({'PCA: ' + str(n_components) + ' comps' if pca_np is not None else 'no PCA'})",
        fontsize=13, fontweight="bold",
    )
    ts_path = os.path.join(out_dir, f"{var_name}_timeseries.png")
    fig_ts.savefig(ts_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ts)
    print(f"  [SAVED] {ts_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 2: Spatial maps at snapshot years  (one figure per snapshot)
    # ──────────────────────────────────────────────────────────────────────────
    for si, slbl, scol in valid_snaps:
        raw_snap  = raw_np[si]    # (H, W)
        norm_snap = norm_np[si]
        pca_snap  = pca_np[si]  if pca_np  is not None else None

        n_cols = 3 if pca_snap is not None else 2
        fig_sp, axes_sp = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 5),
                                       constrained_layout=True)
        if n_cols == 2:
            axes_sp = list(axes_sp) + [None]

        # Raw  –  use symlog norm so both near-zero (ocean) and peaks are visible
        linthresh_raw = max(float(np.percentile(raw_snap[raw_snap > 0], 10)), 1e-10) \
                        if (raw_snap > 0).any() else 1.0
        raw_hi = float(np.nanpercentile(np.abs(raw_snap), 99.5))

        panels = [
            (axes_sp[0], raw_snap,  "Raw",          "YlOrRd",
             SymLogNorm(linthresh=linthresh_raw, vmin=0, vmax=raw_hi)),
            (axes_sp[1], norm_snap, "Normalised",   "RdBu_r",
             None),   # plain linear, centered
            (axes_sp[2], pca_snap,  "PCA-filtered", "RdBu_r",
             None),
        ]
        for ax, data, label, cmap, snorm in panels:
            if ax is None or data is None:
                continue
            if snorm is not None:
                im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                                   norm=snorm, shading="auto")
            else:
                vabs = max(abs(float(data.min())), abs(float(data.max())), 0.01)
                im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                                   vmin=-vabs, vmax=vabs, shading="auto")
            ax.set_title(
                f"{label}\nmin={data.min():.3g}  max={data.max():.3g}  "
                f"mean={data.mean():.3g}  std={data.std():.3g}",
                fontsize=9,
            )
            ax.set_xlabel("Longitude");  ax.set_ylabel("Latitude")
            fig_sp.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        fig_sp.suptitle(
            f"{var_name}  –  Year {slbl}   "
            f"{'(PCA: ' + str(n_components) + ' comps)' if pca_snap is not None else ''}",
            fontsize=13, fontweight="bold",
        )
        sp_path = os.path.join(out_dir, f"{var_name}_maps_{slbl}.png")
        fig_sp.savefig(sp_path, dpi=150, bbox_inches="tight")
        plt.close(fig_sp)
        print(f"  [SAVED] {sp_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 3: Comparison strip  (all snapshot years × all stages)
    # ──────────────────────────────────────────────────────────────────────────
    stages   = ["Raw", "Normalised"] + (["PCA-filtered"] if pca_np is not None else [])
    n_stages = len(stages)
    n_snaps  = len(valid_snaps)

    fig_cmp, axes_cmp = plt.subplots(
        n_stages, n_snaps,
        figsize=(5.5 * n_snaps, 4.5 * n_stages),
        constrained_layout=True,
    )
    # Ensure 2-D indexing
    if n_stages == 1:
        axes_cmp = axes_cmp[np.newaxis, :]
    if n_snaps == 1:
        axes_cmp = axes_cmp[:, np.newaxis]

    stage_arrays = [raw_np, norm_np] + ([pca_np] if pca_np is not None else [])
    stage_cmaps  = ["YlOrRd", "RdBu_r"] + (["RdBu_r"] if pca_np is not None else [])

    for row, (arr, cmap, stage) in enumerate(zip(stage_arrays, stage_cmaps, stages)):
        # Compute colour limits across all snapshot years for this stage
        snap_vals = np.concatenate([arr[si].ravel() for si, _, _ in valid_snaps])
        if stage == "Raw":
            lo_c, hi_c = 0, float(np.nanpercentile(snap_vals, 99.5))
            linthresh = max(float(np.percentile(snap_vals[snap_vals > 0], 5)), 1e-10) \
                        if (snap_vals > 0).any() else 1.0
            shared_norm = SymLogNorm(linthresh=linthresh, vmin=lo_c, vmax=hi_c)
        else:
            vabs = max(abs(float(snap_vals.min())), abs(float(snap_vals.max())), 0.01)
            shared_norm = None
            lo_c, hi_c = -vabs, vabs

        for col, (si, slbl, scol) in enumerate(valid_snaps):
            ax = axes_cmp[row, col]
            data = arr[si]
            if shared_norm is not None:
                im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                                   norm=shared_norm, shading="auto")
            else:
                im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                                   vmin=lo_c, vmax=hi_c, shading="auto")
            ax.set_title(
                f"Year {slbl}\nmin={data.min():.3g}  max={data.max():.3g}",
                fontsize=8.5, color=scol,
            )
            if col == 0:
                ax.set_ylabel(stage, fontsize=10, fontweight="bold")
            fig_cmp.colorbar(im, ax=ax, shrink=0.8, pad=0.015)

    fig_cmp.suptitle(
        f"{var_name}  –  Comparison strip  "
        f"(Raw / Normalised{' / PCA-filtered' if pca_np is not None else ''})",
        fontsize=13, fontweight="bold",
    )
    cmp_path = os.path.join(out_dir, f"{var_name}_comparison_strip.png")
    fig_cmp.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cmp)
    print(f"  [SAVED] {cmp_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 4: Value distribution histograms  (raw | norm | pca)
    # ──────────────────────────────────────────────────────────────────────────
    n_hist = 3 if pca_np is not None else 2
    fig_h, axes_h = plt.subplots(1, n_hist, figsize=(5 * n_hist, 4),
                                  constrained_layout=True)
    if n_hist == 2:
        axes_h = list(axes_h) + [None]

    hist_panels = [
        (axes_h[0], raw_np,  "Raw",          "steelblue",   True),
        (axes_h[1], norm_np, "Normalised",   "darkorange",  False),
        (axes_h[2], pca_np,  "PCA-filtered", "forestgreen", False),
    ]
    for ax, arr, label, color, do_log_x in hist_panels:
        if ax is None or arr is None:
            continue
        flat = arr.ravel()
        finite = flat[np.isfinite(flat)]
        if do_log_x:
            pos = finite[finite > 0]
            if len(pos):
                ax.hist(np.log10(pos), bins=120, color=color, alpha=0.75,
                        density=True, edgecolor="none")
                ax.set_xlabel("log₁₀(value)")
            else:
                ax.hist(finite, bins=120, color=color, alpha=0.75,
                        density=True, edgecolor="none")
                ax.set_xlabel("Value")
        else:
            ax.hist(finite, bins=120, color=color, alpha=0.75,
                    density=True, edgecolor="none")
            ax.set_xlabel("Value")
        ax.set_title(f"{var_name}  –  {label}\nn={len(finite):,}", fontsize=10)
        ax.set_ylabel("Density");  ax.grid(True, alpha=0.25)
        ax.annotate(
            f"p1={np.percentile(finite,1):.3g}  "
            f"p50={np.percentile(finite,50):.3g}  "
            f"p99={np.percentile(finite,99):.3g}",
            xy=(0.02, 0.96), xycoords="axes fraction", fontsize=8,
            va="top", color="dimgray",
        )

    if pca_np is not None:
        # Extra panel: residual distribution
        residual = (norm_np - pca_np).ravel()
        axes_h[-1].hist(residual, bins=120, color="purple", alpha=0.75,
                        density=True, edgecolor="none")
        axes_h[-1].set_xlabel("Value");  axes_h[-1].set_ylabel("Density")
        axes_h[-1].set_title(
            f"{var_name}  –  Residual (norm − PCA)\n"
            f"std={residual.std():.4f}", fontsize=10,
        )
        axes_h[-1].grid(True, alpha=0.25)

    fig_h.suptitle(f"{var_name}  –  Value distributions", fontsize=13, fontweight="bold")
    hist_path = os.path.join(out_dir, f"{var_name}_histograms.png")
    fig_h.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    print(f"  [SAVED] {hist_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 5: PCA scree + leading EOF patterns  (if PCA enabled)
    # ──────────────────────────────────────────────────────────────────────────
    if pca_obj is not None:
        cumvar = np.cumsum(pca_obj.explained_variance_ratio_) * 100
        n_eofs = min(4, pca_obj.n_components_)
        components = pca_obj.components_.reshape(pca_obj.n_components_, H, W)

        fig_pca, axes_pca = plt.subplots(
            2, max(n_eofs, 2),
            figsize=(5 * max(n_eofs, 2), 8),
            constrained_layout=True,
        )
        # Row 0: scree plot (span across all columns)
        ax_scree = fig_pca.add_axes([0.06, 0.55, 0.88, 0.38])
        ax_scree.plot(np.arange(1, len(cumvar) + 1), cumvar, "o-", ms=4,
                      color="steelblue", lw=1.5)
        ax_scree.axhline(90, color="orange", ls="--", lw=1, label="90 %")
        ax_scree.axhline(95, color="red",    ls="--", lw=1, label="95 %")
        ax_scree.axhline(99, color="darkred",ls=":",  lw=1, label="99 %")
        ax_scree.set_xlabel("Number of components")
        ax_scree.set_ylabel("Cumulative variance explained (%)")
        ax_scree.set_title(
            f"{var_name}  –  PCA scree  "
            f"({n_components} comps kept → {cumvar[n_components-1]:.1f}% var)",
            fontsize=11,
        )
        ax_scree.legend(fontsize=9); ax_scree.grid(True, alpha=0.25)

        # Row 1: first n_eofs EOF spatial patterns
        for k in range(n_eofs):
            ax = fig_pca.axes[k] if k < len(fig_pca.axes) else fig_pca.add_subplot(2, n_eofs, n_eofs + k + 1)
            # Use the bottom row subplots (created by subplots above, but we'll just add them)
        # Create new figure for EOF maps
        fig_eof, axes_eof = plt.subplots(1, n_eofs, figsize=(5.5 * n_eofs, 4),
                                          constrained_layout=True)
        if n_eofs == 1:
            axes_eof = [axes_eof]
        for k, ax in enumerate(axes_eof[:n_eofs]):
            eof = components[k]
            vabs = float(np.abs(eof).max())
            im = ax.pcolormesh(lons, lats, eof, cmap="RdBu_r",
                               vmin=-vabs, vmax=vabs, shading="auto")
            pct_k = pca_obj.explained_variance_ratio_[k] * 100
            ax.set_title(f"EOF {k+1}  ({pct_k:.2f}% var)", fontsize=9)
            ax.set_xlabel("Lon"); ax.set_ylabel("Lat" if k == 0 else "")
            fig_eof.colorbar(im, ax=ax, shrink=0.8)
        fig_eof.suptitle(f"{var_name}  –  Leading EOF patterns", fontsize=12, fontweight="bold")
        eof_path = os.path.join(out_dir, f"{var_name}_PCA_EOFs.png")
        fig_eof.savefig(eof_path, dpi=150, bbox_inches="tight")
        plt.close(fig_eof)
        print(f"  [SAVED] {eof_path}")

        # Save scree separately too
        fig_scree2, ax_sc2 = plt.subplots(figsize=(8, 4))
        ax_sc2.plot(np.arange(1, len(cumvar) + 1), cumvar, "o-", ms=4,
                    color="steelblue", lw=1.5)
        ax_sc2.axhline(90, color="orange", ls="--", label="90 %")
        ax_sc2.axhline(95, color="red",    ls="--", label="95 %")
        ax_sc2.axhline(99, color="darkred",ls=":",  label="99 %")
        ax_sc2.axvline(n_components, color="black", ls="-.", lw=1.5,
                       label=f"kept ({n_components})")
        ax_sc2.set_xlabel("Number of components"); ax_sc2.set_ylabel("Cumulative var (%)")
        ax_sc2.set_title(f"{var_name}  –  PCA scree", fontsize=12)
        ax_sc2.legend(); ax_sc2.grid(True, alpha=0.25)
        scree_path = os.path.join(out_dir, f"{var_name}_PCA_scree.png")
        fig_scree2.savefig(scree_path, dpi=150, bbox_inches="tight")
        plt.close(fig_scree2)
        plt.close(fig_pca)
        print(f"  [SAVED] {scree_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Conditioning-data diagnostic script")
    p.add_argument("--cond_file",    required=True,
                   help="Path to the conditioning NetCDF file (e.g. emissions.nc)")
    p.add_argument("--cond_vars",    nargs="+", default=["CO2", "SO2"],
                   help="Variable names to diagnose")
    p.add_argument("--n_components", nargs="+", type=int, default=None,
                   help="PCA components per variable (same order as --cond_vars). "
                        "Pass 0 or omit to skip PCA.")
    p.add_argument("--out_dir",      default="./cond_diagnostics",
                   help="Output directory for figures")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nOutput directory: {args.out_dir}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading: {args.cond_file}")
    ds = xr.open_dataset(args.cond_file)
    print(ds)

    all_years = ds["year"].values if "year" in ds.coords else \
                ds[list(ds.data_vars)[0]].coords.get("year",
                    ds[list(ds.data_vars)[0]].coords.get("time")).values

    print(f"\nYears in file: {all_years[0]} – {all_years[-1]}  ({len(all_years)} steps)")

    # Resolve n_components per variable
    n_comp_map = {}
    if args.n_components is not None:
        if len(args.n_components) == 1:
            n_comp_map = {v: args.n_components[0] for v in args.cond_vars}
        elif len(args.n_components) == len(args.cond_vars):
            n_comp_map = dict(zip(args.cond_vars, args.n_components))
        else:
            raise ValueError("--n_components must be length 1 or match --cond_vars length")
    else:
        n_comp_map = {v: None for v in args.cond_vars}

    # ── Diagnose each variable ────────────────────────────────────────────────
    for var_name in args.cond_vars:
        if var_name not in ds:
            print(f"  [WARN] '{var_name}' not found in dataset – skipping.")
            continue

        da_raw = ds[var_name]
        # Ensure dims are (year, lat, lon)
        if "year" not in da_raw.dims:
            da_raw = da_raw.rename({"time": "year"}) if "time" in da_raw.dims else da_raw

        da_raw = da_raw.astype("float32")
        da_raw.name = var_name     # required by norm_zscore

        n_comp = n_comp_map.get(var_name)
        if n_comp == 0:
            n_comp = None

        diagnose_variable(
            da_raw      = da_raw,
            var_name    = var_name,
            n_components= n_comp,
            out_dir     = args.out_dir,
            years       = all_years,
        )

    print(f"\n{'='*60}")
    print(f"  All diagnostics saved to: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()