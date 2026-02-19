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
import sys
import warnings
from typing import Optional, Dict, List, Tuple

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers  (mirrors climate_dataset.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def norm_zscore(
    da: xr.DataArray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    mode: str = "percentile",   # "percentile" | "zscore"
    n_std: float = 3.0,         # only used when mode="zscore"
) -> xr.DataArray:
    """
    Log1p normalisation to [-1, 1].

    mode='percentile'  (default)
        lo = p(lo_pct) of log1p(non-ocean values)
        hi = p(hi_pct) of log1p(non-ocean values)
        Robust to outliers; the bulk of real-emission cells fill [-1, 1].

    mode='zscore'
        lo = mean - n_std * std   of log1p(non-ocean values)
        hi = mean + n_std * std
        Keeps the Gaussian shape centred; n_std=3 maps ±3σ → ±1.
        Useful when you want the model to see relative anomalies rather than
        the absolute rank of each cell.

    Both modes:
      - Operate on log1p(da) so transform and statistics are in the same space.
      - Compute statistics on non-ocean (above-floor) cells only.
      - Pin below-floor cells to exactly -1.
      - Hard-clip output to [-1, 1] via numpy (bypasses xarray edge-cases).
    """
    floor = {"SUL": 1e-9, "CO2": 1e-3}.get(da.name, 0.0)

    vals     = da.values.flatten().astype(np.float64)
    log_vals = np.log1p(np.clip(vals, 0.0, None))
    mask     = vals > floor

    if mask.sum() == 0:
        result = xr.full_like(da, -1.0).astype("float32")
        result.attrs.update(norm_lo=0.0, norm_hi=1.0)
        return result

    if mode == "percentile":
        lo = float(np.percentile(log_vals[mask], lo_pct))
        hi = float(np.percentile(log_vals[mask], hi_pct))
    elif mode == "zscore":
        mu  = float(log_vals[mask].mean())
        std = float(log_vals[mask].std())
        lo  = mu - n_std * std
        hi  = mu + n_std * std
    else:
        raise ValueError(f"mode must be 'percentile' or 'zscore', got '{mode}'")

    # Transform DataArray in log-space, stretch [lo, hi] -> [-1, 1]
    log_da = np.log1p(da.clip(min=0))
    normed = 2.0 * (log_da - lo) / max(hi - lo, 1e-30) - 1.0

    # Pin ocean / below-floor cells to exactly -1
    normed = xr.where(da <= floor, -1.0, normed)

    # Hard clip via numpy — guarantees [-1, 1] regardless of outliers
    out = np.clip(normed.values, -1.0, 1.0).astype(np.float32)
    result = xr.DataArray(out, dims=da.dims, coords=da.coords, name=da.name)
    result.attrs["norm_lo"]   = lo
    result.attrs["norm_hi"]   = hi
    result.attrs["norm_mode"] = mode
    return result


def normalize_var(da: xr.DataArray, norm_mode: str = "percentile", n_std: float = 3.0) -> xr.DataArray:
    if da.name in ("CO2", "SO2", "SUL"):
        return norm_zscore(da, mode=norm_mode, n_std=n_std).fillna(0)
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


def make_symlog_norm(data: np.ndarray, pct_hi=99.5, linthresh_pct=5.0):
    """Build a SymLogNorm that works for both raw (positive-only) and
    signed (normalised / PCA) arrays.

    * ``linthresh`` is set to the *linthresh_pct*-th percentile of the
      absolute non-zero values so the linear region spans the typical
      noise floor, not just one tick.
    * ``vmin`` / ``vmax`` are clipped at the *pct_hi* percentile of |data|
      so a handful of extreme outliers don't crush the colour scale.
    """
    flat = data.ravel()
    finite = flat[np.isfinite(flat)]
    abs_nonzero = np.abs(finite[finite != 0])

    if len(abs_nonzero) == 0:
        return SymLogNorm(linthresh=1e-6, vmin=-1, vmax=1)

    linthresh = max(float(np.percentile(abs_nonzero, linthresh_pct)), 1e-10)
    hi = float(np.nanpercentile(np.abs(finite), pct_hi))

    # For all-positive data keep vmin=0; for signed data use -hi
    if float(finite.min()) >= 0:
        vmin, vmax = 0.0, hi
    else:
        vmin, vmax = -hi, hi

    return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)


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
    da_raw    : xr.DataArray,
    var_name  : str,
    n_components: Optional[int],
    out_dir   : str,
    years     : np.ndarray,
    norm_mode : str = "percentile",
    n_std     : float = 3.0,
):
    print(f"\n{'='*60}")
    print(f"  Diagnosing: {var_name}")
    print(f"{'='*60}")

    lats = da_raw.lat.values
    lons = da_raw.lon.values
    raw_np  = da_raw.values.astype(np.float32)   # (T, H, W)
    T, H, W = raw_np.shape

    # ── 1. Normalize ──────────────────────────────────────────────────────────
    da_norm  = normalize_var(da_raw, norm_mode=norm_mode, n_std=n_std)
    norm_np  = da_norm.values.astype(np.float32)  # (T, H, W)
    lo_val   = da_norm.attrs.get("norm_lo", float("nan"))
    hi_val   = da_norm.attrs.get("norm_hi", float("nan"))
    print(f"  Norm params: log1p(x) percentiles → lo={lo_val:.4f}  hi={hi_val:.4f}")
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
        f"[norm={norm_mode}]  "
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

        # Shared linear scale for norm/PCA panels (both already in [-1, 1])
        signed_data = [d for d in [norm_snap, pca_snap] if d is not None]
        vabs = max(float(np.abs(np.concatenate([d.ravel() for d in signed_data])).max()), 0.01)

        panels = [
            (axes_sp[0], raw_snap,  "Raw",          "YlOrRd",  True),
            (axes_sp[1], norm_snap, "Normalised",   "RdBu_r",  False),
            (axes_sp[2], pca_snap,  "PCA-filtered", "RdBu_r",  False),
        ]
        for ax, data, label, cmap, use_symlog in panels:
            if ax is None or data is None:
                continue
            if use_symlog:
                im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                                   norm=make_symlog_norm(data), shading="auto")
            else:
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
        snap_vals = np.concatenate([arr[si].ravel() for si, _, _ in valid_snaps])

        if stage == "Raw":
            # SymLogNorm for raw — shared across snapshot columns
            shared_norm = make_symlog_norm(snap_vals)
            vmin_l, vmax_l = None, None   # unused when norm is set
        else:
            # Linear symmetric scale for normalised / PCA-filtered
            shared_norm = None
            vabs = max(float(np.abs(snap_vals).max()), 0.01)
            vmin_l, vmax_l = -vabs, vabs

        for col, (si, slbl, scol) in enumerate(valid_snaps):
            ax = axes_cmp[row, col]
            data = arr[si]
            if shared_norm is not None:
                im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                                   norm=shared_norm, shading="auto")
            else:
                im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                                   vmin=vmin_l, vmax=vmax_l, shading="auto")
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
            im = ax.pcolormesh(lons, lats, eof, cmap="RdBu_r",
                               norm=make_symlog_norm(eof), shading="auto")
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


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL-CONDITIONING DIAGNOSTIC
#  Visualises how the conditioning tensor is processed through every phase of
#  the UNetModel3D, loaded from a saved checkpoint.
# ══════════════════════════════════════════════════════════════════════════════

def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class _HookStore:
    """Lightweight container that registers forward hooks and collects outputs."""

    def __init__(self):
        self.records: Dict[str, "torch.Tensor"] = {}
        self._handles = []

    def register(self, module: "torch.nn.Module", name: str) -> None:
        # Capture the name in the closure; import torch inside the hook so it
        # is always resolved at call-time regardless of lazy import ordering.
        _name = name

        def _hook(mod, inp, out):
            import torch as _torch  # always available here – the model already ran
            if isinstance(out, _torch.Tensor):
                self.records[_name] = out.detach().cpu()
            elif (isinstance(out, (tuple, list))
                  and len(out) > 0
                  and isinstance(out[0], _torch.Tensor)):
                self.records[_name] = out[0].detach().cpu()

        handle = module.register_forward_hook(_hook)
        self._handles.append(handle)

    def remove_all(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _load_model_from_checkpoint(checkpoint_path: str, model_config_path: str):
    """
    Load UNetModel3D from an EMA checkpoint.
    Returns (model, device).
    """
    torch = _try_import_torch()
    if torch is None:
        raise ImportError("PyTorch is required for model diagnostics.")

    # Lazy imports so the rest of the script still works without torch
    import sys, os
    # Allow the user's repo to be on the path
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from ema_pytorch import EMA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [MODEL] Using device: {device}")

    # Load config and build model
    cfg = OmegaConf.load(model_config_path)
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg
    from models.video_net import UNetModel3D
    model = instantiate(model_cfg)

    # Load checkpoint
    chkpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema_sd = chkpt.get("EMA", None)
    if ema_sd is None:
        raise KeyError("Checkpoint does not contain an 'EMA' key.")

    ema = EMA(model, beta=0.9999, update_after_step=100, update_every=10)
    ema.ema_model.load_state_dict(ema_sd)
    ema.eval()
    model = ema.ema_model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [MODEL] Loaded: {n_params/1e6:.1f}M parameters")
    return model, device


def _build_cond_tensor(
    cond_np_dict: Dict[str, np.ndarray],   # {var_name: (T, H, W)}
    year_idx: int,
    seq_len: int,
    device,
) -> "torch.Tensor":
    """
    Build a (1, C, seq_len, H, W) conditioning tensor for a single year slice.
    year_idx is the centre year; we replicate it seq_len times (static snapshot).
    """
    import torch
    channels = []
    for arr in cond_np_dict.values():
        frame = arr[year_idx]                          # (H, W)
        seq   = np.stack([frame] * seq_len, axis=0)   # (seq_len, H, W)
        channels.append(seq)
    cond = np.stack(channels, axis=0)                 # (C, seq_len, H, W)
    cond = torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(device)
    return cond


def _rms(t: "torch.Tensor") -> float:
    """Root-mean-square of all elements."""
    import torch
    return float(t.float().pow(2).mean().sqrt().item())


def _collect_phase_stats(
    model,
    noise: "torch.Tensor",
    cond_tensor: "torch.Tensor",
    null_tensor: "torch.Tensor",
    device,
) -> Tuple[Dict, Dict]:
    """
    Run two forward passes (cond ON and cond OFF) with hooks at every UNet phase.
    Returns (stats_cond, stats_null) where each is a dict:
        phase_name -> {"rms": float, "mean": float, "std": float, "shape": tuple}
    """
    import torch

    def _run_with_hooks(cond_input):
        store = _HookStore()
        # ── encoder phases ──────────────────────────────────────────────────
        if model.cond_encoder is not None:
            store.register(model.cond_encoder, "cond_encoder")
            store.register(model.cond_scale,   "cond_scale")
            store.register(model.cond_shift,   "cond_shift")
        store.register(model.time_mlp,      "time_mlp")
        store.register(model.input_conv,    "input_conv")
        store.register(model.input_temp_op, "input_temp_op")

        # ── down blocks ─────────────────────────────────────────────────────
        for idx, block_list in enumerate(model.downs):
            block1, block2, s_attn, t_attn, down = block_list
            store.register(block1, f"down_{idx}_block1")
            store.register(block2, f"down_{idx}_block2")
            store.register(down,   f"down_{idx}_sample")

        # ── bottleneck ──────────────────────────────────────────────────────
        store.register(model.mid_block1,       "mid_block1")
        store.register(model.mid_spatial_attn, "mid_spatial_attn")
        store.register(model.mid_block2,       "mid_block2")

        # ── up blocks ───────────────────────────────────────────────────────
        for idx, block_list in enumerate(model.ups):
            block1, block2, s_attn, t_attn, up = block_list
            store.register(block1, f"up_{idx}_block1")
            store.register(block2, f"up_{idx}_block2")
            store.register(up,    f"up_{idx}_sample")

        store.register(model.out_conv, "out_conv")

        # ── dummy timestep (mid-noise level for a fair probe) ───────────────
        t_val = torch.tensor([0.0], device=device)

        with torch.no_grad():
            model(noise, t_val, cond_map=cond_input)

        store.remove_all()

        stats = {}
        for name, tensor in store.records.items():
            flat = tensor.float().reshape(-1)
            stats[name] = {
                "rms":   float(flat.pow(2).mean().sqrt()),
                "mean":  float(flat.mean()),
                "std":   float(flat.std()),
                "shape": tuple(tensor.shape),
            }
        return stats

    stats_cond = _run_with_hooks(cond_tensor)
    stats_null = _run_with_hooks(null_tensor)
    return stats_cond, stats_null


def _collect_output_diff_spatial(
    model,
    noise: "torch.Tensor",
    cond_tensor: "torch.Tensor",
    device,
) -> np.ndarray:
    """
    Returns the mean absolute spatial difference between cond and null outputs.
    Shape: (H, W)
    """
    import torch
    null_tensor = torch.zeros_like(cond_tensor)
    t_val = torch.tensor([0.0], device=device)

    with torch.no_grad():
        out_cond = model(noise, t_val, cond_map=cond_tensor)
        out_null = model(noise, t_val, cond_map=null_tensor)

    diff = (out_cond - out_null).abs()               # (1, C, T, H, W)
    diff_map = diff.squeeze(0).mean(dim=(0, 1))       # mean over C and T → (H, W)
    return diff_map.cpu().numpy()


def _collect_cond_vectors(
    model,
    cond_tensor: "torch.Tensor",
    device,
) -> Dict[str, np.ndarray]:
    """
    Capture the raw intermediate embedding vectors from the conditioning pathway:
    cond_encoder output, cond_scale, cond_shift, time_emb (before/after injection).
    """
    import torch
    vectors = {}

    class _Grabber:
        def __init__(self, key):
            self.key = key
        def __call__(self, mod, inp, out):
            import torch as _torch
            if isinstance(out, _torch.Tensor):
                vectors[self.key] = out.squeeze().detach().cpu().numpy()

    handles = []
    if model.cond_encoder is not None:
        handles.append(model.cond_encoder.register_forward_hook(_Grabber("encoder_feat")))
        handles.append(model.cond_scale.register_forward_hook(_Grabber("scale")))
        handles.append(model.cond_shift.register_forward_hook(_Grabber("shift")))

    # Capture time_mlp BEFORE cond injection by hooking time_mlp output
    handles.append(model.time_mlp.register_forward_hook(_Grabber("time_emb_raw")))

    t_val = torch.tensor([0.0], device=device)
    with torch.no_grad():
        model(noise := torch.zeros(
            1, model.input_conv.in_channels,
            cond_tensor.shape[2], cond_tensor.shape[3], cond_tensor.shape[4],
            device=device), t_val, cond_map=cond_tensor)

    for h in handles:
        h.remove()
    return vectors


# ──────────────────────────────────────────────────────────────────────────────
# Figure builders
# ──────────────────────────────────────────────────────────────────────────────

_PHASE_ORDER = [
    "cond_encoder", "cond_scale", "cond_shift", "time_mlp",
    "input_conv", "input_temp_op",
    "down_0_block1", "down_0_block2", "down_0_sample",
    "down_1_block1", "down_1_block2", "down_1_sample",
    "down_2_block1", "down_2_block2", "down_2_sample",
    "down_3_block1", "down_3_block2", "down_3_sample",
    "mid_block1", "mid_spatial_attn", "mid_block2",
    "up_0_block1",  "up_0_block2",  "up_0_sample",
    "up_1_block1",  "up_1_block2",  "up_1_sample",
    "up_2_block1",  "up_2_block2",  "up_2_sample",
    "up_3_block1",  "up_3_block2",  "up_3_sample",
    "out_conv",
]

_SECTION_COLORS = {
    "cond": "#e07b54",
    "time": "#5b8dd9",
    "input": "#8ecf72",
    "down":  "#c488d6",
    "mid":   "#e8c23a",
    "up":    "#4ec4c4",
    "out":   "#d95f5f",
}

def _phase_color(name: str) -> str:
    if name.startswith("cond"):   return _SECTION_COLORS["cond"]
    if name.startswith("time"):   return _SECTION_COLORS["time"]
    if name.startswith("input"):  return _SECTION_COLORS["input"]
    if name.startswith("down"):   return _SECTION_COLORS["down"]
    if name.startswith("mid"):    return _SECTION_COLORS["mid"]
    if name.startswith("up"):     return _SECTION_COLORS["up"]
    return _SECTION_COLORS["out"]


def plot_phase_activation_norms(
    stats_cond: Dict,
    stats_null: Dict,
    year_label: str,
    out_dir: str,
):
    """
    Bar-chart showing per-phase RMS activations with cond (solid) vs null (hatched),
    plus a second panel for the absolute difference.
    """
    phases = [p for p in _PHASE_ORDER if p in stats_cond]

    rms_c   = np.array([stats_cond[p]["rms"] for p in phases])
    rms_n   = np.array([stats_null[p]["rms"] for p in phases])
    rms_diff = rms_c - rms_n

    x = np.arange(len(phases))
    colors = [_phase_color(p) for p in phases]

    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(phases) * 0.55), 13),
                              constrained_layout=True)

    # ── Panel 1: RMS side-by-side ────────────────────────────────────────────
    ax = axes[0]
    bars_c = ax.bar(x - 0.2, rms_c, 0.38, color=colors, alpha=0.85, label="Cond ON")
    bars_n = ax.bar(x + 0.2, rms_n, 0.38, color=colors, alpha=0.40,
                    hatch="///", edgecolor="white", label="Cond OFF (null)")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("RMS activation", fontsize=10)
    ax.set_title(f"Per-phase RMS activations  –  Year {year_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate cond encoder phases
    for i, p in enumerate(phases):
        if p.startswith("cond"):
            ax.annotate("↑ cond\nencoder", xy=(i - 0.2, rms_c[i]),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=6.5, ha="center", color=_SECTION_COLORS["cond"],
                        fontweight="bold")

    # ── Panel 2: RMS difference (cond - null) ────────────────────────────────
    ax2 = axes[1]
    bar_diff = ax2.bar(x, rms_diff, 0.55, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases, rotation=55, ha="right", fontsize=7.5)
    ax2.set_ylabel("ΔRMS  (cond − null)", fontsize=10)
    ax2.set_title("Conditioning effect per phase  (positive = cond amplifies activations)",
                  fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # Colour bars by sign for clarity
    for bar, val in zip(bar_diff, rms_diff):
        bar.set_color("#d9534f" if val < 0 else "#5cb85c")
        bar.set_alpha(0.80)

    # ── Panel 3: Relative effect (%) ─────────────────────────────────────────
    ax3 = axes[2]
    eps = 1e-10
    rms_rel = (rms_diff / (rms_n + eps)) * 100
    bar_rel = ax3.bar(x, rms_rel, 0.55, alpha=0.80)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(phases, rotation=55, ha="right", fontsize=7.5)
    ax3.set_ylabel("Relative effect (%)", fontsize=10)
    ax3.set_title("Relative conditioning effect per phase  ( (cond−null)/null × 100 )",
                  fontsize=11)
    ax3.grid(axis="y", alpha=0.3)

    for bar, val in zip(bar_rel, rms_rel):
        bar.set_color("#d9534f" if val < 0 else "#5cb85c")
        bar.set_alpha(0.80)

    # Legend for section colours
    legend_patches = [mpatches.Patch(color=c, label=k.capitalize())
                      for k, c in _SECTION_COLORS.items()]
    fig.legend(handles=legend_patches, loc="upper right",
               title="UNet section", fontsize=8, ncol=2, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(
        f"UNet conditioning diagnostic  –  Year {year_label}\n"
        f"Each bar = one forward-hook capture point through the network",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out_path = os.path.join(out_dir, f"model_phase_norms_{year_label}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def plot_conditioning_flow(
    cond_vectors_per_year: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
):
    """
    Shows how the conditioning embedding vectors (encoder feat, scale, shift,
    time_emb) look at different years.  One column per year, one row per
    embedding stage.
    """
    stages = ["encoder_feat", "scale", "shift", "time_emb_raw"]
    stage_labels = {
        "encoder_feat": "Cond encoder\nfeature",
        "scale":        "FiLM scale\nγ(cond)",
        "shift":        "FiLM shift\nβ(cond)",
        "time_emb_raw": "Time emb\n(pre-injection)",
    }
    years_available = list(cond_vectors_per_year.keys())
    n_years  = len(years_available)
    n_stages = len(stages)

    fig, axes = plt.subplots(n_stages, n_years,
                             figsize=(5 * n_years, 3.2 * n_stages),
                             constrained_layout=True)
    if n_stages == 1:
        axes = axes[np.newaxis, :]
    if n_years == 1:
        axes = axes[:, np.newaxis]

    for row, stage in enumerate(stages):
        for col, yr_lbl in enumerate(years_available):
            ax = axes[row, col]
            vec = cond_vectors_per_year[yr_lbl].get(stage)
            if vec is None:
                ax.set_visible(False)
                continue

            vec = vec.ravel()
            # Split into head/tail if very long
            if len(vec) > 256:
                ax.plot(vec[:256], lw=0.8, color=_phase_color(stage))
                ax.set_xlabel(f"dim (first 256 of {len(vec)})", fontsize=7.5)
            else:
                ax.bar(np.arange(len(vec)), vec, color=_phase_color(stage), alpha=0.7, width=1.0)
                ax.set_xlabel("Dimension", fontsize=7.5)

            ax.axhline(0, color="black", lw=0.6, ls="--")
            if col == 0:
                ax.set_ylabel(stage_labels.get(stage, stage), fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(f"Year {yr_lbl}", fontsize=10, fontweight="bold")

            ax.annotate(
                f"μ={vec.mean():.3g}  σ={vec.std():.3g}\n"
                f"min={vec.min():.3g}  max={vec.max():.3g}",
                xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=7, va="top", color="dimgray",
            )
            ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Conditioning embedding flow through the encoder\n"
        "(columns = years, rows = embedding stage)",
        fontsize=13, fontweight="bold",
    )
    out_path = os.path.join(out_dir, "model_cond_flow.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def plot_scale_shift_evolution(
    scale_per_year: Dict[str, np.ndarray],
    shift_per_year: Dict[str, np.ndarray],
    out_dir: str,
):
    """
    Box-plot style evolution of FiLM scale and shift vectors across years,
    showing how the model's attention to conditioning evolves over time.
    """
    years  = list(scale_per_year.keys())
    n_yrs  = len(years)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, data_dict, title, color in zip(
        axes,
        [scale_per_year, shift_per_year],
        ["FiLM scale γ(cond)  – mean ± 2σ across embedding dims",
         "FiLM shift β(cond)  – mean ± 2σ across embedding dims"],
        [_SECTION_COLORS["cond"], _SECTION_COLORS["time"]],
    ):
        means  = np.array([data_dict[yr].mean()      for yr in years])
        stds   = np.array([data_dict[yr].std()       for yr in years])
        p25    = np.array([np.percentile(data_dict[yr], 25) for yr in years])
        p75    = np.array([np.percentile(data_dict[yr], 75) for yr in years])

        x = np.arange(n_yrs)
        ax.fill_between(x, means - 2 * stds, means + 2 * stds, alpha=0.2, color=color)
        ax.fill_between(x, p25, p75, alpha=0.35, color=color, label="IQR")
        ax.plot(x, means, "o-", color=color, lw=2.0, ms=6, label="mean")

        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=30, fontsize=8.5)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="black", lw=0.7, ls="--")

    fig.suptitle(
        "FiLM conditioning gate evolution across snapshot years\n"
        "Larger γ/β deviation from identity → stronger conditioning signal",
        fontsize=12, fontweight="bold",
    )
    out_path = os.path.join(out_dir, "model_film_evolution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def plot_output_diff_maps(
    diff_maps: Dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    out_dir: str,
):
    """
    Spatial maps of |output(cond) − output(null)| at each snapshot year.
    Bright regions are where the model's prediction is most changed by the
    conditioning signal.
    """
    years  = list(diff_maps.keys())
    n      = len(years)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Shared colour scale
    all_vals = np.concatenate([d.ravel() for d in diff_maps.values()])
    vmax     = float(np.nanpercentile(np.abs(all_vals), 99.5))

    for ax, yr_lbl in zip(axes, years):
        diff = diff_maps[yr_lbl]
        im   = ax.pcolormesh(lons, lats, diff, cmap="inferno",
                              vmin=0, vmax=vmax, shading="auto")
        ax.set_title(
            f"Year {yr_lbl}\n"
            f"mean={diff.mean():.4g}  max={diff.max():.4g}",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, shrink=0.80, label="|Δ output|")

    fig.suptitle(
        "Model output sensitivity to conditioning  |out(cond) − out(null)|\n"
        "Bright = model here is most affected by CO₂/SO₂ conditioning",
        fontsize=13, fontweight="bold",
    )
    out_path = os.path.join(out_dir, "model_output_diff_maps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


def plot_phase_heatmap(
    stats_per_year: Dict[str, Tuple[Dict, Dict]],
    out_dir: str,
):
    """
    2-D heatmap: rows = UNet phases, columns = snapshot years.
    Cell value = relative conditioning effect (%) = (rms_cond - rms_null) / rms_null * 100.
    Gives an instant overview of which phases respond to conditioning at which time.
    """
    years  = list(stats_per_year.keys())
    # Union of all phases present
    all_phases = [p for p in _PHASE_ORDER
                  if any(p in stats_per_year[yr][0] for yr in years)]

    matrix = np.zeros((len(all_phases), len(years)))
    for col, yr in enumerate(years):
        sc, sn = stats_per_year[yr]
        for row, phase in enumerate(all_phases):
            if phase in sc and phase in sn:
                rms_c = sc[phase]["rms"]
                rms_n = sn[phase]["rms"] + 1e-10
                matrix[row, col] = (rms_c - rms_n) / rms_n * 100

    vabs = max(float(np.abs(matrix).max()), 0.1)
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(years)), max(8, 0.35 * len(all_phases))),
                            constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=10)
    ax.set_yticks(range(len(all_phases)))
    ax.set_yticklabels(all_phases, fontsize=7.5)
    ax.set_xlabel("Snapshot year", fontsize=11)
    ax.set_ylabel("UNet phase", fontsize=11)
    ax.set_title(
        "Conditioning effect heatmap\n"
        "(rms_cond − rms_null) / rms_null  ×  100%\n"
        "Red = cond reduces activation, Blue = cond amplifies",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Relative effect (%)")

    # Horizontal lines separating UNet sections
    section_starts = {}
    for i, p in enumerate(all_phases):
        section = p.split("_")[0] if "_" in p else p
        if section not in section_starts:
            section_starts[section] = i
            if i > 0:
                ax.axhline(i - 0.5, color="white", lw=1.5)

    out_path = os.path.join(out_dir, "model_phase_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Top-level model diagnostic orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def diagnose_model_conditioning(
    checkpoint_path: str,
    model_config_path: str,
    cond_np_dict: Dict[str, np.ndarray],    # {var_name: (T, H, W)} – normalized
    years: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    out_dir: str,
    seq_len: int = 10,
    snap_years: List[int] = None,
):
    """
    Orchestrates all model-conditioning diagnostics.

    Parameters
    ----------
    checkpoint_path   : path to .pt checkpoint with 'EMA' key
    model_config_path : path to the Hydra YAML config that defines cfg.model
    cond_np_dict      : dict of normalised conditioning arrays (T, H, W) per variable
    years             : 1-D array of years corresponding to axis-0 of cond_np_dict values
    lats, lons        : coordinate arrays
    out_dir           : directory to save all figures
    seq_len           : temporal length of dummy input (match training seq_len)
    snap_years        : years to probe (default [1850, 2000, 2100])
    """
    torch = _try_import_torch()
    if torch is None:
        print("[WARN] PyTorch not available – skipping model diagnostics.")
        return

    if snap_years is None:
        snap_years = [1850, 2000, 2100]

    print(f"\n{'='*60}")
    print(f"  MODEL CONDITIONING DIAGNOSTIC")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Config     : {model_config_path}")
    print(f"{'='*60}")

    model, device = _load_model_from_checkpoint(checkpoint_path, model_config_path)

    # Resolve snapshot year indices
    snap_info = []
    for target_yr in snap_years:
        diffs   = np.abs(years - target_yr)
        nearest = int(np.argmin(diffs))
        if diffs[nearest] <= 20:
            snap_info.append((nearest, str(int(years[nearest]))))
    if not snap_info:
        print("  [WARN] No snapshot years found in dataset – aborting model diagnostic.")
        return

    # Build a fixed noise tensor (same for all years, so differences are purely from cond)
    first_cond = next(iter(cond_np_dict.values()))
    H, W = first_cond.shape[1], first_cond.shape[2]
    in_channels = model.input_conv.in_channels
    noise = torch.randn(1, in_channels, seq_len, H, W, device=device) * 0.1

    # ── 1. Per-phase activation-norm stats for each snapshot year ─────────────
    stats_per_year: Dict[str, Tuple[Dict, Dict]] = {}
    for (yr_idx, yr_lbl) in snap_info:
        print(f"\n  [PROBE] Year {yr_lbl} ...")
        cond_t = _build_cond_tensor(cond_np_dict, yr_idx, seq_len, device)
        null_t = torch.zeros_like(cond_t)
        sc, sn = _collect_phase_stats(model, noise, cond_t, null_t, device)
        stats_per_year[yr_lbl] = (sc, sn)
        plot_phase_activation_norms(sc, sn, yr_lbl, out_dir)

    # ── 2. Multi-year phase heatmap ───────────────────────────────────────────
    plot_phase_heatmap(stats_per_year, out_dir)

    # ── 3. Conditioning embedding flow for each year ──────────────────────────
    cond_vectors_per_year: Dict[str, Dict[str, np.ndarray]] = {}
    for (yr_idx, yr_lbl) in snap_info:
        cond_t = _build_cond_tensor(cond_np_dict, yr_idx, seq_len, device)
        vecs   = _collect_cond_vectors(model, cond_t, device)
        cond_vectors_per_year[yr_lbl] = vecs

    plot_conditioning_flow(cond_vectors_per_year, out_dir)

    # ── 4. FiLM scale/shift evolution ────────────────────────────────────────
    scale_dict = {yr: vecs["scale"] for yr, vecs in cond_vectors_per_year.items()
                  if "scale" in vecs}
    shift_dict = {yr: vecs["shift"] for yr, vecs in cond_vectors_per_year.items()
                  if "shift" in vecs}
    if scale_dict:
        plot_scale_shift_evolution(scale_dict, shift_dict, out_dir)

    # ── 5. Output spatial sensitivity maps ───────────────────────────────────
    diff_maps: Dict[str, np.ndarray] = {}
    for (yr_idx, yr_lbl) in snap_info:
        cond_t = _build_cond_tensor(cond_np_dict, yr_idx, seq_len, device)
        diff_maps[yr_lbl] = _collect_output_diff_spatial(model, noise, cond_t, device)
    plot_output_diff_maps(diff_maps, lats, lons, out_dir)

    print(f"\n  [DONE] All model diagnostics written to: {out_dir}")


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
    p.add_argument("--norm_mode",    default="percentile",
                   choices=["percentile", "zscore"],
                   help="Normalisation mode: 'percentile' (p1/p99 of log1p) or "
                        "'zscore' (mean ± n_std of log1p). Default: percentile")
    p.add_argument("--n_std",        type=float, default=3.0,
                   help="Std devs used in zscore mode (default: 3.0)")
    p.add_argument("--out_dir",      default="./cond_diagnostics",
                   help="Output directory for figures")
    # ── Model-conditioning diagnostic (optional) ─────────────────────────────
    p.add_argument("--checkpoint",    default=None,
                   help="Path to a .pt checkpoint file (must contain 'EMA' key). "
                        "If provided, runs model-conditioning diagnostics.")
    p.add_argument("--model_config",  default="./configs/config_aero.yaml",
                   help="Path to the Hydra YAML config that defines cfg.model. "
                        "Required when --checkpoint is set. (default: ./configs/config_aero.yaml)")
    p.add_argument("--snap_years",   nargs="+", type=int, default=[1850, 2000, 2100],
                   help="Snapshot years to probe in model diagnostics. (default: 1850 2000 2100)")
    p.add_argument("--seq_len",      type=int, default=10,
                   help="Temporal length of the dummy noise tensor fed to the model. "
                        "Should match the seq_len used during training. (default: 10)")
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
    # Also collect normalised arrays for the model diagnostic below
    normed_arrays: Dict[str, np.ndarray] = {}
    lats_arr = lons_arr = None

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
            norm_mode   = args.norm_mode,
            n_std       = args.n_std,
        )

        # Stash normalised array for the model diagnostic
        da_norm = normalize_var(da_raw, norm_mode=args.norm_mode, n_std=args.n_std)
        normed_arrays[var_name] = da_norm.values.astype(np.float32)
        if lats_arr is None and "lat" in da_raw.coords:
            lats_arr = da_raw.lat.values
            lons_arr = da_raw.lon.values

    # ── Model-conditioning diagnostic (optional) ──────────────────────────────
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        elif not os.path.isfile(args.model_config):
            print(f"[ERROR] Model config not found: {args.model_config}")
        elif not normed_arrays:
            print("[WARN] No normalised arrays available – skipping model diagnostic.")
        else:
            diagnose_model_conditioning(
                checkpoint_path   = args.checkpoint,
                model_config_path = args.model_config,
                cond_np_dict      = normed_arrays,
                years             = all_years,
                lats              = lats_arr,
                lons              = lons_arr,
                out_dir           = args.out_dir,
                seq_len           = args.seq_len,
                snap_years        = args.snap_years,
            )

    print(f"\n{'='*60}")
    print(f"  All diagnostics saved to: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()