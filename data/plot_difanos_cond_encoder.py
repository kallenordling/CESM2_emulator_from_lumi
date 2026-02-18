"""
Conditioning Encoder Diagnostic
================================
Uses ClimateDataset for all data loading and normalization — including PCA
denoising if configured — so the diagnostic always reflects exactly what the
model receives.

Usage (data only — no model needed):
    python diagnose_cond_encoder.py \\
        --data_dir /path/to/data \\
        --cond_file emissions.nc \\
        --cond_vars CO2 SO2 \\
        --target_vars TREFHT \\
        --realizations r1i1p1f1 \\
        --data_only

Full diagnostic with trained model:
    python diagnose_cond_encoder.py \\
        --data_dir /path/to/data \\
        --cond_file emissions.nc \\
        --checkpoint /path/to/best_epoch_1700.pt \\
        --config_path ./configs/config_aero.yaml \\
        --n_components_cond 40

PCA options (both default to None = disabled):
    --n_components_target N    PCA components for climate target fields
    --n_components_cond   N    PCA components for CO2/SO2 conditioning maps
"""

import argparse
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import xarray as xr

# ─────────────────────────────────────────────
# Import from climate_dataset.py
# ─────────────────────────────────────────────
from climate_dataset import (
    ClimateDataset,
    scale_cumulative_linear,
    scale_emis_m1_p1_log10,
    scale_emis_0_1_log10,
    scale_quantile_transform,
    normalize,
)


# ─────────────────────────────────────────────
# Latitude-weighted area mean
# ─────────────────────────────────────────────

def lat_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Compute a latitude-weighted spatial mean, collapsing all spatial dims.

    Looks for a coordinate named 'lat' or 'latitude'. If neither is found,
    falls back to a simple arithmetic mean with a warning.

    Parameters
    ----------
    da : xr.DataArray
        Array with at least one spatial dimension.

    Returns
    -------
    xr.DataArray
        Array with spatial dims collapsed, weighted by cos(lat).
    """
    # Find the latitude coordinate name
    lat_name = None
    for candidate in ("lat", "latitude"):
        if candidate in da.coords:
            lat_name = candidate
            break

    if lat_name is None:
        spatial_dims = [d for d in da.dims if d != "year"]
        print(f"  [WARN] No lat/latitude coord found — falling back to unweighted mean")
        return da.mean(dim=spatial_dims)

    # Build cosine-of-latitude weights (shape matches the lat dim)
    weights = np.cos(np.deg2rad(da[lat_name])).clip(min=0)  # [0, 1], zero at poles
    weights = weights / weights.sum()                        # normalise

    # Weighted mean over all spatial dims; xarray broadcasts weights over non-lat dims
    spatial_dims = [d for d in da.dims if d != "year"]
    return da.weighted(weights).mean(dim=spatial_dims)


# ─────────────────────────────────────────────
# Additional alternative normalizations for comparison
# ─────────────────────────────────────────────

def scale_sqrt_m1_p1(da: xr.DataArray):
    """Alternative: sqrt compression then min-max to [-1, 1]"""
    x = da.clip(min=0)
    sx = np.sqrt(x)
    lo = float(sx.min(skipna=True))
    hi = float(sx.max(skipna=True))
    z01 = (sx - lo) / max(hi - lo, 1e-30)
    return (2.0 * z01 - 1.0).astype("float32")


def scale_linear_pctile_clip(da: xr.DataArray, lo_pct=1.0, hi_pct=99.0):
    """Linear normalization with percentile clipping to handle outliers."""
    lo = float(da.quantile(lo_pct / 100.0, skipna=True))
    hi = float(da.quantile(hi_pct / 100.0, skipna=True))
    clipped = da.clip(min=lo, max=hi)
    z01 = (clipped - lo) / max(hi - lo, 1e-30)
    return (2.0 * z01 - 1.0).astype("float32")


def scale_spatial_mean_linear(da: xr.DataArray):
    """
    First reduce to latitude-weighted spatial mean per year, then min-max to [-1, 1].
    Avoids ocean-zero domination by collapsing spatial dims first.
    Then broadcast back to full grid shape.
    """
    ts = lat_weighted_mean(da)  # [year]
    lo = float(ts.min(skipna=True))
    hi = float(ts.max(skipna=True))
    ts_normed = (2.0 * (ts - lo) / max(hi - lo, 1e-30) - 1.0)
    # Broadcast back to original shape (every grid cell gets the same value per year)
    return ts_normed.broadcast_like(da).astype("float32")


def scale_zscore(da: xr.DataArray) -> xr.DataArray:
    """
    Z-score normalization: (x - mean) / std computed over ALL values globally.

    Result is unbounded but typically falls within ±3 for well-behaved fields.
    Useful for revealing whether the encoder sees genuinely different signal
    magnitudes vs. just shifted baselines.
    """
    mu    = float(da.mean(skipna=True))
    sigma = float(da.std(skipna=True))
    return ((da - mu) / max(sigma, 1e-30)).astype("float32")


# ─────────────────────────────────────────────
# Hook-based encoder inspection
# ─────────────────────────────────────────────

class EncoderProbe:
    """Forward hooks on each layer of cond_encoder to capture activations."""

    def __init__(self, cond_encoder, cond_scale=None, cond_shift=None):
        self.activations = OrderedDict()
        self.hooks = []

        for i, layer in enumerate(cond_encoder):
            name = f"{i}_{layer.__class__.__name__}"
            hook = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

        if cond_scale is not None:
            self.hooks.append(
                cond_scale.register_forward_hook(self._make_hook("scale_output")))
        if cond_shift is not None:
            self.hooks.append(
                cond_shift.register_forward_hook(self._make_hook("shift_output")))

    def _make_hook(self, name):
        def hook_fn(module, inp, out):
            self.activations[name] = out.detach().cpu()
        return hook_fn

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_normalization_comparison(cond_ds, cond_vars, cached_normed, save_path):
    """
    Compare normalization strategies on the latitude-weighted spatial-mean time series.
    Uses pre-computed cached_normed[var][label] to avoid recomputation.
    """
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']

    n_vars = len(cond_vars)
    fig, axes = plt.subplots(n_vars, 2, figsize=(20, 6 * n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    for row, var in enumerate(cond_vars):
        da = cond_ds[var]
        years = da.year.values
        raw_ts = lat_weighted_mean(da)  # latitude-weighted

        # ── Left panel: all normalizations on spatial-mean time series ──
        ax = axes[row, 0]

        ax2 = ax.twinx()
        ax2.plot(years, raw_ts.values, color='black', linewidth=2.5,
                 alpha=0.3, linestyle='-', label='raw (right axis)')
        ax2.set_ylabel("Raw value", color='black', alpha=0.5)
        ax2.tick_params(axis='y', labelcolor='gray')

        for (label, normed), col in zip(cached_normed[var].items(), colors):
            if normed is None:
                continue
            ts = lat_weighted_mean(normed)  # latitude-weighted
            ax.plot(years, ts.values, color=col, linewidth=2, label=label)

        ax.set_ylim(-1.15, 1.15)
        ax.axhline(-1, color='gray', ls='--', alpha=0.4)
        ax.axhline(1, color='gray', ls='--', alpha=0.4)
        ax.set_title(f"{var} — Lat-weighted spatial mean after normalization", fontsize=13)
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalized value")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        # ── Right panel: future-period zoom ──
        ax = axes[row, 1]
        future_mask = years >= 2015

        ax2 = ax.twinx()
        ax2.plot(years[future_mask], raw_ts.values[future_mask],
                 color='black', linewidth=2.5, alpha=0.3, linestyle='-',
                 label='raw (right axis)')
        ax2.set_ylabel("Raw value", color='black', alpha=0.5)
        ax2.tick_params(axis='y', labelcolor='gray')

        for (label, normed), col in zip(cached_normed[var].items(), colors):
            if normed is None:
                continue
            ts = lat_weighted_mean(normed)  # latitude-weighted
            ax.plot(years[future_mask], ts.values[future_mask],
                    color=col, linewidth=2, label=label)
            fvals = ts.values[future_mask]
            rng = fvals.max() - fvals.min()
            ax.annotate(f"Δ={rng:.3f}",
                        xy=(2085, fvals[-1]),
                        fontsize=8, color=col)

        ax.set_title(f"{var} — Future period zoom (2015–2100)", fontsize=13)
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalized value")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_spatial_maps(dataset: ClimateDataset, save_path: str):
    """
    Show spatial maps at representative years across the full processing pipeline:
      Row 0 — Raw (straight from NetCDF, physical units)
      Row 1 — Normalized, pre-PCA  (dataset.dataset_cond)
      Row 2 — Model input, post-PCA (dataset.tensor_data_cond)   [only if PCA active]

    This makes it immediately obvious what information the encoder actually sees.
    """
    ds_raw = xr.open_dataset(os.path.join(dataset.data_dir, dataset.cond_file))
    ds_raw = ds_raw[dataset.cond_vars]

    all_years = dataset.dataset_cond.year.values
    candidate_years = [all_years[0], 2015, 2050, all_years[-1]]
    years_to_show = [y for y in candidate_years if y in all_years]
    year_indices  = [int(np.where(all_years == y)[0][0]) for y in years_to_show]

    pca_active = dataset._pca_cond is not None
    n_rows = 3 if pca_active else 2
    n_cols = len(years_to_show)

    # tensor_data_cond: (n_vars, T, H, W)
    cond_np = dataset.tensor_data_cond.numpy()

    for v_idx, var in enumerate(dataset.cond_vars):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 3.8 * n_rows))
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        row_labels = ["Raw (physical units)",
                      "Normalized — pre-PCA  (dataset.dataset_cond)"]
        if pca_active:
            n_comps = dataset._pca_cond[v_idx].n_components_
            var_pct = dataset._pca_cond[v_idx].explained_variance_ratio_.sum() * 100
            row_labels.append(
                f"Model input — post-PCA  ({n_comps} comps, {var_pct:.1f}% var)"
            )

        for col, (yr, t_idx) in enumerate(zip(years_to_show, year_indices)):
            # ── Row 0: raw ────────────────────────────────────────────────────
            raw_data = ds_raw[var].sel(year=yr).values if yr in ds_raw.year.values \
                       else np.full_like(cond_np[v_idx, t_idx], np.nan)
            ax = axes[0, col]
            vmin_r, vmax_r = np.nanmin(raw_data), np.nanmax(raw_data)
            im = ax.imshow(raw_data, aspect='auto', cmap='viridis',
                           vmin=vmin_r, vmax=vmax_r, origin='lower')
            ax.set_title(f"year={yr}\n"
                         f"min={vmin_r:.2e}  max={vmax_r:.2e}", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            if col == 0:
                ax.set_ylabel(row_labels[0], fontsize=9)

            # ── Row 1: normalized pre-PCA ─────────────────────────────────────
            norm_data = dataset.dataset_cond[var].sel(year=yr).values
            ax = axes[1, col]
            im = ax.imshow(norm_data, aspect='auto', cmap='RdBu_r',
                           vmin=-1, vmax=1, origin='lower')
            ax.set_title(f"min={norm_data.min():.3f}  max={norm_data.max():.3f}", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            if col == 0:
                ax.set_ylabel(row_labels[1], fontsize=9)

            # ── Row 2: model input post-PCA ───────────────────────────────────
            if pca_active:
                model_data = cond_np[v_idx, t_idx]
                ax = axes[2, col]
                im = ax.imshow(model_data, aspect='auto', cmap='RdBu_r',
                               vmin=-1, vmax=1, origin='lower')
                ax.set_title(f"min={model_data.min():.3f}  max={model_data.max():.3f}",
                             fontsize=8)
                plt.colorbar(im, ax=ax, shrink=0.8)
                if col == 0:
                    ax.set_ylabel(row_labels[2], fontsize=9)

        pca_tag = (f"  [PCA: {dataset._pca_cond[v_idx].n_components_} comps]"
                   if pca_active else "")
        fig.suptitle(f"{var} — Processing pipeline{pca_tag}", fontsize=14)
        plt.tight_layout()
        var_save = save_path.replace(".png", f"_{var}.png")
        plt.savefig(var_save, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {var_save}")
        plt.close()

    ds_raw.close()


def plot_spatial_maps_normalized(dataset: ClimateDataset, cached_normed: dict,
                                  save_path: str):
    """
    Show spatial maps AFTER each normalization method at representative years.
    The top row always shows "★ ClimateDataset model input" (post-PCA), so the
    reader can immediately compare every alternative to what the model actually sees.
    """
    all_years = dataset.dataset_cond.year.values
    candidate_years = [all_years[0], 2015, 2050, all_years[-1]]
    years_to_show  = [y for y in candidate_years if y in all_years]
    year_indices   = [int(np.where(all_years == y)[0][0]) for y in years_to_show]
    cond_np = dataset.tensor_data_cond.numpy()   # (n_vars, T, H, W)

    for v_idx, var in enumerate(dataset.cond_vars):
        methods = cached_normed[var]
        n_methods = len(methods) + 1              # +1 for the ClimateDataset row
        n_years   = len(years_to_show)

        fig, axes = plt.subplots(n_methods, n_years,
                                 figsize=(5 * n_years, 3.5 * n_methods))
        if n_methods == 1:
            axes = axes[np.newaxis, :]
        if n_years == 1:
            axes = axes[:, np.newaxis]

        pca_active = dataset._pca_cond is not None
        pca_tag = ""
        if pca_active:
            n_comps = dataset._pca_cond[v_idx].n_components_
            var_pct = dataset._pca_cond[v_idx].explained_variance_ratio_.sum() * 100
            pca_tag = f"  PCA {n_comps} comps ({var_pct:.1f}% var)"
        model_row_label = f"★ ClimateDataset model input{pca_tag}"

        # ── Row 0: ClimateDataset model input (post-PCA) ──────────────────────
        for col, (yr, t_idx) in enumerate(zip(years_to_show, year_indices)):
            ax = axes[0, col]
            data = cond_np[v_idx, t_idx]
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                           vmin=-1, vmax=1, origin='lower')
            ax.set_title(f"year={yr}\nmin={data.min():.3f}  max={data.max():.3f}",
                         fontsize=8)
            if col == 0:
                ax.set_ylabel(model_row_label, fontsize=9, color='darkgreen',
                              fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)

        # ── Rows 1+: alternative normalization methods ─────────────────────────
        for row_offset, (label, normed) in enumerate(methods.items()):
            row = row_offset + 1
            if normed is None:
                for col in range(n_years):
                    axes[row, col].text(0.5, 0.5, "Error",
                                        transform=axes[row, col].transAxes,
                                        ha='center', fontsize=8)
                    if col == 0:
                        axes[row, col].set_ylabel(label, fontsize=9)
                continue

            for col, yr in enumerate(years_to_show):
                ax = axes[row, col]
                data = normed.sel(year=yr).values
                im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                               vmin=-1, vmax=1, origin='lower')
                ax.set_title(f"year={yr}\nmin={data.min():.3f}  max={data.max():.3f}",
                             fontsize=8)
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)
                plt.colorbar(im, ax=ax, shrink=0.8)

        plt.suptitle(f"{var} — Normalised spatial maps  (all methods vs model input)",
                     fontsize=13)
        plt.tight_layout()
        var_save = save_path.replace(".png", f"_{var}.png")
        plt.savefig(var_save, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {var_save}")
        plt.close()


def plot_encoder_activations(activations, sample_labels, save_path):
    """Activation distributions at each encoder layer for different emission levels."""
    layer_names = list(activations.keys())
    n_layers = len(layer_names)
    n_samples = len(sample_labels)

    fig, axes = plt.subplots(n_layers, 1, figsize=(14, 3.5 * n_layers))
    if n_layers == 1:
        axes = [axes]

    colors = plt.cm.coolwarm(np.linspace(0, 1, n_samples))

    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        acts = activations[layer_name]

        for j, (act, label) in enumerate(zip(acts, sample_labels)):
            vals = act.flatten().numpy()
            ax.hist(vals, bins=80, alpha=0.5, label=label, color=colors[j],
                    density=True, histtype='stepfilled', linewidth=1.5)
            ax.axvline(vals.mean(), color=colors[j], ls='--', alpha=0.8)

        ax.set_title(f"Layer: {layer_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_scale_shift(scale_vals, shift_vals, emission_levels, save_path):
    """Scale/shift vector properties vs emission level."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Norms
    scale_norms = [s.norm().item() for s in scale_vals]
    shift_norms = [s.norm().item() for s in shift_vals]

    axes[0].plot(emission_levels, scale_norms, 'ro-', lw=2, ms=8)
    axes[0].set_title("||scale|| vs emission level")
    axes[0].set_xlabel("Normalized emission (lat-weighted spatial mean)")
    axes[0].set_ylabel("||scale||")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(emission_levels, shift_norms, 'bo-', lw=2, ms=8)
    axes[1].set_title("||shift|| vs emission level")
    axes[1].set_xlabel("Normalized emission (lat-weighted spatial mean)")
    axes[1].set_ylabel("||shift||")
    axes[1].grid(True, alpha=0.3)

    # Cosine similarity matrix
    n = len(scale_vals)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = torch.nn.functional.cosine_similarity(
                scale_vals[i].flatten().unsqueeze(0),
                scale_vals[j].flatten().unsqueeze(0)
            ).item()

    im = axes[2].imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_xticks(range(n))
    axes[2].set_yticks(range(n))
    labels = [f"{e:.2f}" for e in emission_levels]
    axes[2].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_title("Cosine sim of scale vectors\n(should vary, not all ~1)")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_embedding_pca(scale_vals, shift_vals, emission_levels, save_path):
    """PCA of [scale||shift] embeddings colored by emission level."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[SKIP] sklearn not installed, skipping PCA plot")
        return

    embeddings = np.stack([
        torch.cat([s.flatten(), sh.flatten()]).numpy()
        for s, sh in zip(scale_vals, shift_vals)
    ])

    if embeddings.shape[0] < 3:
        print("[SKIP] Need >= 3 samples for PCA")
        return

    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=emission_levels,
                    cmap='coolwarm', s=100, edgecolors='black', lw=0.5)
    plt.colorbar(sc, ax=ax, label="Emission level (lat-weighted spatial mean)")

    for i, lvl in enumerate(emission_levels):
        ax.annotate(f"  {lvl:.2f}", (proj[i, 0], proj[i, 1]), fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA of [scale || shift] — should separate by emission level")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Main routines
# ─────────────────────────────────────────────

def run_data_diagnostic(dataset: ClimateDataset, output_dir: str):
    """Normalization comparison — no model needed.

    Uses ``dataset.dataset_cond`` (the normalised pre-PCA xarray) for the
    multi-method comparison, and ``dataset.tensor_data_cond`` (post-PCA,
    exactly what the encoder sees) for distribution and spatial-pipeline plots.
    """
    # Open raw file for physical-unit stats and the raw spatial row.
    # Filter to the same year selection as dataset.dataset_cond so all
    # downstream .sel(year=...) calls are consistent.
    cond_file_path = os.path.join(dataset.data_dir, dataset.cond_file)
    selected_years = dataset.dataset_cond.year.values
    ds_raw = xr.open_dataset(cond_file_path)[dataset.cond_vars].sel(
        year=selected_years, method="nearest"
    )
    cond_vars = dataset.cond_vars

    # ── Print raw stats ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RAW DATA STATISTICS")
    print("=" * 60)
    for var in cond_vars:
        da = ds_raw[var]
        ts = lat_weighted_mean(da)
        vals = da.values.flatten()
        vals = vals[~np.isnan(vals)]

        print(f"\n{var}:")
        print(f"  Shape:               {da.shape}  dims={da.dims}")
        print(f"  Global min:          {vals.min():.6e}")
        print(f"  Global max:          {vals.max():.6e}")
        print(f"  Mean:                {vals.mean():.6e}")
        print(f"  Median:              {np.median(vals):.6e}")
        print(f"  % zeros:             {(vals == 0).sum() / len(vals) * 100:.1f}%")
        print(f"  % < 1e-10:           {(np.abs(vals) < 1e-10).sum() / len(vals) * 100:.1f}%")
        print(f"  Max/median:          {vals.max() / max(np.median(vals), 1e-30):.1f}x")
        print(f"  Year range:          {da.year.values[0]} -> {da.year.values[-1]}")
        print(f"  First year lat-wtd mean: {float(ts.isel(year=0)):.6e}")
        print(f"  Last  year lat-wtd mean: {float(ts.isel(year=-1)):.6e}")

    # ── ClimateDataset model-input stats (post-PCA) ───────────────────────────
    pca_active = dataset._pca_cond is not None
    cond_np = dataset.tensor_data_cond.numpy()   # (n_vars, T, H, W)
    all_years = dataset.dataset_cond.year.values

    print("\n" + "=" * 60)
    print("CLIMATADATASET MODEL-INPUT STATS (post-PCA)" if pca_active
          else "CLIMATADATASET NORMALISED STATS")
    print("=" * 60)
    for v_idx, var in enumerate(cond_vars):
        vals = cond_np[v_idx].ravel()
        ts   = cond_np[v_idx].mean(axis=(1, 2))
        pca_info = ""
        if pca_active:
            n_comps = dataset._pca_cond[v_idx].n_components_
            var_pct = dataset._pca_cond[v_idx].explained_variance_ratio_.sum() * 100
            pca_info = f"  PCA: {n_comps} comps  {var_pct:.2f}% var retained"
        print(f"\n{var}:{pca_info}")
        print(f"  min={vals.min():.4f}  max={vals.max():.4f}  "
              f"mean={vals.mean():.4f}  std={vals.std():.4f}")
        print(f"  temporal mean range: [{ts.min():.4f}, {ts.max():.4f}]")

    # ── Compute alternative normalizations on raw data for comparison ─────────
    # dataset.dataset_cond already holds normalize() output — add it as the
    # first (starred) entry so the comparison is centred on the actual method.
    method_fns = OrderedDict([
        ("QuantileTransformer",        scale_quantile_transform),
        ("log10+quantile",             scale_emis_m1_p1_log10),
        ("spatial-mean linear",        scale_cumulative_linear),
        ("sqrt + min-max",             scale_sqrt_m1_p1),
        ("linear pctile-clip (1-99%)", scale_linear_pctile_clip),
        ("spatial-mean-first",         scale_spatial_mean_linear),
        ("z-score (μ=0, σ=1)",         scale_zscore),
    ])

    # cached_normed[var][label] = xr.DataArray  (all applied to raw ds_raw)
    cached_normed = {}
    for var in cond_vars:
        da = ds_raw[var]
        cached_normed[var] = OrderedDict()

        # First entry: the actual ClimateDataset normalization (from xr_data)
        cached_normed[var]["★ normalize() [ClimateDataset — pre-PCA]"] = \
            dataset.dataset_cond[var]

        for label, fn in method_fns.items():
            print(f"  Computing {label} for {var}...", end=" ", flush=True)
            try:
                cached_normed[var][label] = fn(da)
                print("OK")
            except Exception as e:
                cached_normed[var][label] = None
                print(f"ERROR: {e}")

    # ── Print normalized stats ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NORMALIZATION COMPARISON (latitude-weighted spatial-mean time series)")
    print("=" * 60)
    for var in cond_vars:
        print(f"\n{var}:")
        for label, normed in cached_normed[var].items():
            if normed is None:
                print(f"  {label}: SKIPPED (error)")
                continue
            ts = lat_weighted_mean(normed)
            future = ts.sel(year=slice(2015, 2100))
            hist   = ts.sel(year=slice(1850, 2014))
            print(f"\n  {label}:")
            print(f"    Full:       [{float(ts.min()):.4f}, {float(ts.max()):.4f}]")
            print(f"    Historical: [{float(hist.min()):.4f}, {float(hist.max()):.4f}]  "
                  f"delta={float(hist.max()) - float(hist.min()):.4f}")
            print(f"    Future:     [{float(future.min()):.4f}, {float(future.max()):.4f}]  "
                  f"delta={float(future.max()) - float(future.min()):.4f}")
            print(f"    Future std: {float(future.std()):.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_normalization_comparison(
        ds_raw, cond_vars, cached_normed,
        os.path.join(output_dir, "diag_normalization.png"),
    )
    plot_spatial_maps(
        dataset,
        os.path.join(output_dir, "diag_spatial_maps.png"),
    )
    plot_spatial_maps_normalized(
        dataset, cached_normed,
        os.path.join(output_dir, "diag_spatial_maps_normalized.png"),
    )
    _plot_encoder_input_distributions(
        dataset,
        os.path.join(output_dir, "diag_encoder_input_distributions.png"),
    )

    ds_raw.close()


def _plot_encoder_input_distributions(dataset: ClimateDataset, save_path: str):
    """
    Plot the distribution of the values ACTUALLY fed into the encoder:
    dataset.tensor_data_cond (post-PCA if enabled).

    One row per conditioning variable; three columns:
      [density histogram | CDF | lat-weighted mean vs year]

    A second dashed overlay shows the pre-PCA normalised values from
    dataset.dataset_cond so the effect of PCA denoising is visible.
    """
    cond_vars = dataset.cond_vars
    all_years = dataset.dataset_cond.year.values
    pca_active = dataset._pca_cond is not None
    cond_np = dataset.tensor_data_cond.numpy()   # (n_vars, T, H, W)

    n_vars = len(cond_vars)
    fig, axes = plt.subplots(n_vars, 3, figsize=(18, 5 * n_vars), squeeze=False)
    pca_tag = ""
    if pca_active:
        pca_tag = " + PCA denoising"
    fig.suptitle(
        f"Encoder input distributions — ClimateDataset.tensor_data_cond"
        f"  (normalize(){pca_tag})\nall years × all grid points",
        fontsize=14, fontweight="bold",
    )

    for row, var in enumerate(cond_vars):
        # ── Post-PCA values (what the encoder actually receives) ──────────────
        vals = cond_np[row].ravel().astype(np.float64)
        vals = vals[~np.isnan(vals)]

        # ── Pre-PCA values (normalized xarray, before PCA) ───────────────────
        vals_prenorm = dataset.dataset_cond[var].values.ravel().astype(np.float64)
        vals_prenorm = vals_prenorm[~np.isnan(vals_prenorm)]

        n       = len(vals)
        vmean   = vals.mean()
        vmedian = np.median(vals)
        vstd    = vals.std()
        vmin, vmax = vals.min(), vals.max()
        p01  = np.percentile(vals, 1)
        p99  = np.percentile(vals, 99)
        zeros_pct = 100.0 * (vals == 0).sum() / max(n, 1)

        stat_str = (
            f"n={n:,}\n"
            f"min={vmin:.3e}  max={vmax:.3e}\n"
            f"mean={vmean:.3e}  std={vstd:.3e}\n"
            f"p01={p01:.3e}  p99={p99:.3e}\n"
            f"zeros={zeros_pct:.1f}%"
        )

        # ── col 0: density histogram ──────────────────────────────────────────
        ax0 = axes[row, 0]
        ax0.hist(vals, bins=100, color="royalblue", edgecolor="none",
                 alpha=0.75, density=True, histtype='stepfilled',
                 label="model input (post-PCA)" if pca_active else "model input")
        if pca_active:
            ax0.hist(vals_prenorm, bins=100, color="orange", edgecolor="none",
                     alpha=0.45, density=True, histtype='stepfilled',
                     label="pre-PCA normalised")
        ax0.axvline(vmean,   color="red",    ls="--", lw=1.5, label="mean")
        ax0.axvline(vmedian, color="orange", ls="--", lw=1.5, label="median")
        ax0.legend(fontsize=8)
        ax0.text(0.02, 0.97, stat_str, transform=ax0.transAxes,
                 fontsize=7, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax0.set_title(f"{var}  —  encoder input distribution", fontsize=11)
        ax0.set_xlabel(f"{var} (normalised)")
        ax0.set_ylabel("Density")
        ax0.grid(True, alpha=0.3)

        # ── col 1: CDF ────────────────────────────────────────────────────────
        ax1 = axes[row, 1]
        sv  = np.sort(vals)
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax1.plot(sv, cdf, lw=1.5, color="royalblue",
                 label="model input (post-PCA)" if pca_active else "model input")
        if pca_active:
            sv_pre = np.sort(vals_prenorm)
            cdf_pre = np.arange(1, len(sv_pre) + 1) / len(sv_pre)
            ax1.plot(sv_pre, cdf_pre, lw=1.5, color="orange", ls="--",
                     label="pre-PCA normalised")
        for pct, col in [(0.05, "orange"), (0.50, "red"), (0.95, "orange")]:
            i = min(np.searchsorted(cdf, pct), len(sv) - 1)
            ax1.axvline(sv[i], color=col, ls=":", lw=1.2,
                        label=f"p{int(pct*100)}={sv[i]:.3f}")
        ax1.legend(fontsize=7)
        ax1.set_title("CDF")
        ax1.set_xlabel(f"{var} (normalised)")
        ax1.set_ylabel("Cumulative probability")
        ax1.grid(True, alpha=0.3)

        # ── col 2: spatial mean vs year ───────────────────────────────────────
        ax2 = axes[row, 2]
        # Post-PCA: mean over H, W dims
        ts_post = cond_np[row].mean(axis=(1, 2))      # (T,)
        ax2.plot(all_years, ts_post, lw=1.8, color="royalblue",
                 label="model input (post-PCA)" if pca_active else "model input")
        if pca_active:
            # Pre-PCA spatial mean
            ts_pre = dataset.dataset_cond[var].mean(
                dim=[d for d in dataset.dataset_cond[var].dims if d != "year"]
            ).values
            ax2.plot(all_years, ts_pre, lw=1.5, color="orange", ls="--",
                     label="pre-PCA normalised")
        ax2.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax2.set_title("Spatial mean vs year")
        ax2.set_xlabel("Year")
        ax2.set_ylabel(f"{var} (normalised)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {save_path}")
    plt.close()


def run_encoder_diagnostic(checkpoint_path: str, dataset: ClimateDataset,
                            config_path: str, output_dir: str, device: str = "cpu"):
    """Full diagnostic: data + encoder layer activations + scale/shift.

    Samples conditioning tensors directly from ``dataset.tensor_data_cond``
    (the same post-PCA data the model sees during training/generation) rather
    than re-normalising from the raw file.
    """
    # First run data diagnostic using the same dataset
    run_data_diagnostic(dataset, output_dir)

    # Load model
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    print(f"\n[INFO] Loading config: {config_path}")
    conf = OmegaConf.load(config_path)
    model = instantiate(conf.model)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    chkpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "EMA" in chkpt:
        model.load_state_dict(chkpt["EMA"], strict=False)
        print("[INFO] Loaded EMA weights")
    elif "Unet" in chkpt:
        model.load_state_dict(chkpt["Unet"], strict=False)
        print("[INFO] Loaded Unet weights")

    # Restore PCA state from checkpoint if present
    if "PCA" in chkpt and chkpt["PCA"] is not None:
        dataset.set_pca_state(chkpt["PCA"])
        print("[INFO] Restored PCA state from checkpoint")

    model.eval().to(device)

    if model.cond_encoder is None:
        print("[ERROR] Model has no cond_encoder!")
        return

    # Hook up the encoder
    probe = EncoderProbe(model.cond_encoder, model.cond_scale, model.cond_shift)

    # ── Plot the actual normalised/PCA values fed into the encoder ────────────
    print("[INFO] Plotting encoder-input distributions...")
    _plot_encoder_input_distributions(
        dataset,
        os.path.join(output_dir, "diag_encoder_input_distributions.png"),
    )

    # ── Sample ~10 representative years from tensor_data_cond ─────────────────
    # tensor_data_cond: (n_vars, T, H, W) — this IS what the encoder receives
    all_years = dataset.dataset_cond.year.values
    cond_np   = dataset.tensor_data_cond.numpy()      # (n_vars, T, H, W)
    n_vars, T, H, W = cond_np.shape

    t_indices = np.linspace(0, T - 1, 10, dtype=int)
    sample_years = all_years[t_indices]
    print(f"[INFO] Sampling years: {sample_years}")

    all_layer_acts = OrderedDict()
    scale_vals, shift_vals = [], []
    emission_levels, sample_labels = [], []

    for t_idx, year in zip(t_indices, sample_years):
        # Extract single time step: (n_vars, 1, H, W) → unsqueeze → (1, n_vars, 1, H, W)
        frame = cond_np[:, t_idx: t_idx + 1, :, :]            # (n_vars, 1, H, W)
        cond_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device)

        # Use spatial mean of first cond_var as emission-level label
        emis_level = float(cond_np[0, t_idx].mean())
        emission_levels.append(emis_level)
        sample_labels.append(f"Year {year}\n(mean={emis_level:.3f})")

        probe.clear()
        with torch.no_grad():
            cond_feat = model.cond_encoder(cond_tensor)
            cond_feat_flat = cond_feat.view(cond_feat.shape[0], -1)
            scale = model.cond_scale(cond_feat_flat)
            shift = model.cond_shift(cond_feat_flat)

        scale_vals.append(scale.cpu())
        shift_vals.append(shift.cpu())

        for layer_name, act in probe.activations.items():
            if layer_name not in all_layer_acts:
                all_layer_acts[layer_name] = []
            all_layer_acts[layer_name].append(act)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_encoder_activations(all_layer_acts, sample_labels,
                             os.path.join(output_dir, "diag_encoder_layers.png"))
    plot_scale_shift(scale_vals, shift_vals, emission_levels,
                     os.path.join(output_dir, "diag_scale_shift.png"))
    plot_embedding_pca(scale_vals, shift_vals, emission_levels,
                       os.path.join(output_dir, "diag_embedding_pca.png"))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENCODER DIAGNOSTIC SUMMARY")
    print("=" * 60)

    print(f"\nScale vectors across emission levels:")
    for lvl, s in zip(emission_levels, scale_vals):
        print(f"  emis={lvl:+.3f}  ||scale||={s.norm().item():.4f}  "
              f"mean={s.mean().item():.4f}  std={s.std().item():.4f}")

    print(f"\nShift vectors across emission levels:")
    for lvl, s in zip(emission_levels, shift_vals):
        print(f"  emis={lvl:+.3f}  ||shift||={s.norm().item():.4f}  "
              f"mean={s.mean().item():.4f}  std={s.std().item():.4f}")

    scale_stacked = torch.stack(scale_vals)
    shift_stacked = torch.stack(shift_vals)
    scale_var = scale_stacked.std(dim=0).mean().item()
    shift_var = shift_stacked.std(dim=0).mean().item()

    print(f"\nCross-sample variation (std across emission levels):")
    print(f"  Scale: {scale_var:.6f}")
    print(f"  Shift: {shift_var:.6f}")

    if scale_var < 0.01 and shift_var < 0.01:
        print("\n  WARNING: VERY LOW — encoder produces nearly identical embeddings!")
        print("     -> GroupNorm is likely destroying the signal.")
        print("     -> Try removing GroupNorm from cond_encoder.")
    elif scale_var < 0.05:
        print("\n  WARNING: LOW — encoder barely distinguishes emission levels.")
    else:
        print("\n  OK: Reasonable variation. Encoder does respond to emissions.")

    probe.remove_hooks()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose conditioning encoder — reads dataset params from config_aero.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config (primary source of dataset params) ─────────────────────────────
    parser.add_argument("--config_path", type=str, default="./configs/config_aero.yaml",
                        help="Path to config_aero.yaml (provides data_dir, cond_file, "
                             "variables, dataset._target_, etc.)")

    # ── Optional overrides (take priority over config when supplied) ──────────
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override config.data_dir")
    parser.add_argument("--cond_file", type=str, default=None,
                        help="Override config.cond_file (filename relative to data_dir)")
    parser.add_argument("--cond_vars", nargs="+", default=None,
                        help="Override conditioning variables, e.g. CO2 SO2")
    parser.add_argument("--target_vars", nargs="+", default=None,
                        help="Override target variables, e.g. TREFHT")
    parser.add_argument("--realizations", nargs="+", default=None,
                        help="Override realization(s) to load")

    # ── PCA options ───────────────────────────────────────────────────────────
    parser.add_argument("--n_components_target", type=int, default=None,
                        help="PCA components for target fields (None = disabled)")
    parser.add_argument("--n_components_cond", type=int, default=None,
                        help="PCA components for conditioning fields (None = disabled)")

    # ── Model / run options ───────────────────────────────────────────────────
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt) — omit for data-only mode")
    parser.add_argument("--output_dir", type=str, default="./diagnostics")
    parser.add_argument("--data_only", action="store_true",
                        help="Run data diagnostics only (no model or target files needed)")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load config_aero.yaml ─────────────────────────────────────────────────
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    print(f"[INFO] Loading config: {args.config_path}")
    cfg = OmegaConf.load(args.config_path)

    # Dataset lives at cfg.data.train
    ds_cfg = cfg.data.train

    # ── Resolve each param: config value is default, CLI flag overrides ───────
    data_dir     = args.data_dir     or str(ds_cfg.data_dir)
    cond_file    = args.cond_file    or str(ds_cfg.cond_file)
    cond_vars    = args.cond_vars    or list(OmegaConf.to_object(ds_cfg.cond_vars))
    target_vars  = args.target_vars  or list(OmegaConf.to_object(ds_cfg.target_vars))
    # Use only the first realization for diagnostics to keep it fast
    realizations = (args.realizations
                    or [list(OmegaConf.to_object(ds_cfg.realizations))[0]])

    # PCA: config values are the default; CLI can override to a different N or
    # pass 0 to explicitly disable (0 is treated as None = disabled).
    def _pca(cli_val, cfg_val):
        if cli_val is not None:
            return cli_val if cli_val > 0 else None
        return int(cfg_val) if cfg_val else None

    n_components_target = _pca(args.n_components_target,
                                ds_cfg.get("n_components_target"))
    n_components_cond   = _pca(args.n_components_cond,
                                ds_cfg.get("n_components_cond"))

    print(f"[INFO] data_dir            = {data_dir}")
    print(f"[INFO] cond_file           = {cond_file}")
    print(f"[INFO] cond_vars           = {cond_vars}")
    print(f"[INFO] target_vars         = {target_vars}")
    print(f"[INFO] realizations        = {realizations}")
    print(f"[INFO] n_components_target = {n_components_target}")
    print(f"[INFO] n_components_cond   = {n_components_cond}")

    # ── Build ClimateDataset via hydra instantiate (mirrors generate_ssp370.py) ─
    # cond_only=True skips loading target realization NetCDFs, which is all we
    # need for --data_only mode and avoids needing every realization on disk.
    cond_only = args.data_only or (args.checkpoint is None)
    print(f"[INFO] Building ClimateDataset via hydra instantiate "
          f"(cond_only={cond_only})...")

    dataset = instantiate(
        ds_cfg,                        # _target_ + fixed kwargs from config
        data_dir=data_dir,
        realizations=realizations,
        target_vars=target_vars,
        cond_vars=cond_vars,
        cond_file=cond_file,
        n_components_target=n_components_target,
        n_components_cond=n_components_cond,
    )

    print(f"[INFO] Dataset loaded — cond tensor shape: "
          f"{tuple(dataset.tensor_data_cond.shape)}  (n_vars, T, H, W)")
    if dataset._pca_cond is not None:
        for v_idx, var in enumerate(dataset.cond_vars):
            pca = dataset._pca_cond[v_idx]
            print(f"[INFO] PCA cond/{var}: {pca.n_components_} comps, "
                  f"{pca.explained_variance_ratio_.sum()*100:.2f}% var")

    # ── Run diagnostics ───────────────────────────────────────────────────────
    if args.data_only or args.checkpoint is None:
        run_data_diagnostic(dataset, args.output_dir)
    else:
        run_encoder_diagnostic(
            args.checkpoint, dataset, args.config_path,
            args.output_dir, args.device,
        )

    print(f"\n[DONE] Diagnostics saved to: {args.output_dir}")