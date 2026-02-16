"""
Conditioning Encoder Diagnostic
================================
Uses the ACTUAL normalization from climate_dataset.py, plus alternatives.

Usage (data only — no model needed):
    python diagnose_cond_encoder.py --cond_file /path/to/cond.nc --data_only

Full diagnostic with trained model:
    python diagnose_cond_encoder.py \
        --checkpoint /path/to/best_epoch_1700.pt \
        --cond_file /path/to/cond.nc \
        --config_path ./configs/config_aero.yaml
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
# Import normalizations from YOUR climate_dataset.py
# ─────────────────────────────────────────────
from climate_dataset import (
    scale_cumulative_linear,    # current active method
    scale_emis_m1_p1_log10,     # previous log10+quantile method
    scale_emis_0_1_log10,
    normalize,                  # the main dispatch function
)


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
    First reduce to spatial mean per year, then min-max to [-1, 1].
    Avoids ocean-zero domination by collapsing spatial dims first.
    Then broadcast back to full grid shape.
    """
    spatial_dims = [d for d in da.dims if d != "year"]
    ts = da.mean(dim=spatial_dims)  # [year]
    lo = float(ts.min(skipna=True))
    hi = float(ts.max(skipna=True))
    ts_normed = (2.0 * (ts - lo) / max(hi - lo, 1e-30) - 1.0)
    # Broadcast back to original shape (every grid cell gets the same value per year)
    return ts_normed.broadcast_like(da).astype("float32")


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

def plot_normalization_comparison(cond_ds, cond_vars, save_path):
    """
    Compare normalization strategies on the spatial-mean time series.
    Shows how much temporal dynamic range each approach preserves.
    """
    methods = OrderedDict([
        ("current: linear min-max",       lambda da: scale_cumulative_linear(da)),
        ("previous: log10+quantile",      lambda da: scale_emis_m1_p1_log10(da)),
        ("sqrt + min-max",                lambda da: scale_sqrt_m1_p1(da)),
        ("linear pctile-clip (1-99%)",    lambda da: scale_linear_pctile_clip(da)),
        ("spatial-mean-first, then linear", lambda da: scale_spatial_mean_linear(da)),
    ])
    colors = ['red', 'orange', 'green', 'blue', 'purple']

    n_vars = len(cond_vars)
    fig, axes = plt.subplots(n_vars, 2, figsize=(20, 6 * n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    for row, var in enumerate(cond_vars):
        da = cond_ds[var]
        spatial_dims = [d for d in da.dims if d != "year"]
        years = da.year.values

        # ── Left panel: all normalizations on spatial-mean time series ──
        ax = axes[row, 0]
        for (label, fn), col in zip(methods.items(), colors):
            try:
                normed = fn(da)
                ts = normed.mean(dim=spatial_dims)
                ax.plot(years, ts.values, color=col, linewidth=2, label=label)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes)

        ax.set_ylim(-1.15, 1.15)
        ax.axhline(-1, color='gray', ls='--', alpha=0.4)
        ax.axhline(1, color='gray', ls='--', alpha=0.4)
        ax.set_title(f"{var} — Spatial mean after normalization", fontsize=13)
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalized value")
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        # ── Right panel: future-period zoom ──
        ax = axes[row, 1]
        for (label, fn), col in zip(methods.items(), colors):
            try:
                normed = fn(da)
                ts = normed.mean(dim=spatial_dims)
                future_mask = years >= 2015
                ax.plot(years[future_mask], ts.values[future_mask],
                        color=col, linewidth=2, label=label)
                # Annotate the range
                fvals = ts.values[future_mask]
                rng = fvals.max() - fvals.min()
                ax.annotate(f"Δ={rng:.3f}",
                            xy=(2085, fvals[-1]),
                            fontsize=8, color=col)
            except Exception:
                pass

        ax.set_title(f"{var} — Future period zoom (2015–2100)", fontsize=13)
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalized value")
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_spatial_maps(cond_ds, cond_vars, save_path):
    """
    Show spatial maps at early / late years for each var.
    Reveals the ocean-zero problem and spatial structure.
    """
    years_to_show = [cond_ds.year.values[0], 2015, 2050, cond_ds.year.values[-1]]
    years_to_show = [y for y in years_to_show if y in cond_ds.year.values]

    fig, axes = plt.subplots(len(cond_vars), len(years_to_show),
                              figsize=(5 * len(years_to_show), 4 * len(cond_vars)))
    if len(cond_vars) == 1:
        axes = axes[np.newaxis, :]
    if len(years_to_show) == 1:
        axes = axes[:, np.newaxis]

    for row, var in enumerate(cond_vars):
        da = cond_ds[var]
        vmin = float(da.min(skipna=True))
        vmax = float(da.max(skipna=True))

        for col, yr in enumerate(years_to_show):
            ax = axes[row, col]
            data = da.sel(year=yr).values
            im = ax.imshow(data, aspect='auto', cmap='viridis',
                           vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(f"{var} year={yr}\nmin={data.min():.2e} max={data.max():.2e}",
                         fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Raw spatial maps — check for ocean zeros / outlier cells", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
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
    axes[0].set_xlabel("Normalized emission (spatial mean)")
    axes[0].set_ylabel("||scale||")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(emission_levels, shift_norms, 'bo-', lw=2, ms=8)
    axes[1].set_title("||shift|| vs emission level")
    axes[1].set_xlabel("Normalized emission (spatial mean)")
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
    plt.colorbar(sc, ax=ax, label="Emission level (norm. spatial mean)")

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

def run_data_diagnostic(cond_file, cond_vars, output_dir):
    """Normalization comparison — no model needed."""
    ds = xr.open_dataset(cond_file)
    ds = ds[cond_vars]

    # ── Print raw stats ──
    print("\n" + "=" * 60)
    print("RAW DATA STATISTICS")
    print("=" * 60)
    for var in cond_vars:
        da = ds[var]
        spatial_dims = [d for d in da.dims if d != "year"]
        ts = da.mean(dim=spatial_dims)
        vals = da.values.flatten()
        vals = vals[~np.isnan(vals)]

        print(f"\n{var}:")
        print(f"  Shape:       {da.shape}  dims={da.dims}")
        print(f"  Global min:  {vals.min():.6e}")
        print(f"  Global max:  {vals.max():.6e}")
        print(f"  Mean:        {vals.mean():.6e}")
        print(f"  Median:      {np.median(vals):.6e}")
        print(f"  % zeros:     {(vals == 0).sum() / len(vals) * 100:.1f}%")
        print(f"  % < 1e-10:   {(np.abs(vals) < 1e-10).sum() / len(vals) * 100:.1f}%")
        print(f"  Max/median:  {vals.max() / max(np.median(vals), 1e-30):.1f}x")
        print(f"  Year range:  {da.year.values[0]} -> {da.year.values[-1]}")
        print(f"  First year spatial mean: {float(ts.isel(year=0)):.6e}")
        print(f"  Last  year spatial mean: {float(ts.isel(year=-1)):.6e}")

    # ── Print normalized stats for each method ──
    methods = OrderedDict([
        ("current: scale_cumulative_linear", scale_cumulative_linear),
        ("previous: scale_emis_m1_p1_log10", scale_emis_m1_p1_log10),
        ("sqrt + min-max",                   scale_sqrt_m1_p1),
        ("linear pctile-clip (1-99%)",       scale_linear_pctile_clip),
        ("spatial-mean-first",               scale_spatial_mean_linear),
    ])

    print("\n" + "=" * 60)
    print("NORMALIZATION COMPARISON (spatial-mean time series)")
    print("=" * 60)
    for var in cond_vars:
        da = ds[var]
        spatial_dims = [d for d in da.dims if d != "year"]
        print(f"\n{var}:")
        for label, fn in methods.items():
            try:
                normed = fn(da)
                ts = normed.mean(dim=spatial_dims)
                future = ts.sel(year=slice(2015, 2100))
                hist = ts.sel(year=slice(1850, 2014))
                print(f"\n  {label}:")
                print(f"    Full:       [{float(ts.min()):.4f}, {float(ts.max()):.4f}]")
                print(f"    Historical: [{float(hist.min()):.4f}, {float(hist.max()):.4f}]  "
                      f"delta={float(hist.max()) - float(hist.min()):.4f}")
                print(f"    Future:     [{float(future.min()):.4f}, {float(future.max()):.4f}]  "
                      f"delta={float(future.max()) - float(future.min()):.4f}")
                print(f"    Future std: {float(future.std()):.4f}")
            except Exception as e:
                print(f"  {label}: ERROR -- {e}")

    # ── Plots ──
    plot_normalization_comparison(ds, cond_vars,
                                  os.path.join(output_dir, "diag_normalization.png"))
    plot_spatial_maps(ds, cond_vars,
                      os.path.join(output_dir, "diag_spatial_maps.png"))


def run_encoder_diagnostic(checkpoint_path, cond_file, config_path,
                            cond_vars, output_dir, device="cpu"):
    """Full diagnostic: data + encoder layer activations + scale/shift."""

    # First run data diagnostic
    run_data_diagnostic(cond_file, cond_vars, output_dir)

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

    model.eval().to(device)

    if model.cond_encoder is None:
        print("[ERROR] Model has no cond_encoder!")
        return

    # Hook up the encoder
    probe = EncoderProbe(model.cond_encoder, model.cond_scale, model.cond_shift)

    # Load & normalize conditioning data using YOUR normalize()
    ds_raw = xr.open_dataset(cond_file)[cond_vars]
    ds_normed = ds_raw.map(normalize)

    # Pick ~10 representative years
    all_years = ds_normed.year.values
    idx = np.linspace(0, len(all_years) - 1, 10, dtype=int)
    sample_years = all_years[idx]
    print(f"[INFO] Sampling years: {sample_years}")

    all_layer_acts = OrderedDict()
    scale_vals, shift_vals = [], []
    emission_levels, sample_labels = [], []

    for year in sample_years:
        year_ds = ds_normed.sel(year=[year])

        # Stack cond_vars into [C, T, H, W] -> [1, C, T, H, W]
        arrays = [year_ds[v].values for v in cond_vars]
        stacked = np.stack(arrays, axis=0)  # [C, T, H, W] or [C, H, W]
        if stacked.ndim == 3:
            stacked = stacked[:, np.newaxis, :, :]  # add T dim
        cond_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)

        emis_level = float(cond_tensor.mean())
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

    # ── Plots ──
    plot_encoder_activations(all_layer_acts, sample_labels,
                              os.path.join(output_dir, "diag_encoder_layers.png"))
    plot_scale_shift(scale_vals, shift_vals, emission_levels,
                      os.path.join(output_dir, "diag_scale_shift.png"))
    plot_embedding_pca(scale_vals, shift_vals, emission_levels,
                        os.path.join(output_dir, "diag_embedding_pca.png"))

    # ── Summary ──
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose conditioning encoder")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--cond_file", type=str, required=True,
                        help="Path to conditioning NetCDF file")
    parser.add_argument("--config_path", type=str, default="./configs/config_aero.yaml")
    parser.add_argument("--cond_vars", nargs="+", default=["CO2", "SO2"])
    parser.add_argument("--output_dir", type=str, default="./diagnostics")
    parser.add_argument("--data_only", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_only or args.checkpoint is None:
        run_data_diagnostic(args.cond_file, args.cond_vars, args.output_dir)
    else:
        run_encoder_diagnostic(
            args.checkpoint, args.cond_file, args.config_path,
            args.cond_vars, args.output_dir, args.device,
        )

    print(f"\n[DONE] Diagnostics saved to: {args.output_dir}")