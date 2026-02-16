"""
Conditioning Encoder Diagnostic
================================
Visualizes the conditioning signal at every stage:
  1. Raw cumulative emissions (before any normalization)
  2. After your log10+quantile normalization (what enters the model)
  3. After each layer inside cond_encoder (Conv3d, GroupNorm, SiLU, Pool)
  4. Final scale & shift vectors that modulate the time embedding

This shows exactly where the dynamic range gets destroyed.

Usage:
    python diagnose_cond_encoder.py \
        --checkpoint /path/to/best_epoch_1700.pt \
        --cond_file /path/to/emissions_cond.nc \
        --config_path ./configs/config_aero.yaml

Or without a checkpoint (just checks the data normalization stages):
    python diagnose_cond_encoder.py \
        --cond_file /path/to/emissions_cond.nc \
        --data_only
"""

import argparse
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import xarray as xr

# ─────────────────────────────────────────────
# 1. REPLICATE YOUR NORMALIZATION FUNCTIONS
#    (copied from climate_dataset.py so this
#     script is self-contained)
# ─────────────────────────────────────────────

def scale_emis_0_1_log10(da, low_pct=1.0, high_pct=99.5, floor=1e-30):
    x = da.clip(min=0)
    x = xr.where(x > 0, x, floor)
    lx = np.log10(x)
    lo = float(lx.quantile(low_pct / 100.0, skipna=True))
    hi = float(lx.quantile(high_pct / 100.0, skipna=True))
    z = (lx - lo) / (hi - lo)
    return z.clip(0, 1).fillna(0).astype("float32"), lo, hi


def scale_emis_m1_p1_log10(da, low_pct=1.0, high_pct=99.5, floor=1e-30):
    z01, lo, hi = scale_emis_0_1_log10(da, low_pct, high_pct, floor)
    return (2.0 * z01 - 1.0).astype("float32"), lo, hi


def scale_linear_m1_p1(da):
    """Alternative: simple min-max to [-1, 1]"""
    lo = float(da.min(skipna=True))
    hi = float(da.max(skipna=True))
    z01 = (da - lo) / (hi - lo)
    return (2.0 * z01 - 1.0).astype("float32"), lo, hi


def scale_sqrt_m1_p1(da):
    """Alternative: sqrt compression then min-max to [-1, 1]"""
    x = da.clip(min=0)
    sx = np.sqrt(x)
    lo = float(sx.min(skipna=True))
    hi = float(sx.max(skipna=True))
    z01 = (sx - lo) / (hi - lo)
    return (2.0 * z01 - 1.0).astype("float32"), lo, hi


# ─────────────────────────────────────────────
# 2. HOOK-BASED ENCODER INSPECTION
# ─────────────────────────────────────────────

class EncoderProbe:
    """Registers forward hooks on each layer of the cond_encoder
    to capture intermediate activations."""

    def __init__(self, cond_encoder: nn.Sequential, cond_scale: nn.Module, cond_shift: nn.Module):
        self.activations = OrderedDict()
        self.hooks = []

        # Hook every sub-module in the encoder
        for i, layer in enumerate(cond_encoder):
            name = f"{i}_{layer.__class__.__name__}"
            hook = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

        # Also hook scale and shift outputs
        if cond_scale is not None:
            hook = cond_scale.register_forward_hook(self._make_hook("scale_output"))
            self.hooks.append(hook)
        if cond_shift is not None:
            hook = cond_shift.register_forward_hook(self._make_hook("shift_output"))
            self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook_fn

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ─────────────────────────────────────────────
# 3. PLOTTING FUNCTIONS
# ─────────────────────────────────────────────

def plot_normalization_comparison(cond_ds, cond_vars, save_path="diag_normalization.png"):
    """
    Plot 1: Compare different normalization strategies side by side.
    Shows how much dynamic range each approach preserves.
    """
    fig, axes = plt.subplots(len(cond_vars), 4, figsize=(22, 5 * len(cond_vars)))
    if len(cond_vars) == 1:
        axes = axes[np.newaxis, :]

    for row, var in enumerate(cond_vars):
        da = cond_ds[var]

        # Compute the spatial mean per year for a clean time-series view
        raw_ts = da.mean(dim=[d for d in da.dims if d != "year"])

        # Current normalization: log10 + quantile
        norm_log, lo_log, hi_log = scale_emis_m1_p1_log10(da)
        norm_log_ts = norm_log.mean(dim=[d for d in norm_log.dims if d != "year"])

        # Alternative 1: linear min-max
        norm_lin, lo_lin, hi_lin = scale_linear_m1_p1(da)
        norm_lin_ts = norm_lin.mean(dim=[d for d in norm_lin.dims if d != "year"])

        # Alternative 2: sqrt + min-max
        norm_sqrt, lo_sq, hi_sq = scale_sqrt_m1_p1(da)
        norm_sqrt_ts = norm_sqrt.mean(dim=[d for d in norm_sqrt.dims if d != "year"])

        years = da.year.values

        # Panel 1: Raw values
        ax = axes[row, 0]
        ax.plot(years, raw_ts.values, 'k-', linewidth=2)
        ax.set_title(f"{var} — Raw cumulative emissions")
        ax.set_ylabel("Raw value")
        ax.grid(True, alpha=0.3)

        # Panel 2: Current log10 normalization
        ax = axes[row, 1]
        ax.plot(years, norm_log_ts.values, 'r-', linewidth=2)
        ax.set_title(f"{var} — Current: log10+quantile\n"
                     f"Range: [{norm_log_ts.min().values:.3f}, {norm_log_ts.max().values:.3f}]")
        ax.set_ylabel("Normalized value")
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Highlight the "compressed" region in the future
        future_mask = years >= 2015
        if future_mask.any():
            future_vals = norm_log_ts.values[future_mask]
            ax.annotate(f"Future range:\n{future_vals.min():.3f} to {future_vals.max():.3f}",
                        xy=(2060, future_vals.mean()), fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 3: Linear normalization
        ax = axes[row, 2]
        ax.plot(years, norm_lin_ts.values, 'b-', linewidth=2)
        ax.set_title(f"{var} — Alternative: linear min-max\n"
                     f"Range: [{norm_lin_ts.min().values:.3f}, {norm_lin_ts.max().values:.3f}]")
        ax.set_ylabel("Normalized value")
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        if future_mask.any():
            future_vals = norm_lin_ts.values[future_mask]
            ax.annotate(f"Future range:\n{future_vals.min():.3f} to {future_vals.max():.3f}",
                        xy=(2060, future_vals.mean()), fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 4: Sqrt normalization
        ax = axes[row, 3]
        ax.plot(years, norm_sqrt_ts.values, 'g-', linewidth=2)
        ax.set_title(f"{var} — Alternative: sqrt + min-max\n"
                     f"Range: [{norm_sqrt_ts.min().values:.3f}, {norm_sqrt_ts.max().values:.3f}]")
        ax.set_ylabel("Normalized value")
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        if future_mask.any():
            future_vals = norm_sqrt_ts.values[future_mask]
            ax.annotate(f"Future range:\n{future_vals.min():.3f} to {future_vals.max():.3f}",
                        xy=(2060, future_vals.mean()), fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_encoder_activations(activations, sample_labels, save_path="diag_encoder_layers.png"):
    """
    Plot 2: Activations at each encoder layer for different emission levels.
    If GroupNorm is collapsing the signal, you'll see the distributions
    converging after GroupNorm layers.
    """
    layer_names = list(activations.keys())
    n_layers = len(layer_names)
    n_samples = len(sample_labels)

    fig, axes = plt.subplots(n_layers, 1, figsize=(14, 3.5 * n_layers))
    if n_layers == 1:
        axes = [axes]

    colors = plt.cm.coolwarm(np.linspace(0, 1, n_samples))

    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        acts = activations[layer_name]  # list of [1, C, ...] tensors

        for j, (act, label) in enumerate(zip(acts, sample_labels)):
            vals = act.flatten().numpy()
            ax.hist(vals, bins=80, alpha=0.5, label=label, color=colors[j],
                    density=True, histtype='stepfilled', linewidth=1.5)

            # Also show mean and std as text
            mean_val = vals.mean()
            std_val = vals.std()
            ax.axvline(mean_val, color=colors[j], linestyle='--', alpha=0.8, linewidth=1)

        ax.set_title(f"Layer: {layer_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_scale_shift_vs_emissions(scale_vals, shift_vals, emission_levels,
                                   save_path="diag_scale_shift.png"):
    """
    Plot 3: How scale and shift change as a function of emission level.
    If conditioning works, you should see a monotonic trend.
    If it's flat, the encoder is not distinguishing emission levels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scale vector norms
    ax = axes[0]
    scale_norms = [s.norm().item() for s in scale_vals]
    ax.plot(emission_levels, scale_norms, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel("Emission level (spatial mean of normalized input)")
    ax.set_ylabel("||scale||")
    ax.set_title("Scale vector magnitude vs emission level")
    ax.grid(True, alpha=0.3)

    # Shift vector norms
    ax = axes[1]
    shift_norms = [s.norm().item() for s in shift_vals]
    ax.plot(emission_levels, shift_norms, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel("Emission level (spatial mean of normalized input)")
    ax.set_ylabel("||shift||")
    ax.set_title("Shift vector magnitude vs emission level")
    ax.grid(True, alpha=0.3)

    # Pairwise cosine similarity of scale vectors
    ax = axes[2]
    n = len(scale_vals)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_sim = torch.nn.functional.cosine_similarity(
                scale_vals[i].flatten().unsqueeze(0),
                scale_vals[j].flatten().unsqueeze(0)
            ).item()
            sim_matrix[i, j] = cos_sim

    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [f"{e:.2f}" for e in emission_levels]
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Cosine similarity of scale vectors\n(should vary, not all ~1.0)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_embedding_pca(scale_vals, shift_vals, emission_levels,
                        save_path="diag_embedding_pca.png"):
    """
    Plot 4: PCA of the scale/shift embeddings colored by emission level.
    If conditioning works, points should separate by emission level.
    """
    from sklearn.decomposition import PCA

    # Stack scale and shift into one embedding per sample
    embeddings = []
    for s, sh in zip(scale_vals, shift_vals):
        emb = torch.cat([s.flatten(), sh.flatten()]).numpy()
        embeddings.append(emb)
    embeddings = np.stack(embeddings)

    if embeddings.shape[0] < 3:
        print("[SKIP] Need at least 3 samples for PCA plot")
        return

    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=emission_levels,
                    cmap='coolwarm', s=100, edgecolors='black', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="Emission level (norm. spatial mean)")

    for i, lvl in enumerate(emission_levels):
        ax.annotate(f"  {lvl:.2f}", (proj[i, 0], proj[i, 1]), fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of [scale || shift] embeddings\n"
                 "Points should separate by emission level")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 4. MAIN DIAGNOSTIC ROUTINE
# ─────────────────────────────────────────────

def run_data_only_diagnostic(cond_file, cond_vars=None, output_dir="."):
    """Run just the normalization comparison (no model needed)."""
    if cond_vars is None:
        cond_vars = ["CO2", "SO2"]

    print(f"[INFO] Loading conditioning file: {cond_file}")
    ds = xr.open_dataset(cond_file)
    ds = ds[cond_vars]

    # Print raw statistics
    print("\n" + "=" * 60)
    print("RAW DATA STATISTICS")
    print("=" * 60)
    for var in cond_vars:
        da = ds[var]
        ts = da.mean(dim=[d for d in da.dims if d != "year"])
        print(f"\n{var}:")
        print(f"  Shape: {da.shape}")
        print(f"  Global min: {float(da.min()):.4e}")
        print(f"  Global max: {float(da.max()):.4e}")
        print(f"  Year 1850 spatial mean: {float(ts.isel(year=0)):.4e}")
        print(f"  Year 2100 spatial mean: {float(ts.isel(year=-1)):.4e}")
        print(f"  Ratio (2100/1850): {float(ts.isel(year=-1)) / max(float(ts.isel(year=0)), 1e-30):.2f}x")

    # Print normalized statistics for each method
    print("\n" + "=" * 60)
    print("NORMALIZED DATA STATISTICS")
    print("=" * 60)
    for var in cond_vars:
        da = ds[var]
        print(f"\n{var}:")

        for method_name, method_fn in [
            ("log10+quantile (current)", lambda d: scale_emis_m1_p1_log10(d)[0]),
            ("linear min-max", lambda d: scale_linear_m1_p1(d)[0]),
            ("sqrt + min-max", lambda d: scale_sqrt_m1_p1(d)[0]),
        ]:
            normed = method_fn(da)
            ts = normed.mean(dim=[d for d in normed.dims if d != "year"])

            # Check how much dynamic range remains in the future period
            future = ts.sel(year=slice("2015", "2100"))
            hist = ts.sel(year=slice("1850", "2014"))

            print(f"\n  {method_name}:")
            print(f"    Full range: [{float(ts.min()):.4f}, {float(ts.max()):.4f}]")
            print(f"    Historical range: [{float(hist.min()):.4f}, {float(hist.max()):.4f}]")
            print(f"    Future range:     [{float(future.min()):.4f}, {float(future.max()):.4f}]")
            print(f"    Future std:       {float(future.std()):.4f}")
            print(f"    Future dynamic range: {float(future.max()) - float(future.min()):.4f}")

    save_path = os.path.join(output_dir, "diag_normalization.png")
    plot_normalization_comparison(ds, cond_vars, save_path=save_path)


def run_encoder_diagnostic(checkpoint_path, cond_file, config_path,
                            cond_vars=None, output_dir=".", device="cpu"):
    """Run the full encoder diagnostic with a trained model."""
    if cond_vars is None:
        cond_vars = ["CO2", "SO2"]

    # First run the data diagnostic
    run_data_only_diagnostic(cond_file, cond_vars, output_dir)

    # Load model
    print(f"\n[INFO] Loading config: {config_path}")

    # Try to load with hydra/omegaconf
    try:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        conf = OmegaConf.load(config_path)
        model = instantiate(conf.model)
    except Exception as e:
        print(f"[ERROR] Could not instantiate model from config: {e}")
        print("  Make sure config_path points to your config_aero.yaml")
        return

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    chkpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try loading EMA weights
    if "EMA" in chkpt:
        model.load_state_dict(chkpt["EMA"], strict=False)
        print("[INFO] Loaded EMA weights")
    elif "Unet" in chkpt:
        model.load_state_dict(chkpt["Unet"], strict=False)
        print("[INFO] Loaded Unet weights")

    model.eval().to(device)

    # Check that the model has a cond_encoder
    if model.cond_encoder is None:
        print("[ERROR] Model has no cond_encoder! Nothing to diagnose.")
        return

    # Set up hooks
    probe = EncoderProbe(model.cond_encoder, model.cond_scale, model.cond_shift)

    # Load conditioning data and create samples at different emission levels
    print(f"\n[INFO] Loading conditioning data: {cond_file}")
    ds = xr.open_dataset(cond_file)
    ds = ds[cond_vars]

    # Pick ~8 representative years spanning the full range
    all_years = ds.year.values
    sample_years = np.linspace(0, len(all_years) - 1, 8, dtype=int)
    sample_years = all_years[sample_years]
    print(f"[INFO] Sampling years: {sample_years}")

    # Process each year through the encoder
    all_layer_acts = OrderedDict()  # layer_name -> list of activations
    scale_vals = []
    shift_vals = []
    emission_levels = []
    sample_labels = []

    for year in sample_years:
        # Get conditioning for this year (expand to batch of 1)
        year_ds = ds.sel(year=[year])

        # Normalize with current method
        normed_arrays = []
        for var in cond_vars:
            normed, _, _ = scale_emis_m1_p1_log10(year_ds[var])
            normed_arrays.append(normed.values)

        # Stack into [1, C, T, H, W] tensor
        stacked = np.stack(normed_arrays, axis=0)  # [C, T, H, W]
        cond_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)

        # Record emission level (spatial mean of normalized input)
        emis_level = float(cond_tensor.mean())
        emission_levels.append(emis_level)
        sample_labels.append(f"Year {year}\n(mean={emis_level:.3f})")

        # Forward through encoder
        probe.clear()
        with torch.no_grad():
            cond_feat = model.cond_encoder(cond_tensor)
            cond_feat_flat = cond_feat.view(cond_feat.shape[0], -1)
            scale = model.cond_scale(cond_feat_flat)
            shift = model.cond_shift(cond_feat_flat)

        scale_vals.append(scale.cpu())
        shift_vals.append(shift.cpu())

        # Store activations per layer
        for layer_name, act in probe.activations.items():
            if layer_name not in all_layer_acts:
                all_layer_acts[layer_name] = []
            all_layer_acts[layer_name].append(act)

    # Plot encoder layer activations
    save_path = os.path.join(output_dir, "diag_encoder_layers.png")
    plot_encoder_activations(all_layer_acts, sample_labels, save_path=save_path)

    # Plot scale/shift vs emission level
    save_path = os.path.join(output_dir, "diag_scale_shift.png")
    plot_scale_shift_vs_emissions(scale_vals, shift_vals, emission_levels, save_path=save_path)

    # Plot PCA of embeddings
    try:
        save_path = os.path.join(output_dir, "diag_embedding_pca.png")
        plot_embedding_pca(scale_vals, shift_vals, emission_levels, save_path=save_path)
    except ImportError:
        print("[SKIP] sklearn not available for PCA plot")

    # Print summary
    print("\n" + "=" * 60)
    print("ENCODER DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"\nScale vector norms across emission levels:")
    for lvl, s in zip(emission_levels, scale_vals):
        print(f"  emission={lvl:+.3f}  ||scale||={s.norm().item():.4f}  "
              f"mean={s.mean().item():.4f}  std={s.std().item():.4f}")

    print(f"\nShift vector norms across emission levels:")
    for lvl, s in zip(emission_levels, shift_vals):
        print(f"  emission={lvl:+.3f}  ||shift||={s.norm().item():.4f}  "
              f"mean={s.mean().item():.4f}  std={s.std().item():.4f}")

    # Key diagnostic: do the embeddings vary?
    scale_stacked = torch.stack(scale_vals)  # [N, 1, time_dim]
    scale_range = scale_stacked.std(dim=0).mean().item()
    shift_stacked = torch.stack(shift_vals)
    shift_range = shift_stacked.std(dim=0).mean().item()

    print(f"\nCross-sample variation (std of embeddings across emission levels):")
    print(f"  Scale: {scale_range:.6f}")
    print(f"  Shift: {shift_range:.6f}")

    if scale_range < 0.01 and shift_range < 0.01:
        print("\n  ⚠️  VERY LOW VARIATION — the encoder produces nearly identical")
        print("     embeddings regardless of emission level!")
        print("     → GroupNorm is likely destroying the signal.")
        print("     → Try removing GroupNorm from cond_encoder.")
    elif scale_range < 0.05:
        print("\n  ⚠️  LOW VARIATION — the encoder barely distinguishes emission levels.")
        print("     → Consider both removing GroupNorm AND using linear normalization.")
    else:
        print("\n  ✓ Reasonable variation detected. The encoder does respond to emissions.")

    probe.remove_hooks()


# ─────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose conditioning encoder")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--cond_file", type=str, required=True,
                        help="Path to conditioning NetCDF file")
    parser.add_argument("--config_path", type=str, default="./configs/config_aero.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--cond_vars", nargs="+", default=["CO2", "SO2"],
                        help="Conditioning variable names")
    parser.add_argument("--output_dir", type=str, default="./diagnostics",
                        help="Directory to save diagnostic plots")
    parser.add_argument("--data_only", action="store_true",
                        help="Only run normalization diagnostics (no model needed)")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_only or args.checkpoint is None:
        run_data_only_diagnostic(
            cond_file=args.cond_file,
            cond_vars=args.cond_vars,
            output_dir=args.output_dir,
        )
    else:
        run_encoder_diagnostic(
            checkpoint_path=args.checkpoint,
            cond_file=args.cond_file,
            config_path=args.config_path,
            cond_vars=args.cond_vars,
            output_dir=args.output_dir,
            device=args.device,
        )

    print("\n[DONE] All diagnostics saved to:", args.output_dir)