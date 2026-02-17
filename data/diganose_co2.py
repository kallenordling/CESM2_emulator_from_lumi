"""
CO2 Emission Normalization Diagnostic
======================================
Analyzes your emission data and tests normalization methods that
preserve BOTH global-mean temporal trend AND spatial structure.

Usage:
    python diagnose_co2_norm.py --cond_file /path/to/compressed.nc --var CO2
"""

import argparse
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Normalization methods to compare
# ─────────────────────────────────────────────

def norm_log10_quantile_all(da, lo_pct=1.0, hi_pct=99.5, floor=1e-30):
    """Current: log10+quantile over ALL cells (including ocean zeros)."""
    x = da.clip(min=0)
    x = xr.where(x > 0, x, floor)
    lx = np.log10(x)
    lo = float(lx.quantile(lo_pct / 100.0, skipna=True))
    hi = float(lx.quantile(hi_pct / 100.0, skipna=True))
    z = (lx - lo) / max(hi - lo, 1e-30)
    return (2.0 * z - 1.0).fillna(0).astype("float32"), "log10+quantile (all cells)"


def norm_log10_quantile_nonzero(da, lo_pct=1.0, hi_pct=99.5, floor=1e-30):
    """log10+quantile on NON-ZERO cells only. Ocean → -1."""
    real_mask = da > floor
    lx = np.log10(da.where(real_mask))
    lo = float(lx.quantile(lo_pct / 100.0, skipna=True))
    hi = float(lx.quantile(hi_pct / 100.0, skipna=True))
    z = (lx - lo) / max(hi - lo, 1e-30)
    z = z.clip(max=1.0)  # soft bottom, hard top
    result = (2.0 * z - 1.0)
    result = result.fillna(-1.0)  # ocean → -1
    return result.astype("float32"), "log10+quantile (non-zero only)"


def norm_log10_quantile_nonzero_clipped(da, lo_pct=1.0, hi_pct=99.5, floor=1e-30):
    """log10+quantile on NON-ZERO cells, clipped [0,1]. Ocean → -1."""
    real_mask = da > floor
    lx = np.log10(da.where(real_mask))
    lo = float(lx.quantile(lo_pct / 100.0, skipna=True))
    hi = float(lx.quantile(hi_pct / 100.0, skipna=True))
    z = (lx - lo) / max(hi - lo, 1e-30)
    z = z.clip(0, 1)
    result = (2.0 * z - 1.0)
    result = result.fillna(-1.0)
    return result.astype("float32"), "log10+quantile (non-zero, clipped)"


def norm_log10_fixed_range(da, floor=1e-30):
    """log10 with FIXED range from data min/max of non-zero cells.
    No quantile clipping — uses the full observed range."""
    real_mask = da > floor
    lx = np.log10(da.where(real_mask))
    lo = float(lx.min(skipna=True))
    hi = float(lx.max(skipna=True))
    z = (lx - lo) / max(hi - lo, 1e-30)
    result = (2.0 * z - 1.0)
    result = result.fillna(-1.0)
    return result.astype("float32"), "log10 fixed min/max (non-zero)"


def norm_spatial_mean_linear(da):
    """Spatial-mean-first: collapse to yearly mean, then linear [-1,1]."""
    spatial_dims = [d for d in da.dims if d != "year"]
    ts = da.mean(dim=spatial_dims)
    lo = float(ts.min(skipna=True))
    hi = float(ts.max(skipna=True))
    normed = (2.0 * (ts - lo) / max(hi - lo, 1e-30) - 1.0)
    return normed.broadcast_like(da).astype("float32"), "spatial-mean-first linear"


def norm_rank_temporal(da, floor=1e-30):
    """Per-cell temporal rank normalization.
    Each cell gets its temporal percentile rank → preserves both
    spatial pattern AND temporal ordering without jumps."""
    from scipy.stats import rankdata

    real_mask = da > floor
    result = da.copy(deep=True).astype("float32")

    # For each spatial location, rank values over time
    vals = da.values  # [year, lat, lon]
    n_years = vals.shape[0]
    out = np.full_like(vals, -1.0, dtype=np.float32)

    for i in range(vals.shape[1]):
        for j in range(vals.shape[2]):
            col = vals[:, i, j]
            mask = col > floor
            if mask.sum() > 1:
                ranks = rankdata(col[mask]) / mask.sum()  # [0, 1]
                out[mask, i, j] = (2.0 * ranks - 1.0).astype(np.float32)
            elif mask.sum() == 1:
                out[mask, i, j] = 0.0

    result.values[:] = out
    return result, "per-cell temporal rank"


def norm_dual_channel(da, floor=1e-30):
    """Return TWO channels:
    ch0 = spatial-mean-first (global temporal trend)
    ch1 = log10+quantile non-zero (spatial structure)
    """
    ch0, _ = norm_spatial_mean_linear(da)
    ch1, _ = norm_log10_quantile_nonzero(da, floor=floor)
    return ch0, ch1, "dual: ch0=global-mean, ch1=log10-spatial"


# ─────────────────────────────────────────────
# Diagnostic analysis
# ─────────────────────────────────────────────

def analyze_data(da, var_name):
    """Print comprehensive data statistics."""
    vals = da.values.flatten()
    vals = vals[~np.isnan(vals)]
    spatial_dims = [d for d in da.dims if d != "year"]
    ts = da.mean(dim=spatial_dims)
    years = da.year.values

    print(f"\n{'='*70}")
    print(f"  {var_name} DATA ANALYSIS")
    print(f"{'='*70}")
    print(f"  Shape: {da.shape}  dims: {da.dims}")
    print(f"  Years: {years[0]} → {years[-1]} ({len(years)} timesteps)")

    print(f"\n  --- Value Distribution ---")
    print(f"  Global min:    {vals.min():.6e}")
    print(f"  Global max:    {vals.max():.6e}")
    print(f"  Mean:          {vals.mean():.6e}")
    print(f"  Median:        {np.median(vals):.6e}")
    print(f"  Std:           {vals.std():.6e}")

    print(f"\n  --- Zero/Near-Zero Analysis ---")
    for thresh in [0, 1e-30, 1e-20, 1e-15, 1e-10]:
        pct = (np.abs(vals) <= thresh).sum() / len(vals) * 100
        print(f"  % <= {thresh:.0e}: {pct:.1f}%")

    pos = vals[vals > 1e-30]
    if len(pos) > 0:
        print(f"\n  --- Non-Zero Distribution ({len(pos)}/{len(vals)} = {len(pos)/len(vals)*100:.1f}%) ---")
        print(f"  log10 range: [{np.log10(pos.min()):.1f}, {np.log10(pos.max()):.1f}]  "
              f"({np.log10(pos.max()) - np.log10(pos.min()):.1f} orders of magnitude)")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    {p:3d}th percentile: {np.percentile(pos, p):.6e}  "
                  f"(log10 = {np.log10(np.percentile(pos, p)):.2f})")

    print(f"\n  --- Temporal Evolution (spatial mean) ---")
    idx = np.linspace(0, len(years)-1, min(12, len(years)), dtype=int)
    for i in idx:
        print(f"    Year {years[i]:4d}: {float(ts.isel(year=i)):.6e}")

    # Check spatial variation per year
    print(f"\n  --- Spatial Variation Over Time ---")
    spatial_std = da.std(dim=spatial_dims)
    spatial_max = da.max(dim=spatial_dims)
    for i in idx:
        yr = years[i]
        yr_data = da.sel(year=yr)
        nonzero_count = int((yr_data > 1e-30).sum())
        total = int(np.prod(yr_data.shape))
        print(f"    Year {yr:4d}: mean={float(ts.isel(year=i)):.4e}  "
              f"std={float(spatial_std.isel(year=i)):.4e}  "
              f"max={float(spatial_max.isel(year=i)):.4e}  "
              f"nonzero={nonzero_count}/{total} ({nonzero_count/total*100:.0f}%)")


def evaluate_normalization(da, norm_fn, var_name, floor=1e-30):
    """Evaluate how well a normalization preserves temporal trend and spatial structure."""
    spatial_dims = [d for d in da.dims if d != "year"]
    years = da.year.values

    result = norm_fn(da)
    if isinstance(result, tuple) and len(result) == 3:
        # Dual channel
        normed_label = result[2]
        normed = result[1]  # use spatial channel for evaluation
    else:
        normed, normed_label = result

    ts_raw = da.mean(dim=spatial_dims).values
    ts_norm = normed.mean(dim=spatial_dims).values

    # Metric 1: Correlation between raw and normalized temporal trends
    corr = np.corrcoef(ts_raw, ts_norm)[0, 1]

    # Metric 2: Smoothness (no sudden jumps) — max year-to-year change
    diffs = np.abs(np.diff(ts_norm))
    max_jump = diffs.max()
    mean_jump = diffs.mean()

    # Metric 3: Dynamic range in the future period
    future_mask = years >= 2015
    if future_mask.any():
        future_range = ts_norm[future_mask].max() - ts_norm[future_mask].min()
    else:
        future_range = ts_norm.max() - ts_norm.min()

    # Metric 4: Spatial discrimination — std across spatial dims per year
    spatial_stds = normed.std(dim=spatial_dims).values
    avg_spatial_std = spatial_stds.mean()

    # Metric 5: Does it use the full [-1, 1] range?
    full_range = float(normed.max()) - float(normed.min())

    return {
        "label": normed_label,
        "temporal_corr": corr,
        "max_jump": max_jump,
        "mean_jump": mean_jump,
        "future_range": future_range,
        "avg_spatial_std": avg_spatial_std,
        "full_range": full_range,
        "normed": normed,
        "ts_norm": ts_norm,
    }


def plot_comparison(da, methods, var_name, output_dir):
    """Generate comprehensive comparison plots."""
    spatial_dims = [d for d in da.dims if d != "year"]
    years = da.year.values
    ts_raw = da.mean(dim=spatial_dims).values

    results = []
    for fn in methods:
        try:
            results.append(evaluate_normalization(da, fn, var_name))
        except Exception as e:
            print(f"  ERROR with method: {e}")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # ── Plot 1: Time series comparison with raw on secondary axis ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    ax2 = ax.twinx()
    ax2.plot(years, ts_raw, 'k-', linewidth=2.5, alpha=0.25, label='raw (right axis)')
    ax2.set_ylabel("Raw value", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    for r, c in zip(results, colors):
        ax.plot(years, r["ts_norm"], color=c, linewidth=2,
                label=f'{r["label"]} (corr={r["temporal_corr"]:.3f})')
        # Mark data points to show where the 5-year gaps are
        ax.scatter(years, r["ts_norm"], color=c, s=8, alpha=0.5)

    ax.set_ylim(-1.15, 1.15)
    ax.axhline(-1, color='gray', ls='--', alpha=0.3)
    ax.axhline(1, color='gray', ls='--', alpha=0.3)
    ax.set_title(f"{var_name} — Spatial mean: normalized vs raw", fontsize=13)
    ax.set_ylabel("Normalized value")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Future zoom
    ax = axes[1]
    future_mask = years >= 2015
    if future_mask.any():
        ax2 = ax.twinx()
        ax2.plot(years[future_mask], ts_raw[future_mask], 'k-', linewidth=2.5,
                 alpha=0.25, label='raw')
        ax2.set_ylabel("Raw value", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        for r, c in zip(results, colors):
            ax.plot(years[future_mask], r["ts_norm"][future_mask],
                    color=c, linewidth=2, label=r["label"])

        ax.set_title(f"{var_name} — Future period zoom (2015–2100)", fontsize=13)
        ax.set_ylabel("Normalized value")
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"norm_timeseries_{var_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {path}")
    plt.close()

    # ── Plot 2: Year-to-year jumps ──
    fig, ax = plt.subplots(figsize=(16, 5))
    for r, c in zip(results, colors):
        diffs = np.abs(np.diff(r["ts_norm"]))
        ax.plot(years[1:], diffs, color=c, linewidth=1.5,
                label=f'{r["label"]} (max={r["max_jump"]:.3f})')
    ax.set_title(f"{var_name} — Year-to-year |jump| in spatial mean (lower = smoother)")
    ax.set_ylabel("|Δ normalized value|")
    ax.set_xlabel("Year")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"norm_jumps_{var_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {path}")
    plt.close()

    # ── Plot 3: Spatial maps at key years ──
    key_years = [years[0], 1950, 2015, 2050, years[-1]]
    key_years = [y for y in key_years if y in years]

    for r in results:
        fig, axes_row = plt.subplots(1, len(key_years),
                                      figsize=(5 * len(key_years), 4))
        if len(key_years) == 1:
            axes_row = [axes_row]

        normed = r["normed"]
        for col, yr in enumerate(key_years):
            ax = axes_row[col]
            data = normed.sel(year=yr).values
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                           vmin=-1, vmax=1, origin='lower')
            ax.set_title(f"year={yr}\nmin={data.min():.3f} max={data.max():.3f}\n"
                         f"mean={data.mean():.3f}", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f"{var_name} — {r['label']}", fontsize=12)
        plt.tight_layout()
        safe_label = r["label"].replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(output_dir, f"norm_spatial_{var_name}_{safe_label}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {path}")
        plt.close()

    # ── Summary table ──
    print(f"\n{'='*90}")
    print(f"  NORMALIZATION COMPARISON SUMMARY — {var_name}")
    print(f"{'='*90}")
    print(f"  {'Method':<45} {'Corr':>6} {'MaxJump':>8} {'FutRng':>7} {'SpatStd':>8} {'Range':>6}")
    print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*6}")

    best_score = -1
    best_method = None

    for r in results:
        score = (
            r["temporal_corr"] * 0.3          # temporal trend preserved
            + (1 - r["max_jump"]) * 0.25      # smoothness (no jumps)
            + r["future_range"] * 0.2         # future discrimination
            + min(r["avg_spatial_std"], 0.5) * 0.15  # spatial structure
            + min(r["full_range"], 2.0) / 2.0 * 0.1   # uses full range
        )

        marker = ""
        if score > best_score:
            best_score = score
            best_method = r["label"]

        print(f"  {r['label']:<45} {r['temporal_corr']:>6.3f} {r['max_jump']:>8.4f} "
              f"{r['future_range']:>7.3f} {r['avg_spatial_std']:>8.4f} {r['full_range']:>6.2f}")

    print(f"\n  >>> RECOMMENDED: {best_method}")
    print(f"      (Score: {best_score:.3f})")
    print(f"\n  Scoring weights: temporal_corr=0.3, smoothness=0.25, "
          f"future_range=0.2, spatial_std=0.15, full_range=0.1")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond_file", type=str, required=True)
    parser.add_argument("--var", type=str, default="CO2")
    parser.add_argument("--output_dir", type=str, default="./diagnostics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = xr.open_dataset(args.cond_file)
    if args.var not in ds:
        print(f"Variable '{args.var}' not found. Available: {list(ds.data_vars)}")
        # Try first variable
        args.var = list(ds.data_vars)[0]
        print(f"Using: {args.var}")

    da = ds[args.var]

    # Analyze raw data
    analyze_data(da, args.var)

    # Define methods to compare
    methods = [
        norm_log10_quantile_all,
        norm_log10_quantile_nonzero,
        norm_log10_quantile_nonzero_clipped,
        norm_log10_fixed_range,
        norm_spatial_mean_linear,
    ]

    # Try rank-based too (can be slow for large grids)
    try:
        from scipy.stats import rankdata
        methods.append(norm_rank_temporal)
    except ImportError:
        print("[SKIP] scipy not available for rank-based normalization")

    # Run comparison
    plot_comparison(da, methods, args.var, args.output_dir)

    print(f"\n[DONE] All diagnostics saved to: {args.output_dir}")