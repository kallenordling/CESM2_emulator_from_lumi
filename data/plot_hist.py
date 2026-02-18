"""
Conditioning File Histogram Plotter
=====================================
Plots value distributions for each variable in a conditioning NetCDF file,
pooling ALL years and ALL grid points into a single sample.
Also produces matching figures after several normalization strategies.

Usage:
    python plot_cond_histograms.py --cond_file /path/to/cond.nc
    python plot_cond_histograms.py --cond_file /path/to/cond.nc --vars CO2 SO2
    python plot_cond_histograms.py --cond_file /path/to/cond.nc --drop_zeros
    python plot_cond_histograms.py --cond_file /path/to/cond.nc --no_norm
"""

import argparse
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr


# ── Normalization strategies (self-contained, operate on xr.DataArray) ────────

def norm_minmax(da: xr.DataArray) -> xr.DataArray:
    """Global min-max linear scaling to [-1, 1]."""
    lo = float(da.min(skipna=True))
    hi = float(da.max(skipna=True))
    return ((2.0 * (da - lo) / max(hi - lo, 1e-30)) - 1.0).astype("float32")


def norm_log10_minmax(da: xr.DataArray) -> xr.DataArray:
    """log10(x + eps) then min-max to [-1, 1]. Good for heavy-tailed / sparse fields."""
    pos_vals = da.values[da.values > 0]
    eps = float(pos_vals.min()) * 1e-3 if len(pos_vals) > 0 else 1e-30
    log_da = np.log10(da.clip(min=0) + eps)
    return norm_minmax(log_da)


def norm_pctile_clip(da: xr.DataArray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> xr.DataArray:
    """Percentile clip then linear scaling to [-1, 1]. Robust to outliers."""
    lo = float(da.quantile(lo_pct / 100.0, skipna=True))
    hi = float(da.quantile(hi_pct / 100.0, skipna=True))
    clipped = da.clip(min=lo, max=hi)
    return ((2.0 * (clipped - lo) / max(hi - lo, 1e-30)) - 1.0).astype("float32")


def norm_zscore(da: xr.DataArray) -> xr.DataArray:
    """Z-score: (x - mean) / std. Not bounded to [-1,1] but useful for comparison."""
    mu = float(da.mean(skipna=True))
    sigma = float(da.std(skipna=True))
    return ((da - mu) / max(sigma, 1e-30)).astype("float32")


def norm_quantile(da: xr.DataArray, n_quantiles: int = 1000) -> xr.DataArray:
    """
    sklearn QuantileTransformer to uniform [0,1] then rescale to [-1, 1].
    Flattens the distribution — best for revealing structure in heavy-tailed data.
    """
    try:
        from sklearn.preprocessing import QuantileTransformer
    except ImportError:
        raise ImportError("scikit-learn is required for quantile normalization")

    vals = da.values.ravel().astype(np.float64)
    nan_mask = np.isnan(vals)
    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, int((~nan_mask).sum())),
        output_distribution="uniform",
        random_state=0,
    )
    out = np.full_like(vals, np.nan)
    out[~nan_mask] = qt.fit_transform(vals[~nan_mask].reshape(-1, 1)).ravel()
    normed_vals = (out * 2.0 - 1.0).astype(np.float32)
    return xr.DataArray(normed_vals.reshape(da.shape), dims=da.dims, coords=da.coords)


# ── Try to import normalization functions from climate_dataset.py ─────────────
# These are added to the registry only if the module is importable.

_CLIMATE_DATASET_NORMS: OrderedDict = OrderedDict()

try:
    from climate_dataset import (
        scale_cumulative_linear,
        scale_emis_m1_p1_log10,
        scale_emis_0_1_log10,
        scale_quantile_transform,
        normalize,
    )
    _CLIMATE_DATASET_NORMS = OrderedDict([
        ("★ normalize (current)",       (normalize,                {})),
        ("★ scale_quantile_transform",  (scale_quantile_transform, {})),
        ("★ scale_emis_m1_p1_log10",    (scale_emis_m1_p1_log10,   {})),
        ("★ scale_emis_0_1_log10",      (scale_emis_0_1_log10,     {})),
        ("★ scale_cumulative_linear",   (scale_cumulative_linear,  {})),
    ])
    print("[INFO] climate_dataset.py found — its normalizations will be included (★).")
except ImportError:
    print("[WARN] climate_dataset.py not importable — skipping its normalizations.")


# Registry: label -> (function, kwargs)
# climate_dataset methods come first so they appear at the top of every figure.
NORM_METHODS: OrderedDict = OrderedDict(
    list(_CLIMATE_DATASET_NORMS.items()) + [
        ("min-max [-1,1]",       (norm_minmax,       {})),
        ("log10 + min-max",      (norm_log10_minmax, {})),
        ("pctile-clip p1-p99",   (norm_pctile_clip,  {"lo_pct": 1.0, "hi_pct": 99.0})),
        ("z-score (μ=0, σ=1)",   (norm_zscore,       {})),
        ("quantile → uniform",   (norm_quantile,     {})),
    ]
)


def apply_normalizations(da: xr.DataArray) -> "OrderedDict[str, xr.DataArray | None]":
    """Apply all normalization methods to a DataArray; store None on error."""
    results = OrderedDict()
    for label, (fn, kwargs) in NORM_METHODS.items():
        print(f"    [{label}]...", end=" ", flush=True)
        try:
            results[label] = fn(da, **kwargs)
            print("OK")
        except Exception as exc:
            results[label] = None
            print(f"ERROR: {exc}")
    return results


# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_flat(ds: xr.Dataset, var: str, drop_zeros: bool, drop_nan: bool) -> np.ndarray:
    """Return all values of `var` as a 1-D numpy array."""
    vals = ds[var].values.ravel().astype(np.float64)
    if drop_nan:
        vals = vals[~np.isnan(vals)]
    if drop_zeros:
        vals = vals[vals != 0.0]
    return vals


def percentile_stats(vals: np.ndarray) -> dict:
    return {
        "n":       len(vals),
        "min":     vals.min(),
        "p01":     np.percentile(vals, 1),
        "p05":     np.percentile(vals, 5),
        "median":  np.median(vals),
        "mean":    vals.mean(),
        "p95":     np.percentile(vals, 95),
        "p99":     np.percentile(vals, 99),
        "max":     vals.max(),
        "std":     vals.std(),
        "zeros_%": 100.0 * (vals == 0).sum() / max(len(vals), 1),
    }


def stat_text(s: dict) -> str:
    return (
        f"n={s['n']:,}\n"
        f"min={s['min']:.3e}   max={s['max']:.3e}\n"
        f"mean={s['mean']:.3e}  std={s['std']:.3e}\n"
        f"p01={s['p01']:.3e}   p99={s['p99']:.3e}\n"
        f"zeros={s['zeros_%']:.1f}%"
    )


def lat_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """Latitude-cosine-weighted spatial mean, preserving the year dimension."""
    lat_name = next((c for c in ("lat", "latitude") if c in da.coords), None)
    spatial_dims = [d for d in da.dims if d != "year"]
    if lat_name is None:
        return da.mean(dim=spatial_dims)
    weights = np.cos(np.deg2rad(da[lat_name])).clip(min=0)
    return da.weighted(weights).mean(dim=spatial_dims)


def _draw_hist(ax, vals: np.ndarray, color: str, xlabel: str,
               title: str, bins: int, stats_box: bool = True):
    """Draw a single density histogram with mean/median lines."""
    if len(vals) == 0:
        ax.text(0.5, 0.5, "no values", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title(title)
        return
    s = percentile_stats(vals)
    ax.hist(vals, bins=bins, color=color, edgecolor="none", alpha=0.85, density=True)
    ax.axvline(s["mean"],   color="red",    ls="--", lw=1.5, label="mean")
    ax.axvline(s["median"], color="orange", ls="--", lw=1.5, label="median")
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if stats_box:
        ax.text(0.02, 0.97, stat_text(s), transform=ax.transAxes,
                fontsize=7, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))


def _draw_cdf(ax, vals: np.ndarray, color: str, xlabel: str, title: str | None = None):
    """Draw a CDF with p5/p50/p95 marker lines."""
    if len(vals) == 0:
        return
    sv = np.sort(vals)
    cdf = np.arange(1, len(sv) + 1) / len(sv)
    ax.plot(sv, cdf, lw=1.5, color=color)
    for pct, col in [(0.05, "orange"), (0.50, "red"), (0.95, "orange")]:
        i = min(np.searchsorted(cdf, pct), len(sv) - 1)
        ax.axvline(sv[i], color=col, ls=":", lw=1.2,
                   label=f"p{int(pct*100)}={sv[i]:.3f}")
    ax.legend(fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)


# ── Raw per-variable figure ───────────────────────────────────────────────────

def plot_variable_raw(ds: xr.Dataset, var: str, drop_zeros: bool,
                      bins: int, output_dir: str):
    vals_full = load_flat(ds, var, drop_zeros=False, drop_nan=True)
    vals_nz   = load_flat(ds, var, drop_zeros=True,  drop_nan=True)
    vals_pos  = vals_nz[vals_nz > 0] if len(vals_nz) > 0 else np.array([])
    s_full    = percentile_stats(vals_full)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{var}  —  raw  —  all years × all grid points  "
        f"({s_full['n']:,} values, {s_full['zeros_%']:.1f}% zeros)",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (0,0) Linear — all
    _draw_hist(fig.add_subplot(gs[0, 0]), vals_full, "steelblue",
               var, "Linear scale (all values)", bins)

    # (0,1) Linear — non-zero
    ax1 = fig.add_subplot(gs[0, 1])
    if len(vals_nz) > 0:
        _draw_hist(ax1, vals_nz, "darkorange", var, "Linear scale (non-zero only)", bins)
    else:
        ax1.text(0.5, 0.5, "no non-zero values", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=11, color="gray")
        ax1.set_title("Linear scale (non-zero only)")

    # (0,2) Log10 — positive
    ax2 = fig.add_subplot(gs[0, 2])
    if len(vals_pos) > 0:
        log_vals = np.log10(vals_pos)
        _draw_hist(ax2, log_vals, "seagreen", f"log₁₀({var})",
                   "Log₁₀ scale (positive only)", bins, stats_box=False)
        lmin, lmax = log_vals.min(), log_vals.max()
        ticks = np.arange(np.floor(lmin), np.ceil(lmax) + 1)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"10^{int(t)}" for t in ticks], fontsize=7)
    else:
        ax2.text(0.5, 0.5, "no positive values\n(log scale N/A)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color="gray")
        ax2.set_title("Log₁₀ scale (positive only)")

    # (1,0) CDF
    ax3 = fig.add_subplot(gs[1, 0])
    _draw_cdf(ax3, vals_full, "steelblue", var, "CDF (all values)")

    # (1,1) Lat-weighted mean vs year
    ax4 = fig.add_subplot(gs[1, 1])
    da = ds[var]
    ts = lat_weighted_mean(da)
    ax4.plot(da.year.values, ts.values, lw=1.8, color="navy")
    ax4.set_title("Lat-weighted spatial mean vs year")
    ax4.set_xlabel("Year")
    ax4.set_ylabel(f"{var} (spatial mean)")
    ax4.grid(True, alpha=0.3)

    # (1,2) Box-per-decade
    ax5 = fig.add_subplot(gs[1, 2])
    years = da.year.values
    decade_starts = np.arange((years[0] // 10) * 10, years[-1] + 1, 10)
    box_data, box_labels = [], []
    for d in decade_starts:
        mask = (years >= d) & (years < d + 10)
        if not mask.sum():
            continue
        dv = da.isel(year=np.where(mask)[0]).values.ravel()
        dv = dv[~np.isnan(dv)]
        if len(dv):
            box_data.append(dv)
            box_labels.append(str(d))
    if box_data:
        bp = ax5.boxplot(box_data, labels=box_labels, showfliers=False,
                         patch_artist=True, medianprops=dict(color="red", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")
        ax5.set_title("Distribution by decade (no outliers)")
        ax5.set_xlabel("Decade start")
        ax5.set_ylabel(var)
        ax5.tick_params(axis="x", rotation=45)
        ax5.grid(True, axis="y", alpha=0.3)

    plt.savefig(os.path.join(output_dir, f"hist_{var}_raw.png"),
                dpi=150, bbox_inches="tight")
    print(f"  [SAVED] hist_{var}_raw.png")
    plt.close()


# ── Normalized per-variable figure ───────────────────────────────────────────

def plot_variable_normalized(ds: xr.Dataset, var: str,
                              normed_dict: "OrderedDict[str, xr.DataArray | None]",
                              bins: int, output_dir: str):
    """
    One figure per variable — one row per normalization method.
    Columns: density histogram | CDF | lat-weighted mean vs year
    """
    valid = [(lbl, da) for lbl, da in normed_dict.items() if da is not None]
    if not valid:
        print(f"  [SKIP] {var}: all normalizations failed")
        return

    n_methods = len(valid)
    fig, axes = plt.subplots(n_methods, 3,
                              figsize=(18, 4.5 * n_methods),
                              squeeze=False)
    fig.suptitle(
        f"{var}  —  after normalization  —  all years × all grid points",
        fontsize=14, fontweight="bold",
    )

    years = ds[var].year.values

    for row, (label, normed_da) in enumerate(valid):
        vals = normed_da.values.ravel().astype(np.float64)
        vals = vals[~np.isnan(vals)]
        s = percentile_stats(vals)

        # col 0: density histogram
        ax0 = axes[row, 0]
        if len(vals):
            ax0.hist(vals, bins=bins, color="mediumpurple",
                     edgecolor="none", alpha=0.85, density=True)
            ax0.axvline(s["mean"],   color="red",    ls="--", lw=1.5, label="mean")
            ax0.axvline(s["median"], color="orange", ls="--", lw=1.5, label="median")
            ax0.legend(fontsize=8)
            ax0.text(0.02, 0.97, stat_text(s), transform=ax0.transAxes,
                     fontsize=7, va="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax0.set_ylabel(f"{label}\n\nDensity", fontsize=8)
        ax0.set_xlabel(f"{var} (normalized)", fontsize=8)
        if row == 0:
            ax0.set_title("Density histogram", fontsize=11)
        ax0.grid(True, alpha=0.3)

        # col 1: CDF
        ax1 = axes[row, 1]
        _draw_cdf(ax1, vals, "mediumpurple", f"{var} (normalized)",
                  "CDF" if row == 0 else None)

        # col 2: lat-weighted mean vs year
        ax2 = axes[row, 2]
        ts = lat_weighted_mean(normed_da)
        ax2.plot(years, ts.values, lw=1.8, color="darkgreen")
        ax2.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax2.set_xlabel("Year", fontsize=8)
        ax2.set_ylabel("Normalized spatial mean", fontsize=8)
        if row == 0:
            ax2.set_title("Lat-weighted spatial mean vs year", fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hist_{var}_normalized.png"),
                dpi=150, bbox_inches="tight")
    print(f"  [SAVED] hist_{var}_normalized.png")
    plt.close()


# ── Overview figures ──────────────────────────────────────────────────────────

def plot_overview_raw(ds: xr.Dataset, cond_vars: list, drop_zeros: bool,
                      bins: int, output_dir: str):
    """One-panel-per-variable summary of raw distributions."""
    n = len(cond_vars)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Raw distributions — all years × all grid points",
                 fontsize=13, fontweight="bold")

    for idx, var in enumerate(cond_vars):
        ax = axes[idx // ncols][idx % ncols]
        vals = load_flat(ds, var, drop_zeros=drop_zeros, drop_nan=True)
        if not len(vals):
            ax.text(0.5, 0.5, f"{var}\n(no values)", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        s = percentile_stats(vals)
        ax.hist(vals, bins=bins, edgecolor="none", alpha=0.85, density=True)
        ax.set_title(
            f"{var}\nmean={s['mean']:.2e}  std={s['std']:.2e}  zeros={s['zeros_%']:.1f}%",
            fontsize=9,
        )
        ax.set_xlabel(var, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.axvline(s["mean"],   color="red",    ls="--", lw=1.2)
        ax.axvline(s["median"], color="orange", ls="--", lw=1.2)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_overview_raw.png"),
                dpi=150, bbox_inches="tight")
    print(f"  [SAVED] hist_overview_raw.png")
    plt.close()


def plot_overview_normalized(cond_vars: list, all_normed: dict,
                              bins: int, output_dir: str):
    """
    Grid: rows = normalization methods, cols = variables.
    Each cell = density histogram of the normalized values.
    """
    methods  = list(NORM_METHODS.keys())
    n_vars   = len(cond_vars)
    n_methods = len(methods)
    colors   = plt.cm.tab10(np.linspace(0, 0.9, n_vars))

    fig, axes = plt.subplots(n_methods, n_vars,
                              figsize=(5 * n_vars, 3.5 * n_methods),
                              squeeze=False)
    fig.suptitle("Normalized distributions — rows=methods, cols=variables",
                 fontsize=13, fontweight="bold")

    for row, method in enumerate(methods):
        for col, var in enumerate(cond_vars):
            ax = axes[row, col]
            normed_da = all_normed[var].get(method)

            if normed_da is None:
                ax.text(0.5, 0.5, "error", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                vals = normed_da.values.ravel().astype(np.float64)
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    s = percentile_stats(vals)
                    ax.hist(vals, bins=bins, color=colors[col],
                            edgecolor="none", alpha=0.85, density=True)
                    ax.axvline(s["mean"],   color="red",   ls="--", lw=1.0)
                    ax.axvline(s["median"], color="black", ls=":",  lw=1.0)
                    ax.set_title(
                        f"{var} | {method[:22]}\n"
                        f"μ={s['mean']:.3f}  σ={s['std']:.3f}",
                        fontsize=7,
                    )

            if col == 0:
                ax.set_ylabel(method, fontsize=8)
            ax.set_xlabel("normalized value", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_overview_normalized.png"),
                dpi=150, bbox_inches="tight")
    print(f"  [SAVED] hist_overview_normalized.png")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of conditioning NetCDF, raw and after normalization"
    )
    parser.add_argument("--cond_file",  required=True,
                        help="Path to conditioning NetCDF file")
    parser.add_argument("--vars",       nargs="+", default=None,
                        help="Variables to plot (default: all data_vars)")
    parser.add_argument("--output_dir", default="./hist_output")
    parser.add_argument("--bins",       type=int, default=100,
                        help="Histogram bins (default: 100)")
    parser.add_argument("--drop_zeros", action="store_true",
                        help="Exclude exact zeros from raw non-zero panel")
    parser.add_argument("--no_norm",    action="store_true",
                        help="Skip normalization figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Opening {args.cond_file}")
    ds = xr.open_dataset(args.cond_file)

    cond_vars = args.vars if args.vars else list(ds.data_vars)
    print(f"[INFO] Variables : {cond_vars}")
    print(f"[INFO] Dimensions: {dict(ds.dims)}")

    # ── File summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FILE SUMMARY (raw)")
    print("=" * 60)
    for var in cond_vars:
        vals = load_flat(ds, var, drop_zeros=False, drop_nan=True)
        s = percentile_stats(vals)
        print(f"\n{var}  shape={ds[var].shape}  dims={ds[var].dims}")
        print(f"  min={s['min']:.4e}  max={s['max']:.4e}")
        print(f"  mean={s['mean']:.4e}  std={s['std']:.4e}")
        print(f"  p01={s['p01']:.4e}  p99={s['p99']:.4e}")
        print(f"  zeros={s['zeros_%']:.2f}%  n={s['n']:,}")

    # ── Raw plots ─────────────────────────────────────────────────────────────
    print("\n[INFO] Plotting raw distributions...")
    plot_overview_raw(ds, cond_vars, drop_zeros=args.drop_zeros,
                      bins=args.bins, output_dir=args.output_dir)
    for var in cond_vars:
        print(f"  {var}...")
        plot_variable_raw(ds, var, drop_zeros=args.drop_zeros,
                          bins=args.bins, output_dir=args.output_dir)

    # ── Normalized plots ──────────────────────────────────────────────────────
    if not args.no_norm:
        print("\n[INFO] Computing normalizations...")
        all_normed: dict = {}
        for var in cond_vars:
            print(f"  {var}:")
            all_normed[var] = apply_normalizations(ds[var])

        print("\n[INFO] Plotting normalized distributions...")
        plot_overview_normalized(cond_vars, all_normed,
                                 bins=args.bins, output_dir=args.output_dir)
        for var in cond_vars:
            print(f"  {var}...")
            plot_variable_normalized(ds, var, all_normed[var],
                                     bins=args.bins, output_dir=args.output_dir)

    print(f"\n[DONE] All figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()