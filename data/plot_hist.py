"""
Conditioning File Histogram Plotter
=====================================
Plots value distributions for each variable in a conditioning NetCDF file,
pooling ALL years and ALL grid points into a single sample.

Usage:
    python plot_cond_histograms.py --cond_file /path/to/cond.nc
    python plot_cond_histograms.py --cond_file /path/to/cond.nc --vars CO2 SO2
    python plot_cond_histograms.py --cond_file /path/to/cond.nc --log --drop_zeros
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr


# ── Helpers ──────────────────────────────────────────────────────────────────

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
        "n":        len(vals),
        "min":      vals.min(),
        "p01":      np.percentile(vals, 1),
        "p05":      np.percentile(vals, 5),
        "median":   np.median(vals),
        "mean":     vals.mean(),
        "p95":      np.percentile(vals, 95),
        "p99":      np.percentile(vals, 99),
        "max":      vals.max(),
        "std":      vals.std(),
        "zeros_%":  100.0 * (vals == 0).sum() / max(len(vals), 1),
    }


def stat_text(s: dict) -> str:
    return (
        f"n={s['n']:,}\n"
        f"min={s['min']:.3e}   max={s['max']:.3e}\n"
        f"mean={s['mean']:.3e}  std={s['std']:.3e}\n"
        f"p01={s['p01']:.3e}   p99={s['p99']:.3e}\n"
        f"zeros={s['zeros_%']:.1f}%"
    )


# ── Per-variable figure (linear + log side by side + CDF) ────────────────────

def plot_variable(ds: xr.Dataset, var: str, drop_zeros: bool, bins: int,
                  output_dir: str):
    vals_full = load_flat(ds, var, drop_zeros=False, drop_nan=True)
    vals_nz   = load_flat(ds, var, drop_zeros=True,  drop_nan=True)

    s_full = percentile_stats(vals_full)
    s_nz   = percentile_stats(vals_nz)

    has_pos = (vals_nz > 0).any()

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{var}  —  all years × all grid points  "
        f"(total {s_full['n']:,} values, {s_full['zeros_%']:.1f}% zeros)",
        fontsize=14, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── (0,0) Linear histogram — full data ───────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(vals_full, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax0.set_title("Linear scale (all values)")
    ax0.set_xlabel(var)
    ax0.set_ylabel("Count")
    ax0.axvline(s_full["mean"],   color="red",    ls="--", lw=1.5, label="mean")
    ax0.axvline(s_full["median"], color="orange", ls="--", lw=1.5, label="median")
    ax0.legend(fontsize=8)
    ax0.text(0.02, 0.97, stat_text(s_full), transform=ax0.transAxes,
             fontsize=7, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # ── (0,1) Linear histogram — non-zero only ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    if len(vals_nz) > 0:
        ax1.hist(vals_nz, bins=bins, color="darkorange", edgecolor="none", alpha=0.85)
        ax1.axvline(s_nz["mean"],   color="red",    ls="--", lw=1.5, label="mean")
        ax1.axvline(s_nz["median"], color="navy",   ls="--", lw=1.5, label="median")
        ax1.legend(fontsize=8)
        ax1.text(0.02, 0.97, stat_text(s_nz), transform=ax1.transAxes,
                 fontsize=7, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    else:
        ax1.text(0.5, 0.5, "no non-zero values", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=11, color="gray")
    ax1.set_title("Linear scale (non-zero values only)")
    ax1.set_xlabel(var)
    ax1.set_ylabel("Count")

    # ── (0,2) Log10 histogram — positive values ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    vals_pos = vals_nz[vals_nz > 0] if len(vals_nz) > 0 else np.array([])
    if len(vals_pos) > 0 and has_pos:
        log_vals = np.log10(vals_pos)
        ax2.hist(log_vals, bins=bins, color="seagreen", edgecolor="none", alpha=0.85)
        ax2.axvline(np.log10(np.median(vals_pos)), color="orange", ls="--",
                    lw=1.5, label="median")
        ax2.axvline(np.log10(vals_pos.mean()),     color="red",    ls="--",
                    lw=1.5, label="mean")
        ax2.set_xlabel(f"log₁₀({var})")
        ax2.legend(fontsize=8)

        # annotate a few powers of 10 on x-axis
        lmin, lmax = log_vals.min(), log_vals.max()
        ticks = np.arange(np.floor(lmin), np.ceil(lmax) + 1)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"10^{int(t)}" for t in ticks], fontsize=7)
    else:
        ax2.text(0.5, 0.5, "no positive values\n(log scale N/A)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color="gray")
        ax2.set_xlabel(f"log₁₀({var})")
    ax2.set_title("Log₁₀ scale (positive values only)")
    ax2.set_ylabel("Count")

    # ── (1,0) CDF — all values ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_v = np.sort(vals_full)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax3.plot(sorted_v, cdf, lw=1.5, color="steelblue")
    for pct, col in [(0.05, "orange"), (0.50, "red"), (0.95, "orange")]:
        idx = np.searchsorted(cdf, pct)
        idx = min(idx, len(sorted_v) - 1)
        ax3.axvline(sorted_v[idx], color=col, ls=":", lw=1.2,
                    label=f"p{int(pct*100)}={sorted_v[idx]:.2e}")
    ax3.set_title("CDF (all values)")
    ax3.set_xlabel(var)
    ax3.set_ylabel("Cumulative probability")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── (1,1) Temporal mean per year — lat-weighted ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    da = ds[var]
    years = da.year.values

    lat_name = next((c for c in ("lat", "latitude") if c in da.coords), None)
    if lat_name is not None:
        weights = np.cos(np.deg2rad(da[lat_name])).clip(min=0)
        ts = da.weighted(weights).mean(dim=[d for d in da.dims if d != "year"])
    else:
        ts = da.mean(dim=[d for d in da.dims if d != "year"])

    ax4.plot(years, ts.values, lw=1.8, color="navy")
    ax4.set_title("Lat-weighted spatial mean vs year")
    ax4.set_xlabel("Year")
    ax4.set_ylabel(f"{var} (spatial mean)")
    ax4.grid(True, alpha=0.3)

    # ── (1,2) Box-per-decade ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    decade_starts = np.arange((years[0] // 10) * 10, years[-1] + 1, 10)
    box_data, box_labels = [], []
    for d in decade_starts:
        mask = (years >= d) & (years < d + 10)
        if mask.sum() == 0:
            continue
        decade_vals = da.isel(year=np.where(mask)[0]).values.ravel()
        decade_vals = decade_vals[~np.isnan(decade_vals)]
        if len(decade_vals) > 0:
            box_data.append(decade_vals)
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

    plt.savefig(os.path.join(output_dir, f"hist_{var}.png"),
                dpi=150, bbox_inches="tight")
    print(f"[SAVED] hist_{var}.png")
    plt.close()


# ── Summary overview figure (all vars on one page) ────────────────────────────

def plot_overview(ds: xr.Dataset, cond_vars: list, drop_zeros: bool,
                  bins: int, output_dir: str):
    n = len(cond_vars)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle("Conditioning variables — value distributions (all years, all grid points)",
                 fontsize=13, fontweight="bold")

    for idx, var in enumerate(cond_vars):
        ax = axes[idx // ncols][idx % ncols]
        vals = load_flat(ds, var, drop_zeros=drop_zeros, drop_nan=True)
        if len(vals) == 0:
            ax.text(0.5, 0.5, f"{var}\n(no values)", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.hist(vals, bins=bins, edgecolor="none", alpha=0.85)
        s = percentile_stats(vals)
        ax.set_title(
            f"{var}\nmean={s['mean']:.2e}  std={s['std']:.2e}  zeros={s['zeros_%']:.1f}%",
            fontsize=9
        )
        ax.set_xlabel(var, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.axvline(s["mean"],   color="red",    ls="--", lw=1.2)
        ax.axvline(s["median"], color="orange", ls="--", lw=1.2)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # hide empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_overview.png"),
                dpi=150, bbox_inches="tight")
    print(f"[SAVED] hist_overview.png")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of all values in a conditioning NetCDF file"
    )
    parser.add_argument("--cond_file", required=True,
                        help="Path to conditioning NetCDF file")
    parser.add_argument("--vars", nargs="+", default=None,
                        help="Variables to plot (default: all in file)")
    parser.add_argument("--output_dir", default="./hist_output",
                        help="Directory to save figures")
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of histogram bins (default: 100)")
    parser.add_argument("--drop_zeros", action="store_true",
                        help="Exclude exact zeros from non-zero panels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Opening {args.cond_file}")
    ds = xr.open_dataset(args.cond_file)

    cond_vars = args.vars if args.vars else list(ds.data_vars)
    print(f"[INFO] Variables: {cond_vars}")
    print(f"[INFO] Dimensions: {dict(ds.dims)}")

    # ── Print file-level summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FILE SUMMARY")
    print("=" * 60)
    for var in cond_vars:
        vals = load_flat(ds, var, drop_zeros=False, drop_nan=True)
        s = percentile_stats(vals)
        print(f"\n{var}  shape={ds[var].shape}  dims={ds[var].dims}")
        print(f"  min={s['min']:.4e}  max={s['max']:.4e}")
        print(f"  mean={s['mean']:.4e}  std={s['std']:.4e}")
        print(f"  p01={s['p01']:.4e}  p99={s['p99']:.4e}")
        print(f"  zeros={s['zeros_%']:.2f}%  n={s['n']:,}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_overview(ds, cond_vars, drop_zeros=args.drop_zeros,
                  bins=args.bins, output_dir=args.output_dir)

    for var in cond_vars:
        print(f"\n[INFO] Plotting {var}...")
        plot_variable(ds, var, drop_zeros=args.drop_zeros,
                      bins=args.bins, output_dir=args.output_dir)

    print(f"\n[DONE] Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()