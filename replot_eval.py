#!/usr/bin/env python3
"""replot_eval.py
================
Re-generate evaluation plots from existing NetCDF output produced by eval_aero.py.
No model checkpoint or GPU required — reads only the saved TREFHT_*.nc files.

Usage
-----
    python replot_eval.py /path/to/eval_output_dir
    python replot_eval.py /path/to/eval_output_dir --emis-dir /scratch/.../emulator_data
    python replot_eval.py /path/to/eval_output_dir --out-dir /path/to/new_plots_dir

Arguments
---------
output_dir   Directory containing TREFHT_hist.nc, TREFHT_ssp370.nc, etc.
--emis-dir   Root directory containing emissions_*.nc files (needed for TCRE plot).
             Defaults to /scratch/project_462001328/emulator_data.
--out-dir    Where to save new plots. Defaults to output_dir (overwrites originals).
"""

import argparse
import os
import re
import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy import stats
import csv

# ── constants (must match eval_aero.py) ───────────────────────────────────────
BASELINE_START = 1850
BASELINE_END   = 1900

EXPERIMENT_META = {
    "hist":   {
        "map_years": [1900, 2000, 2014],
        "color":     "#1f77b4",
        "cond_file": "emissions_hist_timefixed.nc",
        "time_dim":  "time",
    },
    "ssp370": {
        "map_years": [2015, 2050, 2100],
        "color":     "#d62728",
        "cond_file": "emissions_ssp370_timefixed.nc",
        "time_dim":  "time",
    },
    "aaer":   {
        "map_years": [1900, 2000, 2050],
        "color":     "#ff7f0e",
        "cond_file": "emissions_aero_only_timefixed.nc",
        "time_dim":  "time",
    },
    "ghg":    {
        "map_years": [1900, 2000, 2050],
        "color":     "#2ca02c",
        "cond_file": "emissions_ghg_only_timefixed.nc",
        "time_dim":  "time",
    },
}

# Module-level lat/lon — set from first NetCDF loaded
LAT = None
LON = None


# ── helpers ────────────────────────────────────────────────────────────────────

def area_weighted_gmean(data: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Area-weighted global mean.  data: (..., H, W), lat: (H,)."""
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    w = w / w.mean()
    return (data * w).mean(axis=(-2, -1))


def _nearest_year(years: np.ndarray, target: int) -> int:
    idx = np.argmin(np.abs(years - target))
    return int(years[idx])


def extract_years(time_values) -> np.ndarray:
    """Extract integer years from xarray time coordinate values."""
    try:
        return np.array([int(str(t)[:4]) for t in time_values])
    except Exception:
        return time_values.astype(int)


def load_co2_global_annual(cond_file: str, time_dim: str, lat: np.ndarray):
    """Load area-weighted global-mean CO2 per year from an emissions file."""
    if not os.path.exists(cond_file):
        return None, None
    ds = xr.open_dataset(cond_file)
    if "CO2" not in ds:
        ds.close()
        return None, None
    co2 = ds["CO2"].values.astype(np.float64)
    years = extract_years(ds[time_dim].values)
    ds.close()
    return years, area_weighted_gmean(co2, lat)


# ── NetCDF loader ──────────────────────────────────────────────────────────────

def load_experiment_nc(nc_path: str):
    """Load one TREFHT_<name>.nc and return all arrays needed for plotting.

    Returns
    -------
    lat, lon          : 1-D coordinate arrays
    gen_years         : (T,)  model year axis
    gen_ensemble      : (N, T, H, W)  per-member absolute temperature [°C]
    baseline_map      : (H, W)  1850-1900 climatology used for anomalies
    cesm_years        : (T_c,) or None
    cesm_ensemble     : (M, T_c, H, W) or None
    """
    global LAT, LON

    ds = xr.open_dataset(nc_path)

    lat = ds["lat"].values
    lon = ds["lon"].values
    gen_years = ds["year"].values.astype(int)

    if LAT is None:
        LAT, LON = lat, lon

    # Model ensemble: gather per-member absolute fields
    n_model = int(ds.attrs.get("n_model_ensemble", 1))
    gen_members = []
    for m in range(1, n_model + 1):
        key = f"TREFHT_model_m{m}"
        if key in ds:
            gen_members.append(ds[key].values.astype(np.float32))
    if not gen_members:
        gen_members = [ds["TREFHT_model_mean"].values.astype(np.float32)]
    gen_ensemble = np.stack(gen_members, axis=0)  # (N, T, H, W)

    baseline_map = ds["baseline_map"].values.astype(np.float64)

    # CESM2 ensemble: gather per-member absolute fields
    cesm_years = None
    cesm_ensemble = None
    if "TREFHT_cesm_mean" in ds:
        cesm_years = ds["cesm_year"].values.astype(int)
        # Count available member variables
        cesm_keys = sorted(
            [v for v in ds.data_vars
             if re.fullmatch(r"TREFHT_cesm_m\d+", v)],
            key=lambda s: int(re.search(r"\d+$", s).group())
        )
        if cesm_keys:
            cesm_members = [ds[k].values.astype(np.float32) for k in cesm_keys]
            cesm_ensemble = np.stack(cesm_members, axis=0)  # (M, T_c, H, W)
        else:
            cesm_ensemble = ds["TREFHT_cesm_mean"].values[np.newaxis].astype(np.float32)

    ds.close()
    return lat, lon, gen_years, gen_ensemble, baseline_map, cesm_years, cesm_ensemble


# ── plot functions (copied from eval_aero.py, LAT/LON read from module globals)

def plot_tcre(results: dict, out_path: str):
    """ΔT vs cumulative CO2 diagram (hist + ssp370)."""
    TCRE_SCENARIOS = {"hist", "ssp370"}

    have_co2 = any(
        results.get(sc, {}).get("co2_annual") is not None
        for sc in TCRE_SCENARIOS
    )
    if not have_co2:
        print("[TCRE] No CO2 data available — skipping TCRE plot.")
        print("       Re-run with --emis-dir pointing at the emissions files.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    ax_main, ax_bias = axes

    co2_years_parts, co2_annual_parts = [], []
    for sc in ("hist", "ssp370"):
        d = results.get(sc, {})
        if d.get("co2_annual") is not None:
            co2_years_parts.append(d["co2_years"])
            co2_annual_parts.append(d["co2_annual"])

    co2_all_years  = np.concatenate(co2_years_parts)
    co2_all_annual = np.concatenate(co2_annual_parts)
    co2_cumulative = np.cumsum(co2_all_annual)
    co2_lookup = dict(zip(co2_all_years.astype(int), co2_cumulative))

    def get_cumco2(years):
        return np.array([co2_lookup.get(int(y), np.nan) for y in years])

    cesm_slope_all, model_slope_all = [], []

    for sc in ("hist", "ssp370"):
        d = results.get(sc)
        if d is None:
            continue
        c = d["color"]
        gen_ens   = d["gen_anom_ens"]
        gen_years = d["gen_years"]
        gen_mean  = gen_ens.mean(axis=0)
        N_gen     = gen_ens.shape[0]

        cumco2 = get_cumco2(gen_years)
        valid  = ~np.isnan(cumco2)

        if N_gen == 1:
            ax_main.plot(cumco2[valid], gen_mean[valid], color=c, lw=1.8,
                         label=f"{sc} model")
        else:
            lo = gen_ens[:, valid].min(axis=0)
            hi = gen_ens[:, valid].max(axis=0)
            ax_main.fill_between(cumco2[valid], lo, hi, color=c, alpha=0.18)
            ax_main.plot(cumco2[valid], gen_mean[valid], color=c, lw=1.8,
                         label=f"{sc} model (N={N_gen})")

        x_v, y_v = cumco2[valid], gen_mean[valid]
        if len(x_v) > 2:
            slope, intercept = np.polyfit(x_v, y_v, 1)
            model_slope_all.append((sc, slope))

        if d.get("cesm_anom") is not None:
            cesm_ens   = d["cesm_anom_ens"]
            cesm_mean  = d["cesm_anom"]
            cesm_years = d["cesm_years"]
            N_cesm     = cesm_ens.shape[0]

            cumco2_c = get_cumco2(cesm_years)
            valid_c  = ~np.isnan(cumco2_c)

            if N_cesm == 1:
                ax_main.plot(cumco2_c[valid_c], cesm_mean[valid_c],
                             color=c, lw=1.8, ls="--", alpha=0.8,
                             label=f"{sc} CESM2")
            else:
                lo_c = cesm_ens[:, valid_c].min(axis=0)
                hi_c = cesm_ens[:, valid_c].max(axis=0)
                ax_main.fill_between(cumco2_c[valid_c], lo_c, hi_c,
                                     facecolor=c, alpha=0.10, lw=0)
                ax_main.fill_between(cumco2_c[valid_c], lo_c, hi_c,
                                     facecolor="none", hatch="///",
                                     edgecolor=c, alpha=0.4, lw=0)
                ax_main.plot(cumco2_c[valid_c], cesm_mean[valid_c],
                             color=c, lw=1.8, ls="--", alpha=0.8,
                             label=f"{sc} CESM2 (N={N_cesm})")

            xc, yc = cumco2_c[valid_c], cesm_mean[valid_c]
            if len(xc) > 2:
                slope_c, _ = np.polyfit(xc, yc, 1)
                cesm_slope_all.append((sc, slope_c))

            common_y, ig, ic = np.intersect1d(gen_years, cesm_years,
                                               return_indices=True)
            cumco2_common = get_cumco2(common_y)
            valid_b = ~np.isnan(cumco2_common)
            diff = gen_mean[ig] - cesm_mean[ic]
            ax_bias.plot(cumco2_common[valid_b], diff[valid_b],
                         color=c, lw=1.5, label=f"{sc}")
            if N_gen > 1:
                dlo = gen_ens[:, ig][:, valid_b].min(axis=0) - cesm_mean[ic][valid_b]
                dhi = gen_ens[:, ig][:, valid_b].max(axis=0) - cesm_mean[ic][valid_b]
                ax_bias.fill_between(cumco2_common[valid_b], dlo, dhi,
                                     color=c, alpha=0.15)

    def _combined_regression(key_ens, key_years):
        xs, ys = [], []
        for sc in ("hist", "ssp370"):
            d = results.get(sc)
            if d is None or d.get(key_ens) is None:
                continue
            yr = d[key_years]
            en = d[key_ens].mean(axis=0)
            cc = get_cumco2(yr)
            v  = ~np.isnan(cc)
            xs.append(cc[v]); ys.append(en[v])
        if not xs:
            return None, None
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        return np.polyfit(xs, ys, 1)

    m_slope, m_int = _combined_regression("gen_anom_ens", "gen_years")
    c_slope, c_int = _combined_regression("cesm_anom_ens", "cesm_years")

    x_range = np.array([co2_cumulative.min(), co2_cumulative.max()])
    if m_slope is not None:
        ax_main.plot(x_range, m_slope * x_range + m_int,
                     color="k", lw=1.2, ls="-",
                     label=f"Model fit  slope={m_slope:.4f} °C/unit")
    if c_slope is not None:
        ax_main.plot(x_range, c_slope * x_range + c_int,
                     color="k", lw=1.2, ls="--",
                     label=f"CESM2 fit   slope={c_slope:.4f} °C/unit")

    ax_main.axhline(0, color="k", lw=0.6, ls=":")
    ax_main.set_xlabel("Cumulative CO₂ (area-weighted global mean, native units)")
    ax_main.set_ylabel("Global-mean TREFHT anomaly re 1850–1900 (°C)")
    ax_main.set_title("TCRE — ΔT vs cumulative CO₂  (hist + ssp370)")
    ax_main.legend(fontsize=7, ncol=2)
    ax_main.grid(True, alpha=0.25)

    ref_band_drawn_tcre = False
    for sc in ("hist", "ssp370"):
        d = results.get(sc)
        if d is None or d.get("ref_diff") is None or d.get("ref_years") is None:
            continue
        ry  = d["ref_years"]
        rd  = np.abs(d["ref_diff"])
        cc  = get_cumco2(ry)
        v   = ~np.isnan(cc)
        if v.any():
            label = "±|member diff| (nat. var.)" if not ref_band_drawn_tcre else None
            ax_bias.fill_between(cc[v], -rd[v], rd[v], color="grey", alpha=0.20,
                                 zorder=0, label=label)
            ref_band_drawn_tcre = True

    ax_bias.axhline(0, color="k", lw=0.9)
    ax_bias.set_xlabel("Cumulative CO₂ (area-weighted global mean, native units)")
    ax_bias.set_ylabel("Bias: model − CESM2 (°C)")
    ax_bias.set_title("TCRE bias")
    ax_bias.legend(fontsize=8)
    ax_bias.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_timeseries(results: dict, out_path: str):
    """Global-mean anomaly time series, all experiments."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    last_n_gen = 1
    for name, d in results.items():
        c = d["color"]
        gen_anom_ens  = d["gen_anom_ens"]
        gen_anom_mean = gen_anom_ens.mean(axis=0)
        gen_years     = d["gen_years"]
        N_gen = gen_anom_ens.shape[0]
        last_n_gen = N_gen

        if N_gen == 1:
            ax_top.plot(gen_years, gen_anom_mean, color=c, lw=2.0,
                        label=f"{name} (model)")
        else:
            gen_lo = gen_anom_ens.min(axis=0)
            gen_hi = gen_anom_ens.max(axis=0)
            ax_top.fill_between(gen_years, gen_lo, gen_hi, color=c, alpha=0.18)
            ax_top.plot(gen_years, gen_anom_mean, color=c, lw=2.0,
                        label=f"{name} (model mean ± spread, N={N_gen})")

        if d.get("cesm_anom") is not None:
            cesm_anom_ens  = d["cesm_anom_ens"]
            cesm_anom_mean = d["cesm_anom"]
            cesm_years     = d["cesm_years"]
            N_cesm = cesm_anom_ens.shape[0]

            if N_cesm == 1:
                ax_top.plot(cesm_years, cesm_anom_mean, color=c, lw=2.0,
                            ls="--", alpha=0.8, label=f"{name} (CESM2)")
            else:
                cesm_lo = cesm_anom_ens.min(axis=0)
                cesm_hi = cesm_anom_ens.max(axis=0)
                ax_top.fill_between(cesm_years, cesm_lo, cesm_hi,
                                    facecolor=c, alpha=0.10, lw=0)
                ax_top.fill_between(cesm_years, cesm_lo, cesm_hi,
                                    facecolor="none", hatch="///",
                                    edgecolor=c, alpha=0.4, lw=0)
                ax_top.plot(cesm_years, cesm_anom_mean, color=c, lw=2.0,
                            ls="--", alpha=0.8,
                            label=f"{name} (CESM2 mean ± spread, N={N_cesm})")

            common, idx_gen, idx_cs = np.intersect1d(
                gen_years, cesm_years, return_indices=True
            )
            diff_mean = gen_anom_mean[idx_gen] - cesm_anom_mean[idx_cs]
            ax_bot.plot(common, diff_mean, color=c, lw=1.5, label=name)
            if N_gen > 1:
                diff_lo = gen_anom_ens[:, idx_gen].min(axis=0) - cesm_anom_mean[idx_cs]
                diff_hi = gen_anom_ens[:, idx_gen].max(axis=0) - cesm_anom_mean[idx_cs]
                ax_bot.fill_between(common, diff_lo, diff_hi, alpha=0.15, color=c)

    ref_band_drawn = False
    for name, d in results.items():
        if d.get("ref_diff") is not None and d.get("ref_years") is not None:
            rd = np.abs(d["ref_diff"])
            ry = d["ref_years"]
            label = "±|member diff| (nat. var.)" if not ref_band_drawn else None
            ax_bot.fill_between(ry, -rd, rd, color="grey", alpha=0.20,
                                zorder=0, label=label)
            ref_band_drawn = True

    ax_top.axhline(0, color="k", lw=0.6, ls=":")
    ax_top.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12,
                   label="baseline period")
    ax_top.set_ylabel("TREFHT anomaly (°C)")
    ax_top.set_title(
        f"Global-mean temperature anomaly vs 1850–1900  "
        f"(model solid, CESM2 dashed — {last_n_gen}-member ensemble)"
    )
    ax_top.legend(fontsize=8, ncol=2)
    ax_top.grid(True, alpha=0.25)

    ax_bot.axhline(0, color="k", lw=0.9)
    ax_bot.axvspan(BASELINE_START, BASELINE_END, color="grey", alpha=0.12)
    ax_bot.set_xlabel("Year")
    ax_bot.set_ylabel("Bias: model − CESM2 (°C)")
    ax_bot.set_title(
        "Model bias relative to CESM2 ensemble mean"
        + (" (shaded = model min/max spread)" if last_n_gen > 1 else "")
    )
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_anomaly_maps(name: str, gen_data: np.ndarray, gen_years: np.ndarray,
                      baseline_map: np.ndarray, map_years: list,
                      cesm_data, cesm_years,
                      out_path: str,
                      gen_ensemble=None,
                      cesm_ensemble=None):
    """Spatial anomaly maps at requested years."""
    has_cesm = (cesm_data is not None) and (cesm_years is not None)
    n_cols = len(map_years)
    n_rows = 3 if has_cesm else 1
    row_labels = ["Model"]
    if has_cesm:
        row_labels += ["CESM2", "Model − CESM2"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    vmax_anom = 4.0
    vmax_diff = 2.0
    cmap = plt.cm.RdBu_r
    norm_anom = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax_anom, vmax=vmax_anom)
    norm_diff = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax_diff, vmax=vmax_diff)

    def _plot_panel(ax, data, norm, title):
        data_cyc, lon_cyc = add_cyclic_point(data, coord=LON)
        da = xr.DataArray(
            data_cyc, dims=["lat", "lon"],
            coords={"lat": LAT, "lon": lon_cyc},
            attrs={"units": "°C"},
        )
        da.plot.pcolormesh(
            ax=ax, cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            add_colorbar=True,
            cbar_kwargs={"label": "°C", "shrink": 0.75},
        )
        ax.add_feature(cfeature.COASTLINE, lw=0.5)
        ax.add_feature(cfeature.BORDERS, lw=0.3, linestyle=":")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                          color="grey", alpha=0.5, linestyle="--")
        gl.top_labels   = False
        gl.right_labels = False
        ax.set_title(title, fontsize=9)
        gmean = float(area_weighted_gmean(data[np.newaxis], LAT)[0])
        ax.text(0.98, 0.03, f"GM: {gmean:+.2f}°C",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    for col, yr_target in enumerate(map_years):
        yr_gen  = _nearest_year(gen_years, yr_target)
        idx_gen = int(np.where(gen_years == yr_gen)[0][0])
        anom_gen = gen_data[idx_gen] - baseline_map

        _plot_panel(axes[0, col], anom_gen, norm_anom,
                    f"{name} model  ({yr_gen})")

        if has_cesm:
            yr_cs  = _nearest_year(cesm_years, yr_target)
            idx_cs = int(np.where(cesm_years == yr_cs)[0][0])
            anom_cs = cesm_data[idx_cs] - baseline_map

            _plot_panel(axes[1, col], anom_cs, norm_anom,
                        f"{name} CESM2  ({yr_cs})")
            _plot_panel(axes[2, col], anom_gen - anom_cs, norm_diff,
                        f"Model − CESM2  ({yr_gen})")

            if gen_ensemble is not None and cesm_ensemble is not None:
                gen_members  = gen_ensemble[:, idx_gen]  - baseline_map
                cesm_members = cesm_ensemble[:, idx_cs] - baseline_map
                _, pvals = stats.ttest_ind(gen_members, cesm_members,
                                           axis=0, equal_var=False)
                sig_mask = pvals < 0.05
                lon_idx, lat_idx = np.meshgrid(np.arange(sig_mask.shape[1]),
                                               np.arange(sig_mask.shape[0]))
                sig_lats = LAT[lat_idx[sig_mask]]
                sig_lons = LON[lon_idx[sig_mask]]
                ax_diff = axes[2, col]
                ax_diff.scatter(sig_lons, sig_lats, transform=ccrs.PlateCarree(),
                                color="k", s=0.3, alpha=0.5, linewidths=0,
                                rasterized=True, zorder=5)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f"TREFHT anomaly vs 1850–1900 — {name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def save_csv(results: dict, out_path: str):
    """Save global-mean anomaly and bias to CSV."""
    rows = []
    for name, d in results.items():
        gen_anom_mean = d["gen_anom_ens"].mean(axis=0)
        gen_years     = d["gen_years"]

        if d.get("cesm_anom") is not None:
            cesm_anom_mean = d["cesm_anom"]
            cesm_years     = d["cesm_years"]
            common, idx_gen, idx_cs = np.intersect1d(
                gen_years, cesm_years, return_indices=True
            )
            cesm_lookup = {int(yr): float(cesm_anom_mean[i])
                           for yr, i in zip(common, idx_cs)}
        else:
            cesm_lookup = {}

        for i, yr in enumerate(gen_years):
            yr_int = int(yr)
            model_anom = float(gen_anom_mean[i])
            cesm_anom  = cesm_lookup.get(yr_int, float("nan"))
            bias       = model_anom - cesm_anom if not np.isnan(cesm_anom) else float("nan")
            rows.append({
                "experiment":      name,
                "year":            yr_int,
                "model_anom_degC": round(model_anom, 4),
                "cesm_anom_degC":  round(cesm_anom, 4) if not np.isnan(cesm_anom) else "",
                "bias_degC":       round(bias, 4)       if not np.isnan(bias)      else "",
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "year", "model_anom_degC",
                           "cesm_anom_degC", "bias_degC"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → saved {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-generate eval plots from existing TREFHT_*.nc output."
    )
    parser.add_argument(
        "output_dir",
        help="Directory containing TREFHT_hist.nc, TREFHT_ssp370.nc, etc.",
    )
    parser.add_argument(
        "--emis-dir",
        default="/scratch/project_462001328/emulator_data",
        help="Root directory of emissions_*.nc files (for TCRE CO2 axis). "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write new plots. Defaults to output_dir.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    timeseries_results = {}

    for name, meta in EXPERIMENT_META.items():
        nc_path = os.path.join(args.output_dir, f"TREFHT_{name}.nc")
        if not os.path.exists(nc_path):
            print(f"[SKIP] {nc_path} not found")
            continue

        print(f"\n[{name}] Loading {nc_path} …")
        lat, lon, gen_years, gen_ensemble, baseline_map, cesm_years, cesm_ensemble = \
            load_experiment_nc(nc_path)

        n_model = gen_ensemble.shape[0]
        n_cesm  = cesm_ensemble.shape[0] if cesm_ensemble is not None else 0
        print(f"  model members={n_model}  CESM2 members={n_cesm}"
              f"  years {gen_years[0]}–{gen_years[-1]}")

        gen_data  = gen_ensemble.mean(axis=0)   # (T, H, W)
        cesm_data = cesm_ensemble.mean(axis=0) if cesm_ensemble is not None else None

        # ── anomaly maps ──────────────────────────────────────────────────────
        map_out = os.path.join(out_dir, f"anomaly_maps_{name}.png")
        print(f"  Plotting anomaly maps …")
        plot_anomaly_maps(
            name         = name,
            gen_data     = gen_data,
            gen_years    = gen_years,
            baseline_map = baseline_map,
            map_years    = meta["map_years"],
            cesm_data    = cesm_data,
            cesm_years   = cesm_years,
            out_path     = map_out,
            gen_ensemble = gen_ensemble,
            cesm_ensemble= cesm_ensemble,
        )

        # ── global-mean anomaly per member ────────────────────────────────────
        bl_scalar = float(area_weighted_gmean(baseline_map[np.newaxis], lat)[0])

        gen_anom_ens = np.stack(
            [area_weighted_gmean(gen_ensemble[m], lat) - bl_scalar
             for m in range(n_model)],
            axis=0,
        )   # (N, T)

        cesm_anom_ens = cesm_anom = cesm_years_out = None
        if cesm_ensemble is not None:
            cesm_anom_ens = np.stack(
                [area_weighted_gmean(cesm_ensemble[m], lat) - bl_scalar
                 for m in range(n_cesm)],
                axis=0,
            )   # (M, T_c)
            cesm_anom     = cesm_anom_ens.mean(axis=0)
            cesm_years_out = cesm_years

        # ── natural-variability reference band from CESM2 member diff ─────────
        ref_diff = ref_years_out = None
        if cesm_anom_ens is not None and cesm_anom_ens.shape[0] >= 2:
            ref_diff      = cesm_anom_ens[0] - cesm_anom_ens[1]
            ref_years_out = cesm_years_out

        # ── CO2 for TCRE (hist / ssp370 only) ─────────────────────────────────
        co2_years_raw = co2_annual_raw = None
        if name in ("hist", "ssp370") and lat is not None:
            cond_file = os.path.join(args.emis_dir, meta["cond_file"])
            co2_years_raw, co2_annual_raw = load_co2_global_annual(
                cond_file, meta["time_dim"], lat
            )
            if co2_annual_raw is None:
                print(f"  [CO2] not found at {cond_file} — TCRE will be skipped")

        timeseries_results[name] = dict(
            gen_anom_ens  = gen_anom_ens,
            gen_years     = gen_years,
            cesm_anom_ens = cesm_anom_ens,
            cesm_anom     = cesm_anom,
            cesm_years    = cesm_years_out,
            color         = meta["color"],
            co2_years     = co2_years_raw,
            co2_annual    = co2_annual_raw,
            ref_diff      = ref_diff,
            ref_years     = ref_years_out,
        )

    if not timeseries_results:
        print("\n[ERROR] No NetCDF files found — nothing to plot.")
        return

    # ── global-mean time series ────────────────────────────────────────────────
    ts_out  = os.path.join(out_dir, "global_mean_anomaly.png")
    csv_out = os.path.join(out_dir, "global_mean_anomaly.csv")
    print(f"\n[PLOT] Time series → {ts_out}")
    plot_timeseries(timeseries_results, ts_out)
    print(f"[CSV]  Global anomaly + bias → {csv_out}")
    save_csv(timeseries_results, csv_out)

    # ── TCRE ──────────────────────────────────────────────────────────────────
    tcre_out = os.path.join(out_dir, "tcre.png")
    print(f"[PLOT] TCRE → {tcre_out}")
    plot_tcre(timeseries_results, tcre_out)

    print(f"\n[DONE] Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
