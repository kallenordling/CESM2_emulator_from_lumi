"""Plot raw CO2 and SUL emissions from all experiment conditioning files.

Each experiment gets its own y-axis so lines with very different magnitudes
(e.g. aaer CO2 max=0.02 vs ssp370 CO2 max=28.3) are all clearly visible.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMIS_DIR = os.environ.get("EMIS_DIR", "/scratch/project_462001328/emulator_data")
OUT_DIR  = os.environ.get("OUT_DIR",  "/scratch/project_462001328/eval_output")

EXPERIMENTS = {
    "hist":   f"{EMIS_DIR}/emissions_hist_timefixed.nc",
    "ssp370": f"{EMIS_DIR}/emissions_ssp370_timefixed.nc",
    "ssp126": f"{EMIS_DIR}/emissions_ssp126_only_timefixed.nc",
    "aaer":   f"{EMIS_DIR}/emissions_aero_only_timefixed.nc",
    "ghg":    f"{EMIS_DIR}/emissions_ghg_only_timefixed.nc",
}
COLORS     = {"hist": "#1f77b4", "ssp370": "#d62728", "ssp126": "#9467bd",
              "aaer": "#ff7f0e", "ghg": "#2ca02c"}
LINESTYLE  = {"hist": "-",       "ssp370": "--",       "ssp126": "-",
              "aaer": "-.",      "ghg": ":"}
COND_VARS  = ["CO2", "SUL"]
# Alternate variable names to try when the primary name is missing.
VAR_ALIASES = {"SUL": ["SO2", "sul"], "CO2": ["co2"]}


def gmean_ts(da, time_dim):
    w = np.cos(np.deg2rad(da["lat"].values))
    w /= w.mean()
    return (da.values * w[np.newaxis, :, np.newaxis]).mean(axis=(1, 2))


def extract_years(time_vals):
    try:
        return [int(str(v)[:4]) for v in time_vals]
    except Exception:
        return list(range(len(time_vals)))


n_exp = len(EXPERIMENTS)
n_var = len(COND_VARS)

fig, axes = plt.subplots(n_var, n_exp,
                         figsize=(5 * n_exp, 4 * n_var),
                         sharey=False, sharex=False)

for vi, var in enumerate(COND_VARS):
    for ei, (exp_name, cond_file) in enumerate(EXPERIMENTS.items()):
        ax = axes[vi, ei]
        try:
            ds = xr.open_dataset(cond_file)
            resolved_var = var if var in ds else next(
                (alias for alias in VAR_ALIASES.get(var, []) if alias in ds),
                None,
            )
            if resolved_var is None:
                ax.text(0.5, 0.5, f"{var}\nnot in file",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_title(f"{exp_name}", fontsize=10)
                ds.close()
                continue
            da = ds[resolved_var]
            time_dim = [d for d in da.dims if d not in ("lat", "lon")][0]
            years = extract_years(ds[time_dim].values)
            ts    = gmean_ts(da, time_dim)

            ax.plot(years, ts,
                    color=COLORS[exp_name], lw=1.8,
                    linestyle=LINESTYLE[exp_name], label=exp_name)
            title_var = var if resolved_var == var else f"{var} (as {resolved_var})"
            ax.set_title(f"{exp_name}  —  {title_var}", fontsize=10)
            ax.set_xlabel("Year", fontsize=8)
            ax.set_ylabel(var, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

            # Annotate min/max
            raw_max = float(da.max())
            raw_min = float(da.min())
            ax.text(0.02, 0.97, f"max={raw_max:.3g}\nmin={raw_min:.3g}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

            print(f"  [{exp_name}] {var}: min={raw_min:.4g}  max={raw_max:.4g}")
            ds.close()
        except FileNotFoundError:
            ax.text(0.5, 0.5, "file not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")
            ax.set_title(f"{exp_name}  —  {var}", fontsize=10)
        except Exception as e:
            ax.set_title(f"{exp_name}  —  {var} ERROR", fontsize=10)
            print(f"  [{exp_name}] {var}: ERROR — {e}")

fig.suptitle("Raw emissions per experiment — each panel has its own y-axis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
out = f"{OUT_DIR}/emissions.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved → {out}")
