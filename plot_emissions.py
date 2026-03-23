"""Plot raw CO2 and SUL emissions from all experiment conditioning files."""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMIS_DIR = "/scratch/project_462001328/emulator_data"

EXPERIMENTS = {
    "hist":   f"{EMIS_DIR}/emissions_hist_timefixed.nc",
    "ssp370": f"{EMIS_DIR}/emissions_ssp370_timefixed.nc",
    "aaer":   f"{EMIS_DIR}/emissions_aero_only_timefixed.nc",
    "ghg":    f"{EMIS_DIR}/emissions_ghg_only_timefixed.nc",
}
COLORS = {"hist": "#1f77b4", "ssp370": "#d62728", "aaer": "#ff7f0e", "ghg": "#2ca02c"}
COND_VARS = ["CO2", "SUL"]


def gmean_ts(da, time_dim):
    w = np.cos(np.deg2rad(da["lat"].values))
    w /= w.mean()
    return (da.values * w[np.newaxis, :, np.newaxis]).mean(axis=(1, 2))


def extract_years(time_vals):
    try:
        return [int(str(v)[:4]) for v in time_vals]
    except Exception:
        return list(range(len(time_vals)))


fig, axes = plt.subplots(len(COND_VARS), 1, figsize=(12, 5 * len(COND_VARS)), sharex=False)

for vi, var in enumerate(COND_VARS):
    ax = axes[vi]
    for exp_name, cond_file in EXPERIMENTS.items():
        try:
            ds = xr.open_dataset(cond_file)
            if var not in ds:
                print(f"  [{exp_name}] {var} not in file — skipping")
                ds.close()
                continue
            da = ds[var]
            time_dim = [d for d in da.dims if d not in ("lat", "lon")][0]
            years = extract_years(ds[time_dim].values)
            ts = gmean_ts(da, time_dim)
            ax.plot(years, ts, label=exp_name, color=COLORS[exp_name], lw=1.8)
            print(f"  [{exp_name}] {var}: min={float(da.min()):.4g}  max={float(da.max()):.4g}")
            ds.close()
        except FileNotFoundError:
            print(f"  [{exp_name}] file not found: {cond_file}")
        except Exception as e:
            print(f"  [{exp_name}] {var}: ERROR — {e}")

    ax.set_title(f"{var} — global mean emissions", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(var)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Conditioning emissions — all experiments", fontsize=14, fontweight="bold")
plt.tight_layout()
out = "/scratch/project_462001328/eval_output/emissions.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved → {out}")
