"""Plot CO2 and SUL global-mean time series from every cond file listed in
configs/config_data.yaml under experiment_configs.

Reads (scenario_name, cond_file) pairs straight from the active training
config so the figure always reflects the files the trainer is actually
seeing.  No hardcoded experiment list.

Usage:
    python plot_cond_from_config.py
    OUT_DIR=/mnt/lumi_sc2/eval_output \
    PATH_REMAP=/scratch/project_462001328:/mnt/lumi_sc2 \
        python plot_cond_from_config.py

PATH_REMAP (optional): "src:dst" — applied as a literal-prefix replace on
each cond_file path before opening.  Useful when the YAML stores LUMI
scratch paths but you're plotting from a local mount.
"""
import os
import numpy as np
import xarray as xr
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG     = os.environ.get("CONFIG",  "configs/config_data.yaml")
OUT_DIR    = os.environ.get("OUT_DIR", "/scratch/project_462001328/eval_output")
PATH_REMAP = os.environ.get("PATH_REMAP", "")

COND_VARS   = ["CO2", "SUL"]
VAR_ALIASES = {"SUL": ["SO2", "sul"], "CO2": ["co2"]}
COLORS      = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f"]


def remap(path: str) -> str:
    if not PATH_REMAP or ":" not in PATH_REMAP:
        return path
    src, dst = PATH_REMAP.split(":", 1)
    return path.replace(src, dst, 1) if path.startswith(src) else path


def gmean_ts(da: xr.DataArray, time_dim: str) -> np.ndarray:
    arr = da.transpose(time_dim, "lat", "lon").values.astype(np.float64)
    w = np.cos(np.deg2rad(da["lat"].values))
    w /= w.mean()
    return (arr * w[np.newaxis, :, np.newaxis]).mean(axis=(1, 2))


def extract_years(time_vals) -> list:
    try:
        return [int(str(v)[:4]) for v in time_vals]
    except Exception:
        return list(range(len(time_vals)))


def resolve_var(ds: xr.Dataset, var: str):
    if var in ds:
        return var
    for alias in VAR_ALIASES.get(var, []):
        if alias in ds:
            return alias
    return None


# ── Read config ─────────────────────────────────────────────────────────────
with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

experiments = cfg.get("experiment_configs", []) or []
if not experiments:
    raise SystemExit(f"No experiment_configs in {CONFIG}")

print(f"[load] {CONFIG}: {len(experiments)} experiments")
for e in experiments:
    print(f"  - {e['scenario_name']:8s} → {e['cond_file']}")

# ── Plot grid ───────────────────────────────────────────────────────────────
n_exp = len(experiments)
n_var = len(COND_VARS)
fig, axes = plt.subplots(n_var, n_exp,
                         figsize=(4.5 * n_exp, 4 * n_var),
                         sharey=False, sharex=False, squeeze=False)

for ei, exp in enumerate(experiments):
    name = exp["scenario_name"]
    path = remap(exp["cond_file"])
    color = COLORS[ei % len(COLORS)]

    try:
        ds = xr.open_dataset(path)
    except FileNotFoundError:
        for vi in range(n_var):
            axes[vi, ei].text(0.5, 0.5, f"{name}\nfile not found",
                              ha="center", va="center", color="red",
                              transform=axes[vi, ei].transAxes, fontsize=10)
            axes[vi, ei].set_title(name, fontsize=10)
        continue

    time_dim = next((d for d in ds.dims if d not in ("lat", "lon")), None)
    years = extract_years(ds[time_dim].values) if time_dim else []

    for vi, var in enumerate(COND_VARS):
        ax = axes[vi, ei]
        resolved = resolve_var(ds, var)
        if resolved is None or time_dim is None:
            ax.text(0.5, 0.5, f"{var}\nnot in file", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{name} — {var}", fontsize=10)
            continue

        da = ds[resolved]
        ts = gmean_ts(da, time_dim)
        ax.plot(years, ts, color=color, lw=1.8)

        title_var = var if resolved == var else f"{var} (as {resolved})"
        ax.set_title(f"{name} — {title_var}", fontsize=10)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel(var, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        ax.text(0.02, 0.97,
                f"min={float(da.min()):.3g}\nmax={float(da.max()):.3g}\n"
                f"yrs {min(years)}–{max(years)}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.7, ec="none"))
        print(f"  [{name}] {var}: min={float(da.min()):.4g}  "
              f"max={float(da.max()):.4g}  years {min(years)}-{max(years)}")
    ds.close()

fig.suptitle(f"Conditioning files from {CONFIG} (experiment_configs)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
out = f"{OUT_DIR}/cond_files_from_config.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved → {out}")
