"""Diagnose whether ssp126 conditioning is in- or out-of-distribution
relative to the training scenarios (hist, ssp370, aaer, ghg).

Produces diag_cond_distribution.png with:
  1. Global-mean cumulative CO2 vs year (all scenarios overlaid)
  2. Global-mean SUL vs year (all scenarios overlaid)
  3. 2D trajectory in (CO2, SUL) cond-space — the region ssp126 traverses
     that no training scenario visits is where model predictions are
     extrapolations.

Also prints a text summary: per-scenario min/max and whether ssp126's
range extends beyond the union of the training scenarios.

Run on LUMI:
    python diag_cond_distribution.py
"""
import lumi_paths as L
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMIS_DIR = os.environ.get("EMIS_DIR", f"{L.DATA}")
OUT_DIR  = os.environ.get("OUT_DIR",  f"{L.EVAL_OUT}")
os.makedirs(OUT_DIR, exist_ok=True)

# (name, path, is_training, color, linestyle)
EXPERIMENTS = [
    ("hist",   f"{EMIS_DIR}/emissions_hist_timefixed.nc",           True,  "#1f77b4", "-"),
    ("ssp370", f"{EMIS_DIR}/emissions_ssp370_timefixed.nc",         True,  "#d62728", "-"),
    ("aaer",   f"{EMIS_DIR}/emissions_aero_only_timefixed.nc",      True,  "#ff7f0e", "-"),
    ("ghg",    f"{EMIS_DIR}/emissions_ghg_only_timefixed.nc",       True,  "#2ca02c", "-"),
    ("ssp126", f"{EMIS_DIR}/emissions_ssp126_only_timefixed.nc",     False, "#9467bd", "--"),
]
COND_VARS   = ["CO2", "SUL"]
VAR_ALIASES = {"SUL": ["SO2", "sul"], "CO2": ["co2"]}


def resolve_var(ds, var):
    if var in ds:
        return var
    for alias in VAR_ALIASES.get(var, []):
        if alias in ds:
            return alias
    return None


def gmean_ts(da):
    time_dim = [d for d in da.dims if d not in ("lat", "lon")][0]
    arr = da.transpose(time_dim, "lat", "lon").values.astype(np.float64)
    w = np.cos(np.deg2rad(da["lat"].values))
    w /= w.mean()
    return (arr * w[np.newaxis, :, np.newaxis]).mean(axis=(1, 2))


def extract_years(time_vals):
    try:
        return np.array([int(str(v)[:4]) for v in time_vals])
    except Exception:
        return np.arange(len(time_vals))


def load_scenario(path):
    ds = xr.open_dataset(path)
    time_dim = [d for d in ds.dims if d not in ("lat", "lon")][0]
    out = {"years": extract_years(ds[time_dim].values)}
    for var in COND_VARS:
        resolved = resolve_var(ds, var)
        out[var] = gmean_ts(ds[resolved]) if resolved else None
    ds.close()
    return out


# ── Load all scenarios ─────────────────────────────────────────────────────────
data = {}
for name, path, *_ in EXPERIMENTS:
    try:
        data[name] = load_scenario(path)
        yrs = data[name]["years"]
        print(f"  [{name}] years {yrs[0]}–{yrs[-1]} ({len(yrs)} steps)")
    except Exception as e:
        print(f"  [{name}] FAILED: {type(e).__name__}: {e}")

# ── Text summary: ssp126 OOD check ─────────────────────────────────────────────
print("\n" + "=" * 72)
print("Range summary (global-mean, area-weighted):")
print(f"{'Scenario':10s}  {'Var':4s}  {'Min':>12s}  {'Max':>12s}  {'Years':>12s}")
print("-" * 72)
for name, *_ in EXPERIMENTS:
    d = data.get(name)
    if d is None:
        continue
    for var in COND_VARS:
        ts = d.get(var)
        if ts is None:
            continue
        yrs = d["years"]
        print(f"{name:10s}  {var:4s}  {ts.min():12.4g}  {ts.max():12.4g}  "
              f"{yrs[0]}–{yrs[-1]}")

# OOD verdict: does ssp126 reach values outside the union of training ranges?
train_names = [n for n, _, is_tr, *_ in EXPERIMENTS if is_tr]
print("\n" + "=" * 72)
print("ssp126 OOD check (global-mean range vs union of training scenarios):")
for var in COND_VARS:
    train_vals = np.concatenate([
        data[n][var] for n in train_names
        if data.get(n) is not None and data[n].get(var) is not None
    ])
    ssp126_ts = data.get("ssp126", {}).get(var)
    if ssp126_ts is None:
        print(f"  {var}: ssp126 missing — skip")
        continue
    tr_lo, tr_hi = train_vals.min(), train_vals.max()
    s_lo, s_hi   = ssp126_ts.min(), ssp126_ts.max()
    below = s_lo < tr_lo
    above = s_hi > tr_hi
    verdict = "INSIDE training range" if not (below or above) else \
              f"OUTSIDE ({'below ' if below else ''}{'above' if above else ''})"
    print(f"  {var}: train=[{tr_lo:.4g}, {tr_hi:.4g}]  "
          f"ssp126=[{s_lo:.4g}, {s_hi:.4g}]  → {verdict}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

def plot_ts(ax, var):
    for name, _, is_train, color, ls in EXPERIMENTS:
        d = data.get(name)
        if d is None or d.get(var) is None:
            continue
        lw = 1.8 if is_train else 3.2
        alpha = 0.55 if is_train else 1.0
        ax.plot(d["years"], d[var], color=color, lw=lw, ls=ls, label=name,
                alpha=alpha)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Global-mean {var}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

plot_ts(axes[0], "CO2")
axes[0].set_title("CO2 (cumulative) — is ssp126 inside training envelope?")

plot_ts(axes[1], "SUL")
axes[1].set_title("SUL — is ssp126 inside training envelope?")

# Panel 3: 2D trajectory in cond-space
ax = axes[2]
for name, _, is_train, color, ls in EXPERIMENTS:
    d = data.get(name)
    if d is None or d.get("CO2") is None or d.get("SUL") is None:
        continue
    lw = 1.8 if is_train else 3.2
    alpha = 0.55 if is_train else 1.0
    ax.plot(d["CO2"], d["SUL"], color=color, lw=lw, ls=ls, label=name,
            alpha=alpha)
    # Mark start (open) and end (filled) points
    ax.scatter(d["CO2"][0], d["SUL"][0], facecolors="none",
               edgecolors=color, s=60, lw=1.5, zorder=5)
    ax.scatter(d["CO2"][-1], d["SUL"][-1], color=color, s=80,
               edgecolor="black", lw=0.8, zorder=6)
ax.set_xlabel("Global-mean cumulative CO2")
ax.set_ylabel("Global-mean SUL")
ax.set_title("Cond-space trajectory (○ = start year, ● = end year)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=9)

fig.suptitle(
    "Conditioning distribution — ssp126 (model-only) vs training scenarios",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
out = f"{OUT_DIR}/diag_cond_distribution.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved → {out}")
