"""Verify that the normalization reference file covers all experiment emission ranges.

For each conditioning variable (CO2, SUL), checks:
  1. The raw min/max in the reference file vs each experiment file
  2. Whether any experiment values fall outside the reference range (→ clipped to ±1)
  3. Plots the normalized time series — values hitting ±1 indicate clipping

Run on LUMI:
    python diagnose_conditioning.py
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
NORM_REF_PATH = "/scratch/project_462001112/emulator_data/emissions_co2_so2_regridded.nc"
EMIS_DIR      = "/scratch/project_462001328/emulator_data"

EXPERIMENTS = {
    "hist":   f"{EMIS_DIR}/emissions_hist_timefixed.nc",
    "ssp370": f"{EMIS_DIR}/emissions_ssp370_timefixed.nc",
    "aaer":   f"{EMIS_DIR}/emissions_aero_only_timefixed.nc",
    "ghg":    f"{EMIS_DIR}/emissions_ghg_only_timefixed.nc",
}
COND_VARS = ["CO2", "SUL"]
COLORS    = {"hist": "#1f77b4", "ssp370": "#d62728", "aaer": "#ff7f0e", "ghg": "#2ca02c"}

# ── Load reference min/max ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Reference file: {NORM_REF_PATH}")
ref_ds  = xr.open_dataset(NORM_REF_PATH)
ref_mm  = {}
for var in COND_VARS:
    if var in ref_ds:
        lo, hi = float(ref_ds[var].min()), float(ref_ds[var].max())
        ref_mm[var] = (lo, hi)
        print(f"  {var:4s}  ref range: [{lo:.4g}, {hi:.4g}]")
    else:
        print(f"  {var:4s}  NOT FOUND in reference file")
ref_ds.close()

# ── Helper: area-weighted global mean time series ─────────────────────────────
def gmean_ts(da, time_dim):
    w = np.cos(np.deg2rad(da["lat"].values))
    w /= w.mean()
    return (da.values * w[np.newaxis, :, np.newaxis]).mean(axis=(1, 2))

def extract_years(time_vals):
    try:
        return [int(str(v)[:4]) for v in time_vals]
    except Exception:
        return list(range(len(time_vals)))

# ── Verify each experiment ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Experiment range check (values outside ref range → clipped to ±1)")
print(f"{'Exp':8s}  {'Var':4s}  {'Raw min':>12s}  {'Raw max':>12s}  "
      f"{'Ref min':>12s}  {'Ref max':>12s}  {'Clipped?':>10s}")
print("-" * 80)

results = {}
for exp_name, cond_file in EXPERIMENTS.items():
    results[exp_name] = {}
    try:
        ds = xr.open_dataset(cond_file)
    except FileNotFoundError:
        print(f"  {exp_name}: file not found — {cond_file}")
        continue

    time_dim = None
    for var in COND_VARS:
        if var not in ds:
            results[exp_name][var] = None
            print(f"{exp_name:8s}  {var:4s}  {'(missing)':>12s}")
            continue

        da = ds[var]
        if time_dim is None:
            time_dim = [d for d in da.dims if d not in ("lat", "lon")][0]

        raw_lo = float(da.min())
        raw_hi = float(da.max())

        if var in ref_mm:
            ref_lo, ref_hi = ref_mm[var]
            clipped_lo = raw_lo < ref_lo
            clipped_hi = raw_hi > ref_hi
            clipped = clipped_lo or clipped_hi
            flag = "YES ⚠" if clipped else "no"
        else:
            ref_lo = ref_hi = float("nan")
            clipped = False
            flag = "no ref"

        print(f"{exp_name:8s}  {var:4s}  {raw_lo:>12.4g}  {raw_hi:>12.4g}  "
              f"{ref_lo:>12.4g}  {ref_hi:>12.4g}  {flag:>10s}")

        years   = extract_years(ds[time_dim].values)
        raw_gm  = gmean_ts(da, time_dim)

        # Normalized
        if var in ref_mm:
            mean_val  = (ref_lo + ref_hi) / 2
            range_val = ref_hi - ref_lo
            norm_vals = np.clip((da.values - mean_val) / (range_val / 2), -1.0, 1.0)
            norm_gm   = (norm_vals * (np.cos(np.deg2rad(da["lat"].values)) /
                                      np.cos(np.deg2rad(da["lat"].values)).mean())
                         [:, np.newaxis]).mean(axis=(1, 2)) if norm_vals.ndim == 3 else norm_vals
        else:
            norm_gm = None

        results[exp_name][var] = dict(years=years, raw_gm=raw_gm, norm_gm=norm_gm,
                                      raw_lo=raw_lo, raw_hi=raw_hi, clipped=clipped)
    ds.close()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(COND_VARS), 2, figsize=(14, 5 * len(COND_VARS)))

for vi, var in enumerate(COND_VARS):
    ax_raw  = axes[vi, 0]
    ax_norm = axes[vi, 1]

    if var in ref_mm:
        ax_raw.axhline(ref_mm[var][0], color="k", lw=1.0, ls="--", alpha=0.5,
                       label=f"ref min={ref_mm[var][0]:.3g}")
        ax_raw.axhline(ref_mm[var][1], color="k", lw=1.0, ls=":",  alpha=0.5,
                       label=f"ref max={ref_mm[var][1]:.3g}")

    for exp_name, exp_data in results.items():
        d = exp_data.get(var)
        if d is None:
            continue
        lw    = 1.5
        label = exp_name + (" ⚠ clipped" if d["clipped"] else "")
        ax_raw.plot(d["years"], d["raw_gm"], label=label, color=COLORS[exp_name], lw=lw)
        if d["norm_gm"] is not None:
            ax_norm.plot(d["years"], d["norm_gm"], label=label, color=COLORS[exp_name], lw=lw)

    ax_norm.axhline( 1.0, color="k", lw=0.8, ls="--", alpha=0.5, label="clip +1")
    ax_norm.axhline(-1.0, color="k", lw=0.8, ls="--", alpha=0.5, label="clip -1")

    for ax, title in [(ax_raw,  f"{var} — raw global mean"),
                      (ax_norm, f"{var} — normalized (model input)")]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

fig.suptitle("Conditioning normalization check — does reference cover all experiments?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = "/scratch/project_462001328/eval_output/diagnose_conditioning.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved → {out}")
