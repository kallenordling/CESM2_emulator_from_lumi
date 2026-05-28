"""TCRE (cumCO2 → GMT) curve diagnostic — model vs CESM2, per scenario.

Joins existing best_ep*/global_mean_anomaly.csv with cumCO2 from the emissions
cond files. No new model sampling needed. Reveals where the model's forcing→
temperature response diverges from CESM2 — by scenario and by forcing magnitude.

A linear TCRE curve would mean ΔT ∝ cumCO2 with identical slope across
scenarios. Deviations:
  - Different slopes across scenarios → forcing-pathway sensitivity (CO2 vs SUL)
  - Bowed/saturating curve in one scenario → cond magnitude maps non-linearly
  - Vertical offset between model and CESM2 curves at low cumCO2 → baked-in
    baseline bias (the warm bias).
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

EMU_DIR = "/mnt/lumi_sc2/emulator_data"
EVAL_DIR = "/mnt/lumi_sc2/eval_output/run_slope-tcre"
COND_FILES = {
    "hist":   f"{EMU_DIR}/emissions_hist_only_timefixed.nc",
    "ssp370": f"{EMU_DIR}/emissions_ssp370_only_timefixed.nc",
    "aaer":   f"{EMU_DIR}/emissions_aaer_only_timefixed.nc",
    "ghg":    f"{EMU_DIR}/emissions_ghg_only_timefixed.nc",
    "ssp126": f"{EMU_DIR}/emissions_ssp126_only_timefixed.nc",
}
SCEN_COLORS = {
    "hist":   "#666666",
    "ssp370": "#d62728",
    "ssp126": "#1f77b4",
    "ghg":    "#2ca02c",
    "aaer":   "#9467bd",
}


def load_cumco2_per_year():
    """Return dict[scen][year] = cumCO2 (sum-over-grid, units = Gt cumulative)."""
    out = {}
    for scen, path in COND_FILES.items():
        ds = xr.open_dataset(path)
        tdim = "year" if "year" in ds.dims else "time"
        years = ds[tdim].values.astype(int)
        co2 = ds["CO2"].values  # (T, lat, lon) cumulative per-gridpoint
        cum = co2.sum(axis=(1, 2))
        out[scen] = {int(y): float(cum[i]) for i, y in enumerate(years)}
        ds.close()
    return out


def load_eval_csv(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["experiment"]].append(
                (int(r["year"]),
                 float(r["model_anom_degC"]),
                 float(r["cesm_anom_degC"]),
                 float(r["bias_degC"])))
    return {s: sorted(v) for s, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=985,
                    help="closest epoch to use from best_ep* (default 985)")
    ap.add_argument("--out", default="diag_tcre_curve.png")
    args = ap.parse_args()

    # Pick the eval dir nearest the requested epoch
    candidates = sorted(glob.glob(f"{EVAL_DIR}/best_ep*"))
    eps = [int(os.path.basename(d).replace("best_ep", "")) for d in candidates]
    best_idx = int(np.argmin(np.abs(np.array(eps) - args.epoch)))
    eval_dir = candidates[best_idx]
    print(f"[eval] using {eval_dir} (closest to ep{args.epoch})")

    csv_path = os.path.join(eval_dir, "global_mean_anomaly.csv")
    by_scen = load_eval_csv(csv_path)
    cum = load_cumco2_per_year()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_curve, ax_bias = axes

    summary = []
    for scen in ["hist", "ssp370", "ssp126", "aaer", "ghg"]:
        if scen not in by_scen:
            continue
        rows = by_scen[scen]
        yrs    = np.array([r[0] for r in rows])
        mod    = np.array([r[1] for r in rows])
        cesm   = np.array([r[2] for r in rows])
        bias   = np.array([r[3] for r in rows])
        cum_y  = np.array([cum[scen].get(int(y), np.nan) for y in yrs])

        col = SCEN_COLORS.get(scen, "#000")
        # Left: GMT vs cumCO2 (model solid, CESM2 dashed)
        ax_curve.plot(cum_y, mod,  color=col, lw=1.8, label=f"{scen} model")
        ax_curve.plot(cum_y, cesm, color=col, lw=1.5, ls="--", alpha=0.7,
                      label=f"{scen} CESM2")
        # Right: bias vs cumCO2
        ax_bias.plot(cum_y, bias, color=col, lw=1.8, label=scen)

        # Fit TCRE slopes (mod and cesm) on linear regime — drop ssp126 outliers
        valid = np.isfinite(cum_y) & np.isfinite(mod)
        if valid.sum() > 3:
            m_slope, m_int = np.polyfit(cum_y[valid], mod[valid], 1)
            c_slope, c_int = np.polyfit(cum_y[valid], cesm[valid], 1)
            summary.append((scen, m_slope, c_slope, m_int, c_int,
                            float(bias.mean()), float(bias[:10].mean()),
                            float(bias[-10:].mean()), len(rows)))

    ax_curve.set_xlabel("cumCO2 (Gt, grid sum — model's cond signal)")
    ax_curve.set_ylabel("GMT anomaly (°C)")
    ax_curve.set_title(f"GMT vs cumCO2 — model vs CESM2 (ep ≈ {args.epoch})")
    ax_curve.grid(alpha=.3); ax_curve.legend(fontsize=8, ncol=2)
    ax_curve.axhline(0, color="grey", lw=0.6)

    ax_bias.set_xlabel("cumCO2 (Gt, grid sum)")
    ax_bias.set_ylabel("bias = model − CESM2 (°C)")
    ax_bias.set_title("bias vs cumCO2 — where does it grow?")
    ax_bias.grid(alpha=.3); ax_bias.legend()
    ax_bias.axhline(0, color="grey", lw=0.6)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[plot] {args.out}")

    print("\n[summary] per-scenario TCRE-style slopes (K per Gt cumCO2)")
    print(f"{'scen':>8} {'model_slope':>12} {'cesm_slope':>12} {'ratio':>7} "
          f"{'mod_int':>9} {'cesm_int':>9} {'int_diff':>9} "
          f"{'bias_mean':>10} {'bias_first10':>13} {'bias_last10':>12}")
    for scen, ms, cs, mi, ci, bm, bf, bl, n in summary:
        ratio = ms / cs if abs(cs) > 1e-9 else float("nan")
        print(f"{scen:>8} {ms:12.4e} {cs:12.4e} {ratio:7.2f} "
              f"{mi:+9.3f} {ci:+9.3f} {(mi-ci):+9.3f} "
              f"{bm:+10.3f} {bf:+13.3f} {bl:+12.3f}")
    print("\n  Reading:")
    print("  - ratio ≈ 1.0 → model TCRE matches CESM2; ≠ 1 means scenario-specific sensitivity error.")
    print("  - int_diff (model intercept − CESM2 intercept) is the OFFSET part of the bias —")
    print("    a non-zero int_diff at cumCO2≈0 means baked-in baseline bias.")
    print("  - large bias_last10−bias_first10 within one scenario = bias grows with forcing.")


if __name__ == "__main__":
    main()
