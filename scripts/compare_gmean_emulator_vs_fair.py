#!/usr/bin/env python3
"""
Compare the emulator's global-mean temperature timeseries against FaIR, for the
CMIP7 scenarios.

READ THIS BEFORE INTERPRETING THE OUTPUT
----------------------------------------
FaIR is NOT ground truth here and agreement is NOT validation. Three reasons:

 1. Different CO2. FaIR is driven by the ScenarioMIP protocol paper's
    ILLUSTRATIVE pathways (van Vuuren et al. 2026 Fig. 1 = Sanderson & Smith
    2025), which the paper itself labels provisional pending the IAM runs. The
    emulator is driven by the FINAL gridded IAM quantification on ESGF. For H
    these differ by ~25% at 2100 (FaIR 71 vs ESGF 53 GtCO2/yr).
 2. Different species. FaIR responds to CH4, N2O, 41 halocarbons, OC/NH3/NOx/
    VOC/CO and volcanic forcing. The emulator sees only CO2, SUL and BC.
 3. No CESM2 truth exists for CMIP7, so neither side can be scored.

Consequence: FaIR's full-forcing run should read WARM relative to a
correctly-behaving emulator on H (its CO2 is too high), while its CO2-only run
should read COOL (no non-CO2 greenhouse warming). A well-behaved emulator
plausibly sits BETWEEN them. Use this as a plausibility bracket, not a target.

Emulator source (either works; NetCDF preferred and available first)
-------------------------------------------------------------------
--eval-dir     directory of eval_cmip7.py's <VAR>_<experiment>.nc files. Reads
               <VAR>_model_gmean_mean_anom plus the per-member
               <VAR>_model_gmean_m<N>_anom for the ensemble spread. Works as soon
               as individual experiments finish, before the combined CSV exists.
--emulator-csv eval_cmip7.py's global_mean_anomaly_cmip7_<unit>.csv, written only
               after ALL experiments complete.

Both sides are anomalies vs 1850-1900, so no re-baselining is applied.

Usage
-----
    python scripts/compare_gmean_emulator_vs_fair.py \
        --eval-dir /scratch/project_462001328/eval_output/cmip7 \
        --out plots/gmean_emulator_vs_fair.png --csv plots/gmean_comparison.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# emulator experiment -> (FaIR scenario in the reference CSVs, colour, label)
SCEN = {
    "hist_cmip7": (None, "#1f77b4", "CMIP7 historical"),
    "h":          ("h",  "#d62728", "h = H (High)"),
    "vl":         ("vl", "#2ca02c", "vl = VL (Very Low)"),
}


def from_netcdf(eval_dir, var):
    """Per-experiment global-mean anomaly + ensemble spread from the eval NetCDFs."""
    import xarray as xr
    out = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, f"{var}_*.nc"))):
        name = os.path.basename(path)[len(var) + 1:-3]
        ds = xr.open_dataset(path)          # lazy: only 1-D vars are touched
        key = f"{var}_model_gmean_mean_anom"
        if key not in ds:
            print(f"  [skip] {os.path.basename(path)}: no {key}")
            continue
        years = ds["year"].values.astype(int)
        mean = ds[key].values
        # Members are <var>_model_gmean_m<N>_anom; the ensemble mean is
        # <var>_model_gmean_mean_anom. Excluding on "mean" alone matches
        # NOTHING, because "gmean" contains "mean" — filter on the exact
        # ensemble-mean prefix instead.
        mem = [ds[v].values for v in ds.data_vars
               if v.startswith(f"{var}_model_gmean_m") and v.endswith("_anom")
               and not v.startswith(f"{var}_model_gmean_mean")]
        out[name] = dict(years=years, mean=mean,
                         members=np.stack(mem) if mem else None)
        n = len(mem) if mem else 0
        print(f"  {name:12s} {years[0]}-{years[-1]}  {n} members  "
              f"end {mean[-1]:+.3f}")
    return out


def from_csv(path, var_unit):
    out = {}
    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.startswith("model_anom")), None)
    if col is None:
        print(f"ERROR: no model_anom* column in {path}; got {list(df.columns)}",
              file=sys.stderr)
        return None
    for name, g in df.groupby("experiment"):
        g = g.sort_values("year")
        out[str(name)] = dict(years=g["year"].values.astype(int),
                              mean=g[col].values, members=None)
        print(f"  {name:12s} {g['year'].min()}-{g['year'].max()}  "
              f"end {g[col].values[-1]:+.3f}")
    return out


def load_fair(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return {sc: g.sort_values("year") for sc, g in df.groupby("emulator_scenario")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare emulator vs FaIR global-mean temperature for CMIP7",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir", help="dir of eval_cmip7.py <VAR>_<exp>.nc files")
    src.add_argument("--emulator-csv", help="global_mean_anomaly_cmip7_*.csv")
    ap.add_argument("--var", default="TREFHT")
    ap.add_argument("--fair-csv", default="reference_data/fair_cmip7_gsat.csv",
                    help="FaIR FULL-forcing reference")
    ap.add_argument("--fair-co2only-csv",
                    default="reference_data/fair_cmip7_gsat_co2only.csv",
                    help="FaIR CO2-only reference")
    ap.add_argument("--out", default="plots/gmean_emulator_vs_fair.png")
    ap.add_argument("--csv", default=None, help="write the comparison table here")
    ap.add_argument("--year-min", type=int, default=1850)
    ap.add_argument("--year-max", type=int, default=2100)
    args = ap.parse_args()

    units = "°C" if args.var == "TREFHT" else "mm/day"

    print("[emulator]")
    emu = (from_netcdf(args.eval_dir, args.var) if args.eval_dir
           else from_csv(args.emulator_csv, args.var))
    if not emu:
        print("ERROR: no emulator data found", file=sys.stderr)
        return 1

    full = load_fair(args.fair_csv)
    co2o = load_fair(args.fair_co2only_csv)
    print(f"\n[fair] full-forcing: {'ok' if full else 'MISSING ' + args.fair_csv}")
    print(f"[fair] CO2-only    : {'ok' if co2o else 'MISSING ' + str(args.fair_co2only_csv)}")

    # ── junction continuity: does the scenario start where hist ended? ───────
    if "hist_cmip7" in emu:
        h = emu["hist_cmip7"]
        last_y, last_v = int(h["years"][-1]), float(h["mean"][-1])
        print(f"\n[junction] hist_cmip7 ends {last_y}: {last_v:+.3f}{units}")
        for name in emu:
            if name == "hist_cmip7":
                continue
            e = emu[name]
            print(f"           {name:4s} starts {int(e['years'][0])}: "
                  f"{float(e['mean'][0]):+.3f}{units}  "
                  f"step {float(e['mean'][0]) - last_v:+.3f}{units}")
        print("           (a large step suggests the OOD fresh-fit PCA basis is "
              "shifting the cond — cf. the ssp126 cold-start)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))

    rows = []
    for name, e in emu.items():
        fsc, colr, lab = SCEN.get(name, (None, "#7f7f7f", name))
        m = (e["years"] >= args.year_min) & (e["years"] <= args.year_max)
        yy, vv = e["years"][m], e["mean"][m]
        ax.plot(yy, vv, color=colr, lw=2.2, label=f"emulator {lab}", zorder=3)
        if e["members"] is not None:
            ax.fill_between(yy, e["members"][:, m].min(axis=0),
                            e["members"][:, m].max(axis=0),
                            color=colr, alpha=0.18, lw=0, zorder=2)

        if fsc and full and fsc in full:
            f = full[fsc]
            fm = (f["year"] >= args.year_min) & (f["year"] <= args.year_max)
            ax.plot(f["year"][fm], f["p50"][fm], color=colr, lw=1.4, ls="--",
                    zorder=3, label=f"FaIR full-forcing {fsc} (median)")
            ax.fill_between(f["year"][fm], f["p33"][fm], f["p66"][fm],
                            color=colr, alpha=0.10, lw=0, zorder=1)
        if fsc and co2o and fsc in co2o:
            c = co2o[fsc]
            cm = (c["year"] >= args.year_min) & (c["year"] <= args.year_max)
            ax.plot(c["year"][cm], c["p50"][cm], color=colr, lw=1.2, ls=":",
                    zorder=3, label=f"FaIR CO$_2$-only {fsc} (median)")

        for yr in (2050, 2100):
            if yr not in e["years"]:
                continue
            ev = float(e["mean"][list(e["years"]).index(yr)])
            r = dict(experiment=name, year=yr, emulator=round(ev, 3))
            if fsc and full and fsc in full:
                f = full[fsc]; fr = f[f.year == yr]
                if len(fr):
                    r["fair_full_p50"] = round(float(fr["p50"].values[0]), 3)
                    r["fair_full_p33"] = round(float(fr["p33"].values[0]), 3)
                    r["fair_full_p66"] = round(float(fr["p66"].values[0]), 3)
                    r["diff_vs_full"] = round(ev - r["fair_full_p50"], 3)
            if fsc and co2o and fsc in co2o:
                c = co2o[fsc]; cr = c[c.year == yr]
                if len(cr):
                    r["fair_co2only_p50"] = round(float(cr["p50"].values[0]), 3)
                    r["diff_vs_co2only"] = round(ev - r["fair_co2only_p50"], 3)
            rows.append(r)

    ax.axhline(0, ls=":", color="k", lw=0.5)
    ax.set_xlabel("year")
    ax.set_ylabel(f"global-mean {args.var} anomaly vs 1850-1900, {units}")
    ax.set_title("CMIP7: diffusion emulator vs FaIR global-mean temperature\n"
                 "FaIR is a PLAUSIBILITY BRACKET, not truth — different CO2 "
                 "(illustrative vs final IAM) and different species; no CESM2 "
                 "run exists for CMIP7", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")

    if rows:
        t = pd.DataFrame(rows)
        print(f"\nGlobal-mean anomaly vs 1850-1900 ({units})")
        print(t.to_string(index=False))
        if args.csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
            t.to_csv(args.csv, index=False)
            print(f"\nwrote {args.csv}")
        print("\nReminder: FaIR full-forcing should read WARM on h (its CO2 is "
              "~25% too high) and\nFaIR CO2-only should read COOL (no non-CO2 "
              "GHGs). Sitting between them is the\nexpected place for a "
              "well-behaved emulator — it is not a pass/fail test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
