#!/usr/bin/env python3
"""
Extract a FaIR GSAT reference for the CMIP7 scenarios the emulator runs.

The emulator's CMIP7 eval (eval_cmip7.py) has NO CESM2 truth — CESM2 was never
run under CMIP7 forcing — so a simple climate model is the only available check
on its global-mean response. This turns the FaIR output from
chrisroadmap-cmip7-scenariomip into a tidy per-scenario/per-year CSV keyed by
the emulator's scenario names, on the emulator's baseline and year range, so the
two can be plotted or differenced directly.

Both sides express anomalies vs 1850-1900, so the values are directly comparable
WITHOUT any re-baselining.

Name mapping (ESGF gridded  ->  FaIR scenario)
---------------------------------------------
    h   ->  high-extension
    vl  ->  verylow

  * `h`: FaIR's `high-extension` and `high-overshoot` are IDENTICAL through 2100
    (the notebook copies one onto the other and their CO2 pathways share every
    anchor to 2100; both give 3.38 K in 2100). They diverge only after, so within
    the emulator's 2024-2100 window the choice does not matter.
  * `vl`: ESGF publishes `vl`; FaIR has both `verylow` (VLLO) and
    `verylow-overshoot` (VLHO), which are NOT identical (1.50 vs 1.58 K in 2100).
    `verylow` is used as the closer match, but ESGF's metadata carries no long
    name to confirm this, so treat it as the documented assumption it is --
    override with --map if the IIASA docs say otherwise.

Usage
-----
    # after running run_scenarios.py in the FaIR repo
    python scripts/make_fair_reference.py \
        --fair-csv ~/Downloads/chrisroadmap-cmip7-scenariomip-3129623/output/temperature_summary.csv \
        --out reference_data/fair_cmip7_gsat.csv

    # different mapping
    python scripts/make_fair_reference.py --fair-csv ... --map h=high-overshoot vl=verylow-overshoot

    # overlay against an emulator eval run
    python scripts/make_fair_reference.py --fair-csv ... \
        --emulator-csv /scratch/.../eval_output/cmip7/global_mean_anomaly_cmip7_degC.csv \
        --plot fair_vs_emulator.png
"""

import argparse
import os
import sys

import pandas as pd

DEFAULT_MAP = {"h": "high-extension", "vl": "verylow"}
PCTS = ["p5", "p17", "p33", "p50", "p66", "p83", "p95"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract FaIR GSAT reference for the emulator's CMIP7 scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fair-csv", required=True,
                    help="temperature_summary.csv from run_scenarios.py")
    ap.add_argument("--out", default="reference_data/fair_cmip7_gsat.csv")
    ap.add_argument("--map", nargs="+", default=None,
                    help="Override the scenario mapping, e.g. h=high-overshoot")
    ap.add_argument("--year-min", type=int, default=1850)
    ap.add_argument("--year-max", type=int, default=2100)
    ap.add_argument("--emulator-csv", default=None,
                    help="Optional: eval_cmip7.py's global_mean_anomaly CSV, to "
                         "join and (with --plot) overlay")
    ap.add_argument("--plot", default=None, help="Write a comparison PNG here")
    args = ap.parse_args()

    mapping = dict(DEFAULT_MAP)
    if args.map:
        for item in args.map:
            if "=" not in item:
                print(f"ERROR: --map entry {item!r} must be scenario=fair-name",
                      file=sys.stderr)
                return 1
            k, v = item.split("=", 1)
            mapping[k.strip()] = v.strip()

    if not os.path.exists(args.fair_csv):
        print(f"ERROR: {args.fair_csv} not found.\n"
              "Run the FaIR experiment first:\n"
              "  cd ~/Downloads/chrisroadmap-cmip7-scenariomip-3129623\n"
              "  python run_scenarios.py", file=sys.stderr)
        return 1

    fair = pd.read_csv(args.fair_csv)
    have = set(fair["scenario"].unique())
    missing = [v for v in mapping.values() if v not in have]
    if missing:
        print(f"ERROR: FaIR CSV has no scenario(s) {missing}.\n"
              f"Available: {sorted(have)}", file=sys.stderr)
        return 1

    rows = []
    for emu_name, fair_name in mapping.items():
        sub = fair[(fair["scenario"] == fair_name)
                   & (fair["year"] >= args.year_min)
                   & (fair["year"] <= args.year_max)].copy()
        sub.insert(0, "emulator_scenario", emu_name)
        sub = sub.rename(columns={"scenario": "fair_scenario"})
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["year"] = out["year"].astype(int)
    out = out[["emulator_scenario", "fair_scenario", "short", "year"] + PCTS]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} rows, "
          f"{args.year_min}-{args.year_max}, anomaly vs 1850-1900)")

    print("\nFaIR GSAT anomaly vs 1850-1900 (K), median [p33-p66]")
    for emu_name, fair_name in mapping.items():
        s = out[out["emulator_scenario"] == emu_name]
        for yr in (2050, 2100):
            r = s[s["year"] == yr]
            if len(r):
                r = r.iloc[0]
                print(f"  {emu_name:3s} ({fair_name:16s}) {yr}: "
                      f"{r['p50']:5.2f}  [{r['p33']:5.2f} – {r['p66']:5.2f}]")

    # ── optional join/overlay against the emulator ──────────────────────────
    if args.emulator_csv:
        if not os.path.exists(args.emulator_csv):
            print(f"\nWARNING: emulator CSV not found: {args.emulator_csv}")
            return 0
        emu = pd.read_csv(args.emulator_csv)
        col = next((c for c in emu.columns if c.startswith("model_anom")), None)
        if col is None:
            print(f"\nWARNING: no model_anom* column in {args.emulator_csv}; "
                  f"got {list(emu.columns)}")
            return 0
        print("\nEmulator vs FaIR (K), where both have the scenario/year:")
        for emu_name in mapping:
            e = emu[emu["experiment"] == emu_name]
            if not len(e):
                print(f"  {emu_name}: not present in the emulator CSV")
                continue
            f = out[out["emulator_scenario"] == emu_name]
            j = e.merge(f, on="year", suffixes=("_emu", "_fair"))
            for yr in (2050, 2100):
                r = j[j["year"] == yr]
                if len(r):
                    r = r.iloc[0]
                    inside = r["p33"] <= r[col] <= r["p66"]
                    print(f"  {emu_name:3s} {yr}: emulator {r[col]:5.2f}  "
                          f"FaIR {r['p50']:5.2f} [{r['p33']:5.2f}–{r['p66']:5.2f}]  "
                          f"diff {r[col]-r['p50']:+5.2f}"
                          f"{'  (within p33-p66)' if inside else ''}")

        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            colors = {"h": "#d62728", "vl": "#2ca02c"}
            fig, ax = plt.subplots(figsize=(9, 5))
            for emu_name in mapping:
                c = colors.get(emu_name, "#7f7f7f")
                f = out[out["emulator_scenario"] == emu_name]
                ax.fill_between(f["year"], f["p33"], f["p66"], color=c,
                                alpha=0.25, lw=0,
                                label=f"FaIR {emu_name} (p33-p66)")
                ax.plot(f["year"], f["p50"], color=c, lw=1.5, ls="--",
                        label=f"FaIR {emu_name} median")
                e = emu[emu["experiment"] == emu_name]
                if len(e):
                    ax.plot(e["year"], e[col], color=c, lw=2.0,
                            label=f"emulator {emu_name}")
            ax.set_xlabel("year")
            ax.set_ylabel("GSAT anomaly vs 1850-1900, K")
            ax.set_title("CMIP7 scenarios: diffusion emulator vs FaIR v2.2.0\n"
                         "(no CESM2 truth exists for CMIP7)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=130)
            print(f"\nwrote {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
