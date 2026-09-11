#!/usr/bin/env python3
"""
Plot the CMIP7 ScenarioMIP emissions that drive FaIR, for all 7 pathways.

Source: chrisroadmap's `cmip7-scenariomip` repo,
`data/emissions/extensions_1750-2500.csv` — 7 scenarios x 52 species x
1750.5-2500.5, annual, global totals.

WHAT THESE NUMBERS ARE, AND ARE NOT
-----------------------------------
These are the ILLUSTRATIVE pathways behind Figure 1 of the ScenarioMIP protocol
paper (van Vuuren et al. 2026, GMD 19:2627, citing Sanderson and Smith 2025).
That paper is explicit that they are indicative: "The final emission
trajectories will depend on the finalized IAM runs but are expected to be
roughly consistent with the illustrations provided here."

The FINAL IAM quantification is a separate publication (Zenodo 19825038), and
THAT is what was harmonized, infilled and gridded into the input4MIPs
`IIASA-IAMC-*-1-1-0` files the emulator is conditioned on. For High the two
differ substantially — 71.3 GtCO2/yr FFI at 2100 here against 53.2 in the
gridded set, ~34% — because the IAM came in well below the illustration.

So this figure documents the illustrative pathways. It is NOT a target the
emulator should reproduce, and a disagreement between these curves and the
emulator's conditioning is expected rather than a defect. See the
fair_cmip7_scenariomip note.

Scenario codes (protocol -> this file):
    H  high-extension    HL high-overshoot     M  medium-extension
    ML medium-overshoot  L  low                VL verylow
    LN verylow-overshoot (see note: LN mapping was never fully resolved)

Usage:
    python scripts/plot_fair_cmip7_emissions.py
    python scripts/plot_fair_cmip7_emissions.py --year-end 2500
    python scripts/plot_fair_cmip7_emissions.py --scenarios high-extension verylow
    python scripts/plot_fair_cmip7_emissions.py --dump-data plots/fair_cmip7.csv
"""
import argparse
import os
import sys

import numpy as np

ROOT = "/home/nordling/Downloads/chrisroadmap-cmip7-scenariomip-3129623"
CSV = "data/emissions/extensions_1750-2500.csv"

# Protocol order: warmest to coolest, so the legend reads like the fan.
ORDER = ["high-extension", "high-overshoot", "medium-extension",
         "medium-overshoot", "low", "verylow", "verylow-overshoot"]
CODE = {"high-extension": "H", "high-overshoot": "HL", "medium-extension": "M",
        "medium-overshoot": "ML", "low": "L", "verylow": "VL",
        "verylow-overshoot": "LN?"}
COL = {"high-extension": "#7b241c", "high-overshoot": "#c0392b",
       "medium-extension": "#b9770e", "medium-overshoot": "#e08e0b",
       "low": "#1e8449", "verylow": "#148f77",
       "verylow-overshoot": "#2471a3"}
# Overshoot pathways dashed: they are not variants of their parent, they turn.
STYLE = {s: ("--" if "overshoot" in s else "-") for s in ORDER}

# The three channels the emulator is conditioned on. CO2 is assembled from its
# two components because the emulator's channel is total anthropogenic CO2.
PANELS = [
    ("CO2", ["CO2 FFI", "CO2 AFOLU"], "total anthropogenic CO$_2$",
     "Gt CO$_2$ yr$^{-1}$"),
    ("CO2cum", None, "cumulative CO$_2$ from 1850", "Gt CO$_2$"),
    ("Sulfur", ["Sulfur"], "SO$_2$ (the SUL channel)", "Mt SO$_2$ yr$^{-1}$"),
    ("BC", ["BC"], "black carbon", "Mt BC yr$^{-1}$"),
]
CUM_FROM = 1850


def load(path):
    """(years, {(scenario, variable): series}) from the wide CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    ycols = [c for c in df.columns if c.replace(".", "").isdigit()]
    # Columns are mid-year (1750.5); floor to the calendar year they describe.
    years = np.array([int(float(c)) for c in ycols])
    data = {}
    for _, r in df.iterrows():
        data[(r["scenario"], r["variable"])] = r[ycols].to_numpy(dtype=float)
    return years, data


def series(data, sc, parts):
    """Sum the component variables, or None if any is absent."""
    out = None
    for v in parts:
        s = data.get((sc, v))
        if s is None:
            return None
        out = s.copy() if out is None else out + s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--scenarios", nargs="+", default=ORDER)
    ap.add_argument("--year-start", type=int, default=1850)
    ap.add_argument("--year-end", type=int, default=2100,
                    help="2500 to show the post-2100 extensions")
    ap.add_argument("--out", default="plots/fair_cmip7_emissions")
    ap.add_argument("--dump-data", help="also write the plotted series as CSV")
    args = ap.parse_args()

    path = os.path.join(args.root, CSV)
    if not os.path.exists(path):
        print(f"[fair] not found: {path}", file=sys.stderr)
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years, data = load(path)
    print(f"[fair] {path}")
    print(f"[fair] {len(years)} years {years[0]}-{years[-1]}, "
          f"{len({k[0] for k in data})} scenarios")

    m = (years >= args.year_start) & (years <= args.year_end)
    yr = years[m]

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9.5, 3.4 * len(PANELS)),
                             sharex=True)
    rows = []
    for ax, (key, parts, title, unit) in zip(axes, PANELS):
        for sc in args.scenarios:
            if key == "CO2cum":
                # Cumulative from CUM_FROM, integrating the TOTAL CO2 — the
                # quantity the emulator's CO2 channel actually carries.
                tot = series(data, sc, ["CO2 FFI", "CO2 AFOLU"])
                if tot is None:
                    continue
                s = np.cumsum(np.where(years >= CUM_FROM, tot, 0.0))[m]
            else:
                tot = series(data, sc, parts)
                if tot is None:
                    print(f"  [skip] {sc}/{key}", file=sys.stderr)
                    continue
                s = tot[m]
            ax.plot(yr, s, color=COL.get(sc, "#555"), ls=STYLE.get(sc, "-"),
                    lw=1.8, label=f"{CODE.get(sc, '?')}  {sc}")
            rows += [(sc, key, int(y), float(v)) for y, v in zip(yr, s)]
        ax.axhline(0, color="k", lw=.8, alpha=.5)
        ax.set_ylabel(f"{title}\n{unit}", fontsize=9)
        ax.grid(alpha=.3)
        ax.axvline(2100, color="k", lw=.8, ls=":", alpha=.45)
        # Several pairs coincide exactly through 2100 — the extension and its
        # overshoot twin share a pathway and only separate afterwards (the
        # notebook literally copies high-overshoot onto high-extension before
        # the post-2100 CO2 overrides diverge them). Overlapping curves here
        # are the data, not a plotting fault; --year-end 2500 separates them.
    axes[0].legend(frameon=False, fontsize=7.5, ncol=1, loc="upper left")
    axes[-1].set_xlabel("year")
    axes[0].set_title("CMIP7 ScenarioMIP — ILLUSTRATIVE emissions driving FaIR "
                      "(protocol Fig. 1), not the gridded IAM set", fontsize=11)

    # State the discrepancy on the figure, with the measured number, so the
    # plot cannot be mistaken for the emulator's conditioning.
    h = series(data, "high-extension", ["CO2 FFI"])
    if h is not None and (years == 2100).any():
        v = float(h[years == 2100][0])
        axes[0].annotate(
            f"High FFI at 2100 = {v:.1f} Gt CO$_2$/yr here;\n"
            f"the gridded IIASA-IAMC set used by the emulator has 53.2\n"
            f"(illustrative pathway vs final IAM quantification)",
            xy=(2098, v), xytext=(1858, v * .34), fontsize=8, alpha=.9,
            arrowprops=dict(arrowstyle="->", lw=.8, alpha=.6))

    fig.text(0.01, 0.004,
             "Source: chrisroadmap/cmip7-scenariomip data/emissions/"
             "extensions_1750-2500.csv (van Vuuren et al. 2026, GMD 19:2627, "
             "Fig. 1; final IAM quantification published separately, Zenodo "
             "19825038). Overshoot pathways dashed; each coincides with its "
             "parent extension through 2100 and separates only after — "
             "use --year-end 2500 to see it.", fontsize=7, alpha=.65)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"[fair] wrote {args.out}.{ext}")

    if args.dump_data:
        import csv
        os.makedirs(os.path.dirname(args.dump_data) or ".", exist_ok=True)
        with open(args.dump_data, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "quantity", "year", "value"])
            w.writerows(rows)
        print(f"[fair] wrote {args.dump_data} ({len(rows)} rows)")

    print("\n  2100 values (illustrative):")
    print(f"  {'scenario':20s} {'CO2 tot':>9s} {'cum since 1850':>15s} "
          f"{'SO2':>8s} {'BC':>7s}")
    for sc in args.scenarios:
        tot = series(data, sc, ["CO2 FFI", "CO2 AFOLU"])
        if tot is None or not (years == 2100).any():
            continue
        i = int(np.where(years == 2100)[0][0])
        cum = float(np.cumsum(np.where(years >= CUM_FROM, tot, 0.0))[i])
        so2 = series(data, sc, ["Sulfur"])
        bc = series(data, sc, ["BC"])
        print(f"  {sc:20s} {tot[i]:9.2f} {cum:15.0f} "
              f"{so2[i]:8.2f} {bc[i]:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
