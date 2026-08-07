#!/usr/bin/env python3
"""
Plot FaIR's CO2 FFI and CO2 AFOLU against the corresponding ESGF input4MIPs
gridded CO2, for the CMIP7 scenarios.

Pairing
-------
    FaIR `CO2 FFI`    <->  ESGF  CO2_em_anthro + CO2_em_AIR_anthro
                           (fossil fuel + industry, incl. aviation) — LIKE FOR LIKE
    FaIR `CO2 AFOLU`  <->  (nothing)

There is NO gridded AFOLU / land-use CO2 anywhere in CMIP7 input4MIPs — verified
against the ESGF variable facet: the only non-anthropogenic CO2 published is bare
`CO2` from DRES-CMIP-BB4CMIP7 (open biomass burning), which is historical-only
with no scenario counterpart, and is a component of AFOLU rather than AFOLU
itself. AFOLU is therefore plotted as the FaIR line alone, to show the magnitude
of what the gridded inventory the emulator sees omits.

File selection — the inputs4mips directory holds BOTH generations
-----------------------------------------------------------------
Used (CMIP7):      CEDS-CMIP-2025-04-18, IIASA-IAMC-{h,vl}-1-1-0
Deliberately NOT:  CEDS-CMIP-2024-10-21, CEDS-CMIP-2024-11-25,
                   IAMC-AIM-ssp370, IAMC-IMAGE-ssp126, IAMC-MESSAGE-GLOBIOM-ssp245
                   (these are the CMIP6-era files the old pipeline uses)

Scenario mapping, confirmed against the ScenarioMIP-CMIP7 protocol
(van Vuuren et al. 2026, GMD 19:2627 — official codes H/HL/M/ML/L/VL/LN):
    ESGF h  = H  (High)      -> FaIR high-extension
    ESGF vl = VL (Very Low)  -> FaIR verylow

Global totals use make_cmip7_cond.py's exact recipe on the RAW files: sum
sectors (or levels for aircraft) -> annual MEAN of the kg m-2 s-1 rate -> x cell
area x s/yr / 1e12. The raw files are used rather than the built cond files
because the cond files are deflated ~4.7x by bilinear regridding of a
per-gridpoint (extensive) quantity — see cond_regrid_extensive_deflation.

Usage
-----
    python scripts/plot_co2_ffi_afolu_vs_esgf.py \
        --raw-dir /home/nordling/mnt/lumi_sc2/emulator_data/emission_data/inputs4mips \
        --fair-emissions ~/Downloads/chrisroadmap-cmip7-scenariomip-3129623/output/emissions_adjusted.csv \
        --out plots/co2_ffi_afolu_vs_esgf.png --series-csv plots/co2_series.csv

    # reuse a previously dumped series (instant; no ~9 GB re-read)
    python scripts/plot_co2_ffi_afolu_vs_esgf.py --series-csv plots/co2_series.csv \
        --fair-emissions ... --out ...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_co2_emulator_vs_fair import from_raw          # noqa: E402

HIST_SOURCE = "CEDS-CMIP-2025-04-18"
SCEN_VERSION = "1-1-0"
# ESGF scenario -> (FaIR scenario, colour, label)
SCEN = {
    "h":  ("high-extension", "#d62728", "H (High)"),
    "vl": ("verylow",        "#2ca02c", "VL (Very Low)"),
}
HIST_END = 2023


def esgf_series(raw_dir, scenarios, y0, y1):
    """Global annual GtCO2/yr per scenario: CEDS hist <=2023 + IIASA >=2024."""
    print(f"[esgf] historical {HIST_SOURCE} …", flush=True)
    hist = from_raw(raw_dir, f"CO2-em-anthro_*CMIP_{HIST_SOURCE}_gn_*.nc", y0, HIST_END)
    if hist is None:
        print(f"ERROR: no CO2-em-anthro {HIST_SOURCE} files in {raw_dir}", file=sys.stderr)
        return None
    air = from_raw(raw_dir, f"CO2-em-AIR-anthro_*CMIP_{HIST_SOURCE}_gn_*.nc", y0, HIST_END)
    if air is None:
        print("ERROR: historical CO2-em-AIR-anthro files missing — FFI must include "
              "aviation to be comparable to FaIR's CO2 FFI", file=sys.stderr)
        return None
    hist = hist.add(air, fill_value=0)

    out = {}
    for sc in scenarios:
        src = f"IIASA-IAMC-{sc}-{SCEN_VERSION}"
        print(f"[esgf] scenario {sc} ({src}) …", flush=True)
        s = from_raw(raw_dir, f"CO2-em-anthro_*ScenarioMIP_{src}_gn_*.nc",
                     HIST_END + 1, y1)
        a = from_raw(raw_dir, f"CO2-em-AIR-anthro_*ScenarioMIP_{src}_gn_*.nc",
                     HIST_END + 1, y1)
        if s is None:
            print(f"  WARNING: no files for {src}; skipping")
            continue
        if a is not None:
            s = s.add(a, fill_value=0)
        # IIASA files are sparse (~17 sampled years over 2022-2100), not annual
        full = np.arange(int(s.index.min()), int(s.index.max()) + 1)
        if len(full) != len(s):
            print(f"  [{sc}] sparse ({len(s)} yrs) — interpolating to annual", flush=True)
            s = s.reindex(full).interpolate(method="index")
        out[sc] = pd.concat([hist, s]).sort_index()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot FaIR CO2 FFI / AFOLU against ESGF input4MIPs gridded CO2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw-dir", default=None,
                    help="inputs4mips dir (omit if --series-csv already exists)")
    ap.add_argument("--fair-emissions", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["h", "vl"])
    ap.add_argument("--year-min", type=int, default=1850)
    ap.add_argument("--year-max", type=int, default=2100)
    ap.add_argument("--out", default="plots/co2_ffi_afolu_vs_esgf.png")
    ap.add_argument("--series-csv", default=None,
                    help="Cache of the ESGF global series; read if it exists, "
                         "written after computing so re-plots are instant")
    args = ap.parse_args()

    # ── ESGF side (cached if available — the raw read is ~9 GB over sshfs) ──
    esgf = None
    if args.series_csv and os.path.exists(args.series_csv):
        df = pd.read_csv(args.series_csv)
        esgf = {sc: g.set_index("year")["GtCO2_per_yr"]
                for sc, g in df.groupby("scenario")}
        print(f"[esgf] reusing cached series from {args.series_csv} "
              f"({', '.join(esgf)})")
    if esgf is None:
        if not args.raw_dir:
            print("ERROR: need --raw-dir (no cached --series-csv found)", file=sys.stderr)
            return 1
        esgf = esgf_series(args.raw_dir, args.scenarios, args.year_min, args.year_max)
        if not esgf:
            return 1
        if args.series_csv:
            rows = [dict(scenario=sc, year=int(y), GtCO2_per_yr=float(v))
                    for sc, s in esgf.items() for y, v in s.items()]
            os.makedirs(os.path.dirname(os.path.abspath(args.series_csv)), exist_ok=True)
            pd.DataFrame(rows).to_csv(args.series_csv, index=False)
            print(f"[out] {args.series_csv}")

    # ── FaIR side ───────────────────────────────────────────────────────────
    fa = pd.read_csv(args.fair_emissions, index_col=0)
    ycols = [c for c in fa.columns if c != "Scenario"]
    fyears = np.floor(np.array([float(c) for c in ycols])).astype(int)

    def fair(scen, specie):
        r = fa[(fa.index == specie) & (fa.Scenario == scen)]
        if not len(r):
            return None
        s = pd.Series(r[ycols].values[0].astype(float), index=fyears)
        return s[(s.index >= args.year_min) & (s.index <= args.year_max)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    tab = []
    for i, sc in enumerate(args.scenarios):
        if sc not in esgf:
            continue
        fsc, col, lab = SCEN.get(sc, (sc, "#7f7f7f", sc))
        a = ax[i] if i < len(ax) else ax[-1]
        e = esgf[sc]
        e = e[(e.index >= args.year_min) & (e.index <= args.year_max)]
        ffi = fair(fsc, "CO2 FFI")
        afo = fair(fsc, "CO2 AFOLU")

        a.plot(e.index, e.values, color="k", lw=2.2,
               label="ESGF input4MIPs: anthro + aircraft\n(what the emulator sees)")
        if ffi is not None:
            a.plot(ffi.index, ffi.values, color=col, lw=1.8, ls="--",
                   label=f"FaIR CO$_2$ FFI ({fsc})")
        if afo is not None:
            a.plot(afo.index, afo.values, color=col, lw=1.4, ls=":",
                   label="FaIR CO$_2$ AFOLU\n(NO gridded counterpart exists)")
            a.fill_between(afo.index, 0, afo.values, color=col, alpha=0.12, lw=0)

        a.set_title(f"{sc}  =  {lab}")
        a.set_xlabel("year")
        a.axhline(0, ls=":", color="k", lw=0.5)
        a.grid(alpha=0.3)
        a.legend(fontsize=7.5, loc="upper left")
        a.set_xlim(args.year_min, args.year_max)

        for yr in (2020, 2050, 2100):
            if yr in e.index and ffi is not None and yr in ffi.index:
                tab.append(dict(scenario=sc, fair_scenario=fsc, year=yr,
                                esgf_anthro_air=round(float(e.loc[yr]), 3),
                                fair_ffi=round(float(ffi.loc[yr]), 3),
                                fair_afolu=(round(float(afo.loc[yr]), 3)
                                            if afo is not None and yr in afo.index
                                            else np.nan),
                                ratio_esgf_over_ffi=round(float(e.loc[yr] / ffi.loc[yr]), 4)
                                if float(ffi.loc[yr]) else np.nan))

    ax[0].set_ylabel("CO$_2$ emissions, GtCO$_2$ yr$^{-1}$")
    fig.suptitle("FaIR CO$_2$ FFI / AFOLU  vs  ESGF input4MIPs gridded CO$_2$ — CMIP7\n"
                 "ESGF = CEDS-CMIP-2025-04-18 (hist, ≤2023) + IIASA-IAMC-{h,vl}-1-1-0 "
                 "(≥2024), anthro + aircraft", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")

    if tab:
        t = pd.DataFrame(tab)
        print("\nGtCO2/yr — ESGF (anthro+air) vs FaIR FFI, with FaIR AFOLU for scale")
        print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
