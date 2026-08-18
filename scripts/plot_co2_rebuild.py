#!/usr/bin/env python3
"""
Document the ssp370 CO2 fix: raw annual emissions vs the cumulative channel.

Two panels, because the bug is invisible in one of them and obvious in the other:

  TOP    annual CO2 emissions on the NATIVE grid — the raw input4MIPs data as
         make_co2_files.py leaves it. All scenarios agree at 2015 and diverge
         slowly. Nothing wrong here, which is the point: the sources were never
         the problem.

  BOTTOM cumulative CO2 as the emulator receives it, ssp370 shown BOTH ways.
         The shipped file accumulated at ~1.96x the correct rate; the rebuilt
         one lands on ssp126's 2015 value exactly, as a shared branch point
         requires.

Usage:
    python scripts/plot_co2_rebuild.py
    python scripts/plot_co2_rebuild.py --stage DIR --out plots/co2_rebuild
"""
import argparse
import os
import sys

import numpy as np

STAGE = os.path.expanduser("~/data_staging/bc_rebuild")
MOUNT = "/home/nordling/mnt/lumi_sc2/emulator_data"

COL = {"ssp370": "#c0392b", "ssp126": "#2471a3", "ssp245": "#e08e0b",
       "hist": "#444444"}
NICE = {"ssp370": "SSP3-7.0", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5"}


def gsum(path, var="CO2"):
    """Global sum per year. These are per-gridpoint extensive fields, so a
    plain sum is the global total."""
    import xarray as xr
    if not os.path.exists(path):
        return None, None
    ds = xr.open_dataset(path)
    c = "year" if "year" in ds.coords else "time"
    v = var if var in ds else list(ds.data_vars)[0]
    y = np.asarray(ds[c].values, dtype=int)
    g = ds[v].sum(dim=("lat", "lon")).values
    ds.close()
    return y, g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=STAGE)
    ap.add_argument("--mount", default=MOUNT)
    ap.add_argument("--out", default="plots/co2_rebuild")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S, M = args.stage, args.mount
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)

    # ── raw annual, native grid ──────────────────────────────────────────────
    yh, gh = gsum(f"{S}/CO2_cumulative_Gt_per_gridpoint_hist.nc")
    if yh is not None:
        m = yh <= 2014          # the file runs to 2022; the splice is at 2014
        ax1.plot(yh[m], gh[m], color=COL["hist"], lw=2, label="historical (CEDS)")
    for sc in ("ssp370", "ssp126", "ssp245"):
        for base in (S, M):
            y, g = gsum(f"{base}/CO2_cumulative_Gt_per_gridpoint_{sc}.nc")
            if y is not None:
                # decadal samples: mark them, so "10 points for 86 years" is
                # visible rather than implied
                ax1.plot(y, g, color=COL[sc], lw=1.6, marker="o", ms=3.5,
                         label=f"{NICE[sc]} (decadal)")
                break
    ax1.set_ylabel("annual CO$_2$ emissions\nGt CO$_2$ yr$^{-1}$ (native grid)")
    ax1.set_title("Raw input4MIPs CO$_2$ — the sources were never wrong")
    ax1.grid(alpha=.3)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax1.axvline(2015, color="k", lw=.8, ls=":", alpha=.5)
    ax1.annotate("all scenarios share 2015:\n34.90 anthro + 0.76 aircraft",
                 xy=(2015, 35.7), xytext=(1875, 30), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=.8, alpha=.6), alpha=.8)

    # ── cumulative, cond files ───────────────────────────────────────────────
    yh2, gh2 = gsum(f"{S}/emissions_hist_only_timefixed_bc.nc")
    if yh2 is not None:
        ax2.plot(yh2, gh2, color=COL["hist"], lw=2, label="historical")
    pairs = [
        ("ssp370", f"{S}/emissions_ssp370_only_timefixed_bc_co2fix.nc", "-", 2.2,
         "SSP3-7.0 REBUILT"),
        ("ssp370", f"{S}/emissions_ssp370_only_timefixed_bc.nc", "--", 1.6,
         "SSP3-7.0 shipped (1.96x)"),
        ("ssp126", f"{M}/emissions_ssp126_only_timefixed_co2fix.nc", "-", 1.8, None),
        ("ssp245", f"{M}/emissions_ssp245_only_timefixed.nc", "-", 1.8, None),
    ]
    for sc, p, ls, lw, lab in pairs:
        y, g = gsum(p)
        if y is None:
            print(f"  [skip] {p}", file=sys.stderr)
            continue
        ax2.plot(y, g, color=COL[sc], lw=lw, ls=ls, label=lab or NICE[sc])
    ax2.set_ylabel("cumulative CO$_2$\nGt CO$_2$ (emulator input space)")
    ax2.set_xlabel("year")
    ax2.set_title("Cumulative CO$_2$ as the emulator receives it")
    ax2.grid(alpha=.3)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.axvline(2015, color="k", lw=.8, ls=":", alpha=.5)
    ax2.annotate("rebuilt lands on 317.39,\nidentical to SSP1-2.6",
                 xy=(2015, 317), xytext=(2020, 900), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=.8, alpha=.6), alpha=.85)

    fig.text(0.01, 0.005,
             "Cumulative panel is in emulator input space: the extensive regrid "
             "deflates totals ~4.7x vs the native grid above.",
             fontsize=7, alpha=.65)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"[co2] wrote {args.out}.{ext}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
