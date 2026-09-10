#!/usr/bin/env python3
"""
Every conditioning file in one figure: all three channels, all experiments.

One row per channel (CO2, BC, SUL), one line per cond file in --cond-dir. This
is the companion to plot_raw_vs_cond_emissions.py, which compares two of these
against the published input4MIPs data; here nothing is compared, the point is to
see what each EXPERIMENT actually carries -- including the three that figure
leaves out:

  aaer   aerosols vary, CO2 pinned to its 1850 value
  ghg    CO2 varies, BOTH aerosols pinned to 1850
  ramip  ssp370 CO2 with ssp126 aerosols, the hybrid used for the removal test

Values are plotted in Gt/Tg. The cond channels are stored as "Gt per gridpoint",
so summing the grid already gives Gt and only the aerosols need Gt -> Tg. They
still carry the pipeline's ~4.7x regrid deflation, so these are NOT real-world
totals -- see plot_raw_vs_cond_emissions.py for the side-by-side that measures it.

Usage:
    python scripts/plot_cond_files_emissions.py --cond-dir ~/data_staging/bc_rebuild
"""
import argparse
import glob
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# label -> filename stem, in the order they should be drawn and listed.
FILES = [
    ("historical",      "emissions_hist_only_timefixed_bc_co2fix.nc",              "0.25", 2.0),
    ("ssp370",          "emissions_ssp370_only_timefixed_bc_co2fix.nc",            "#c1121f", 1.5),
    ("ssp245",          "emissions_ssp245_only_timefixed_bc_co2fix.nc",            "#e07a00", 1.5),
    ("ssp126",          "emissions_ssp126_only_timefixed_bc_co2fix.nc",            "#0077b6", 1.5),
    ("aaer",            "emissions_aaer_only_timefixed_bc_co2fix.nc",              "#2a9d8f", 1.5),
    ("ghg",             "emissions_ghg_only_timefixed_bc_co2fix.nc",               "#7b2cbf", 1.5),
    ("ramip 370co2/126aer",
     "emissions_ssp370co2_ssp126aer_bc_2015-2079_co2fix.nc",                       "#b5179e", 1.5),
]
UNIT = {"CO2": ("Gt CO$_2$ (cumulative)", 1.0),
        "BC":  ("Tg BC / yr", 1e3),
        "SUL": ("Tg SO$_2$ / yr", 1e3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-dir", default=os.path.expanduser("~/data_staging/bc_rebuild"))
    ap.add_argument("--out", default="plots/cond_files_emissions.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    known = {f for _, f, _, _ in FILES}
    present = {os.path.basename(p) for p in glob.glob(os.path.join(args.cond_dir, "emissions_*.nc"))}
    for extra in sorted(present - known):
        print(f"  [note] not plotted (unknown to this script): {extra}")

    series = {}
    for label, fname, _, _ in FILES:
        path = os.path.join(args.cond_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] absent: {fname}")
            continue
        ds = xr.open_dataset(path)
        # hist/ssp/aaer/ghg carry `year`; the RAMIP hybrid carries `time`.
        coord = "year" if "year" in ds.coords else "time"
        yrs = np.asarray(ds[coord].values).astype(int)
        for sp in UNIT:
            if sp in ds.data_vars:
                series[(sp, label)] = (yrs, ds[sp].sum(("lat", "lon")).values * UNIT[sp][1])
        ds.close()

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, sp in zip(axes, ["CO2", "BC", "SUL"]):
        for label, _, color, lw in FILES:
            if (sp, label) not in series:
                continue
            y, v = series[(sp, label)]
            # ghg pins both aerosols and aaer pins CO2; a pinned channel is a flat
            # line that would otherwise look like a plotting bug, so say so.
            flat = np.allclose(v, v[0])
            ax.plot(y, v, color=color, lw=lw, alpha=0.9,
                    label=f"{label}  (pinned)" if flat else label,
                    ls=":" if flat else "-")
        ax.axvline(2015, color="r", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylabel(UNIT[sp][0])
        ax.set_title(f"{sp}", fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        ax.set_xlim(1850, 2100)
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    axes[2].set_xlabel("Year")
    # Several of these files SHARE a channel by construction -- aaer holds ssp370
    # aerosols, ghg holds ssp370 CO2, ramip holds ssp370 CO2 and ssp126 aerosols --
    # so lines sitting exactly on top of each other are agreement, not a bug.
    fig.suptitle(f"Conditioning channels, every file in {args.cond_dir}\n"
                 "Gt/Tg after the pipeline's ~4.7x regrid deflation — not real-world totals.  "
                 "Coincident lines share a channel by construction.",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
