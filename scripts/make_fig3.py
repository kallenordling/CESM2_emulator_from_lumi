#!/usr/bin/env python3
"""
================================================================================
 FIGURE 3 — distribution of global means over each experiment's last 20 years
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig3.py

Everything configurable is in the SETTINGS block below. There are no
command-line options and no helper functions: the script runs top to bottom in
seven numbered steps, so it can be read as a description of how the figure is
made. Same shape as make_fig1.py and make_fig2.py.

WHAT THE FIGURE SHOWS
---------------------
One panel per experiment and variable. Each histogram pools EVERY member-year
of the last 20 years — 25 members x 20 years = 500 values for the emulator, and
6 to 11 members x 20 years for CESM2 — and asks whether the two samples are
drawn from the same distribution.

WHY POOL MEMBER-YEARS
---------------------
The timeseries figures compare ENSEMBLE MEANS, which average internal
variability away and so test the forced response. This tests something the mean
cannot: the SPREAD. A diffusion model can reproduce a trajectory perfectly while
generating too little year-to-year variability, and the two figures together
separate those questions — figure 1 asks whether the emulator warms correctly,
this one asks whether its climate fluctuates correctly.

Values are ABSOLUTE global means, not anomalies. A histogram of anomalies would
hide any offset in the mean state, which is half of what a distribution
comparison is for.

WHAT TO WATCH FOR
-----------------
Three ways the distributions can differ, and they mean different things:
  * shifted    — a bias in the mean state (the same thing figure 1 reports)
  * too narrow — under-dispersion: too little internal variability
  * different shape — skew or tails the emulator does not reproduce

The last 20 years still contain a warming trend in the forced scenarios, so
part of every distribution's width is that trend rather than internal
variability. It affects both sides identically, so the comparison stays fair,
but do not read the width as internal variability alone.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

# The emulator's evaluation output: one NetCDF per scenario, holding
# {var}_model with dims (member, year, lat, lon).
EVAL_DIR = "/home/nordling/mnt/lumi_sc/eval_output/manual/ep0860_ens25_absolute"

# The CESM2 reference, built by scripts/make_cesm2_reference.py: one NetCDF per
# variable and scenario, holding {var}_cesm with the same dimensions. Held-out
# members only, already in the emulator's units.
REFERENCE_DIR = "/home/nordling/mnt/lumi_sc/emulator_data/cesm2_reference"

OUT = "plots/fig3.png"                 # the .pdf sibling is written alongside

# LaTeX table of the distribution statistics, for \input into the paper.
TABLE = "plots/fig3_distributions.tex"

# How many years at the END of each experiment go into the histogram.
N_YEARS = 20

# Variables, with the unit shown on the axis and in the table.
VARIABLES = {"TREFHT": ("Temperature", "$^{\\circ}$C", "degC"),
             "PRECT":  ("Precipitation", "mm day$^{-1}$", "mm/day")}

# The experiments, in plotting order: key -> (label, colour).
# Okabe-Ito colours, distinguishable in greyscale and to colour-blind readers.
SCENARIOS = {
    "hist":   ("Historical",                "#0072B2"),
    "ssp370": ("SSP3-7.0",                  "#D55E00"),
    "aaer":   ("Aerosol-only (AAER)",       "#E69F00"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "#009E73"),
}

# Histogram resolution. Both sides share one set of bin edges per panel, or the
# shapes cannot be compared.
N_BINS = 24

# =============================================================================

import os

import matplotlib
matplotlib.use("Agg")                  # write files; no display needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

# =============================================================================
#  STEP 1 — collect the emulator's global means for the last N years
# =============================================================================
# The eval file holds maps, so the global mean is computed here: cos(lat)-
# weighted, because grid cells shrink towards the poles and an unweighted mean
# over this grid is wrong by degrees, not decimals.
#
# `.sel(year=slice(-N_YEARS, None))` would select YEARS -20 to 0, which do not
# exist; the last N years are an INDEX selection, hence isel.

emulator = {}          # (variable, scenario) -> 1-D array of member-years
for variable in VARIABLES:
    for scenario in SCENARIOS:
        path = f"{EVAL_DIR}/{variable}_{scenario}.nc"
        dataset = xr.open_dataset(path)
        field = dataset[f"{variable}_model"].isel(year=slice(-N_YEARS, None))
        weights = np.cos(np.deg2rad(field["lat"]))
        global_mean = field.weighted(weights).mean(("lat", "lon")).compute()
        years = global_mean["year"].values
        # Flatten (member, year) into one sample: every member-year is one draw
        # from the model's climate over this window.
        emulator[(variable, scenario)] = global_mean.values.ravel()
        dataset.close()
        print(f"[step 1] {variable:6s} {scenario:7s} emulator: "
              f"{global_mean.sizes['member']:2d} members x {len(years)} years "
              f"({years[0]}-{years[-1]}) = {emulator[(variable, scenario)].size} values")

# =============================================================================
#  STEP 2 — the same for CESM2
# =============================================================================
# One file per variable and scenario, already restricted to HELD-OUT members
# and converted to the emulator's units by make_cesm2_reference.py. Scoring
# against members the emulator trained on would be marking its own homework;
# that filtering happened when the reference was built.

cesm = {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        path = f"{REFERENCE_DIR}/{variable}_{scenario}.nc"
        dataset = xr.open_dataset(path)
        field = dataset[f"{variable}_cesm"].isel(year=slice(-N_YEARS, None))
        weights = np.cos(np.deg2rad(field["lat"]))
        global_mean = field.weighted(weights).mean(("lat", "lon")).compute()
        years = global_mean["year"].values
        cesm[(variable, scenario)] = global_mean.values.ravel()
        member_count = global_mean.sizes["member"]
        dataset.close()
        print(f"[step 2] {variable:6s} {scenario:7s} CESM2:    "
              f"{member_count:2d} members x {len(years)} years "
              f"({years[0]}-{years[-1]}) = {cesm[(variable, scenario)].size} values")

# =============================================================================
#  STEP 3 — describe each distribution
# =============================================================================
# Mean and standard deviation, plus the 5th and 95th percentiles because a
# distribution can match in the middle and differ in the tails.
#
# The RATIO of standard deviations is the number the figure exists to produce:
# below 1 means the emulator generates too little variability, which is the
# characteristic failure of a generative model that has learned the mean state
# but not the noise around it.

statistics = {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        emulator_values = emulator[(variable, scenario)]
        cesm_values = cesm[(variable, scenario)]
        statistics[(variable, scenario)] = dict(
            emulator_mean=float(np.mean(emulator_values)),
            cesm_mean=float(np.mean(cesm_values)),
            emulator_sd=float(np.std(emulator_values, ddof=1)),
            cesm_sd=float(np.std(cesm_values, ddof=1)),
            emulator_p05=float(np.percentile(emulator_values, 5)),
            cesm_p05=float(np.percentile(cesm_values, 5)),
            emulator_p95=float(np.percentile(emulator_values, 95)),
            cesm_p95=float(np.percentile(cesm_values, 95)),
        )
        row = statistics[(variable, scenario)]
        row["mean_difference"] = row["emulator_mean"] - row["cesm_mean"]
        row["sd_ratio"] = row["emulator_sd"] / row["cesm_sd"]
        print(f"[step 3] {variable:6s} {scenario:7s} "
              f"mean {row['emulator_mean']:8.3f} vs {row['cesm_mean']:8.3f} "
              f"(diff {row['mean_difference']:+.3f}), "
              f"sd {row['emulator_sd']:.3f} vs {row['cesm_sd']:.3f} "
              f"(ratio {row['sd_ratio']:.2f})")

# =============================================================================
#  STEP 4 — lay out the figure
# =============================================================================
# One row per variable, one column per experiment.

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
figure, axes = plt.subplots(len(VARIABLES), len(SCENARIOS),
                            figsize=(4.0 * len(SCENARIOS), 3.2 * len(VARIABLES)),
                            squeeze=False)

# =============================================================================
#  STEP 5 — draw the histograms
# =============================================================================
# Both sides share one set of bin edges per panel, spanning the combined range:
# with different edges the two shapes would not be comparable.
#
# density=True, not counts. The emulator contributes 500 member-years and CESM2
# between 120 and 220, so raw counts would show the sample sizes rather than the
# distributions.

EMULATOR_COLOUR = "#D55E00"
CESM_COLOUR = "#0072B2"

for row_index, variable in enumerate(VARIABLES):
    variable_label, unit_axis, _ = VARIABLES[variable]
    for column_index, scenario in enumerate(SCENARIOS):
        axis = axes[row_index][column_index]
        emulator_values = emulator[(variable, scenario)]
        cesm_values = cesm[(variable, scenario)]

        low = min(emulator_values.min(), cesm_values.min())
        high = max(emulator_values.max(), cesm_values.max())
        bin_edges = np.linspace(low, high, N_BINS + 1)

        axis.hist(cesm_values, bins=bin_edges, density=True,
                  color=CESM_COLOUR, alpha=0.45, label="CESM2")
        axis.hist(emulator_values, bins=bin_edges, density=True,
                  histtype="step", color=EMULATOR_COLOUR, lw=2.0,
                  label="Emulator")

        # The means, so a shift is readable even where the shapes overlap.
        axis.axvline(np.mean(cesm_values), color=CESM_COLOUR, lw=1.4, ls="--")
        axis.axvline(np.mean(emulator_values), color=EMULATOR_COLOUR,
                     lw=1.4, ls="--")

        row = statistics[(variable, scenario)]
        axis.set_title(f"{SCENARIOS[scenario][0]}\n"
                       f"$\\Delta$mean {row['mean_difference']:+.3f}, "
                       f"sd ratio {row['sd_ratio']:.2f}",
                       fontsize=9, loc="left")
        axis.set_xlabel(f"{variable_label} ({unit_axis})")
        if column_index == 0:
            axis.set_ylabel("Probability density")
        axis.text(0.03, 0.95, f"({'abcdefgh'[row_index * len(SCENARIOS) + column_index]})",
                  transform=axis.transAxes, fontweight="bold", va="top", fontsize=9)

# =============================================================================
#  STEP 6 — legend and title
# =============================================================================

legend_entries = [
    Patch(facecolor=CESM_COLOUR, alpha=0.45, label="CESM2 (held-out members)"),
    Line2D([], [], color=EMULATOR_COLOUR, lw=2.0, label="Emulator"),
    Line2D([], [], color="0.35", lw=1.4, ls="--", label="distribution mean"),
]
# tight_layout FIRST, then the title and legend are placed relative to the
# settled axes; doing it the other way round lets tight_layout move the axes out
# from under them, which is how the legend ends up on top of the title.
figure.tight_layout()
figure.suptitle(
    f"Global-mean distributions over the last {N_YEARS} years of each experiment"
    f"\nevery member-year pooled; shared bins per panel; densities, not counts",
    fontsize=10.5, y=1.10)
legend = figure.legend(handles=legend_entries, frameon=False, ncols=3,
                       loc="lower center", bbox_to_anchor=(0.5, 1.005))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
for path in (OUT, os.path.splitext(OUT)[0] + ".pdf"):
    figure.savefig(path, bbox_inches="tight", bbox_extra_artists=[legend])
    print(f"[step 6] wrote {path}")

# =============================================================================
#  STEP 7 — the same numbers as a LaTeX table
# =============================================================================
# Plain LaTeX with borders: \toprule and \cmidrule need \usepackage{booktabs},
# and without it \cmidrule silently typesets "(lr)2-3" into the table.

table_rows = []
for variable in VARIABLES:
    unit_table = VARIABLES[variable][2]
    for scenario in SCENARIOS:
        row = statistics[(variable, scenario)]
        table_rows.append(
            f"{VARIABLES[variable][0]} & {SCENARIOS[scenario][0]} & "
            f"{row['emulator_mean']:.3f} & {row['cesm_mean']:.3f} & "
            f"{row['mean_difference']:+.3f} & "
            f"{row['emulator_sd']:.3f} & {row['cesm_sd']:.3f} & "
            f"{row['sd_ratio']:.2f} \\\\")

table_tex = "\n".join([
    r"\begin{tabular}{|l|l|r|r|r|r|r|r|}",
    r"\hline",
    r"\textbf{Variable} & \textbf{Experiment} & "
    r"\multicolumn{3}{c|}{\textbf{Mean}} & "
    r"\multicolumn{3}{c|}{\textbf{Standard deviation}} \\",
    r"\cline{3-8}",
    r" & & Emulator & CESM2 & Difference & Emulator & CESM2 & Ratio \\",
    r"\hline",
    *table_rows,
    r"\hline",
    r"\end{tabular}",
])

os.makedirs(os.path.dirname(TABLE) or ".", exist_ok=True)
with open(TABLE, "w") as handle:
    handle.write(
        f"% Global-mean distributions over the last {N_YEARS} years of each\n"
        f"% experiment, pooling every member-year: emulator 25 members,\n"
        f"% CESM2 6-11 held-out members. Units are degC for temperature and\n"
        f"% mm/day for precipitation. An sd ratio below 1 means the emulator\n"
        f"% generates too little variability.\n"
        f"% Generated by scripts/make_fig3.py — do not edit by hand.\n")
    handle.write(table_tex + "\n")
print(f"[step 7] wrote {TABLE}")
