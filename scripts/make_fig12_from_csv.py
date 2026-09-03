#!/usr/bin/env python3
"""
================================================================================
 FIGURES 1 AND 2, REBUILT FROM THE EXPORTED CSVs
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig12_from_csv.py

Everything configurable is in the SETTINGS block below. No command-line options
and no helper functions: the script runs top to bottom in nine numbered steps.

WHAT THIS IS, AND WHY IT EXISTS
-------------------------------
scripts/make_fig1.py and scripts/make_fig2.py build the same two figures from
the RAW output — the evaluation NetCDFs and the CESM2 training trees — reading
several GB of maps over a network mount and computing the global means here.
That is the authoritative path, and it is slow: minutes at best, and on a
degraded sshfs mount it does not finish at all.

This script instead reads the SIXTEEN CSVs that scripts/make_fig12_csv.py
exported, which already hold exactly the numbers those figures plot: one file
per variable, experiment and side, years as rows and ensemble members as
columns. 280 KB in total, so it runs in about a second on any machine.

Two consequences worth knowing:

  * It reproduces the figures WITHOUT the raw data, so anyone with the repo can
    redraw them, and the numbers in the paper are traceable to a file a reader
    can open.
  * It cannot notice anything the export dropped. The CSVs are already
    reduced — global means, held-out members, member counts matched. Change any
    of that and it must be changed in make_fig12_csv.py and re-exported; the
    settings below cannot recover what is not in the files.

The figures themselves are the same as make_fig1/make_fig2 produce, and the
docstrings there explain the design. In brief:

Panel (a): the anomaly vs 1850-1900, four experiments, emulator (solid) against
held-out CESM2 (dashed with circles), with CESM2's member range shaded.

Panels (b-d): the bias — emulator ensemble mean minus CESM2 ensemble mean — over
a grey band showing the full range of CESM2's own members about their mean. A
bias line inside that band is indistinguishable from internal variability.

PRECIPITATION IS SHOWN AS A PERCENTAGE, not an absolute rate: a few hundredths
of a mm/day means nothing without the ~2.9 mm/day it is relative to. Figure 2's
panels (b)-(d) are therefore in PERCENTAGE POINTS.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

# Where scripts/make_fig12_csv.py wrote its output. Expected inside:
#     <variable>_<scenario>_<side>.csv   years as rows, members as columns
#     baselines.csv                      each side's own 1850-1900 mean
DATA_DIR = "plots/fig12_data"

OUT = "plots/{name}.png"          # the .pdf sibling is written alongside
TABLE = "plots/{name}_skill.tex"  # LaTeX table of the same numbers

# The two figures. key -> (output name, label, panel (a) y-label, table unit,
#                          whether the anomaly is a PERCENTAGE of the baseline)
VARIABLES = {
    "TREFHT": ("fig1", "Temperature",
               "GMST anomaly (°C, vs 1850–1900)", r"$^{\circ}$C", False),
    "PRECT":  ("fig2", "Precipitation",
               "Precipitation change (%, vs 1850–1900)", r"\%-points", True),
}

# The four experiments, in plotting order: key -> (legend label, colour).
# Colours are Okabe-Ito, which stay distinguishable in greyscale and to
# colour-blind readers.
SCENARIOS = {
    "hist":   ("Historical",                "#0072B2"),
    "ssp370": ("SSP3-7.0",                  "#D55E00"),
    "aaer":   ("Aerosol-only (AAER)",       "#E69F00"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "#009E73"),
}

BASELINE = (1850, 1900)   # the anomaly reference window, for the shaded span
YEAR_MAX = 2100

# =============================================================================

import os

import matplotlib
matplotlib.use("Agg")                      # no display needed; write files only
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

print(__doc__.split("WHAT THIS IS")[0])

# =============================================================================
#  STEP 1 — read the CSVs
# =============================================================================
# One file per variable, experiment and side. Rows are years, columns are
# ensemble members, values are ABSOLUTE global means — degrees Celsius and
# mm/day — cos(lat)-weighted when they were exported.
#
# Nothing is selected or filtered here. The export already restricted CESM2 to
# HELD-OUT members and capped the emulator to the same count per experiment, so
# every column in every file belongs in the figure.

series = {}          # (variable, scenario, side) -> DataFrame(year x member)
for variable in VARIABLES:
    for scenario in SCENARIOS:
        for side in ("emulator", "cesm2"):
            path = f"{DATA_DIR}/{variable}_{scenario}_{side}.csv"
            series[(variable, scenario, side)] = pd.read_csv(path,
                                                             index_col="year")
        print(f"[step 1] {variable:6s} {scenario:7s} "
              f"emulator {series[(variable, scenario, 'emulator')].shape}, "
              f"CESM2 {series[(variable, scenario, 'cesm2')].shape}  "
              f"(year x member)")

# baselines.csv carries each side's own 1850-1900 mean, which is what turns the
# absolute values above into the anomalies the figures plot. It is read rather
# than recomputed because ssp370 begins in 2015 and has no baseline period of
# its own: the export gave it the historical one, on both sides, and that
# convention has to survive the round trip.
baseline_frame = pd.read_csv(f"{DATA_DIR}/baselines.csv")
baseline = {(row.variable, row.scenario, row.side): row.baseline
            for row in baseline_frame.itertuples()}
print(f"[step 1] {len(baseline)} baselines from {DATA_DIR}/baselines.csv")

# =============================================================================
#  STEP 2 — absolute values become anomalies
# =============================================================================
# Each side is referenced to ITS OWN pre-industrial. The emulator's absolute
# climate is close to CESM2's but not identical, and a constant offset between
# them is not what these figures are about: subtracting each side's own baseline
# removes it and leaves the RESPONSE, which is the quantity being compared.
#
# For precipitation the baseline is the DENOMINATOR as well, not just an offset:
# the anomaly is expressed as a percentage of that side's own ~2.9 mm/day, so a
# small difference in the mean state cannot masquerade as a difference in
# response.

anomaly = {}
for variable, (_, _, _, _, as_percent) in VARIABLES.items():
    for scenario in SCENARIOS:
        for side in ("emulator", "cesm2"):
            base = baseline[(variable, scenario, side)]
            values = series[(variable, scenario, side)] - base
            if as_percent:
                values = 100.0 * values / base
            anomaly[(variable, scenario, side)] = values

# =============================================================================
#  STEP 3 — the numbers, on the years both sides cover
# =============================================================================
# The ranges differ — CESM2's aaer and ghg trees stop in 2050 while the emulator
# runs to 2100 — so every statistic below is computed on the intersection.
#
# The bias is the difference of the two ENSEMBLE MEANS. The band it is judged
# against is the full min-to-max range of CESM2's own members about their mean:
# where the bias sits inside it, the emulator differs from CESM2 by no more than
# the most extreme pair of CESM2 runs differ from each other.

stats, bias_series = {}, {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        emulator_frame = anomaly[(variable, scenario, "emulator")]
        cesm_frame = anomaly[(variable, scenario, "cesm2")]

        years = emulator_frame.index.intersection(cesm_frame.index)
        years = years[years <= YEAR_MAX]

        emulator_mean = emulator_frame.loc[years].mean(axis=1)
        cesm_mean = cesm_frame.loc[years].mean(axis=1)
        difference = emulator_mean - cesm_mean

        # CESM2's members as deviations from their own mean, per year.
        deviation = cesm_frame.loc[years].sub(cesm_mean, axis=0)
        spread_low = deviation.min(axis=1)
        spread_high = deviation.max(axis=1)

        bias_series[(variable, scenario)] = (years.values, difference,
                                             spread_low, spread_high)
        stats[(variable, scenario)] = dict(
            n_emu=emulator_frame.shape[1], n_cesm=cesm_frame.shape[1],
            bias=float(difference.mean()),
            rmse=float(np.sqrt((difference ** 2).mean())),
            # Correlation of the two ENSEMBLE-MEAN series. High values here are
            # mostly the shared forced trend, so read it next to the bias
            # rather than on its own.
            corr=float(np.corrcoef(emulator_mean, cesm_mean)[0, 1]),
            inside=float(((difference >= spread_low)
                          & (difference <= spread_high)).mean()) * 100)
        row = stats[(variable, scenario)]
        print(f"[step 3] {variable:6s} {scenario:7s} r {row['corr']:.3f}, "
              f"bias {row['bias']:+.3f}, rmse {row['rmse']:.3f}, "
              f"{row['inside']:.0f}% of years within CESM2's own spread")

# =============================================================================
#  STEP 4 — one figure per variable
# =============================================================================
# One wide overview panel on top, and beneath it one bias panel per experiment.
# hist and ssp370 share a bias panel because they are one continuous
# trajectory: hist ends in 2014, ssp370 begins in 2015.

bias_panels = [(("hist", "ssp370"), "Historical + SSP3-7.0"),
               (("aaer",), SCENARIOS["aaer"][0]),
               (("ghg",),  SCENARIOS["ghg"][0])]

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

for variable, (name, label, y_label, unit_tex, as_percent) in VARIABLES.items():
    fig = plt.figure(figsize=(9.5, 7.6))
    grid = fig.add_gridspec(2, len(bias_panels), height_ratios=[2.3, 1.0],
                            hspace=0.30, wspace=0.12)
    ax_main = fig.add_subplot(grid[0, :])
    ax_bias = [fig.add_subplot(grid[1, i]) for i in range(len(bias_panels))]

    # -------------------------------------------------------------------------
    #  STEP 5 — draw panel (a), one experiment at a time
    # -------------------------------------------------------------------------
    # Per experiment: CESM2's member range as shading, CESM2's mean as a dashed
    # line with open circles, and the emulator's mean as a thick solid line.
    #
    # The circles matter. Solid-vs-dashed in the same colour is unreadable where
    # the two curves coincide, which is most of the record — a marker shape
    # survives overlap, greyscale and print size.

    for scenario, (scenario_label, colour) in SCENARIOS.items():
        cesm_frame = anomaly[(variable, scenario, "cesm2")]
        cesm_frame = cesm_frame.loc[cesm_frame.index <= YEAR_MAX]
        emulator_frame = anomaly[(variable, scenario, "emulator")]
        emulator_frame = emulator_frame.loc[emulator_frame.index <= YEAR_MAX]

        cesm_years = cesm_frame.index.values
        cesm_mean = cesm_frame.mean(axis=1)
        low, high = cesm_frame.min(axis=1), cesm_frame.max(axis=1)

        ax_main.fill_between(cesm_years, low, high, color=colour, alpha=0.26,
                             lw=0, zorder=1)
        for edge in (low, high):                  # a thin edge pins the band
            ax_main.plot(cesm_years, edge, color=colour, lw=0.7, alpha=0.55,
                         zorder=1)
        ax_main.plot(cesm_years, cesm_mean, color=colour, lw=1.2, ls="--",
                     marker="o", markersize=3.4, markevery=8,
                     markerfacecolor="white", markeredgecolor=colour, zorder=5,
                     path_effects=[pe.withStroke(linewidth=3.0,
                                                 foreground="white")])
        ax_main.plot(emulator_frame.index.values, emulator_frame.mean(axis=1),
                     color=colour, lw=2.6, zorder=4, label=scenario_label)

    # -------------------------------------------------------------------------
    #  STEP 6 — draw the bias panels
    # -------------------------------------------------------------------------
    # The grey band is the min-to-max range of CESM2's members about their own
    # mean, per year. Where the coloured bias line sits inside it, the emulator
    # differs from CESM2 by no more than one CESM2 member differs from another.

    for i, (group, title) in enumerate(bias_panels):
        ax = ax_bias[i]
        ax.axhline(0, lw=0.8, color="0.3", zorder=1)
        for scenario in group:
            years, difference, spread_low, spread_high = \
                bias_series[(variable, scenario)]
            ax.fill_between(years, spread_low, spread_high,
                            color="0.45", alpha=0.22, lw=0, zorder=0)
            ax.plot(years, difference, color=SCENARIOS[scenario][1], lw=1.4,
                    zorder=3)
        ax.set_title(
            f"{title}\nn = "
            f"{'/'.join(str(stats[(variable, k)]['n_emu']) for k in group)} "
            f"emulator, "
            f"{'/'.join(str(stats[(variable, k)]['n_cesm']) for k in group)} "
            f"CESM2", fontsize=9, loc="left", pad=4)
        ax.text(0.02, 0.94, f"({'bcd'[i]})", transform=ax.transAxes,
                fontweight="bold", va="top", fontsize=9)
        ax.set_xlabel("Year")
        if i == 0:
            ax.set_ylabel(("Bias (percentage points)" if as_percent
                           else "Bias (degC)") + "\nensemble means")
        else:
            # All three panels share one y-scale, so repeating the tick labels
            # adds nothing and collides with the neighbouring panel's text.
            ax.tick_params(labelleft=False)

    # Every bias panel gets ONE y-scale, so a bias in one is the same size on
    # the page as a bias in another, and ONE x-range — the same 1850-2100 as
    # panel (a), so the columns line up under it. aaer and ghg end in 2050 and
    # simply stop there, which is honest: the run is short, not the bias small.
    limit = 1.15 * max(
        max(abs(float(d.min())), abs(float(d.max())),
            abs(float(lo.min())), abs(float(hi.max())))
        for (v, _), (_, d, lo, hi) in bias_series.items() if v == variable)
    for ax in ax_bias:
        ax.set_ylim(-limit, limit)
        ax.set_xlim(BASELINE[0], YEAR_MAX)
        # Prune the ticks at both ends: with three panels side by side spanning
        # the same years, a label on the right edge of one lands on the label at
        # the left edge of the next.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both",
                                               steps=[1, 5, 10]))

    # -------------------------------------------------------------------------
    #  STEP 7 — legends, and finish panel (a)
    # -------------------------------------------------------------------------
    # Two legends, stacked above the panel so they cover no data: the scenario
    # colours, and what the line styles and shadings mean.

    style = [
        Line2D([], [], color="0.35", lw=2.6, label="EMULATOR — ensemble mean"),
        Line2D([], [], color="0.35", lw=1.2, ls="--", marker="o",
               markersize=3.4, markerfacecolor="white", markeredgecolor="0.35",
               label="CESM2 — held-out ensemble mean"),
        Patch(facecolor="0.35", alpha=0.26,
              label="CESM2 member range (min–max)"),
    ]
    legend_scenarios = ax_main.legend(frameon=False, ncols=4, loc="lower left",
                                      bbox_to_anchor=(0.0, 1.14),
                                      handlelength=2.2)
    ax_main.add_artist(legend_scenarios)
    legend_style = ax_main.legend(handles=style, frameon=False, ncols=2,
                                  fontsize=8.2, loc="lower left",
                                  bbox_to_anchor=(0.0, 1.005), handlelength=2.6)

    ax_main.axhline(0, ls=":", lw=0.8, color="0.3")
    ax_main.axvspan(*BASELINE, color="0.9", alpha=0.6, lw=0, zorder=0)
    ax_main.set_ylabel(y_label)
    ax_main.set_xlabel("Year")
    ax_main.set_xlim(BASELINE[0], YEAR_MAX)
    ax_main.text(0.005, 0.97, "(a)", transform=ax_main.transAxes,
                 fontweight="bold", va="top")

    # -------------------------------------------------------------------------
    #  STEP 8 — save
    # -------------------------------------------------------------------------
    # bbox_extra_artists keeps the legends inside the tight bounding box;
    # without it the top row gets cropped, because they sit outside the axes.

    out_path = OUT.format(name=name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for path in (out_path, os.path.splitext(out_path)[0] + ".pdf"):
        fig.savefig(path, bbox_inches="tight",
                    bbox_extra_artists=[legend_scenarios, legend_style])
        print(f"[step 8] wrote {path}")
    plt.close(fig)

# =============================================================================
#  STEP 9 — the same numbers as LaTeX tables
# =============================================================================
# Written as a bare `tabular` so it can be \\input inside whatever table
# environment the paper wants, with its own caption and placement. Plain LaTeX:
# \hline and | rules, no booktabs. \toprule/\midrule/\cmidrule need
# \usepackage{booktabs}, and without it \cmidrule(lr){2-3} does not error — it
# prints "(lr)2-3" into the table as text.

for variable, (name, label, _, unit_tex, _) in VARIABLES.items():
    rows_tex = []
    for scenario in SCENARIOS:
        row = stats[(variable, scenario)]
        rows_tex.append(
            f"{SCENARIOS[scenario][0]} & {row['n_emu']} & {row['n_cesm']} & "
            f"{row['corr']:.3f} & {row['rmse']:.3f} & {row['bias']:+.3f} & "
            f"{row['inside']:.0f} \\\\")

    table_tex = "\n".join([
        r"\begin{tabular}{|l|r|r|r|r|r|r|}",
        r"\hline",
        r"\textbf{Experiment} & \multicolumn{2}{c|}{\textbf{Members}} & "
        r"\textbf{$r$} & \textbf{RMSE} & \textbf{Bias} & \textbf{In band} \\",
        r"\cline{2-3}",
        rf" & Emulator & CESM2 & & ({unit_tex}) & ({unit_tex}) & (\%) \\",
        r"\hline",
        *rows_tex,
        r"\hline",
        r"\end{tabular}",
    ])

    table_path = TABLE.format(name=name)
    os.makedirs(os.path.dirname(table_path) or ".", exist_ok=True)
    with open(table_path, "w") as handle:
        handle.write(f"% {label}: emulated vs held-out CESM2 global means.\n"
                     "% r and RMSE compare the two ENSEMBLE-MEAN series over the\n"
                     "% years both cover; 'in band' is the share of those years\n"
                     "% where the difference falls inside CESM2's own member\n"
                     "% range.\n"
                     f"% Built from {DATA_DIR}/ by\n"
                     "% scripts/make_fig12_from_csv.py — do not edit by hand.\n")
        handle.write(table_tex + "\n")
    print(f"[step 9] wrote {table_path}")

    print(f"\n{label}: emulator vs held-out CESM2 "
          f"({'percentage points' if VARIABLES[variable][4] else 'degC'}, "
          f"years both sides cover)")
    print(pd.DataFrame({s: stats[(variable, s)] for s in SCENARIOS})
          .T.round(3).to_string())
