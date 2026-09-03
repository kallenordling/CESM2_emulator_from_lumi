#!/usr/bin/env python3
"""
================================================================================
 FIGURES 3 AND 4, REBUILT FROM THE EXPORTED CSVs
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig34_from_csv.py

Everything configurable is in the SETTINGS block below. No command-line options
and no helper functions: the script runs top to bottom in seven numbered steps.

WHAT THIS IS
------------
The CSV twin of scripts/make_fig3.py, exactly as make_fig12_from_csv.py is the
CSV twin of make_fig1/make_fig2. It reads the sixteen files that
scripts/make_fig12_csv.py exported — years as rows, ensemble members as
columns, absolute global means — and needs nothing else. 280 KB of text, about
a second, no network mount.

make_fig3.py remains the authoritative path: it walks the evaluation NetCDFs
and the CESM2 reference files and computes the global means itself. This script
cannot notice anything that export dropped. The CSVs are already reduced to
global means over held-out members with the counts matched, so changing any of
that means re-running make_fig12_csv.py, not editing the settings here.

WHAT THE FIGURES SHOW
---------------------
TWO figures — fig03 for temperature, fig04 for precipitation — each a 2x2 grid
of the four experiments. Each histogram pools EVERY member-year of the window
N_YEARS selects and asks whether the two samples are drawn from the same
distribution.

WHY POOL MEMBER-YEARS
---------------------
A single ensemble mean per experiment says nothing about spread, and spread is
the thing a generative emulator most easily gets wrong: it can reproduce the
forced trajectory while generating too little year-to-year variability around
it. Pooling every member and every year in the window gives the distribution
that variability actually produces.

WHICH VALUES, AND THE ONE TRAP
------------------------------
ANOMALIES, each side referenced to ITS OWN 1850-1900 mean — otherwise the
comparison is dominated by a constant offset in the mean state rather than by
the response.

The trap, which bit once already in make_fig3.py: the baseline must come from
the FULL record, not from the truncated window. The last 20 years of a scenario
do not contain 1850-1900, so computing the baseline after truncation yields
NaN and an empty figure. Here the baselines are simply READ from
baselines.csv, which is what makes that mistake unavailable — and it is also
how ssp370, which begins in 2015 and has no pre-industrial of its own, inherits
the historical baseline on both sides.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

# Where scripts/make_fig12_csv.py wrote its output. Expected inside:
#     <variable>_<scenario>_<side>.csv   years as rows, members as columns
#     baselines.csv                      each side's own 1850-1900 mean
DATA_DIR = "plots/fig12_data"

FIGURE_NAME = {"TREFHT": "fig03", "PRECT": "fig04"}
# Each figure gets its OWN FOLDER, holding the figure and the LaTeX table of
# its statistics: plots/fig03/{fig03.png, fig03.pdf, fig03_stats.tex}.
OUT = "plots/{name}/{name}.png"          # the .pdf sibling is written alongside
TABLE = "plots/{name}/{name}_stats.tex"

# How many years at the END of each experiment go into the figure and the
# statistics. 0 = every year of the record.
#
# READ THE SD RATIO DIFFERENTLY DEPENDING ON THIS. Over a short window the
# spread is mostly internal variability. Over the FULL record it is dominated by
# the forced trend — hist spans 165 years and about 1 degC of warming — so the
# raw standard deviations below measure "how much does the climate move over the
# record", not "how noisy is it". The KS test is unaffected: it runs on
# residuals with each side's ensemble-mean trajectory removed, which subtracts
# the trend whatever the window length.
N_YEARS = 20

N_BINS = 24

# Variables:
#   label       -> the noun used in captions
#   axis_label  -> the full x-axis label, units included. MATPLOTLIB text, not
#                  LaTeX: mathtext handles $^{\\circ}$, but a percent sign is
#                  written bare, since \\% would print the backslash
#   unit_tex    -> unit for the table header and caption (typeset)
#   unit_plain  -> the same unit for the % comment header (not typeset)
#   as_percent  -> express the anomaly as a PERCENTAGE of that side's own
#                  baseline rather than as an absolute difference
#
# PRECIPITATION IS SHOWN AS A PERCENTAGE, matching figure 2. A few hundredths of
# a mm/day means nothing without the ~2.9 mm/day it is relative to, and dividing
# each side by its OWN baseline stops a small difference in the mean state from
# masquerading as a difference in variability.
VARIABLES = {
    "TREFHT": ("Temperature", "Temperature anomaly ($^{\\circ}$C)",
               "$^{\\circ}$C", "degC", False),
    "PRECT":  ("Precipitation", "Precipitation change (%)",
               "\\%-points", "%-points", True),
}

# The experiments, in plotting order: key -> (label, colour).
# Okabe-Ito colours, distinguishable in greyscale and to colour-blind readers.
SCENARIOS = {
    "hist":   ("Historical",                "#0072B2"),
    "ssp370": ("SSP3-7.0",                  "#D55E00"),
    "aaer":   ("Aerosol-only (AAER)",       "#E69F00"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "#009E73"),
}

# =============================================================================

import os

import matplotlib
matplotlib.use("Agg")                      # no display needed; write files only
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

print(__doc__.split("WHAT THIS IS")[0])

# =============================================================================
#  STEP 1 — read the CSVs, and turn them into anomalies
# =============================================================================
# One file per variable, experiment and side: rows are years, columns are
# ensemble members, values are ABSOLUTE cos(lat)-weighted global means.
#
# The baselines are READ rather than recomputed. That is deliberate — see the
# trap in the docstring — and it is also how ssp370 inherits the historical
# baseline on both sides.
#
# Everything downstream wants (member, year), so the frames are transposed on
# the way in: `.mean(axis=0)` is then the ensemble-mean trajectory and
# `.mean(axis=1)` is one number per member, which is exactly the distinction
# the two statistical tests in step 6 turn on.

baseline_frame = pd.read_csv(f"{DATA_DIR}/baselines.csv")
baseline = {(row.variable, row.scenario, row.side): row.baseline
            for row in baseline_frame.itertuples()}

by_member = {}   # (variable, scenario, side) -> array (member, year)
pooled = {}      # (variable, scenario, side) -> flat array of member-years
# The window is the LAST N_YEARS of each record, and the records end at
# different times — hist in 2014, ssp370 in 2100, aaer and ghg in 2050 — so
# "the last 20 years" is a different span in every panel. The actual years are
# kept here and shown in the panel titles and the table caption, because a
# reader cannot otherwise tell which decades a panel covers.
years_used = {}  # (variable, scenario) -> (first_year, last_year)
for variable in VARIABLES:
    as_percent = VARIABLES[variable][4]
    for scenario in SCENARIOS:
        for side in ("emulator", "cesm2"):
            frame = pd.read_csv(f"{DATA_DIR}/{variable}_{scenario}_{side}.csv",
                                index_col="year")
            base = baseline[(variable, scenario, side)]
            anomaly = frame - base
            if as_percent:
                # Percentage of that side's OWN baseline, so the two sides stay
                # comparable even though their mean states differ slightly.
                anomaly = 100.0 * anomaly / base

            # Truncate to the window AFTER the baseline has been applied.
            window = anomaly.iloc[-N_YEARS:] if N_YEARS else anomaly

            by_member[(variable, scenario, side)] = window.values.T
            pooled[(variable, scenario, side)] = window.values.T.ravel()

            span = (int(window.index[0]), int(window.index[-1]))
            # Both sides must cover the same years, or the histograms would be
            # comparing different decades to each other.
            assert years_used.setdefault((variable, scenario), span) == span, (
                f"{variable} {scenario}: {side} covers {span}, the other side "
                f"covers {years_used[(variable, scenario)]}")

        emulator_shape = by_member[(variable, scenario, "emulator")].shape
        cesm_shape = by_member[(variable, scenario, "cesm2")].shape
        print(f"[step 1] {variable:6s} {scenario:7s} "
              f"emulator {emulator_shape[0]:2d} members x {emulator_shape[1]} "
              f"years, CESM2 {cesm_shape[0]:2d} x {cesm_shape[1]}"
              + ("   (ssp370 baseline inherited from hist)"
                 if scenario == "ssp370" else ""))

# =============================================================================
#  STEP 2 — describe each distribution
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
        emulator_values = pooled[(variable, scenario, "emulator")]
        cesm_values = pooled[(variable, scenario, "cesm2")]
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
        print(f"[step 2] {variable:6s} {scenario:7s} "
              f"mean {row['emulator_mean']:8.3f} vs {row['cesm_mean']:8.3f} "
              f"(diff {row['mean_difference']:+.3f}), "
              f"sd {row['emulator_sd']:.3f} vs {row['cesm_sd']:.3f} "
              f"(ratio {row['sd_ratio']:.2f})")

# =============================================================================
#  STEP 3 — draw one 2x2 figure per variable
# =============================================================================
# Rows and columns are just the four experiments wrapped two-by-two; with only
# four panels a 2x2 block sits better on a page than a 1x4 strip, and leaves the
# panels wide enough to read the distribution shapes.
#
# Both sides share one set of bin edges per panel, spanning the combined range:
# with different edges the two shapes would not be comparable. density=True, not
# counts — the two sides need not contribute the same number of member-years,
# so raw counts would show the sample sizes rather than the distributions.

EMULATOR_COLOUR = "#D55E00"
CESM_COLOUR = "#0072B2"

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

for variable in VARIABLES:
    variable_label, axis_label, _, _, _ = VARIABLES[variable]
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))

    for panel_index, scenario in enumerate(SCENARIOS):
        axis = axes[panel_index // 2][panel_index % 2]
        emulator_values = pooled[(variable, scenario, "emulator")]
        cesm_values = pooled[(variable, scenario, "cesm2")]

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
        first_year, last_year = years_used[(variable, scenario)]
        axis.set_title(f"{SCENARIOS[scenario][0]} ({first_year}\u2013{last_year})\n"
                       f"$\\Delta$mean {row['mean_difference']:+.3f}, "
                       f"sd ratio {row['sd_ratio']:.2f}",
                       fontsize=9.5, loc="left")
        axis.text(0.03, 0.95, f"({'abcd'[panel_index]})",
                  transform=axis.transAxes, fontweight="bold", va="top",
                  fontsize=9)
        # Axis labels only on the outside, so the panels are not repetitive.
        if panel_index // 2 == 1:
            axis.set_xlabel(axis_label)
        if panel_index % 2 == 0:
            axis.set_ylabel("Probability density")

    # =========================================================================
    #  STEP 4 — legend, then save
    # =========================================================================
    # tight_layout FIRST, so the legend is placed relative to the settled axes;
    # the other way round lets tight_layout move the axes out from under it.
    legend_entries = [
        Patch(facecolor=CESM_COLOUR, alpha=0.45,
              label="CESM2 (held-out members)"),
        Line2D([], [], color=EMULATOR_COLOUR, lw=2.0, label="Emulator"),
        Line2D([], [], color="0.35", lw=1.4, ls="--", label="distribution mean"),
    ]
    figure.tight_layout()
    # No figure title: the paper's caption says what the figure is, and a
    # heading repeating it wastes the space above the panels.
    legend = figure.legend(handles=legend_entries, frameon=False, ncols=3,
                           loc="lower center", bbox_to_anchor=(0.5, 1.005))

    out_png = OUT.format(name=FIGURE_NAME[variable])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    for path in (out_png, os.path.splitext(out_png)[0] + ".pdf"):
        # The legend sits OUTSIDE the axes, so it has to be named here; a
        # tight bbox crops whatever it is not told about.
        figure.savefig(path, bbox_inches="tight", bbox_extra_artists=[legend])
        print(f"[step 4] wrote {path}")
    plt.close(figure)

# =============================================================================
#  STEP 5 — are the means the same, and are the distributions the same?
# =============================================================================
# The obvious thing — a t-test and a KS test on the pooled member-years — IS
# WRONG, and badly. Both assume independent samples, but consecutive years
# inside one member are strongly correlated: over these windows the lag-1
# autocorrelation of the pooled values runs 0.56 to 0.93, because every member
# carries the same forced trend across the window. The effective sample size is
# a small fraction of the nominal one, and the tests return absurd confidence —
# p ~ 1e-26 for aaer, which no honest reading of a few hundred correlated
# points supports.
#
# So each question gets a form of the data that satisfies its test:
#
# ARE THE MEANS THE SAME?  Welch's t-test on the MEMBER MEANS — one number per
#   member. Members are independent realizations, so these are genuinely
#   independent units. Welch rather than Student because the two sides need not
#   have equal variance.
#
# ARE THE DISTRIBUTIONS THE SAME?  Two-sample Kolmogorov-Smirnov, and Levene
#   for the variances, on values with each side's OWN ensemble-mean trajectory
#   removed. That subtracts the forced signal — the trend the members share —
#   and leaves internal variability. It also removes the correlation: the lag-1
#   autocorrelation of the residuals is below 0.06, so the samples now behave as
#   independent draws. This asks the question the figure is really about: does
#   the emulator's spread ABOUT its own trajectory match CESM2's?

tests = {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        emulator_series = by_member[(variable, scenario, "emulator")]
        cesm_series = by_member[(variable, scenario, "cesm2")]

        # means: one value per member
        mean_test = stats.ttest_ind(emulator_series.mean(axis=1),
                                    cesm_series.mean(axis=1), equal_var=False)

        # distributions: internal variability, forced signal removed
        emulator_residual = (emulator_series
                             - emulator_series.mean(axis=0, keepdims=True)).ravel()
        cesm_residual = (cesm_series
                         - cesm_series.mean(axis=0, keepdims=True)).ravel()
        ks_test = stats.ks_2samp(emulator_residual, cesm_residual)
        variance_test = stats.levene(emulator_residual, cesm_residual)

        tests[(variable, scenario)] = dict(
            mean_p=float(mean_test.pvalue),
            ks_p=float(ks_test.pvalue),
            levene_p=float(variance_test.pvalue))
        print(f"[step 5] {variable:6s} {scenario:7s} "
              f"means p={mean_test.pvalue:.4f} "
              f"{'DIFFER' if mean_test.pvalue < 0.05 else 'same'}, "
              f"distribution p={ks_test.pvalue:.3f} "
              f"{'DIFFER' if ks_test.pvalue < 0.05 else 'same'}, "
              f"variance p={variance_test.pvalue:.3f}")

# =============================================================================
#  STEP 6 — the same numbers as a LaTeX table
# =============================================================================
# A COMPLETE `table` float — caption, label and tabular — so \\input drops it
# straight into the paper with no wrapper. The label is \\label{tab:fig03} and
# so on, matching the figure it belongs to.
#
# The caption is built from the same numbers as the table, so it cannot go stale
# against the settings: the window length and the member counts are read off
# N_YEARS and the data rather than typed in.
#
# Plain LaTeX with borders: \toprule and \cmidrule need \usepackage{booktabs},
# and without it \cmidrule silently typesets "(lr)2-3" into the table.

# A p-value formatter: below 0.001 the exact value is noise, so say so.
def format_p(value):
    return "$<$0.001" if value < 0.001 else f"{value:.3f}"


for variable in VARIABLES:
    variable_label, _, unit_tex, unit_plain, as_percent = VARIABLES[variable]

    table_rows = []
    for scenario in SCENARIOS:
        row = statistics[(variable, scenario)]
        test = tests[(variable, scenario)]
        table_rows.append(
            f"{SCENARIOS[scenario][0]} & "
            f"{row['mean_difference']:+.3f} & {format_p(test['mean_p'])} & "
            f"{'no' if test['mean_p'] < 0.05 else 'yes'} & "
            f"{row['sd_ratio']:.2f} & {format_p(test['ks_p'])} & "
            f"{'no' if test['ks_p'] < 0.05 else 'yes'} \\\\")

    # Member counts actually used, so neither the caption nor the comment
    # header can go stale.
    counts = {}
    for side in ("emulator", "cesm2"):
        sizes = sorted({by_member[(variable, s, side)].shape[0]
                        for s in SCENARIOS})
        counts[side] = (f"{sizes[0]}" if len(sizes) == 1
                        else f"{sizes[0]}--{sizes[-1]}")

    # The experiments end in different years, so name the spans rather than
    # leaving "the last 20 years" to be resolved against four different records.
    spans = "; ".join(
        f"{SCENARIOS[s][0]} {years_used[(variable, s)][0]}--"
        f"{years_used[(variable, s)][1]}" for s in SCENARIOS)
    window_text = f"the last {N_YEARS} years" if N_YEARS else "the full record"
    span_text = f" ({spans})" if N_YEARS else ""

    referenced = ("as a percentage change from each side's own 1850--1900 "
                  "mean" if as_percent else
                  "as anomalies relative to each side's own 1850--1900 mean")

    caption = (
        f"Distributions of global-mean {variable_label.lower()} over "
        f"{window_text} of each experiment{span_text}, pooling every member "
        f"and every year, {referenced}. The "
        f"MEAN columns give the difference between the emulator's and CESM2's "
        f"pooled means, in {unit_tex}, and Welch's $t$-test on the member "
        f"means ({counts['emulator']} emulator members against "
        f"{counts['cesm2']} CESM2 members), members being independent "
        f"realizations. The DISTRIBUTION columns give the ratio of standard "
        f"deviations, emulator over CESM2, and a two-sample "
        f"Kolmogorov--Smirnov test on values from which each side's own "
        f"ensemble-mean trajectory has been removed --- that is, on internal "
        f"variability with the shared forced trend taken out, which is what "
        f"makes the samples independent enough to test. ``Same?'' is $p>0.05$.")

    table_tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        # The 6-7 column tables overflow article's text block by 60-100pt at
        # full size. \footnotesize and tighter column padding bring them inside it
        # without needing graphicx for \resizebox; both are scoped by the
        # table environment, so neither leaks into the surrounding document.
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{FIGURE_NAME[variable]}}}",
        r"\begin{tabular}{|l|r|r|c|r|r|c|}",
        r"\hline",
        r"\textbf{Experiment} & \multicolumn{3}{c|}{\textbf{Mean}} & "
        r"\multicolumn{3}{c|}{\textbf{Distribution}} \\",
        r"\cline{2-7}",
        r" & Difference (%s) & $p$ & Same? & SD ratio & $p$ & Same? \\" % unit_tex,
        r"\hline",
        *table_rows,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_path = TABLE.format(name=FIGURE_NAME[variable])
    os.makedirs(os.path.dirname(table_path) or ".", exist_ok=True)
    with open(table_path, "w") as handle:
        handle.write(
            f"% {variable_label} ({unit_plain}): emulated vs held-out CESM2.\n"
            f"% {('last %d years' % N_YEARS) if N_YEARS else 'full record'}.\n"
            f"% Built from {DATA_DIR}/ by scripts/make_fig34_from_csv.py\n"
            "% — do not edit by hand. The caption below says what the numbers\n"
            "% are; \\input this file directly.\n")
        handle.write(table_tex + "\n")
    print(f"[step 6] wrote {table_path}")
