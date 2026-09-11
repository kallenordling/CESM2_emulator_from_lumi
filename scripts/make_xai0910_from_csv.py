#!/usr/bin/env python3
"""
================================================================================
 FIGURES 9 AND 10 — THE NONLINEARITY N(t), IN CESM2 AND IN THE EMULATOR
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig1112_from_csv.py

Everything configurable is in the SETTINGS block below. No command-line options
and no helper functions: the script runs top to bottom in eight numbered steps.

WHAT THIS IS
------------
The global-mean, time-resolved twin of scripts/paper_fig_attribution.py, which
computes the same residual but as decadal MAPS. This one reads the sixteen CSVs
that scripts/make_fig12_csv.py exported and needs nothing else, exactly as
make_fig12_from_csv.py and make_fig34_from_csv.py do.

WHAT THE FIGURES SHOW
---------------------
TWO figures — xai09 for temperature, xai10 for precipitation — each two panels:

    (a) the three ingredients, as anomalies: the all-forcing run, GHG-only,
        aerosol-only, and the SUM of the last two, for both sides. Where the
        sum tracks the all-forcing curve the response is additive; where it
        does not, the gap between them IS panel (b).

    (b) N(t) itself, for CESM2 and for the emulator, with 90% bands.

THE QUANTITY
------------
    N(t) = Delta(ALL)(t) - Delta(GHG)(t) - Delta(AAER)(t)

Delta is the anomaly relative to each side's OWN 1850-1900 mean, so N is a
difference of three responses and every climatological offset has already
cancelled. It is computed SEPARATELY on each side, which is the point: CESM2's
N is a property of the climate model, the emulator's N is what the emulator
reproduces, and the two are then comparable without either being a reference
for the other.

If the response were additive, N would be zero. It is not — in CESM2 itself,
before any emulator is involved.

ALL IS hist SPLICED TO ssp370
-----------------------------
There is no single "all-forcing" experiment in the CSVs: hist runs 1850-2014
and ssp370 takes over in 2015. They are one continuous LENS2 trajectory and
carry the SAME member names (LE2-1231.001 and so on), so they are spliced PER
MEMBER, not per ensemble mean. The single-forcing runs stop in 2050, so N is
defined on 1850-2050 and the figure stops there.

N IS NOT A PURE INTERACTION TERM — THE PAPER MUST SAY SO
--------------------------------------------------------
The single-forcing runs hold ozone, land use, biomass burning, solar and
volcanic forcing at 1850 levels. Everything the all-forcing run has and they do
not therefore lands in N alongside the genuine nonlinearity:

    N = (forcings absent from both single-forcing runs) + (nonlinearity)

Before 2015 the volcanic eruptions dominate it outright, which is visible in
panel (b) as downward spikes at Krakatoa, Agung, El Chichon and Pinatubo. That
is why the statistics below are computed on a POST-2015 window, where no major
eruption is prescribed and the single-forcing runs are still alive: 2041-2050,
the last clean decade, the same window paper_fig_attribution.py defaults to.
"""

# =============================================================================
#  SETTINGS — everything configurable lives here
# =============================================================================

# Where scripts/make_fig12_csv.py wrote its output. Expected inside:
#     <variable>_<scenario>_<side>.csv   years as rows, members as columns
#     baselines.csv                      each side's own 1850-1900 mean
DATA_DIR = "plots/xai12_data"

FIGURE_NAME = {"TREFHT": "xai09", "PRECT": "xai10"}
# Each figure gets its OWN FOLDER, holding the figure and the LaTeX table of
# its statistics: plots/xai09/{xai09.png, xai09.pdf, xai09_stats.tex}.
OUT = "plots/{name}/{name}.png"          # the .pdf sibling is written alongside
TABLE = "plots/{name}/{name}_stats.tex"
# The residual itself, as a CSV, so the numbers behind the figure are readable
# without re-running anything.
SERIES_CSV = "plots/{name}/{name}_residual.csv"

BASELINE = (1850, 1900)

# The single-forcing runs end here, and so does N.
YEAR_MAX = 2050

# The window the statistics are computed on. POST-2015, so no prescribed
# eruption is inside it, and ending at YEAR_MAX because that is where the
# single-forcing runs stop. Matches paper_fig_attribution.py's default.
WINDOW = (2041, 2050)

# N(t) is a difference of three noisy ensemble means and is spiky year to year.
# The raw series is drawn thin and a centred running mean over this many years
# is drawn over it. 1 disables the smoothing.
SMOOTH_YEARS = 11

# Variables:
#   label       -> the noun used in captions
#   axis_label  -> full axis label for the anomaly panel, units included.
#                  MATPLOTLIB text, not LaTeX: mathtext handles $^{\\circ}$,
#                  but a percent sign is written bare
#   unit_tex    -> unit for the table header and caption (typeset)
#   as_percent  -> express the anomaly as a PERCENTAGE of that side's own
#                  baseline rather than as an absolute difference
#
# PRECIPITATION IS SHOWN AS A PERCENTAGE, matching figures 2 and 4. Each
# experiment divides by its OWN baseline; those differ by under 0.2% between
# experiments (2.930-2.936 mm/day), far below anything visible here.
VARIABLES = {
    "TREFHT": ("Temperature", "Temperature anomaly ($^{\\circ}$C)",
               "$^{\\circ}$C", False),
    "PRECT":  ("Precipitation", "Precipitation change (%)", "\\%", True),
}

# The three ingredients, in plotting order: key -> (label, colour).
# Okabe-Ito colours, distinguishable in greyscale and to colour-blind readers.
# "all" is the spliced hist+ssp370 trajectory built in step 3.
INGREDIENTS = {
    "all":  ("All-forcing (hist + SSP3-7.0)", "#0072B2"),
    "ghg":  ("Greenhouse-gas-only (GHG)",     "#009E73"),
    "aaer": ("Aerosol-only (AAER)",           "#E69F00"),
}
SUM_COLOUR = "#CC79A7"        # GHG + AAER, the curve N measures the gap to

# The two sides, in the residual panel.
SIDES = {"cesm2": ("CESM2", "#000000"), "emulator": ("Emulator", "#D55E00")}

# =============================================================================

import os

import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
#  STEP 1 — read the CSVs
# =============================================================================
# Rows are years, columns are ensemble members, values are ABSOLUTE global
# means — degrees Celsius and mm/day — cos(lat)-weighted when they were
# exported.
#
# Nothing is selected or filtered here. The export already restricted CESM2 to
# HELD-OUT members and capped the emulator to the same count per experiment.

series = {}          # (variable, scenario, side) -> DataFrame(year x member)
for variable in VARIABLES:
    for scenario in ("hist", "ssp370", "ghg", "aaer"):
        for side in ("emulator", "cesm2"):
            path = f"{DATA_DIR}/{variable}_{scenario}_{side}.csv"
            series[(variable, scenario, side)] = pd.read_csv(path,
                                                             index_col="year")
        print(f"[step 1] {variable:6s} {scenario:7s} "
              f"emulator {series[(variable, scenario, 'emulator')].shape}, "
              f"CESM2 {series[(variable, scenario, 'cesm2')].shape}  "
              f"(year x member)")

# =============================================================================
#  STEP 2 — each side's 1850-1900 baseline, from the same files
# =============================================================================
# Computed from the frames just read rather than taken from baselines.csv, for
# the reason make_fig12_from_csv.py gives: the anomaly should not be able to
# drift from the absolute values it is derived from.
#
# ssp370 begins in 2015 and has no pre-industrial of its own. It inherits the
# historical baseline on BOTH sides — which is also exactly what the splice in
# step 3 requires, since a jump in baseline at 2014/2015 would appear in N as a
# step change that is an artefact of the bookkeeping and nothing else.

baseline = {}
for variable in VARIABLES:
    for side in ("emulator", "cesm2"):
        for scenario in ("hist", "ghg", "aaer"):
            window = series[(variable, scenario, side)].loc[BASELINE[0]:BASELINE[1]]
            baseline[(variable, scenario, side)] = float(window.values.mean())
        baseline[(variable, "ssp370", side)] = baseline[(variable, "hist", side)]

for variable in VARIABLES:
    for scenario in ("hist", "ssp370", "ghg", "aaer"):
        print(f"[step 2] {variable:6s} {scenario:7s} baselines "
              f"emulator {baseline[(variable, scenario, 'emulator')]:8.3f}, "
              f"CESM2 {baseline[(variable, scenario, 'cesm2')]:8.3f}"
              + ("   (inherited from hist)" if scenario == "ssp370" else ""))

# =============================================================================
#  STEP 3 — anomalies, and the spliced all-forcing trajectory
# =============================================================================
# Each side is referenced to ITS OWN pre-industrial, so the constant offset
# between the emulator's mean state and CESM2's is gone before any subtraction
# happens and N is a difference of RESPONSES.
#
# The splice is PER MEMBER. hist and ssp370 carry the same member names on both
# sides — LE2-1231.001 in one is the same run continued in the other — so
# concatenating them column by column rebuilds the continuous trajectory each
# realization actually followed. Splicing the ensemble MEANS instead would give
# the same curve here but would throw away the member axis that every interval
# below is computed on.

anomaly = {}         # (variable, ingredient, side) -> DataFrame(year x member)
for variable, (_, _, _, as_percent) in VARIABLES.items():
    for side in ("emulator", "cesm2"):
        for scenario in ("hist", "ssp370", "ghg", "aaer"):
            base = baseline[(variable, scenario, side)]
            values = series[(variable, scenario, side)] - base
            if as_percent:
                values = 100.0 * values / base
            anomaly[(variable, scenario, side)] = values

        hist_part = anomaly[(variable, "hist", side)]
        ssp_part = anomaly[(variable, "ssp370", side)]
        if list(hist_part.columns) != list(ssp_part.columns):
            raise SystemExit(
                f"hist and ssp370 member columns differ for {variable}/{side} — "
                f"the per-member splice would silently pair unrelated runs")
        combined = pd.concat([hist_part.loc[hist_part.index < 2015], ssp_part])
        anomaly[(variable, "all", side)] = combined.loc[combined.index <= YEAR_MAX]
        print(f"[step 3] {variable:6s} {side:8s} all-forcing = hist "
              f"{hist_part.index.min()}-2014 + ssp370 2015-{YEAR_MAX}, "
              f"{anomaly[(variable, 'all', side)].shape} (year x member)")

# =============================================================================
#  STEP 4 — N(t), and the band around it
# =============================================================================
#     N(t) = Delta(ALL) - Delta(GHG) - Delta(AAER)
#
# on the years all three cover, which the single-forcing runs cap at 2050.
#
# THE BAND IS AN INTERVAL ON N, NOT A MEMBER SPREAD. N combines three ensemble
# MEANS, each carrying its own sampling error, so the three variances add:
#
#     se(t) = sqrt( var_all/n_all + var_ghg/n_ghg + var_aaer/n_aaer )
#
# with the variances taken across members at that year. The band is +/-1.645
# se, a 90% interval, matching the intervals in figures 1 and 2. It is drawn
# per year, so it says nothing about whether N is jointly nonzero across a
# stretch of years — the window statistics in step 5 are for that.

residual, ingredient_mean, counts = {}, {}, {}
for variable in VARIABLES:
    for side in ("emulator", "cesm2"):
        frames = {k: anomaly[(variable, k, side)] for k in INGREDIENTS}
        years = frames["all"].index
        for frame in frames.values():
            years = years.intersection(frame.index)
        years = years[years <= YEAR_MAX]

        means = {k: frames[k].loc[years].mean(axis=1) for k in INGREDIENTS}
        variances = {k: frames[k].loc[years].var(axis=1, ddof=1) for k in INGREDIENTS}
        n = {k: frames[k].shape[1] for k in INGREDIENTS}

        value = means["all"] - means["ghg"] - means["aaer"]
        standard_error = np.sqrt(sum(variances[k] / n[k] for k in INGREDIENTS))

        residual[(variable, side)] = pd.DataFrame(
            {"N": value, "se": standard_error,
             "low": value - 1.645 * standard_error,
             "high": value + 1.645 * standard_error,
             "sum_ghg_aaer": means["ghg"] + means["aaer"]},
            index=years)
        ingredient_mean[(variable, side)] = pd.DataFrame(means, index=years)
        counts[(variable, side)] = n
        print(f"[step 4] {variable:6s} {side:8s} N on {years.min()}-{years.max()}, "
              f"n = " + "/".join(f"{k} {n[k]}" for k in INGREDIENTS))

# =============================================================================
#  STEP 5 — the window statistics
# =============================================================================
# One number per MEMBER — its own time mean over the window — for each of the
# three experiments. Members are independent realizations, so these are the
# independent units an interval is entitled to assume; the year-by-year values
# are not, and using them would shrink every interval below by a factor of
# several.
#
# N is then a LINEAR COMBINATION of three group means with different member
# counts, so its variance is the sum of the three, and the degrees of freedom
# come from Satterthwaite — the same approximation Welch's test makes for two
# groups, written out for three.
#
# The emulator-minus-CESM2 comparison at the end combines all SIX groups the
# same way. It is the honest way to ask whether the emulator's nonlinearity
# differs from CESM2's, and with six sampling errors stacked up it is a
# demanding test: read a wide interval as too few members, not as agreement.

stats = {}
for variable in VARIABLES:
    per_side = {}
    for side in ("emulator", "cesm2"):
        member_means, member_var, n = {}, {}, {}
        for key in INGREDIENTS:
            frame = anomaly[(variable, key, side)].loc[WINDOW[0]:WINDOW[1]]
            values = frame.mean(axis=0).values          # one per member
            member_means[key] = float(values.mean())
            member_var[key] = float(values.var(ddof=1)) / len(values)
            n[key] = len(values)

        value = member_means["all"] - member_means["ghg"] - member_means["aaer"]
        variance = sum(member_var[k] for k in INGREDIENTS)
        degrees_freedom = variance ** 2 / sum(
            member_var[k] ** 2 / (n[k] - 1) for k in INGREDIENTS)
        half_width = sstats.t.ppf(0.95, degrees_freedom) * np.sqrt(variance)

        per_side[side] = dict(
            N=value, half=half_width, var=variance, df=degrees_freedom,
            all=member_means["all"], ghg=member_means["ghg"],
            aaer=member_means["aaer"], var_by=member_var, n=n,
            # N as a share of the all-forcing response it is a part of. This is
            # what makes the residual interpretable: 0.4 K means little until
            # it is set against the 2.0 K the run actually warmed.
            share=100.0 * value / member_means["all"],
            detected=abs(value) > half_width)

        print(f"[step 5] {variable:6s} {side:8s} {WINDOW[0]}-{WINDOW[1]}: "
              f"ALL {member_means['all']:+.3f} - GHG {member_means['ghg']:+.3f} "
              f"- AAER {member_means['aaer']:+.3f} = N {value:+.3f} "
              f"+/-{half_width:.3f} ({per_side[side]['share']:.1f}% of ALL)")

    # Emulator minus CESM2, with all six groups' sampling errors added.
    difference = per_side["emulator"]["N"] - per_side["cesm2"]["N"]
    variance = per_side["emulator"]["var"] + per_side["cesm2"]["var"]
    degrees_freedom = variance ** 2 / sum(
        per_side[s]["var_by"][k] ** 2 / (per_side[s]["n"][k] - 1)
        for s in ("emulator", "cesm2") for k in INGREDIENTS)
    half_width = sstats.t.ppf(0.95, degrees_freedom) * np.sqrt(variance)

    # The ratio is the headline: how much of CESM2's nonlinearity the emulator
    # produces. Reported only where CESM2's own N is resolved, since a ratio to
    # a number indistinguishable from zero means nothing.
    stats[variable] = dict(
        per_side=per_side, diff=difference, diff_half=half_width,
        diff_detected=abs(difference) > half_width,
        ratio=(per_side["emulator"]["N"] / per_side["cesm2"]["N"]
               if per_side["cesm2"]["detected"] else np.nan))
    print(f"[step 5] {variable:6s} emulator - CESM2 = {difference:+.3f} "
          f"+/-{half_width:.3f}"
          + (f", ratio {stats[variable]['ratio']:.2f}x"
             if np.isfinite(stats[variable]["ratio"]) else ", ratio n/a"))

# =============================================================================
#  STEP 6 — one figure per variable
# =============================================================================

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

for variable, (name, axis_label, unit_tex, as_percent) in VARIABLES.items():
    figure_name = FIGURE_NAME[variable]
    fig = plt.figure(figsize=(9.5, 7.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.26)
    ax_top = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1], sharex=ax_top)

    # -------------------------------------------------------------------------
    #  PANEL (a) — the three ingredients, and their sum
    # -------------------------------------------------------------------------
    # CESM2 dashed with open circles, the emulator thick and solid, as in
    # figures 1 and 2. The circles matter: solid-vs-dashed in one colour is
    # unreadable where the curves coincide, and here they mostly do — a marker
    # shape survives overlap, greyscale and print size.
    #
    # The SUM curve is the whole argument of the figure. Where it departs from
    # the all-forcing curve, the response is not the sum of its parts, and the
    # size of that departure is panel (b).

    for key, (label, colour) in INGREDIENTS.items():
        for side in ("cesm2", "emulator"):
            values = ingredient_mean[(variable, side)][key]
            if side == "emulator":
                ax_top.plot(values.index, values.values, color=colour, lw=2.4,
                            zorder=4, label=label)
            else:
                ax_top.plot(values.index, values.values, color=colour, lw=1.2,
                            ls="--", marker="o", markersize=3.4, markevery=10,
                            markerfacecolor="white", markeredgecolor=colour,
                            zorder=5,
                            path_effects=[pe.withStroke(linewidth=3.0,
                                                        foreground="white")])

    for side in ("cesm2", "emulator"):
        total = residual[(variable, side)]["sum_ghg_aaer"]
        if side == "emulator":
            ax_top.plot(total.index, total.values, color=SUM_COLOUR, lw=2.4,
                        zorder=3, label="GHG + AAER (the sum)")
        else:
            ax_top.plot(total.index, total.values, color=SUM_COLOUR, lw=1.2,
                        ls="--", marker="o", markersize=3.4, markevery=10,
                        markerfacecolor="white", markeredgecolor=SUM_COLOUR,
                        zorder=3)

    ax_top.axhline(0, ls=":", lw=0.8, color="0.3")
    ax_top.axvspan(*BASELINE, color="0.9", alpha=0.6, lw=0, zorder=0)
    ax_top.axvspan(*WINDOW, color="0.75", alpha=0.45, lw=0, zorder=0)
    ax_top.set_ylabel(axis_label)
    ax_top.text(0.005, 0.97, "(a)", transform=ax_top.transAxes,
                fontweight="bold", va="top")
    ax_top.tick_params(labelbottom=False)

    legend_curves = ax_top.legend(frameon=False, ncols=2, loc="upper left",
                                  bbox_to_anchor=(0.0, 0.97), fontsize=9,
                                  handlelength=2.2)
    ax_top.add_artist(legend_curves)
    ax_top.legend(handles=[
        Line2D([], [], color="0.35", lw=2.4, label="Emulator"),
        Line2D([], [], color="0.35", lw=1.2, ls="--", marker="o",
               markersize=3.4, markerfacecolor="white", markeredgecolor="0.35",
               label="CESM2 (held out)")],
        frameon=False, ncols=1, fontsize=8.6, loc="lower left",
        bbox_to_anchor=(0.01, 0.02), handlelength=2.6)

    # -------------------------------------------------------------------------
    #  PANEL (b) — N(t)
    # -------------------------------------------------------------------------
    # Thin raw line plus a centred running mean, because N is a difference of
    # three noisy ensemble means and the year-to-year scatter is not the signal.
    # The shaded band is the 90% interval on N from step 4.

    for side, (side_label, colour) in SIDES.items():
        frame = residual[(variable, side)]
        ax_bottom.fill_between(frame.index, frame["low"], frame["high"],
                               color=colour, alpha=0.16, lw=0, zorder=1)
        ax_bottom.plot(frame.index, frame["N"], color=colour, lw=0.8,
                       alpha=0.45, zorder=2)
        if SMOOTH_YEARS > 1:
            smooth = frame["N"].rolling(SMOOTH_YEARS, center=True,
                                        min_periods=SMOOTH_YEARS).mean()
            ax_bottom.plot(smooth.index, smooth.values, color=colour, lw=2.4,
                           zorder=3,
                           label=f"{side_label}  ({SMOOTH_YEARS}-yr mean)")
        else:
            ax_bottom.plot(frame.index, frame["N"], color=colour, lw=2.4,
                           zorder=3, label=side_label)

    ax_bottom.axhline(0, lw=1.0, color="0.2", zorder=4)
    ax_bottom.axvspan(*WINDOW, color="0.75", alpha=0.45, lw=0, zorder=0)
    ax_bottom.set_xlim(BASELINE[0], YEAR_MAX)
    ax_bottom.set_xlabel("Year")
    ax_bottom.set_ylabel(("N = ALL - GHG - AAER (%)" if as_percent
                          else "N = ALL - GHG - AAER ($^{\\circ}$C)"))
    ax_bottom.text(0.005, 0.96, "(b)", transform=ax_bottom.transAxes,
                   fontweight="bold", va="top")

    # The window numbers, on the panel, so the figure carries its own headline.
    cesm_row, emulator_row = (stats[variable]["per_side"]["cesm2"],
                              stats[variable]["per_side"]["emulator"])
    unit_plain = "%" if as_percent else "degC"
    ax_bottom.annotate(
        f"{WINDOW[0]}-{WINDOW[1]}:  CESM2 {cesm_row['N']:+.3f} "
        f"$\\pm${cesm_row['half']:.3f} {unit_plain} "
        f"({cesm_row['share']:.0f}% of the all-forcing response)\n"
        f"{' ' * 12}emulator {emulator_row['N']:+.3f} "
        f"$\\pm${emulator_row['half']:.3f} {unit_plain} "
        f"({emulator_row['share']:.0f}%)",
        xy=(0.015, 0.05), xycoords="axes fraction", fontsize=8.6, va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82,
                  edgecolor="0.8"))

    # Upper RIGHT: the residual grows through the record, so the top-left
    # corner is the one corner panel (b) always has free.
    ax_bottom.legend(frameon=False, ncols=1, loc="upper left",
                     bbox_to_anchor=(0.07, 0.99), fontsize=9)

    # Room under the curves for the annotation box, which sits at the bottom
    # left and would otherwise be clipped by the axes.
    low, high = ax_bottom.get_ylim()
    ax_bottom.set_ylim(low - 0.30 * (high - low), high)

    # -------------------------------------------------------------------------
    #  STEP 7 — save the figure and the residual series
    # -------------------------------------------------------------------------

    out_path = OUT.format(name=figure_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for path in (out_path, os.path.splitext(out_path)[0] + ".pdf"):
        fig.savefig(path, bbox_inches="tight")
        print(f"[step 7] wrote {path}")
    plt.close(fig)

    export = pd.DataFrame(index=residual[(variable, "cesm2")].index)
    for side in SIDES:
        frame = residual[(variable, side)]
        for column in ("N", "se", "low", "high"):
            export[f"{side}_{column}"] = frame[column]
    export.index.name = "year"
    csv_path = SERIES_CSV.format(name=figure_name)
    export.to_csv(csv_path, float_format="%.6f")
    print(f"[step 7] wrote {csv_path}")

# =============================================================================
#  STEP 8 — the same numbers as LaTeX tables
# =============================================================================
# A COMPLETE `table` float — caption, label and tabular — so \input drops it
# straight into the paper with no wrapper, matching fig01-fig04.
#
# Plain LaTeX: \hline and | rules, no booktabs.

for variable, (name, _, unit_tex, as_percent) in VARIABLES.items():
    figure_name = FIGURE_NAME[variable]
    row = stats[variable]
    cesm_row, emulator_row = row["per_side"]["cesm2"], row["per_side"]["emulator"]

    rows_tex = []
    for side, side_row in (("CESM2", cesm_row), ("Emulator", emulator_row)):
        rows_tex.append(
            f"{side} & {side_row['all']:+.3f} & {side_row['ghg']:+.3f} & "
            f"{side_row['aaer']:+.3f} & {side_row['N']:+.3f} & "
            f"$\\pm${side_row['half']:.3f} & {side_row['share']:.1f} & "
            f"{'yes' if side_row['detected'] else 'unresolved'} \\\\")
    rows_tex.append(r"\hline")
    rows_tex.append(
        f"Emulator $-$ CESM2 & & & & {row['diff']:+.3f} & "
        f"$\\pm${row['diff_half']:.3f} & "
        + (f"{100 * row['ratio']:.0f}" if np.isfinite(row["ratio"]) else "---")
        + f" & {'yes' if row['diff_detected'] else 'unresolved'} \\\\")

    n_text = ", ".join(
        f"{INGREDIENTS[k][0].split(' (')[0].lower()} {cesm_row['n'][k]}"
        for k in INGREDIENTS)
    quantity = ("percentage change in global-mean precipitation"
                if as_percent else
                "global-mean temperature anomaly, in $^{\\circ}$C")

    caption = (
        f"The nonlinearity $N(t) = \\Delta(\\mathrm{{ALL}}) - "
        f"\\Delta(\\mathrm{{GHG}}) - \\Delta(\\mathrm{{AAER}})$ in the "
        f"{quantity}, averaged over {WINDOW[0]}--{WINDOW[1]}. $\\Delta$ is the "
        f"anomaly relative to each side's own 1850--1900 mean, and ALL is the "
        f"historical run spliced per member to SSP3-7.0 at 2015. $N$ is "
        f"computed separately on each side, so neither is a reference for the "
        f"other: CESM2's $N$ is a property of the climate model, and the "
        f"emulator's is what the emulator reproduces. Were the response "
        f"additive, $N$ would be zero; it is not, in CESM2 itself. The "
        f"``share'' column gives $N$ as a percentage of that side's own "
        f"all-forcing response, and on the last row the emulator's $N$ as a "
        f"percentage of CESM2's. Intervals are 90\\% confidence intervals, "
        f"given as half-widths, over the per-member time means "
        f"({n_text} members per side); $N$ combines three ensemble means, so "
        f"three sampling errors add, and the last row combines six. "
        f"``Resolved'' says whether the interval excludes zero --- "
        f"``unresolved'' is too few members to tell, not evidence of "
        f"additivity. "
        f"NOTE THAT $N$ IS NOT A PURE INTERACTION TERM: the single-forcing "
        f"runs hold ozone, land use, biomass burning, solar and volcanic "
        f"forcing at 1850 levels, so $N$ = (those forcings) + (nonlinearity). "
        f"The window is post-2015 for that reason --- before 2015 the "
        f"volcanic eruptions dominate $N$ outright, as the spikes in "
        f"figure~\\ref{{fig:{figure_name}}}(b) show.")

    table_tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{" + caption + "}",
        r"\label{tab:" + figure_name + "}",
        r"\begin{tabular}{|l|r|r|r|r|c|r|c|}",
        r"\hline",
        r"\textbf{Side} & \textbf{ALL} & \textbf{GHG} & \textbf{AAER} & "
        r"\textbf{$N$} & \textbf{90\% CI} & \textbf{Share} & "
        r"\textbf{Resolved?} \\",
        r" & \multicolumn{5}{c|}{(" + unit_tex + r")} & (\%) & \\",
        r"\hline",
        *rows_tex,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_path = TABLE.format(name=figure_name)
    os.makedirs(os.path.dirname(table_path) or ".", exist_ok=True)
    with open(table_path, "w") as handle:
        handle.write(
            f"% {name}: the nonlinearity N = ALL - GHG - AAER, "
            f"{WINDOW[0]}-{WINDOW[1]}.\n"
            f"% Built from {DATA_DIR}/ by scripts/make_fig1112_from_csv.py\n"
            "% — do not edit by hand. The caption below says what the numbers\n"
            "% are; \\input this file directly.\n")
        handle.write(table_tex + "\n")
    print(f"[step 8] wrote {table_path}")
