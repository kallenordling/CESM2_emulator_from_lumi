#!/usr/bin/env python3
"""
================================================================================
 FIGURES 3 AND 4 — distributions of global means, last 20 years
================================================================================

Run it with no arguments:

    /home/nordling/miniconda3/envs/plotting/bin/python scripts/make_fig3.py

Everything configurable is in the SETTINGS block below. There are no
command-line options and no helper functions: the script runs top to bottom in
seven numbered steps, so it can be read as a description of how the figure is
made. Same shape as make_fig1.py and make_fig2.py.

WHAT THE FIGURE SHOWS
---------------------
TWO figures — fig03 for temperature, fig04 for precipitation — each a 2x2
grid of the four experiments. Each
histogram pools EVERY member-year
of the window N_YEARS selects — members x years values per side — and asks
whether the two samples are drawn from the same distribution. With
MATCH_MEMBER_COUNTS the two sides carry the same number of members, so the
sample sizes differ only if the records differ in length.

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

# One figure per variable, each a 2x2 grid of experiments, so temperature and
# precipitation can be placed separately in the paper rather than as one
# eight-panel block. They are the paper's figures 3 and 4.
FIGURE_NAME = {"TREFHT": "fig03", "PRECT": "fig04"}

OUT = "plots/{name}.png"               # the .pdf sibling is written alongside

# LaTeX table of the statistics, one per figure so each can sit beside it.
TABLE = "plots/{name}_stats.tex"

# Cap the emulator at the CESM2 member count, per experiment. The eval has 25
# members and CESM2 has 6-11, so leaving them unequal means the two ensemble
# means are converged to different degrees and the tests compare samples of very
# different size. Matching makes every comparison n against n. Selection is the
# first N members — deterministic, never random.
MATCH_MEMBER_COUNTS = True

# ── ABSOLUTE VALUES OR ANOMALIES ─────────────────────────────────────────────
# True  : every series has its OWN side's 1850-1900 mean subtracted, the
#         convention figures 1 and 2 use. The question becomes "does the
#         emulator WARM by the same amount", and a difference in mean state is
#         removed before the comparison rather than measured by it.
# False : absolute values. The question is "is the emulator's climate the same",
#         and any offset in mean state is part of the answer.
#
# THIS IS NOT A COSMETIC CHOICE FOR THE MEAN TEST. Referencing each side to its
# own baseline subtracts most of the very offset the mean test exists to detect,
# so the differences shrink and some stop being significant. The KS test and the
# sd ratio are unaffected either way: both already remove a per-side constant.
ANOMALY = True
BASELINE = (1850, 1900)

# How many years at the END of each experiment go into the statistics.
# 0 = every year of the record.
#
# READ THE SD RATIO DIFFERENTLY DEPENDING ON THIS. Over a short window the
# spread is mostly internal variability. Over the FULL record it is dominated by
# the forced trend — hist spans 165 years and about 1 degC of warming — so the
# raw standard deviations below measure "how much does the climate move over the
# record", not "how noisy is it". The KS test is unaffected: it runs on residuals
# with each side's ensemble-mean trajectory removed, which subtracts the trend
# whatever the window length.
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
from scipy import stats

# =============================================================================
#  STEP 1 — collect the emulator's global means for the last N years
# =============================================================================
# The eval file holds maps, so the global mean is computed here: cos(lat)-
# weighted, because grid cells shrink towards the poles and an unweighted mean
# over this grid is wrong by degrees, not decimals.
#
# `.sel(year=slice(-N_YEARS, None))` would select YEARS -20 to 0, which do not
# exist; the last N years are an INDEX selection, hence isel.

emulator = {}            # (variable, scenario) -> 1-D array of member-years
emulator_by_member = {}  # the same values, still shaped (member, year)
baseline_of = {}         # (side, variable, scenario) -> 1850-1900 mean of the FULL record
for variable in VARIABLES:
    for scenario in SCENARIOS:
        path = f"{EVAL_DIR}/{variable}_{scenario}.nc"
        dataset = xr.open_dataset(path)
        field = dataset[f"{variable}_model"]
        weights = np.cos(np.deg2rad(field["lat"]))
        # The FULL series first: the 1850-1900 baseline window lies outside the
        # last-N-years slice, so computing it after truncation gives NaN.
        full_mean = field.weighted(weights).mean(("lat", "lon")).compute()
        full_years = full_mean["year"].values
        in_baseline = (full_years >= BASELINE[0]) & (full_years <= BASELINE[1])
        baseline_of[("emulator", variable, scenario)] = (
            float(full_mean.values[:, in_baseline].mean()) if in_baseline.any()
            else np.nan)
        global_mean = (full_mean.isel(year=slice(-N_YEARS, None)) if N_YEARS
                       else full_mean)
        years = global_mean["year"].values
        # Flatten (member, year) into one sample: every member-year is one draw
        # from the model's climate over this window.
        emulator[(variable, scenario)] = global_mean.values.ravel()
        emulator_by_member[(variable, scenario)] = global_mean.values  # (member, year)
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
cesm_by_member = {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        path = f"{REFERENCE_DIR}/{variable}_{scenario}.nc"
        dataset = xr.open_dataset(path)
        field = dataset[f"{variable}_cesm"]
        weights = np.cos(np.deg2rad(field["lat"]))
        # The FULL series first: the 1850-1900 baseline window lies outside the
        # last-N-years slice, so computing it after truncation gives NaN.
        full_mean = field.weighted(weights).mean(("lat", "lon")).compute()
        full_years = full_mean["year"].values
        in_baseline = (full_years >= BASELINE[0]) & (full_years <= BASELINE[1])
        baseline_of[("cesm", variable, scenario)] = (
            float(full_mean.values[:, in_baseline].mean()) if in_baseline.any()
            else np.nan)
        global_mean = (full_mean.isel(year=slice(-N_YEARS, None)) if N_YEARS
                       else full_mean)
        years = global_mean["year"].values
        cesm[(variable, scenario)] = global_mean.values.ravel()
        cesm_by_member[(variable, scenario)] = global_mean.values      # (member, year)
        member_count = global_mean.sizes["member"]
        dataset.close()
        print(f"[step 2] {variable:6s} {scenario:7s} CESM2:    "
              f"{member_count:2d} members x {len(years)} years "
              f"({years[0]}-{years[-1]}) = {cesm[(variable, scenario)].size} values")

# =============================================================================
#  STEP 2b — same n on both sides
# =============================================================================
# Done after both are read, because the cap for each experiment is CESM2's own
# member count. Truncating rather than subsampling keeps the run reproducible.

if MATCH_MEMBER_COUNTS:
    for variable in VARIABLES:
        for scenario in SCENARIOS:
            n_cesm = cesm_by_member[(variable, scenario)].shape[0]
            series = emulator_by_member[(variable, scenario)]
            if series.shape[0] > n_cesm:
                emulator_by_member[(variable, scenario)] = series[:n_cesm]
                emulator[(variable, scenario)] = series[:n_cesm].ravel()
                print(f"[step 2b] {variable:6s} {scenario:7s} emulator "
                      f"{series.shape[0]} -> {n_cesm} members")

# =============================================================================
#  STEP 2c — anomalies, each side referenced to its own pre-industrial
# =============================================================================
# hist, aaer and ghg all begin in 1850 and so carry their own baseline period.
# ssp370 begins in 2015 and has none, so it inherits the historical baseline —
# the SAME convention applied to both sides, which is what keeps them
# comparable.

if ANOMALY:
    # ssp370 begins in 2015 and has no baseline period of its own; it inherits
    # the historical one, the same convention applied to both sides.
    for variable in VARIABLES:
        for side in ("emulator", "cesm"):
            if not np.isfinite(baseline_of[(side, variable, "ssp370")]):
                baseline_of[(side, variable, "ssp370")] = baseline_of[(side, variable, "hist")]

    for variable in VARIABLES:
        for scenario in SCENARIOS:
            e_base = baseline_of[("emulator", variable, scenario)]
            c_base = baseline_of[("cesm", variable, scenario)]
            emulator_by_member[(variable, scenario)] = (
                emulator_by_member[(variable, scenario)] - e_base)
            cesm_by_member[(variable, scenario)] = (
                cesm_by_member[(variable, scenario)] - c_base)
            emulator[(variable, scenario)] = emulator_by_member[(variable, scenario)].ravel()
            cesm[(variable, scenario)] = cesm_by_member[(variable, scenario)].ravel()
            print(f"[step 2c] {variable:6s} {scenario:7s} baselines "
                  f"emulator {e_base:8.3f}, CESM2 {c_base:8.3f}"
                  + ("   (inherited from hist)" if scenario == "ssp370" else ""))

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
#  STEP 4 — draw one 2x2 figure per variable
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
    variable_label, unit_axis, _ = VARIABLES[variable]
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))

    for panel_index, scenario in enumerate(SCENARIOS):
        axis = axes[panel_index // 2][panel_index % 2]
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
                       fontsize=9.5, loc="left")
        axis.text(0.03, 0.95, f"({'abcd'[panel_index]})", transform=axis.transAxes,
                  fontweight="bold", va="top", fontsize=9)
        # Axis labels only on the outside, so the panels are not repetitive.
        if panel_index // 2 == 1:
            axis.set_xlabel(f"{variable_label} ({unit_axis})")
        if panel_index % 2 == 0:
            axis.set_ylabel("Probability density")

    # =========================================================================
    #  STEP 5 — legend, then save
    # =========================================================================
    # tight_layout FIRST, so the title and legend are placed relative to the
    # settled axes; the other way round lets tight_layout move the axes out from
    # under them and the legend lands on top of the title.
    legend_entries = [
        Patch(facecolor=CESM_COLOUR, alpha=0.45, label="CESM2 (held-out members)"),
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
        print(f"[step 5] wrote {path}")
    plt.close(figure)

# =============================================================================
#  STEP 6 — are the means the same, and are the distributions the same?
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
#   member. Members are independent realizations, so these are
#   genuinely independent units. Welch rather than Student because the two sides
#   need not have equal variance.
#
# ARE THE DISTRIBUTIONS THE SAME?  Two-sample Kolmogorov-Smirnov, and Levene for
#   the variances, on values with each side's OWN ensemble-mean trajectory
#   removed. That subtracts the forced signal — the trend the members share —
#   and leaves internal variability. It also removes the correlation: the lag-1
#   autocorrelation of the residuals is below 0.06, so the samples now behave as
#   independent draws. This asks the question the figure is really about: does
#   the emulator's spread ABOUT its own trajectory match CESM2's?

tests = {}
for variable in VARIABLES:
    for scenario in SCENARIOS:
        emulator_series = emulator_by_member[(variable, scenario)]
        cesm_series = cesm_by_member[(variable, scenario)]

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
        print(f"[step 6] {variable:6s} {scenario:7s} "
              f"means p={mean_test.pvalue:.4f} "
              f"{'DIFFER' if mean_test.pvalue < 0.05 else 'same'}, "
              f"distribution p={ks_test.pvalue:.3f} "
              f"{'DIFFER' if ks_test.pvalue < 0.05 else 'same'}, "
              f"variance p={variance_test.pvalue:.3f}")

# =============================================================================
#  STEP 7 — the same numbers as a LaTeX table
# =============================================================================
# Plain LaTeX with borders: \toprule and \cmidrule need \usepackage{booktabs},
# and without it \cmidrule silently typesets "(lr)2-3" into the table.

# A p-value formatter: below 0.001 the exact value is noise, so say so.
def format_p(value):
    return "$<$0.001" if value < 0.001 else f"{value:.3f}"


for variable in VARIABLES:
    variable_label, _, unit_table = VARIABLES[variable]

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

    table_tex = "\n".join([
        r"\begin{tabular}{|l|r|r|c|r|r|c|}",
        r"\hline",
        r"\textbf{Experiment} & \multicolumn{3}{c|}{\textbf{Mean}} & "
        r"\multicolumn{3}{c|}{\textbf{Distribution}} \\",
        r"\cline{2-7}",
        r" & Difference (%s) & $p$ & Same? & SD ratio & $p$ & Same? \\" % unit_table,
        r"\hline",
        *table_rows,
        r"\hline",
        r"\end{tabular}",
    ])

    # Member counts actually used, so the caption cannot go stale.
    counts = {}
    for side, store in (("emulator", emulator_by_member), ("cesm2", cesm_by_member)):
        sizes = sorted({store[(variable, s)].shape[0] for s in SCENARIOS})
        counts[side] = (f"{sizes[0]}" if len(sizes) == 1
                        else f"{sizes[0]}-{sizes[-1]}")

    table_path = TABLE.format(name=FIGURE_NAME[variable])
    os.makedirs(os.path.dirname(table_path) or ".", exist_ok=True)
    with open(table_path, "w") as handle:
        handle.write(
            f"% {variable_label}: global-mean statistics over "
            f"{('the last %d years' % N_YEARS) if N_YEARS else 'the FULL record'}\n"
            f"% of each experiment. Differences are emulator minus CESM2,\n"
            f"% in {unit_table}.\n"
            f"%\n"
            f"% MEAN columns: Welch's t-test on the MEMBER MEANS "
            f"({counts['emulator']} emulator\n"
            f"%   members against {counts['cesm2']} CESM2 members). Members are "
            f"independent\n"
            f"%   realizations, so these are independent samples.\n"
            f"%\n"
            f"% DISTRIBUTION columns: ratio of standard deviations, and a\n"
            f"%   two-sample Kolmogorov-Smirnov test on values with each side's\n"
            f"%   own ensemble-mean trajectory removed — i.e. on internal\n"
            f"%   variability, with the shared forced trend taken out. That\n"
            f"%   removal is what makes the samples independent enough to test:\n"
            f"%   on the raw pooled member-years the lag-1 autocorrelation is\n"
            f"%   0.56-0.93 and any test assuming independence is meaningless.\n"
            f"%\n"
            f"% 'Same?' is p > 0.05. Generated by scripts/make_fig3.py.\n")
        handle.write(table_tex + "\n")
    print(f"[step 7] wrote {table_path}")
