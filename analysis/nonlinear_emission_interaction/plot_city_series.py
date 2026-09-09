#!/usr/bin/env python3
"""Figures 15 and 16 — four cities, every training experiment.

  figure_15_city_timeseries  TREFHT at each city, 1850-2100, emulator against
                             CESM2, one colour per experiment
  figure_16_city_histograms  distribution over the LAST 20 YEARS of each
                             experiment, same four cities

Point series rather than global means on purpose: a global mean averages away
both the aerosol fingerprint and most of the ensemble spread, and the question
here is whether the local climate is right in places with very different
forcing histories -- Helsinki (high-latitude, strong European sulfate history),
Tokyo (Asian aerosol), Sydney (Southern Hemisphere, little local aerosol),
Sao Paulo (tropical, biomass burning).

Training experiments only. hist and ssp370 are the historical/future pair;
aaer holds greenhouse gases at 1850 and varies aerosols, ghg does the reverse,
so the two single-forcing runs are where an aerosol error shows up undiluted.

Member counts differ a lot -- CESM2 has 11 hist, 10 aaer, 10 ghg but only 3
ssp370, against 5 emulator members throughout -- so the shaded bands are NOT
comparable in width between experiments. The band is min-max across members.

    ~/miniconda3/envs/plotting/bin/python \\
        analysis/nonlinear_emission_interaction/plot_city_series.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULT = os.path.join(HERE, "results", "city_series.npz")
FIGDIR = os.path.join(HERE, "figures")
# Paper figures follow the repo convention plots/figNN/figNN.{png,pdf}. The
# previous occupants of 05/06 were moved to plots/archive, so these slots are
# free; scripts/make_fig5.py still NAMES its outputs fig05/fig06 though, so
# re-running it would overwrite these.
PAPER = {"timeseries": "fig05", "histograms": "fig06"}


def outputs(kind, local_stem):
    """Both the working copy and the numbered paper copy."""
    paths = [os.path.join(FIGDIR, local_stem + ext) for ext in (".png", ".pdf")]
    n = PAPER.get(kind)
    if n:
        d = os.path.join(REPO, "plots", n)
        os.makedirs(d, exist_ok=True)
        paths += [os.path.join(d, n + ext) for ext in (".png", ".pdf")]
    return paths
if not os.path.exists(RESULT):
    sys.exit(f"[error] {RESULT} not found — run dump_city_series.py on LUMI")

d = np.load(RESULT, allow_pickle=True)
cities = [str(c) for c in d["cities"]]
cells = d["cell_latlon"]
exps = [e for e in (str(x) for x in d["experiments"]) if f"model_{e}" in d.files]

COL = {"hist": "#2F5D7C", "ssp370": "#B4451F",
       "aaer": "#2F7A4F", "ghg": "#7B5EA7"}
LBL = {"hist": "hist", "ssp370": "ssp370",
       "aaer": "aaer (GHG fixed at 1850)", "ghg": "ghg (aerosol fixed at 1850)"}

SMOOTH_N = 10


def _smooth(x, n=SMOOTH_N):
    """Centred n-year running mean, FULLY COVERED windows only.

    mode="valid" rather than "same": with "same" the edge windows are divided
    by the full width even though they are only partly filled, which biases the
    first and last n/2 points toward zero and would show up as a spurious
    trend at both ends of the correlation.
    """
    return np.convolve(x, np.ones(n) / n, mode="valid")


def skill(e):
    """r and RMSE between the two ENSEMBLE MEANS on their overlapping years.

    The emulator's members are different realisations from CESM2's, so
    internal variability is unsynchronised and CANNOT correlate. r on the
    annual series therefore measures agreement on the FORCED signal only, and
    is diluted by noise that no model could match; r10 (both series smoothed
    with a 10-year running mean) isolates that forced part. RMSE is on the
    annual ensemble means, so it carries both the forced error and the
    residual noise of a 5-member vs 3-to-11-member mean.
    """
    if f"cesm_{e}" not in d.files:
        return None
    my, cy = d[f"years_{e}"], d[f"cesm_years_{e}"]
    common = np.intersect1d(my, cy)
    if len(common) < 10:
        return None
    mi = np.searchsorted(my, common)
    ci_ = np.searchsorted(cy, common)
    out = {}
    for k in range(len(cities)):
        m = d[f"model_{e}"][:, k, :].mean(0)[mi]
        c = d[f"cesm_{e}"][:, k, :].mean(0)[ci_]
        r = float(np.corrcoef(m, c)[0, 1])
        # "valid" already drops the partial windows, so both series shorten
        # by n-1 identically and stay aligned -- no trimming needed.
        ms, cs_ = _smooth(m), _smooth(c)
        r10 = float(np.corrcoef(ms, cs_)[0, 1]) if len(ms) > 3 else np.nan
        out[k] = (r, r10, float(np.sqrt(np.mean((m - c) ** 2))), len(common))
    return out


SKILL = {e: skill(e) for e in exps}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})
os.makedirs(FIGDIR, exist_ok=True)

# ── Figure 15 — time series ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True)
for ci, city in enumerate(cities):
    ax = axes[ci // 2][ci % 2]
    for e in exps:
        c = COL.get(e, "0.4")
        m, yr = d[f"model_{e}"][:, ci, :], d[f"years_{e}"]
        ax.plot(yr, m.mean(0), color=c, lw=1.7, zorder=4, label=f"{LBL[e]} — emulator")
        ax.fill_between(yr, m.min(0), m.max(0), color=c, alpha=0.16, lw=0, zorder=2)
        if f"cesm_{e}" in d.files:
            cm, cyr = d[f"cesm_{e}"][:, ci, :], d[f"cesm_years_{e}"]
            ax.plot(cyr, cm.mean(0), color=c, lw=1.4, ls="--", zorder=3,
                    label=f"{LBL[e]} — CESM2")
    la, lo = cells[ci]
    ax.set_title(f"{city}   (cell {la:+.2f}, {lo:+.2f})", fontsize=11,
                 loc="left", pad=5, fontweight="bold")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylabel("TREFHT  [°C]")
    rows = [f"{'':>10s} {'r':>6s} {'r10':>6s} {'RMSE':>6s}"]
    for e in exps:
        sk = SKILL.get(e)
        if sk is None:
            continue
        r, r10, rmse, _ = sk[ci]
        rows.append(f"{e:>10s} {r:>6.2f} {r10:>6.2f} {rmse:>6.2f}")
    ax.text(0.015, 0.975, "\n".join(rows), transform=ax.transAxes, ha="left",
            va="top", fontsize=6.9, family="monospace",
            bbox=dict(fc="white", ec="0.75", lw=0.6, alpha=0.9, pad=3))
for ax in axes[1]:
    ax.set_xlabel("year")
# One shared legend under the panels rather than inside the top-left one: the
# entries are identical in all four, and inside it covered the 1850-1950 part
# of the Helsinki series.
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.2,
           bbox_to_anchor=(0.5, -0.045))
fig.suptitle(
    "Near-surface temperature at four cities, every TRAINING experiment — "
    "solid = emulator (5 members), dashed = CESM2\n"
    "band = min-max across members. Member counts differ (CESM2: 11 hist, "
    "10 aaer, 10 ghg, only 3 ssp370), so band widths\nare not comparable "
    "between experiments. Nearest gridpoint on the 192x288 grid.\n"
    "Inset: r and RMSE [°C] between the two ENSEMBLE MEANS. Members are "
    "different realisations, so internal variability cannot correlate — "
    "r is the forced-signal agreement, r10 the same on 10-year means.",
    fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0.035, 1, 0.93))
for p in outputs("timeseries", "figure_15_city_timeseries"):
    fig.savefig(p, bbox_inches="tight"); print(f"[plot] wrote {p}")
plt.close(fig)

# ── Figure 16 — last-20-year distributions ──────────────────────────────────
NLAST = 20
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
stats = {}
for ci, city in enumerate(cities):
    ax = axes[ci // 2][ci % 2]
    for e in exps:
        c = COL.get(e, "0.4")
        m = d[f"model_{e}"][:, ci, -NLAST:].ravel()
        ax.hist(m, bins=18, density=True, histtype="step", lw=1.8, color=c,
                zorder=4, label=f"{LBL[e]} — emulator")
        if f"cesm_{e}" in d.files:
            cm = d[f"cesm_{e}"][:, ci, -NLAST:].ravel()
            ax.hist(cm, bins=18, density=True, histtype="stepfilled", lw=0,
                    color=c, alpha=0.17, zorder=2)
            ax.hist(cm, bins=18, density=True, histtype="step", lw=1.1, ls="--",
                    color=c, zorder=3, label=f"{LBL[e]} — CESM2")
            # A pointwise RMSE between two unpaired sample sets is
            # meaningless -- the members are different realisations and the
            # sample counts differ (5x20 vs up to 11x20). The Q-Q RMSE, the
            # RMS gap between matched quantiles, is a proper distance between
            # the two distributions and reduces to |mean bias| when they
            # differ only by a shift.
            q = np.linspace(0.02, 0.98, 49)
            qq = float(np.sqrt(np.mean(
                (np.quantile(m, q) - np.quantile(cm, q)) ** 2)))
            stats[(city, e)] = (m.mean(), cm.mean(), m.std(), cm.std(), qq)
    la, lo = cells[ci]
    ax.set_title(f"{city}   (cell {la:+.2f}, {lo:+.2f})", fontsize=11,
                 loc="left", pad=5, fontweight="bold")
    ax.set_xlabel("TREFHT  [°C]")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25, lw=0.5)
    rows = [f"{'':>10s} {'bias':>6s} {'sd/sd':>6s} {'QQ':>5s}"]
    for e in exps:
        st = stats.get((city, e))
        if st is None:
            continue
        mm, cc, ms_, cs_, qq = st
        rows.append(f"{e:>10s} {mm - cc:>+6.2f} {ms_ / cs_:>6.2f} {qq:>5.2f}")
    ax.text(0.985, 0.97, "\n".join(rows), transform=ax.transAxes, ha="right",
            va="top", fontsize=6.9, family="monospace",
            bbox=dict(fc="white", ec="0.75", lw=0.6, alpha=0.9, pad=3))
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.2,
           bbox_to_anchor=(0.5, -0.045))
fig.suptitle(
    f"Distribution over the LAST {NLAST} YEARS of each training experiment — "
    "step = emulator, filled/dashed = CESM2\n"
    "hist ends 2014, ssp370 and the single-forcing runs end 2100, so the "
    "panels compare different periods per experiment.\n"
    "All members pooled; the emulator contributes 5 x 20 samples, CESM2 up to "
    "11 x 20.\n"
    "Inset: mean bias [°C], the sd ratio emulator/CESM2, and the Q-Q RMSE — "
    "the RMS gap between matched quantiles, which is a\nproper distance for "
    "unpaired samples where a pointwise RMSE would not be.",
    fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0.035, 1, 0.93))
for p in outputs("histograms", "figure_16_city_histograms"):
    fig.savefig(p, bbox_inches="tight"); print(f"[plot] wrote {p}")
plt.close(fig)

print(f"\n[plot] last-{NLAST}-year mean and sd, emulator vs CESM2 [°C]:")
print(f"{'city':>11s} {'exp':>7s} | {'bias':>6s} {'sd/sd':>6s} {'QQ':>5s} | "
      f"{'r':>6s} {'r10':>6s} {'RMSE':>6s}")
for ci_, city in enumerate(cities):
    for e in exps:
        st = stats.get((city, e))
        sk = SKILL.get(e)
        if st is None or sk is None:
            continue
        mm, cc, ms_, cs_, qq = st
        r, r10, rmse, _ = sk[ci_]
        print(f"{city:>11s} {e:>7s} | {mm - cc:>+6.2f} {ms_ / cs_:>6.2f} "
              f"{qq:>5.2f} | {r:>6.2f} {r10:>6.2f} {rmse:>6.2f}")
