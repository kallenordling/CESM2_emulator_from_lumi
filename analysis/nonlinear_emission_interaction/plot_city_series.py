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
RESULT = os.path.join(HERE, "results", "city_series.npz")
FIGDIR = os.path.join(HERE, "figures")
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
    if ci == 0:
        ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper left")
for ax in axes[1]:
    ax.set_xlabel("year")
fig.suptitle(
    "Near-surface temperature at four cities, every TRAINING experiment — "
    "solid = emulator (5 members), dashed = CESM2\n"
    "band = min-max across members. Member counts differ (CESM2: 11 hist, "
    "10 aaer, 10 ghg, only 3 ssp370), so band widths\nare not comparable "
    "between experiments. Nearest gridpoint on the 192x288 grid.",
    fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for p in (os.path.join(FIGDIR, "figure_15_city_timeseries.png"),
          os.path.join(FIGDIR, "figure_15_city_timeseries.pdf")):
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
            stats[(city, e)] = (m.mean(), cm.mean(), m.std(), cm.std())
    la, lo = cells[ci]
    ax.set_title(f"{city}   (cell {la:+.2f}, {lo:+.2f})", fontsize=11,
                 loc="left", pad=5, fontweight="bold")
    ax.set_xlabel("TREFHT  [°C]")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25, lw=0.5)
    if ci == 0:
        ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper left")
fig.suptitle(
    f"Distribution over the LAST {NLAST} YEARS of each training experiment — "
    "step = emulator, filled/dashed = CESM2\n"
    "hist ends 2014, ssp370 and the single-forcing runs end 2100, so the "
    "panels compare different periods per experiment.\n"
    "All members pooled; the emulator contributes 5 x 20 samples, CESM2 up to "
    "11 x 20.",
    fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for p in (os.path.join(FIGDIR, "figure_16_city_histograms.png"),
          os.path.join(FIGDIR, "figure_16_city_histograms.pdf")):
    fig.savefig(p, bbox_inches="tight"); print(f"[plot] wrote {p}")
plt.close(fig)

print(f"\n[plot] last-{NLAST}-year mean and sd, emulator vs CESM2 [°C]:")
print(f"{'city':>11s} {'experiment':>8s} | {'emu mean':>9s} {'cesm':>8s} "
      f"{'bias':>7s} | {'emu sd':>7s} {'cesm sd':>8s}")
for (city, e), (mm, cc, ms, cs) in stats.items():
    print(f"{city:>11s} {e:>8s} | {mm:>9.2f} {cc:>8.2f} {mm - cc:>+7.2f} | "
          f"{ms:>7.2f} {cs:>8.2f}")
