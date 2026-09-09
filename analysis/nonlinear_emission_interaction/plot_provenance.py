#!/usr/bin/env python3
"""Figure 19 — does the conditioning agree with raw input4MIPs?

Three levels of the same quantity, one column per species:

  row 1  RAW    input4MIPs, all sectors, area-integrated to Tg/yr
  row 2  COND   the *_bc_co2fix.nc files the training config points at,
                area-integrated the same way, in the cond file's own units
  row 3  RATIO  RAW / COND

The RATIO row is the point of the figure. A roughly constant ratio means the
cond files are a faithfully rescaled copy of input4MIPs and the emulator is
self-consistent in its own units -- the recorded ~4.7x deflation from a regrid
that treats an extensive field as intensive. A ratio that DRIFTS with time, or
differs between scenarios, would mean the cond files misrepresent the shape of
the forcing trajectory, which no amount of internal consistency would excuse
and which would invalidate every cross-scenario comparison.

Two things the reader has to know to read it:

  * historical BC is CEDS-2025 while SO2 is CEDS-2017. That asymmetry is in the
    cond files themselves, so the raw side is pinned to match; a glob over both
    vintages double-counts 1850-2014, which is how the first run of the dump
    reported 439 years of a 274-year record.
  * ScenarioMIP anthro files are stored as DECADAL means, so the ssp370 raw
    line has ~10 points over 2015-2100. That is the file's resolution.

CO2 is absent: the input4MIPs CO2 files staged here are CO2-em-AIR-anthro,
i.e. aviation only, which is not what the CO2 cond channel is built from.

    ~/miniconda3/envs/plotting/bin/python \\
        analysis/nonlinear_emission_interaction/plot_provenance.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULT = os.path.join(HERE, "results", "provenance.npz")
FIGDIR = os.path.join(HERE, "figures")
PAPER = "fig19"

if not os.path.exists(RESULT):
    sys.exit(f"[error] {RESULT} not found — run dump_provenance.py on LUMI")

d = np.load(RESULT, allow_pickle=True)
CHANS = [c for c in ("BC", "SUL") if f"raw_tg_hist_{c}" in d.files]
SCEN = [s for s in ("hist", "ssp370") if f"cond_years_{s}" in d.files]
COL = {"hist": "#2F5D7C", "ssp370": "#B4451F"}
VINT = {"BC": "CEDS-2025", "SUL": "CEDS-2017"}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})
os.makedirs(FIGDIR, exist_ok=True)

fig, axes = plt.subplots(3, len(CHANS), figsize=(6.6 * len(CHANS), 9.4),
                         sharex=True)
if len(CHANS) == 1:
    axes = axes[:, None]

summary = []
for c, ch in enumerate(CHANS):
    ax_raw, ax_cond, ax_rat = axes[0][c], axes[1][c], axes[2][c]
    for s in SCEN:
        rk, tk = f"raw_years_{s}_{ch}", f"raw_tg_{s}_{ch}"
        if rk not in d.files:
            continue
        ry, rt = d[rk], d[tk]
        cy, ci = d[f"cond_years_{s}"], d[f"cond_int_{s}_{ch}"]
        col = COL[s]
        mk = "o" if s == "ssp370" else None       # decadal points are sparse
        ax_raw.plot(ry, rt, color=col, lw=1.6, marker=mk, ms=3.5, label=s)
        ax_cond.plot(cy, ci, color=col, lw=1.6, label=s)

        # ratio on the years both actually have
        common = np.intersect1d(ry, cy)
        if len(common) >= 3:
            r = rt[np.searchsorted(ry, common)] / ci[np.searchsorted(cy, common)]
            ax_rat.plot(common, r, color=col, lw=1.6, marker=mk, ms=3.5, label=s)
            lo, hi = np.percentile(r, [5, 95])
            summary.append((ch, s, float(np.median(r)), float(lo), float(hi),
                            float(hi / lo)))

    ax_raw.set_title(f"{ch}   (raw vintage: {VINT.get(ch, '?')})", fontsize=11,
                     loc="left", pad=5, fontweight="bold")
    ax_raw.set_ylabel("RAW input4MIPs  [Tg/yr]")
    ax_cond.set_ylabel("COND file, area-integrated\n[file units]")
    ax_rat.set_ylabel("RATIO  raw / cond")
    ax_rat.set_xlabel("year")
    for a in (ax_raw, ax_cond, ax_rat):
        a.grid(alpha=0.25, lw=0.5)
    ax_raw.legend(frameon=False, fontsize=8)
    # annotate how constant the ratio is — the actual question
    txt = []
    for chh, s, med, lo, hi, spread in summary:
        if chh != ch:
            continue
        txt.append(f"{s:>7s}  median {med:.3g}   p5-p95 spread {spread:.2f}x")
    if txt:
        ax_rat.text(0.02, 0.04, "\n".join(txt), transform=ax_rat.transAxes,
                    ha="left", va="bottom", fontsize=7.6, family="monospace",
                    bbox=dict(fc="white", ec="0.75", lw=0.6, alpha=0.9, pad=3))

fig.suptitle(
    "Provenance check: raw input4MIPs against the conditioning the emulator "
    "reads\n"
    "The RATIO row is the test. Constant = the cond files are a faithfully "
    "rescaled copy and the emulator is self-consistent in its own units.\n"
    "Drifting or scenario-dependent = the cond files misrepresent the shape of "
    "the forcing, which internal consistency would not excuse.\n"
    "ssp370 raw is DECADAL (the file's own resolution). CO2 omitted: the "
    "staged input4MIPs CO2 is aviation-only, not the cond channel's source.",
    fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.93))

paths = [os.path.join(FIGDIR, f"figure_19_provenance{e}") for e in (".png", ".pdf")]
pd = os.path.join(REPO, "plots", PAPER)
os.makedirs(pd, exist_ok=True)
paths += [os.path.join(pd, PAPER + e) for e in (".png", ".pdf")]
for p in paths:
    fig.savefig(p, bbox_inches="tight"); print(f"[plot] wrote {p}")
plt.close(fig)

print("\n[plot] raw/cond ratio — constant is what we want:")
print(f"{'channel':>8s} {'scenario':>8s} {'median':>12s} {'p5':>12s} "
      f"{'p95':>12s} {'spread':>8s}")
for ch, s, med, lo, hi, spread in summary:
    flag = "  <== NOT constant" if spread > 1.25 else ""
    print(f"{ch:>8s} {s:>8s} {med:>12.4g} {lo:>12.4g} {hi:>12.4g} "
          f"{spread:>7.2f}x{flag}")
