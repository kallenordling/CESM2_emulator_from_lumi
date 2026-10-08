#!/usr/bin/env python3
"""Figures for the regional error budget (Analysis A).

Two figures per variable, both driven by error_budget_regions.csv:

  maps  AR6 land regions filled with the DIFFERENTIAL error -- the scenario's
        error minus ssp370's -- one panel per scenario. Regions whose error
        does not clear 2x the CESM2 noise floor are hatched, so a reader
        cannot mistake an unresolved region for a resolved one. Filling by raw
        error instead would mostly paint the common bias, which for
        precipitation is the ITCZ dipole present in every scenario.

  bars  the ten regions carrying the most squared error, with their
        |error|/floor printed. Bars below the 2x line are drawn hollow.

RAMIP temperature has ONE CESM2 member, so its floor is a spatial proxy rather
than an ensemble spread; its panel is labelled NOISE-LIMITED. Its errors are
3-7x that proxy and spatially coherent, but it is the weakest evidence here
and the figure says so rather than letting the colour imply otherwise.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_error_budget.py
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:                                            # noqa: BLE001
    HAVE_CARTOPY = False
import regionmask

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--csv", default="plots/error_budget/error_budget_regions.csv")
ap.add_argument("--outdir", default="plots/error_budget")
ap.add_argument("--top", type=int, default=10)
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

rows = list(csv.DictReader(open(args.csv)))
for r in rows:
    for k in ("raw", "diff", "floor", "ratio", "share"):
        try:
            r[k] = float(r[k])
        except (ValueError, TypeError):
            r[k] = np.nan
    r["n_ref"] = int(r["n_ref"])

VARS = sorted({r["var"] for r in rows})
SCEN = ["ssp126", "ssp245", "ssp370-126aer"]
LABEL = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370-126aer": "RAMIP ssp370-126aer"}
UNIT = {"TREFHT": "degC", "PRECT": "mm/day"}
ar6 = regionmask.defined_regions.ar6.land
NAME2ID = {ar6[i].name: ar6[i].number for i in range(len(ar6))}

for var in VARS:
    # ── map figure ───────────────────────────────────────────────────────────
    sub = [r for r in rows if r["var"] == var]
    use_diff = not all(np.isnan(r["diff"]) for r in sub)
    key = "diff" if use_diff else "raw"
    vmax = np.nanpercentile([abs(r[key]) for r in sub], 95) or 1.0

    proj = {"projection": ccrs.Robinson()} if HAVE_CARTOPY else {}
    fig, axes = plt.subplots(1, len(SCEN), figsize=(5.0 * len(SCEN), 3.4),
                             constrained_layout=True, subplot_kw=proj)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, SCEN):
        d = {r["region"]: r for r in sub if r["scenario"] == sc}
        if not d:
            ax.axis("off"); continue
        vals = np.full(len(ar6), np.nan)
        sig = np.zeros(len(ar6), bool)
        for nm, r in d.items():
            if nm in NAME2ID:
                vals[NAME2ID[nm]] = r[key]
                sig[NAME2ID[nm]] = r["ratio"] >= 2.0
        cmap = plt.get_cmap("RdBu_r")
        norm = plt.Normalize(-vmax, vmax)
        for i in range(len(ar6)):
            if np.isnan(vals[i]):
                continue
            poly = ar6[i].polygon
            geoms = getattr(poly, "geoms", [poly])
            for g in geoms:
                xs, ys = g.exterior.xy
                kw = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
                ax.fill(xs, ys, color=cmap(norm(vals[i])), **kw)
                if not sig[i]:                       # unresolved -> hatched
                    ax.fill(xs, ys, facecolor="none", hatch="////",
                            edgecolor="0.35", linewidth=0.0, **kw)
                ax.plot(xs, ys, color="0.3", lw=0.3, **kw)
        if HAVE_CARTOPY:
            try:
                ax.coastlines(linewidth=0.35, color="0.2"); ax.set_global()
            except Exception as exc:                  # noqa: BLE001
                print(f"[plot] coastline failed: {exc}")
        n = d[next(iter(d))]["n_ref"]
        note = "  NOISE-LIMITED" if n == 1 else ""
        ax.set_title(f"{LABEL[sc]}\n{n} CESM2 member{'s' if n > 1 else ''}{note}",
                     fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
    lab = ("differential error (scenario - ssp370)" if use_diff else "raw error")
    fig.colorbar(sm, ax=list(axes), shrink=0.75, label=f"{lab} [{UNIT[var]}]")
    fig.suptitle(f"{var}: where each unseen scenario's error lives "
                 f"(AR6 land regions; hatched = below 2x the CESM2 noise floor)",
                 fontsize=11)
    out = os.path.join(args.outdir, f"error_budget_map_{var}")
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png/.pdf")

    # ── bar figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(SCEN), figsize=(4.8 * len(SCEN), 4.2),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, SCEN):
        d = sorted([r for r in sub if r["scenario"] == sc],
                   key=lambda r: -r["share"])[:args.top]
        if not d:
            ax.axis("off"); continue
        y = np.arange(len(d))[::-1]
        for yi, r in zip(y, d):
            resolved = r["ratio"] >= 2.0
            ax.barh(yi, r["share"],
                    color=("#B4451F" if r[key] > 0 else "#2F5D7C") if resolved else "none",
                    edgecolor=("#B4451F" if r[key] > 0 else "#2F5D7C"),
                    hatch=None if resolved else "///")
            ax.text(r["share"] + 0.4, yi, f"{r['ratio']:.1f}x", va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels([r["region"] for r in d], fontsize=8)
        ax.set_xlabel("share of global squared error (%)", fontsize=8)
        ax.set_title(LABEL[sc], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"{var}: regions carrying the error  (filled = clears 2x the "
                 f"noise floor, hollow = does not; label is |error|/floor)",
                 fontsize=11)
    out = os.path.join(args.outdir, f"error_budget_bars_{var}")
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png/.pdf")
