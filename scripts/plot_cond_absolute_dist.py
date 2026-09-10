#!/usr/bin/env python3
"""
The same saturation story told in ABSOLUTE units instead of normalised ones.

Everything so far has been drawn in [-1, 1], where a ceiling is invisible once
you are on it. These two figures put the emissions back in the units the files
carry and draw the CEILINGS on top, so you can see which cells cross them and
when:

  histograms  distribution of the positive cells, pooled over the training cond
              files, one panel per species, log x. Vertical lines mark the v1
              clip and the asinh ceiling at p99.5 (the first arm) and p99.9
              (asinh99). The legend carries the share of MASS above each line —
              the quantity that decides how much of a region is flattened.

  timeseries  per region, the per-cell maximum and the 90th percentile of the
              emitting cells, 1850-2100, against those same ceilings. Where a
              curve sits above a line, that region's strongest cells are pinned.

Units are the cond files' own: Gt CO2 cumulative per gridpoint, Gt SO2/yr and
Gt BC/yr. The regrid deflates extensive sums ~4.7x, so these are the emulator's
input space, NOT real-world emissions — see the regrid-deflation note.

Usage
-----
    python scripts/plot_cond_absolute_dist.py
    python scripts/plot_cond_absolute_dist.py --species SUL BC --scenario ssp126
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_asinh_cond_maps import (
    DEFAULT_DATA_DIR, REGIONS, SPECIES, CLIP_PCTL, FIT_FILES,
    cond_path, region_mask,
)
from sweep_asinh_cond_params import load_record

UNITS = {"CO2": "Gt CO$_2$ cumulative / gridpoint",
         "SUL": "Gt SO$_2$ / yr / gridpoint",
         "BC":  "Gt BC / yr / gridpoint"}
CEILINGS = [("v1 clip", CLIP_PCTL, "tab:red", "-"),
            ("asinh p99.5", 99.5, "tab:orange", "--"),
            ("asinh p99.9 (asinh99)", 99.9, "tab:green", ":")]
BOXES = dict(REGIONS, Arabia=(15, 30, 35, 55))


def pooled_positive(data_dir, species):
    chunks = []
    for tag in FIT_FILES:
        with xr.open_dataset(cond_path(data_dir, tag)) as ds:
            a = np.asarray(ds[species].values, dtype=float).ravel()
        chunks.append(a[np.isfinite(a) & (a > 0)])
    return np.concatenate(chunks)


def ceiling_values(data_dir, species, pos):
    """The three ceilings in ABSOLUTE units, on one pooled distribution."""
    with_all = []
    for tag in FIT_FILES:
        with xr.open_dataset(cond_path(data_dir, tag)) as ds:
            a = np.asarray(ds[species].values, dtype=float).ravel()
        with_all.append(a[np.isfinite(a)])
    v1_hi = float(np.percentile(np.concatenate(with_all), CLIP_PCTL[species][1]))
    return [("v1 clip", v1_hi, "tab:red", "-"),
            ("asinh p99.5", float(np.percentile(pos, 99.5)), "tab:orange", "--"),
            ("asinh p99.9 (asinh99)", float(np.percentile(pos, 99.9)), "tab:green", ":")]


def draw_histograms(data_dir, species, outdir):
    fig, axes = plt.subplots(1, len(species), figsize=(4.6 * len(species), 3.8),
                             constrained_layout=True, squeeze=False)
    for ax, v in zip(axes[0], species):
        pos = pooled_positive(data_dir, v)
        # The regrid leaves a floor of denormal-scale values (down to 1e-51)
        # that carry no mass and would otherwise stretch the axis over 50
        # decades. Show the top 10 decades and say what was left out.
        floor = pos.max() * 1e-10
        shown = pos[pos >= floor]
        below = 100.0 * (pos.size - shown.size) / pos.size
        bins = np.logspace(np.log10(floor), np.log10(pos.max()), 120)
        ax.hist(shown, bins=bins, color="0.6", edgecolor="none")
        ax.set_xlim(floor, pos.max())
        ax.text(0.02, 0.04, f"{below:.0f}% of emitting cells below axis "
                            f"(<1e-10 of the maximum)",
                transform=ax.transAxes, fontsize=7, va="bottom")
        ax.set_xscale("log"); ax.set_yscale("log")
        total = pos.sum()
        for label, val, colour, style in ceiling_values(data_dir, v, pos):
            share = 100.0 * pos[pos >= val].sum() / total
            ax.axvline(val, color=colour, ls=style, lw=1.6,
                       label=f"{label}: {share:.0f}% of mass above")
        ax.set_title(v, fontsize=11)
        ax.set_xlabel(UNITS[v], fontsize=9)
        ax.set_ylabel("emitting cells", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Conditioning cells in absolute units, with the clip ceilings "
                 "(hist + ssp370 + aaer + ghg)", fontsize=11)
    path = os.path.join(outdir, "cond_absolute_histograms.png")
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def draw_timeseries(data_dir, scenario, species, regions, outdir):
    fig, axes = plt.subplots(len(species), len(regions),
                             figsize=(3.0 * len(regions), 2.7 * len(species)),
                             constrained_layout=True, squeeze=False,
                             sharex=True, sharey="row")
    for i, v in enumerate(species):
        field, years, lat, lon = load_record(data_dir, scenario, v)
        pos = pooled_positive(data_dir, v)
        ceils = ceiling_values(data_dir, v, pos)
        for j, name in enumerate(regions):
            m = region_mask(lat, lon, BOXES[name])
            cells = field[:, m]
            hi = cells.max(axis=1)
            p90 = np.array([np.percentile(c[c > 0], 90) if (c > 0).any() else np.nan
                            for c in cells])
            ax = axes[i][j]
            ax.plot(years, hi, color="k", lw=1.3, label="strongest cell")
            ax.plot(years, p90, color="tab:blue", lw=1.1, label="p90 of emitting cells")
            for label, val, colour, style in ceils:
                ax.axhline(val, color=colour, ls=style, lw=1.2, label=label)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{v}\n{UNITS[v]}", fontsize=8)
            if i == len(species) - 1:
                ax.set_xlabel("year", fontsize=9)
        del field
    axes[0][-1].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"Regional cell values against the clip ceilings — hist + {scenario}",
                 fontsize=11)
    path = os.path.join(outdir, f"cond_absolute_timeseries_{scenario}.png")
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--scenario", default="ssp370")
    ap.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    ap.add_argument("--regions", nargs="+",
                    default=["E China", "India", "Arabia", "Europe", "E US"])
    ap.add_argument("--outdir", default="plots/asinh_cond")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("wrote", draw_histograms(args.data_dir, args.species, args.outdir))
    print("wrote", draw_timeseries(args.data_dir, args.scenario, args.species,
                                   args.regions, args.outdir))


if __name__ == "__main__":
    main()
