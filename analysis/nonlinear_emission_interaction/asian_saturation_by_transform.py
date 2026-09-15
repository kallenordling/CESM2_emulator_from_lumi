#!/usr/bin/env python3
"""
Does the normalisation decide whether Asian emissions saturate?

No model is involved: saturation is a property of the CONDITIONING PIPELINE, so
this pushes the cond files through the exact stages the network receives --
normalise -> gaussian smoothing -> joint PCA truncation -- under three maps:

  v1      linear, anchored at percentiles of all cells, CLIPPED to [-1, 1]
          (the precip-bc / paper checkpoint normalisation)
  asinh   asinh(v/s) with s = 10% of the positive-cell median, top at p99.9 of
          positive cells, clipped (the asinh99 arm, SUL and BC)
  minmax  linear, anchored at the minimum and maximum of the SMOOTHED field,
          NO clip (the minmax arm)

and tracks the region-mean conditioning from 1850 to 2100 against the region's
actual emissions from the same files.

HOW TO READ IT
Every curve is rescaled to 0-1 over its own record, so the panels compare
SHAPE, not magnitude. A faithful channel follows the black emission curve.
Saturation is the conditioning reaching its ceiling early and FLATTENING while
emissions keep climbing -- the network then cannot tell 1990 from 2040.

THE NUMBER: late-rise retention. Of each curve's full 1850->peak rise, the
share that happens between 1980 and the emission peak. If emissions gain 40%
of their rise after 1980 but a channel gains only 5%, that channel is saturated
over exactly the decades that matter.

    ~/miniconda3/envs/plotting/bin/python \\
        analysis/nonlinear_emission_interaction/asian_saturation_by_transform.py
"""
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
COND_DIR = os.path.expanduser("~/mnt/lumi_sc/emulator_data")
FIT_FILES = ["hist", "ssp370", "aaer", "ghg"]
SPECIES = ["SUL", "BC"]
SIGMA = 2.0
N_COMP = 5
V1_PCTL = (5, 95)
ASINH_SCALE_FRAC, ASINH_TOP_PCTL = 0.10, 99.9
REGIONS = {  # lat0, lat1, lon0, lon1 (0-360)
    "East China":        (20, 45, 100, 123),
    "India":             (8, 30, 70, 90),
    "Arabian Peninsula": (12, 32, 35, 60),
    "Europe":            (40, 60, 350, 30),   # wraps the meridian
}
LATE_FROM = 1980
COLORS = {"v1 (clip)": "#D55E00", "asinh (p99.9)": "#0072B2", "minmax (no clip)": "#009E73"}


def path(tag):
    return f"{COND_DIR}/emissions_{tag}_only_timefixed_bc_co2fix.nc"


def smooth(a):
    a = gaussian_filter1d(a, SIGMA, axis=-1, mode="wrap")
    return gaussian_filter1d(a, SIGMA, axis=-2, mode="reflect")


def box(lat, lon, lat0, lat1, lon0, lon1):
    lm = (lat >= lat0) & (lat <= lat1)
    om = (lon >= lon0) & (lon <= lon1) if lon0 <= lon1 else (lon >= lon0) | (lon <= lon1)
    return lm[:, None] & om[None, :]


def anchors(sp):
    """The three maps' parameters, fitted exactly as training fits them."""
    allc, pos, smax = [], [], 0.0
    for t in FIT_FILES:
        with xr.open_dataset(path(t)) as ds:
            a = np.asarray(ds[sp].values, float)
        f = a[np.isfinite(a)]
        allc.append(f.ravel()); pos.append(f[f > 0])
        smax = max(smax, float(np.nanmax(smooth(a))))
    allc, pos = np.concatenate(allc), np.concatenate(pos)
    lo, hi = np.percentile(allc, V1_PCTL)
    return {"v1": (float(lo), float(hi)),
            "asinh": (float(np.percentile(pos, 50)) * ASINH_SCALE_FRAC,
                      float(np.percentile(pos, ASINH_TOP_PCTL))),
            "minmax": (0.0, smax)}


def transforms(p):
    lo, hi = p["v1"]; s, top = p["asinh"]; mlo, mhi = p["minmax"]
    return {
        "v1 (clip)": lambda x: np.clip((x - (lo + hi) / 2) / ((hi - lo) / 2), -1, 1),
        "asinh (p99.9)": lambda x: np.clip(2 * np.arcsinh(x / s) / np.arcsinh(top / s) - 1, -1, 1),
        "minmax (no clip)": lambda x: 2 * (x - mlo) / (mhi - mlo) - 1,
    }


def unit01(y):
    return (y - y.min()) / (y.max() - y.min() + 1e-30)


def main():
    with xr.open_dataset(path("hist")) as h, xr.open_dataset(path("ssp370")) as s:
        hy, sy = h["year"].values, s["year"].values
        lat, lon = h["lat"].values, h["lon"].values
        raw = {sp: np.concatenate([np.asarray(h[sp].values, float)[hy <= 2014],
                                   np.asarray(s[sp].values, float)[sy >= 2015]]) for sp in SPECIES}
    years = np.concatenate([hy[hy <= 2014], sy[sy >= 2015]])
    masks = {r: box(lat, lon, *b) for r, b in REGIONS.items()}

    series = {}   # (sp, region, name) -> region-mean series
    for sp in SPECIES:
        p = anchors(sp)
        print(f"[fit] {sp}: v1 hi={p['v1'][1]:.3e}  asinh s={p['asinh'][0]:.3e} top={p['asinh'][1]:.3e}  "
              f"minmax hi={p['minmax'][1]:.3e}")
        for r, m in masks.items():
            series[(sp, r, "emissions")] = raw[sp][:, m].mean(axis=1)
        for name, fn in transforms(p).items():
            z = smooth(fn(raw[sp]))                                  # normalise -> smooth
            T, H, W = z.shape
            pca = PCA(n_components=N_COMP).fit(z.reshape(T, -1))     # joint basis, hist+ssp370
            zp = pca.inverse_transform(pca.transform(z.reshape(T, -1))).reshape(T, H, W)
            for r, m in masks.items():
                series[(sp, r, name)] = zp[:, m].mean(axis=1)

    # ---- the number ----------------------------------------------------------
    names = ["emissions"] + list(COLORS)
    print(f"\nLATE-RISE RETENTION: share of each curve's 1850->peak rise that happens "
          f"between {LATE_FROM} and the emission peak")
    print(f"{'species':8s}{'region':20s}{'peak':>6s}" + "".join(f"{n:>18s}" for n in names))
    rows = []
    for sp in SPECIES:
        for r in REGIONS:
            e = series[(sp, r, "emissions")]
            ipk = int(np.argmax(e)); pk = int(years[ipk])
            i80 = int(np.where(years == LATE_FROM)[0][0])
            vals = []
            for n in names:
                y = series[(sp, r, n)]
                total = y[ipk] - y[0]
                vals.append(100 * (y[ipk] - y[i80]) / total if pk > LATE_FROM and abs(total) > 1e-30 else np.nan)
            rows.append((sp, r, pk, vals))
            print(f"{sp:8s}{r:20s}{pk:6d}" + "".join(f"{v:17.1f}%" for v in vals))

    # ---- the figure ----------------------------------------------------------
    fig, axes = plt.subplots(len(REGIONS), len(SPECIES), figsize=(11, 2.6 * len(REGIONS)),
                             sharex=True, constrained_layout=True, squeeze=False)
    letters = "abcdefghijklmnop"
    for j, sp in enumerate(SPECIES):
        for i, r in enumerate(REGIONS):
            ax = axes[i][j]
            ax.plot(years, unit01(series[(sp, r, "emissions")]), color="k", lw=2.4,
                    label="emissions (cond file)")
            for n, c in COLORS.items():
                ax.plot(years, unit01(series[(sp, r, n)]), color=c, lw=1.6, label=n)
            ax.axvline(LATE_FROM, color="0.6", ls=":", lw=1)
            ax.axvline(2014.5, color="0.8", lw=0.8)
            ax.set_ylim(-0.05, 1.08)
            ax.set_title(f"{r} — {sp}", fontsize=10, loc="left")
            ax.text(0.01, 0.97, f"({letters[i * len(SPECIES) + j]})", transform=ax.transAxes,
                    fontweight="bold", va="top", fontsize=9)
            ret = next(v for s_, r_, _, v in rows if s_ == sp and r_ == r)
            ax.text(0.99, 0.04, "late-rise kept: " + "  ".join(
                f"{n.split()[0]} {v:.0f}%" for n, v in zip(names, ret) if not np.isnan(v)),
                transform=ax.transAxes, ha="right", fontsize=7, color="0.25")
            if j == 0:
                ax.set_ylabel("rescaled 0-1")
            if i == len(REGIONS) - 1:
                ax.set_xlabel("year")
    axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.suptitle("Does the normalisation saturate Asian emissions? Region-mean conditioning after "
                 "normalise -> smooth -> PCA, vs the emissions it encodes (each curve rescaled 0-1)",
                 fontsize=10.5)
    out = os.path.join(HERE, "figures", "figure_41_asian_saturation_by_transform")
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=160, bbox_inches="tight")
    print(f"\nwrote {out}.png/.pdf")


if __name__ == "__main__":
    main()
