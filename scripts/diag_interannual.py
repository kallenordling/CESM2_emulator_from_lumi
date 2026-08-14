#!/usr/bin/env python3
"""
Diagnostic: does the emulator produce realistic YEAR-TO-YEAR variability?

TWO SEPARATE QUESTIONS, and the model can pass one and fail the other:

  AMPLITUDE   is the size of the interannual wiggle right?
  STRUCTURE   is it correlated in time, as the real climate is?

The architecture predicts a specific answer. The model is memoryless per year:
the frame dimension is 1, so each year is an independent diffusion sample
conditioned only on that year's forcing. Nothing carries ocean heat content or
an ENSO phase from one year to the next. So the expectation is:

  amplitude  can be right — the sampler's spread is free to match
  structure   should be WHITE: lag-1 autocorrelation ~0, whereas CESM2's
              global mean carries real persistence (ENSO, ocean memory)

If that is what comes out, it bounds what the emulator can be used for: forced
response and distribution of annual states, yes; anything depending on sequence
— run-lengths, consecutive-year extremes, decadal excursions — no. It also
speaks to the daily-output plan, where temporal structure is most of the point.

METHOD
------
The forced response is removed by subtracting the ENSEMBLE MEAN of each side
year by year, which is exact and assumption-free (no polynomial fit, no
filter). What remains per member is internal variability. Then, per member:

    sd        std over years
    r1..r5    lag-k autocorrelation

averaged over members. Both sides are treated identically. The DoF correction
(1 - 1/n) is applied to the variance, since the mean subtracted was estimated
from the same n members.

Usage
-----
    python scripts/diag_interannual.py --eval-dir <eval>/ep0490_ens25 \\
        --experiment hist --vars TREFHT PRECT
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr


def gmean_members(ds, prefix):
    names = sorted([v for v in ds.data_vars if re.fullmatch(rf"{prefix}_m\d+", v)],
                   key=lambda x: int(x.rsplit("_m", 1)[1]))
    if not names:
        return None
    return np.stack([ds[n].values for n in names])          # (member, year)


def autocorr(x, k):
    """Lag-k autocorrelation of a 1-D series (already zero-mean-ish)."""
    x = x - x.mean()
    if k == 0:
        return 1.0
    denom = np.dot(x, x)
    return float(np.dot(x[:-k], x[k:]) / denom) if denom > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--experiment", default="hist")
    ap.add_argument("--vars", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--max-lag", type=int, default=5)
    ap.add_argument("--out", default="plots/diag_interannual.png")
    ap.add_argument("--csv", default="plots/interannual.csv")
    args = ap.parse_args()

    rows, series = [], {}
    for var in args.vars:
        p = os.path.join(args.eval_dir, f"{var}_{args.experiment}.nc")
        if not os.path.exists(p):
            print(f"[{var}] MISSING {p}")
            continue
        ds = xr.open_dataset(p)
        E = gmean_members(ds, f"{var}_model_gmean")
        C = gmean_members(ds, f"{var}_cesm_gmean")
        if E is None or C is None:
            print(f"[{var}] missing member series — skipping")
            continue
        ey = ds["year"].values.astype(int)
        cy = (ds["cesm_year"].values.astype(int) if "cesm_year" in ds.dims else ey)
        yrs = np.intersect1d(ey, cy)
        E = E[:, np.searchsorted(ey, yrs)]
        C = C[:, np.searchsorted(cy, yrs)]
        print(f"[{var}] emulator {E.shape[0]} members, CESM2 {C.shape[0]}, "
              f"{yrs.min()}-{yrs.max()} ({len(yrs)} yr)")

        out = {}
        for tag, M in (("emulator", E), ("cesm2", C)):
            n = M.shape[0]
            R = M - M.mean(axis=0, keepdims=True)      # remove forced response
            R = R / np.sqrt(1.0 - 1.0 / n)             # DoF correction
            sd = R.std(axis=1, ddof=1).mean()
            acf = [np.mean([autocorr(R[i], k) for i in range(n)])
                   for k in range(args.max_lag + 1)]
            out[tag] = dict(n=n, sd=float(sd), acf=acf)
            print(f"  {tag:9s} n={n:2d}  sd={sd:.4f}  "
                  + "  ".join(f"r{k}={acf[k]:+.3f}" for k in range(1, 4)))

        e, c = out["emulator"], out["cesm2"]
        rows.append(dict(var=var, experiment=args.experiment,
                         n_emu=e["n"], n_cesm=c["n"],
                         sd_emulator=round(e["sd"], 4),
                         sd_cesm2=round(c["sd"], 4),
                         sd_ratio=round(e["sd"] / c["sd"], 3),
                         **{f"r{k}_emulator": round(e["acf"][k], 3)
                            for k in range(1, args.max_lag + 1)},
                         **{f"r{k}_cesm2": round(c["acf"][k], 3)
                            for k in range(1, args.max_lag + 1)}))
        series[var] = out

    if not rows:
        print("nothing computed", file=sys.stderr)
        return 1
    t = pd.DataFrame(rows)
    print("\nInterannual variability of the global mean")
    print(t.to_string(index=False))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        t.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.6), squeeze=False)
    lags = np.arange(0, args.max_lag + 1)
    for j, (var, o) in enumerate(series.items()):
        ax = axes[0][j]
        ax.plot(lags, o["cesm2"]["acf"], "o-", color="#0072B2", lw=2,
                label=f"CESM2 ({o['cesm2']['n']} members)")
        ax.plot(lags, o["emulator"]["acf"], "s-", color="#D55E00", lw=2,
                label=f"Emulator ({o['emulator']['n']} members)")
        ax.axhline(0, ls=":", lw=1, color="0.4")
        ax.set_xlabel("lag (years)"); ax.set_ylabel("autocorrelation")
        ax.set_title(f"({'ab'[j]})  {var}   sd ratio "
                     f"{o['emulator']['sd']/o['cesm2']['sd']:.2f}",
                     loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Interannual variability of the global mean, forced response "
                 "removed\nflat at zero = each year drawn independently",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
