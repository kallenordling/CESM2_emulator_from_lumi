#!/usr/bin/env python3
"""Global-mean precipitation against global-mean temperature, one dot per year.

WHAT IT SHOWS
-------------
The emulator predicts TREFHT and PRECT as two channels of one field, so the
question is not only whether each is right on its own but whether they move
TOGETHER the way CESM2's do. Plotting annual global means against each other
answers that directly: the slope dP/dT is the model's HYDROLOGICAL SENSITIVITY,
reported here in %/K of the 1850-1900 climatology, and CESM2's is a physical
constraint the emulator was never explicitly trained on.

Each dot is one year's ensemble-mean global mean; with --members every
member-year is drawn faintly behind them, which shows whether the emulator's
year-to-year scatter about the line matches CESM2's.

The fit is ordinary least squares of P on T over the scenario's own years. For
scenarios that turn around (aaer, and ssp126 late) a single slope is a summary,
not a law — the cloud's shape carries the rest.

Usage
-----
    python scripts/paper_fig_tp_scatter.py --eval-dir <eval>/best_ep0860
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

BASELINE = (1850, 1900)
DEFAULT_SCENARIOS = ["hist", "ssp370", "aaer", "ghg"]

LABEL = {"hist": "Historical", "ssp370": "SSP3-7.0", "aaer": "Aerosol-only",
         "ghg": "GHG-only", "ssp126": "SSP1-2.6 (unseen)",
         "ssp245": "SSP2-4.5 (unseen)",
         "ssp370-126aer": "SSP3-7.0 with SSP1-2.6 aerosols"}
C_EMU, C_CESM = "#D55E00", "#0072B2"


def members(ds, prefix):
    names = sorted([v for v in ds.data_vars
                    if re.fullmatch(rf"{prefix}_m\d+", v)],
                   key=lambda x: int(x.rsplit("_m", 1)[1]))
    return names


def series(path, var, side):
    """(years, (member, year)) ABSOLUTE global means — not anomalies."""
    ds = xr.open_dataset(path)
    names = members(ds, f"{var}_{side}_gmean")
    if not names:
        return None, None
    M = np.stack([ds[n].values for n in names])
    yv = "year" if side == "model" else ("cesm_year" if "cesm_year" in ds
                                         else "year")
    return ds[yv].values.astype(int), M


def ols(x, y):
    """slope, intercept, r — plain least squares, y on x."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return b[0], b[1], r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--members", action="store_true",
                    help="also draw every member-year faintly behind the "
                         "ensemble-mean dots")
    ap.add_argument("--year-range", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--out", default="plots/paper_fig_tp_scatter.png")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    E = args.eval_dir
    data, rows = {}, []
    for sc in args.scenarios:
        d = {}
        missing = False
        for var in ("TREFHT", "PRECT"):
            p = os.path.join(E, f"{var}_{sc}.nc")
            if not os.path.exists(p):
                print(f"[{sc}] MISSING {p} — skipping scenario")
                missing = True
                break
            for side in ("cesm", "model"):
                y, M = series(p, var, side)
                if M is None:
                    # Out-of-training scenarios carry no CESM2 precipitation in
                    # the eval file. That costs the reference, not the panel —
                    # the emulator's own T-P relationship is still the point.
                    print(f"[{sc}/{side}] no {var} members — that side dropped")
                    continue
                d[(var, side)] = (y, M)
        if missing:
            continue

        # Temperature and precipitation are stored in separate files, and the
        # CESM2 reference can differ in member count between them, so pair them
        # on the YEAR axis and use each side's own ensemble mean.
        rec = {}
        for side in ("cesm", "model"):
            if ("TREFHT", side) not in d or ("PRECT", side) not in d:
                continue
            yt, T = d[("TREFHT", side)]
            yp, P = d[("PRECT", side)]
            yrs = np.intersect1d(yt, yp)
            if args.year_range:
                lo, hi = args.year_range
                yrs = yrs[(yrs >= lo) & (yrs <= hi)]
            if yrs.size == 0:
                continue
            t = T[:, np.searchsorted(yt, yrs)]
            p_ = P[:, np.searchsorted(yp, yrs)]
            # The two variables' CESM2 ensembles can differ in size (11 hist
            # members for temperature, 5 for precipitation), so a member-year
            # cloud is only meaningful when the counts match — and even then
            # only if member i is the same realization in both files, which
            # holds for the eval writer's ordering.
            rec[side] = dict(yrs=yrs, t=t, p=p_, tm=t.mean(0), pm=p_.mean(0),
                             n=t.shape[0], nt=t.shape[0], np=p_.shape[0],
                             paired=(t.shape[0] == p_.shape[0]))
        if not rec:
            continue
        data[sc] = rec

        for side, r in rec.items():
            sl, ic, cc = ols(r["tm"], r["pm"])
            base = float(np.nanmean(
                r["pm"][(r["yrs"] >= BASELINE[0]) & (r["yrs"] <= BASELINE[1])]))
            # A scenario starting in 2015 has no 1850-1900 window of its own;
            # fall back to the series' own mean so %/K stays comparable rather
            # than becoming NaN.
            if not np.isfinite(base):
                base = float(np.nanmean(r["pm"]))
            rows.append(dict(
                scenario=sc, side={"cesm": "CESM2", "model": "emulator"}[side],
                n_members_T=r["nt"], n_members_P=r["np"], years=f"{r['yrs'].min()}-{r['yrs'].max()}",
                dP_dT_mm_day_per_K=round(float(sl), 4),
                dP_dT_pct_per_K=round(float(100.0*sl/base), 3),
                r=round(float(cc), 3),
                T_mean=round(float(r["tm"].mean()), 3),
                P_mean=round(float(r["pm"].mean()), 4)))
            print(f"[{sc}/{side}] {r['nt']}T/{r['np']}P members  dP/dT = {sl:.4f} mm/day/K "
                  f"= {100*sl/base:.2f} %/K   r = {cc:.3f}")

    if not data:
        print("no data", file=sys.stderr)
        return 1

    t = pd.DataFrame(rows)
    print("\nHydrological sensitivity, global annual means")
    print(t.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})

    n = len(data)
    ncol = 2 if n <= 4 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6*ncol, 4.0*nrow),
                             squeeze=False)
    for k, (sc, rec) in enumerate(data.items()):
        ax = axes[k//ncol][k % ncol]
        for side, col, lab in (("cesm", C_CESM, "CESM2"),
                               ("model", C_EMU, "Emulator")):
            if side not in rec:
                continue
            r = rec[side]
            if args.members and r["paired"]:
                ax.scatter(r["t"].ravel(), r["p"].ravel(), s=2, alpha=0.10,
                           color=col, lw=0)
            ax.scatter(r["tm"], r["pm"], s=9, alpha=0.65, color=col, lw=0)
            sl, ic, cc = ols(r["tm"], r["pm"])
            xx = np.linspace(r["tm"].min(), r["tm"].max(), 10)
            ax.plot(xx, sl*xx + ic, color=col, lw=2,
                    label=f"{lab} ({r['nt']}m):  {sl:.3f} mm/day/K")
        ax.set_title(f"({'abcdefgh'[k]})  {LABEL.get(sc, sc)}", loc="left",
                     fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        if k % ncol == 0:
            ax.set_ylabel("Global-mean precipitation (mm day$^{-1}$)")
        if k//ncol == nrow - 1:
            ax.set_xlabel("Global-mean temperature (°C)")
    for k in range(n, nrow*ncol):
        axes[k//ncol][k % ncol].set_visible(False)
    fig.suptitle("Precipitation against temperature, one dot per year "
                 "(global annual means, absolute)\n"
                 "the slope is the hydrological sensitivity — a coupling the "
                 "emulator was never trained on explicitly",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"\nwrote {args.out}")
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
