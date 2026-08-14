#!/usr/bin/env python3
"""
Paper figure: the RAMIP ssp370-126aer experiment, emulator vs CESM2.

TWO VIEWS, --mode
-----------------
anomaly (default)
    The experiment's own response: ssp370-126aer as an ANOMALY vs each side's
    own 1850-1900 climatology. Temperature in K, precipitation as percent
    change (a few hundredths of a mm/day means nothing without the ~2.9 mm/day
    it is relative to — same convention as paper_fig_timeseries.py).

signal
    The aerosol-removal signal, ssp370-126aer MINUS ssp370, computed separately
    on each side. That is the quantity RAMIP exists to isolate, and it needs no
    baseline at all because the control subtracts the climatology.

Each side is referenced to ITSELF in both views — the emulator to its own
emulated historical, CESM2 to CESM2 historical — so any climatological offset
between them cancels and what is compared is the RESPONSE.

For --mode signal the CESM2 control is RAMIP's OWN 10-member ssp370, not the
3-member CMIP6 ssp370: differencing across ensembles would leave a
model-configuration difference inside the answer.

UNCERTAINTY
-----------
Bands are +/-2 x the standard error of the ensemble mean (or of the difference
of two means, in signal mode), which is the right scale for "could this be
sampling noise". The two sides' spreads mean different things — CESM2's is
climate internal variability, the emulator's is diffusion sampling noise — so
read the bands as sampling error, not as model uncertainty.

Usage
-----
    python scripts/paper_fig_ramip.py \\
        --emu-pert <eval>/ramip_ens25/TREFHT_ssp370-126aer.nc \\
        --emu-hist <eval>/ep0490_ens25/TREFHT_hist.nc
    # add --vars TREFHT PRECT once the emulator's PRECT run exists
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

DATA = "/home/nordling/mnt/lumi_sc/emulator_data"
BASELINE = (1850, 1900)

VARS = {
    # The eval NetCDFs store the CESM2 reference already converted (tas K->degC,
    # pr -> mm/day), while the ramip_*.nc references hold raw CMIP6 units. The
    # baseline comes from the former and the series from the latter, so the
    # reference must be converted here or an "anomaly" of 274 K comes out.
    "TREFHT": dict(nc="tas", label="Temperature", percent=False, offset=-273.15,
                   ylab="Temperature anomaly (K, vs 1850–1900)",
                   ylab_sig="Aerosol-removal warming (K)", colour="#D55E00"),
    # percent applies to the ANOMALY only. In signal mode the two experiments
    # are differenced directly, so the result is an absolute mm/day difference
    # with no baseline to divide by — labelling it "%" overstates it ~35x.
    "PRECT":  dict(nc="pr", label="Precipitation", percent=True, scale=86400.0,
                   ylab="Precipitation change (%, vs 1850–1900)",
                   ylab_sig="Aerosol-removal precipitation change "
                            "(mm day$^{-1}$)",
                   unit_sig="mm/day", colour="#0072B2"),
}
C_EMU, C_CESM = "#D55E00", "#0072B2"


def _gmean_members(ds, prefix):
    """Per-member global-mean series already stored in an eval NetCDF."""
    names = sorted([v for v in ds.data_vars
                    if re.fullmatch(rf"{prefix}_m\d+", v)],
                   key=lambda x: int(x.rsplit("_m", 1)[1]))
    if not names:
        return None
    return np.stack([ds[n].values for n in names])          # (member, year)


def emu_gmean(path, var):
    ds = xr.open_dataset(path)
    M = _gmean_members(ds, f"{var}_model_gmean")
    if M is None:
        raise KeyError(f"{path}: no {var}_model_gmean_m* fields")
    return ds["year"].values.astype(int), M


def cesm_gmean_from_eval(path, var):
    """CESM2 reference global means as stored in an eval NetCDF (hist)."""
    ds = xr.open_dataset(path)
    M = _gmean_members(ds, f"{var}_cesm_gmean")
    if M is None:
        return None, None
    return ds["year"].values.astype(int), M


def cesm_gmean_from_ref(path, ncvar, scale=1.0, offset=0.0):
    """CESM2 global means computed from an annual multi-member reference."""
    ds = xr.open_dataset(path)
    w = np.cos(np.deg2rad(ds["lat"]))
    g = (ds[ncvar] * scale + offset).weighted(w).mean(("lat", "lon"))
    g = g.transpose("member", "year")
    return ds["year"].values.astype(int), g.values


def baseline_of(years, M):
    m = (years >= BASELINE[0]) & (years <= BASELINE[1])
    return float(np.nanmean(M[:, m])) if m.any() else np.nan


def anom(M, base, percent):
    return 100.0 * (M - base) / base if percent else M - base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["anomaly", "signal"], default="anomaly")
    ap.add_argument("--baseline", choices=["hist", "self"], default="hist",
                    help="'hist' references each side to its own 1850-1900 "
                         "climatology (needs the hist eval files); 'self' uses "
                         "the first --baseline-years of the experiment itself, "
                         "which needs no historical run at all and cancels any "
                         "drift the two sides carry into 2015")
    ap.add_argument("--baseline-years", type=int, default=10,
                    help="length of the --baseline self window")
    ap.add_argument("--vars", nargs="+", default=["TREFHT", "PRECT"],
                    choices=sorted(VARS))
    ap.add_argument("--emu-dir", default=None,
                    help="eval dir holding <VAR>_ssp370-126aer.nc")
    ap.add_argument("--emu-hist-dir", default=None,
                    help="eval dir holding <VAR>_hist.nc for the baselines")
    ap.add_argument("--emu-ctrl-dir", default=None,
                    help="eval dir holding <VAR>_ssp370.nc (--mode signal only)")
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--experiment", default="ssp370-126aer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dump-data", default=None, metavar="DIR")
    args = ap.parse_args()

    if args.out is None:
        _tag = args.mode + ("" if args.baseline == "hist" else "_selfbase")
        args.out = f"plots/paper_fig_ramip_{_tag}.png"
    if args.emu_hist_dir is None:
        args.emu_hist_dir = args.emu_ctrl_dir

    series, missing = {}, []
    for var in args.vars:
        V = VARS[var]
        suffix = "" if var == "TREFHT" else f"_{V['nc']}"
        p_emu = os.path.join(args.emu_dir, f"{var}_{args.experiment}.nc")
        p_cesm = f"{args.data_root}/cmip6/ramip_{args.experiment}{suffix}.nc"
        if not os.path.exists(p_emu):
            # The emulator's PRECT run for this experiment does not exist yet:
            # its eval crashed in the plotting stage before writing it. Report
            # it plainly and keep the CESM2 side, rather than dropping the
            # variable silently.
            print(f"[{var}] emulator file MISSING: {p_emu}")
            missing.append(var)
        if not os.path.exists(p_cesm):
            print(f"[{var}] CESM2 reference MISSING: {p_cesm} — skipping")
            continue

        sc, off = V.get("scale", 1.0), V.get("offset", 0.0)
        cy, CM = cesm_gmean_from_ref(p_cesm, V["nc"], sc, off)
        ey, EM = (emu_gmean(p_emu, var) if os.path.exists(p_emu) else (None, None))

        # ── baselines ────────────────────────────────────────────────────────
        if args.baseline == "self":
            # First N years of the experiment, per side. No historical run is
            # involved, so this also removes any 1850-1900 climatology mismatch
            # and any drift accumulated before 2015 from the comparison — what
            # is left is purely the change ACROSS the experiment.
            b0 = int(cy.min()); b1 = b0 + args.baseline_years - 1
            cm = (cy >= b0) & (cy <= b1)
            c_base = float(np.nanmean(CM[:, cm]))
            if EM is None:
                e_base = np.nan
            else:
                em = (ey >= b0) & (ey <= b1)
                e_base = float(np.nanmean(EM[:, em]))
            print(f"[{var}] baseline {b0}-{b1} (experiment's own first "
                  f"{args.baseline_years} yr): emulator {e_base:.4f}, "
                  f"CESM2 {c_base:.4f}")
        else:
            hp = os.path.join(args.emu_hist_dir or "", f"{var}_hist.nc")
            if not os.path.exists(hp):
                print(f"[{var}] hist file MISSING: {hp} — cannot build a "
                      f"baseline (use --baseline self to avoid needing it)")
                continue
            hy, HE = emu_gmean(hp, var)
            e_base = baseline_of(hy, HE)
            chy, HC = cesm_gmean_from_eval(hp, var)
            if HC is None:
                print(f"[{var}] no CESM2 members in {hp}")
                continue
            c_base = baseline_of(chy, HC)
            print(f"[{var}] baseline 1850-1900: emulator {e_base:.4f}, "
                  f"CESM2 {c_base:.4f}")

        if args.mode == "signal":
            cp = f"{args.data_root}/cmip6/ramip_ssp370{suffix}.nc"
            pe = os.path.join(args.emu_ctrl_dir or "", f"{var}_ssp370.nc")
            if not os.path.exists(cp):
                print(f"[{var}] signal mode needs {cp}; skipping")
                continue
            cy2, CC = cesm_gmean_from_ref(cp, V["nc"], sc, off)
            # The emulator side needs BOTH its perturbed run and its control.
            # Either can be absent — the ssp370-126aer eval crashed before
            # writing PRECT — in which case the CESM2 signal is still shown,
            # exactly as anomaly mode already does.
            have_emu = EM is not None and os.path.exists(pe)
            if EM is not None and not os.path.exists(pe):
                print(f"[{var}] emulator control MISSING: {pe}")
            if have_emu:
                ey2, EC = emu_gmean(pe, var)
                yrs = np.intersect1d(np.intersect1d(cy, ey),
                                     np.intersect1d(cy2, ey2))
            else:
                yrs = np.intersect1d(cy, cy2)
            gi = lambda y, A: A[:, np.searchsorted(y, yrs)]
            c_v = gi(cy, CM).mean(0) - gi(cy2, CC).mean(0)
            c_se = np.sqrt(gi(cy, CM).var(0, ddof=1)/CM.shape[0]
                           + gi(cy2, CC).var(0, ddof=1)/CC.shape[0])
            nc_ = CM.shape[0]
            if have_emu:
                e_v = gi(ey, EM).mean(0) - gi(ey2, EC).mean(0)
                e_se = np.sqrt(gi(ey, EM).var(0, ddof=1)/EM.shape[0]
                               + gi(ey2, EC).var(0, ddof=1)/EC.shape[0])
                ne_ = EM.shape[0]
            else:
                e_v = e_se = None
                ne_ = 0
            ylab = V["ylab_sig"]
        else:
            yrs = cy if EM is None else np.intersect1d(cy, ey)
            Ca = anom(CM[:, np.searchsorted(cy, yrs)], c_base, V["percent"])
            c_v = Ca.mean(0)
            c_se = Ca.std(0, ddof=1) / np.sqrt(Ca.shape[0])
            nc_ = Ca.shape[0]
            if EM is None:
                e_v = e_se = None; ne_ = 0
            else:
                Ea = anom(EM[:, np.searchsorted(ey, yrs)], e_base, V["percent"])
                e_v = Ea.mean(0)
                e_se = Ea.std(0, ddof=1) / np.sqrt(Ea.shape[0])
                ne_ = Ea.shape[0]
            ylab = (V["ylab"] if args.baseline == "hist"
                    else V["ylab"].replace("1850–1900", f"{int(yrs.min())}–"
                                           f"{int(yrs.min())+args.baseline_years-1}"))

        series[var] = dict(yrs=yrs, c=c_v, cse=c_se, e=e_v, ese=e_se,
                           ylab=ylab, nc=nc_, ne=ne_,
                           unit=(V.get("unit_sig", "K") if args.mode == "signal"
                                 else ("%" if V["percent"] else "K")))
        print(f"[{var}] {yrs.min()}-{yrs.max()}  CESM2 {nc_} members, "
              f"emulator {ne_} members")

    if not series:
        print("no data to plot", file=sys.stderr)
        return 1

    # ── numbers ──────────────────────────────────────────────────────────────
    rows = []
    for var, s in series.items():
        for lo in range(int(s["yrs"].min())//10*10, int(s["yrs"].max()), 10):
            hi = lo + 9
            m = (s["yrs"] >= lo) & (s["yrs"] <= hi)
            if m.sum() < 8:      # skip partial decades (2015-2019 is not "2010-2019")
                continue
            r = dict(var=var, decade=f"{lo}-{hi}", unit=s["unit"],
                     cesm=round(float(s["c"][m].mean()), 3),
                     cesm_se=round(float(np.sqrt((s["cse"][m]**2).mean())), 3))
            if s["e"] is not None:
                r.update(emulator=round(float(s["e"][m].mean()), 3),
                         emulator_se=round(float(np.sqrt((s["ese"][m]**2).mean())), 3),
                         difference=round(float(s["e"][m].mean()-s["c"][m].mean()), 3))
            rows.append(r)
    t = pd.DataFrame(rows)
    print(f"\nssp370-126aer, {args.mode}")
    print(t.to_string(index=False))

    if args.dump_data:
        os.makedirs(args.dump_data, exist_ok=True)
        d = []
        for var, s in series.items():
            for nm, arr in (("cesm2", s["c"]), ("cesm2_se", s["cse"]),
                            ("emulator", s["e"]), ("emulator_se", s["ese"])):
                if arr is None:
                    continue
                d += [dict(var=var, year=int(y), source=nm, value=float(v),
                           unit=s["unit"]) for y, v in zip(s["yrs"], arr)]
        p = os.path.join(args.dump_data, f"ramip_{args.mode}"
                         f"{'' if args.baseline == 'hist' else '_selfbase'}.csv")
        pd.DataFrame(d).to_csv(p, index=False)
        print(f"[data] {p}")

    # ── plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 3.3*n + 0.4), squeeze=False,
                             sharex=True)
    for i, (var, s) in enumerate(series.items()):
        ax = axes[i][0]
        ax.fill_between(s["yrs"], s["c"]-2*s["cse"], s["c"]+2*s["cse"],
                        color=C_CESM, alpha=0.20, lw=0)
        ax.plot(s["yrs"], s["c"], color=C_CESM, lw=2.2,
                label=f"CESM2 (RAMIP, {s['nc']} members)")
        if s["e"] is not None:
            ax.fill_between(s["yrs"], s["e"]-2*s["ese"], s["e"]+2*s["ese"],
                            color=C_EMU, alpha=0.20, lw=0)
            ax.plot(s["yrs"], s["e"], color=C_EMU, lw=2.2,
                    label=f"Emulator ({s['ne']} members)")
        else:
            ax.text(0.5, 0.08, "emulator run for this variable not generated yet",
                    transform=ax.transAxes, ha="center", fontsize=8.5,
                    color=C_EMU, style="italic")
        ax.set_ylabel(s["ylab"])
        ax.legend(frameon=False, loc="upper left")
        ax.set_title(f"({'abc'[i]})  {VARS[var]['label']}", loc="left",
                     fontsize=10)
        ax.set_xlim(s["yrs"].min(), s["yrs"].max())
        if args.mode == "signal":
            ax.axhline(0, ls=":", lw=0.8, color="0.3")
    axes[-1][0].set_xlabel("Year")
    _bl = ("1850–1900" if args.baseline == "hist" else
           f"{int(list(series.values())[0]['yrs'].min())}–"
           f"{int(list(series.values())[0]['yrs'].min())+args.baseline_years-1}")
    ttl = (f"SSP3-7.0 with SSP1-2.6 aerosols (ssp370-126aer): anomaly vs {_bl}"
           if args.mode == "anomaly" else
           "ssp370-126aer minus ssp370: the aerosol-removal signal")
    fig.suptitle(ttl + "\nbands: ±2 SE of the ensemble mean", fontsize=10,
                 y=1.005)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"\nwrote {args.out}")
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    if missing:
        print(f"\nNOT PLOTTED (emulator side): {', '.join(missing)} — the eval "
              f"that produced {args.experiment} crashed in its plotting stage "
              f"before writing them. Rerun it to complete the figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
