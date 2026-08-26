#!/usr/bin/env python3
"""
Forcing-agent attribution: is the all-forcing response the sum of its parts?

THE TEST
--------
CESM2's single-forcing runs vary ONE agent and hold everything else at 1850:
`ghg` (greenhouse gases only) and `aaer` (anthropogenic aerosols only). The
all-forcing response is `hist` up to 2014 and `ssp370` after it. Every series
is an anomaly vs 1850-1900, so they can be added, and the residual

    R  =  ALL  -  (GHG + AAER)

is what the two agents fail to explain between them. R is computed separately
for CESM2 and for the emulator, which is the point of the exercise: an emulator
that has learned the agents' JOINT response reproduces CESM2's R, while one
that has merely learned two marginals produces R ~ 0 by construction.

WHAT R IS NOT
-------------
R is not a pure interaction term. The single-forcing runs omit ozone, land use,
biomass burning, solar and volcanic forcing, so

    R  =  (other forcings)  +  (nonlinear interaction)

and over the historical period the volcanic contribution DOMINATES: Krakatoa,
Santa Maria, Agung, El Chichon and Pinatubo appear as sharp negative spikes of
several tenths of a K that have nothing to do with GHG-aerosol interaction. The
emulator has no volcanic input at all, so it cannot reproduce them and should
not be judged on them. Read the interaction claim on the POST-2015 window,
where ssp370 carries only background volcanic aerosol and the residual is
dominated by the genuine nonlinearity. `--interaction-window` sets it.

Precipitation is handled in percent of the 1850-1900 climatology. Percent
anomalies share one denominator, so they add exactly as absolute ones do.

Usage
-----
    python scripts/paper_fig_attribution.py \\
        --eval-dir <eval>/best_ep0860 --vars TREFHT PRECT --maps
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

BASELINE = (1850, 1900)

VARS = {
    "TREFHT": dict(label="Temperature", percent=False, unit="K",
                   ylab="Temperature anomaly (K, vs 1850–1900)",
                   cmap="RdBu_r"),
    "PRECT":  dict(label="Precipitation", percent=True, unit="%",
                   unit_map="mm day$^{-1}$",
                   ylab="Precipitation change (%, vs 1850–1900)",
                   cmap="BrBG"),
}

# Agent colours, colour-blind safe (Okabe-Ito). ALL and SUM deliberately share
# no hue with the agents: the figure's subject is whether they coincide.
C = dict(all="#000000", sum="#D55E00", ghg="#CC79A7", aaer="#0072B2",
         res="#009E73")

# Boxes for "where are the interactions largest". Chosen to cover the regions
# where GHG and aerosol forcing are both strong and where the literature puts
# the aerosol fingerprint, plus two tropical bands for precipitation.
REGIONS = [
    ("Arctic (60-90N)",        60,  90,   0, 360),
    ("N. mid-lat (30-60N)",    30,  60,   0, 360),
    ("Tropics (20S-20N)",     -20,  20,   0, 360),
    ("S. mid-lat (60-30S)",   -60, -30,   0, 360),
    ("Antarctic (90-60S)",    -90, -60,   0, 360),
    ("Europe",                 35,  70, 350, 40),
    ("East Asia",              20,  50,  100, 145),
    ("South Asia",             5,   35,   65, 100),
    ("N. America",             25,  60, 235, 300),
    ("N. Atlantic",            40,  65, 300, 350),
    ("N. Pacific",             30,  60,  150, 230),
    ("Sahel / W. Africa",       5,  20,  340, 40),
    ("Maritime Continent",    -10,  10,   95, 150),
]


# ── readers ──────────────────────────────────────────────────────────────────
def _members(ds, prefix, suffix=""):
    names = sorted([v for v in ds.data_vars
                    if re.fullmatch(rf"{prefix}_m\d+{suffix}", v)],
                   key=lambda x: int(re.search(r"_m(\d+)", x).group(1)))
    return names


def gmean_series(path, var, side):
    """(years, (member, year)) anomaly series, already baselined by eval_aero."""
    ds = xr.open_dataset(path)
    names = _members(ds, f"{var}_{side}_gmean", "_anom")
    if not names:
        return None, None
    M = np.stack([ds[n].values for n in names])
    yv = "year" if side == "model" else ("cesm_year" if "cesm_year" in ds
                                         else "year")
    return ds[yv].values.astype(int), M


def window_maps(path, var, side, lo, hi):
    """(member, lat, lon) mean anomaly map over [lo, hi]."""
    ds = xr.open_dataset(path)
    names = _members(ds, f"{var}_{side}", "_anom")
    if not names:
        return None, None, None
    yv = "year" if side == "model" else ("cesm_year" if "cesm_year" in ds
                                         else "year")
    yrs = ds[yv].values.astype(int)
    idx = np.where((yrs >= lo) & (yrs <= hi))[0]
    if idx.size == 0:
        return None, None, None
    out = np.stack([ds[n].isel({yv: idx}).values.mean(axis=0) for n in names])
    return out, ds["lat"].values, ds["lon"].values


def gmean_baseline(path, var, side):
    """1850-1900 mean of the ABSOLUTE global-mean members.

    The eval NetCDFs store `*_anom` as an absolute difference in the variable's
    own units, so precipitation arrives in mm/day. The rest of the paper reports
    precipitation as a percent of the pre-industrial climatology, and that
    conversion needs the climatology itself — which only the non-anomaly fields
    carry.
    """
    ds = xr.open_dataset(path)
    names = _members(ds, f"{var}_{side}_gmean")
    names = [n for n in names if not n.endswith("_anom")]
    if not names:
        return np.nan
    M = np.stack([ds[n].values for n in names])
    yv = "year" if side == "model" else ("cesm_year" if "cesm_year" in ds
                                         else "year")
    y = ds[yv].values.astype(int)
    m = (y >= BASELINE[0]) & (y <= BASELINE[1])
    return float(np.nanmean(M[:, m])) if m.any() else np.nan


def _mean_se(M):
    return M.mean(0), M.std(0, ddof=1) / np.sqrt(M.shape[0])


def area_weights(lat, lon):
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    return w / w.sum()


def region_mask(lat, lon, s, n, w, e):
    la = (lat >= s) & (lat <= n)
    lo = ((lon >= w) & (lon <= e)) if w <= e else ((lon >= w) | (lon <= e))
    return la[:, None] & lo[None, :]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--vars", nargs="+", default=["TREFHT", "PRECT"],
                    choices=sorted(VARS))
    ap.add_argument("--interaction-window", nargs=2, type=int,
                    default=[2041, 2050], metavar=("LO", "HI"),
                    help="the window the interaction claim is read on. Must "
                         "sit after 2015 (no large eruptions in ssp370) and "
                         "inside the single-forcing runs' 2050 end.")
    ap.add_argument("--maps", action="store_true",
                    help="also write residual maps for --interaction-window")
    ap.add_argument("--decompose", action="store_true",
                    help="second figure: where the non-additivity comes from — "
                         "conditioning additivity, the Arctic excess split by "
                         "scenario, and whether R has its own geography")
    ap.add_argument("--cond-dir",
                    default="/home/nordling/mnt/lumi_sc2/emulator_data",
                    help="directory holding the per-scenario conditioning files")
    ap.add_argument("--cond-template",
                    default="emissions_{scen}_only_timefixed_bc_co2fix.nc")
    ap.add_argument("--allow-no-cartopy", action="store_true")
    ap.add_argument("--out", default="plots/paper_fig_attribution.png")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    E = args.eval_dir
    lo_i, hi_i = args.interaction_window
    if lo_i < 2015:
        print(f"[warn] interaction window {lo_i}-{hi_i} reaches before 2015; "
              f"volcanic forcing sits in the residual there and is NOT an "
              f"interaction. See the module docstring.", file=sys.stderr)

    series, rows, region_rows, maps = {}, [], [], {}
    for var in args.vars:
        V = VARS[var]
        S = {}
        for side in ("cesm", "model"):
            got = {}
            for agent, fname in (("hist", "hist"), ("ssp370", "ssp370"),
                                 ("ghg", "ghg"), ("aaer", "aaer")):
                p = os.path.join(E, f"{var}_{fname}.nc")
                if not os.path.exists(p):
                    print(f"[{var}/{side}] MISSING {p}")
                    continue
                y, M = gmean_series(p, var, side)
                if M is None:
                    print(f"[{var}/{side}] no members in {p}")
                    continue
                got[agent] = (y, M)
            if not {"hist", "ssp370", "ghg", "aaer"} <= set(got):
                print(f"[{var}/{side}] incomplete set — skipping this side")
                continue

            # mm/day -> percent of the 1850-1900 climatology. One shared
            # denominator per side, so the series stay exactly additive.
            if V["percent"]:
                b = gmean_baseline(os.path.join(E, f"{var}_hist.nc"), var, side)
                if not np.isfinite(b) or b <= 0:
                    print(f"[{var}/{side}] no usable baseline for the percent "
                          f"conversion — leaving values in native units")
                else:
                    print(f"[{var}/{side}] baseline {b:.4f} mm/day "
                          f"(percent conversion)")
                    got = {k: (y, M * (100.0 / b)) for k, (y, M) in got.items()}

            # ALL forcing = hist spliced to ssp370. Both are anomalies vs the
            # same 1850-1900 baseline, so the join needs no offset; the two
            # come from different ensembles, hence mean+SE per segment rather
            # than a single member axis.
            ah, sh = _mean_se(got["hist"][1])
            as_, ss = _mean_se(got["ssp370"][1])
            y_all = np.concatenate([got["hist"][0], got["ssp370"][0]])
            v_all = np.concatenate([ah, as_])
            e_all = np.concatenate([sh, ss])

            yg, Mg = got["ghg"]; ya, Ma = got["aaer"]
            vg, eg = _mean_se(Mg); va, ea = _mean_se(Ma)
            yrs = np.intersect1d(np.intersect1d(y_all, yg), ya)
            g = lambda y, a: a[np.searchsorted(y, yrs)]
            v_all, e_all = g(y_all, v_all), g(y_all, e_all)
            vg, eg = g(yg, vg), g(yg, eg)
            va, ea = g(ya, va), g(ya, ea)

            v_sum = vg + va
            e_sum = np.sqrt(eg**2 + ea**2)
            v_res = v_all - v_sum
            e_res = np.sqrt(e_all**2 + e_sum**2)
            S[side] = dict(yrs=yrs, all=v_all, all_se=e_all, ghg=vg, ghg_se=eg,
                           aaer=va, aaer_se=ea, sum=v_sum, sum_se=e_sum,
                           res=v_res, res_se=e_res,
                           n=dict(hist=got["hist"][1].shape[0],
                                  ssp370=got["ssp370"][1].shape[0],
                                  ghg=Mg.shape[0], aaer=Ma.shape[0]))
            print(f"[{var}/{side}] {yrs.min()}-{yrs.max()}  members "
                  f"{S[side]['n']}")

        if not S:
            continue
        series[var] = S

        # ── decadal numbers ──────────────────────────────────────────────────
        for side, s in S.items():
            for lo in range(int(s["yrs"].min())//10*10, int(s["yrs"].max()), 10):
                hi = lo + 9
                m = (s["yrs"] >= lo) & (s["yrs"] <= hi)
                if m.sum() < 8:
                    continue
                rows.append(dict(
                    var=var, side={"cesm": "CESM2", "model": "emulator"}[side],
                    decade=f"{lo}-{hi}", unit=V["unit"],
                    all=round(float(s["all"][m].mean()), 3),
                    ghg=round(float(s["ghg"][m].mean()), 3),
                    aaer=round(float(s["aaer"][m].mean()), 3),
                    sum=round(float(s["sum"][m].mean()), 3),
                    residual=round(float(s["res"][m].mean()), 3),
                    residual_se=round(float(np.sqrt((s["res_se"][m]**2).mean())), 3),
                    residual_pct_of_all=round(
                        100.0 * float(s["res"][m].mean()) /
                        float(s["all"][m].mean()), 1)
                    if abs(float(s["all"][m].mean())) > 1e-9 else np.nan))

        # ── maps ─────────────────────────────────────────────────────────────
        if args.maps:
            M = {}
            for side in S:
                parts = {}
                ok = True
                for agent, fname in (("all", "ssp370"), ("ghg", "ghg"),
                                     ("aaer", "aaer")):
                    a, lat, lon = window_maps(
                        os.path.join(E, f"{var}_{fname}.nc"), var, side,
                        lo_i, hi_i)
                    if a is None:
                        print(f"[{var}/{side}] no map for {agent} in "
                              f"{lo_i}-{hi_i}")
                        ok = False
                        break
                    parts[agent] = a
                if not ok:
                    continue
                # Maps stay in native units. A percent residual would divide
                # by the LOCAL climatology, which is near zero over the deserts
                # and turns a meaningless absolute error into a huge percentage
                # — the masking problem paper_fig_maps.py handles with a floor.
                # The residual is a difference of differences; mm/day is the
                # honest unit for it.
                res = (parts["all"].mean(0) - parts["ghg"].mean(0)
                       - parts["aaer"].mean(0))
                se = np.sqrt(sum(p.var(0, ddof=1)/p.shape[0]
                                 for p in parts.values()))
                M[side] = dict(res=res, se=se, lat=lat, lon=lon,
                               all=parts["all"].mean(0),
                               ghg=parts["ghg"].mean(0),
                               aaer=parts["aaer"].mean(0))
            if M:
                maps[var] = M
                # regional breakdown, ranked by |residual|
                ref = M.get("cesm", list(M.values())[0])
                lat, lon = ref["lat"], ref["lon"]
                W = area_weights(lat, lon)
                for name, s_, n_, w_, e_ in REGIONS:
                    mk = region_mask(lat, lon, s_, n_, w_, e_)
                    ww = W * mk
                    ww = ww / ww.sum()
                    r = dict(var=var, region=name, unit=V["unit"],
                             window=f"{lo_i}-{hi_i}")
                    for side, d in M.items():
                        tag = {"cesm": "cesm", "model": "emu"}[side]
                        r[f"{tag}_residual"] = round(float((d["res"]*ww).sum()), 3)
                        r[f"{tag}_residual_se"] = round(
                            float(np.sqrt((d["se"]**2 * ww**2).sum())
                                  * np.sqrt(mk.sum())), 3)
                        r[f"{tag}_all"] = round(float((d["all"]*ww).sum()), 3)
                    region_rows.append(r)

    if not series:
        print("no data", file=sys.stderr)
        return 1

    t = pd.DataFrame(rows)
    print("\nAdditivity test:  residual  R = ALL - (GHG + AAER)")
    print(t.to_string(index=False))

    # The headline number, on the clean window only.
    print(f"\nInteraction window {lo_i}-{hi_i} (post-2015: no major eruptions)")
    for var, S in series.items():
        V = VARS[var]
        for side, s in S.items():
            m = (s["yrs"] >= lo_i) & (s["yrs"] <= hi_i)
            if m.sum() == 0:
                continue
            r = float(s["res"][m].mean())
            se = float(np.sqrt((s["res_se"][m]**2).mean()))
            a = float(s["all"][m].mean())
            print(f"  {var:7s} {'CESM2' if side=='cesm' else 'emulator':9s} "
                  f"R = {r:+.3f} +/- {se:.3f} {V['unit']}  "
                  f"({100*r/a:+.1f}% of the all-forcing response {a:.3f}), "
                  f"|R|/SE = {abs(r)/se:.1f}")

    if region_rows:
        rt = pd.DataFrame(region_rows)
        print(f"\nWhere the residual is largest ({lo_i}-{hi_i}), by |CESM2 R|")
        for var in rt["var"].unique():
            sub = rt[rt["var"] == var].copy()
            key = "cesm_residual" if "cesm_residual" in sub else "emu_residual"
            sub = sub.reindex(sub[key].abs().sort_values(ascending=False).index)
            print(f"\n[{var}]")
            print(sub.to_string(index=False))

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})

    nv = len(series)
    fig, axes = plt.subplots(nv, 2, figsize=(11.0, 3.5*nv + 0.5),
                             squeeze=False, sharex=True)
    for i, (var, S) in enumerate(series.items()):
        V = VARS[var]
        for j, side in enumerate(("cesm", "model")):
            ax = axes[i][j]
            if side not in S:
                ax.set_visible(False)
                continue
            s = S[side]
            for key, lab, lw in (("all", "All forcing (hist+SSP3-7.0)", 2.2),
                                 ("sum", "GHG + AAER (sum)", 2.0),
                                 ("ghg", "GHG only", 1.3),
                                 ("aaer", "Aerosol only", 1.3),
                                 ("res", "Residual  R = All − sum", 1.6)):
                ls = "--" if key == "sum" else "-"
                ax.plot(s["yrs"], s[key], color=C[key], lw=lw, ls=ls, label=lab)
                if key in ("all", "sum", "res"):
                    ax.fill_between(s["yrs"], s[key]-2*s[f"{key}_se"],
                                    s[key]+2*s[f"{key}_se"],
                                    color=C[key], alpha=0.15, lw=0)
            ax.axhline(0, ls=":", lw=0.8, color="0.4")
            ax.axvspan(lo_i, hi_i, color="0.85", alpha=0.5, lw=0, zorder=0)
            ttl = "CESM2" if side == "cesm" else "Emulator"
            ax.set_title(f"({'abcdef'[i*2+j]})  {V['label']} — {ttl}",
                         loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel(V["ylab"])
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=8.0)
            ax.set_xlim(s["yrs"].min(), s["yrs"].max())
        # one y-scale per row so the two sides are comparable by eye
        ys = [axes[i][j].get_ylim() for j in (0, 1) if axes[i][j].get_visible()]
        if len(ys) == 2:
            lim = (min(y[0] for y in ys), max(y[1] for y in ys))
            for j in (0, 1):
                axes[i][j].set_ylim(lim)
    for j in (0, 1):
        axes[-1][j].set_xlabel("Year")
    fig.suptitle("Forcing-agent additivity: does All forcing equal GHG + Aerosol?"
                 f"\nshaded band = interaction window {lo_i}–{hi_i}; "
                 "before 2015 the residual also holds volcanic and other forcings",
                 fontsize=10, y=1.005)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"\nwrote {args.out}")

    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
        if region_rows:
            rp = os.path.splitext(args.csv)[0] + "_regions.csv"
            pd.DataFrame(region_rows).to_csv(rp, index=False)
            print(f"wrote {rp}")

    # ── residual maps ────────────────────────────────────────────────────────
    if maps:
        try:
            import cartopy.crs as ccrs
            proj = dict(projection=ccrs.Robinson(central_longitude=0))
            HAVE = True
        except ImportError:
            HAVE = False
            if not args.allow_no_cartopy:
                print("[maps] cartopy missing; rerun in the plotting env or "
                      "pass --allow-no-cartopy", file=sys.stderr)
                return 2
            proj = {}
        for var, M in maps.items():
            V = VARS[var]
            sides = [s for s in ("cesm", "model") if s in M]
            fig, axs = plt.subplots(1, len(sides) + 1,
                                    figsize=(5.2*(len(sides)+1), 3.4),
                                    subplot_kw=proj, squeeze=False)
            lat = M[sides[0]]["lat"]; lon = M[sides[0]]["lon"]
            vmax = max(np.nanpercentile(np.abs(M[s]["res"]), 98)
                       for s in sides)
            panels = [(s, M[s]["res"],
                       f"{'CESM2' if s=='cesm' else 'Emulator'}  R")
                      for s in sides]
            if len(sides) == 2:
                panels.append(("diff", M["model"]["res"] - M["cesm"]["res"],
                               "Emulator R − CESM2 R"))
            for k, (tag, fld, ttl) in enumerate(panels):
                ax = axs[0][k]
                kw = dict(cmap=V["cmap"], vmin=-vmax, vmax=vmax, shading="auto")
                if HAVE:
                    kw["transform"] = ccrs.PlateCarree()
                    ax.coastlines(lw=0.4)
                im = ax.pcolormesh(lon, lat, fld, **kw)
                ax.set_title(ttl, fontsize=9.5)
                plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04,
                             shrink=0.85,
                             label=V.get("unit_map", V["unit"]))
            fig.suptitle(f"{V['label']}: residual R = All − (GHG + Aerosol), "
                         f"{lo_i}–{hi_i}", fontsize=10, y=1.04)
            mp = (os.path.splitext(args.out)[0] + f"_maps_{var}.png")
            fig.savefig(mp, bbox_inches="tight")
            fig.savefig(os.path.splitext(mp)[0] + ".pdf", bbox_inches="tight")
            print(f"wrote {mp}")
    if args.decompose:
        rc = decompose(args, plt)
        if rc:
            return rc
    return 0


def _wcorr(x, y, w):
    xm, ym = (x*w).sum(), (y*w).sum()
    cov = ((x-xm)*(y-ym)*w).sum()
    return (cov / np.sqrt(((x-xm)**2*w).sum() * ((y-ym)**2*w).sum()),
            cov / ((x-xm)**2*w).sum())


def decompose(args, plt):
    """Where does the non-additivity come from — the inputs or the model?

    Three questions, three panels:
      (a) are the CONDITIONING fields additive? If cond(all) = cond(ghg) +
          cond(aaer) but the response is not additive, the nonlinearity is
          manufactured by the model rather than read off its inputs — a linear
          response operator would return R = 0 by construction.
      (b) which scenario's regional bias builds the emulator's Arctic excess?
      (c) does R have its own geography, or is it the warming pattern rescaled?
          r(R, ALL) answers that; r(R, AEROSOL) tests the physical mechanism,
          since aerosol efficacy falling in a warmer base state predicts R to be
          ANTI-correlated with the aerosol-only response.
    """
    E, (lo_i, hi_i) = args.eval_dir, args.interaction_window
    var = "TREFHT" if "TREFHT" in args.vars else args.vars[0]
    V = VARS[var]

    # ── (a) conditioning ─────────────────────────────────────────────────────
    cond = {}
    for scen in ("ssp370", "ghg", "aaer"):
        p = os.path.join(args.cond_dir, args.cond_template.format(scen=scen))
        if not os.path.exists(p):
            print(f"[decompose] cond MISSING {p}")
            cond = None
            break
        cond[scen] = xr.open_dataset(p).sel(year=slice(lo_i, hi_i)).mean("year")

    # ── (b, c) response fields ───────────────────────────────────────────────
    P = {}
    for side in ("cesm", "model"):
        for ag, f in (("all", "ssp370"), ("ghg", "ghg"), ("aaer", "aaer")):
            a, lat, lon = window_maps(os.path.join(E, f"{var}_{f}.nc"),
                                      var, side, lo_i, hi_i)
            if a is None:
                print(f"[decompose] no maps for {side}/{ag}")
                return 2
            P[(side, ag)] = a.mean(0)
    W = area_weights(lat, lon)
    arc = region_mask(lat, lon, 60, 90, 0, 360)
    Wa = W * arc; Wa = Wa / Wa.sum()
    R = {s: P[(s, "all")] - P[(s, "ghg")] - P[(s, "aaer")]
         for s in ("cesm", "model")}
    am = lambda x: float((x*Wa).sum())

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    # (a) conditioning additivity, each channel normalised by its own all-forcing
    # value — the three channels differ by six orders of magnitude, so absolute
    # units would show one bar and two slivers.
    ax = fig.add_subplot(gs[0, 0])
    if cond is None:
        ax.text(0.5, 0.5, "conditioning files not found", ha="center",
                transform=ax.transAxes)
    else:
        chans = [c for c in ("CO2", "SUL", "BC") if c in cond["ssp370"]]
        xs = np.arange(len(chans)); wbar = 0.26
        g = lambda d, c: float((d[c].values * W).sum())
        base = [g(cond["ssp370"], c) for c in chans]
        for k, (scen, lab, col) in enumerate((
                ("ghg", "GHG-only", C["ghg"]), ("aaer", "Aerosol-only", C["aaer"]))):
            ax.bar(xs + (k-1)*wbar, [g(cond[scen], c)/b
                                     for c, b in zip(chans, base)],
                   wbar, color=col, label=lab)
        ax.bar(xs + wbar, [(g(cond["ghg"], c) + g(cond["aaer"], c))/b
                           for c, b in zip(chans, base)],
               wbar, color=C["sum"], label="GHG + Aerosol")
        for i, c in enumerate(chans):
            tot = (g(cond["ghg"], c) + g(cond["aaer"], c))/base[i]
            d = 1.0 - tot
            # clear of BOTH the reference line and the bar, which overshoots 1
            # wherever the single-forcing files leak a little of the other agent
            ax.annotate(f"{100*d:+.2f}%", (xs[i]+wbar, max(1.0, tot) + 0.03),
                        ha="center", fontsize=8, color="0.25")
        ax.axhline(1.0, ls=":", lw=1.0, color="0.3")
        ax.set_xticks(xs); ax.set_xticklabels(chans)
        ax.set_ylabel("fraction of the all-forcing field")
        ax.set_ylim(0, 1.25)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("(a)  The INPUT is additive\n"
                 f"conditioning channels, {lo_i}–{hi_i} mean", loc="left",
                 fontsize=9.5)

    # (b) waterfall: CESM2 Arctic R -> emulator Arctic R
    ax = fig.add_subplot(gs[0, 1])
    steps = [("CESM2 R", am(R["cesm"]), "0.45", None)]
    run = am(R["cesm"])
    for ag, sgn, lab in (("all", +1, "+ All-forcing bias"),
                         ("ghg", -1, "− GHG-only bias"),
                         ("aaer", -1, "− Aerosol-only bias")):
        d = sgn * am(P[("model", ag)] - P[("cesm", ag)])
        steps.append((lab, d, C[ag], run)); run += d
    steps.append(("Emulator R", am(R["model"]), "0.15", None))
    for i, (lab, val, col, bot) in enumerate(steps):
        if bot is None:
            ax.bar(i, val, 0.62, color=col)
            ax.annotate(f"{val:+.2f}", (i, val), ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
        else:
            ax.bar(i, val, 0.62, bottom=bot, color=col, alpha=0.85)
            ax.annotate(f"{val:+.2f}", (i, bot + val), ha="center",
                        va="bottom" if val > 0 else "top", fontsize=8.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], rotation=20, ha="right",
                       fontsize=8)
    ax.set_ylabel(f"Arctic-mean residual ({V['unit']})")
    ax.axhline(0, lw=0.8, color="0.3")
    ax.set_title("(b)  The emulator's Arctic excess is three\n"
                 "same-signed scenario biases, not one", loc="left",
                 fontsize=9.5)

    # (c, d) does R have its own geography?
    for j, (ref, name) in enumerate((("all", "All-forcing response"),
                                     ("aaer", "Aerosol-only response"))):
        ax = fig.add_subplot(gs[1, j])
        w = W.ravel()
        for side, col, lab in (("cesm", C["all"], "CESM2"),
                               ("model", C["sum"], "Emulator")):
            x, y = P[(side, ref)].ravel(), R[side].ravel()
            r, sl = _wcorr(x, y, w)
            k = slice(None, None, 11)          # thin for legibility only
            ax.scatter(x[k], y[k], s=1.5, alpha=0.20, color=col, lw=0)
            xx = np.linspace(x.min(), x.max(), 10)
            ax.plot(xx, (y*w).sum() + sl*(xx - (x*w).sum()), color=col, lw=2,
                    label=f"{lab}:  r = {r:+.2f}")
        ax.axhline(0, ls=":", lw=0.8, color="0.4")
        ax.set_xlabel(f"{name} ({V['unit']})")
        if j == 0:
            ax.set_ylabel(f"Residual R ({V['unit']})")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        note = ("R is NOT the warming pattern rescaled"
                if ref == "all" else
                "R is ANTI-correlated with the aerosol response\n"
                "— aerosol efficacy falls in a warmer base state")
        ax.set_title(f"({'cd'[j]})  {note}", loc="left", fontsize=9.5)

    fig.suptitle(f"Where the non-additivity comes from — {V['label']}, "
                 f"{lo_i}–{hi_i}\n"
                 "additive inputs, non-additive response: the model "
                 "manufactures R rather than reading it off the conditioning",
                 fontsize=10.5, y=0.99)
    out = os.path.splitext(args.out)[0] + "_decompose.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
