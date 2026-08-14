#!/usr/bin/env python3
"""
Paper figure: the RAMIP aerosol-removal signal, emulator vs CESM2.

THE QUANTITY IS A DIFFERENCE OF DIFFERENCES
-------------------------------------------
RAMIP's ssp370-126aer is ssp370's greenhouse forcing with ssp126's aerosols.
The signal it isolates is

    dT(t) = ssp370-126aer(t) - ssp370(t)

i.e. the warming unmasked by removing the aerosol load, and it is computed
SEPARATELY on each side. That construction is what makes the comparison fair:
each side differences against its own control, so any climatological offset
between emulator and CESM2 cancels exactly and no 1850-1900 baseline is needed.

Both controls must come from the same ensemble as their perturbed run. CESM2
uses RAMIP's own 10-member ssp370, NOT the 3-member CMIP6 ssp370 — differencing
across ensembles would leave a model-configuration difference in the answer.

UNCERTAINTY
-----------
Bands are the standard error of the difference of two ensemble means,

    SE = sqrt(s_a^2/n_a + s_b^2/n_b)

which is the right scale for "could this difference be sampling noise?", and is
~sqrt(n) smaller than the inter-member spread. Note the two sides' spreads mean
different things: CESM2's is climate internal variability, the emulator's is
diffusion sampling noise. Neither is an uncertainty on the FORCED response in
the same sense, so read the bands as sampling error, not as model spread.

Usage
-----
    python scripts/paper_fig_ramip.py \\
        --emu-pert  <eval>/ramip_ens25/TREFHT_ssp370-126aer.nc \\
        --emu-ctrl  <eval>/ep0490_ens25/TREFHT_ssp370.nc
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

DATA = "/home/nordling/mnt/lumi_sc/emulator_data"
VARS = {
    "TREFHT": dict(nc="tas", label="Temperature", unit="K", cmap="RdBu_r",
                   ylab="Aerosol-removal warming (K)"),
    "PRECT":  dict(nc="pr", label="Precipitation", unit="mm day$^{-1}$",
                   cmap="BrBG", scale=86400.0,
                   ylab="Aerosol-removal precipitation change (mm day$^{-1}$)"),
}
C_EMU, C_CESM = "#D55E00", "#0072B2"


def area_w(lat, lon):
    return np.broadcast_to(np.cos(np.deg2rad(np.asarray(lat)))[:, None],
                           (len(lat), len(lon)))


def cesm(path, ncvar, scale=1.0):
    """(years, members, lat, lon) from an annual multi-member reference."""
    ds = xr.open_dataset(path)
    da = (ds[ncvar] * scale).transpose("year", "member", "lat", "lon")
    return (ds["year"].values.astype(int), da.values,
            ds["lat"].values, ds["lon"].values)


def emulator(path, var):
    """(years, members, lat, lon) from an eval NetCDF's per-member maps."""
    ds = xr.open_dataset(path)
    names = sorted([v for v in ds.data_vars
                    if re.fullmatch(rf"{var}_model_m\d+", v)],
                   key=lambda x: int(x.rsplit("_m", 1)[1]))
    if not names:
        raise KeyError(f"{path}: no per-member {var}_model_m* fields")
    M = np.stack([ds[n].values for n in names])          # (mem, yr, lat, lon)
    return (ds["year"].values.astype(int), M.transpose(1, 0, 2, 3),
            ds["lat"].values, ds["lon"].values)


def gmean(F, lat, lon):
    """(year, member, lat, lon) -> (year, member) area-weighted global mean."""
    w = area_w(lat, lon)
    return (F * w).sum(axis=(2, 3)) / w.sum()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--var", default="TREFHT", choices=sorted(VARS))
    ap.add_argument("--emu-pert", required=True,
                    help="eval NetCDF for the emulator's ssp370-126aer run")
    ap.add_argument("--emu-ctrl", required=True,
                    help="eval NetCDF for the emulator's ssp370 run "
                         "(same checkpoint, or the comparison is meaningless)")
    ap.add_argument("--cesm-pert", default=None)
    ap.add_argument("--cesm-ctrl", default=None)
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--n-years", type=int, default=10,
                    help="final-decade window for the map panels")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dump-data", default=None, metavar="DIR")
    args = ap.parse_args()

    V = VARS[args.var]
    suffix = "" if args.var == "TREFHT" else f"_{V['nc']}"
    if args.cesm_pert is None:
        args.cesm_pert = f"{args.data_root}/cmip6/ramip_ssp370-126aer{suffix}.nc"
    if args.cesm_ctrl is None:
        args.cesm_ctrl = f"{args.data_root}/cmip6/ramip_ssp370{suffix}.nc"
    if args.out is None:
        args.out = f"plots/paper_fig_ramip_{args.var}.png"

    for p in (args.emu_pert, args.emu_ctrl, args.cesm_pert, args.cesm_ctrl):
        if not os.path.exists(p):
            print(f"ERROR: missing input {p}", file=sys.stderr)
            if "PRECT" in p or V["nc"] == "pr":
                print("       (the emulator's PRECT run for ssp370-126aer was "
                      "never written — its eval crashed in plotting before "
                      "that stage)", file=sys.stderr)
            return 1

    sc = V.get("scale", 1.0)
    print(f"[{args.var}] loading …", flush=True)
    cy, CP, lat, lon = cesm(args.cesm_pert, V["nc"], sc)
    _,  CC, _, _     = cesm(args.cesm_ctrl, V["nc"], sc)
    ey, EP, _, _     = emulator(args.emu_pert, args.var)
    sy, EC, _, _     = emulator(args.emu_ctrl, args.var)

    yrs = np.intersect1d(np.intersect1d(cy, ey), sy)
    ic, ie, isx = (np.searchsorted(cy, yrs), np.searchsorted(ey, yrs),
                   np.searchsorted(sy, yrs))
    CP, CC, EP, EC = CP[ic], CC[ic], EP[ie], EC[isx]
    print(f"  CESM2    {CP.shape[1]} vs {CC.shape[1]} members")
    print(f"  emulator {EP.shape[1]} vs {EC.shape[1]} members")
    print(f"  years    {yrs.min()}-{yrs.max()}")

    # ── global-mean signal, per side ─────────────────────────────────────────
    gCP, gCC = gmean(CP, lat, lon), gmean(CC, lat, lon)
    gEP, gEC = gmean(EP, lat, lon), gmean(EC, lat, lon)
    c_sig = gCP.mean(1) - gCC.mean(1)
    e_sig = gEP.mean(1) - gEC.mean(1)
    c_se = np.sqrt(gCP.var(1, ddof=1)/gCP.shape[1] + gCC.var(1, ddof=1)/gCC.shape[1])
    e_se = np.sqrt(gEP.var(1, ddof=1)/gEP.shape[1] + gEC.var(1, ddof=1)/gEC.shape[1])

    rows = []
    for lo in range(int(yrs.min())//10*10, int(yrs.max()), 10):
        hi = lo + 9
        m = (yrs >= lo) & (yrs <= hi)
        if m.sum() < 5:
            continue
        rows.append(dict(decade=f"{lo}-{hi}",
                         cesm=round(float(c_sig[m].mean()), 3),
                         cesm_se=round(float(np.sqrt((c_se[m]**2).mean())), 3),
                         emulator=round(float(e_sig[m].mean()), 3),
                         emulator_se=round(float(np.sqrt((e_se[m]**2).mean())), 3),
                         difference=round(float(e_sig[m].mean()-c_sig[m].mean()), 3),
                         unit=V["unit"] if args.var == "TREFHT" else "mm/day"))
    t = pd.DataFrame(rows)
    print(f"\nAerosol-removal signal ({args.var})")
    print(t.to_string(index=False))

    # ── final-decade maps ────────────────────────────────────────────────────
    keep = yrs >= yrs.max() - args.n_years + 1
    c_map = CP[keep].mean((0, 1)) - CC[keep].mean((0, 1))
    e_map = EP[keep].mean((0, 1)) - EC[keep].mean((0, 1))
    w = area_w(lat, lon)
    pc = float(np.average(c_map, weights=w)); pe = float(np.average(e_map, weights=w))
    cw = np.average((c_map-pc)*(e_map-pe), weights=w)
    corr = cw / np.sqrt(np.average((c_map-pc)**2, weights=w)
                        * np.average((e_map-pe)**2, weights=w))
    print(f"\nfinal decade {int(yrs[keep].min())}-{int(yrs[keep].max())}: "
          f"pattern r = {corr:.3f}, global mean emu {pe:+.3f} vs CESM2 {pc:+.3f}")

    if args.dump_data:
        os.makedirs(args.dump_data, exist_ok=True)
        d = [dict(year=int(y), source=s, value=float(v))
             for s, arr in (("cesm2", c_sig), ("emulator", e_sig),
                            ("cesm2_se", c_se), ("emulator_se", e_se))
             for y, v in zip(yrs, arr)]
        p = os.path.join(args.dump_data, f"ramip_signal_{args.var}.csv")
        pd.DataFrame(d).to_csv(p, index=False)
        print(f"[data] {p}")

    # ── plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        HAVE_CARTOPY = True
    except ImportError:
        HAVE_CARTOPY = False
        print("[warn] cartopy missing — map panels will be flat lat/lon")

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig = plt.figure(figsize=(9.6, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.10)
    ax = fig.add_subplot(gs[0, :])

    for sig, se, c, lab, n in ((c_sig, c_se, C_CESM, "CESM2 (RAMIP)", CP.shape[1]),
                               (e_sig, e_se, C_EMU, "Emulator", EP.shape[1])):
        ax.fill_between(yrs, sig-2*se, sig+2*se, color=c, alpha=0.20, lw=0)
        ax.plot(yrs, sig, color=c, lw=2.2, label=f"{lab} ({n} members)")
    ax.axhline(0, ls=":", lw=0.8, color="0.3")
    ax.set_ylabel(V["ylab"]); ax.set_xlabel("Year")
    ax.set_xlim(yrs.min(), yrs.max())
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a)  Aerosol-removal signal: ssp370-126aer minus ssp370, "
                 "each side vs its own control", loc="left", fontsize=9.5)
    ax.text(0.99, 0.03, "bands: ±2 SE of the difference of ensemble means",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.4, color="0.3")

    vmax = float(np.nanpercentile(np.abs(np.concatenate(
        [c_map.ravel(), e_map.ravel()])), 99)) or 1.0
    proj = dict(projection=ccrs.Robinson(central_longitude=0)) if HAVE_CARTOPY else {}
    im = None
    for j, (M, ttl) in enumerate(((c_map, "CESM2 (RAMIP)"), (e_map, "Emulator"))):
        axm = fig.add_subplot(gs[1, j], **proj)
        kw = dict(cmap=V["cmap"], vmin=-vmax, vmax=vmax, shading="auto")
        if HAVE_CARTOPY:
            kw["transform"] = ccrs.PlateCarree()
        im = axm.pcolormesh(lon, lat, M, **kw)
        if HAVE_CARTOPY:
            axm.coastlines(linewidth=0.35, color="0.25"); axm.set_global()
        axm.set_title(f"({'bc'[j]})  {ttl}   {int(yrs[keep].min())}–"
                      f"{int(yrs[keep].max())}", loc="left", fontsize=9.5)
    cb = fig.colorbar(im, ax=[fig.axes[1], fig.axes[2]], orientation="vertical",
                      fraction=0.02, pad=0.012, extend="both")
    cb.set_label(f"aerosol-removal signal ({V['unit']})", fontsize=8.5)
    fig.text(0.5, 0.02, f"pattern correlation r = {corr:.3f}", ha="center",
             fontsize=8.5, color="0.25")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"\nwrote {args.out}")
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
