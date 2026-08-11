#!/usr/bin/env python3
"""
Paper figure: emulated vs held-out CESM2 DISTRIBUTIONS over the final decade.

One panel per experiment (historical, SSP3-7.0, AAER, GHG). Each pools every
grid point, every year of the last N years, and every ensemble member into one
distribution, then overlays emulator against CESM2.

Where the timeseries figure asks "is the forced response right?", this asks "is
the whole distribution right?" — including the tails, which a global mean cannot
show. That matters most for precipitation, which is strongly right-skewed: an
emulator can match the mean while badly missing the extremes.

AREA WEIGHTING
--------------
Grid cells are not equal area, so an unweighted histogram over-counts the poles
by ~cos(lat). Every count is weighted by cos(lat), making these true
area-fraction distributions.

REFERENCE = HELD-OUT MEMBERS ONLY
---------------------------------
CESM2 is read from the training trees, restricted to members absent from the
data config's experiment_configs, and truncated to the same ensemble size as the
emulator (--n-ref-members, default 5). The eval NetCDF's own CESM arrays are NOT
used: for aaer/ghg they contain all ten members, nine of which are training data.

Usage
-----
    python scripts/paper_fig_histograms.py --var TREFHT \\
        --eval-dir /path/to/eval_output/run_mseyb_BCprect/best_ep0490

    python scripts/paper_fig_histograms.py --var PRECT --eval-dir ... --n-years 10
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VAR = "TREFHT"

VARMETA = {
    "TREFHT": dict(unit="°C", unit_plain="degC",
                   xlab="Near-surface air temperature (°C)",
                   title="temperature",
                   tree_scale={None: 1.0, "K": 1.0, "degC": 1.0},
                   tree_offset={"K": -273.15, "degC": 0.0, None: 0.0}),
    "PRECT":  dict(unit="mm day$^{-1}$", unit_plain="mm/day",
                   xlab="Precipitation (mm day$^{-1}$)",
                   title="precipitation",
                   tree_scale={"m/s": 86400.0 * 1000.0, "mm/day": 1.0, None: 1.0},
                   tree_offset={"m/s": 0.0, "mm/day": 0.0, None: 0.0}),
}

SCEN = {
    "hist":   ("Historical",                "hist",   "hist",   "#0072B2"),
    "ssp370": ("SSP3-7.0",                  "ssp370", "ssp370", "#D55E00"),
    "aaer":   ("Aerosol-only (AAER)",       "aaer",   "AAER",   "#E69F00"),
    "ghg":    ("Greenhouse-gas-only (GHG)", "ghg",    "GHG",    "#009E73"),
}
CFG_KEY = {k: k for k in SCEN}


def unseen_members(tree_root: Path, subdir: str, trained: set) -> list:
    d = tree_root / subdir
    have = {p.name for p in d.iterdir() if p.is_dir() and p.name != "diagnostics"}
    return sorted(have - set(trained))


def read_emulator_block(nc_path: Path, n_years: int):
    """(values, weights) pooled over members x last n_years x grid."""
    ds = xr.open_dataset(nc_path)
    years = ds["year"].values.astype(int)
    keep = years >= years.max() - n_years + 1
    names = [v for v in ds.data_vars
             if v.startswith(f"{VAR}_model_m") and not v.endswith("_anom")
             and not v.startswith(f"{VAR}_model_mean")]
    if not names:
        raise KeyError(f"{nc_path}: no per-member {VAR}_model_m* fields")
    lat = ds["lat"].values
    w2d = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None],
                          (len(lat), ds.sizes["lon"]))
    vals, wts = [], []
    for nm in sorted(names, key=lambda x: int(x.rsplit("_m", 1)[1])):
        a = ds[nm].values[keep]                      # (t, lat, lon)
        vals.append(a.ravel())
        wts.append(np.broadcast_to(w2d, a.shape).ravel())
    ds.close()
    yr = years[keep]
    return (np.concatenate(vals), np.concatenate(wts),
            int(yr.min()), int(yr.max()), len(names))


def read_cesm_block(tree_root: Path, subdir: str, members: list, n_years: int):
    """Same, from the held-out training-tree members, converted to model units."""
    meta = VARMETA[VAR]
    vals, wts, y0, y1 = [], [], None, None
    for i, mem in enumerate(members, 1):
        files = sorted((tree_root / subdir / mem).glob("*.nc"))
        if not files:
            print(f"      [{i}/{len(members)}] {mem}: no chunks, skipped", flush=True)
            continue
        ds = xr.open_mfdataset(files, combine="by_coords", decode_times=False)
        if VAR not in ds:
            raise KeyError(f"{tree_root/subdir/mem}: no {VAR!r} "
                           f"(--tree-root must point at the {VAR} tree)")
        u = ds[VAR].attrs.get("units")
        scale = meta["tree_scale"].get(u)
        off = meta["tree_offset"].get(u, 0.0)
        if scale is None:
            raise ValueError(f"{mem}: {VAR} units {u!r} have no conversion to "
                             f"{meta['unit_plain']}")
        tdim = "time" if "time" in ds[VAR].dims else "year"
        years = np.asarray(ds[tdim].values).astype(int)
        keep = years >= years.max() - n_years + 1
        if i == 1:
            print(f"      units {u!r} -> x{scale:g}{off:+g} "
                  f"[{meta['unit_plain']}]", flush=True)
        a = (ds[VAR].isel({tdim: np.where(keep)[0]}).values * scale + off)
        lat = ds["lat"].values
        w2d = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape[-2:])
        vals.append(a.ravel())
        wts.append(np.broadcast_to(w2d, a.shape).ravel())
        yy = years[keep]
        y0 = int(yy.min()) if y0 is None else min(y0, int(yy.min()))
        y1 = int(yy.max()) if y1 is None else max(y1, int(yy.max()))
        ds.close()
        print(f"      [{i}/{len(members)}] {subdir}/{mem} {yy.min()}-{yy.max()}",
              flush=True)
    return np.concatenate(vals), np.concatenate(wts), y0, y1, len(vals)


def wq(x, w, q):
    """Weighted quantile."""
    i = np.argsort(x)
    x, w = x[i], w[i]
    c = np.cumsum(w) - 0.5 * w
    return np.interp(np.asarray(q) * w.sum(), c, x)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Emulator vs held-out CESM2 distributions over the final decade",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--var", default="TREFHT", choices=sorted(VARMETA))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--data-root",
                    default="/home/nordling/mnt/lumi_sc2/emulator_data")
    ap.add_argument("--tree-root", default=None,
                    help="default <data-root>/training_data/<var>")
    ap.add_argument("--data-config",
                    default="configs/config_data_ybias_BCprect.yaml")
    ap.add_argument("--n-years", type=int, default=10,
                    help="pool the last N years of each experiment")
    ap.add_argument("--n-ref-members", type=int, default=5,
                    help="CESM2 members to use; matches the emulator's count")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    global VAR
    VAR = args.var
    META = VARMETA[VAR]
    if args.out is None:
        args.out = f"plots/paper_fig_hist_{VAR}.png"
    if args.tree_root is None:
        args.tree_root = os.path.join(args.data_root, "training_data", VAR)
    tree_root, eval_dir = Path(args.tree_root), Path(args.eval_dir)
    print(f"[var] {VAR} ({META['unit_plain']})   last {args.n_years} years")

    import yaml
    cfg = yaml.safe_load(open(args.data_config))
    trained = {e["scenario_name"]: set(e.get("realizations", []))
               for e in cfg["experiment_configs"]}

    data = {}
    for sc, (label, ncname, sub, colour) in SCEN.items():
        p = eval_dir / f"{VAR}_{ncname}.nc"
        if not p.exists():
            print(f"[skip] {sc}: {p} not found")
            continue
        print(f"\n[{sc}] emulator …", flush=True)
        ev, ew, ey0, ey1, n_emu = read_emulator_block(p, args.n_years)
        mems = unseen_members(tree_root, sub, trained.get(CFG_KEY[sc], set()))
        if args.n_ref_members > 0:
            mems = mems[:args.n_ref_members]
        print(f"[{sc}] CESM2 held-out {mems} …", flush=True)
        cv, cw, cy0, cy1, n_c = read_cesm_block(tree_root, sub, mems, args.n_years)
        data[sc] = dict(ev=ev, ew=ew, cv=cv, cw=cw, n_emu=n_emu, n_c=n_c,
                        ey=(ey0, ey1), cy=(cy0, cy1))
        print(f"[{sc}] emulator {ey0}-{ey1} ({ev.size:,} pts), "
              f"CESM2 {cy0}-{cy1} ({cv.size:,} pts)")

    if not data:
        print("no data", file=sys.stderr)
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 10.5, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    rows = []
    for i, (ax, (sc, d)) in enumerate(zip(axes.flat, data.items())):
        label, _, _, colour = SCEN[sc]
        ev, ew, cv, cw = d["ev"], d["ew"], d["cv"], d["cw"]

        # shared bins spanning both, clipped to the 0.1-99.9 percentile so a
        # handful of extreme cells do not flatten the body of the distribution
        lo = min(wq(ev, ew, 0.001), wq(cv, cw, 0.001))
        hi = max(wq(ev, ew, 0.999), wq(cv, cw, 0.999))
        bins = np.linspace(lo, hi, args.bins + 1)

        ax.hist(cv, bins=bins, weights=cw, density=True, histtype="stepfilled",
                color="0.55", alpha=0.45, label="CESM2 (held-out)")
        ax.hist(ev, bins=bins, weights=ew, density=True, histtype="step",
                color=colour, lw=1.8, label="Emulator")

        st = {}
        for nm, v, w in (("emu", ev, ew), ("cesm", cv, cw)):
            m = np.average(v, weights=w)
            sd = np.sqrt(np.average((v - m) ** 2, weights=w))
            q = wq(v, w, [0.01, 0.5, 0.99])
            st[nm] = (m, sd, q)
        (em, esd, eq), (cm, csd, cq) = st["emu"], st["cesm"]
        rows.append(dict(scenario=sc, years=f"{d['ey'][0]}-{d['ey'][1]}",
                         n_emu=d["n_emu"], n_cesm=d["n_c"],
                         emu_mean=round(em, 3), cesm_mean=round(cm, 3),
                         d_mean=round(em - cm, 3),
                         emu_sd=round(esd, 3), cesm_sd=round(csd, 3),
                         d_sd=round(esd - csd, 3),
                         d_p1=round(eq[0] - cq[0], 3),
                         d_p50=round(eq[1] - cq[1], 3),
                         d_p99=round(eq[2] - cq[2], 3)))

        ax.set_title(f"({'abcd'[i]}) {label}", loc="left")
        ax.set_xlabel(META["xlab"])
        if i % 2 == 0:
            ax.set_ylabel("Area-weighted density")
        ax.text(0.97, 0.95,
                f"{d['ey'][0]}–{d['ey'][1]}\n"
                f"n = {d['n_emu']} emulator, {d['n_c']} CESM2\n"
                f"Δmean {em-cm:+.2f}, Δsd {esd-csd:+.2f} "
                f"{META['unit_plain']}",
                transform=ax.transAxes, fontsize=7.4, va="top", ha="right",
                color="0.25")

    axes.flat[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Emulated vs held-out CESM2 {META['title']} distribution, "
                 f"final {args.n_years} years "
                 f"(all grid points, years and members; area-weighted)",
                 fontsize=10.5, y=0.995)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".pdf")), bbox_inches="tight")
    print(f"\nwrote {args.out}")
    print(f"wrote {Path(args.out).with_suffix('.pdf')}")

    t = pd.DataFrame(rows)
    print(f"\nDistribution comparison ({META['unit_plain']}); "
          f"d_* = emulator minus CESM2")
    print(t.to_string(index=False))
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
