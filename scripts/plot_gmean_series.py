#!/usr/bin/env python3
"""Global-mean series, absolute and anomaly, from the CURRENT eval format.

paper_fig_timeseries.py and plot_absolute.py both expect an older eval schema
with per-member `*_gmean_*` variables. The eval code no longer writes those --
it writes one `{var}_model` (member, year, lat, lon) array (eval_aero.py:1325)
-- so those scripts crash or, worse, emit an EMPTY figure for any arm evaluated
with current code. This reads the compact format directly.

CESM2 reference comes from the cached held-out member series
(plots/fig{1,2}_cesm2_members.csv), so no member tree is re-read.

    ~/miniconda3/envs/plotting/bin/python scripts/plot_gmean_series.py \
        --eval-dir <eval>/best_ep0520 --out-dir plots/paper_mmlin_interim
"""
import argparse, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCEN = {"hist": ("Historical", "tab:blue"), "ssp370": ("SSP3-7.0", "tab:red"),
        "aaer": ("Aerosol-only", "tab:green"), "ghg": ("GHG-only", "tab:purple")}
META = {"TREFHT": dict(unit="$^\\circ$C", csv="plots/fig1_cesm2_members.csv", k2c=True),
        "PRECT": dict(unit="mm/day", csv="plots/fig2_cesm2_members.csv", k2c=False)}
BASE = (1850, 1900)


def gmean(da):
    """cos(lat)-weighted global mean -> (member, year)."""
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(("lat", "lon"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--var", nargs="+", default=["TREFHT", "PRECT"])
    ap.add_argument("--label", default="emulator")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for var in args.var:
        meta = META[var]
        ref = pd.read_csv(meta["csv"]) if os.path.exists(meta["csv"]) else None
        emu = {}
        for key in SCEN:
            p = f"{args.eval_dir}/{var}_{key}.nc"
            if not os.path.exists(p):
                continue
            with xr.open_dataset(p) as ds:
                g = np.asarray(gmean(ds[f"{var}_model"]).values, float)   # (member, year)
                yrs = np.asarray(ds["year"].values).astype(int)
            if meta["k2c"] and np.nanmean(g) > 100:
                g = g - 273.15
            emu[key] = (yrs, g)
        if not emu:
            print(f"[skip] {var}: no eval files in {args.eval_dir}")
            continue

        # CESM2 pre-industrial baseline, taken ONCE from the historical record.
        # ssp370 starts in 2015 and has no 1850-1900 period of its own, so
        # without this its "anomaly" was the absolute temperature (~15 degC on
        # an anomaly axis). Same convention as the emulator side and as every
        # other figure: scenarios without a pre-industrial inherit hist's.
        ref_base = None
        if ref is not None:
            rh = ref[ref["scenario"] == "hist"]
            if len(rh):
                ph = rh.pivot_table(index="year", columns="member", values="gmean")
                vh = ph.values
                if meta["k2c"] and np.nanmean(vh) > 100:
                    vh = vh - 273.15
                yh = ph.index.values.astype(int)
                bb = (yh >= BASE[0]) & (yh <= BASE[1])
                if bb.any():
                    ref_base = float(np.nanmean(np.nanmean(vh, axis=1)[bb]))

        for mode in ("absolute", "anomaly"):
            fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
            for key, (label, col) in SCEN.items():
                if key not in emu:
                    continue
                yrs, g = emu[key]
                m = g.mean(0)
                if mode == "anomaly":
                    b = (yrs >= BASE[0]) & (yrs <= BASE[1])
                    off = m[b].mean() if b.any() else (
                        emu["hist"][1].mean(0)[(emu["hist"][0] >= BASE[0]) &
                                               (emu["hist"][0] <= BASE[1])].mean())
                    m = m - off
                    g = g - off
                ax.plot(yrs, m, color=col, lw=1.6, label=f"{label} ({args.label})")
                if g.shape[0] > 1:
                    ax.fill_between(yrs, g.min(0), g.max(0), color=col, alpha=0.18, lw=0)
                if ref is not None:
                    r = ref[ref["scenario"] == key]
                    if len(r):
                        piv = r.pivot_table(index="year", columns="member", values="gmean")
                        rv = piv.values
                        if meta["k2c"] and np.nanmean(rv) > 100:
                            rv = rv - 273.15
                        ry = piv.index.values.astype(int)
                        rm = np.nanmean(rv, axis=1)
                        if mode == "anomaly":
                            own = (ry >= BASE[0]) & (ry <= BASE[1])
                            base_val = (float(rm[own].mean()) if own.any()
                                        else ref_base)
                            if base_val is None:
                                continue          # cannot anomalise: draw nothing
                            rm = rm - base_val
                        ax.plot(ry, rm, color=col, lw=1.2, ls="--", alpha=0.9,
                                label=f"{label} (CESM2)")
            ax.set_xlabel("year")
            ax.set_ylabel(f"global-mean {var} ({meta['unit']})"
                          + ("" if mode == "absolute" else f", anomaly vs {BASE[0]}-{BASE[1]}"))
            ax.set_title(f"{var} global mean — {mode}. Solid: {args.label} "
                         f"(shading = member range); dashed: held-out CESM2", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            out = os.path.join(args.out_dir, f"gmean_series_{var}_{mode}")
            for ext in (".png", ".pdf"):
                fig.savefig(out + ext, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[WROTE] {out}.png/.pdf")


if __name__ == "__main__":
    main()
