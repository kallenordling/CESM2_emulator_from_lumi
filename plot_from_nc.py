"""Plot emulator vs CESM2 global-mean anomaly from eval-generated NetCDFs.

Reads <output_dir>/<VAR>_<exp>.nc (written by eval_aero.py) for the user-chosen
experiments and overlays, per experiment:
  * emulator  = solid line, <VAR>_model_gmean_mean_anom (+ member min/max band)
  * CESM2     = dashed line, <VAR>_cesm_gmean_mean_anom  (+ member min/max band)
A bottom panel shows the bias (emulator − CESM2) on common years.

Usage:
  python plot_from_nc.py <output_dir> <exp1> [exp2 ...] [--var TREFHT]
                         [--out FILE] [--no-spread] [--title TXT]
e.g.
  python plot_from_nc.py /mnt/lumi_sc2/eval_output/manual/ep0852_v2 ssp126 ssp245 ssp370
"""
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "hist": "#1f77b4", "ssp370": "#d62728", "ssp126": "#9467bd",
    "ssp245": "#17becf", "ghg": "#2ca02c", "aaer": "#ff7f0e",
}


def _years(da):
    d = "year" if "year" in da.dims else ("cesm_year" if "cesm_year" in da.dims else da.dims[0])
    v = da[d].values
    return (v.astype(int) if np.issubdtype(np.asarray(v).dtype, np.number)
            else np.array([int(str(x)[:4]) for x in v]))


def _members(ds, var_pat):
    """Stack all per-member series matching var_pat.format(i) → (N, T) or None."""
    mem, i = [], 1
    while var_pat.format(i) in ds:
        mem.append(ds[var_pat.format(i)].values)
        i += 1
    return np.stack(mem, axis=0) if mem else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("experiments", nargs="+", help="experiment names to plot")
    ap.add_argument("--var", default="TREFHT")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-spread", action="store_true", help="hide member min/max bands")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    V = args.var

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    cyc = plt.cm.tab10(np.linspace(0, 1, 10))
    plotted = 0

    for k, exp in enumerate(args.experiments):
        path = os.path.join(args.output_dir, f"{V}_{exp}.nc")
        if not os.path.isfile(path):
            print(f"  [skip] not found: {path}")
            continue
        ds = xr.open_dataset(path)
        c = COLORS.get(exp, cyc[k % 10])

        # emulator (model)
        mg = ds[f"{V}_model_gmean_mean_anom"]
        my = _years(mg)
        ax.plot(my, mg.values, color=c, lw=2.0, label=f"{exp} emulator")
        if not args.no_spread:
            mm = _members(ds, f"{V}_model_gmean_m{{}}_anom")
            if mm is not None:
                ax.fill_between(my, mm.min(0), mm.max(0), color=c, alpha=0.15, lw=0)

        # CESM2
        cg_name = f"{V}_cesm_gmean_mean_anom"
        if cg_name in ds:
            cg = ds[cg_name]
            cy = _years(cg)
            ax.plot(cy, cg.values, color=c, lw=2.0, ls="--", alpha=0.85,
                    label=f"{exp} CESM2")
            if not args.no_spread:
                cm = _members(ds, f"{V}_cesm_gmean_m{{}}_anom")
                if cm is not None:
                    ax.fill_between(cy, cm.min(0), cm.max(0), facecolor="none",
                                    hatch="///", edgecolor=c, alpha=0.35, lw=0)
            # bias on common years
            common, im, ic = np.intersect1d(my, cy, return_indices=True)
            axb.plot(common, mg.values[im] - cg.values[ic], color=c, lw=1.5, label=exp)
        plotted += 1
        ds.close()

    if not plotted:
        raise SystemExit("no experiments plotted (no matching NetCDFs found)")

    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_ylabel(f"{V} anomaly (°C)")
    ax.set_title(args.title or f"Emulator (solid) vs CESM2 (dashed) — {', '.join(args.experiments)}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    axb.axhline(0, color="k", lw=0.8)
    axb.set_ylabel("bias (emu − CESM2) °C"); axb.set_xlabel("year")
    axb.grid(alpha=0.3); axb.legend(fontsize=8)

    out = args.out or os.path.join(
        args.output_dir, f"timeseries_{'_'.join(args.experiments)}.png")
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
